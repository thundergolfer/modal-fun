"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""
import datetime
import dataclasses
from email.base64mime import body_decode
import json
import pathlib
import sys
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import modal

import config
import podcast
import search
import web


volume = modal.SharedVolume().persist("dataset-cache-vol")
app_image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "https://github.com/openai/whisper/archive/5d8d3e75a4826fe5f01205d81c3017a805fc2bf9.tar.gz",
            "dacite",
            "jiwer",
            "ffmpeg-python",
            "gql[all]~=3.0.0a5",
            "pandas",
            "loguru==0.6.0",
            "torchaudio==0.12.1",
        ]
    )
    .apt_install(
        [
            "ffmpeg",
        ]
    )
)
search_image = modal.Image.debian_slim().pip_install(
    ["scikit-learn~=0.24.2", "tqdm~=4.46.0", "numpy~=1.23.3", "dacite"]
)

stub = modal.Stub("whisper-pod-transcriber", image=app_image)
web_app = FastAPI()


def utc_now() -> datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def create_transcript_path(
    guid_hash: str, model: Optional[config.ModelSpec] = None
) -> pathlib.Path:
    if model is None:
        model = config.supported_whisper_models["base.en"]  # Assumption
    model_slug = f"whisper-{model.name.replace('.', '-')}"
    return config.TRANSCRIPTIONS_DIR / f"{guid_hash}-{model_slug}.json"


@web_app.get("/all")
async def all_transcripts():
    import dacite
    from collections import defaultdict

    episodes_by_show = defaultdict(list)
    if config.METADATA_DIR.exists():
        for file in config.METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes_by_show[ep.show].append(ep)

    body = web.html_all_transcripts_header()
    for show, episodes_by_show in episodes_by_show.items():
        episode_part = f"""<div class="font-bold text-center text-green-500 text-xl mt-6">{show}</div>"""
        episode_part += web.html_episode_list(episodes_by_show)
        body += episode_part
    content = web.html_page(
        title="Modal Podcast Transcriber | All Transcripts", body=body
    )
    return HTMLResponse(content=content, status_code=200)


@web_app.get("/transcripts/{podcast_id}/{episode_guid_hash}")
async def episode_transcript_page(podcast_id: str, episode_guid_hash):
    import dacite

    _pod_id = podcast_id  # TODO(Jonathon): Check episode matches podcast ID.

    episode_metadata_path = config.METADATA_DIR / f"{episode_guid_hash}.json"
    transcription_path = create_transcript_path(episode_guid_hash)
    with open(transcription_path, "r") as f:
        data = json.load(f)
    with open(episode_metadata_path, "r") as f:
        metadata = json.load(f)
        episode = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=metadata)

    segments_ul_html = web.html_transcript_list(data["segments"])
    episode_header_html = web.html_episode_header(episode)
    body = episode_header_html + segments_ul_html
    content = web.html_page(
        title="Modal Podcast Transcriber | Episode Transcript", body=body
    )
    return HTMLResponse(content=content, status_code=200)


@web_app.get("/transcripts/{podcast_id}")
async def podcast_transcripts_page(podcast_id: str):
    import dacite

    pod_metadata_path = config.PODCAST_METADATA_DIR / f"{podcast_id}.json"
    if not pod_metadata_path.exists():
        return HTMLResponse(content=web.html_podcast_404_page(), status_code=404)
    else:
        with open(pod_metadata_path, "r") as f:
            data = json.load(f)
            pod_metadata = dacite.from_dict(
                data_class=podcast.PodcastMetadata, data=data
            )

    podcast_header_html = web.html_podcast_header(pod_metadata)
    podcast_episodes = []
    if config.METADATA_DIR.exists():
        for file in config.METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                if str(ep.podcast_id) == podcast_id:
                    podcast_episodes.append(ep)

    transcript_list_html = web.html_episode_list(podcast_episodes)
    body = podcast_header_html + transcript_list_html
    content = web.html_page(title="Modal Podcast Transcriber | Transcripts", body=body)
    return HTMLResponse(content=content, status_code=200)


def is_podcast_recently_transcribed(podcast_id: str):
    if not config.COMPLETED_DIR.exists():
        return False
    completion_marker_path = config.COMPLETED_DIR / f"{podcast_id}.txt"
    return completion_marker_path.exists()


@web_app.post("/podcasts")
async def podcasts_endpoint(request: Request):
    import dataclasses

    form = await request.form()
    name = form["podcast"]
    podcasts_response = []
    for podcast in search_podcast(name):
        data = dataclasses.asdict(podcast)
        if is_podcast_recently_transcribed(podcast.id):
            data["recently_transcribed"] = "true"
        else:
            data["recently_transcribed"] = "false"
        podcasts_response.append(data)
    return JSONResponse(content=podcasts_response)


@stub.asgi(
    mounts=[modal.Mount("/assets", local_dir=config.ASSETS_PATH)],
    shared_volumes={config.CACHE_DIR: volume},
)
def fastapi_app():
    import fastapi.staticfiles

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app


@stub.function(schedule=modal.Period(hours=4))
def refresh_index():
    print(f"Running scheduled index refresh at {utc_now()}")
    index()


@stub.function(
    image=app_image,
    secret=modal.ref("podchaser"),
)
def search_podcast(name):
    from gql import gql

    print(f"Searching for '{name}'")
    client = podcast.create_podchaser_client()
    podcasts_raw = podcast.search_podcast_name(gql, client, name, max_results=3)
    print(f"Found {len(podcasts_raw)} results for '{name}'")
    return [
        podcast.PodcastMetadata(
            id=pod["id"],
            title=pod["title"],
            description=pod["description"],
            html_description=pod["htmlDescription"],
            web_url=pod["webUrl"],
        )
        for pod in podcasts_raw
    ]


@stub.function(
    image=search_image,
    shared_volumes={config.CACHE_DIR: volume},
)
def index():
    import dacite
    import dataclasses
    from collections import defaultdict

    print("Starting transcript indexing process.")
    config.SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = defaultdict(list)
    guid_hash_to_episodes = {}
    if config.METADATA_DIR.exists():
        for file in config.METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes[ep.show].append(ep)
                guid_hash_to_episodes[ep.guid_hash] = ep

    print(f"Loaded {len(guid_hash_to_episodes)} podcast episodes.")

    transcripts = {}
    if config.TRANSCRIPTIONS_DIR.exists():
        for file in config.TRANSCRIPTIONS_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                guid_hash = file.stem.split("-")[0]
                transcripts[guid_hash] = data

    # Important: These have to be the same length and have same episode order.
    # i-th element of indexed_episodes is the episode indexed by the i-th element
    # of search_records
    indexed_episodes = []
    search_records = []
    for key, value in transcripts.items():
        idxd_episode = guid_hash_to_episodes.get(key)
        if idxd_episode:
            search_records.append(
                search.SearchRecord(
                    title=idxd_episode.title,
                    text=value["text"],
                )
            )
            # Prepare records for JSON serialization
            indexed_episodes.append(dataclasses.asdict(idxd_episode))

    print(
        f"Matched {len(search_records)} transcripts against episode metadata records."
    )

    filepath = pathlib.Path(config.SEARCH_DIR, "jall.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(indexed_episodes, f)

    print(
        "calc feature vectors for all transcripts, keeping track of most similar podcasts"
    )
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    filepath = pathlib.Path(config.SEARCH_DIR, "sim_tfidf_svm.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(sim_svm, f)

    print("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    filepath = pathlib.Path(config.SEARCH_DIR, "search.json")
    print(f"writing {filepath}")
    with open(filepath, "w") as f:
        json.dump(search_dict, f)


@web_app.post("/transcribe")
async def transcribe_job(request: Request):
    form = await request.form()
    pod_id = form["podcast_id"]
    call = transcribe_podcast.submit(podcast_id=pod_id)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse(status_code=202)

    return result


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    secret=modal.ref("podchaser"),
    concurrency_limit=2,
)
def transcribe_podcast(podcast_id: str):
    from gql import gql

    pod_metadata: podcast.PodcastMetadata = podcast.fetch_podcast(gql, podcast_id)
    metadata_path = config.PODCAST_METADATA_DIR / f"{podcast_id}.json"
    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(pod_metadata), f)
    print(f"Wrote podcast metadata to {metadata_path}")

    temp_limit = config.transcripts_per_podcast_limit
    print(f"Fetching {temp_limit} podcast episodes to transcribe.")
    episodes = fetch_episodes(show_name=pod_metadata.title, podcast_id=podcast_id)
    # Most recent episodes
    episodes.sort(key=lambda ep: ep.publish_date, reverse=True)
    completed = []
    for result in process_episode.map(episodes[:temp_limit], order_outputs=False):
        print("Processed:")
        print(result.title)
        completed.append(result.title)

    config.COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    completion_marker_path = config.COMPLETED_DIR / f"{podcast_id}.txt"
    with open(completion_marker_path, "w") as f:
        f.write(str(utc_now()))
    print(f"Marked podcast {podcast_id} as recently transcribed.")
    return completed  # Need to return something for function call polling to work.


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    gpu=True,
    concurrency_limit=10,
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: config.ModelSpec,
):
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model.name, device=device)
    result = model.transcribe(str(audio_filepath), language="en")
    print(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
)
def process_episode(episode: podcast.EpisodeMetadata):
    config.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    config.TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = config.RAW_AUDIO_DIR / episode.guid_hash
    podcast.store_original_audio(
        url=episode.original_download_link,
        destination=destination_path,
    )

    model = config.supported_whisper_models["base.en"]
    print(f"Using the {model.name} model which has {model.params} parameters.")

    metadata_path = config.METADATA_DIR / f"{episode.guid_hash}.json"
    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(episode), f)
    print(f"Wrote episode metadata to {metadata_path}")

    transcription_path = create_transcript_path(episode.guid_hash, model)
    if transcription_path.exists():
        print(
            f"Transcription already exists for '{episode.title}' with ID {episode.guid_hash}."
        )
        print(f"Skipping GPU transcription.")
    else:
        transcribe_episode(
            audio_filepath=destination_path,
            result_path=transcription_path,
            model=model,
        )
    return episode


@stub.function(
    image=app_image,
    secret=modal.ref("podchaser"),
    shared_volumes={config.CACHE_DIR: volume},
)
def fetch_episodes(show_name: str, podcast_id: str, max_episodes=100):
    import hashlib
    from gql import gql

    client = podcast.create_podchaser_client()
    episodes_raw = podcast.fetch_episodes_data(
        gql, client, podcast_id, max_episodes=max_episodes
    )
    print(f"Retreived {len(episodes_raw)} raw episodes")
    episodes = [
        podcast.EpisodeMetadata(
            podcast_id=podcast_id,
            podcast_title=show_name,
            show=show_name,
            title=ep["title"],
            publish_date=ep["airDate"],
            description=ep["description"],
            episode_url=ep["url"],
            html_description=ep["htmlDescription"],
            guid=ep["guid"],
            guid_hash=hashlib.md5(ep["guid"].encode("utf-8")).hexdigest(),
            original_download_link=ep["audioUrl"],
        )
        for ep in episodes_raw
        if "guid" in ep
    ]
    no_guid_count = len(episodes) - len(episodes_raw)
    print(f"{no_guid_count} episodes had no GUID and couldn't be used.")
    return episodes


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "transcribe":
        show_name = sys.argv[2]
        podcast_id = config.podchaser_podcast_ids[show_name]
        with stub.run() as app:
            print(f"Modal app ID -> {app.app_id}")
            transcribe_podcast(podcast_id=podcast_id)
    elif cmd == "serve":
        stub.serve()
    elif cmd == "index":
        with stub.run():
            index()
    elif cmd == "search-podcast":
        with stub.run():
            for pod in search_podcast(sys.argv[2]):
                print(pod)
    else:
        exit(
            f"Unknown command {cmd}. Supported commands: [transcribe, run, serve, index, search-podcast]"
        )
