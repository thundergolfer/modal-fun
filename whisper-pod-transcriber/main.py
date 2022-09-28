"""
whisper-pod-transcriber uses OpenAI's Whisper modal to do speech-to-text transcription
of podcasts.
"""
import datetime
from dataclasses import dataclass
import dataclasses
from importlib.resources import path
import json
import os
import pathlib
from re import M
import sys

from typing import Any, NamedTuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import modal

import podcast
import search


CACHE_DIR = "/cache"
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
METADATA_DIR = pathlib.Path(CACHE_DIR, "metadata")
COMPLETED_DIR = pathlib.Path(CACHE_DIR, "completed")
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")
SEARCH_DIR = pathlib.Path(CACHE_DIR, "search")
assets_path = pathlib.Path(__file__).parent / "web"
podchaser_podcast_ids = {
    "ezra_klein_nyt": 1582975,
    "ezra_klein_vox": 82327,
    "lex_fridman": 721928,
    "The Joe Rogan Experience": 10829,
}


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


supported_whisper_models = {
    "tiny.en": ModelSpec(name="tiny.en", params="39M", relative_speed=32),
    "base.en": ModelSpec(name="base.en", params="74M", relative_speed=16),
    "small.en": ModelSpec(name="small.en", params="244M", relative_speed=6),
    "medium.en": ModelSpec(name="medium.en", params="769M", relative_speed=2),
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}


volume = modal.SharedVolume().persist("dataset-cache-vol")
app_image = (
    modal.DebianSlim()
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
web_image = modal.DebianSlim().pip_install(["dacite"])
search_image = modal.DebianSlim().pip_install(
    ["scikit-learn~=0.24.2", "tqdm~=4.46.0", "numpy~=1.17.3", "dacite"]
)

stub = modal.Stub("whisper-pod-transcriber", image=app_image)
web_app = FastAPI()


@web_app.get("/episodes")
async def episodes():
    import dacite
    from collections import defaultdict

    episodes_by_show = defaultdict(list)
    all_episodes = []
    episodes_content = ""
    if METADATA_DIR.exists():
        for file in METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes_by_show[ep.show].append(ep)
                all_episodes.append(ep)

    for show, episodes_by_show in episodes_by_show.items():
        episodes_content += f"<h4>{show}</h4>\n"
        episodes_content += "\n<ul>"
        for ep in episodes_by_show:
            episodes_content += f"\n<li>{ep.title} - {ep.publish_date}</li>"
        episodes_content += "\n</ul>"
    content = f"""
    <html>
        <h1>Modal Transcriber!</h1>
        <h3>Transcribed episodes!</h3>
        {episodes_content}
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


@web_app.get("/transcripts/{podcast_id}/{episode_guid_hash}")
async def episode_transcript_page(podcast_id: str, episode_guid_hash):
    model_slug = "whisper-base-en"  # TODO: Hardcoded for now.
    transcription_path = TRANSCRIPTIONS_DIR / f"{episode_guid_hash}-{model_slug}.json"
    with open(transcription_path, "r") as f:
        data = json.load(f)

    segments_ul_html = """<ul class="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">"""
    for segment in data["segments"]:
        segment_li = f"""<li class="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg">
            {segment["text"]}
        </li>
        """
        segments_ul_html += segment_li
    segments_ul_html += "</ul>"

    content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Modal Podcast Transcriber | Episode Transcript</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>

    <body class="bg-gray-50">
        <div class="mx-auto max-w-md py-16">
            Hello and welcome to the individual transcripts page!

            episode hash: {episode_guid_hash}
        </div>
        <div class="mx-auto max-w-4xl py-16">
            <h3>Transcript</h3>
            <div>
                {segments_ul_html}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


@web_app.get("/transcripts/{podcast_id}")
async def podcast_transcripts_page(podcast_id: str):
    import dacite

    # 1. Read all metadata files, filtering for podcast_id
    # 3. Read all transcripts using guid and hardcoded model id
    podcast_episodes = []
    if METADATA_DIR.exists():
        for file in METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                # Hack: Some files in this directory aren't episode metadata
                if isinstance(data, list):
                    continue
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                if str(ep.podcast_id) == podcast_id:
                    podcast_episodes.append(ep)

    transcript_list_html = """<ul class="bg-white rounded-lg border border-gray-200 w-384 text-gray-900">"""
    for ep in podcast_episodes:
        episode_li = f"""<li class="px-6 py-2 border-b border-gray-200 w-full rounded-t-lg">
            <a href="/transcripts/{ep.podcast_id}/{ep.guid_hash}" class="text-blue-700 no-underline hover:underline">
                {ep.title}
            </a> | {ep.show}
        </li>
        """
        transcript_list_html += episode_li
    transcript_list_html += "</ul>"

    content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Modal Podcast Transcriber | Transcripts</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>

    <body class="bg-gray-50">
        <div className="mx-auto max-w-md py-16">
            Hello and welcome to the transcripts page!
            {podcast_id}
            Found {len(podcast_episodes)} for you!
            <div class="flex justify-center">
                {transcript_list_html}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


def is_podcast_recently_transcribed(podcast_id: str):
    if not COMPLETED_DIR.exists():
        return False
    completion_marker_path = COMPLETED_DIR / f"{podcast_id}.txt"
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


@web_app.get("/old")
async def root(query: str = ""):
    import dacite
    import json
    from collections import defaultdict

    all_json = pathlib.Path(SEARCH_DIR, "jall.json")

    all_episodes = []
    if METADATA_DIR.exists():
        for file in METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                all_episodes.append(ep)
    print("load all indexed episodes")
    with open(all_json, "r") as f:
        items_data = json.load(f)
        indexed_eps = [
            dacite.from_dict(data_class=podcast.EpisodeMetadata, data=x)
            for x in items_data
        ]

    search_results_html = ""
    if query:
        search_results = search_transcripts(items=indexed_eps, query=query)
        list_items = []
        for score, episode in search_results:
            list_items.append(
                f"""<li>
                    <span>show: {episode.show}</span></br>
                    <span>score: {score}</span></br>
                    <span>title: {episode.title}</span></br>
                    <span>url: {episode.episode_url}</span></br>
                </li>"""
            )
        search_results_html = "\n".join(list_items)
    else:
        search_results_html = ""

    content = f"""
    <html>
        <h1>Modal Transcriber!</h1>
        <section>
            <h4>Your query:</h4>
            <h5>{query}</h5>
            <h4>Your search results:</h4>
            <ul>{search_results_html}</ul>
        </section>
    </html>
    """
    return HTMLResponse(content=content, status_code=200)


@stub.asgi(
    mounts=[modal.Mount("/assets", local_dir=assets_path)],
    shared_volumes={CACHE_DIR: volume},
)
def fastapi_app():
    import fastapi.staticfiles

    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app


def _write_json(obj, filepath, msg="") -> None:
    suffix = f"; {msg}" if msg else ""
    print(f"writing {filepath}{suffix}")
    with open(filepath, "w") as f:
        json.dump(obj, f)


def utc_now() -> datetime:
    return datetime.datetime.now(datetime.timezone.utc)


@stub.function(schedule=modal.Period(hours=4))
def refresh_index():
    print(f"Running scheduled index refresh at {utc_now()}")
    index()


def search_transcripts(query: str, items: list[podcast.EpisodeMetadata]):
    query_parts = query.lower().strip().split()  # split by spaces
    search_json = pathlib.Path(SEARCH_DIR, "search.json")

    print("load search dictionary")
    with open(search_json, "r") as f:
        search_dict = json.load(f)

    n = len(items)
    scores = []
    for i, sd in enumerate(search_dict):
        score = sum(sd.get(q, 0) for q in query_parts)
        if score == 0:
            continue  # no match whatsoever, don't include
        score += (
            1.0 * (n - i) / n
        )  # give a small boost to more recent episodes (low index)
        scores.append((score, items[i]))
    scores.sort(reverse=True, key=lambda x: x[0])  # descending
    return scores


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
            web_url=pod["webUrl"],
        )
        for pod in podcasts_raw
    ]


@stub.function(
    image=search_image,
    shared_volumes={CACHE_DIR: volume},
)
def index():
    import dacite
    import dataclasses
    from collections import defaultdict

    print("Starting transcript indexing process.")
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = defaultdict(list)
    guid_hash_to_episodes = {}
    if METADATA_DIR.exists():
        for file in METADATA_DIR.iterdir():
            with open(file, "r") as f:
                data = json.load(f)
                # Hack: Some files in this directory aren't episode metadata
                if isinstance(data, list):
                    continue
                ep = dacite.from_dict(data_class=podcast.EpisodeMetadata, data=data)
                episodes[ep.show].append(ep)
                guid_hash_to_episodes[ep.guid_hash] = ep

    print(f"Loaded {len(guid_hash_to_episodes)} podcast episodes.")

    transcripts = {}
    if TRANSCRIPTIONS_DIR.exists():
        for file in TRANSCRIPTIONS_DIR.iterdir():
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

    _write_json(indexed_episodes, pathlib.Path(SEARCH_DIR, "jall.json"))

    print(
        "calculate feature vectors for all transcripts and keep track of most similar other episodes"
    )
    X, v = search.calculate_tfidf_features(search_records)
    sim_svm = search.calculate_similarity_with_svm(X)
    _write_json(
        sim_svm,
        pathlib.Path(SEARCH_DIR, "sim_tfidf_svm.json"),
    )

    print("calculate the search index to support search")
    search_dict = search.build_search_index(search_records, v)
    _write_json(
        search_dict,
        pathlib.Path(SEARCH_DIR, "search.json"),
    )


@web_app.post("/transcribe")
async def transcribe_job(request: Request):
    # Use aio_lookup since we're in an async context.
    form = await request.form()
    pod_name = form["podcast_name"]
    pod_id = form["podcast_id"]
    call = transcribe_podcast.submit(name=pod_name, podcast_id=pod_id)
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
    shared_volumes={CACHE_DIR: volume},
    concurrency_limit=2,
)
def transcribe_podcast(name: str, podcast_id: str):
    episodes = fetch_episodes(show_name=name, podcast_id=podcast_id)
    # Most recent episodes
    episodes.sort(key=lambda ep: ep.publish_date, reverse=True)
    temp_limit = 5  # TODO: Remove when basics are working
    for result in process_episode.map(episodes[:temp_limit], order_outputs=False):
        print("Processed:")
        print(result.title)

    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    completion_marker_path = COMPLETED_DIR / f"{podcast_id}.txt"
    with open(completion_marker_path, "w") as f:
        f.write(str(utc_now()))
    print(f"Marked podcast {podcast_id} as recently transcribed.")


@stub.function(
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    gpu=True,
    concurrency_limit=10,
)
def transcribe_episode(
    audio_filepath: pathlib.Path,
    result_path: pathlib.Path,
    model: ModelSpec,
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
    shared_volumes={CACHE_DIR: volume},
)
def process_episode(episode: podcast.EpisodeMetadata):
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    destination_path = RAW_AUDIO_DIR / episode.guid_hash
    podcast.store_original_audio(
        url=episode.original_download_link,
        destination=destination_path,
    )

    model = supported_whisper_models["base.en"]
    model_slug = f"whisper-{model.name.replace('.', '-')}"
    print(f"Using the {model.name} model which has {model.params} parameters.")

    metadata_path = METADATA_DIR / f"{episode.guid_hash}.json"
    with open(metadata_path, "w") as f:
        json.dump(dataclasses.asdict(episode), f)
    print(f"Wrote episode metadata to {metadata_path}")

    transcription_path = TRANSCRIPTIONS_DIR / f"{episode.guid_hash}-{model_slug}.json"
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
    shared_volumes={CACHE_DIR: volume},
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
        podcast_id = podchaser_podcast_ids[show_name]
        with stub.run() as app:
            print(f"Modal app ID -> {app.app_id}")
            transcribe_podcast(name=show_name, podcast_id=podcast_id)
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
