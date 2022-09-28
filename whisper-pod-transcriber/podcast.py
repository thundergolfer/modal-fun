import dataclasses
import os
import pathlib
import urllib.request

from typing import NamedTuple, Optional, Union


@dataclasses.dataclass
class EpisodeMetadata:
    podcast_id: Union[str, int]
    show: str  # TODO: Rename to `podcast_title`
    title: str
    publish_date: str  # The publish date of the episode as specified by the publisher
    description: str
    html_description: str
    guid: str  # The unique identifier of this episode within the context of the podcast
    guid_hash: str  # Hash the guid because Podchaser app adopted a new shit format which isn't good for filenames, eg, gid://art19-episode-locator/V0/ycwWQomSBBS6eeJxkT_94I0WTMgR2fL9XNA7vWXD1Kc
    episode_url: Optional[str]  # link to episode on Podchaser
    original_download_link: str


@dataclasses.dataclass
class PodcastMetadata:
    id: str
    title: str
    description: str
    web_url: str


class DownloadResult(NamedTuple):
    data: bytes
    content_type: str


def download_podcast_file(url: str) -> DownloadResult:
    with urllib.request.urlopen(url) as response:
        return DownloadResult(
            data=response.read(),
            content_type=response.headers["content-type"],
        )


def create_podchaser_client():
    from gql import gql, Client
    from gql.transport.aiohttp import AIOHTTPTransport

    transport = AIOHTTPTransport(url="https://api.podchaser.com/graphql")
    client = Client(transport=transport, fetch_schema_from_transport=True)
    podchaser_client_id = os.environ.get("PODCHASER_CLIENT_ID")
    podchaser_client_secret = os.environ.get("PODCHASER_CLIENT_SECRET")

    if not podchaser_client_id or not podchaser_client_secret:
        exit(
            "Must provide both PODCHASER_CLIENT_ID and PODCHASER_CLIENT_SECRET as environment vars."
        )

    query = gql(
        """
        mutation {{
            requestAccessToken(
                input: {{
                    grant_type: CLIENT_CREDENTIALS
                    client_id: "{client_id}"
                    client_secret: "{client_secret}"
                }}
            ) {{
                access_token
                token_type
            }}
        }}
    """.format(
            client_id=podchaser_client_id,
            client_secret=podchaser_client_secret,
        )
    )

    result = client.execute(query)

    access_token = result["requestAccessToken"]["access_token"]
    transport = AIOHTTPTransport(
        url="https://api.podchaser.com/graphql",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    return Client(transport=transport, fetch_schema_from_transport=True)


def search_podcast_name(gql, client, name, max_results=5):
    podcasts = []
    has_more_pages = True
    current_page = 0
    max_episodes_per_request = 100  # Max allowed by API
    while has_more_pages:
        search_podcast_name_query = gql(
            """
            query {{
                podcasts(searchTerm: "{name}", first: {max_episodes_per_request}, page: {current_page}) {{
                    paginatorInfo {{
                        currentPage,
                        hasMorePages,
                        lastPage,
                    }},
                    data {{
                        id,
                        title,
                        description,
                        webUrl,
                    }}
                }}
            }}
            """.format(
                name=name,
                max_episodes_per_request=max_episodes_per_request,
                current_page=current_page,
            )
        )
        print(f"Querying Podchaser for podcasts. Current page: {current_page}.")
        result = client.execute(search_podcast_name_query)
        has_more_pages = result["podcasts"]["paginatorInfo"]["hasMorePages"]
        podcasts_in_page = result["podcasts"]["data"]
        podcasts.extend(podcasts_in_page)
        if len(podcasts) >= max_results:
            return podcasts[:max_results]
        current_page += 1
    return podcasts


def fetch_episodes_data(gql, client, podcast_id, max_episodes=100):
    """
    NYT Episodes:
    curl https://podbay.fm/api/podcast?slug=the-ezra-klein-show-280811&page=0&reverse=false&refresh=true | jq .

    Vox.com era episodes:
    curl https://podbay.fm/api/podcast?slug=the-ezra-klein-show&reverse=true&page=0 | jq .
    """
    max_episodes_per_request = 100  # Max allowed by API
    episodes = []
    has_more_pages = True
    current_page = 0
    while has_more_pages:
        list_episodes_query = gql(
            """
            query getPodList {{
                podcast(identifier: {{id: "{id}", type: PODCHASER}}) {{
                    episodes(first: {max_episodes_per_request}, page: {current_page}) {{
                        paginatorInfo {{
                          count
                          currentPage
                          firstItem
                          hasMorePages
                          lastItem
                          lastPage
                          perPage
                          total
                        }}
                        data {{
                          id
                          title
                          airDate
                          audioUrl  
                          description
                          htmlDescription
                          guid
                          url
                        }}
                    }}
                }}
            }}
        """.format(
                id=podcast_id,
                max_episodes_per_request=max_episodes_per_request,
                current_page=current_page,
            )
        )

        print(f"Fetching {max_episodes_per_request} episodes from API.")
        result = client.execute(list_episodes_query)
        has_more_pages = result["podcast"]["episodes"]["paginatorInfo"]["hasMorePages"]
        episodes_in_page = result["podcast"]["episodes"]["data"]
        episodes.extend(episodes_in_page)
        current_page += 1
        if len(episodes) >= max_episodes:
            break
    return episodes


def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def store_original_audio(
    url: str, destination: pathlib.Path, overwrite: bool = False
) -> None:
    if destination.exists():
        if overwrite:
            print(
                f"Audio file exists at {destination} but overwrite option is specified."
            )
        else:
            print(f"Audio file exists at {destination}, skipping download.")
            return

    podcast_download_result = download_podcast_file(url=url)
    humanized_bytes_str = sizeof_fmt(num=len(podcast_download_result.data))
    print(f"Downloaded {humanized_bytes_str} episode from URL.")
    with open(destination, "wb") as f:
        f.write(podcast_download_result.data)

    print(f"Stored audio episode at {destination}.")
