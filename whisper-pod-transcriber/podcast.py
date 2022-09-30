import dataclasses
import os
import pathlib
import urllib.request

from typing import NamedTuple, Optional, Union


@dataclasses.dataclass
class EpisodeMetadata:
    # Unique ID of podcast this episode is associated with.
    podcast_id: Union[str, int]
    # Title of podcast this episode is associated with.
    show: str  # TODO: Rename to `podcast_title`
    title: str
    # The publish date of the episode as specified by the publisher
    publish_date: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    description: str
    # HTML markup description. Suitable for display in UI.
    html_description: str
    # The unique identifier of this episode within the context of the podcast
    guid: str
    # Hash the guid into something appropriate for filenames.
    guid_hash: str  
    # Link to episode on Podchaser website.
    episode_url: Optional[str]
    # Link to audio file for episode. Typically an .mp3 file.
    original_download_link: str


@dataclasses.dataclass
class PodcastMetadata:
    # Unique ID for a podcast
    id: str
    # Title of podcast, eg. 'The Joe Rogan Experience'.
    title: str
    # Plaintext description of episode. nb: has whitespace issues so not suitable in UI.
    # TODO: Grab html_description as well.
    description: str
    # Link to podcast on Podchaser website.
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
    """
    Use's Podchaser's graphql API to get an new access token and instantiate
    a graphql client with it.
    """
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


def search_podcast_name(gql, client, name, max_results=5) -> list[dict]:
    """Search for a podcast by name/title. eg. 'Joe Rogan Experience' or 'Serial'."""
    podcasts = []
    has_more_pages = True
    current_page = 0
    # TODO: Remove pagination. Just need one request.
    max_episodes_per_request = max_results
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


def fetch_episodes_data(gql, client, podcast_id, max_episodes=100) -> list[dict]:
    """
    Use the Podchaser API to grab a podcast's episodes.
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
