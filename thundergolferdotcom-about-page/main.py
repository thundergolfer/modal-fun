import base64
import dataclasses
import json
import os
import urllib.parse
import urllib.request

import modal

stub = modal.Stub(name="thundergolferdotcom-about-page")
bs4_image = modal.Image.debian_slim().pip_install(["beautifulsoup4"])


@dataclasses.dataclass(frozen=True)
class SpotifyTrack:
    name: str
    artist: str
    link: str


@dataclasses.dataclass(frozen=True)
class Book:
    title: str
    authors: list[str]
    # Goodreads.com link to the book
    link: str
    cover_image_link: str
    # The dominant color of the book's cover, for use in CSS rendering
    # of the book.
    cover_image_color: tuple[float, float, float]


@dataclasses.dataclass
class TwitterInfo:
    display_name: str
    handle: str
    byline: str
    link: str


@dataclasses.dataclass
class GithubContribution:
    repo: str
    date: str
    link: str


@dataclasses.dataclass
class GithubInfo:
    username: str
    # List of 'active days in the last N days' stats. (active, total)
    active_days: list[tuple[int, int]]
    # List of N most recent public contributions on Github
    recent_contributions: list[GithubContribution]
    total_stars: int


@dataclasses.dataclass()
class AboutMeStats:
    # Top N most listened tracks on my Spotify account.
    spotify: list[SpotifyTrack]
    # N most recently finished bookreads on my Goodreads account
    goodreads: list[Book]
    # Profile data for my Twitter account
    twitter: TwitterInfo
    # Github profile data and activity stats
    github: GithubInfo


SPOTIFY_CLIENT_ID = "a38982a07d3c4071967f35b5e84ef599"


@stub.function(secret=modal.Secret.from_name("spotify-aboutme"))
def create_spotify_refresh_token(code: str):
    auth_str = (
        os.environ["SPOTIFY_CLIENT_ID"] + ":" + os.environ["SPOTIFY_CLIENT_SECRET"]
    )
    encoded_client_id_and_secret = base64.b64encode(auth_str.encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=urllib.parse.urlencode(
            {
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": "http://localhost:3000/callback",
            }
        ).encode(),
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36",
            "Authorization": f"Basic {encoded_client_id_and_secret}",
        },
    )
    response = urllib.request.urlopen(req).read().decode()
    return json.load(response)["refresh_token"]


def manual_spotify_auth() -> None:
    """Instrunctions: https://leerob.io/blog/spotify-api-nextjs"""
    redirect_uri = urllib.parse.quote(
        "http://localhost:3000/callback", safe=""
    )  # This must match what's in the Spotify app's settings
    authorize_url = (
        "https://accounts.spotify.com/"
        f"authorize?client_id={SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri={redirect_uri}"
        "&scope=user-read-currently-playing%20user-top-read"
    )

    code = input(
        f"Visit {authorize_url} and then paste back the resulting code in the URL.\nCode: "
    ).strip()
    with stub.run():
        refresh_token = create_spotify_refresh_token(code)
    print(f"SPOTIFY_REFRESH_TOKEN: {refresh_token}")
    print(
        "Save the refresh_token back into the `spotify-aboutme` secret in Modal as SPOTIFY_REFRESH_TOKEN"
    )


@stub.function(secret=modal.Secret.from_name("spotify-aboutme"))
def request_spotify_top_tracks(max_tracks=5) -> list[SpotifyTrack]:
    client_auth = (
        os.environ["SPOTIFY_CLIENT_ID"] + ":" + os.environ["SPOTIFY_CLIENT_SECRET"]
    )
    refresh_token = os.environ["SPOTIFY_REFRESH_TOKEN"]
    encoded_client_id_and_secret = base64.b64encode(client_auth.encode()).decode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=urllib.parse.urlencode(
            {
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode(),
        headers={
            "Authorization": f"Basic {encoded_client_id_and_secret}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    response = urllib.request.urlopen(req).read().decode()
    access_token = json.loads(response)["access_token"]

    req = urllib.request.Request(
        "https://api.spotify.com/v1/me/top/tracks",
        headers={
            "Authorization": f"Bearer {access_token}",
        },
    )
    response = urllib.request.urlopen(req).read().decode()
    data = json.loads(response)
    tracks = data["items"]
    top_tracks = []
    for tr in tracks[:max_tracks]:
        top_tracks.append(
            SpotifyTrack(
                name=tr["name"],
                artist=", ".join(a["name"] for a in tr["artists"]),
                link=tr["external_urls"]["spotify"],
            )
        )
    return top_tracks


@stub.function(image=bs4_image)
def request_goodreads_reads(max_books=5) -> list[Book]:
    """
    Setting @stub.function(interactive=True, ...) was really helpful in writing this function.
    """
    from bs4 import BeautifulSoup

    url = "https://www.goodreads.com/review/list/88184044?shelf=read&sort=date_read"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36",
        },
    )
    html_doc = urllib.request.urlopen(req).read().decode()
    soup = BeautifulSoup(html_doc, "html.parser")
    # print(soup.prettify())

    books_table = soup.find("tbody", {"id": "booksBody"})
    books = []
    for book_item in books_table.find_all("tr", limit=max_books):
        title_data = book_item.find("td", {"class": "title"})
        title_link = title_data.find("a")
        title = title_link.get("title").strip()
        book_href = "https://goodreads.com" + title_link.get("href")

        cover_data = book_item.find("td", {"class": "cover"})
        cover_link = cover_data.find("img").get("src")
        # Resize the cover:
        # The thumbnail in the results are tiny, but can be enlarged by modifying the SY25
        # url part to be a bigger number, like SY600.
        cover_link = cover_link.replace("SY75", "SY600")

        author_data = book_item.find("td", {"class": "author"})
        author = author_data.find("a").get_text().strip()
        books.append(
            Book(
                title=title,
                authors=[author],
                link=book_href,
                cover_image_link=cover_link,
                cover_image_color=(-1.0, -1.0, -1.0),  # TODO: Provide valid data
            )
        )
    return books


@stub.webhook(secret=modal.Secret.from_name("spotify-aboutme"))
def about_me():
    stats = AboutMeStats(
        spotify=request_spotify_top_tracks(),
        goodreads=request_goodreads_reads(),
        twitter=TwitterInfo(
            display_name="Jonathon Belotti",
            handle="thundergolfer",
            byline="",
            link="",
        ),
        github=GithubInfo(
            username="thundergolfer",
            active_days=[],
            recent_contributions=[],
            total_stars=1_000_000_000,
        ),
    )
    return dataclasses.asdict(stats)


if __name__ == "__main__":
    stub.serve()
    # manual_spotify_auth()

    # with stub.run():
    #     print(request_goodreads_reads())

    # with stub.run():
    # request_spotify_top_tracks()
