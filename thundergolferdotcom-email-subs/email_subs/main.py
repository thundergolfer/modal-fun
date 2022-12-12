import os
import time
import uuid
from datetime import datetime, timezone
from typing import NamedTuple

import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from . import datastore
from . import emailer
from . import email_copy

CACHE_DIR = "/cache"
DB_PATH = CACHE_DIR + "/emailsubs.db"
# If modifying these scopes, delete the file token.json.
SCOPES = [
    # Used for sanity-checking OAuth success
    "https://www.googleapis.com/auth/gmail.readonly",
    # Required to send out the 'new blog post' email notifications
    "https://www.googleapis.com/auth/gmail.send",
]

app_name = "thundergolferdotcom-email-subs"
volume = modal.SharedVolume().persist(f"{app_name}-vol")
image = modal.Image.debian_slim().pip_install_from_requirements(
    requirements_txt="./requirements.txt"
)
stub = modal.Stub(name=app_name, image=image)
stub.confirmation_code_to_email = modal.Dict()
web_app = FastAPI()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_age(dt: datetime, base: datetime) -> float:
    return (base - dt).total_seconds()
    

class BlogEntry(NamedTuple):
    title: str
    link: str
    published_datetime: datetime


def fetch_my_blog_posts_from_rss() -> list[BlogEntry]:
    import feedparser

    feed = feedparser.parse("https://thundergolfer.com/feed.xml")
    return [
        BlogEntry(
            title=entry["title"],
            link=entry["link"],
            published_datetime=datetime.fromtimestamp(
                time.mktime(entry["published_parsed"]),
                timezone.utc,
            ),
        )
        for entry in feed.entries
    ]


@stub.function(shared_volumes={CACHE_DIR: volume})
def setup_db():
    """
    Only need to run this once for a Modal app. 
    Creates and initializes an SQLite DB on a Modal persistent volume.
    """
    print("Setting up new DB... (this should only run once)")
    conn = datastore.get_db(DB_PATH)
    datastore.init(conn)
    print("☑️ DB setup done")


@stub.function(shared_volumes={CACHE_DIR: volume})
def reset_db():
    """
    ⚠️ Only use during testing. Don't reset production DB as it could cause
    duplication notifications to be sent to subscribers.
    """
    raise NotImplementedError()


# Check relatively frequently for new posts, because subscribers should
# be among first to hear of a new post.
@stub.function(
    schedule=modal.Period(hours=3), 
    shared_volumes={CACHE_DIR: volume},
    secret=modal.Secret.from_name("gmail"),
)
def notify_subscribers_of_new_posts():
    # Create emailer
    creds = Credentials.from_authorized_user_info(
        info={
            "refresh_token": os.environ["GMAIL_AUTH_REFRESH_TOKEN"],
            "client_id": os.environ["GMAIL_AUTH_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_AUTH_CLIENT_SECRET"],
        }, 
        scopes=SCOPES
    )
    sender = emailer.GmailSender(creds)

    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    notifications = store.list_notifications()
    seen_links = set(n.blogpost_link for n in notifications)

    now = utc_now()
    two_days_in_secs = 60 * 60 * 24 * 2
    posts_for_notification: list[BlogEntry] = []
    posts = fetch_my_blog_posts_from_rss()
    print(f"Got {len(posts)} from RSS feed.")
    for post in posts:
        if utc_age(post.published_datetime, now) > two_days_in_secs:
            # Defensively ignoring old blog posts. Likely shouldn't push these out.
            print(f"'{post.title}' too old")
            continue
        if post.link in seen_links:
            print(f"'{post.title}' already sent to subscribers")
            continue
        posts_for_notification.append(post)

    if not posts_for_notification:
        print(f"No new posts @ {now}. Done for now.")
    else:
        print(f"Found {len(posts_for_notification)} recent posts not yet sent to subscribers.")

    active_subs = store.list_subs()
    # Important: register notifications in DB.
    # TODO: Ensure DB locked for writes.
    for p in posts_for_notification:
        store.create_notification(
            link=p.link,
            recipients=[s.email for s in active_subs]
        )

    print(f"Sending new post notification email to {len(active_subs)} active email subscribers.")
    for subscriber in active_subs:
        code = subscriber.unsub_code
        unsub_link = f"https://thundergolfer--{app_name}-web.modal.run/unsubscribe?code={code}"
        copy = email_copy.construct_new_blogpost_email(
            blog_links=[p.link for p in posts_for_notification],
            blog_titles=[p.title for p in posts_for_notification],
            unsubscribe_link=unsub_link,
        )
        emailer.send(
            sender=sender,
            subject=copy.subject,
            content=copy.body,
            from_addr="jonathon.i.belotti@gmail.com",
            recipient=subscriber.email,
        )
    print("✅ Done!")
        


@stub.function(
    secret=modal.Secret.from_name("gmail"),
    shared_volumes={CACHE_DIR: volume},
)
def send_confirmation_email(email: str):
    creds = Credentials.from_authorized_user_info(
        info={
            "refresh_token": os.environ["GMAIL_AUTH_REFRESH_TOKEN"],
            "client_id": os.environ["GMAIL_AUTH_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_AUTH_CLIENT_SECRET"],
        }, 
        scopes=SCOPES
    )

    # Create subscriber
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )

    subscriber = store.create_sub(email=email)
    code = subscriber.confirm_code
    confirm_link = f"https://thundergolfer--{app_name}-web.modal.run/confirm?code={code}"
    sender = emailer.GmailSender(creds)
    copy = email_copy.confirm_subscription_email(confirm_link)
    emailer.send(
        sender=sender,
        subject=copy.subject,
        content=copy.body,
        from_addr="jonathon.i.belotti@gmail.com",
        recipient=subscriber.email,
    )


@web_app.get("/confirm")
def confirm(email: str, code: str):
    from fastapi import HTTPException
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    try:
        confirmed = store.confirm_sub(email=email, code=code)
        assert confirmed
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    return {
        "message": f"subscription confirmed for '{email}'"
    }



@web_app.get("/unsubscribe")
def unsubscribe(email: str, code: str):
    # Check code against email. If match, unsubscribe user
    # and send back HTML page showing them they were unsubscribed.
    from fastapi import HTTPException
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    try:
        unsubbed = store.unsub(email=email, code=code)
        assert unsubbed
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    return {
        "message": f"'{email}' is unsubscribed successfully from thundergolfer.com/blog"
    }


@web_app.get("/subscribe")
def subscribe(email: str):
    # 1. check if email is already subscribed
    # 2. send confirmation email if not
    send_confirmation_email.spawn(email=email)
    return {"message": f"Confirmation email sent to '{email}'"}


@stub.asgi()
def web():
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://thundergolfer.com",
            "https://thundergolfer.com",
            "http://localhost:4000",
            "http://localhost:4000/",
            "localhost:4000",
            "localhost:4000/",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return web_app


def _check_labels():
    # Call the Gmail API
    service = build("gmail", "v1", credentials=creds)
    results = service.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])

    if not labels:
        print("No labels found.")
        return
    print("Labels:")
    for label in labels:
        print(label["name"])


def main():
    """
    Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        sender = emailer.GmailSender(creds)
        emailer.send(
            sender=sender,
            subject="Testy McTestFace",
            content="Hello from Modal script!",
            from_addr="jonathon.i.belotti@gmail.com",
            recipients=[
                "jonathon@modal.com",
            ],
        )
    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    # main()
    stub.serve()
    # with stub.run():
    #     notify_subscribers_of_new_posts.call()
