import os
import time
import uuid
import sys
from datetime import datetime, timezone
from typing import NamedTuple

import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
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

modal_workspace_username = "thundergolfer"
app_name = "thundergolferdotcom-email-subs"
nfs = modal.NetworkFileSystem.persisted(f"{app_name}-vol")
image = modal.Image.debian_slim().pip_install_from_requirements(
    requirements_txt="./requirements.txt"
)
stub = modal.Stub(name=app_name, image=image)
web_app = FastAPI()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_age(dt: datetime, base: datetime) -> float:
    return (base - dt).total_seconds()


class Config(NamedTuple):
    """
    Configuration for the app.
    Change these values to be correct for your deployment.
    """

    # The website that from which your users do their subscription by calling the
    # webhook. This value is used to setup CORS properly on the webhook.
    personal_website_domain: str
    # This is used as the FROM address in sent mail, and is associated with
    # the authenticated GMail session.
    maintainer_gmail_address: str
    # URL for blog's RSS feed
    rss_feed_url: str
    # Used in setup and testing. Can be the same as the maintainer email address.
    test_email_address: str
    # Used in email copy
    twitter_username: str


class BlogEntry(NamedTuple):
    title: str
    link: str
    published_datetime: datetime


config = Config(
    personal_website_domain="thundergolfer.com",
    maintainer_gmail_address="jonathon.i.belotti@gmail.com",
    rss_feed_url="https://thundergolfer.com/feed.xml",
    test_email_address="jonathon.bel.melbourne@gmail.com",
    twitter_username="jonobelotti_IO",
)


def fetch_fresh_gmail_creds_from_env():
    creds = Credentials.from_authorized_user_info(
        info={
            "refresh_token": os.environ["GMAIL_AUTH_REFRESH_TOKEN"],
            "client_id": os.environ["GMAIL_AUTH_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_AUTH_CLIENT_SECRET"],
        },
        scopes=SCOPES,
    )
    if creds and creds.expired and creds.refresh_token:
        print("Refreshing credentials...")
        creds.refresh(Request())
    return creds


def fetch_blog_posts_from_rss(feed_url: str) -> list[BlogEntry]:
    import feedparser

    feed = feedparser.parse(feed_url)
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


@stub.function(network_file_systems={CACHE_DIR: nfs})
def setup_db():
    """
    Only need to run this once for a Modal app.
    Creates and initializes an SQLite DB on a Modal persistent volume.
    """
    print("Setting up new DB... (this should only run once)")
    conn = datastore.get_db(DB_PATH)
    datastore.init(conn)
    print("☑️ DB setup done")


@stub.function(network_file_systems={CACHE_DIR: nfs})
def reset_db(notifications=False, subs=False):
    """
    ⚠️ Only use during testing. Don't reset production DB as it could cause
    duplication notifications to be sent to subscribers.
    """
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    if notifications and subs:
        store.delete_everything()
        print("Deleted everything.")
    elif notifications:
        store.delete_notifications()
        print(
            f"Deleted notifications. There are now {len(store.list_notifications())} notifications."
        )
    else:
        print("Deleting nothing.")


def notify_subscribers_of_new_posts_impl(feed_url):
    """
    Implementation broken out to facilitate testing.

    TODO: Merge back into scheduled modal.Function when those support parameters.
    """
    # Create emailer
    creds = fetch_fresh_gmail_creds_from_env()
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
    posts = fetch_blog_posts_from_rss(feed_url)
    print(f"Got {len(posts)} from RSS feed.")
    for post in posts:
        if utc_age(post.published_datetime, now) > two_days_in_secs:
            # Defensively ignoring old blog posts. Likely shouldn't push these out.
            print(f"'{post.title}' too old 👴🏼")
            continue
        if post.link in seen_links:
            print(f"'{post.title}' already sent to subscribers")
            continue
        posts_for_notification.append(post)

    if not posts_for_notification:
        print(f"No new posts @ {now}. Done for now.")
        return
    else:
        print(
            f"Found {len(posts_for_notification)} recent posts not yet sent to subscribers."
        )

    active_subs = store.list_subs()
    # Important: register notifications in DB.
    # TODO: Ensure DB locked for writes.
    for p in posts_for_notification:
        store.create_notification(
            link=p.link, recipients=[s.email for s in active_subs]
        )

    print(
        f"Sending new post notification email to {len(active_subs)} active email subscribers."
    )
    live_web_url = web.web_url
    for subscriber in active_subs:
        code = subscriber.unsub_code
        unsub_link = f"{live_web_url}/unsubscribe?code={code}&email={subscriber.email}"
        copy = email_copy.construct_new_blogpost_email(
            blog_url=f"https://{config.personal_website_domain}",
            blog_name=config.personal_website_domain,
            blog_links=[p.link for p in posts_for_notification],
            blog_titles=[p.title for p in posts_for_notification],
            unsubscribe_link=unsub_link,
            twitter_url=f"https://twitter.com/{config.twitter_username}",
        )
        emailer.send(
            sender=sender,
            subject=copy.subject,
            content=copy.body,
            from_addr=config.maintainer_gmail_address,
            recipient=subscriber.email,
        )
    print("✅ Done!")


@stub.function(
    # Check relatively frequently for new posts, because subscribers should
    # be among first to hear of a new post.
    schedule=modal.Period(hours=3),
    network_file_systems={CACHE_DIR: nfs},
    secret=modal.Secret.from_name("gmail"),
)
def notify_subscribers_of_new_posts():
    """
    Cronjob function that checks for new blog posts and if a new one is found
    sends email notifications to all confirmed subscribers.
    """
    notify_subscribers_of_new_posts_impl(config.feed_url)


@stub.function(
    secret=modal.Secret.from_name("gmail"),
    network_file_systems={CACHE_DIR: nfs},
)
def send_confirmation_email(email: str):
    creds = Credentials.from_authorized_user_info(
        info={
            "refresh_token": os.environ["GMAIL_AUTH_REFRESH_TOKEN"],
            "client_id": os.environ["GMAIL_AUTH_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_AUTH_CLIENT_SECRET"],
        },
        scopes=SCOPES,
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
    live_web_url = web.web_url
    confirm_link = f"{live_web_url}/confirm?email={subscriber.email}&code={code}"
    sender = emailer.GmailSender(creds)
    copy = email_copy.confirm_subscription_email(
        blog_name=config.personal_website_domain,
        blog_url=f"https://{config.personal_website_domain}/blog",
        confirmation_link=confirm_link,
    )
    emailer.send(
        sender=sender,
        subject=copy.subject,
        content=copy.body,
        from_addr=config.maintainer_gmail_address,
        recipient=subscriber.email,
    )


@web_app.get("/wake")
def wake():
    """Used to alleviate serverless cold-starts."""
    return "Hello!"


@web_app.get("/confirm")
def confirm(email: str, code: str):
    """
    Used by email subscribers to confirm their subscription.
    Requires email and code values in query params, which are populated
    for the user in the subscription confirmation email they're sent.
    """
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
    return {"message": f"subscription confirmed for '{email}'"}


@web_app.get("/unsubscribe")
def unsubscribe(email: str, code: str):
    """
    Unsubscribes an email from future updates. 'Unsubscribe' links
    using this endpoint are provided in every subscription email, in
    accordance with email provider requirements.

    ref: https://support.google.com/mail/answer/81126?hl=en
    """
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
    """
    Creates a new subscription for an email and sends a confirmation email
    to that email so that the subscription can be confirmed.
    """
    from fastapi import HTTPException

    if not email:
        raise HTTPException(status_code=400, detail="email cannot be empty")
    send_confirmation_email.spawn(email=email)
    return {"message": f"Confirmation email sent to '{email}'"}


@stub.function(
    # Web app uses datastore to confirm subscriptions and fulfil unsubscriptions.
    network_file_systems={CACHE_DIR: nfs},
)
@modal.asgi_app()
def web():
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{config.personal_website_domain}",
            f"https://{config.personal_website_domain}",
            # Localhost used for development and testing.
            # You may need to change the PORT to something else.
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


def create_refresh_token_and_test_creds():
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
            from_addr=config.maintainer_gmail_address,
            recipient=config.test_email_address,  # Send to yourself
        )
    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "create-refresh-token":
        with stub.run():
            create_refresh_token_and_test_creds()
    elif len(sys.argv) == 0:
        stub.serve()
    else:
        print(
            "usage: python3 -m email_subs.main [create-refresh-token]", file=sys.stderr
        )
