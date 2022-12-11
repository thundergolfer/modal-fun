import os
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

image = modal.Image.debian_slim().pip_install_from_requirements(
    requirements_txt="./requirements.txt"
)
stub = modal.Stub(name="thundergolferdotcom-email-subs")
stub.confirmation_code_to_email = modal.Dict()
web_app = FastAPI()

# Check relatively frequently for new posts, because subscribers should
# be among first to hear of a new post.
@stub.function(schedule=modal.Period(hours=3))
def check_for_new_post():
    # If new, unseen post:
    # 1. mark post as seen
    # 2. send out emails to all confirmed subscribers
    pass


@stub.function
def send_confirmation_email(email: str):
    pass


@web_app.get("/confirm")
def confirm(email: str, code: str):
    pass


@web_app.get("/unsubscribe")
def unsubscribe(email: str, code: str):
    # Check code against email. If match, unsubscribe user
    # and send back HTML page showing them they were unsubscribed.
    pass


@web_app.get("/subscribe")
def subscribe(email: str):
    # 1. check if email is already subscribed
    # 2. send confirmation email if not
    send_confirmation_email.spawn(email="")
    return {"hello": "world"}


@stub.asgi
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


# If modifying these scopes, delete the file token.json.
SCOPES = [
    # Used for sanity-checking OAuth success
    "https://www.googleapis.com/auth/gmail.readonly",
    # Required to send out the 'new blog post' email notifications
    "https://www.googleapis.com/auth/gmail.send",
]


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
    main()
    # stub.serve()
