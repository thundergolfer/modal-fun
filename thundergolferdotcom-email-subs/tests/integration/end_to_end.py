import base64
import uuid
import pathlib
from datetime import datetime, timezone

import modal

from email_subs import datastore
from email_subs import emailer
from email_subs.main import (
    app_name,
    CACHE_DIR,
    DB_PATH,
    image,
    fetch_fresh_gmail_creds_from_env,
    web,
)

test_app_name = app_name + "-test"
volume = modal.SharedVolume()  # Doesn't need to be persisted.
stub = modal.Stub(name=test_app_name, image=image)


web_app_handle = stub.asgi(shared_volumes={CACHE_DIR: volume})(web.get_raw_f())


@stub.function(shared_volumes={CACHE_DIR: volume})
def create_test_db():
    conn = datastore.get_db(DB_PATH)
    datastore.init(conn)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    assert len(store.list_subs()) == 0


@stub.function(
    secret=modal.Secret.from_name("gmail"),
)
def fetch_recent_emails(n: int = 3) -> list[str]:
    creds = fetch_fresh_gmail_creds_from_env()
    sender = emailer.GmailSender(creds)
    gmail_service = sender.service
    results = gmail_service.users().messages().list(userId="me", maxResults=n).execute()
    messages = results.get("messages", [])
    message_bodies = []
    for m in messages:
        r = (
            gmail_service.users()
            .messages()
            .get(userId="me", id=m["id"], format="full")
            .execute()
        )
        body_text = base64.b64decode((r["payload"]["body"]["data"]))
        message_bodies.append(body_text)
    return message_bodies


# TODO: Use @pytest.mark.modal when that's supported.
def test_end_to_end():
    with stub.run():
        test_web_url = web_app_handle.web_url
        print(test_web_url)

        assert not pathlib.Path(DB_PATH).exists()
        # 1. Create a new test DB
        create_test_db.call()

        # 2. Call /subscribe with email

        # 3. Read email from my Gmail and get confirmation link
        messages = fetch_recent_emails.call()

        assert len(messages) > 0

        # 4. GET the confirmation link to confirm subscription

        # 5. Use fake RSS to simulate new blog post and run notification cron

        # 6. Check email blog update received in my Gmail

        # 7. Get the unsubscribe link from the email update and hit it

        # 8. Use fake RSS to simulate another post

        # 9. Check no new email

        # 10. Clean up DB
        pass


if __name__ == "__main__":
    raise SystemExit(test_end_to_end())
