import base64
import email
import email.utils
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.message import Message
from typing import Optional

import modal

from email_subs import datastore
from email_subs import emailer
from email_subs.main import (
    app_name,
    CACHE_DIR,
    DB_PATH,
    image,
    fetch_fresh_gmail_creds_from_env,
    notify_subscribers_of_new_posts_impl,
    send_confirmation_email,
    web,
)

# Test-specific Modal objects:
test_image = image.pip_install(
    "httpx~=0.23.3"
)  # extend prod image to include `httpx` for testing.
# Doesn't need to be persisted, just lives for life of test.
volume = modal.SharedVolume()
stub = modal.Stub(name=f"{app_name}-test", image=test_image)

# Register application functions with Modal test stub.
#
# uses test-specific url and volume
web_app_handle = stub.asgi(shared_volumes={CACHE_DIR: volume})(web.get_raw_f())
# uses test-specific volume
stub.function(
    secret=modal.Secret.from_name("gmail"),
    shared_volumes={CACHE_DIR: volume},
)(send_confirmation_email.get_raw_f())
# avoids arity-restriction of Modal cron functions. we need to pass in a test RSS feed URL.
notify_subscribers_of_new_posts = stub.function(
    secret=modal.Secret.from_name("gmail"),
    shared_volumes={CACHE_DIR: volume},
)(notify_subscribers_of_new_posts_impl)

# The web URL matching regex used by Markdown. ref:
#   http://daringfireball.net/2010/07/improved_regex_for_matching_urls
#   https://gist.github.com/gruber/8891611
URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""


def wait_for_email_sending() -> None:
    wait_secs = 10
    print(f"Waiting {wait_secs}s for emails to be delivered.")
    time.sleep(wait_secs)


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


@stub.function(shared_volumes={CACHE_DIR: volume})
def clear_notifications():
    store = datastore.Datastore(
        conn=datastore.get_db(DB_PATH),
        codegen_fn=lambda: str(uuid.uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    store.delete_notifications()
    assert len(store.list_notifications()) == 0


@stub.webhook(method="GET")
def fake_rss_feed():
    """Fake RSS feed web endpoint used only during test execution."""
    rfc_822_fmt = "%a, %d %b %Y %H:%M:%S %z"
    now = datetime.now(timezone.utc)
    recent_pub_datetime: str = now.strftime(rfc_822_fmt)
    old_pub_datetime = (now - timedelta(days=10)).strftime(rfc_822_fmt)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
  <title>Test RSS feed [thundergolferdotcom-email-subs]</title>
  <link>https://www.abc123.com</link>
  <description>This is a hardcoded feed used for integration testing.</description>
  <item>
    <title>Send Me Too Your Subscribers!</title>
    <description>I am new and should be sent to subscribers.</description>
    <pubDate>{recent_pub_datetime}</pubDate>
    <link>https://abc123.com/bar/</link>
    <guid isPermaLink="true">https://abc123.com/bar/</guid>
  </item>
  <item>
    <title>I am an old post and shouldn't be sent to subscribers</title>
    <description>Don't send me, I'm old. Leave me in peace.</description>
    <pubDate>{old_pub_datetime}</pubDate>
    <link>https://abc123.com/foo/</link>
    <guid isPermaLink="true">https://abc123.com/foo/</guid>
  </item>
</channel>
</rss>"""


@stub.function(
    secret=modal.Secret.from_name("gmail"),
)
def fetch_recent_emails(n: int = 3) -> list[Message]:
    creds = fetch_fresh_gmail_creds_from_env()
    sender = emailer.GmailSender(creds)
    gmail_service = sender.service
    results = gmail_service.users().messages().list(userId="me", maxResults=n).execute()
    gmail_messages = results.get("messages", [])
    messages = []
    for m in gmail_messages:
        r = (
            gmail_service.users()
            .messages()
            .get(userId="me", id=m["id"], format="raw")
            .execute()
        )
        msg_str = base64.urlsafe_b64decode(r["raw"]).decode("UTF-8")
        messages.append(email.message_from_string(str(msg_str)))
    return messages


@stub.function(
    secret=modal.Secret.from_name("gmail"),
)
def trash_email(msg_id: str) -> list[str]:
    """NOTE: Requires delete scope in Gmail."""
    creds = fetch_fresh_gmail_creds_from_env()
    sender = emailer.GmailSender(creds)
    gmail_service = sender.service
    msg = gmail_service.users().messages().trash(userId="me", id=msg_id).execute()
    print(f"Trashed: {msg}")


def _find_endpoint_url(*, msg: str, endpoint: str) -> Optional[str]:
    all_urls = re.findall(URL_REGEX, msg)
    matching_urls = [u for u in all_urls if f"/{endpoint}?" in u]
    if len(matching_urls) > 1:
        raise RuntimeError(
            f"Unexpectedly found more than one URL pointing at endpoint '{endpoint}'"
        )
    elif not matching_urls:
        return None
    return matching_urls[0]


# TODO: Use @pytest.mark.modal when that's supported.
def test_end_to_end():
    import httpx

    # The personal email associated with my Gmail auth.
    test_email_addr = "jonathon.i.belotti@gmail.com"
    test_web_url = web_app_handle.web_url

    # 1. Create a new test DB
    create_test_db.call()

    # 2. Call /subscribe with email
    print("Hitting /subscribe endpoint.")
    httpx.get(f"{test_web_url}/subscribe?email={test_email_addr}")

    wait_for_email_sending()

    # 3. Read email from my Gmail and get confirmation link
    messages = fetch_recent_emails.call()
    assert len(messages) > 0
    confirm_url = None
    confirm_email_id = None
    for msg in messages:
        confirm_url = _find_endpoint_url(msg=msg.as_string(), endpoint="confirm")
        confirm_email_id = msg.get("Message-Id")
        if confirm_url:
            break

    assert confirm_url
    assert confirm_email_id

    # 4. GET the confirmation link to confirm subscription
    print("Hitting /confirm endpoint.")
    httpx.get(confirm_url)

    # 4.2 Delete the confirmation email
    trash_email.spawn(msg_id=confirm_email_id)

    # 5. Use fake RSS to simulate new blog post and run notification cron
    feed_url = fake_rss_feed.web_url
    notify_subscribers_of_new_posts.call(feed_url)

    wait_for_email_sending()

    # 6. Check email blog update received in my Gmail
    messages = fetch_recent_emails.call()
    assert len(messages) > 0
    unsubscribe_url, new_post_email_date, new_post_email_id = None, None, None
    for msg in messages:
        unsubscribe_url = _find_endpoint_url(
            msg=msg.as_string(), endpoint="unsubscribe"
        )
        if unsubscribe_url:
            assert "Send Me Too Your Subscribers!" in msg.as_string()
            new_post_email_date = email.utils.parsedate_tz(msg.get("Date"))
            new_post_email_id = msg.get("Message-Id")
            break

    assert all([unsubscribe_url, new_post_email_id, new_post_email_date])

    # 7. Hit the unsubscribe link from the email
    print("Hitting /unsubscribe endpoint.")
    httpx.get(unsubscribe_url)

    # 7.2 Delete email
    trash_email.spawn(msg_id=new_post_email_id)

    # 8. Use fake RSS again to simulate another post
    clear_notifications.call()
    notify_subscribers_of_new_posts.call(feed_url)

    wait_for_email_sending()

    # 9. Check no new email was sent after unsubscribing.
    for msg in messages:
        unsubscribe_url = _find_endpoint_url(
            msg=msg.as_string(), endpoint="unsubscribe"
        )
        if unsubscribe_url:
            assert "Send Me Too Your Subscribers!" in msg.as_string()
            # check its the email sent previously
            curr_email_date = email.utils.parsedate_tz(msg.get("date"))
            if new_post_email_date < curr_email_date:
                raise AssertionError(
                    "The unsubscribe/ endpoint seems to have failed to work!"
                    f"{curr_email_date} is newer than {new_post_email_date}, so "
                    "another email was sent after unsubcribe/ endpoint was hit."
                )


if __name__ == "__main__":
    with stub.run():
        raise SystemExit(test_end_to_end())
