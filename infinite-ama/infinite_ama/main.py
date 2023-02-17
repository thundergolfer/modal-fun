import os
import time
from datetime import datetime

import modal
from pydantic import BaseModel

from .ama_data import QANDA_SNIPPETS
from .chatbot import get_new_chain1
from .config import USER_SETTINGS
from .datastore import (
    bulk_insert,
    create_db_tables,
    get_engine,
    make_id,
    HnComment,
    PostType,
)
from .ingest import ingest_data, ingest_examples

image = modal.Image.debian_slim().pip_install(
    "httpx",
    "langchain~=0.0.85",
    "loguru",
    "openai~=0.26.5",
    "psycopg2-binary",
    "sqlalchemy",
    "tiktoken~=0.2.0",
    "weaviate-client~=3.11.0",
)
stub = modal.Stub(
    name="infinite-ama", image=image, secrets=[modal.Secret.from_name("neondb")]
)

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "<dbname>",
        "USER": "<user>",
        "PASSWORD": "<password>",
        "HOST": "<endpoint_hostname>",
        "PORT": "<port>",
    }
}


class Item(BaseModel):
    text: str
    # list[(human, ai)]
    history: list[list[str]] = []


@stub.webhook(
    method="POST",
    label="infinite-ama",
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("weaviate"),
    ],
)
def chatbot(item: Item):
    import weaviate
    from langchain.vectorstores import Weaviate

    client = weaviate.Client(
        url=os.environ["WEAVIATE_URL"],
        additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
    )
    vectorstore = Weaviate(client, "Paragraph", "content", attributes=["source"])
    chain = get_new_chain1(vectorstore)
    result = chain({"question": item.text, "chat_history": item.history})
    return {"answer": result["answer"]}


@stub.function(
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("weaviate"),
    ]
)
def ingest():
    """
    Run this ad-hoc with `modal run`. It will setup the chat bot's
    knowledge base data.
    """
    ingest_data(
        weaviate_url=os.environ["WEAVIATE_URL"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        docs=QANDA_SNIPPETS,
    )
    ingest_examples(
        weaviate_url=os.environ["WEAVIATE_URL"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    print("Done!")


def db_config_from_env() -> dict[str, str]:
    required_keys = {"PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"}
    import os

    extracted_env = {k: os.environ[k] for k in required_keys if k in os.environ}

    missing_keys = required_keys - set(extracted_env.keys())
    if missing_keys:
        raise RuntimeError(
            f"Missing required environment variables: {missing_keys}. "
            "Did you forget to add a modal.Secret, or are some keys missing from "
            "the provided modal.Secret?"
        )
    return {k.replace("PG", "").lower(): v for k, v in extracted_env.items()}


def ingest_hn_comments(user: str, engine):
    """
    Use the user API endpoint to get a user's full list of comments and submissions,
    and then filter for comments and ingest only new comments.
    {
        about: "Currently Data Engineering at Canva. Previously at Zendesk and Atlassian.<p>Sydney, Australia",
        created: 1500343271,
        id: "thundergolfer",
        karma: 970,
        submitted: [23736822, 23736800, ...]
    }
    """
    import httpx

    url = f"https://hacker-news.firebaseio.com/v0/user/{user}.json"
    response = httpx.get(url)
    user_data = response.json()
    submissions = user_data["submitted"]

    comment_batch = []
    max_batch_size = 10

    for sub in submissions:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{sub}.json"
        item_response = httpx.get(item_url)
        item_data = item_response.json()
        if item_data["type"] == "comment":
            print("found comment")
            print(item_data)
            comment_batch.append(
                {
                    "user_id": item_data["by"],
                    "id": make_id(post_type=PostType.HN),
                    "hn_id": item_data["id"],
                    "body": item_data["text"],
                    "created_at": datetime.utcfromtimestamp(item_data["time"]),
                }
            )
        else:
            print("not comment! ⚠️")

        if len(comment_batch) == max_batch_size:
            print("Inserting comment batch into DB.")
            bulk_insert(engine=engine, table=HnComment, comments=comment_batch)
            comment_batch = []
        time.sleep(0.5)


@stub.function()
def main():
    import psycopg2

    conn = None
    try:
        db_config = db_config_from_env()
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        print("PostgreSQL database version:")
        cur.execute("SELECT version()")
        db_version = cur.fetchone()
        print(db_version)

        print("Getting engine")
        engine = get_engine(db_config)

        print("Creating DB tables")
        create_db_tables(db_config)

        cur.close()
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

    ingest_hn_comments(user=USER_SETTINGS.hackernews_username, engine=engine)

    print("Done!")
