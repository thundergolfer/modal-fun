import time
import modal

from .config import USER_SETTINGS

image = modal.Image.debian_slim().pip_install("httpx", "loguru", "psycopg2-binary")
stub = modal.Stub(
    name="post-archiver", 
    image=image, 
    secrets=[modal.Secret.from_name("neondb")]
)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': '<dbname>',
        'USER': '<user>',
        'PASSWORD': '<password>',
        'HOST': '<endpoint_hostname>',
        'PORT': '<port>',
    }
}


def db_config_from_env() -> dict[str, str]:
    required_keys = {'PGHOST', 'PGDATABASE', 'PGUSER', "PGPASSWORD"}
    import os
    extracted_env = {
        k: os.environ[k]
        for k in required_keys
        if k in os.environ
    }

    missing_keys = required_keys - set(extracted_env.keys())
    if missing_keys:
        raise RuntimeError(
            f"Missing required environment variables: {missing_keys}. "
            "Did you forget to add a modal.Secret, or are some keys missing from "
            "the provided modal.Secret?"
        )
    return {
        k.replace("PG", "").lower(): v
        for k, v
        in extracted_env.items()
    }


def ingest_hn_comments(user: str):
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
    for sub in submissions:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{sub}.json"
        item_response = httpx.get(item_url)
        item_data = item_response.json()
        if item_data["type"] == "comment":
            print("found comment")
            print(item_data)
        else:
            print("not comment! ⚠️")
        time.sleep(0.5)


@stub.function()
def main():
    import psycopg2
    conn = None
    try:
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**db_config_from_env())
        cur = conn.cursor()
        
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
        db_version = cur.fetchone()
        print(db_version)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
    
    ingest_hn_comments(USER_SETTINGS.hackernews_username)

    print("Done!")
