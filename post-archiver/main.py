import modal

image = modal.Image.debian_slim().pip_install("loguru", "psycopg2-binary")
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


@stub.function()
def main():
    import psycopg2
    print("hello world")

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

    print("Done!")
