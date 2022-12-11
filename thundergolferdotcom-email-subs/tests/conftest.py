import sqlite3
import pytest

from email_subs import datastore

@pytest.fixture(scope="session")
def store():
    db = sqlite3.connect(":memory:?cache=shared")
    datastore.init(db)
    store = datastore.Datastore(
        conn=db,
        codegen_fn=lambda: "1234abcd",
    )
    yield store
