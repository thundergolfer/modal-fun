import pytz
import sqlite3
from datetime import datetime
import pytest

from email_subs import datastore


def test_clock_fn():
    return datetime.fromtimestamp(0, tz=pytz.utc)


@pytest.fixture(scope="session")
def store():
    db = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    datastore.init(db)
    store = datastore.Datastore(
        conn=db,
        codegen_fn=lambda: "1234abcd",
        clock_fn=test_clock_fn,
    )
    yield store
