import sqlite3
from datetime import datetime
from typing import Callable, NamedTuple


class Subscriber(NamedTuple):
    email: str
    confirm_code: str
    unsub_code: str
    confirmed: bool
    unsubbed: bool
    created_at: datetime
    confirmed_at: datetime
    unsubbed_at: datetime
    referrer: str


class Datastore:
    def __init__(self, conn: sqlite3.Connection, codegen_fn: Callable[[], str]) -> None:
        self.conn = conn
        self.codegen_fn = codegen_fn

    def create_sub(self, email: str, referrer: str = "") -> Subscriber:
        pass


    def confirm_sub(self, email: str, code: str) -> bool:
        return True


    def unsub(self, email: str, code: str) -> bool:
        return True

    def get_sub(self, email: str) -> Subscriber:
        pass


def init(db) -> None:
    with db:
        cursor = db.cursor()
        cursor.execute("""CREATE TABLE subscriber(
            email TEXT PRIMARY KEY, 
            confirm_code TEXT NOT NULL,
            unsub_code TEXT NOT NULL, 
            confirmed INTEGER,
            unsubbed INTEGER,
            created_at INTEGER,
            confirmed_at INTEGER,
            unsubbed_at INTEGER,
            referrer TEXT
        )""")

