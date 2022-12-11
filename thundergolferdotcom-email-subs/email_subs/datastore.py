import sqlite3
from datetime import datetime
from typing import NamedTuple


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


def create_sub(email: str) -> Subscriber:
    pass


def confirm_sub(email: str, code: str) -> bool:
    return True


def unsub(email: str, code: str) -> bool:
    return True
