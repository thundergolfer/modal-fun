import json
import sqlite3
from datetime import datetime
from typing import Callable, NamedTuple, Optional
from uuid import uuid4


class StatLine(NamedTuple):
    id: str
    data: dict[str, list[str]]
    created_at: datetime


class Datastore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        codegen_fn: Callable[[], str],
        clock_fn: Callable[[], datetime],
    ) -> None:
        self.conn = conn
        self.codegen_fn = codegen_fn
        self.clock_fn = clock_fn

    def add_stat_line(self, data: dict[str, list[str]]) -> StatLine:
        id_ = str(uuid4())
        created_at = self.clock_fn()
        data_str = json.dumps(data)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO stat VALUES(?, ?, ?)",
                (id_, data_str, created_at),
            )
        return StatLine(
            id=id_,
            created_at=created_at,
            data=data,
        )

    def dump_stat_lines(self) -> list[StatLine]:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, created_at as "[timestamp]", data FROM stat')
            rows = cursor.fetchall()
        return [
            StatLine(
                id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                data=json.loads(row[2]),
            )
            for row in rows
        ]

    def delete_everything(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM stat")


def get_db(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def init(db) -> None:
    with db:
        cursor = db.cursor()
        # Append-only table
        cursor.execute(
            """CREATE TABLE stat(
            id TEXT PRIMARY KEY, 
            data TEXT NOT NULL, 
            created_at INTEGER,
        )"""
        )
