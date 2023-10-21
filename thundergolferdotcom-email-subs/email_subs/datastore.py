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
    deleted_at: datetime
    referrer: str


class Notification(NamedTuple):
    blogpost_link: str
    notified_at: datetime
    # Comma-separated list of emails
    recipients: str


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

    def create_sub(self, email: str, referrer: str = "") -> Subscriber:
        try:
            sub = self.get_sub(email)
            print(f"Returning existing subscriber for '{email}'")
            return sub
        except KeyError:
            print(f"Creating new subscriber for '{email}'")

        sub = Subscriber(
            email=email,
            confirm_code=self.codegen_fn(),
            unsub_code=self.codegen_fn(),
            confirmed=False,
            unsubbed=False,
            created_at=self.clock_fn(),
            confirmed_at=None,
            unsubbed_at=None,
            deleted_at=None,
            referrer=referrer,
        )
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO subscriber VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                sub,
            )
        return sub

    def confirm_sub(self, email: str, code: str) -> bool:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT * FROM subscriber WHERE email = ? AND deleted_at IS NULL",
                (email,),
            )
            if cursor.fetchone() is None:
                raise ValueError(f"No subscriber found for email '{email}'")
            cursor.execute(
                f"""UPDATE subscriber
                SET confirmed = TRUE, confirmed_at = ?
                WHERE email = ? AND confirm_code = ?""",
                (self.clock_fn(), email, code),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Invalid confirmation code for email '{email}'")
        return True

    def unsub(self, email: str, code: str) -> bool:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                f"SELECT * FROM subscriber WHERE email = ? AND confirmed = 1 AND deleted_at IS NULL",
                (email,),
            )
            if cursor.fetchone() is None:
                raise ValueError(f"No confirmed subscriber found for email '{email}'")
            cursor.execute(
                f"""UPDATE subscriber
                SET unsubbed = TRUE, unsubbed_at = ?
                WHERE email = ? AND unsub_code = ?""",
                (self.clock_fn(), email, code),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Invalid unsubscribe code for email '{email}'")
        return True

    def get_sub(self, email: str) -> Subscriber:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM subscriber WHERE email = ? AND deleted_at IS NULL",
                (email,),
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"No subscriber with email '{email}'")
        return Subscriber(
            email=row[0],
            confirm_code=row[1],
            unsub_code=row[2],
            confirmed=bool(row[3]),
            unsubbed=bool(row[4]),
            created_at=row[5],
            confirmed_at=row[6],
            unsubbed_at=row[7],
            deleted_at=row[8],
            referrer=row[9],
        )

    def list_subs(
        self,
        include_unconfirmed: bool = False,
        include_unsubbed: bool = False,
        include_deleted: bool = False,
    ) -> list[Subscriber]:
        query = """SELECT * 
            FROM subscriber
            WHERE email is NOT NULL"""
        if not include_unconfirmed:
            query += " AND confirmed = TRUE"
        if not include_unsubbed:
            query += " AND unsubbed = FALSE"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return [
            Subscriber(
                email=row[0],
                confirm_code=row[1],
                unsub_code=row[2],
                confirmed=bool(row[3]),
                unsubbed=bool(row[4]),
                created_at=row[5],
                confirmed_at=row[6],
                unsubbed_at=row[7],
                deleted_at=row[8],
                referrer=row[9],
            )
            for row in rows
        ]

    def create_notification(self, link: str, recipients: list[str]) -> Notification:
        n = Notification(
            blogpost_link=link,
            notified_at=self.clock_fn(),
            recipients=",".join(recipients),
        )
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO notification VALUES(?, ?, ?)",
                n,
            )
        return n

    def list_notifications(self) -> list[Notification]:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                'SELECT blogpost_link, notified_at as "[timestamp]", recipients FROM notification'
            )
            rows = cursor.fetchall()
        return [
            Notification(
                blogpost_link=row[0],
                notified_at=datetime.fromisoformat(row[1]),
                recipients=row[2],
            )
            for row in rows
        ]

    def delete_notifications(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM notification")

    def delete_everything(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM subscriber")
            cursor.execute("DELETE FROM notification")


def get_db(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path)


def init(db) -> None:
    with db:
        cursor = db.cursor()
        cursor.execute(
            """CREATE TABLE subscriber(
            email TEXT PRIMARY KEY, 
            confirm_code TEXT NOT NULL,
            unsub_code TEXT NOT NULL, 
            confirmed INTEGER,
            unsubbed INTEGER,
            created_at INTEGER,
            confirmed_at INTEGER,
            unsubbed_at INTEGER,
            deleted_at INTEGER,
            referrer TEXT
        )"""
        )
        # Append-only table
        cursor.execute(
            """CREATE TABLE notification(
            blogpost_link TEXT PRIMARY KEY, 
            notified_at INTEGER,
            recipients TEXT
        )"""
        )
