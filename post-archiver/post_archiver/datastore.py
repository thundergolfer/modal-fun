# Table structure:
#
# - Want to do full-text search
# - Want to train chatbot, so need any parent comment for prompt-reply training.
#
# Start simple. (Internal ID, Platform User ID, Platform Comment ID, UTF-8 encoded comment body, Platform Score, Created At, Modified At)

# TODO(Jonathon): This 3-party package is needed locally, so add a script to setup virtualenv and switch to requirements.txt.
import enum

import sqlalchemy
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy import Table
from sqlalchemy.ext.declarative import declarative_base

from .crockfords32 import generate

Base = declarative_base()

# The encoded symbol space does not include I, L, O or U
ID_SYMBOLS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class PostType(enum.Enum):
    HN = "hn"
    REDDIT = "re"


def make_id(post_type: PostType) -> str:
    random_part = generate(length=16, checksum=True)
    return f"{post_type.value}-{random_part}"


class HnComment(Base):
    __tablename__ = "hn_comment"
    id = Column(String(), primary_key=True)
    hn_id = Column(String(), unique=True)
    user_id = Column(String())
    body = Column(String())
    score = Column(Integer())
    created_at = Column(DateTime())
    modified_at = Column(DateTime())


def bulk_insert(engine, table, comments) -> None:
    from sqlalchemy import insert
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.execute(
            insert(table),
            comments,
        )
        session.commit()


def get_engine(db_config: dict[str, str]):
    from sqlalchemy import create_engine

    username = db_config["user"]
    pw = db_config["password"]
    endpoint = db_config["host"]
    conn_str = f"postgresql://{username}:{pw}@{endpoint}/neondb"
    return create_engine(conn_str, echo=True)


def create_db_tables(db_config: dict[str, str]) -> None:
    engine = get_engine(db_config)
    Base.metadata.create_all(engine)
