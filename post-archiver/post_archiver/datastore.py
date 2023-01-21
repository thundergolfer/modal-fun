# Table structure:
#
# - Want to do full-text search
# - Want to train chatbot, so need any parent comment for prompt-reply training.
#
# Start simple. (Internal ID, Platform User ID, Platform Comment ID, UTF-8 encoded comment body, Platform Score, Created At, Modified At)

# TODO(Jonathon): This 3-party package is needed locally, so add a script to setup virtualenv and switch to requirements.txt.
import sqlalchemy
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy import Table
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def make_id(type: str) -> str:
    # TODO: type should be an enum.
    raise NotImplementedError()


class HnComment(Base):
    __tablename__ = "hn_comment"
    id = Column(String(), primary_key=True)
    hn_id = Column(String(), unique=True)
    user_id = Column(String())
    body = Column(String())
    score = Column(Integer())
    created_at = Column(DateTime())
    modified_at = Column(DateTime())


def create_db_tables(db_config: dict[str, str]) -> None:
    from sqlalchemy import create_engine

    username = db_config["user"]
    pw = db_config["password"]
    endpoint = db_config["host"]
    conn_str = f"postgresql://{username}:{pw}@{endpoint}/neondb"
    engine = create_engine(conn_str, echo=True)
    Base.metadata.create_all(engine)
