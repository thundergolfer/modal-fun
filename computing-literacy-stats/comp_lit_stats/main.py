from datetime import datetime, timezone
from pydantic import BaseModel
from uuid import uuid4

import modal
from fastapi import FastAPI

from . import datastore

CACHE_DIR = "/cache"
DB_PATH = CACHE_DIR + "/comp_lit_stats.db"

modal_workspace_username = "thundergolfer"
app_name = "comp-lit-stats"
volume = modal.NetworkFileSystem.persisted(f"{app_name}-vol")
stub = modal.Stub(name=app_name)

web_app = FastAPI()


class AddStatLineRequest(BaseModel):
    left: list[str]
    right: list[str]
    trash: list[str]


@stub.function(network_file_systems={CACHE_DIR: volume})
def setup_db():
    """
    Only need to run this once for a Modal app.
    Creates and initializes an SQLite DB on a Modal persistent volume.
    """
    print("Setting up new DB... (this should only run once)")
    conn = datastore.get_db(DB_PATH)
    datastore.init(conn)
    print("☑️ DB setup done")


@web_app.post("/add")
def add_stat_line(request: AddStatLineRequest):
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    store.add_stat_line(dict(request))


@web_app.get("/summary")
def summarize_stat_lines():
    # Dummy data for now.
    return {
        "left": ["Fairchild Semiconductor", "HyperCard"],
        "right": ["web3"],
        "trash": ["Transhumanism"],
    }


@web_app.get("/dump")
def dump():
    conn = datastore.get_db(DB_PATH)
    store = datastore.Datastore(
        conn=conn,
        codegen_fn=lambda: str(uuid4()),
        clock_fn=lambda: datetime.now(timezone.utc),
    )
    return store.dump_stat_lines()


@stub.function(
    # Web app uses datastore to confirm subscriptions and fulfil unsubscriptions.
    network_file_systems={CACHE_DIR: volume},
)
@modal.asgi_app()
def web():
    from fastapi.middleware.cors import CORSMiddleware

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://thundergolfer.com",
            "https://thundergolfer.com",
            "http://localhost:4000",
            "http://localhost:4000/",
            "localhost:4000",
            "localhost:4000/",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return web_app
