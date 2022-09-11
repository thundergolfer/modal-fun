import argparse
import csv
import enum
import itertools
import os
import pathlib
import sys
from datetime import datetime, timezone
from typing import Any, NamedTuple

import modal

volume = modal.SharedVolume().persist("dataset-cache-vol")
stub = modal.Stub("modal-datasette-covid")
datasette_image = (
    modal.DebianSlim()
    .pip_install(
        [
            "datasette",
            "sqlite-utils",
            "GitPython",
        ]
    )
    .apt_install(["git"])
)

CACHE_DIR = "/cache"
REPO_DIR = pathlib.Path(CACHE_DIR, "COVID-19")
DB_DIR = pathlib.Path(CACHE_DIR, "sqlitedb-files")
DB_PATH = pathlib.Path(DB_DIR, "covid.db")


class Subcommand(enum.Enum):
    QUERY = "query"
    PREP = "prep"


class Command(NamedTuple):
    subcommand: Subcommand
    args: Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_report(filepath):
    mm, dd, yyyy = filepath.stem.split("-")
    day = f"{yyyy}-{mm}-{dd}"
    with filepath.open() as fp:
        for row in csv.DictReader(fp):
            # Weirdly, this column is sometimes \ufeffProvince/State
            province_or_state = (
                row.get("\ufeffProvince/State")
                or row.get("Province/State")
                or row.get("Province_State")
                or None
            )
            country_or_region = row.get("Country_Region") or row.get("Country/Region")
            yield {
                "day": day,
                "country_or_region": country_or_region.strip()
                if country_or_region
                else None,
                "province_or_state": province_or_state.strip()
                if province_or_state
                else None,
                "admin2": row.get("Admin2") or None,
                "fips": row.get("FIPS", "").strip() or None,
                "confirmed": int(float(row["Confirmed"] or 0)),
                "deaths": int(float(row["Deaths"] or 0)),
                "recovered": int(float(row["Recovered"] or 0)),
                "active": int(row["Active"]) if row.get("Active") else None,
                "latitude": row.get("Latitude") or row.get("Lat") or None,
                "longitude": row.get("Longitude") or row.get("Long_") or None,
                "last_update": row.get("Last Update") or row.get("Last_Update") or None,
                "combined_key": row.get("Combined_Key") or None,
            }


def load_daily_reports():
    jhu_csse_base = REPO_DIR
    daily_reports = list(
        (jhu_csse_base / "csse_covid_19_data" / "csse_covid_19_daily_reports").glob(
            "*.csv"
        )
    )
    for filepath in daily_reports:
        yield from load_report(filepath)


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def download_dataset(force=False):
    import git

    if REPO_DIR.exists():
        if not force:
            print("Clone already done. Skipping...")
            return
        else:
            print("Refreshing cloned repo.")
    git_url = "https://github.com/CSSEGISandData/COVID-19"
    git.Repo.clone_from(git_url, repo_dir, depth=1)


@stub.function(schedule=modal.Period(hours=1))
def refresh_db():
    print(f"Running scheduled refresh at {utc_now()}")
    download_dataset(force=True)
    prep_db()


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def prep_db():
    print("Prepping sqlite DB...")
    import sqlite_utils
    from sqlite_utils.db import DescIndex

    DB_DIR.mkdir(parents=True, exist_ok=True)

    db = sqlite_utils.Database(str(DB_PATH))

    # Load John Hopkins CSSE daily reports
    table = db["johns_hopkins_csse_daily_reports"]
    if table.exists():
        table.drop()
    print("Loading daily reports...")
    # Do subset of records to speed up processing
    records = itertools.islice(load_daily_reports(), 10_000)
    table.insert_all(records)
    table.create_index(["day"], if_not_exists=True)
    table.create_index(["province_or_state"], if_not_exists=True)
    table.create_index(["country_or_region"], if_not_exists=True)
    table.create_index(["combined_key"], if_not_exists=True)
    print("DB prepared!")


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def query_db(query):
    import sqlite3

    con = sqlite3.connect(DB_PATH)
    for row in con.execute(query):
        print(row)


@stub.asgi(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def app():
    from datasette.app import Datasette

    return Datasette(files=[DB_PATH]).app()


def parse_args() -> Command:
    parser = argparse.ArgumentParser(prog="datasette-serverless-covid19")
    subparsers = parser.add_subparsers(
        dest="subparser_name", help="Application subcommands."
    )

    query_subparser = subparsers.add_parser(
        Subcommand.QUERY.value, help="Run a query remotely, against the SQLite DB."
    )
    query_subparser.add_argument("--sql", type=str, help="SQLite query.", required=True)

    subparsers.add_parser(
        Subcommand.PREP.value, help="Download dataset and seed SQLite DB."
    )

    args = parser.parse_args(sys.argv[1:])
    return Command(
        subcommand=Subcommand[args.subparser_name.upper()],
        args=args,
    )


if __name__ == "__main__":
    cmd = parse_args()

    with stub.run():
        if cmd.subcommand == Subcommand.QUERY:
            query_db(cmd.args.sql)
        elif cmd.subcommand == Subcommand.PREP:
            download_dataset()
            prep_db()
        else:
            raise AssertionError(f"{cmd.subcommand} subparse not valid.")
