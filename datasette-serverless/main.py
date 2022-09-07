import csv
import itertools
import os
import pathlib

import modal

volume = modal.SharedVolume().persist("dataset-cache-vol")
stub = modal.Stub("modal-datasette-covid")

CACHE_DIR = "/cache"
REPO_DIR = pathlib.Path(CACHE_DIR, "COVID-19")
DB_DIR = pathlib.Path(CACHE_DIR, "sqlitedb-files")

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
def download_dataset():
    import git

    if REPO_DIR.exists():
        print("Clone already done. Skipping...")
        return
    git_url = "https://github.com/CSSEGISandData/COVID-19"
    git.Repo.clone_from(git_url, repo_dir, depth=1)


@stub.function(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def prep_db():
    print("Prepping sqlite DB...")

    import datasette
    import sqlite_utils
    from sqlite_utils.db import DescIndex

    DB_DIR.mkdir(parents=True, exist_ok=True)

    db = sqlite_utils.Database(str(DB_DIR / "covid.db"))

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


@stub.asgi(
    image=datasette_image,
    shared_volumes={CACHE_DIR: volume},
)
def app():
    from datasette.app import Datasette

    return Datasette(files=[str(DB_DIR / "covid.db")]).app()


if __name__ == "__main__":
    with stub.run():
        download_dataset()
        prep_db()

    stub.serve()
