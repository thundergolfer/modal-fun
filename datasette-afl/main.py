"""
datasette-afl is a Modal app that runs cron-scheduled functions and a webhook
to serve up-to-date AFL data that's easy to query from your browser.

I will mostly use this app to ask questions people ask in https://www.reddit.com/r/afl.

For example: When was the last time a team made the top-4 with a percentage under 105%?
"""
import argparse
import datetime
import enum
import logging
import pathlib
import sys

from typing import Any, NamedTuple

import httpx
import pytz
import sqlite_utils

import modal

stub = modal.Stub("modal-datasette-afl")

logging_format_str = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=logging_format_str)
logging.getLogger().setLevel(logging.DEBUG)

DB_DIR = pathlib.Path(".")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = pathlib.Path(DB_DIR, "afl.db")


class Subcommand(enum.Enum):
    QUERY = "query"
    PREP = "prep"


class Command(NamedTuple):
    subcommand: Subcommand
    args: Any


class SquiggleAPI:
    @staticmethod
    def games(year):
        url = f"https://api.squiggle.com.au/?q=games;year={year}"
        r = httpx.get(url)
        data = r.json()
        for game in data["games"]:
            # Convert datetime data to datetime objects.
            game["localtime"] = datetime.datetime.fromisoformat(
                game["localtime"] + game["tz"]
            )
            game["date"] = datetime.datetime.fromisoformat(game["date"] + game["tz"])
            game["updated"] = datetime.datetime.fromisoformat(
                game["updated"] + game["tz"]
            )
            yield game

    @staticmethod
    def teams():
        url = f"https://api.squiggle.com.au/?q=teams;"
        r = httpx.get(url)
        data = r.json()
        for game in data["teams"]:
            del game["logo"]  # Don't care about this URI.
            yield game

    @staticmethod
    def standings(year, round):
        url = f"https://api.squiggle.com.au/?q=standings;year={year};round={round}"
        r = httpx.get(url)
        data = r.json()
        for standing in data["standings"]:
            standing["year"] = year
            standing["round"] = round
            yield standing


def is_afl_active() -> bool:
    """
    Determine whether new stats could be available, based on a rough range of dates
    where it's possible AFL is active.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    # AFL has always only been played between these dates. Even during the COVID-19 disrupted season.
    first_march = datetime.datetime(
        year=now.year, month=3, day=1, tzinfo=datetime.timezone.utc
    )
    first_november = datetime.datetime(
        year=now.year, month=11, day=1, tzinfo=datetime.timezone.utc
    )
    if first_march <= now <= first_november:
        return True
    return False


def load_all_games(table):
    logging.info("Loading all AFL games into DB...")
    # TODO(Jonathon): Find out why Squiggle API doesn't have game data before 2000.
    for year in range(2000, 2024):
        if table.exists():
            row_count = query_db(
                f"SELECT COUNT(*) FROM {table.name} WHERE year = {year}"
            ).fetchone()[0]
            if row_count > 120:
                logging.info(
                    f"Not loading {year} games from API. Already loaded in DB."
                )
                continue  # Probably loaded this year's games into DB already.
        games = SquiggleAPI.games(year)
        table.insert_all(games)


def load_all_teams(table):
    logging.info("Loading all AFL teams into DB...")
    teams = SquiggleAPI.teams()
    table.insert_all(teams)


def load_all_standings(table):
    """Ladder data, broken down by year and round."""
    # The number of rounds per year is not regular.
    # COVID-19 affected seasons, and introduction of new teams
    # expanded the number of rounds.
    logging.info("Loading all AFL standings into DB...")
    num_rounds_in_year = [
        (2000, 22),
        (2001, 22),
        (2002, 22),
        (2003, 22),
        (2004, 22),
        (2005, 22),
        (2006, 22),
        (2007, 22),
        (2008, 22),
        (2009, 22),
        (2010, 22),
        (2011, 24),
        (2012, 23),
        (2013, 23),
        (2014, 23),
        (2015, 23),
        (2016, 23),
        (2017, 23),
        (2018, 23),
        (2019, 23),
        (2020, 18),
        (2021, 23),
        (2022, 23),
        (2023, 23),
    ]
    for year, num_rounds in num_rounds_in_year:
        logging.info(f"Processing standings for {year}")
        if table.exists():
            row_count = query_db(
                f"SELECT COUNT(*) FROM {table.name} WHERE year = {year}"
            ).fetchone()[0]
            # Probably loaded this year's standings into DB already.
            if row_count > 300:
                logging.info(
                    f"Not loading {year} standings from API. Already loaded in DB."
                )
                continue
        for round in range(1, num_rounds + 1):
            standings = SquiggleAPI.standings(year, round)
            table.insert_all(standings)


def prep_db(clean=False):
    """Creates a fresh SQLite table with sensible indexes to support queries run from Datasette web UI."""
    logging.info("Prepping sqlite DB...")
    import sqlite_utils
    from sqlite_utils.db import DescIndex

    DB_DIR.mkdir(parents=True, exist_ok=True)

    db = sqlite_utils.Database(str(DB_PATH))

    table = db["games"]
    if table.exists() and clean:
        table.drop()

    load_all_games(table)
    table.create_index(["winner"], if_not_exists=True)

    table = db["teams"]
    if table.exists() and clean:
        table.drop()

    load_all_teams(table)

    table = db["standings"]
    if table.exists() and clean:
        table.drop()

    load_all_standings(table)


def query_db(query):
    """
    Connect to database stored in Modal shared volume and run a query against it.
    Mostly useful for admin and diagnostics.
    """
    import sqlite3

    con = sqlite3.connect(DB_PATH)
    return con.execute(query)


def parse_args() -> Command:
    parser = argparse.ArgumentParser(prog="datasette-serverless-covid19")
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Application subcommands.",
        required=True,
    )

    query_subparser = subparsers.add_parser(
        Subcommand.QUERY.value, help="Run a query against the SQLite DB."
    )
    query_subparser.add_argument("--sql", type=str, help="SQLite query.", required=True)

    subparsers.add_parser(
        Subcommand.PREP.value, help="Download dataset and seed SQLite DB."
    )

    args = parser.parse_args(sys.argv[1:])
    return Command(
        subcommand=Subcommand[args.subcommand.upper()],
        args=args,
    )


@stub.function(schedule=modal.Period(days=1))
def refresh_db():
    if not is_afl_active():
        logging.info(
            "It's the AFL off-season! No new data is being produced, so there's nothing to do."
        )
        return
    logging.info(
        "Within range of dates where AFL is active. Attempting to refresh with new data."
    )
    logging.warning("TODO(Jonathon): Data refresh not yet implemented!")


if __name__ == "__main__":
    cmd = parse_args()

    if cmd.subcommand == Subcommand.QUERY:
        for row in query_db(cmd.args.sql):
            print(row)
    elif cmd.subcommand == Subcommand.PREP:
        prep_db(clean=False)
    else:
        raise AssertionError(f"{cmd.subcommand} subparse not valid.")
