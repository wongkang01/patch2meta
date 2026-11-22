"""Toy games shared by the pro-data tests.

Champions A to L are old (released 2018); N is new (released 2019-02-10). Each game lists picks in role
order (top, jungle, mid, bot, support).
"""

import pandas as pd
import pytest

CHAMPS = list("ABCDEFGHIJKL") + ["N"]
OLD = pd.Timestamp("2018-01-01")


def game(gid, match, tour, dt, patch, blue, red, bans_blue=(), bans_red=(), winner="blue"):
    return {
        "game_id": gid,
        "match_id": match,
        "tournament": tour,
        "datetime_utc": pd.Timestamp(dt),
        "patch": patch,
        "team_blue": "T1",
        "team_red": "T2",
        "winner": {"blue": "T1", "red": "T2", None: None}[winner],
        "picks_blue": list(blue),
        "picks_red": list(red),
        "bans_blue": list(bans_blue),
        "bans_red": list(bans_red),
        "source": "toy",
        "fearless": False,
    }


@pytest.fixture
def toy_games() -> pd.DataFrame:
    rows = [
        # 9.1, released 2019-01-10 (cut 2019-01-09)
        game("g1", "m1", "T", "2019-01-12", "9.1", "ABCDE", "FGHIJ", "K", "L"),
        game("g2", "m2", "T", "2019-01-13", "9.1", "ABCDE", "FGHIJ", "K"),
        # 9.2, released 2019-01-24 (cut 2019-01-23); g4 is played on 9.1 after 9.2's notes
        game("g3", "m3", "T", "2019-01-25", "9.2", "ABCDE", "FGHIK", "J"),
        game("g3b", "m3b", "T", "2019-01-25", "9.2", "ABCDE", "FGHIJ"),
        game("g4", "m4", "T", "2019-01-26", "9.1", "LBCDE", "FGHIJ"),
        # 9.3, released 2019-02-07 (cut 2019-02-06); g6 lists A twice (picked and banned)
        game("g5", "m5", "T", "2019-02-08", "9.3", "ABCDE", "FGHIJ", "L"),
        game("g6", "m6", "T", "2019-02-09", "9.3", "ABCDE", "FGHIJ", "A"),
        # 9.4, released 2019-02-21 (cut 2019-02-20); N debuts here
        game("g7", "m7", "T", "2019-02-22", "9.4", "NBCDE", "FGHIJ", winner="red"),
        game("g8", "m8", "T", "2019-02-23", "9.4", "ABCDE", "FGHIJ", "N"),
        # 9.5, released 2019-03-07 (cut 2019-03-06)
        game("g9", "m9", "T", "2019-03-08", "9.5", "NBCDE", "FGHIJ"),
        game("g10", "m10", "T", "2019-03-09", "9.5", "ABCDE", "FGHIJ"),
        # 9.6, released 2019-03-21 (cut 2019-03-20); 9.5b is a dropped patch in between
        game("g11", "m11", "T", "2019-03-22", "9.6", "NBCDE", "FGHIJ"),
        game("g12", "m12", "T", "2019-03-23", "9.6", "ABCDE", "FGHIJ"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def toy_wiki() -> pd.DataFrame:
    dates = {
        "9.1": "2019-01-10",
        "9.2": "2019-01-24",
        "9.3": "2019-02-07",
        "9.4": "2019-02-21",
        "9.5": "2019-03-07",
        "9.5b": "2019-03-12",
        "9.6": "2019-03-21",
    }
    return pd.DataFrame(
        {"patch": list(dates), "release_date": list(dates.values()), "redirect_to": [None] * len(dates)}
    )


@pytest.fixture
def toy_release() -> pd.Series:
    r = pd.Series(OLD, index=CHAMPS)
    r["N"] = pd.Timestamp("2019-02-10")
    return r.sort_index()
