"""A tournament is Fearless from 90% of its multi-game series (holds with and without fearless-fix.patch)."""

import pandas as pd
from conftest import game

from patch2meta.pro.fearless import game_fearless

FRESH = {"picks_blue": ["P1", "P2", "P3", "P4", "P5"], "picks_red": ["P6", "P7", "P8", "P9", "P10"]}
REPEAT = {"picks_blue": ["A", "P2", "P3", "P4", "P5"], "picks_red": ["P6", "P7", "P8", "P9", "P10"]}


def cup(name: str, n_fearless: int) -> list[dict]:
    rows = []
    for i in range(10):
        second = FRESH if i < n_fearless else REPEAT  # the blue team picks A again
        rows += [
            game(f"{name}{i}a", f"{name}{i}", name, "2025-01-01 10:00", "25.S1.1", "ABCDE", "FGHIJ"),
            game(f"{name}{i}b", f"{name}{i}", name, "2025-01-01 11:00", "25.S1.1", "ABCDE", "FGHIJ") | second,
        ]
    return rows


def test_a_tournament_is_fearless_from_90_percent_of_its_series():
    g = pd.DataFrame(cup("Nine", 9) + cup("Eight", 8))
    flags = game_fearless(g).groupby(g["tournament"]).first().to_dict()
    assert flags == {"Eight": False, "Nine": True}
