"""Build options for open data choices: each one drops or moves exactly what it says, and the defaults
change nothing."""

import pandas as pd
from conftest import game

from patch2meta.pro.games import apply_choices, league_tier
from patch2meta.pro.patches import build_patch_map, cut_of


def series_games() -> pd.DataFrame:
    rows = [
        # Fearless series s1: g1 is game 1
        game("g2", "s1", "Cup", "2025-01-01 13:00", "25.S1.1", "ABCDE", "FGHIJ", "K", "L"),
        game("g1", "s1", "Cup", "2025-01-01 12:00", "25.S1.1", "ABCDE", "FGHIJ", "K", "L"),
        game("g3", "s1", "Cup", "2025-01-01 14:00", "25.S1.1", "ABCDE", "FGHIJ"),  # no bans, unlike g1, g2
        # Standard series s2: no game is dropped by --fearless-games
        game("g4", "s2", "Cup", "2025-01-02 12:00", "25.S1.1", "ABCDE", "FGHIJ", "K"),
        game("g5", "s2", "Cup", "2025-01-02 13:00", "25.S1.1", "ABCDE", "FGHIJ", "K"),
        # Series s3 records no bans at all
        game("g6", "s3", "Cup", "2025-01-03 12:00", "25.S1.1", "ABCDE", "FGHIJ"),
        game("g7", "s3", "Cup", "2025-01-03 13:00", "25.S1.1", "ABCDE", "FGHIJ"),
    ]
    g = pd.DataFrame(rows)
    g["fearless"] = g["match_id"].eq("s1")
    return g


def kept_ids(g: pd.DataFrame, **options) -> list[str]:
    return sorted(apply_choices(g, **options)[0]["game_id"])


def test_defaults_keep_every_game():
    g = series_games()
    out, log = apply_choices(g)
    assert log == {}
    pd.testing.assert_frame_equal(out, g.reset_index(drop=True))


def test_fearless_game_one_only():
    assert kept_ids(series_games(), fearless_games="first") == ["g1", "g4", "g5", "g6", "g7"]


def test_no_ban_games():
    g = series_games()
    assert kept_ids(g, no_ban_games="drop-blind") == ["g1", "g2", "g4", "g5", "g6", "g7"]
    assert kept_ids(g, no_ban_games="drop-all") == ["g1", "g2", "g4", "g5"]


def test_league_tiers():
    tours = pd.Series([
        "LCK/2026 Season/Spring Season", "LCK CL/2026 Season/Spring Season", "LDL/2025 Season/Split 1",
        "2019 Mid-Season Invitational/Main Event", "2025 Season World Championship/Main Event",
        "2026 First Stand", "LFL/2026 Season/Spring", "2025 Asia Masters", "LCS Proving Grounds/2022",
    ])  # fmt: skip
    assert league_tier(tours).tolist() == [
        "major", "second", "second", "major", "major", "major", "other", "other", "other",
    ]  # fmt: skip


def test_cut_rules_and_minimum(toy_games, toy_wiki):
    assert cut_of(pd.Timestamp("2026-10-21"), "release-day") == pd.Timestamp("2026-10-21 00:00")
    pm = build_patch_map(toy_games, toy_wiki, min_games=2, cut_rule="release-day").set_index("patch")
    assert pm.at["9.2", "cut"] == pd.Timestamp("2019-01-24")
    strict = build_patch_map(toy_games, toy_wiki, min_games=3)
    assert strict.loc[strict["kept"], "patch"].tolist() == ["9.1"]  # the only patch with 3 games
