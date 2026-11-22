"""games.clean() and the 1 Jan 2026 switch from gptilt to Leaguepedia Cargo."""

import pandas as pd
from conftest import CHAMPS, game

from patch2meta.pro import games as games_mod


def test_clean_drops_placeholders_short_drafts_unpatched_and_duplicate_games(monkeypatch):
    monkeypatch.setattr(games_mod, "canonical_names", lambda: frozenset(CHAMPS))
    rows = [
        game(
            "ok",
            "m1",
            "T",
            "2025-01-01",
            "25.S1.1",
            "ABCDE",
            "FGHIJ",
            ["K", "None", ""],
            ["Missing Data", "L"],
        ),
        game("short", "m2", "T", "2025-01-01", "25.S1.1", ["A", "B", "C", "D", "None"], "FGHIJ"),
        game("nopatch", "m3", "T", "2025-01-01", "", "ABCDE", "FGHIJ"),
        game("ok", "m1", "T", "2025-01-01", "25.S1.1", "ABCDE", "FGHIJ"),
        game("oddwin", "m4", "T", "2025-01-02", "25.S1.1", "ABCDE", "FGHIJ") | {"winner": "Somebody"},
    ]
    out, log = games_mod.clean(pd.DataFrame(rows))
    assert out["game_id"].tolist() == ["ok", "oddwin"]
    assert (out.loc[0, "bans_blue"], out.loc[0, "bans_red"]) == (["K"], ["L"])
    assert (log["dropped_incomplete_picks"], log["dropped_no_patch"], log["dropped_duplicate_id"]) == (
        1,
        1,
        1,
    )
    assert pd.isna(out.loc[1, "winner"])


def test_each_source_covers_only_its_side_of_1_january_2026(tmp_path, monkeypatch):
    monkeypatch.setattr(games_mod, "RAW", tmp_path)
    g = pd.DataFrame([
        game("g2025", "a", "T", "2025-12-31 23:59", "25.24", "ABCDE", "FGHIJ"),
        game("g2026", "b", "T", "2026-01-01 00:00", "26.01", "ABCDE", "FGHIJ"),
    ]).drop(columns=["fearless", "source"])  # fmt: skip
    g["datetime_utc"] = g["datetime_utc"].astype(str)
    g.to_parquet(tmp_path / "gptilt_games.parquet")
    pd.DataFrame({
        "GameId": ["c2025", "c2026"], "MatchId": ["a", "b"], "OverviewPage": "T", "Tournament": "T",
        "Team1": "T1", "Team2": "T2", "WinTeam": "T1", "DateTimeUTC": ["2025-12-31 23:59:00", "2026-01-01 00:00:00"],
        "Patch": ["25.24", "26.01"], "Team1Picks": "A,B,C,D,E", "Team2Picks": "F,G,H,I,J", "Team1Bans": "K",
        "Team2Bans": "", "NGameInMatch": "1",
    }).to_parquet(tmp_path / "cargo_games_2026.parquet")  # fmt: skip
    assert games_mod.load_gptilt()["game_id"].tolist() == ["g2025"]
    cargo = games_mod.load_cargo()
    assert cargo["game_id"].tolist() == ["c2026"]
    assert (cargo.iloc[0]["picks_blue"], cargo.iloc[0]["bans_red"]) == (["A", "B", "C", "D", "E"], [])
