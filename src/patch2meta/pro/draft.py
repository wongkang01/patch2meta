"""Draft order, for a possible Δpriority target: data/processed/draft.parquet.

Source: Leaguepedia PicksAndBansS7 (downloaded per year). In that table Team1 is the team that drafts
first (bans B1 and picks P1), which is not always the blue side: in 2026, 3,090 games have
SwitchedSides = 1. The side therefore comes from games.parquet by team name, never by position.

One row per draft action: game_id, order (1 to 20), team, drafts_first, side, action, champion.
Standard draft order (first team = F, second team = S):
  bans  F1 S1 F2 S2 F3 S3 | picks F1 S1 S2 F2 F3 S3 | bans S4 F4 S5 F5 | picks S4 F4 F5 S5
A skipped ban keeps its row with champion None.
"""

import argparse
from pathlib import Path

import pandas as pd

from patch2meta.pro.names import NOT_A_CHAMPION, canonical, canonical_names
from patch2meta.pro.patches import load_patch_map

RAW = Path("data/raw/pro/cargo_draft")
GAMES = Path("data/processed/games.parquet")
DRAFT = Path("data/processed/draft.parquet")

# (team 1 = drafts first / 2 = second, action, index within that team's bans or picks)
ORDER = [
    (1, "Ban", 1), (2, "Ban", 1), (1, "Ban", 2), (2, "Ban", 2), (1, "Ban", 3), (2, "Ban", 3),
    (1, "Pick", 1), (2, "Pick", 1), (2, "Pick", 2), (1, "Pick", 2), (1, "Pick", 3), (2, "Pick", 3),
    (2, "Ban", 4), (1, "Ban", 4), (2, "Ban", 5), (1, "Ban", 5),
    (2, "Pick", 4), (1, "Pick", 4), (1, "Pick", 5), (2, "Pick", 5),
]  # fmt: skip


def long_draft(raw: pd.DataFrame, games: pd.DataFrame, known: frozenset[str] | None = None) -> pd.DataFrame:
    known = canonical_names() if known is None else known
    sides = games.set_index("game_id")[["team_blue", "team_red"]]
    # Cargo returns some games twice, once with empty slots (65 LCS 2023 games); keep the fullest copy.
    slots = [c for c in raw.columns if c.startswith(("Team1Ban", "Team2Ban", "Team1Pick", "Team2Pick"))]
    filled = (raw[slots].notna() & raw[slots].ne("")).sum(axis=1)
    fullest = raw.assign(_filled=filled).sort_values("_filled", ascending=False, kind="stable")
    raw = raw.loc[sorted(fullest.drop_duplicates("GameId").index)]
    raw = raw.join(sides, on="GameId", how="inner")
    rows = []
    for r in raw.itertuples(index=False):
        rec = r._asdict()
        for order, (team, action, i) in enumerate(ORDER, start=1):
            name = rec[f"Team{team}"]
            value = rec.get(f"Team{team}{action}{i}")
            champion = None if value is None or value in NOT_A_CHAMPION else canonical(value, known)
            side = "blue" if name == rec["team_blue"] else "red" if name == rec["team_red"] else None
            rows.append(
                {
                    "game_id": rec["GameId"],
                    "order": order,
                    "team": name,
                    "drafts_first": team == 1,
                    "side": side,
                    "action": action.lower(),
                    "champion": champion,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="patch2meta.pro draft").parse_args(argv)
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no files in {RAW}; run `download draft` first")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    games = pd.read_parquet(GAMES, columns=["game_id", "patch", "datetime_utc", "team_blue", "team_red"])
    draft = long_draft(raw, games)
    draft.to_parquet(DRAFT, index=False)
    kept = set(load_patch_map().query("kept")["patch"])
    g = games[games["patch"].isin(kept)]
    covered = g["game_id"].isin(
        set(draft.loc[draft["champion"].notna(), "game_id"])
    )  # a draft with champions
    no_side = draft.drop_duplicates(["game_id", "team"])["side"].isna().mean()
    print(
        f"{DRAFT}: {len(draft)} actions, {draft['game_id'].nunique()} games; years downloaded: {[f.stem for f in files]}"
    )
    print("  share of kept-patch games with draft order, by year:")
    print("  ", covered.groupby(g["datetime_utc"].dt.year).mean().round(3).to_dict())
    print(f"  teams whose side could not be matched by name: {no_side:.3%}")
