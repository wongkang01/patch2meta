"""One cleaned table of professional games: data/processed/games.parquet (internal to person 2).

Sources: gptilt for 2019-01-01 to 2025-12-31, Leaguepedia Cargo ScoreboardGames from 2026-01-01
(gptilt stops on 25 Oct 2026). Cleaning:
  - champion names mapped to canonical names; placeholders ("None", "Missing Data", "") dropped from bans;
  - games without 5 real picks per side dropped;
  - games without a patch dropped (1,325 in total: 1,136 in 2025, mostly LDL, 180 in 2026, 9 earlier;
    the patch is empty in Leaguepedia too);
  - a champion listed twice in one game is kept as listed. Presence counts it once per game, but pick,
    win and role counts count each listing (39 games list one champion on both teams, all of them
    blind-pick games without bans); the relation graph skips a champion paired with itself.
Each game also gets its Fearless flag (pro/fearless.py).

Three options drop games for open team questions (compared by pro/variants.py). The defaults keep
every game:
  --fearless-games first   A4: in Fearless games, keep only the first game of each series
  --no-ban-games ...       B3: drop-blind drops games without bans whose series has bans in other games;
                           drop-all drops every game without bans
  --leagues ...            D1: major keeps the four major regions' top leagues and the international
                           events; major-second adds those regions' official second divisions
Fearless flags are set before any game is dropped, so the options never change a game's flag.
"""

import argparse
from pathlib import Path

import pandas as pd

from patch2meta.pro.fearless import game_fearless
from patch2meta.pro.names import NOT_A_CHAMPION, canonical, canonical_names

RAW = Path("data/raw/pro")
START = "2019-01-01"
CARGO_FROM = "2026-01-01"
COLUMNS = [
    "game_id", "match_id", "tournament", "datetime_utc", "patch", "team_blue", "team_red", "winner",
    "picks_blue", "picks_red", "bans_blue", "bans_red", "source",
]  # fmt: skip
CHAMP_LISTS = ["picks_blue", "picks_red", "bans_blue", "bans_red"]
# D1 tiers, by the tournament name before its first "/". League players should check both lists.
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "LTA", "LTA North", "LTA South"}
MAJOR_EVENTS = ("Season World Championship", "Mid-Season Invitational", "First Stand")  # after the year
SECOND_DIVISIONS = {
    "LCK CL", "Challengers Korea", "LDL", "EMEA Masters", "European Masters",
    "North American Challengers League", "NA Academy League",
}  # fmt: skip


def load_gptilt(end: str | None = CARGO_FROM) -> pd.DataFrame:
    g = pd.read_parquet(RAW / "gptilt_games.parquet")
    keep = g["datetime_utc"] >= START
    if end is not None:
        keep &= g["datetime_utc"] < end
    g = g[keep].copy()
    for col in CHAMP_LISTS:
        g[col] = g[col].map(list)
    g["source"] = "gptilt"
    return g[COLUMNS]


def split_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [x.strip() for x in value.split(",")]


def load_cargo() -> pd.DataFrame:
    c = pd.read_parquet(RAW / "cargo_games_2026.parquet")
    out = pd.DataFrame(
        {
            "game_id": c["GameId"],
            "match_id": c["MatchId"],
            "tournament": c["OverviewPage"],
            "datetime_utc": c["DateTimeUTC"],
            "patch": c["Patch"].fillna(""),
            # Team1 in ScoreboardGames is taken as the blue side; verify on match pages (plan 7.3).
            "team_blue": c["Team1"],
            "team_red": c["Team2"],
            "winner": c["WinTeam"],
            "picks_blue": c["Team1Picks"].map(split_list),
            "picks_red": c["Team2Picks"].map(split_list),
            "bans_blue": c["Team1Bans"].map(split_list),
            "bans_red": c["Team2Bans"].map(split_list),
            "source": "cargo",
        }
    )
    return out[out["datetime_utc"] >= CARGO_FROM]


def clean(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    known = canonical_names()
    g = games.copy()
    for col in CHAMP_LISTS:
        g[col] = g[col].map(lambda lst: [canonical(x, known) for x in lst if x not in NOT_A_CHAMPION])
    log = {"input": len(g)}
    full_picks = g["picks_blue"].map(len).eq(5) & g["picks_red"].map(len).eq(5)
    log["dropped_incomplete_picks"] = int((~full_picks).sum())
    g = g[full_picks]
    has_patch = g["patch"].ne("")
    log["dropped_no_patch"] = int((~has_patch).sum())
    g = g[has_patch]
    dup = g["game_id"].duplicated()
    log["dropped_duplicate_id"] = int(dup.sum())
    g = g[~dup]
    # A winner that names neither team (rare naming mismatches) is unknown: those games count for
    # presence but not for win rates.
    g.loc[(g["winner"] != g["team_blue"]) & (g["winner"] != g["team_red"]), "winner"] = None
    log["winner_unknown"] = int(g["winner"].isna().sum())
    g["datetime_utc"] = pd.to_datetime(g["datetime_utc"])
    log["output"] = len(g)
    return g.reset_index(drop=True), log


def league_tier(tournament: pd.Series) -> pd.Series:
    """'major', 'second' or 'other' per game."""
    league = tournament.str.split("/").str[0]
    event = league.str.replace(r"^\d{4} ", "", regex=True)
    major = league.isin(MAJOR_LEAGUES) | (league.str.match(r"^\d{4} ") & event.isin(MAJOR_EVENTS))
    tier = pd.Series("other", index=tournament.index)
    tier[league.isin(SECOND_DIVISIONS)] = "second"
    tier[major] = "major"
    return tier


def apply_choices(
    games: pd.DataFrame, fearless_games: str = "all", no_ban_games: str = "keep", leagues: str = "all"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop games for the A4, B3 and D1 options. The defaults return `games` unchanged."""
    g = games
    log: dict[str, int] = {}
    if leagues != "all":
        tier = league_tier(g["tournament"])
        keep = tier.eq("major") | (tier.eq("second") & (leagues == "major-second"))
        log[f"dropped_leagues_{leagues}"] = int((~keep).sum())
        g = g[keep]
    if no_ban_games != "keep":
        no_bans = (g["bans_blue"].map(len) + g["bans_red"].map(len)).eq(0)
        series_has_bans = (~no_bans).groupby(g["match_id"]).transform("any")
        drop = no_bans & series_has_bans if no_ban_games == "drop-blind" else no_bans
        log[f"dropped_no_bans_{no_ban_games}"] = int(drop.sum())
        g = g[~drop]
    if fearless_games == "first":
        order = g.sort_values(["datetime_utc", "game_id"])
        first = ~order["match_id"].duplicated()
        later = g["fearless"] & ~first.reindex(g.index)
        log["dropped_later_fearless_games"] = int(later.sum())
        g = g[~later]
    return g.reset_index(drop=True), log


def overlap_check(cargo: pd.DataFrame) -> pd.DataFrame:
    """Games per 2026 patch in gptilt and in Cargo, for patches gptilt covers."""
    g = pd.read_parquet(RAW / "gptilt_games.parquet", columns=["datetime_utc", "patch"])
    g = g[(g["datetime_utc"] >= CARGO_FROM) & (g["patch"] != "")]
    both = (
        pd.DataFrame(
            {
                "gptilt": g.groupby("patch").size(),
                "cargo": cargo[cargo["patch"] != ""].groupby("patch").size(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    both["diff_pct"] = (both["cargo"] - both["gptilt"]) / both["gptilt"].where(both["gptilt"] > 0) * 100
    return both


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--fearless-games",
        choices=["all", "first"],
        default="all",
        help="A4: count every game, or only game 1 of each Fearless series (default: %(default)s)",
    )
    ap.add_argument(
        "--no-ban-games",
        choices=["keep", "drop-blind", "drop-all"],
        default="keep",
        help="B3: drop games without bans; drop-blind only where the series has bans (default: %(default)s)",
    )
    ap.add_argument(
        "--leagues",
        choices=["all", "major", "major-second"],
        default="all",
        help="D1: which tournaments count (default: %(default)s)",
    )


def run(args: argparse.Namespace) -> None:
    out = args.data / "games.parquet"
    if (RAW / "cargo_games_2026.parquet").exists():
        cargo = load_cargo()
        ov = overlap_check(cargo)
        big = ov[(ov["gptilt"] >= 100) & (ov["diff_pct"].abs() > 2)]
        print(
            "2026 overlap, gptilt vs Cargo (patches where gptilt has 100+ games and counts differ by > 2%):"
        )
        print(big.to_string() if len(big) else "  none")
        raw = pd.concat([load_gptilt(), cargo], ignore_index=True)
    else:
        # Train and dev are complete without Cargo; only test patch 26.20 is short (370 of ~531 games).
        print(
            "WARNING: cargo_games_2026.parquet missing; 2026 comes from gptilt, which stops on 25 Oct 2026."
        )
        print("         Rerun `download games2026` and this step before the test labels are used.")
        raw = load_gptilt(end=None)
    games, log = clean(raw)
    games["fearless"] = game_fearless(games)
    games, dropped = apply_choices(games, args.fearless_games, args.no_ban_games, args.leagues)
    if dropped:
        log = {**log, **dropped, "output": len(games)}
    out.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(out, index=False)
    print(f"{out}: {log}")
    by_source = games.groupby("source").size().to_dict()
    print(f"  by source: {by_source}; fearless share by year:")
    print(games.groupby(games["datetime_utc"].dt.year)["fearless"].mean().round(3).to_dict())
