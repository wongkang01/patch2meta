"""Fearless draft flags, inferred from the drafts.

In Fearless draft a champion picked in one game of a series cannot be picked again in that series.
Leaguepedia has no format field, so the format is inferred per tournament:
  1. A multi-game series is Fearless-looking if no champion is picked twice across its games.
  2. A tournament is Fearless if at least 90% of its multi-game series are Fearless-looking.
  3. Tournaments with only single-game series count as standard.
Every game takes its tournament's flag.
"""

import pandas as pd

THRESHOLD = 0.9


def series_fearless(games: pd.DataFrame) -> pd.Series:
    """1.0 or 0.0 per multi-game series (match_id); series with one game are left out."""
    picks = games[["match_id", "picks_blue", "picks_red"]].copy()
    picks["picks"] = [list(b) + list(r) for b, r in zip(picks["picks_blue"], picks["picks_red"])]
    long = picks[["match_id", "picks"]].explode("picks")
    n_games = games.groupby("match_id").size()
    multi = n_games[n_games >= 2].index
    long = long[long["match_id"].isin(multi)]
    max_repeat = long.groupby(["match_id", "picks"]).size().groupby(level="match_id").max()
    return (max_repeat <= 1).astype(float).rename("series_fearless")


def tournament_fearless(games: pd.DataFrame) -> pd.Series:
    """Share of Fearless-looking series per tournament (NaN when it has no multi-game series)."""
    series = series_fearless(games)
    tour_of_series = games.drop_duplicates("match_id").set_index("match_id")["tournament"]
    return series.groupby(tour_of_series.reindex(series.index)).mean().rename("fearless_share")


def game_fearless(games: pd.DataFrame) -> pd.Series:
    """True for games in a Fearless tournament, aligned with `games.index`."""
    share = tournament_fearless(games)
    return games["tournament"].map(share).fillna(0.0).ge(THRESHOLD).rename("fearless")
