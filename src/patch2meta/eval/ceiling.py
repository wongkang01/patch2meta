"""Noise ceiling: an estimate of the best mean within-patch correlation a model could reach given label noise.

It is a reliability bound that assumes independent sampling noise, applied to Spearman, so a model can
land slightly above it on a given split.

For each patch, series (match_id) are resampled with replacement on P and on P' to estimate each
champion's sampling variance of Δpresence. Series are resampled whole because Fearless draft ties the
games of a series together. See bootstrap.noise_ceiling_patch for the formula.
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from patch2meta.eval.bootstrap import noise_ceiling_patch
from patch2meta.pro.counts import GameTable, build
from patch2meta.pro.labels import champion_release_dates

GAMES = Path("data/processed/games.parquet")
RESULTS = Path("data/results")
N_BOOT = 200


def series_matrix(
    t: GameTable, series_of_game: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per series in `mask`: games where each champion is present (series x champions), and game counts."""
    games = np.flatnonzero(mask)
    codes, _ = pd.factorize(series_of_game[games])
    local = np.full(len(mask), -1)
    local[games] = codes
    keep = mask[t.present_game]
    counts = np.zeros((codes.max() + 1, t.n_champions))
    np.add.at(counts, (local[t.present_game[keep]], t.present_champ[keep]), 1.0)
    return counts, np.bincount(codes).astype(float)


def noise_ceiling(
    labels: pd.DataFrame,
    split: str,
    n_boot: int = N_BOOT,
    seed: int = 0,
    games_path: Path = GAMES,
    save: bool = True,
) -> dict:
    games = pd.read_parquet(games_path)
    t = build(games, sorted(champion_release_dates().index))
    code = {name: i for i, name in enumerate(t.champions)}
    series = games["match_id"].to_numpy()
    per_patch: dict[str, dict[str, float | None]] = {}
    for patch, d in labels.groupby("patch"):
        cur = series_matrix(t, series, t.patch == patch)
        prev = series_matrix(t, series, t.patch == d["prev_patch"].iloc[0])
        base = d[~d["new_champion"]]
        per_patch[patch] = {}
        for j, (name, sub) in enumerate(
            (("all", base), ("changed", base[base["changed"]]), ("unchanged", base[~base["changed"]]))
        ):
            # One stream per patch and subset, so no estimate depends on the other subsets or patches
            # (the `all` ceiling stays put when `changed` flags change, and builds share draws per patch).
            rng = np.random.default_rng([seed, zlib.crc32(str(patch).encode()), j])
            rows = sub["champion"].map(code).to_numpy()
            per_patch[patch][name] = noise_ceiling_patch(
                cur, prev, sub["delta_presence"].to_numpy(), rows, n_boot, rng
            )
    table = pd.DataFrame(per_patch).T
    result = {
        "split": split,
        "n_boot": n_boot,
        "mean": {c: (float(table[c].dropna().mean()) if table[c].notna().any() else None) for c in table},
        "n_patches": {c: int(table[c].notna().sum()) for c in table},
        "per_patch": per_patch,
    }
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"noise_ceiling_{split}.json").write_text(json.dumps(result, indent=2))
    return result
