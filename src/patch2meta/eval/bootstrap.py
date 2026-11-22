"""Bootstrap confidence intervals, the paired model comparison (H1, H2) and the noise ceiling."""

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 0


def patch_ci(values: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> list[float] | None:
    """95% percentile interval of the mean over patches, resampling patches."""
    v = values.dropna().to_numpy()
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def paired_difference(a: pd.Series, b: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean over patches of (a - b), with a 95% interval from resampling patches.

    a and b are per-patch scores of two models on the same subset (e.g. Spearman on changed champions).
    Only patches scored for both models count.
    """
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = both["a"] - both["b"]
    return {
        "mean_difference": float(diff.mean()) if len(diff) else None,
        "ci": patch_ci(diff, n_boot, seed),
        "n_patches": len(diff),
        "share_patches_a_better": float((diff > 0).mean()) if len(diff) else None,
    }


def presence_bootstrap(series_counts: np.ndarray, series_games: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """Presence per champion for `n_boot` resamples of whole series: shape (n_boot, champions)."""
    s = len(series_games)
    weights = rng.multinomial(s, np.full(s, 1.0 / s), size=n_boot)  # (n_boot, series)
    return (weights @ series_counts) / (weights @ series_games)[:, None]


def noise_ceiling_patch(
    cur: tuple[np.ndarray, np.ndarray],
    prev: tuple[np.ndarray, np.ndarray],
    observed_delta: np.ndarray,
    rows: np.ndarray,
    n_boot: int,
    rng,
) -> float | None:
    """sqrt(max(V - N, 0) / V) for one patch and one subset of champions (`rows` = champion codes).

    V is the variance of the observed Δpresence across the subset; N is the mean over the subset of each
    champion's sampling variance, estimated by resampling series on P and on P'.
    """
    if len(rows) < 3:
        return None
    delta_b = presence_bootstrap(*cur, n_boot, rng) - presence_bootstrap(*prev, n_boot, rng)
    noise = float(delta_b[:, rows].var(axis=0).mean())
    v = float(np.var(observed_delta))
    if v <= 0:
        return None
    return float(np.sqrt(max(v - noise, 0.0) / v))
