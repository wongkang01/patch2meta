"""Per-patch metrics. Every metric is computed inside each patch, then averaged over patches.

Conventions:
  - A patch with fewer than MIN_ROWS rows in a subset is skipped for that subset.
  - Constant predictions (or a constant target) inside a patch score Spearman 0: no ranking skill.
    The number of such patches is reported next to the mean.
  - Ties use average ranks (scipy's default).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_ROWS = 3
TOP_K = 10
VAR_FLOOR = 1e-4
PRED, TRUE = "pred_delta_presence", "delta_presence"


def patch_spearman(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """Spearman per patch, and the number of patches scored 0 because a side was constant."""
    out, constant = {}, 0
    for patch, d in df.groupby("patch", sort=False):
        if len(d) < MIN_ROWS:
            continue
        if d[PRED].nunique() < 2 or d[TRUE].nunique() < 2:
            out[patch] = 0.0
            constant += 1
            continue
        out[patch] = float(spearmanr(d[PRED], d[TRUE]).statistic)
    return pd.Series(out, dtype=float), constant


def patch_topk_precision(df: pd.DataFrame, k: int = TOP_K) -> tuple[pd.Series, pd.Series]:
    """|predicted top-k ∩ realised top-k| / k per patch, for risers and for fallers.

    Ties in the target or the prediction are broken by champion name, so results are deterministic.
    Constant predictions score 0.
    """
    risers, fallers = {}, {}
    for patch, d in df.groupby("patch", sort=False):
        if len(d) < 2 * k:
            continue
        if d[PRED].nunique() < 2:
            risers[patch] = fallers[patch] = 0.0
            continue
        d = d.sort_values("champion")
        for out, asc in ((risers, False), (fallers, True)):
            pred = set(d.sort_values(PRED, ascending=asc, kind="stable")["champion"].head(k))
            true = set(d.sort_values(TRUE, ascending=asc, kind="stable")["champion"].head(k))
            out[patch] = len(pred & true) / k
    return pd.Series(risers, dtype=float), pd.Series(fallers, dtype=float)


def label_variance(df: pd.DataFrame, prev_n: pd.Series) -> pd.Series:
    """Sampling variance of Δpresence: p(1-p)/n + p'(1-p')/n'."""
    p, q = df["presence"], df["prev_presence"]
    return p * (1 - p) / df["n_games"] + q * (1 - q) / prev_n


def patch_mae(df: pd.DataFrame, prev_n: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Plain and inverse-variance-weighted MAE per patch."""
    err = (df[PRED] - df[TRUE]).abs()
    w = 1.0 / np.maximum(label_variance(df, prev_n), VAR_FLOOR)
    plain = err.groupby(df["patch"], sort=False).mean()
    weighted = (err * w).groupby(df["patch"], sort=False).sum() / w.groupby(df["patch"], sort=False).sum()
    return plain, weighted
