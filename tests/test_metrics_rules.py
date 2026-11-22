"""Weighted MAE, the patch bootstrap interval, the top-k row minimum and the noise-ceiling formula."""

import numpy as np
import pandas as pd
import pytest

from patch2meta.eval import bootstrap
from patch2meta.eval.bootstrap import noise_ceiling_patch, patch_ci
from patch2meta.eval.metrics import patch_mae, patch_topk_precision


def test_weighted_mae_uses_inverse_label_variance():
    df = pd.DataFrame({
        "patch": "p", "presence": [0.5, 0.0], "prev_presence": [0.5, 0.0], "n_games": 100,
        "delta_presence": [0.0, 0.0], "pred_delta_presence": [0.1, 0.2],
    })  # fmt: skip
    plain, weighted = patch_mae(df, pd.Series([100, 100]))
    # row 1 has variance 0.25/100 + 0.25/100 = 0.005 (weight 200); row 2 has 0, floored at 1e-4 (weight 10,000)
    assert plain["p"] == pytest.approx(0.15)
    assert weighted["p"] == pytest.approx((0.1 * 200 + 0.2 * 10_000) / 10_200)


def test_patch_ci_is_close_to_the_normal_95_percent_interval():
    v = pd.Series(np.linspace(0.0, 1.0, 10))
    lo, hi = patch_ci(v)
    half = 1.96 * v.std(ddof=0) / np.sqrt(len(v))
    assert lo == pytest.approx(v.mean() - half, abs=0.03)
    assert hi == pytest.approx(v.mean() + half, abs=0.03)
    assert patch_ci(pd.Series([0.3])) is None


def test_topk_precision_needs_two_k_rows():
    rows = [
        {"patch": "p", "champion": f"c{i:02d}", "pred_delta_presence": float(i), "delta_presence": float(i)}
        for i in range(19)
    ]
    risers, fallers = patch_topk_precision(pd.DataFrame(rows))
    assert risers.empty and fallers.empty


def test_noise_ceiling_is_the_square_root_of_the_signal_share(monkeypatch):
    observed = np.array([0.0, 1.0, 2.0, 3.0])
    noise = 0.75 * observed.var()  # leaves a signal share of 0.25
    draws = np.repeat(np.array([[-1.0], [1.0]]) * np.sqrt(noise), 4, axis=1)
    calls = iter([draws, np.zeros_like(draws)])
    monkeypatch.setattr(bootstrap, "presence_bootstrap", lambda *args: next(calls))
    ceiling = noise_ceiling_patch((None, None), (None, None), observed, np.arange(4), 2, None)
    assert ceiling == pytest.approx(0.5)
