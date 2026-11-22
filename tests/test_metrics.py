import numpy as np
import pandas as pd
import pytest

from patch2meta.eval.bootstrap import noise_ceiling_patch, paired_difference
from patch2meta.eval.metrics import patch_spearman, patch_topk_precision


def frame(pred, true, patches=("p1", "p2")):
    rows = []
    for p in patches:
        for i, (a, b) in enumerate(zip(pred, true)):
            rows.append({"patch": p, "champion": f"c{i:02d}", "pred_delta_presence": a, "delta_presence": b})
    return pd.DataFrame(rows)


def test_perfect_shuffled_and_constant():
    rng = np.random.default_rng(0)
    true = rng.normal(size=40)
    perfect, constant = patch_spearman(frame(true, true))
    assert perfect.tolist() == [1.0, 1.0] and constant == 0
    shuffled_frames = [frame(rng.permutation(true), true, patches=[f"p{i}"]) for i in range(200)]
    shuffled, _ = patch_spearman(pd.concat(shuffled_frames))
    assert abs(shuffled.mean()) < 0.05
    zeros, constant = patch_spearman(frame(np.zeros(40), true))
    assert zeros.tolist() == [0.0, 0.0] and constant == 2


def test_topk_precision():
    true = np.arange(30, dtype=float)
    risers, fallers = patch_topk_precision(frame(true, true))
    assert risers.tolist() == [1.0, 1.0] and fallers.tolist() == [1.0, 1.0]
    risers, _ = patch_topk_precision(frame(-true, true))
    assert risers.tolist() == [0.0, 0.0]


def test_paired_difference():
    a = pd.Series({"p1": 0.5, "p2": 0.4, "p3": 0.6})
    b = pd.Series({"p1": 0.3, "p2": 0.3, "p3": 0.3, "p4": 0.9})
    res = paired_difference(a, b)
    assert res["n_patches"] == 3
    assert res["mean_difference"] == pytest.approx(0.2)
    assert res["share_patches_a_better"] == 1.0


def test_noise_ceiling_is_one_without_noise_and_lower_with_it():
    rng = np.random.default_rng(0)
    # 40 series of one game each; champion j is present in every game with probability p_j
    p_cur, p_prev = np.linspace(0.05, 0.9, 20), np.linspace(0.9, 0.05, 20)
    deterministic = (np.ones((40, 20)) * (p_cur > 0.5), np.ones(40))
    cur = ((rng.random((400, 20)) < p_cur).astype(float), np.ones(400))
    prev = ((rng.random((400, 20)) < p_prev).astype(float), np.ones(400))
    observed = cur[0].mean(0) - prev[0].mean(0)
    rows = np.arange(20)
    noisy = noise_ceiling_patch(cur, prev, observed, rows, 200, rng)
    assert 0.9 < noisy < 1.0  # large true spread, small noise
    flat = noise_ceiling_patch(deterministic, deterministic, observed, rows, 50, rng)
    assert flat == pytest.approx(1.0)
