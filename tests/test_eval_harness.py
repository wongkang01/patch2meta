"""The eval loader on toy files: the test guard and its log, missing and non-finite predictions,
patch IDs read as numbers, and new-champion rows scored apart from the headline numbers."""

import numpy as np
import pandas as pd
import pytest

from patch2meta.eval import load
from patch2meta.eval.__main__ import score_model


def label_rows(patch: str) -> pd.DataFrame:
    return pd.DataFrame({
        "patch": patch, "prev_patch": "x", "champion": list("ABCDE"), "n_games": 100,
        "presence": [0.5, 0.4, 0.3, 0.2, 0.1], "prev_presence": 0.3, "delta_presence": [0.2, 0.1, 0.0, -0.1, -0.2],
        "fearless_share": 1.0, "changed": False, "new_champion": False,
    })  # fmt: skip


@pytest.fixture
def files(tmp_path, monkeypatch):
    pd.DataFrame({"patch": ["25.04", "26.01"], "split": ["dev", "test"]}).to_csv(
        tmp_path / "pm.csv", index=False
    )
    label_rows("25.04").to_parquet(tmp_path / "labels.parquet")
    label_rows("26.01").to_parquet(tmp_path / "labels_test.parquet")
    for name, path in {
        "LABELS": tmp_path / "labels.parquet",
        "LABELS_TEST": tmp_path / "labels_test.parquet",
        "PATCH_MAP": tmp_path / "pm.csv",
        "RESULTS": tmp_path / "results",
        "TEST_LOG": tmp_path / "results/test_runs.log",
    }.items():
        monkeypatch.setattr(load, name, path)
    return tmp_path


def test_test_labels_need_the_flag_and_a_second_read_needs_another(files):
    with pytest.raises(load.TestSplitGuard):
        load.load_labels(["test"])
    assert set(load.load_labels(["test"], agree_to_test=True)["patch"]) == {"26.01"}
    with pytest.raises(load.TestSplitGuard, match="already read"):
        load.load_labels(["test"], agree_to_test=True)
    load.load_labels(["test"], agree_to_test=True, rescore_test=True)
    assert len((files / "results/test_runs.log").read_text().splitlines()) == 2
    assert set(load.load_labels(["dev"])["patch"]) == {"25.04"}


def test_non_finite_predictions_count_as_missing(files):
    labels = load.load_labels(["dev"])
    preds = labels[["patch", "champion"]].assign(pred_delta_presence=[0.1, np.nan, 0.0, np.inf, -0.1])
    merged, coverage = load.join(labels, preds)
    assert coverage == {"missing_rows": 2, "non_finite_rows": 2, "extra_rows": 0}
    res, _ = score_model(merged, pd.Series(100, index=merged.index), None, None)
    assert res["spearman"]["all"]["n_patches"] == 1  # scored on the three finite rows only


def test_patch_ids_read_as_numbers_are_rejected(files):
    path = files / "m.csv"
    pd.DataFrame({"model": "m", "patch": ["25.4"], "champion": ["A"], "pred_delta_presence": [0.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="not in patch_map"):
        load.load_predictions(path)


def test_new_champion_rows_are_scored_apart(files):
    labels = load.load_labels(["dev"])
    labels.loc[labels["champion"] == "A", "new_champion"] = True
    pred = labels["delta_presence"].where(labels["champion"] != "A", -1.0)  # perfect except the new champion
    merged, _ = load.join(labels, labels[["patch", "champion"]].assign(pred_delta_presence=pred))
    res, _ = score_model(merged, pd.Series(100, index=merged.index), None, None)
    assert res["spearman"]["all"]["mean"] == pytest.approx(1.0)
    assert res["new_champion_pooled"] == {"n_rows": 1, "mae": pytest.approx(1.2), "spearman": None}
