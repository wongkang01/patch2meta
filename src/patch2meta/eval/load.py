"""Read labels and predictions, and join them on (patch, champion)."""

import getpass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from patch2meta.schema import Prediction

LABELS = Path("data/processed/labels.parquet")
LABELS_TEST = Path("data/private/labels_test.parquet")
PATCH_MAP = Path("data/processed/patch_map.csv")
PREDICTIONS = Path("data/predictions")
RESULTS = Path("data/results")
TEST_LOG = RESULTS / "test_runs.log"


class TestSplitGuard(RuntimeError):
    pass


def split_map() -> pd.Series:
    pm = pd.read_csv(PATCH_MAP, dtype={"patch": str})
    return pm.set_index("patch")["split"]


def load_labels(splits: list[str], agree_to_test: bool = False, rescore_test: bool = False) -> pd.DataFrame:
    """Labels for the given splits. Test labels need the explicit agreement flag and the private file.

    Every read of the test labels is appended to data/results/test_runs.log. Once the log has an entry,
    a further read also needs `rescore_test`, so a second test run is always a deliberate choice.
    """
    frames = []
    if {"train", "dev"} & set(splits):
        frames.append(pd.read_parquet(LABELS))
    if "test" in splits:
        if not agree_to_test:
            raise TestSplitGuard("the test split runs once, after the freeze: pass --i-agree-to-score-test")
        if not LABELS_TEST.exists():
            raise TestSplitGuard(f"{LABELS_TEST} is not here; person 2 holds the test labels")
        earlier = TEST_LOG.read_text().splitlines() if TEST_LOG.exists() else []
        if earlier and not rescore_test:
            raise TestSplitGuard(
                f"the test labels were already read {len(earlier)} time(s), last: {earlier[-1]!r}. "
                "Pass --i-agree-to-rescore-test only if the team agreed to another test run"
            )
        RESULTS.mkdir(parents=True, exist_ok=True)
        with TEST_LOG.open("a") as f:
            f.write(f"{datetime.now(UTC).isoformat()} test labels read by {getpass.getuser()}\n")
        frames.append(pd.read_parquet(LABELS_TEST))
    labels = pd.concat(frames, ignore_index=True)
    labels["split"] = labels["patch"].map(split_map())
    return labels[labels["split"].isin(splits)].reset_index(drop=True)


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patch": str, "champion": str, "model": str})
    TypeAdapter(list[Prediction]).validate_python(df.to_dict("records"))
    models = set(df["model"])
    if models != {path.stem}:
        raise ValueError(f"{path}: `model` column {sorted(models)} must equal the file name {path.stem!r}")
    if df.duplicated(["patch", "champion"]).any():
        raise ValueError(f"{path}: duplicate (patch, champion) rows")
    unknown = sorted(set(df["patch"]) - set(split_map().index))
    if unknown:
        raise ValueError(
            f"{path}: patch IDs not in patch_map.csv: {unknown[:10]}. Write and read patch IDs as strings "
            "(pd.read_csv(..., dtype={'patch': str})); a plain read turns 26.10 into 26.1"
        )
    return df


def join(labels: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Left-join predictions onto the label keys; report missing and extra rows.

    A label row counts as missing when the file has no row for it or when its prediction is NaN or
    infinite, so a file that leaves hard rows empty is never reported as complete.
    """
    keys = ["patch", "champion"]
    merged = labels.merge(preds[[*keys, "pred_delta_presence"]], on=keys, how="left", indicator=True)
    in_split = preds[preds["patch"].isin(labels["patch"].unique())]
    extra = len(
        in_split.merge(labels[keys], on=keys, how="left", indicator=True).query("_merge == 'left_only'")
    )
    no_row = merged["_merge"] == "left_only"
    not_finite = ~no_row & ~np.isfinite(merged["pred_delta_presence"].astype(float))
    coverage = {
        "missing_rows": int(no_row.sum() + not_finite.sum()),
        "non_finite_rows": int(not_finite.sum()),
        "extra_rows": extra,
    }
    return merged.drop(columns="_merge"), coverage
