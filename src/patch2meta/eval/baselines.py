"""Baselines that need no model, written for every row of prepatch_stats.parquet (all splits).

no_change       0 for every champion
last_delta      prev_delta_1 (0 when missing): the last pro patch's shift continues
mean_reversion  -prev_presence: champions that are drafted a lot tend to fall
"""

from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from patch2meta.schema import Prediction

PREPATCH = Path("data/processed/prepatch_stats.parquet")
PREDICTIONS = Path("data/predictions")


def baseline_predictions(stats: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "no_change": pd.Series(0.0, index=stats.index),
        "last_delta": stats["prev_delta_1"].fillna(0.0).astype(float),
        "mean_reversion": -stats["prev_presence"].astype(float),
    }


def write_baselines() -> None:
    stats = pd.read_parquet(PREPATCH)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for model, pred in baseline_predictions(stats).items():
        df = pd.DataFrame(
            {
                "model": model,
                "patch": stats["patch"],
                "champion": stats["champion"],
                "pred_delta_presence": pred,
            }
        )
        TypeAdapter(list[Prediction]).validate_python(df.to_dict("records"))
        df.to_csv(PREDICTIONS / f"{model}.csv", index=False)
        print(f"{PREDICTIONS / model}.csv: {len(df)} rows")
