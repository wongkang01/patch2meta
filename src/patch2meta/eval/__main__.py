"""Score predictions. Usage:

  uv run python -m patch2meta.eval baselines                 write no_change, last_delta, mean_reversion
  uv run python -m patch2meta.eval strata [--labeller ID]    row and patch strata from block labels
  uv run python -m patch2meta.eval ceiling --split dev       noise ceiling
  uv run python -m patch2meta.eval score --split dev [--models a,b] [--compare text:lightgbm@changed]

`--split` takes train, dev, test or dev+test (pooled). Test needs --i-agree-to-score-test. Every read of
the test labels is logged in data/results/test_runs.log, and a second read also needs
--i-agree-to-rescore-test. Results go to data/results/<model>_<split>.json and a printed table.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from patch2meta.eval.baselines import write_baselines
from patch2meta.eval.bootstrap import paired_difference, patch_ci
from patch2meta.eval.ceiling import noise_ceiling
from patch2meta.eval.load import (
    PATCH_MAP,
    PREDICTIONS,
    RESULTS,
    TestSplitGuard,
    join,
    load_labels,
    load_predictions,
)
from patch2meta.eval.metrics import PRED, TRUE, patch_mae, patch_spearman, patch_topk_precision
from patch2meta.eval.strata import SYSTEM_TYPES, TYPE_PRIORITY, load_strata, write_strata

FEARLESS_CUTOFF = 0.5


def summary(per_patch: pd.Series, constant: int | None = None) -> dict:
    out = {
        "mean": float(per_patch.mean()) if len(per_patch) else None,
        "ci": patch_ci(per_patch),
        "n_patches": len(per_patch),
    }
    if constant is not None:
        out["n_constant_patches"] = constant
    return out


def spearman_summary(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    per_patch, constant = patch_spearman(df)
    return summary(per_patch, constant), per_patch


def pooled(d: pd.DataFrame) -> dict:
    """MAE and Spearman over every row of a small stratum, pooled across patches.

    New-champion rows number at most 2 per patch, too few for a per-patch Spearman.
    """
    ranked = len(d) >= 3 and d[PRED].nunique() > 1 and d[TRUE].nunique() > 1
    return {
        "n_rows": len(d),
        "mae": float((d[PRED] - d[TRUE]).abs().mean()) if len(d) else None,
        "spearman": float(spearmanr(d[PRED], d[TRUE]).statistic) if ranked else None,
    }


def score_model(merged: pd.DataFrame, prev_n: pd.Series, row_strata, patch_strata) -> tuple[dict, dict]:
    """All metrics for one model on one split, plus per-patch Spearman series for comparisons.

    Rows without a finite prediction are left out here and counted as missing by `join`.
    """
    df = merged[np.isfinite(merged["pred_delta_presence"].astype(float))]
    base = df[~df["new_champion"]]
    subsets = {
        "all": base,
        "changed": base[base["changed"]],
        "unchanged": base[~base["changed"]],
        "new_champion": df[df["new_champion"]],
    }
    res: dict = {"spearman": {}}
    per_patch: dict[str, pd.Series] = {}
    for name, sub in subsets.items():
        res["spearman"][name], per_patch[name] = spearman_summary(sub)
    res["new_champion_pooled"] = pooled(subsets["new_champion"])
    risers, fallers = patch_topk_precision(base)
    res["top10_riser_precision"], res["top10_faller_precision"] = summary(risers), summary(fallers)
    plain, weighted = patch_mae(base, prev_n.reindex(base.index))
    res["mae"], res["weighted_mae"] = summary(plain), summary(weighted)

    fearless = base.groupby("patch")["fearless_share"].first() >= FEARLESS_CUTOFF
    res["by_format"] = {
        fmt: summary(per_patch["all"][per_patch["all"].index.map(fearless) == flag])
        for fmt, flag in (("fearless", True), ("standard", False))
    }
    if row_strata is not None:
        changed = subsets["changed"].merge(row_strata, on=["patch", "champion"], how="left")
        res["within_buffs"] = spearman_summary(changed[changed["intent"] == "buff"])[0]
        res["within_nerfs"] = spearman_summary(changed[changed["intent"] == "nerf"])[0]
        res["by_change_type"] = {
            t: spearman_summary(changed[changed["change_type"] == t])[0] for t in TYPE_PRIORITY
        }
        flags = patch_strata.set_index("patch")
        res["by_system_change"] = {
            t: {
                str(v): summary(per_patch["all"][per_patch["all"].index.map(flags[f"has_{t}"]) == v])
                for v in (True, False)
            }
            for t in SYSTEM_TYPES
        }
    return res, per_patch


def parse_split(value: str) -> list[str]:
    splits = value.split("+")
    if not set(splits) <= {"train", "dev", "test"}:
        raise argparse.ArgumentTypeError(f"bad split {value!r}")
    return splits


def cmd_score(args) -> None:
    labels = load_labels(
        args.split, agree_to_test=args.i_agree_to_score_test, rescore_test=args.i_agree_to_rescore_test
    )
    pm = pd.read_csv(PATCH_MAP, dtype={"patch": str, "prev_patch": str}).set_index("patch")
    prev_n = labels["prev_patch"].map(pm["n_pro_games"])
    row_strata, patch_strata = load_strata()
    files = sorted(PREDICTIONS.glob("*.csv"))
    if args.models:
        wanted = set(args.models.split(","))
        files = [f for f in files if f.stem in wanted]
    split_name = "+".join(args.split)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table, per_patch = [], {}
    for path in files:
        merged, coverage = join(labels, load_predictions(path))
        res, per_patch[path.stem] = score_model(merged, prev_n, row_strata, patch_strata)
        res = {
            "model": path.stem,
            "split": split_name,
            **coverage,
            "complete": coverage["missing_rows"] == 0,
            **res,
        }
        (RESULTS / f"{path.stem}_{split_name}.json").write_text(json.dumps(res, indent=2))
        sp = res["spearman"]
        table.append(
            {
                "model": path.stem,
                "all": sp["all"]["mean"],
                "changed": sp["changed"]["mean"],
                "unchanged": sp["unchanged"]["mean"],
                "new_champ_mae": res["new_champion_pooled"]["mae"],
                "risers@10": res["top10_riser_precision"]["mean"],
                "fallers@10": res["top10_faller_precision"]["mean"],
                "MAE": res["mae"]["mean"],
                "wMAE": res["weighted_mae"]["mean"],
                "missing": coverage["missing_rows"],
            }
        )
    if not table:
        print(f"no prediction files in {PREDICTIONS}")
        return
    print(f"split {split_name}: {labels['patch'].nunique()} patches, {len(labels)} rows")
    print(pd.DataFrame(table).set_index("model").round(3).to_string())
    if not labels["changed"].any():
        print("  no row has changed=True: rebuild the labels with person 1's changes.jsonl to fill it.")
    for spec in args.compare or []:
        pair, _, subset = spec.partition("@")
        a, _, b = pair.partition(":")
        subset = subset or "changed"
        cmp = paired_difference(per_patch[a][subset], per_patch[b][subset])
        (RESULTS / f"compare_{a}_vs_{b}_{subset}_{split_name}.json").write_text(json.dumps(cmp, indent=2))
        print(f"  {a} - {b} on {subset}: {cmp}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="patch2meta.eval")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("baselines")
    st = sub.add_parser("strata")
    st.add_argument("--labeller", default=None)
    for name in ("score", "ceiling"):
        p = sub.add_parser(name)
        p.add_argument("--split", type=parse_split, default=["dev"])
        p.add_argument("--i-agree-to-score-test", action="store_true")
        p.add_argument(
            "--i-agree-to-rescore-test",
            action="store_true",
            help="read the test labels again although test_runs.log already has a run",
        )
    sc = sub.choices["score"]
    sc.add_argument("--models", default=None, help="comma-separated model names (default: every file)")
    sc.add_argument("--compare", action="append", help="A:B@subset, e.g. text:lightgbm@changed")
    args = ap.parse_args()
    try:
        if args.command == "baselines":
            write_baselines()
        elif args.command == "strata":
            write_strata(pd.read_csv(PATCH_MAP, dtype={"patch": str}), args.labeller)
        elif args.command == "ceiling":
            labels = load_labels(
                args.split,
                agree_to_test=args.i_agree_to_score_test,
                rescore_test=args.i_agree_to_rescore_test,
            )
            res = noise_ceiling(labels, "+".join(args.split))
            print(f"noise ceiling ({res['split']}): {res['mean']} over {res['n_patches']} patches")
        else:
            cmd_score(args)
    except TestSplitGuard as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
