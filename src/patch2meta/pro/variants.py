"""One full build per open data choice, compared with the default build.

  uv run python -m patch2meta.pro variants                build every variant, then compare
  uv run python -m patch2meta.pro variants --only NAME    build one variant (repeatable), then compare
  uv run python -m patch2meta.pro variants --compare      compare the variants already built

A variant is a build with one option changed (see `build --help`). Its files go to
data/processed/variants/<name>/ and its test labels to data/private/variants/<name>/, so each folder
can replace data/processed/ as it is. The comparison reads train and dev only and never opens test
labels. It writes data/processed/variants/comparison.json and prints one row per build.

Comparison columns, computed the same way for the default build:
  games, fearless_games        rows of games.parquet, and the rows flagged Fearless
  kept, train, dev, test       kept patches, in total and per split
  last_delta, mean_reversion   dev Spearman of two baselines against the build's own labels
  ceiling_dev                  noise ceiling on dev
  pre_cut_share_dev            median share of P' games played before the cut, on dev
and, against the default build, on train and dev:
  patches_lost, patches_gained     patches in only one of the two builds
  label_patches_changed            patches in both where some champion's Δpresence moves by > LABEL_MOVE
  prev_patch_changed               patches in both whose P' differs
  rank_agreement_dev               mean per-patch Spearman between the two builds' dev Δpresence
  mean_abs_input_move              mean |change| of the prev_presence input
  dev_patches_shared               dev patches in both builds
  mean_reversion_shared, ceiling_shared   [default, variant] over those shared dev patches only
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from patch2meta.eval.baselines import baseline_predictions
from patch2meta.eval.ceiling import noise_ceiling
from patch2meta.eval.metrics import patch_spearman
from patch2meta.pro import steps
from patch2meta.pro.patches import load_patch_map

ROOT = steps.DATA / "variants"
PRIVATE_ROOT = steps.PRIVATE / "variants"
COMPARISON = ROOT / "comparison.json"
LABEL_MOVE = 0.02
KEYS = ["patch", "champion"]
VARIANTS = {
    "a4-game1": ("A4", "In Fearless series, count game 1 only", ["--fearless-games", "first"]),
    "b3-drop-blind": (
        "B3",
        "Drop games without bans whose series has bans in other games",
        ["--no-ban-games", "drop-blind"],
    ),
    "b3-drop-all": ("B3", "Drop every game without bans", ["--no-ban-games", "drop-all"]),
    "c3-release-day": ("C3", "Forecast time is 00:00 UTC on the release day", ["--cut", "release-day"]),
    "d1-major": ("D1", "Major regions' top leagues and international events only", ["--leagues", "major"]),
    "d1-major-second": (
        "D1",
        "Major leagues, international events and second divisions",
        ["--leagues", "major-second"],
    ),
    "d3-min200": ("D3", "A kept patch needs 200 games", ["--min-patch-games", "200"]),
    "d3-min300": ("D3", "A kept patch needs 300 games", ["--min-patch-games", "300"]),
}


def build_variant(name: str) -> None:
    names = list(steps.STEPS)
    folders = ["--data", str(ROOT / name), "--private", str(PRIVATE_ROOT / name)]
    steps.run(
        names,
        steps.parser(f"patch2meta.pro variants {name}", names).parse_args([*folders, *VARIANTS[name][2]]),
    )


def load_build(data: Path) -> dict:
    pm = load_patch_map(data / "patch_map.csv")
    labels = pd.read_parquet(data / "labels.parquet")  # train and dev; test labels stay in the private folder
    labels["split"] = labels["patch"].map(pm.set_index("patch")["split"])
    return {
        "games": pd.read_parquet(data / "games.parquet", columns=["match_id", "fearless"]),
        "pm": pm,
        "labels": labels,
        "inputs": pd.read_parquet(data / "prepatch_stats.parquet"),
        "data": data,
    }


def summary(b: dict) -> dict:
    pm, labels, inputs = b["pm"], b["labels"], b["inputs"]
    kept = pm[pm["kept"]]
    row = {
        "games": len(b["games"]),
        "fearless_games": int(b["games"]["fearless"].sum()),
        "kept": len(kept),
        **{s: int((kept["split"] == s).sum()) for s in ("train", "dev", "test")},
    }
    dev = labels[labels["split"] == "dev"]
    scored = dev[~dev["new_champion"]]
    b["per_patch"] = {}
    for model, pred in baseline_predictions(inputs).items():
        if model != "no_change":  # constant, scores 0 on every patch
            preds = inputs[KEYS].assign(pred_delta_presence=pred.to_numpy())
            b["per_patch"][model] = patch_spearman(scored.merge(preds, on=KEYS))[0]
            row[model] = round(float(b["per_patch"][model].mean()), 3)
    ceiling = noise_ceiling(dev, "dev", games_path=b["data"] / "games.parquet", save=False)
    b["per_patch"]["ceiling"] = pd.Series({p: v["all"] for p, v in ceiling["per_patch"].items()}).dropna()
    row["ceiling_dev"] = round(ceiling["mean"]["all"], 3)
    per_patch = inputs.groupby("patch").agg(prev=("prev_patch", "first"), n=("prev_n_games", "first"))
    info = pm.set_index("patch")
    share = per_patch["n"] / per_patch["prev"].map(info["n_pro_games"])
    row["pre_cut_share_dev"] = round(float(share[per_patch.index.map(info["split"]) == "dev"].median()), 3)
    return row


def against(a: dict, b: dict) -> dict:
    la, lb = a["labels"], b["labels"]
    both = la.merge(lb, on=KEYS, suffixes=("_a", "_b"))
    move = (both["delta_presence_a"] - both["delta_presence_b"]).abs()
    dev = both[(both["split_a"] == "dev") & ~both["new_champion_a"] & ~both["new_champion_b"]]
    agree = [
        spearmanr(d["delta_presence_a"], d["delta_presence_b"]).statistic
        for _, d in dev.groupby("patch")
        if len(d) >= 3 and d["delta_presence_a"].nunique() > 1 and d["delta_presence_b"].nunique() > 1
    ]
    ia = a["inputs"][a["inputs"]["patch"].isin(la["patch"])]
    inputs = ia.merge(b["inputs"], on=KEYS, suffixes=("_a", "_b"))
    shared = {}  # dev scores of both builds on the dev patches they share, so a dropped patch cannot move them
    for name in ("mean_reversion", "ceiling"):
        sa, sb = a["per_patch"][name], b["per_patch"][name]
        common = sa.index.intersection(sb.index)
        shared[f"{name}_shared"] = [round(float(sa[common].mean()), 3), round(float(sb[common].mean()), 3)]
    return {
        "dev_patches_shared": len(common),
        **shared,
        "patches_lost": len(set(la["patch"]) - set(lb["patch"])),
        "patches_gained": len(set(lb["patch"]) - set(la["patch"])),
        "label_patches_changed": int((move.groupby(both["patch"]).max() > LABEL_MOVE).sum()),
        "prev_patch_changed": int(
            (both["prev_patch_a"] != both["prev_patch_b"]).groupby(both["patch"]).any().sum()
        ),
        "rank_agreement_dev": round(float(np.mean(agree)), 3) if agree else None,
        "mean_abs_input_move": round(
            float((inputs["prev_presence_a"] - inputs["prev_presence_b"]).abs().mean()), 4
        ),
    }


def compare(names: list[str]) -> list[dict]:
    base = load_build(steps.DATA)
    rows = [
        {"name": "default", "question": None, "description": "Today's build", "flags": [], **summary(base)}
    ]
    for name in names:
        if not (ROOT / name / "labels.parquet").exists():
            print(f"  {name}: not built, skipped")
            continue
        question, description, flags = VARIANTS[name]
        b = load_build(ROOT / name)
        rows.append({"name": name, "question": question, "description": description, "flags": flags,
                     **summary(b), **against(base, b)})  # fmt: skip
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="patch2meta.pro variants")
    ap.add_argument("--only", action="append", choices=list(VARIANTS), help="build only this variant")
    ap.add_argument("--compare", action="store_true", help="skip the builds and compare what is there")
    args = ap.parse_args(argv)
    names = args.only or list(VARIANTS)
    if not args.compare:
        for name in names:
            print(f"==== variant {name}: {' '.join(VARIANTS[name][2])}")
            shutil.rmtree(ROOT / name, ignore_errors=True)
            build_variant(name)
    rows = compare(list(VARIANTS))
    ROOT.mkdir(parents=True, exist_ok=True)
    COMPARISON.write_text(json.dumps({"label_move": LABEL_MOVE, "builds": rows}, indent=2))
    table = pd.DataFrame(rows).set_index("name").drop(columns=["question", "description", "flags"])
    print(f"{COMPARISON}:")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string())
