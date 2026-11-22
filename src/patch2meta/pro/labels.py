"""Targets and model inputs, one row per (kept patch P, champion released by P's release date).

Writes:
  data/processed/labels.parquet          ChampionPatch rows for train and dev (shared)
  data/private/labels_test.parquet       ChampionPatch rows for test (scored only in the single test run)
  data/processed/prepatch_stats.parquet  PrePatchStats rows for every kept patch, all splits

Targets use all games on P and P'. Inputs use only games dated before cut(P), the forecast time.
`changed` comes from person 1's data/processed/changes.jsonl: a gameplay block for the champion in any
Riot patch after P' up to P.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from patch2meta.pro.counts import ROLES, Counts, GameTable, build, count
from patch2meta.pro.names import canonical, canonical_names
from patch2meta.pro.patches import kept_patches, load_patch_map, patch_key
from patch2meta.schema import ChampionPatch, PrePatchStats

CHAMPIONS = Path("data/raw/pro/cargo_champions.parquet")
CHANGES = Path("data/processed/changes.jsonl")  # person 1's file, read from the team folder in every build
DATA_START = pd.Timestamp("2019-01-01")


def champion_release_dates() -> pd.Series:
    c = pd.read_parquet(CHAMPIONS)
    return pd.Series(pd.to_datetime(c["ReleaseDate"]).to_numpy(), index=c["Name"]).sort_index()


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--allow-unknown-blocks",
        action="store_true",
        help="build even if changes.jsonl has champion names or patch IDs that do not match (default: stop)",
    )


def changed_pairs(
    pm: pd.DataFrame, allow_unknown: bool = False, known: frozenset[str] | None = None
) -> set[tuple[str, str]] | None:
    """(P, champion) pairs with a gameplay champion block in any Riot patch in (P', P]; None without
    changes.jsonl. Blocks with `gameplay` False (only bug fixes or cosmetic lines) do not count.

    A champion block whose name is not a canonical name or alias, or whose patch ID is not in the patch
    map, stops the build: skipping it would leave `changed` False for that champion without warning.
    """
    if not CHANGES.exists():
        return None
    known = canonical_names() if known is None else known
    patches = set(pm["patch"])
    by_patch: dict[str, set[str]] = {}
    unknown_names: set[str] = set()
    unknown_patches: set[str] = set()
    for line in CHANGES.read_text().splitlines():
        if not line.strip():
            continue
        block = json.loads(line)
        if "gameplay" not in block:
            raise ValueError(f"{CHANGES}: blocks have no `gameplay` field; use v2 or later (MANIFEST.md)")
        if block["kind"] != "champion":
            continue
        if block["patch"] not in patches:
            unknown_patches.add(block["patch"])
            continue
        try:
            name = canonical(block["name"], known)
        except KeyError:
            unknown_names.add(block["name"])
            continue
        if block["gameplay"]:
            by_patch.setdefault(block["patch"], set()).add(name)
    problems = []
    if unknown_names:
        problems.append(f"{len(unknown_names)} block names are not champions: {sorted(unknown_names)[:20]}")
    if unknown_patches:
        problems.append(
            f"{len(unknown_patches)} patch IDs are not in the patch map: {sorted(unknown_patches)[:20]}"
        )
    if problems and not allow_unknown:
        raise ValueError(
            f"{CHANGES}: {'; '.join(problems)}. Add aliases to names.py or fix the IDs, "
            "or pass --allow-unknown-blocks to build anyway"
        )
    for p in problems:
        print(f"  warning: {p}")
    pairs = set()
    for p, since in zip(pm["patch"], pm["riot_patches_since_prev"]):
        if isinstance(since, str):
            for q in since.split(";"):
                pairs |= {(p, name) for name in by_patch.get(q, ())}
    return pairs


def first_presence(t: GameTable, kept_idx: np.ndarray, n_kept: int) -> np.ndarray:
    """first[c, k]: earliest datetime at which champion c is picked or banned on kept patch k (NaT if never)."""
    k_of_game = kept_idx[t.present_game]
    ok = k_of_game >= 0
    df = pd.DataFrame({"c": t.present_champ[ok], "k": k_of_game[ok], "dt": t.dt[t.present_game[ok]]})
    first = df.groupby(["c", "k"])["dt"].min()
    out = np.full((t.n_champions, n_kept), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    out[first.index.get_level_values("c"), first.index.get_level_values("k")] = first.to_numpy()
    return out


def next_release_dates(pm: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Release date of the next Riot patch after each patch.

    A champion is a row key on P when it is released before the next Riot patch: new champions often
    go live a few days into their patch (Renata Glasc on 12.4, Qiyana on 9.13) and pro teams draft them
    at once. P's notes announce the champion, so the key set is still known at forecast time.
    """
    dated = pm.dropna(subset=["release_date"]).copy()
    dated = dated.iloc[sorted(range(len(dated)), key=lambda i: patch_key(dated["patch"].iloc[i]))]
    nxt = dated["release_date"].shift(-1)
    return {p: pd.Timestamp(d) for p, d in zip(dated["patch"], nxt) if pd.notna(d)}


def rate(num: np.ndarray, den: int) -> np.ndarray:
    return num / den if den else np.zeros(len(num))


def build_rows(t: GameTable, pm: pd.DataFrame, release: pd.Series, changed: set | None):
    kept = kept_patches(pm)
    index = {p: i for i, p in enumerate(kept)}
    kept_idx = np.array([index.get(p, -1) for p in t.patch])
    info = pm.set_index("patch")
    first = first_presence(t, kept_idx, len(kept))
    has_first = ~np.isnat(first)
    debut_all = np.where(has_first.any(axis=1), has_first.argmax(axis=1), -1)
    champ_code = {name: i for i, name in enumerate(t.champions)}
    recent = {name: release[name] >= DATA_START for name in t.champions}
    key_until = next_release_dates(pm)

    labels, inputs = [], []
    for i in range(1, len(kept)):
        p, pp = kept[i], kept[i - 1]
        cut = np.datetime64(info.at[p, "cut"])
        before = t.dt < cut
        on = [kept_idx == i - j for j in range(4)]  # P, P', P'', P'''
        c_p, c_pp = count(t, on[0]), count(t, on[1])
        pre: list[Counts | None] = [count(t, on[j] & before) if i - j >= 0 else None for j in (1, 2, 3)]
        pres_p, pres_pp = c_p.presence(), c_pp.presence()
        pre_pres = [c.presence() if c is not None and c.n_games else None for c in pre]
        p1 = pre[0]
        fearless_share = float(t.fearless[on[0]].mean())
        prev_fearless = p1.fearless_games / p1.n_games if p1.n_games else 0.0
        pick_rate, ban_rate = rate(p1.picks, p1.n_games), rate(p1.bans, p1.n_games)
        released = release[release < key_until[p]].index if p in key_until else release.index
        for name in released:
            c = champ_code[name]
            new = bool(recent[name] and debut_all[c] >= 0 and i in (debut_all[c], debut_all[c] + 1))
            labels.append(
                {
                    "patch": p,
                    "prev_patch": pp,
                    "champion": name,
                    "n_games": c_p.n_games,
                    "presence": float(pres_p[c]),
                    "prev_presence": float(pres_pp[c]),
                    "delta_presence": float(pres_p[c] - pres_pp[c]),
                    "fearless_share": fearless_share,
                    "changed": bool(changed is not None and (p, name) in changed),
                    "new_champion": new,
                }
            )
            since_debut = None
            if recent[name]:
                seen = np.flatnonzero(first[c, :i] < cut)
                since_debut = int(i - seen[0]) if len(seen) else 0
            inputs.append(
                {
                    "patch": p,
                    "prev_patch": pp,
                    "champion": name,
                    "prev_n_games": p1.n_games,
                    "prev_presence": float(pre_pres[0][c]) if pre_pres[0] is not None else 0.0,
                    "prev_pick_rate": float(pick_rate[c]),
                    "prev_ban_rate": float(ban_rate[c]),
                    "prev_win_rate": float(p1.wins[c] / p1.decided_picks[c]) if p1.decided_picks[c] else None,
                    "prev_main_role": ROLES[int(p1.roles[c].argmax())] if p1.picks[c] else None,
                    "prev_delta_1": (
                        float(pre_pres[0][c] - pre_pres[1][c])
                        if pre_pres[0] is not None and pre_pres[1] is not None
                        else None
                    ),
                    "prev_delta_2": (
                        float(pre_pres[1][c] - pre_pres[2][c])
                        if pre_pres[1] is not None and pre_pres[2] is not None
                        else None
                    ),
                    "prev_fearless_share": prev_fearless,
                    "patches_since_debut": since_debut,
                }
            )
    return pd.DataFrame(labels), pd.DataFrame(inputs)


def validate(df: pd.DataFrame, model: type) -> None:
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    TypeAdapter(list[model]).validate_python(records)


def run(args: argparse.Namespace) -> None:
    out_labels, out_test = args.data / "labels.parquet", args.private / "labels_test.parquet"
    out_inputs = args.data / "prepatch_stats.parquet"
    pm = load_patch_map(args.data / "patch_map.csv")
    release = champion_release_dates()
    games = pd.read_parquet(args.data / "games.parquet")
    t = build(games, sorted(release.index))
    changed = changed_pairs(pm, getattr(args, "allow_unknown_blocks", False))
    if changed is None:
        print(f"  warning: {CHANGES} not found, `changed` is False everywhere")
    labels, inputs = build_rows(t, pm, release, changed)
    validate(labels, ChampionPatch)
    validate(inputs, PrePatchStats)

    split = labels["patch"].map(pm.set_index("patch")["split"])
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    labels[split.isin(["train", "dev"])].to_parquet(out_labels, index=False)
    labels[split == "test"].to_parquet(out_test, index=False)
    inputs.to_parquet(out_inputs, index=False)

    print(
        f"{out_labels}: {int(split.isin(['train', 'dev']).sum())} rows; {out_test}: {int((split == 'test').sum())} rows"
    )
    print(f"{out_inputs}: {len(inputs)} rows over {inputs['patch'].nunique()} patches")
    print("  rows per split:", split.value_counts().to_dict())
    diff = (labels["prev_presence"] - inputs["prev_presence"]).abs()
    print(
        "  median |label prev_presence - input prev_presence| by split:",
        diff.groupby(split).median().round(4).to_dict(),
    )
    per_patch = inputs.groupby("patch").agg(prev_patch=("prev_patch", "first"), n=("prev_n_games", "first"))
    info = pm.set_index("patch")
    share = per_patch["n"] / per_patch["prev_patch"].map(info["n_pro_games"])
    print(
        "  share of P' games before the cut, median by split:",
        share.groupby(per_patch.index.map(info["split"])).median().round(2).to_dict(),
    )
