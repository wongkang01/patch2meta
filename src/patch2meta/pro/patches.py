"""Patch IDs, the patch map and the forecast time of each patch.

Patch IDs are LoL Wiki version titles without the leading "V": "9.22", "9.24b", "25.S1.1", "26.21".
A pro patch is "kept" when it has at least 100 pro games; P' (prev_patch) is the previous kept patch.
The forecast time cut(P) is 00:00 UTC on the day before P's release date, when the notes are out
(26.21: notes 20 Oct 2026, release 21 Oct). Model inputs only use games dated before cut(P).

Two options exist for open team questions (compared by pro/variants.py):
  --min-patch-games N   D3: the minimum pro games for a kept patch (default 100)
  --cut release-day     C3: cut(P) is 00:00 UTC on the release day instead of the day before

Writes data/processed/patch_map.csv (PatchMapRow).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter

from patch2meta.schema import PatchMapRow

RAW = Path("data/raw/pro")
PATCH_MAP = Path("data/processed/patch_map.csv")
PATCHES_JSONL = Path("data/raw/patches.jsonl")
MIN_GAMES = 100
CUT_RULES = {"day-before": 1, "release-day": 0}  # days before the release date
LAST_TEST_PATCH = "26.20"
PATCH_ID = re.compile(r"^\d+\.(S1\.)?\d+b?$")  # excludes non-patch pages such as "PBE"


def patch_key(p: str) -> tuple[int, int, int, bool]:
    """Sort key: '9.24b' < '10.1'; '25.S1.3' < '25.04'; 'b' patches right after their base patch."""
    hotfix = p.endswith("b")
    major, *rest = p.removesuffix("b").split(".")
    if rest[0] == "S1":
        return (int(major), 0, int(rest[1]), hotfix)
    return (int(major), 1, int(rest[0]), hotfix)


def split_of(p: str) -> str:
    major = patch_key(p)[0]
    if 9 <= major <= 14:
        return "train"
    if major == 25:
        return "dev"
    if major == 26 and patch_key(p) <= patch_key(LAST_TEST_PATCH):
        return "test"
    return "none"


def client_version(p: str) -> str:
    """Game client / CommunityDragon version: 26.21 -> 16.21, 25.S1.1 -> 15.1; 9.x to 14.x unchanged."""
    major, *rest = p.removesuffix("b").split(".")
    if int(major) >= 25:
        minor = rest[1] if rest[0] == "S1" else rest[0]
        return f"{int(major) - 10}.{int(minor)}"
    return p


def cut_of(release_date: pd.Timestamp, rule: str = "day-before") -> pd.Timestamp:
    return pd.Timestamp(release_date).normalize() - pd.Timedelta(days=CUT_RULES[rule])


def riot_urls() -> dict[str, str]:
    if not PATCHES_JSONL.exists():
        return {}
    rows = [json.loads(line) for line in PATCHES_JSONL.read_text().splitlines() if line.strip()]
    return {r["wiki_title"].removeprefix("V"): r.get("riot_url") for r in rows}


def build_patch_map(
    games: pd.DataFrame, wiki: pd.DataFrame, min_games: int = MIN_GAMES, cut_rule: str = "day-before"
) -> pd.DataFrame:
    counts = games.groupby("patch").size()
    wiki = wiki[wiki["patch"].astype(str).str.match(PATCH_ID)]
    wiki = wiki.drop_duplicates("patch").set_index("patch")
    patches = sorted(set(counts.index) | set(wiki.index), key=patch_key)
    urls = riot_urls()
    rows = []
    for p in patches:
        in_wiki = p in wiki.index
        redirect = in_wiki and "redirect_to" in wiki and isinstance(wiki.at[p, "redirect_to"], str)
        release = pd.Timestamp(wiki.at[p, "release_date"]) if in_wiki else pd.NaT
        rows.append(
            {
                "patch": p,
                "wiki_title": "V" + p,
                "client_version": client_version(p),
                "riot_url": urls.get(p),
                "release_date": None if pd.isna(release) else release.date(),
                "cut": None if pd.isna(release) else cut_of(release, cut_rule).to_pydatetime(),
                "n_pro_games": int(counts.get(p, 0)),
                "kept": bool(counts.get(p, 0) >= min_games),
                # A version page of its own with a release date: the patch has its own notes.
                "has_notes_page": bool(in_wiki and not redirect and not pd.isna(release)),
            }
        )
    pm = pd.DataFrame(rows)
    kept = pm.loc[pm["kept"], "patch"].tolist()
    prev = dict(zip(kept[1:], kept[:-1]))
    pm["prev_patch"] = pm["patch"].map(prev)
    notes = pm.loc[pm["has_notes_page"], "patch"].tolist()

    def since_prev(p: str) -> str | None:
        if p not in prev:
            return None
        lo, hi = patch_key(prev[p]), patch_key(p)
        return ";".join(q for q in notes if lo < patch_key(q) <= hi)

    pm["riot_patches_since_prev"] = pm["patch"].map(since_prev)
    pm["split"] = [split_of(p) if k else "none" for p, k in zip(pm["patch"], pm["kept"])]
    missing = pm[pm["kept"] & pm["release_date"].isna()]
    if len(missing):
        raise ValueError(f"kept patches without a release date: {missing['patch'].tolist()}")
    return pm.drop(columns="has_notes_page")


def load_patch_map(path: Path = PATCH_MAP) -> pd.DataFrame:
    pm = pd.read_csv(path, dtype={"patch": str, "prev_patch": str, "client_version": str})
    pm["release_date"] = pd.to_datetime(pm["release_date"])
    pm["cut"] = pd.to_datetime(pm["cut"])
    return pm


def kept_patches(pm: pd.DataFrame) -> list[str]:
    return sorted(pm.loc[pm["kept"], "patch"], key=patch_key)


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--min-patch-games",
        type=int,
        default=MIN_GAMES,
        help="D3: minimum pro games for a kept patch (default: %(default)s)",
    )
    ap.add_argument(
        "--cut",
        choices=list(CUT_RULES),
        default="day-before",
        help="C3: forecast time, 00:00 UTC on this day (default: %(default)s)",
    )


def run(args: argparse.Namespace) -> None:
    out = args.data / "patch_map.csv"
    games = pd.read_parquet(args.data / "games.parquet", columns=["patch"])
    wiki = pd.read_csv(RAW / "wiki_release_dates.csv", dtype={"patch": str})
    pm = build_patch_map(games, wiki, args.min_patch_games, args.cut)
    records = pm.astype(object).where(pm.notna(), None).to_dict("records")
    TypeAdapter(list[PatchMapRow]).validate_python(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.to_csv(out, index=False)
    kept = pm[pm["kept"]]
    print(
        f"{out}: {len(pm)} patches, {len(kept)} kept, "
        f"splits {kept['split'].value_counts().to_dict()}, games on kept patches {kept['n_pro_games'].sum()}"
    )
    order = pm.dropna(subset=["release_date"])
    if not order["release_date"].is_monotonic_increasing:
        print("  warning: release dates are not in patch order")
