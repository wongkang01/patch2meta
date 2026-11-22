"""An optional hand-off bundle without 2026 outcomes: data/share/.

  uv run python -m patch2meta.pro share                  # train and dev rows only
  uv run python -m patch2meta.pro share --include-test   # adds the 2026 and 26.22 inputs

The team shares data/processed/ through git (see the README); this bundle is for giving someone the
files without 2026 outcomes. It holds labels.parquet, prepatch_stats.parquet, relation_graph.parquet,
patch_map.csv, SHA256SUMS and HANDOFF.md.

Inputs for a 2026 patch contain the outcomes of earlier 2026 patches: prev_delta_1 of a patch is a
cut copy of the previous kept patch's Δpresence, and graph rows count picks from earlier patches.
Without --include-test the test and 26.22 rows stay out. games.parquet, draft.parquet, data/raw/,
data/private/ and data/predictions/ are never copied, because they hold every 2026 game, from which
the test labels can be recomputed.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from patch2meta.pro import steps
from patch2meta.pro.patches import load_patch_map

OUT = Path("data/share")
MANIFEST = Path("data/raw/pro/manifest.json")
SPLIT_FILES = ["labels.parquet", "prepatch_stats.parquet", "relation_graph.parquet"]
OPEN_SPLITS = {"train", "dev"}
CLOSED_SPLITS = {"test", "none"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_share(data: Path, out: Path, include_test: bool = False) -> dict[str, int]:
    """Write the shareable files to `out` and return their row counts.

    The bundle's labels are always train and dev only, even with --include-test.
    """
    split = load_patch_map(data / "patch_map.csv").set_index("patch")["split"]
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name in SPLIT_FILES:
        keep = OPEN_SPLITS | (CLOSED_SPLITS if include_test and name != "labels.parquet" else set())
        df = pd.read_parquet(data / name)
        df = df[df["patch"].map(split).isin(keep)]
        df.to_parquet(out / name, index=False)
        rows[name] = len(df)
    (out / "patch_map.csv").write_bytes((data / "patch_map.csv").read_bytes())
    rows["patch_map.csv"] = len(split)
    return rows


def handoff_text(rows: dict[str, int], include_test: bool) -> str:
    git = ["git", "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(git, check=False, capture_output=True, text=True).stdout  # empty outside git
    pulled = (
        sorted(v["pulled_utc"] for v in json.loads(MANIFEST.read_text()).values())
        if MANIFEST.exists()
        else []
    )
    scope = "all splits (released at the freeze)" if include_test else "train and dev rows only"
    lines = [
        "# Pro data hand-off",
        "",
        f"Built {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from commit {commit.strip() or 'unknown'}, {scope}.",
        f"Raw data pulled {pulled[0]} to {pulled[-1]}." if pulled else "Raw data pull time unknown.",
        "Copy these files into data/processed/ and check them with `shasum -a 256 -c SHA256SUMS`.",
        "",
        "| file | rows |",
        "|---|---|",
        *[f"| {name} | {n:,} |" for name, n in rows.items()],
        "",
    ]
    if not include_test:
        lines += [
            "Test and 26.22 inputs arrive at the freeze, because inputs for a 2026 patch contain the",
            "outcomes of earlier 2026 patches. Do not ask for games.parquet or draft.parquet, because they",
            "hold every 2026 game.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="patch2meta.pro share")
    ap.add_argument("--data", type=Path, default=steps.DATA, help="build to share (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=OUT, help="hand-off folder (default: %(default)s)")
    ap.add_argument(
        "--include-test", action="store_true", help="after the freeze only: add the test and 26.22 inputs"
    )
    args = ap.parse_args(argv)
    rows = build_share(args.data, args.out, args.include_test)
    names = [*SPLIT_FILES, "patch_map.csv"]
    (args.out / "SHA256SUMS").write_text("".join(f"{sha256(args.out / n)}  {n}\n" for n in names))
    (args.out / "HANDOFF.md").write_text(handoff_text(rows, args.include_test))
    print(f"{args.out}: {rows}")
