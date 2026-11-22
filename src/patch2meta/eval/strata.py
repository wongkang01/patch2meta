"""Change-type and intent strata for the H1 breakdown.

A champion-patch row (P, c) can have several champion blocks: one per Riot patch in (P', P].
Rule (plan section 2):
  change_type  the strongest type among its blocks: rework > mechanic > number_tweak > bug_fix
  intent       the shared intent if all its blocks agree, else "mixed"
Item, rune and map/objective changes belong to no champion row, so they are a patch-level stratum:
a patch "has" a system type when any Riot patch in (P', P] has a block of that type.

Writes data/processed/row_strata.parquet (changed rows) and data/processed/patch_strata.csv.
"""

import json
from pathlib import Path

import pandas as pd

from patch2meta.pro.names import canonical, canonical_names

CHANGES = Path("data/processed/changes.jsonl")
BLOCK_LABELS = Path("data/processed/block_labels.jsonl")
ROW_STRATA = Path("data/processed/row_strata.parquet")
PATCH_STRATA = Path("data/processed/patch_strata.csv")
TYPE_PRIORITY = ["rework", "mechanic", "number_tweak", "bug_fix"]
SYSTEM_TYPES = ["item", "rune", "map_objective"]
KIND_TO_TYPE = {"item": "item", "rune": "rune", "system": "map_objective"}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])


def pick_labeller(labels: pd.DataFrame, labeller: str | None) -> pd.DataFrame:
    """Rows of one labeller: the one given, else the model with the most labels (never 'human')."""
    if labeller is None:
        models = labels.loc[labels["labeller"] != "human", "labeller"]
        if models.empty:
            raise ValueError("block_labels.jsonl has no model labels")
        labeller = models.value_counts().index[0]
    return labels[labels["labeller"] == labeller].drop_duplicates("block_id")


def build_strata(
    blocks: pd.DataFrame, block_labels: pd.DataFrame, pm: pd.DataFrame, known: frozenset[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = blocks[["block_id", "patch", "kind", "name"]].merge(
        block_labels[["block_id", "change_type", "intent"]], on="block_id", how="left"
    )
    since = {
        p: s.split(";") for p, s in zip(pm["patch"], pm["riot_patches_since_prev"]) if isinstance(s, str)
    }
    known = canonical_names() if known is None else known
    champ = b[b["kind"] == "champion"].copy()
    champ["champion"] = [canonical(n, known) for n in champ["name"]]
    rows, patch_rows = [], []
    for p, riot in since.items():
        cb = champ[champ["patch"].isin(riot)]
        for name, d in cb.groupby("champion"):
            types = set(d["change_type"].dropna())
            intents = set(d["intent"].dropna())
            rows.append(
                {
                    "patch": p,
                    "champion": name,
                    "change_type": next((t for t in TYPE_PRIORITY if t in types), None),
                    "intent": intents.pop() if len(intents) == 1 else ("mixed" if intents else None),
                    "n_blocks": len(d),
                }
            )
        sb = b[(b["kind"] != "champion") & b["patch"].isin(riot)]
        types = set(sb["change_type"].dropna()) | {KIND_TO_TYPE[k] for k in sb["kind"]}
        patch_rows.append({"patch": p, **{f"has_{t}": t in types for t in SYSTEM_TYPES}})
    return pd.DataFrame(rows), pd.DataFrame(patch_rows)


def load_strata() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not (ROW_STRATA.exists() and PATCH_STRATA.exists()):
        return None, None
    return pd.read_parquet(ROW_STRATA), pd.read_csv(PATCH_STRATA, dtype={"patch": str})


def write_strata(pm: pd.DataFrame, labeller: str | None = None) -> None:
    for path in (CHANGES, BLOCK_LABELS):
        if not path.exists():
            raise FileNotFoundError(f"{path} is needed for strata")
    rows, patches = build_strata(read_jsonl(CHANGES), pick_labeller(read_jsonl(BLOCK_LABELS), labeller), pm)
    rows.to_parquet(ROW_STRATA, index=False)
    patches.to_csv(PATCH_STRATA, index=False)
    print(f"{ROW_STRATA}: {len(rows)} rows; {PATCH_STRATA}: {len(patches)} patches")
