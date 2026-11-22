"""Canonical champion names.

The canonical name is the current Leaguepedia display name (Cargo `Champions.Name`), for example
"Nunu & Willump" and "Wukong". gptilt and Leaguepedia game tables already use these names (checked on
17 Nov 2026: every name in 2019+ gptilt games is canonical or a placeholder). ALIASES maps any other
spelling, such as patch-note block names or CommunityDragon keys, to the canonical name.
"""

from functools import cache
from pathlib import Path

import pandas as pd

CHAMPIONS = Path("data/raw/pro/cargo_champions.parquet")

# Placeholders in pick and ban lists: a skipped or lost ban, or a game without draft data.
NOT_A_CHAMPION = frozenset({"", "None", "Missing Data"})

# Other spellings -> canonical name. Add entries by hand after inspecting a new source.
ALIASES: dict[str, str] = {
    "MonkeyKing": "Wukong",  # CommunityDragon / Data Dragon key
    "Nunu": "Nunu & Willump",
    "Renata": "Renata Glasc",
    "Kled & Skaarl": "Kled",  # 12.10 notes: a block for Kled's mounted form
    "Mega Gnar": "Gnar",  # 12.10 notes: a block for Gnar's transformed form
}


@cache
def canonical_names() -> frozenset[str]:
    return frozenset(pd.read_parquet(CHAMPIONS, columns=["Name"])["Name"])


def canonical(name: str, known: frozenset[str] | None = None) -> str:
    """The canonical name for `name`. Raises KeyError on an unknown name so mismatches surface."""
    names = canonical_names() if known is None else known
    if name in names:
        return name
    if name in ALIASES:
        return ALIASES[name]
    raise KeyError(f"unknown champion name {name!r}; add it to ALIASES or NOT_A_CHAMPION")
