"""Data contracts shared across the team.

Each model is one row of a file in data/processed/ (JSONL or Parquet). Producers validate rows
with these models before writing; consumers can rely on the fields below. Change a contract here,
in one PR, and tell the people who read that file.

Patch IDs are the LoL Wiki version titles without the leading "V": "9.22", "25.S1.1", "26.21".
"""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class PatchSource(BaseModel):
    """data/raw/patches.jsonl, written by notes.fetch (person 1)."""

    wiki_title: str  # "V9.22"
    release_date: date  # from the wiki infobox; notes usually go up the day before
    riot_url: str | None
    riot_source: Literal["live", "wayback", "missing"]
    wayback_timestamp: str | None  # YYYYMMDDhhmmss of the snapshot used


class ChangeType(StrEnum):
    NUMBER_TWEAK = "number_tweak"  # only values change
    MECHANIC = "mechanic"  # how something works changes; wins over number_tweak in mixed blocks
    REWORK = "rework"  # several abilities redesigned
    BUG_FIX = "bug_fix"
    ITEM = "item"
    RUNE = "rune"
    MAP_OBJECTIVE = "map_objective"


class Intent(StrEnum):
    """What Riot says it is doing, not what we think will happen in pro play."""

    BUFF = "buff"
    NERF = "nerf"
    SHIFT = "shift"  # power moved (e.g. late game to early game), not up or down
    FIX_QOL = "fix_qol"
    UNCLEAR = "unclear"  # no rationale and numbers point both ways


class ChangeLine(BaseModel):
    ability: str | None  # "Q - Piltover Peacemaker" (bare "Piltover Peacemaker" if Riot has no block);
    # None for base stats, items, runes and system lines
    attribute: str  # "base damage"
    old: str | None  # raw text, e.g. "50/90/130/170/210"
    new: str | None
    rel_change: float | None  # mean relative change across ranks, None if not numeric
    raw: str  # the whole line as plain text (wiki templates rendered, numbers kept)
    tag: str | None  # the wiki's label: "Bug Fix", "New Effect", "Removed", "Undocumented", ...
    gameplay: bool  # False for bug fixes and cosmetic lines (VFX, SFX, skins, icons, tooltips)


class ChangeBlock(BaseModel):
    """data/processed/changes.jsonl, written by person 1. One heading in the notes."""

    block_id: str  # f"{patch}:{kind}:{name}"
    patch: str
    kind: Literal["champion", "item", "rune", "system"]
    name: str  # "Caitlyn", "Infinity Edge"; system blocks use fixed names such as "Dragon", "Smite"
    rationale: str | None  # Riot's prose for this block; None when only the wiki was available
    lines: list[ChangeLine]
    gameplay: bool  # any line is gameplay; False for blocks of only bug fixes and cosmetics


class BlockLabel(BaseModel):
    """data/processed/block_labels.jsonl. LLM labels for every block, human labels for a sample."""

    block_id: str
    change_type: ChangeType
    intent: Intent
    labeller: str  # model id, or "human"
    confidence: float | None = None


class ChampionPatch(BaseModel):
    """data/processed/labels.parquet, written by person 2. One champion on one pro patch."""

    patch: str
    prev_patch: str  # P': the previous patch with >= 100 pro games
    champion: str
    n_games: int  # pro games on `patch`
    presence: float  # (picks + bans) / games on `patch`
    prev_presence: float
    delta_presence: float  # presence - prev_presence: the main target
    fearless_share: float  # share of games on `patch` played in Fearless draft
    changed: bool  # the notes for `patch` have a gameplay block (ChangeBlock.gameplay) for this champion


class Prediction(BaseModel):
    """data/predictions/<model>.csv, written by every model. Scored by the shared eval code."""

    model: str
    patch: str
    champion: str
    pred_delta_presence: float
