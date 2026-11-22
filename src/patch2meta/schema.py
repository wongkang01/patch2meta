"""Data contracts shared across the team.

Each model is one row of a file in data/processed/ (JSONL or Parquet). Producers validate rows
with these models before writing; consumers can rely on the fields below. Change a contract here,
in one PR, and tell the people who read that file.

Patch IDs are the LoL Wiki version titles without the leading "V": "9.22", "25.S1.1", "26.21".
"""

from datetime import date, datetime
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
    rel_change: float | None  # mean of (new - old) / |old| across ranks, so the sign is the direction
    # the number moved; None if not numeric, or if old and new are not the same kind of value (% vs flat,
    # bonus vs total AD, level vs per-minute scaling, prose)
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
    """One champion on one pro patch: the target, written by person 2.

    data/processed/labels.parquet holds train and dev rows. The 2026 test rows are held back in
    data/private/labels_test.parquet until the single test run after the model freeze.

    Rows exist for every champion released before the next Riot patch after P (P's notes announce
    new champions), drafted or not, so the row set reveals nothing about games on P. A champion drafted on neither P nor P' has presence 0
    and delta_presence 0.
    """

    patch: str
    prev_patch: str  # P': the previous patch with >= 100 pro games
    champion: str
    n_games: int  # pro games on `patch`
    presence: float  # (picks + bans) / games on `patch`
    prev_presence: float  # (picks + bans) / all games on P'; part of the target, not a model input
    delta_presence: float  # presence - prev_presence: the main target
    fearless_share: float  # share of games on `patch` played in Fearless draft
    # changed: the notes of any Riot patch after P' up to P have a gameplay block (ChangeBlock.gameplay)
    # for this champion; blocks of only bug fixes or cosmetic lines do not count.
    changed: bool
    # Newly drafted champion: `patch` is the first pro patch on which it is picked or banned, or the
    # next pro patch. Its delta_presence then reflects tournament legality, not the notes, so the eval
    # code reports these rows separately and leaves them out of headline scores. Not a model input.
    new_champion: bool = False


class PrePatchStats(BaseModel):
    """data/processed/prepatch_stats.parquet, written by person 2. Model inputs for every split.

    One row per (patch, champion) key of the label files, plus rows for kept patches outside the study
    splits (26.22, for the demo). Rows for 2026 and later patches hold the outcomes of earlier 2026
    patches, so use them only for the frozen models' predictions. Every value uses only games dated before
    the forecast time cut(P) = 00:00 UTC on the day before P's release date, when the notes are out.
    P' games played after that are left out, so prev_presence here can differ from
    ChampionPatch.prev_presence. Never rebuild the target from this file.
    """

    patch: str  # P, the patch being forecast
    prev_patch: str  # P'
    champion: str
    prev_n_games: int  # pro games on P' dated before cut(P)
    prev_presence: float  # (picks + bans) / those games; 0 if not drafted in them
    prev_pick_rate: float  # picks / those games
    prev_ban_rate: float  # bans / those games
    prev_win_rate: float | None  # wins / picks in those games; None if never picked
    prev_main_role: str | None  # most-played role in those games (top, jungle, mid, bot, support)
    prev_delta_1: float | None  # presence on P' minus presence on P'' (games before cut(P)); None if no P''
    prev_delta_2: float | None  # presence on P'' minus presence on P'''; None if no P'''
    prev_fearless_share: float  # share of those P' games played in Fearless draft
    # Pro patches from the champion's first pick or ban up to P' (games before cut(P)): 0 if released
    # but not drafted yet, 1 on the pro patch after its first one, and so on. None for champions
    # released before 2019, whose first pro patch is before the data starts.
    patches_since_debut: int | None


class PatchMapRow(BaseModel):
    """data/processed/patch_map.csv, written by person 2. One Riot patch from 9.1 onward."""

    patch: str  # "26.20"
    wiki_title: str  # "V26.20"
    client_version: str  # "16.20" (CommunityDragon and the game client)
    riot_url: str | None  # from data/raw/patches.jsonl once person 1 has run the fetcher
    release_date: date | None  # LoL Wiki infobox; None for pages without one
    cut: datetime | None  # forecast time: 00:00 UTC on the day before release_date
    n_pro_games: int
    kept: bool  # at least 100 pro games
    prev_patch: str | None  # P' for kept patches
    riot_patches_since_prev: str | None  # Riot patches in (P', P], joined with ";"
    split: Literal["train", "dev", "test", "none"]


class RelationEdge(BaseModel):
    """data/processed/relation_graph.parquet, written by person 2. One directed edge for one patch.

    Built only from games on the 6 kept patches before `patch` that are dated before cut(patch).
    """

    patch: str  # the forecast patch P
    champion: str
    neighbour: str
    relation: Literal["partner", "counter", "substitute"]
    # partner: games where both are picked on the same team; counter: games where they are picked in
    # the same role on opposite teams (lane opponents); substitute: role overlap, sum over roles of
    # share_c(role) * share_n(role).
    weight: float
    share: float  # partner/counter: weight / picks of `champion` in the window; substitute: the overlap
    lift: float | None  # weight * team_games / (picks of champion * picks of neighbour); None for substitute
    window_first: str
    window_last: str
    n_games_window: int


class Prediction(BaseModel):
    """data/predictions/<model>.csv, written by every model. Scored by the shared eval code."""

    model: str
    patch: str
    champion: str
    pred_delta_presence: float
