"""Riot's rationale prose from an official patch notes page, and matching it to wiki blocks.

Each change on a Riot page is a div.patch-change-block with the name in h3.change-title, a one-line
summary in .summary, the designer rationale in blockquote.context, and ability headings in
h4.change-detail-title ("Q - Piltover Peacemaker"). The same layout is used from 9.1 to 26.x.
Blocks under a "Mid-Patch Updates" heading were added after release and are dropped, as are blocks
for game modes, skins and the store.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from patch2meta.schema import ChangeBlock

_MID_PATCH = re.compile(r"mid-?\s*patch", re.IGNORECASE)
_SKIP_SECTIONS = re.compile(
    r"aram|arena|urf|spellbook|nexus blitz|one[- ]for[- ]all|swiftplay|doom bots|brawl|swarm|"
    r"clash|tft|teamfight tactics|skins?\b|chromas?|mythic shop|ranked|patch highlights",
    re.IGNORECASE,
)
# "[NEW] Doran's Bow", "[REWORKED] Statikk Shiv", "new Zeri", "updated Janna"
_TAG = re.compile(r"^(?:\[[A-Z ]+\]|(?:new|updated|removed|reworked)\b)\s*")


@dataclass
class RiotBlock:
    name: str
    section: str  # the h2 it sits under, e.g. "Champions"
    rationale: str | None
    ability_titles: list[str] = field(default_factory=list)  # ["Passive - Headshot", "Q - ..."]
    from_heading: bool = False  # an h4 entry from a big patch's list rather than a titled block


def _text(el: Tag) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).replace(" ,", ",").replace(" .", ".")


def _is_change_block(t: Tag) -> bool:
    return t.name == "div" and "patch-change-block" in (t.get("class") or [])


def _prose(el: Tag) -> list[str]:
    return [p for p in (_text(x) for x in (el.find_all("p") or [el])) if p]


def riot_blocks(html: str) -> list[RiotBlock]:
    """Patch-day change blocks on a Riot patch notes page, in page order.

    Most changes sit in a titled block (h3.change-title) with their own rationale. Big patches
    (12.10, 14.1, 25.S1.1) instead list many changes as h4 headings, in an untitled block or loose
    in the section; each h4 becomes an entry whose rationale is the blockquote under it or, if it
    has none, the blockquotes that open its h2 section.
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[RiotBlock] = []
    section = ""
    section_prose: list[str] = []
    entries: list[tuple[RiotBlock, list[str], list[str]]] = []  # (entry, its own prose, section prose)
    own: list[str] | None = None  # prose of the current h4 entry
    for el in soup.find_all(lambda t: t.name in ("h2", "h4", "blockquote") or _is_change_block(t)):
        if el.name == "h2":
            section, section_prose, own = _text(el), [], None
            continue
        if _MID_PATCH.search(section) or _SKIP_SECTIONS.search(section):
            continue
        parent = el if _is_change_block(el) else el.find_parent(_is_change_block)
        title = parent.select_one("h3.change-title") if parent else None
        if title is not None:  # a titled block; its h4s are ability headings
            if el is parent:
                prose = [_text(s) for s in el.select(".summary")]
                prose += [p for q in el.select("blockquote") for p in _prose(q)]
                out.append(
                    RiotBlock(
                        name=_TAG.sub("", _text(title)),
                        section=section,
                        rationale="\n".join(p for p in prose if p) or None,
                        ability_titles=[_text(h) for h in el.select("h4.change-detail-title")],
                    )
                )
            own = None
        elif el.name == "div":  # an untitled block starts
            own = None
        elif el.name == "h4":
            own = []
            # "Trailblazer (formerly known as ...)", "ARCHANGEL'S STAFF / SERAPH'S EMBRACE"
            for name in re.sub(r"\(.*?\)", "", _TAG.sub("", _text(el))).split("/"):
                entry = RiotBlock(name=name.strip(), section=section, rationale=None, from_heading=True)
                entries.append((entry, own, section_prose))
                out.append(entry)
        else:  # blockquote
            (section_prose if own is None else own).extend(_prose(el))
    for entry, mine, shared in entries:
        entry.rationale = "\n".join(mine or shared) or None
    deduped: list[RiotBlock] = []
    for b in out:  # some pages (13.17) repeat every block
        if not any((o.name, o.section) == (b.name, b.section) for o in deduped):
            deduped.append(b)
    return deduped


_ALIASES = {"nunuwillump": "nunu"}
_KIND_SECTION = {"champion": "champion", "item": "item", "rune": "rune"}


def key(name: str) -> str:
    """Name key that ignores case, accents, spaces and punctuation ("Nunu & Willump" -> "nunu")."""
    k = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    k = re.sub(r"[^a-z0-9]", "", k)
    return _ALIASES.get(k, k)


def _find(block: ChangeBlock, riot: list[RiotBlock]) -> RiotBlock | None:
    k = key(block.name)
    if not k:
        return None
    exact = [r for r in riot if key(r.name) == k]
    loose = [
        r
        for r in riot
        if not r.from_heading  # h4 entries are often topics ("Rift Herald Fight"), so match them exactly
        and len(k) >= 4
        and len(key(r.name)) >= 4
        and (key(r.name).startswith(k) or k.startswith(key(r.name)))
    ]
    want = _KIND_SECTION.get(block.kind)
    for cands in (exact, loose):
        cands.sort(key=lambda r: r.from_heading)  # a titled block beats an h4 entry of the same name
        if want:
            same = [r for r in cands if want in r.section.lower()]
            if same:
                return same[0]
        if cands:
            return cands[0]
    return None


def _ability_label(ability: str, titles: list[str]) -> str:
    """ "Headshot" -> "Passive - Headshot" when Riot's block has that heading."""
    k = key(ability)
    for t in titles:
        _, sep, name = t.partition(" - ")
        if sep and k and (key(name) == k or k in [key(p) for p in name.split("/")]):
            return t
    return ability


def attach(blocks: list[ChangeBlock], riot: list[RiotBlock]) -> list[RiotBlock]:
    """Fill block.rationale and Riot-style ability labels in place; return titled Riot blocks left
    unmatched (h4 entries are left out: many are not changes, e.g. 14.1's "TOP SIDE")."""
    used: set[int] = set()
    for b in blocks:
        r = _find(b, riot)
        if r is None:
            continue
        used.add(id(r))
        b.rationale = r.rationale
        for line in b.lines:
            if line.ability:
                line.ability = _ability_label(line.ability, r.ability_titles)
    return [r for r in riot if id(r) not in used and not r.from_heading]
