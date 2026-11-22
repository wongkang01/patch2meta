"""Turn LoL Wiki patch-page wikitext into ChangeBlocks.

parse_patch(wikitext, patch, subpages) keeps only the "== League of Legends V<patch> ==" section
(so wiki hotfix sections, which match Riot's mid-patch updates, are dropped), keeps Summoner's Rift
sections (champions, items, runes, summoner spells, map and objectives) and drops game modes and the
client. Each ";" heading is one block; each bullet under it is one ChangeLine.
"""

import ast
import itertools
import operator
import re
from dataclasses import dataclass

from patch2meta.schema import ChangeBlock, ChangeLine

# ---------------------------------------------------------------- templates


@dataclass
class Template:
    name: str
    args: list[str]  # positional
    kwargs: dict[str, str]


Token = str | Template


def _scan(text: str, start: int) -> int:
    """Index just past the "}}" closing the template that opens at text[start:start+2]."""
    depth, i = 0, start
    while i < len(text):
        if text.startswith("{{", i):
            depth, i = depth + 1, i + 2
        elif text.startswith("}}", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(text)  # unbalanced: swallow the rest


def _split_args(body: str) -> list[str]:
    """Split a template body on "|" that is not inside a nested template or link."""
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(body):
        two = body[i : i + 2]
        if two in ("{{", "[["):
            depth += 1
            cur.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            cur.append(two)
            i += 2
        elif body[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(body[i])
            i += 1
    parts.append("".join(cur))
    return parts


_KWARG = re.compile(r"^\s*([A-Za-z_][\w ]*?)\s*=(.*)$", re.DOTALL)


def tokenize(text: str) -> list[Token]:
    """Split text into plain strings and top-level templates."""
    out: list[Token] = []
    i = 0
    while (j := text.find("{{", i)) != -1:
        if j > i:
            out.append(text[i:j])
        end = _scan(text, j)
        name, *rest = _split_args(text[j + 2 : end - 2])
        args, kwargs = [], {}
        for a in rest:
            m = _KWARG.match(a)
            if m and "{{" not in m.group(1):
                kwargs[m.group(1).strip()] = m.group(2)
            else:
                args.append(a)
        out.append(Template(name.strip(), args, kwargs))
        i = end
    if i < len(text):
        out.append(text[i:])
    return out


# Templates that render to a name: the display text is the 2nd arg if given, else the 1st.
_NAME_TEMPLATES = {
    "ci", "ii", "ri", "si", "ui", "bi", "iis", "cis", "uis", "sis", "ccs", "ris", "cci", "ccib",
    "fi", "nie", "tip", "rutngt", "univ",
}  # fmt: skip
# Ability templates: {{ai|ability|champion|display}}.
_ABILITY_TEMPLATES = {"ai", "cai", "ais", "cais"}
_CURRENCY = {"g": "gold", "ME": "ME", "BE": "BE", "RP": "RP", "OE": "OE", "BXP": "BXP"}
_SYMBOLS = {
    "times": "×", "plus": "+", "minus": "-", "divided by": "/", "equals": "=", "degree": "°",
    "PlusMinus": "±", "RMB": "right-click",
}  # fmt: skip
_DROP = {"Note", "note", "References", "anchor", "clear", "Release history", "bug"}


def _drop_count(spec: str) -> str:
    """ "150 to 550 3" or "0 to 80 for 11" -> "150 to 550": the rank count is not display text."""
    m = _RANGE.match(spec.strip())
    return f"{m.group(1)} to {m.group(2)}" if m else spec


def _is_formula(t: Template) -> bool:
    """A pp template whose values are one formula in x ("3450+380*(x-1+6)", "3.5*(x-1) for 7")."""
    return bool(t.args) and "x" in t.args[0] and ";" not in t.args[0] and "then" not in t.args[0]


def _formula_values(t: Template) -> list[float]:
    """Values of a formula pp, where x is the 1-based position in the level range given by
    " for n", else by the second argument ("6 to 18 by 1"), else levels 1 to 18."""
    spec, _, count = t.args[0].partition(" for ")
    if count.strip().isdigit():
        n = int(count)
    else:
        n = len(_expand_range(t.args[1], 18)) if len(t.args) > 1 else 18
    return [v for i in range(1, n + 1) if (v := _eval(spec.replace("x", f"({i})"))) is not None]


def _render_values(t: Template) -> str:
    """Display text of an ap, gs or pp template, without its unit."""
    a = t.args
    if t.name == "ap":
        return " / ".join(render(x) for x in a) if len(a) > 1 else render(_drop_count(a[0]))
    if (t.name == "gs" or _is_formula(t)) and (v := _template_values(t)):  # a formula over levels
        return " to ".join(f"{round(x, 2):g}" for x in (v[0], v[-1]) if len(v) > 1 or x is v[0])
    if t.name == "pp":
        return render("; ".join(_drop_count(p) for p in a[0].split(";")).replace("; ", " / "))
    return ""


def _render_template(t: Template) -> str:
    a = t.args
    name = t.name
    if name in _DROP:
        return ""
    if name.startswith("#expr:"):
        v = _expr(t)
        return "" if v is None else f"{v:g}"
    if name in _SYMBOLS:
        return _SYMBOLS[name]
    if t.kwargs.get("icononly", "").strip() == "true":
        return ""
    if not a:
        return ""
    if name in _ABILITY_TEMPLATES:
        return render(a[2] if len(a) > 2 else a[0])
    if name in _NAME_TEMPLATES:
        return render(a[1] if len(a) > 1 else a[0])
    if name == "csl":
        return render(" ".join(reversed(a[:2])))
    if name in _CURRENCY:
        return f"{render(a[0])} {_CURRENCY[name]}"
    if name == "rd":
        return f"{render(a[0])} (melee) / {render(a[1])} (ranged)" if len(a) > 1 else render(a[0])
    if name in ("ap", "gs", "pp"):
        return _render_values(t) + t.kwargs.get("key", "").strip()  # key is the unit, e.g. "%"
    if name in ("sti", "stil"):
        return render(a[-1])
    return render(a[0])  # as, sbc, fd, tt, adaptive, recurring, and anything unknown


_LINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_EXT_LINK = re.compile(r"\[https?://\S+\s+([^\]]*)\]")
_TAGS = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<[^>]+>", re.DOTALL)


def _link(m: re.Match) -> str:
    target, label = m.group(1), m.group(2)
    if re.match(r"^(File|Image|Category):|^[a-z]{2}(-[a-z]+)?:", target):
        return ""
    return label if label is not None else target.split("#")[0]


def render(text: str) -> str:
    """Wikitext to plain text. Templates become their display text, with numbers kept."""
    parts = [p if isinstance(p, str) else _render_template(p) for p in tokenize(text)]
    out = "".join(parts)
    out = _TAGS.sub(" ", out)
    while _LINK.search(out):
        out = _LINK.sub(_link, out)
    out = _EXT_LINK.sub(r"\1", out)
    out = out.replace("'''", "").replace("''", "")
    out = re.sub(r"\(\s*\)", "", out)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------- numbers

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(expr: str) -> float | None:
    """Evaluate simple arithmetic such as "0.5*100" or "(1/3)*90"; None for anything else."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError

    try:
        return ev(ast.parse(expr.replace("%", "").strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None


def _linspace(a: float, b: float, n: int) -> list[float]:
    return [a] if n <= 1 else [a + (b - a) * i / (n - 1) for i in range(n)]


# "a to b", "a to b 3", "a to b for 3" (3 ranks), "a to b by 2" (step 2)
_RANGE = re.compile(r"^(.+?)\s+to\s+(.+?)(?:\s+(?:for\s+)?(\d+)|\s+by\s+([\d.]+))?$")


def _expand_range(spec: str, default_n: int) -> list[float]:
    m = _RANGE.match(spec.strip())
    if not m:
        v = _eval(spec)
        return [] if v is None else [v]
    a, b = _eval(m.group(1)), _eval(m.group(2))
    if a is None or b is None:
        return []
    if m.group(4) and float(m.group(4)) > 0:
        return _linspace(a, b, int(abs(b - a) / float(m.group(4))) + 1)
    return _linspace(a, b, int(m.group(3)) if m.group(3) else default_n)


def _expr(t: Template) -> float | None:
    """Value of a {{#expr:(0.3-0.1)*100 round 2}} parser function."""
    v = _eval(re.sub(r"\s+round\s+\d+\s*$", "", t.name.removeprefix("#expr:")))
    return None if v is None else round(v, 4)


def _template_values(t: Template) -> list[float] | None:
    """Per-rank (ap) or per-level (pp) values, or None if t is not a value template."""
    if t.name.startswith("#expr:"):
        return [] if (v := _expr(t)) is None else [v]
    if t.name == "ap":
        if len(t.args) > 1:
            return [v for x in t.args if (v := _eval(x)) is not None]
        return _expand_range(t.args[0], 5) if t.args else []
    if t.name == "gs":  # {{gs|base|growth|from level|to level}}: the champion stat growth formula
        got = [v for x in t.args[:4] if (v := _eval(x)) is not None]
        if len(got) < min(len(t.args), 4):
            return []
        base, growth, lo, hi = got + [0.0, 0.0, 1.0, 18.0][len(got) :]
        return [base + growth * (n - 1) * (0.7025 + 0.0175 * (n - 1)) for n in range(int(lo), int(hi) + 1)]
    if t.name == "pp" and t.args:
        if _is_formula(t):
            return _formula_values(t)
        pieces = t.args[0].split(";")
        if len(pieces) == 1:
            return _expand_range(pieces[0], 18)
        # Explicit levels that end in a formula ("42; 114; ...; 590; then + 50*x"): keep the explicit
        # ones if there are at least three; fewer would compare one or two levels out of 18.
        lead = list(itertools.takewhile(lambda p: "x" not in p, pieces))
        return (
            [v for p in lead for v in _expand_range(p, 2)]
            if len(lead) >= 3 or len(lead) == len(pieces)
            else []
        )
    if t.name == "rd" and len(t.args) >= 2:
        return [v for x in t.args[:2] if (v := _eval(render(x))) is not None]
    return None


# The first number, or run of numbers joined by "/" ("25 / 35%"), in plain text. Later numbers are
# usually context such as "per 100 bonus AD" or "(+ 0 / 45 / 80)".
_NUM_RUN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:%?\s*/\s*-?\d+(?:\.\d+)?)*")


def _first_values(tokens: list[Token]) -> list[float] | None:
    for t in tokens:
        if isinstance(t, str):
            if m := _NUM_RUN.search(render(t)):
                return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(0))]
            continue
        vals = _template_values(t)
        if vals is None:
            vals = next((v for a in t.args if (v := _first_values(tokenize(a))) is not None), None)
        if vals is not None:
            return vals
    return None


def values(text: str) -> list[float]:
    """Numbers on one side of a change: the first ap/pp/rd template or number run, in text order."""
    return _first_values(tokenize(text)) or []


def rel_change(old: list[float], new: list[float]) -> float | None:
    """Mean of (new - old) / |old| across ranks, so the sign is the direction the number moved even
    when it is negative; a single value is compared with every rank."""
    if not old or not new:
        return None
    if len(old) != len(new):
        if len(old) == 1:
            old = old * len(new)
        elif len(new) == 1:
            new = new * len(old)
        else:  # different rank counts: compare the means
            old, new = [sum(old) / len(old)], [sum(new) / len(new)]
    pairs = [(o, n) for o, n in zip(old, new) if o != 0]
    if not pairs:
        return None
    return sum((n - o) / abs(o) for o, n in pairs) / len(pairs)


# The first value on one side of a change: a number or run of numbers ("25 / 35%", "10 to 30%",
# "40*0.8%"), then its unit, if any: a "%" or the next word.
_VALUE = re.compile(
    r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*%?\s*(?:/|to|\*)\s*-?(?:\d+(?:\.\d+)?|\.\d+))*(?P<pct>\s*%)?\s*(?P<word>[a-z]+)?",
    re.IGNORECASE,
)
_UNITS = {"second": "s", "seconds": "s", "minute": "min", "minutes": "min", "unit": "units", "units": "units"}
# What a value is of ("% bonus AD", "bonus magic resistance", "of your maximum health"), from the words
# after it: the first stat word, and the first qualifier
_STATS = re.compile(
    r"\b(AD|AP|health|mana|armor|resistance|damage|attack speed|movement speed|haste|tenacity|"
    r"critical strike chance|lethality|penetration|gold|experience)\b",
    re.IGNORECASE,
)
_QUALIFIERS = re.compile(r"\b(bonus|total|base|maximum|current|missing)\b", re.IGNORECASE)


@dataclass
class _Shape:
    unit: str  # "%", "s", "min", "units" or "" for none
    stat: str | None
    qualifier: str | None
    per_minute: bool  # "250 (+ 250 per minute from match start)"


def _shape(side: str) -> _Shape | None:
    """What kind of value one side of a change holds, from its plain text; None when it is prose,
    i.e. three or more words come before the first number."""
    side = re.sub(r"\s*\((?:melee|ranged)\)", "", side)  # "8 (melee) / 6 (ranged)%"
    m = _VALUE.search(side)
    if m is None or len(re.findall(r"[A-Za-z]+", side[: m.start()])) >= 3:
        return None
    unit = "%" if m.group("pct") or "%" in m.group(0) else _UNITS.get((m.group("word") or "").lower(), "")
    rest = side[m.start("word") if m.group("word") else m.end() :]
    after = " ".join(rest.split("(")[0].split()[:6])  # "(+ 30% AP)" is a second value
    stat, qual = _STATS.search(after), _QUALIFIERS.search(after)
    return _Shape(
        unit,
        stat and stat.group(1).lower(),
        qual and qual.group(1).lower(),
        "per minute" in side,
    )


def comparable(old: str, new: str) -> bool:
    """Whether the old and new plain text hold the same kind of value, so their ratio means
    something: neither is prose, they do not mix % with flat values, seconds with units, or
    level scaling with per-minute scaling, and they name the same stat and qualifier where both
    name one ("50% bonus AD" vs "15% total AD" is not comparable)."""
    a, b = _shape(old), _shape(new)
    if a is None or b is None or a.per_minute != b.per_minute:
        return False
    if a.unit != b.unit and ("%" in (a.unit, b.unit) or "" not in (a.unit, b.unit)):
        return False
    return all(x is None or y is None or x == y for x, y in ((a.stat, b.stat), (a.qualifier, b.qualifier)))


# ---------------------------------------------------------------- lines

# Not "updated": "updated to include X from Y" is prose.
_VERB = r"(?:increased|reduced|decreased|changed|lowered|raised|adjusted|modified)"
_MASK = "\x00{}\x00"
_TO_FROM = re.compile(
    rf"^(?P<attr>.*?)\s*\b{_VERB}\s+(?:to\s+|(?=[-\d\x00]))(?P<new>.+?)\s+from\s+(?P<old>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_FROM_TO = re.compile(
    rf"^(?P<attr>.*?)\s*\b{_VERB}\s+from\s+(?P<old>.+?)\s+to\s+(?P<new>.+?)\s*\.?\s*$", re.IGNORECASE
)
_ARROW = re.compile(
    rf"^(?P<attr>.*?)\s*(?::|\b{_VERB}\s+from|\b{_VERB}|\bfrom)\s*(?P<old>[^⇒]+?)\s*⇒\s*(?P<new>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)


def _mask(text: str) -> tuple[str, list[str]]:
    """Hide template bodies so "to"/"from" inside them are not matched."""
    saved: list[str] = []
    out, i = [], 0
    while (j := text.find("{{", i)) != -1:
        out.append(text[i:j])
        end = _scan(text, j)
        saved.append(text[j:end])
        out.append(_MASK.format(len(saved) - 1))
        i = end
    out.append(text[i:])
    return "".join(out), saved


def _unmask(text: str, saved: list[str]) -> str:
    return re.sub("\x00(\\d+)\x00", lambda m: saved[int(m.group(1))], text)


def split_change(text: str) -> tuple[str, str, str] | None:
    """(attribute, old, new) wikitext for a "X increased to NEW from OLD" style line, else None."""
    masked, saved = _mask(text)
    for pat in (_TO_FROM, _ARROW, _FROM_TO):
        if m := pat.match(masked):
            return tuple(_unmask(m.group(g), saved) for g in ("attr", "old", "new"))  # type: ignore[return-value]
    return None


_TAG_WORD = (
    r"(?:undocumented|bug ?fix(?:es)?|bug|new effects?|old effects?|removed|new|old|updated|clarification)"
)
_LINE_TAG = re.compile(rf"^({_TAG_WORD}(?:\s*/\s*{_TAG_WORD})*)\s*:\s*", re.IGNORECASE)
_HEADER_TAG = re.compile(rf"^{_TAG_WORD}(?:\s*/\s*{_TAG_WORD})*$", re.IGNORECASE)
# Lines about how the game looks or sounds, or about the client, rather than how it plays
_COSMETIC = re.compile(
    r"\b(?:[VS]FX|VO|voice[- ]?(?:overs?|lines?)|particles?|splash art(?:work)?|skins?|chromas?|icons?|tooltips?|"
    r"animations?|animated|models?|textures?|HUD|visuals?|visually|sounds?|audio|music|artwork|emotes?|"
    r"lore|colou?rblind|portraits?|loading screen|indicators?|cosmetic|renamed|recommended)\b",
    re.IGNORECASE,
)


def make_line(text: str, ability: str | None, prefix: str, tag: str | None, cosmetic: bool) -> ChangeLine:
    """One change line. `tag` and `cosmetic` come from the headers above it; the line's own leading
    label ("Bug Fix: ...") overrides the tag and is cut from the attribute."""
    raw = render(text)
    if m := _LINE_TAG.match(raw):
        tag = m.group(1)
    got = split_change(text)
    attr = _LINE_TAG.sub("", render(got[0]) if got else raw)
    old, new = (render(got[1]) or None, render(got[2]) or None) if got else (None, None)
    rel = rel_change(values(got[1]), values(got[2])) if got and old and new and comparable(old, new) else None
    looks = bool(_COSMETIC.search(prefix + raw))
    bug = tag is not None and "bug" in tag.lower()
    # A bug fix that moves a number ("Critical damage reduced to 75% AD from 100% AD") still counts
    gameplay = not (cosmetic or looks) if rel is not None else not (cosmetic or looks or bug)
    return ChangeLine(
        ability=ability,
        attribute=prefix + attr,
        old=old,
        new=new,
        rel_change=rel,
        raw=raw,
        tag=tag,
        gameplay=gameplay,
    )


# ---------------------------------------------------------------- sections and blocks

Kind = str  # "champion" | "item" | "rune" | "system"

_SECTION_KINDS: dict[str, Kind] = {
    "champions": "champion", "champion": "champion",
    "items": "item",
    "runes": "rune",
    "summoner spells": "system", "game": "system", "games": "system", "monsters": "system",
    "neutral buffs": "system", "buffs": "system", "turrets": "system", "structures": "system",
    "summoner's rift": "system", "minions": "system", "jungle plants": "system", "jungle": "system",
    "map": "system", "elemental rift": "system",
}  # fmt: skip
# Fixed names for system blocks, first match wins. A system heading that matches none (Practice
# Tool, Scoreboard, Pings, ARURF, ...) is client or game-mode content and is dropped.
_SYSTEM_NAMES: list[tuple[str, str]] = [
    ("Elder Dragon", r"\belder\b"),
    ("Dragon", (r"dragon|drakes?\b|^(infernal|mountain|ocean|cloud|chemtech|hextech|elemental) rift$|"
               r"cloudbringer|infernal might|mountainous vigor|oceanic will|hextech prowess|chemtech blight")),
    ("Baron Nashor", r"baron"),
    ("Rift Herald", r"herald"),
    ("Voidgrubs", r"voidgrub|voidmite|touch of the void|hunger of the void"),
    ("Atakhan", r"atakhan|blood rose|bloody petals"),
    ("Scuttle Crab", r"scuttl"),
    ("Jungle camps", (r"gromp|sentinel|brambleback|krug|raptor|murk ?wolf|^monsters?$|monster camps|"
                     r"crest of|^buffs$")),
    ("Jungle plants", r"jungle plants|blast cone|scryer|honeyfruit"),
    ("Smite", r"smite"),
    ("Teleport", r"teleport"),
    ("Flash", r"^flash$"), ("Ignite", r"^ignite$"), ("Ghost", r"^ghost$"), ("Heal", r"^heal$"),
    ("Barrier", r"^barrier$"), ("Exhaust", r"^exhaust$"), ("Cleanse", r"^cleanse$"),
    ("Clarity", r"^clarity$"), ("Summoner spells", r"^summoner spells$"),
    ("Turrets", r"turret|fortification|^structures?$|inhibitor|^nexus$|ohmwrecker"),
    ("Minions", r"minion"),
    ("Bounties", r"bount"),
    ("Experience and gold", r"^experience$|kill experience|^gold\b|gold generation"),
    ("Base and respawn", r"respawn|^fountain$|homeguard|^(empowered )?recall$"),
    ("Vision", r"^wards?$|^vision$|^stealth$|self-reveal|^fog of war$|faelight"),
    ("Map", r"summoner's rift|^map$|^terrain$|^walls$|alcoves|hex-?gates|^gates$"),
    ("Role quests", r"role quests|^quests$"),
    ("Feats of Strength", r"feats of strength"),
    ("Champion base stats", r"^most champions$"),
    ("Combat rules", (r"crowd control|grievous|omnivamp|life ?steal|lethality|^critical strikes$|tenacity|"
                     r"slow resist|^heal(ing|s)$|incoming healing|^energized$|^stasis$|^silence$|^dash$|"
                     r"^attack speed$|^ability haste$|damage modifiers|executions|^basic attacks?$|^ground$|"
                     r"^clones$|summoned units|^missiles$|^tethers$|targeted abilities|charged abilities|"
                     r"chargeable abilities")),
]  # fmt: skip


def system_name(name: str) -> str | None:
    """Fixed name for a Summoner's Rift system heading ("Voidgrub camp" -> "Voidgrubs"), else None."""
    low = name.lower()
    return next((fixed for fixed, pat in _SYSTEM_NAMES if re.search(pat, low)), None)


_STATS_HEADER = re.compile(r"^(base )?stats$|^general$", re.IGNORECASE)
_HEADING = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")
_BARE_HEADING = re.compile(r"^\{\{(?:ci|ii|ri|si)\|[^{}]*\}\}\s*$")
_SUBPAGE = re.compile(r"\[\[(V[^\]|#/]+/[^\]|#]+)(?:\|[^\]]*)?\]\]")


def _clean(wikitext: str, subpages: dict[str, str]) -> list[str]:
    """Drop comments, inline subpages such as [[V14.1/Items]], and join lines that a template spans."""
    text = re.sub(r"<!--.*?-->", "", wikitext, flags=re.DOTALL)
    text = re.sub(r"<noinclude>.*?</noinclude>", "", text, flags=re.DOTALL)
    out: list[str] = []
    for line in text.splitlines():
        if out and out[-1].count("{{") > out[-1].count("}}"):
            out[-1] += " " + line.strip()
            continue
        m = _SUBPAGE.search(line)
        if m and m.group(1) in subpages and not line.lstrip().startswith(("*", ";")):
            out.extend(_clean(subpages[m.group(1)], {}))
            continue
        out.append(line)
    return out


def _depth(line: str) -> int:
    return len(line) - len(line.lstrip("*"))


def parse_patch(wikitext: str, patch: str, subpages: dict[str, str] | None = None) -> list[ChangeBlock]:
    """All Summoner's Rift change blocks in one patch page, in page order."""
    lines = _clean(wikitext, subpages or {})
    blocks: dict[str, ChangeBlock] = {}
    in_main = False
    kind: Kind | None = None  # kind of the current section, None if the section is dropped
    l3_kind: Kind | None = None
    section_title = ""
    block: ChangeBlock | None = None  # None while skipping a dropped system heading
    opened = False  # a block (or a skip) has started since the last section heading
    stack: dict[int, tuple[str, Role]] = {}  # bullet depth -> (header text, role)
    sub = ""  # original name of a system block merged under a fixed name, e.g. "Gromp" in "Jungle camps"

    def open_block(name: str) -> ChangeBlock | None:
        """The block for `name`, or None for a system name outside Summoner's Rift gameplay."""
        nonlocal sub
        assert kind is not None
        sub = ""
        if kind == "system":
            fixed = system_name(name)
            if fixed is None:
                return None
            sub, name = ("" if fixed == name else name), fixed
        bid = f"{patch}:{kind}:{name}"
        if bid not in blocks:
            blocks[bid] = ChangeBlock(
                block_id=bid, patch=patch, kind=kind, name=name, rationale=None, lines=[], gameplay=False
            )
        return blocks[bid]

    bullets = [(i, _depth(ln)) for i, ln in enumerate(lines)]
    for i, line in enumerate(lines):
        if m := _HEADING.match(line):
            level, title = len(m.group(1)), render(m.group(2))
            if level <= 2:
                in_main = title == f"League of Legends V{patch}"
                kind = l3_kind = None
            elif in_main and level == 3:
                kind = l3_kind = _SECTION_KINDS.get(title.lower())
            elif in_main and l3_kind is not None:
                kind = _SECTION_KINDS.get(title.lower(), l3_kind)
            section_title, block, opened, stack = title, None, False, {}
            continue
        if not in_main or kind is None:
            continue
        # ";{{ci|Caitlyn}}", or a bare "{{ci|Fiora}}" line, which some 2019 pages use
        if line.startswith(";") or _BARE_HEADING.match(line):
            name, _, suffix = render(line.removeprefix(";")).partition(" - ")
            block, opened, stack = open_block(name.strip() or section_title), True, {}
            if block and suffix:  # ";{{ri|Lethal Tempo}} - Removed"
                block.lines.append(make_line(suffix, None, sub and f"{sub}: ", None, False))
            continue
        d = bullets[i][1]
        text = line[d:].strip()
        if d == 0 or not text:
            continue
        if not opened:  # bullets straight under a section heading such as "=== Turrets ==="
            block, opened = open_block(section_title), True
        if block is None:
            continue
        stack = {k: v for k, v in stack.items() if k < d}
        nxt = next((bd for j, bd in bullets[i + 1 :] if lines[j].strip()), 0)
        rendered = render(text)
        if nxt > d and split_change(text) is None and len(rendered) <= 80:
            header = rendered.rstrip(":").strip()
            stack[d] = (header, _role(text, header, block.name if kind == "champion" else None))
            continue
        ability, tag, cosmetic = None, None, False
        prefix_parts = [sub] if sub else []
        for k in sorted(stack):
            header, role = stack[k]
            if m := _LINE_TAG.match(header):
                tag = m.group(1)
            if role == "tag":
                tag = header
            elif kind == "champion" and k == 1 and role in ("ability", "other"):
                ability = header
            elif not (kind == "champion" and k == 1 and role == "stats"):
                prefix_parts.append(header)
            cosmetic |= role == "skin"
        prefix = "".join(f"{p}: " for p in prefix_parts)
        block.lines.append(make_line(text, ability, prefix, tag, cosmetic))
    out = [b for b in blocks.values() if b.lines]
    for b in out:
        b.gameplay = any(x.gameplay for x in b.lines)
    return out


Role = str  # "ability" | "stats" | "skin" | "tag" | "note" | "other"


def _role(text: str, header: str, champion: str | None) -> Role:
    """What a header bullet names. In a champion block a depth-1 "ability" or "other" header is
    taken as the ability and "stats" is dropped; other headers become an attribute prefix. Lines
    under a "skin" header are cosmetic; a "tag" header ("Bug Fix") sets the tag of the lines below."""
    names = {t.name for t in tokenize(text) if isinstance(t, Template)}
    if names & _ABILITY_TEMPLATES:
        return "ability"
    if _HEADER_TAG.match(header):
        return "tag"
    if _STATS_HEADER.match(header):
        return "stats"
    # "{{csl|Sona|DJ}}" renders "DJ Sona"; some pages write a skin as plain "Striker Lucian"
    if "csl" in names or (champion and re.search(rf"\w \b{re.escape(champion)}$", header)):
        return "skin"
    if header.endswith(".") or _LINE_TAG.match(header):  # "Recommended runes updated.", "Removed: ..."
        return "note"
    return "other"
