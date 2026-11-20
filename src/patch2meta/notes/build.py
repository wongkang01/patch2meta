"""Build data/processed/changes.jsonl and the labelling sample from the raw patch notes.

Reads what notes.fetch saved (data/raw/patches.jsonl, wiki/*.wikitext, riot/*.html), takes the change
lines from the wiki and the rationale from Riot, and writes:
  data/processed/changes.jsonl        one ChangeBlock per line, patches in release order
  data/processed/changes_sample.csv   a random sample of gameplay blocks for hand labelling (person 5),
                                      spread across years and block kinds: block_id, patch, kind, name,
                                      rationale, lines ("ability: line", one per row of the cell)

Usage: uv run python -m patch2meta.notes.build [--since 12.17] [--sample 150]
"""

import argparse
import collections
import csv
import random
from pathlib import Path

from patch2meta.notes.riot import attach, riot_blocks
from patch2meta.notes.wikitext import parse_patch
from patch2meta.schema import ChangeBlock, PatchSource

RAW = Path("data/raw")
OUT = Path("data/processed")


def build_patch(src: PatchSource) -> tuple[list[ChangeBlock], int]:
    """Blocks for one patch, and how many Riot blocks found no wiki block."""
    title = src.wiki_title
    patch = title.removeprefix("V")
    wiki = (RAW / "wiki" / f"{title}.wikitext").read_text()
    subpages = {p.stem.replace("__", "/"): p.read_text() for p in (RAW / "wiki").glob(f"{title}__*.wikitext")}
    blocks = parse_patch(wiki, patch, subpages)
    html_path = RAW / "riot" / f"{title}.html"
    unmatched = attach(blocks, riot_blocks(html_path.read_text())) if html_path.exists() else []
    for b in blocks:
        for line in b.lines:
            if line.rel_change is not None:
                line.rel_change = round(line.rel_change, 4)
        ChangeBlock.model_validate(b.model_dump())
    return blocks, len(unmatched)


def sample(blocks: list[ChangeBlock], years: dict[str, int], n: int, seed: int = 0) -> list[ChangeBlock]:
    """n blocks drawn round-robin over (year, kind) groups so each group is represented."""
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[ChangeBlock]] = collections.defaultdict(list)
    for b in blocks:
        groups[(years[b.patch], b.kind)].append(b)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    picked: list[ChangeBlock] = []
    while len(picked) < n and any(groups.values()):
        for g in order:
            if groups[g] and len(picked) < n:
                picked.append(groups[g].pop())
    return sorted(picked, key=lambda b: b.block_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="9.1", help="first patch to include, e.g. 12.17")
    ap.add_argument("--sample", type=int, default=150, help="blocks in the labelling sample")
    args = ap.parse_args()

    sources = [PatchSource.model_validate_json(x) for x in (RAW / "patches.jsonl").read_text().splitlines()]
    titles = [s.wiki_title for s in sources]
    sources = sources[titles.index(f"V{args.since}") :]

    OUT.mkdir(parents=True, exist_ok=True)
    blocks: list[ChangeBlock] = []
    for src in sources:
        got, unmatched = build_patch(src)
        with_rationale = sum(b.rationale is not None for b in got)
        print(
            f"{src.wiki_title}: {len(got)} blocks, {with_rationale} with rationale, "
            f"{unmatched} Riot blocks unmatched, riot={src.riot_source}"
        )
        blocks += got

    (OUT / "changes.jsonl").write_text("".join(b.model_dump_json() + "\n" for b in blocks))
    kinds = collections.Counter(b.kind for b in blocks)
    print(f"wrote {len(blocks)} blocks from {len(sources)} patches to {OUT / 'changes.jsonl'}: {dict(kinds)}")

    years = {s.wiki_title.removeprefix("V"): s.release_date.year for s in sources}
    with (OUT / "changes_sample.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_id", "patch", "kind", "name", "rationale", "lines"])
        for b in sample([b for b in blocks if b.gameplay], years, args.sample):
            lines = "\n".join(f"{x.ability}: {x.raw}" if x.ability else x.raw for x in b.lines)
            w.writerow([b.block_id, b.patch, b.kind, b.name, b.rationale or "", lines])
    print(f"wrote a {args.sample}-block sample to {OUT / 'changes_sample.csv'}")


if __name__ == "__main__":
    main()
