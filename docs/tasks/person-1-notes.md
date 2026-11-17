# Person 1: patch notes

You turn every patch's notes into change blocks.
Everyone downstream reads your `data/processed/changes.jsonl`, so you are on the critical path.

## Links

- Riot patch notes index (en-us): https://www.leagueoflegends.com/en-us/news/tags/patch-notes/
  Use en-us, not en-sg: en-sg only lists 12.17 onward, while en-us pages go back to 9.1.
- LoL Wiki version pages, e.g. https://wiki.leagueoflegends.com/en-us/V9.22
  Raw wikitext: `https://wiki.leagueoflegends.com/en-us/api.php?action=parse&page=V9.22&prop=wikitext&format=json`
- Wayback CDX API (fallback only): https://web.archive.org/cdx/search/cdx

## What the two sources give

| | Wiki | Riot page |
|---|---|---|
| Change lines (old ⇒ new) | Yes, in a fixed template | Yes, in HTML |
| Riot's rationale prose | No | Yes |
| Prev/next patch, release date, Riot URL | Yes (infobox) | No |

Take the change lines from the wiki and the rationale from the Riot page.

## Steps

1. Run `uv run python -m patch2meta.notes.fetch`.
   It walks the wiki chain from V9.1, saves `data/raw/wiki/*.wikitext` and `data/raw/riot/*.html`, and writes `data/raw/patches.jsonl`.
   A full run takes about 45 minutes and covers 192 patches (9.1 to the latest).
   9.4 to 9.17 come from Wayback, because those Riot pages are gone; all others are live.
   A patch whose notes are not out yet shows `riot_source: missing`; rerun the script after release.
2. Parse the wikitext into blocks.
   - `;{{ci|Caitlyn}}` starts a champion block; items, runes and system sections (map, objectives, dragons) have their own headings.
   - `* {{ai|Headshot|Caitlyn}}` names the ability; `** ... increased to X from Y.` is a change line.
   - Strip templates to plain text (`{{ap|70 to 270}}` → `70 to 270`) but keep the numbers.
   - Fill `ChangeLine.old`, `new` and `rel_change` (mean relative change across ranks; `None` if not numeric).
3. Extract rationale from the Riot HTML.
   Each block is a `patch-change-block`; the name is in `change-title`, the one-line summary in `summary`, and the designer rationale in `blockquote context`.
   The same layout appears on 9.22 and 26.21, so one parser should cover all eras; check a few patches from each year.
   Match Riot blocks to wiki blocks by name.
4. Strip mid-patch updates.
   Riot adds "Mid-Patch Updates" sections after release.
   Drop them from both sources so the model only sees what was public on patch day.
5. Write `data/processed/changes.jsonl`, one `ChangeBlock` per line, validated with `patch2meta.schema`.
6. Export a 150-block sample (random, spread across years and block kinds) as CSV for person 5: `block_id, patch, kind, name, rationale, lines`.

## Order

Do 12.17 onward first and share it, so persons 3 to 5 can start on real data.
Then backfill 9.1 to 12.16.

## Targets

- Nov 19: 12.17 onward in `changes.jsonl`; sample CSV to person 5
- Nov 21: all patches
- Nov 23: model freeze

## Who waits on you

Person 3 (LightGBM features, CommunityDragon check), person 4 (text model input), person 5 (labelling).
If you change a field in `ChangeBlock`, tell them in the PR.
