# Processed data

File names stay fixed; the version and build date live here.
When you replace a file, bump its version, update its row in the same commit, and add a line to the history.
Check which copy you have with `shasum -a 256 <file> | cut -c1-12`.

| File | Version | Built | Rows | SHA-256 (first 12) | Producer | Contract |
|---|---|---|---|---|---|---|
| `changes.jsonl` | v3 | 2026-11-22 | 9,075 blocks | `e5e3d6cfac95` | 1 | `ChangeBlock` |
| `changes_sample.csv` | v3 | 2026-11-22 | 150 blocks | `f84ac5b80ebd` | 1 | gameplay `ChangeBlock`s for hand labelling |
| `labels.parquet` | v1 | 2026-11-22 | 24,349 rows (train and dev) | `0923cc468a3a` | 2 | `ChampionPatch` |
| `prepatch_stats.parquet` | v1 | 2026-11-22 | 27,797 rows | `bae0a41e5a72` | 2 | `PrePatchStats` |
| `relation_graph.parquet` | v1 | 2026-11-22 | 1,826,684 edges | `a4de2257f6d2` | 2 | `RelationEdge` |
| `patch_map.csv` | v1 | 2026-11-22 | 194 patches | `f15ee007fddc` | 2 | `PatchMapRow` |
| `games.parquet` | v1 | 2026-11-22 | 109,095 games | `0c01830801ee` | 2 | games table (`pro/games.py`) |
| `draft.parquet` | v1 | 2026-11-22 | 1,752,880 draft actions | `b2fcf41f0785` | 2 | draft order (`pro/draft.py`) |
| `variants/comparison.json` | v1 | 2026-11-22 | 9 builds | `9b2b68b5af79` | 2 | data-choice comparison, train and dev only (`pro/variants.py`) |
| `data/private/labels_test.parquet` | v1 | 2026-11-22 | 3,275 rows (test) | `61f30f20edd9` | 2 | `ChampionPatch` |

Patches 9.1 to 26.23 (192 patches).
Rebuild with `uv run python -m patch2meta.notes.build` after fetching `data/raw/`.
Person 2's files are built from `data/raw/pro/` (`pro download`) with `uv run python -m patch2meta.pro build`, `pro draft` and `pro variants`.
`labels.parquet` v1 uses `changes.jsonl` v3. The 2026 test labels are in `data/private/labels_test.parquet` (last row); score them only in the single test run after the freeze.

## History

- Person 2's test labels v1, 2026-11-22: `data/private/labels_test.parquet` committed; score it only in the single test run after the freeze.
- Person 2's files v1, 2026-11-22: first commit of the labels, inputs, graph, patch map, games, draft order and the data-choice comparison.
- v3, 2026-11-22: `rel_change` is (new - old) / |old| and null when old and new are not the same kind of value; 26.01 `gs` values parsed; units from `key=%` shown in `old`, `new` and `raw`.
- v2, 2026-11-20: added `tag` and `gameplay`; system blocks use 31 fixed names; rationale for 12.10 and 14.1; the sample draws only gameplay blocks.
- v1, 2026-11-19: first build. Copies of v1 and v2 were shared by hand; switch to this version.
