# Processed data

File names stay fixed; the version and build date live here.
When you replace a file, bump its version, update its row in the same commit, and add a line to the history.
Check which copy you have with `shasum -a 256 <file> | cut -c1-12`.

| File | Version | Built | Rows | SHA-256 (first 12) | Producer | Contract |
|---|---|---|---|---|---|---|
| `changes.jsonl` | v3 | 2026-11-22 | 9,075 blocks | `e5e3d6cfac95` | 1 | `ChangeBlock` |
| `changes_sample.csv` | v3 | 2026-11-22 | 150 blocks | `f84ac5b80ebd` | 1 | gameplay `ChangeBlock`s for hand labelling |

Patches 9.1 to 26.23 (192 patches).
Rebuild with `uv run python -m patch2meta.notes.build` after fetching `data/raw/`.

## History

- v3, 2026-11-22: `rel_change` is (new - old) / |old| and null when old and new are not the same kind of value; 26.01 `gs` values parsed; units from `key=%` shown in `old`, `new` and `raw`.
- v2, 2026-11-20: added `tag` and `gameplay`; system blocks use 31 fixed names; rationale for 12.10 and 14.1; the sample draws only gameplay blocks.
- v1, 2026-11-19: first build. Copies of v1 and v2 were shared by hand; switch to this version.
