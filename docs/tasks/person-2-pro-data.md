# Person 2: pro data and evaluation

You build the target every model predicts, and the one piece of code that scores them all.

## Links

- gptilt pro games (Leaguepedia): https://huggingface.co/datasets/gptilt/lol-esports-matches
- Oracle's Elixir: https://oracleselixir.com/tools/downloads
- Leaguepedia pick-and-ban tables (draft order): https://lol.fandom.com/wiki/Special:CargoTables

## Steps

1. Load gptilt games from 2019 to 2026.
2. Map each game to a patch ID: the wiki title without the leading "V" ("9.22", "25.S1.1", "26.21").
   The game client calls 26.x patches 16.x; map those too.
   Use `data/raw/patches.jsonl` from person 1 for the full patch list.
3. Keep patches with at least 100 pro games.
   For each, `prev_patch` is the previous kept patch.
4. Per champion per patch: presence = (picks + bans) / games, then `delta_presence = presence - prev_presence`.
   Include every champion seen in pro play on either patch.
5. Fearless draft: flag each tournament's format and store the share of Fearless games per patch in `fearless_share`.
6. `changed`: true when `changes.jsonl` has a block for the champion with `gameplay` true (bug-fix and cosmetic-only blocks do not count).
7. Write `data/processed/labels.parquet` as `ChampionPatch` rows.
8. Relation graph from the same games: partners (same team), counters (opposite teams, same role) and substitutes (same role), with counts as weights.
9. Check whether draft order is available (Leaguepedia or Oracle's Elixir).
   If it is, tell the team; it allows a second target on pick order.

## Evaluation code (`src/patch2meta/eval/`)

Every model writes `data/predictions/<model>.csv` (`Prediction` rows), and your code scores them all.

- Mean within-patch Spearman between predicted and realised Δpresence, for changed, unchanged and all champions.
- Rank correlation within buffs and within nerfs, top-10 riser and faller precision, inverse-variance-weighted MAE.
- Broken down by change type (from `block_labels.jsonl`) and by Fearless vs standard.
- Noise ceiling: bootstrap games within each patch.
- Confidence intervals: bootstrap patches.
- Splits: train 2019 to 2024, dev 2025, test 2026 (26.01 to 26.20).
  The test split must only run when the team agrees to score it, once, after the freeze.

Write a "no change" predictions file (all zeros) first so the harness can be tested before any model exists.

## Targets

- Nov 19: `labels.parquet` and the patch mapping
- Nov 20: eval harness running on the no-change baseline
- Nov 22: relation graph
- Nov 23: model freeze

## Who waits on you

Everyone needs `labels.parquet`; person 3 needs the relation graph; persons 3 to 5 need the eval harness.
