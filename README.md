# Patch2Meta

Forecast how each League of Legends champion's pro draft priority (pick + ban presence) shifts on a new patch, from the patch notes alone, before teams play it.
SUTD 50.040 Natural Language Processing custom project, 2026 September term.

## Setup

```bash
git clone https://github.com/wongkang01/patch2meta.git && cd patch2meta
uv sync                       # Python 3.12, core deps
uv add <package>              # add what your part needs, commit pyproject.toml and uv.lock
uv run python -m patch2meta.notes.fetch   # raw patch notes for all patches, about 45 minutes
uv run python -m patch2meta.notes.build   # data/processed/changes.jsonl and the labelling sample, seconds
```

On the SUTD GPU cluster (one shared group account), clone into the home directory, since only `~` survives restarts.
Idle instances stop after 2 hours, so save checkpoints to `~/patch2meta/checkpoints/` as you go.
The account password is shared in the team chat, never in this repo.

## Data sources

| Source | Gives | Coverage | Access |
|---|---|---|---|
| [LoL Wiki version pages](https://wiki.leagueoflegends.com/en-us/V9.22) | Change lines in templated wikitext; patch chain and release dates | 9.1 to now | MediaWiki API |
| [Riot patch notes, en-us](https://www.leagueoflegends.com/en-us/news/tags/patch-notes/) | The same lines plus Riot's rationale prose | 9.1 to now (the en-sg index only lists 12.17 on) | Public HTML |
| Wayback Machine | Fallback if a Riot page disappears | First capture after release | CDX API |
| gptilt (Leaguepedia) on Hugging Face | Pro games with picks and bans | 2011 to 2026 | HF datasets |
| Oracle's Elixir | Cross-check, draft order | 2014 to 2026 | CSV downloads |
| CommunityDragon | Ability text; checks parsed numbers | 10.1 on | Static JSON |
| Reddit via Arctic Shift | Crowd baseline from patch preview threads | | Arctic Shift API |

The wiki has no rationale text, so the prose comes from the Riot pages.
Riot adds "Mid-Patch Updates" sections after publishing; the parser must strip them so the model sees only patch-day information.

## Pipeline and owners

Fill in names. Each person's instructions are in [`docs/tasks/`](docs/tasks/).

| # | Owner | Owns | Writes |
|---|---|---|---|
| 1 | | Notes: fetch, strip mid-patch sections, split into blocks, parse numbers | `changes.jsonl` |
| 2 | | Pro data: labels, patch mapping, Fearless flags, relation graph; shared eval code | `labels.parquet`, relation graph |
| 3 | | CommunityDragon diffs and ability text, LightGBM, graph stage | `predictions/lightgbm.csv`, `predictions/graph.csv` |
| 4 | | Text models (DeBERTa-v3, ModernBERT), Qwen3-8B QLoRA, ablations, GPU bookings | `predictions/<model>.csv` |
| 5 | | Block labelling with an LLM, crowd and zero-shot LLM baselines, demo, release | `block_labels.jsonl`, `predictions/crowd.csv` |

Code goes in `src/patch2meta/<area>/` (`notes`, `pro`, `eval`, `baselines`, `models`, `crowd`, `demo`), each area owned by one person.

Critical path: notes → `changes.jsonl` → text model → freeze.
Everyone else can build against stub rows that match `schema.py` and switch to real files when they land.
Person 1 should finish 12.17 onward first so modelling can start early.

## Data contracts

`src/patch2meta/schema.py` defines every shared file as a pydantic model.
Validate rows against it before writing, and change a contract only in a PR that tells its readers.

| File | Model | Producer |
|---|---|---|
| `data/raw/patches.jsonl` | `PatchSource` | 1 |
| `data/processed/changes.jsonl` | `ChangeBlock` | 1 |
| `data/processed/changes_sample.csv` | 150 gameplay `ChangeBlock`s for hand labelling | 1 |
| `data/processed/block_labels.jsonl` | `BlockLabel` | 5 |
| `data/processed/labels.parquet` | `ChampionPatch` | 2 |
| `data/predictions/<model>.csv` | `Prediction` | 3, 4, 5 |

`data/processed/` is committed, so `git pull` brings the latest files.
`data/processed/MANIFEST.md` lists each file's current version, build date and row count; bump it in the same commit as the file.
File names stay fixed so nobody's code breaks; older versions are in git history.
`data/raw/` (60 MB) stays local and is rebuilt by the fetch script.

## Block labels

An LLM labels every block with a `ChangeType` and a stated `Intent` (definitions in `schema.py`).
Intent is what Riot says it is doing ("it's time to nerf her" is a nerf even if Q damage goes up), not a prediction.
One person labels 100 random blocks, sampled across eras and block kinds, without seeing the model's labels; we report agreement per label.

## Evaluation

Train on 2019 to 2024 patches, dev on 2025, test on 2026 (26.01 to 26.20), scored once after the model freeze.
Primary metric: mean within-patch Spearman between predicted and realised Δpresence, for changed, unchanged and all champions.
All models are scored by the same code in `src/patch2meta/eval/`.

## Conventions

- Branch per task, PR into `main`, one reviewer.
- `uv run ruff check . && uv run ruff format .` before pushing.
- No credentials or API keys in git. Put keys in `.env`.
