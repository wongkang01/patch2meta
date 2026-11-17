# Person 4: text models and GPU

You train the models that read the patch notes.
Keep other work off your plate: this is the critical path.

## GPU account (do first)

1. Find the group number in the sheet linked in the eDimension announcement of 20 Oct.
2. Log in at http://login2.gpucluster.sutd.edu.sg:8888/ as `50040_group_<number>` with the initial password from the announcement.
3. Change the password straight away and post the new one in the team chat (never in this repo).
4. Read the "GPU Guides" under Content → Final project.
5. Keep a booking list in the team chat: the group shares one account.
   Idle instances stop after 2 hours and only `~` persists, so checkpoint to `~/patch2meta/checkpoints/`.

## Steps

1. Start now on stub rows that match `ChampionPatch` and `ChangeBlock`; switch to real files when persons 1 and 2 publish them.
2. Input per champion per patch: its own change blocks (lines + rationale), then its ability text (from person 3), then the patch's item, map and objective changes.
   Unchanged champions get an empty change block, so the model can learn how system changes affect each kit.
3. Fine-tune DeBERTa-v3-base and ModernBERT-base.
   Concatenate the pooled embedding with pre-patch statistics and parsed numeric features; a small MLP predicts Δpresence.
   Loss: Huber weighted by inverse label variance (from game counts) plus a within-patch pairwise ranking term (RankNet style).
4. Qwen3-8B with QLoRA and a regression head on the same inputs (2 to 4 GPU-hours).
5. Ablations on dev: text only (no pre-patch stats); drop rationale; drop ability text; drop system-change context.
6. Leave-one-sentence-out attribution: which lines drive each prediction (for error analysis and the demo).
7. Write `data/predictions/<model>.csv` for each model and ablation.
   Tune on dev (2025) only; never look at 2026.

## Targets

- Nov 18: GPU account set up, training loop runs on stubs
- Nov 21: text model v1 on dev with real data
- Nov 23: Qwen3 and ablations; model freeze
