# Person 5: block labels, crowd and LLM baselines, demo

You label every change block, build two baselines that do not use our models, and build the demo.

## 1. Block labels

Every block gets a change type and a stated intent; the definitions are the `ChangeType` and `Intent` enums in `src/patch2meta/schema.py`.
Person 3 and the eval code use these labels to break results down by change type, which H1 depends on.

Rules:
- Mixed blocks (numbers and a mechanic change): label `mechanic`.
- Intent is what Riot *says*, not what you think will happen.
  Caitlyn 26.21 is a `nerf` ("it's time to nerf her"), even though her Q damage goes up.
- No rationale and numbers pointing both ways: `unclear`.

Steps, in this order:
1. Write a one-page guideline: the definitions above plus 5 to 10 worked examples.
2. Human labels first.
   From person 1's sample CSV, label 100 blocks yourself without looking at any model output.
   Save as `BlockLabel` rows with `labeller="human"`.
3. LLM labels for every block in `changes.jsonl`.
   Use a strong model (Claude Opus, GPT) with the guideline in the prompt, or a decision model (OpenRouter Jev, Gemini Decision 1) that returns calibrated probabilities.
   Save as `BlockLabel` rows with `labeller=<model id>` and `confidence` where available.
4. Compare the LLM with your 100 human labels: agreement per label and a confusion matrix.
   If one label is weak, tighten its definition in the prompt and rerun.
   If you try more than one model, keep the one that agrees best.
5. Write `data/processed/block_labels.jsonl`.

The labels describe the notes, not pro play, so it does not matter whether the LLM has read about past metas.

## 2. Baselines

- Crowd: Reddit patch preview threads posted before the notes go live, via the Arctic Shift API (https://github.com/ArthurHeitmann/arctic_shift).
  An LLM extracts winner and loser calls per champion; rank champions by net votes.
  Write `data/predictions/crowd.csv`.
- Zero-shot LLM: a hosted LLM gets the same inputs as the text model and predicts Δpresence.
  Score it only on patches released after that model's training cutoff.
  Write `data/predictions/llm_zeroshot.csv`.

Hosted LLM spend for the whole project should stay around US$10 to 15; keep API keys in `.env`.

## 3. Demo and release

- A patch-day page: champions expected to rise and fall, with intervals, and the note lines behind each call (from person 4's attribution).
  Replays any 2019 to 2026 patch next to what teams actually drafted.
  Build it on fake predictions now; switch to real ones after the freeze.
- Release the parsed change records (with source links) and per-champion stats on Hugging Face.

## Targets

- Nov 19: guideline written
- Nov 20: 100 human labels (sample from person 1 on Nov 19)
- Nov 21: LLM labels for all blocks, agreement reported
- Nov 22: crowd baseline
- Nov 23: model freeze
- Nov 30: demo with real predictions
