# Person 3: game data, LightGBM and the graph stage

You build the strongest model that sees only numbers.
The project's main claim is that prose beats it, so it has to be a fair, strong baseline.

## Links

- CommunityDragon raw game data: https://raw.communitydragon.org/
  Do not use Riot's Data Dragon for ability values: they are zero after patch 12.10.

## Steps

1. Ability text per champion per patch from CommunityDragon (10.1 onward).
   Person 4 uses it as text model input; you use it for similarity edges.
2. Check person 1's parsed numbers against a CommunityDragon diff of each patch.
   Report the mismatch rate and send person 1 the failures.
3. LightGBM on `labels.parquet` + `changes.jsonl` features:
   - pre-patch statistics (prev presence, recent trend)
   - parsed buff or nerf sign, number of changed attributes, summed relative change per ability
   - counts of system changes by type
   - the same features summed over each champion's partners, counters and substitutes (relation graph from person 2)
   - Fearless share
   It sees every number the text model sees, and no prose.
   Tune on dev (2025) only.
4. Graph stage for unchanged champions: a champion's indirect shift is a learned weighted sum of its neighbours' predicted shifts.
   Build it first on LightGBM's predictions; switch to the text model's predictions when person 4 has them.
   Compare edge weights from game statistics with weights from ability-text embedding similarity.
   The embeddings need a short GPU run; book it with person 4.
5. Write `data/predictions/lightgbm.csv` and `data/predictions/graph.csv`.

## Targets

- Nov 19: ability text extracted
- Nov 21: LightGBM on dev
- Nov 23: graph stage on dev; model freeze

## Also

Help person 5 if they need a second labeller for borderline cases.
