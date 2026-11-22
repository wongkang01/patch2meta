# Open data choices (22 Nov 2026)

Each open question about the pro data is listed here with who should answer it, what today's build does, the options and Person 2's recommendation.
Today's build stays the default until the team answers.
The Fearless rule questions (A1 and A2) are in the team message and in `docs/checks/fearless-fix.patch`.

Five questions were answered in full builds, one build per option, with `uv run python -m patch2meta.pro variants`.
The numbers come from `data/processed/variants/comparison.json`, which reads train and dev only and never opens test labels.

## Options built and compared

Today's build has 109,095 games and keeps 130 train, 24 dev and 19 test patches.
On dev, the mean-reversion baseline scores 0.152 and the noise ceiling is 0.895.

| Question | Option | Games | Kept patches (train / dev / test) | Dev rank agreement | Patches with a moved label | Mean reversion, shared dev patches | Noise ceiling, shared dev patches |
|---|---|---|---|---|---|---|---|
| A4 | Count only game 1 of each Fearless series | 95,588 | 130 / 22 / 19 | 0.775 | 40 | 0.158 → 0.195 | 0.905 → 0.908 |
| B3 | Drop games without bans whose series has bans in other games | 108,949 | 130 / 24 / 19 | 0.999 | 7 | 0.152 → 0.151 | 0.895 → 0.894 |
| B3 | Drop every game without bans | 108,170 | 127 / 24 / 18 | 0.997 | 20 | 0.152 → 0.155 | 0.895 → 0.895 |
| C3 | Forecast time at 00:00 UTC on the release day | 109,095 | 130 / 24 / 19 | 1.000 | 0 | 0.152 → 0.153 | 0.895 → 0.895 |
| D1 | Major leagues and international events only | 16,031 | 59 / 13 / 11 | 0.469 | 71 | 0.166 → 0.192 | 0.911 → 0.854 |
| D1 | Major leagues, international events and their second divisions | 31,314 | 83 / 16 / 16 | 0.492 | 98 | 0.156 → 0.162 | 0.907 → 0.886 |
| D3 | A kept patch needs 200 games | 109,095 | 106 / 22 / 19 | 1.000 | 16 | 0.152 → 0.152 | 0.896 → 0.896 |
| D3 | A kept patch needs 300 games | 109,095 | 90 / 19 / 18 | 0.944 | 19 | 0.140 → 0.155 | 0.904 → 0.911 |

The columns compare each option with today's build:
- **Dev rank agreement** is the mean per-patch Spearman between the two builds' dev Δpresence. A value of 1 means the option ranks champions exactly as today's build does.
- **Patches with a moved label** counts train and dev patches in both builds where some champion's Δpresence moves by more than 0.02.
- **Mean reversion** and **noise ceiling** on shared dev patches give today's value, then the option's value, over the dev patches both builds keep. A dropped patch therefore cannot move them.

In short, A4 and D1 change what the target measures, while B3, C3 and D3 barely change the labels.

### A4. Every game or only game 1 in Fearless series (League players, Fazeela)

In Fearless series, later games draft from the champions that are left.
Counting only game 1 drops 13,507 of the 23,577 Fearless games.
The dev ranking then agrees with today's at only 0.775, because game 1 shows which champions teams want on the patch, while later games add the ones that remain.
Labels do not get noisier, and two dev patches (25.21 and 25.23) fall under 100 games.
Training is mostly standard draft and dev and test are mostly Fearless, and counting only game 1 makes Fearless patches closer to standard draft.

Options: keep every game (today); count game 1 only; weight games by their position in the series.
Person 2 makes no recommendation.

### B3. Games without bans (League players)

925 games have no bans recorded, for example blind-pick games, where a champion can appear on both teams.
Only 146 of them sit in a series with bans in other games, such as the deciding games of Demacia Cup 2025.
Dropping those 146 leaves the dev ranking at 0.999 of today's, and dropping all 925 leaves it at 0.997.

Options: count them like any game, with picks only (today); drop them from presence.
Recommended: count them.

### C3. Forecast time (League players)

Inputs use only P' games played before the forecast time, 00:00 UTC on the day before P's release, when the notes are usually out.
The median share of P' games before that time is 0.30 in train, 0.61 in dev and 0.57 in test, because pro leagues often stay on a patch after the next one is released.
A release-day forecast time raises the dev share to 0.68, moves the baselines by less than 0.004 and changes no label.

Options: keep the day before release (today); use the release day, which adds P' games at a small risk of leakage.
Recommended: keep the day before release.

### D1. Which tournaments count (everyone)

Presence uses all 2,402 Leaguepedia tournaments, including minor leagues that may lag behind on patches.
In 2025, major leagues (LCK, LPL, LEC, LCS, LTA, MSI, First Stand and Worlds) play 18% of the games on kept patches.
Keeping only major leagues leaves 83 of 174 kept patches, and adding their second divisions leaves 115.
Either way the dev ranking agrees with today's at under 0.5, and the noise ceiling falls, so labels get noisier.

Options: keep every tournament (today); major leagues and their second divisions only; weight games by league tier.
Recommended: keep every tournament.

### D3. Minimum games for a kept patch (everyone)

20 of 194 Riot patches have fewer than 100 pro games and are dropped.
Their notes still reach the next kept patch through `riot_patches_since_prev`.
Raising the minimum to 200 games drops 27 kept patches and 300 drops 47, while the remaining dev labels barely get cleaner.

Options: keep 100 (today); raise it, for fewer and cleaner patches.
Recommended: keep 100.

## Questions without a build

| ID | Who | Question | Today | Options | Recommended |
|---|---|---|---|---|---|
| A3 | League players | Is one Fearless flag per tournament enough, or do some events change format between stages? First Stand 2026 lists "Hard Fearless Draft" for its knockout stage only. | One flag per tournament | Per tournament is fine; name the events that mix formats so they can be split by stage | Per tournament |
| B2 | League players | In 2026 the team that drafts first is on the red side in 3,054 of 11,108 drafts; before 2026 that happened in only 2 drafts. Should priority follow draft position or side? This matters only if the team uses Δpriority. | Both are stored | Priority by draft position; priority by side | Draft position |
| C1 | Wong Kang, Fazeela | When P' is two or more Riot patches before P, should the text input hold the notes of every Riot patch in between? | `changed` already uses every patch in `riot_patches_since_prev`; the text input is not decided | Concatenate those notes; use only P's notes; drop these rows from training | Concatenate |
| C2 | Wong Kang, League players | Riot sometimes balances a champion mid-patch, and pros usually play the hotfixed version. Should mid-patch changes stay stripped from the notes? | Stripped | Keep them stripped and report it as a limitation; keep them as separate blocks marked as hotfixes | Separate hotfix blocks |
| C4 | League players | Which champions were disabled (bugs) or not yet legal in pro play on a 2025 or 2026 patch? Their presence drops for reasons the notes do not explain. | Not handled | Send a list, and those rows get a flag and leave the headline scores; mention it as a limitation | Send a list |
| C5 | League players | Should a full rework count like a new champion for two patches? | Reworks are ordinary rows | Flag reworks like new champions (needs a list of rework patches); leave them as ordinary rows | Ordinary rows |
| D2 | everyone | Leaguepedia gives no patch for 1,325 games (1,136 in 2025, mostly LDL, and 180 in 2026, mostly 2026 Asia Masters). Should the patch be recovered from the game date? | Dropped; the `games.py` docstring lists them | Keep dropping them; recover them by date for major events only | Keep dropping |
