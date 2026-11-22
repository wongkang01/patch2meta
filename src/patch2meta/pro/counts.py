"""Champion counts over any subset of games, vectorised with numpy.

`GameTable` turns games.parquet into integer arrays once; `count(table, mask)` then counts picks, bans,
wins, roles and presence for the games selected by a boolean mask. Presence counts each champion at
most once per game, even when a source lists it twice.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROLES = ["top", "jungle", "mid", "bot", "support"]  # pick lists are in this order


@dataclass
class GameTable:
    champions: list[str]  # code -> canonical name
    patch: np.ndarray  # patch ID per game
    dt: np.ndarray  # datetime64[ns] per game
    fearless: np.ndarray  # bool per game
    picks: np.ndarray  # int, shape (games, 2 sides, 5 roles); side 0 = blue
    side_won: np.ndarray  # int8, shape (games, 2): 1 won, 0 lost, -1 winner unknown
    present_game: np.ndarray  # one entry per (game, champion) picked or banned in it
    present_champ: np.ndarray
    ban_game: np.ndarray  # one entry per (game, champion) banned in it
    ban_champ: np.ndarray

    @property
    def n_champions(self) -> int:
        return len(self.champions)

    def code(self, name: str) -> int:
        return self.champions.index(name)


@dataclass
class Counts:
    n_games: int
    present: np.ndarray  # games where the champion is picked or banned
    picks: np.ndarray  # pick slots
    bans: np.ndarray  # games where the champion is banned
    wins: np.ndarray  # pick slots on the winning side
    decided_picks: np.ndarray  # pick slots in games with a known winner
    roles: np.ndarray  # shape (champions, 5): pick slots per role
    fearless_games: int

    def presence(self) -> np.ndarray:
        return self.present / self.n_games if self.n_games else np.zeros_like(self.present, dtype=float)


def build(games: pd.DataFrame, champions: list[str]) -> GameTable:
    code = {name: i for i, name in enumerate(champions)}
    n = len(games)
    picks = np.empty((n, 2, 5), dtype=np.int32)
    present_game, present_champ, ban_game, ban_champ = [], [], [], []
    for i, (pb, pr, bb, br) in enumerate(
        zip(games["picks_blue"], games["picks_red"], games["bans_blue"], games["bans_red"])
    ):
        picks[i, 0] = [code[c] for c in pb]
        picks[i, 1] = [code[c] for c in pr]
        bans = {code[c] for c in bb} | {code[c] for c in br}
        for c in bans:
            ban_game.append(i)
            ban_champ.append(c)
        for c in bans | set(picks[i].ravel().tolist()):
            present_game.append(i)
            present_champ.append(c)
    winner = games["winner"].to_numpy()
    blue_won = winner == games["team_blue"].to_numpy()
    red_won = winner == games["team_red"].to_numpy()
    known = pd.notna(winner) & (blue_won | red_won)
    side_won = np.full((n, 2), -1, dtype=np.int8)
    side_won[known, 0] = blue_won[known]
    side_won[known, 1] = red_won[known]
    return GameTable(
        champions=list(champions),
        patch=games["patch"].to_numpy(),
        dt=pd.to_datetime(games["datetime_utc"]).to_numpy(),
        fearless=games["fearless"].to_numpy(dtype=bool),
        picks=picks,
        side_won=side_won,
        present_game=np.asarray(present_game, dtype=np.int64),
        present_champ=np.asarray(present_champ, dtype=np.int64),
        ban_game=np.asarray(ban_game, dtype=np.int64),
        ban_champ=np.asarray(ban_champ, dtype=np.int64),
    )


def count(t: GameTable, mask: np.ndarray) -> Counts:
    """Counts over the games where `mask` is True."""
    k = t.n_champions
    sel = np.flatnonzero(mask)
    picks = t.picks[sel]  # (games, 2, 5)
    won = t.side_won[sel]  # (games, 2)
    flat = picks.reshape(-1)
    side_won = np.repeat(won, 5, axis=1).reshape(-1)
    roles = np.tile(np.arange(5), 2 * len(sel))
    return Counts(
        n_games=len(sel),
        present=np.bincount(t.present_champ[mask[t.present_game]], minlength=k),
        picks=np.bincount(flat, minlength=k),
        bans=np.bincount(t.ban_champ[mask[t.ban_game]], minlength=k),
        wins=np.bincount(flat[side_won == 1], minlength=k),
        decided_picks=np.bincount(flat[side_won >= 0], minlength=k),
        roles=np.bincount(flat * 5 + roles, minlength=k * 5).reshape(k, 5),
        fearless_games=int(t.fearless[sel].sum()),
    )
