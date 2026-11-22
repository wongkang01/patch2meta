"""Relation graph between champions, one graph per forecast patch: relation_graph.parquet (RelationEdge).

For forecast patch P the window is the `--window` kept patches before P, restricted to games dated
before cut(P), so no edge uses information from after the notes. Relations:
  partner     picked on the same team in the same game
  counter     picked in the same role on opposite teams (lane opponents, not proven counter-picks)
  substitute  role overlap: sum over roles of share_c(role) * share_n(role)
Partner and counter edges need at least `--min-games` games. Substitute edges need a role overlap of at
least 0.5 (in practice, the same main role); a top-k cut would drop equally good substitutes at random,
since all regular bot laners overlap at about 1.0.
"""

import argparse

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from patch2meta.pro.counts import GameTable, build
from patch2meta.pro.labels import champion_release_dates
from patch2meta.pro.patches import kept_patches, load_patch_map
from patch2meta.schema import RelationEdge

MIN_OVERLAP = 0.5
_OFF_DIAG = [(a, b) for a in range(5) for b in range(5) if a != b]


def pair_events(t: GameTable) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-game pair codes c * K + n: partner pairs (games, 40) and counter pairs (games, 10),
    plus team-level pick codes (games, 10) and role-pick codes c * 5 + role (games, 10)."""
    k = t.n_champions
    p = t.picks.astype(np.int64)  # (games, 2, 5)
    a = np.array([x for x, _ in _OFF_DIAG])
    b = np.array([y for _, y in _OFF_DIAG])
    partner = np.concatenate([p[:, s, a] * k + p[:, s, b] for s in (0, 1)], axis=1)
    counter = np.concatenate([p[:, 0, :] * k + p[:, 1, :], p[:, 1, :] * k + p[:, 0, :]], axis=1)
    picks = p.reshape(len(p), 10)
    role_picks = picks * 5 + np.tile(np.arange(5), 2)
    return partner, counter, picks, role_picks


def edges_for_window(
    t: GameTable, games_mask: np.ndarray, events: tuple, min_games: int
) -> list[tuple[int, int, str, float, float, float | None]]:
    k = t.n_champions
    partner, counter, picks, role_picks = (e[games_mask] for e in events)
    n_games = int(games_mask.sum())
    team_games = 2 * n_games
    pick_n = np.bincount(picks.ravel(), minlength=k).astype(float)
    out = []
    for relation, codes in (("partner", partner), ("counter", counter)):
        w = np.bincount(codes.ravel(), minlength=k * k).reshape(k, k)
        # Blind-pick games can put one champion on both teams; it is not its own partner or counter.
        np.fill_diagonal(w, 0)
        cs, ns = np.nonzero(w >= min_games)
        for c, n in zip(cs.tolist(), ns.tolist()):
            weight = float(w[c, n])
            out.append(
                (c, n, relation, weight, weight / pick_n[c], weight * team_games / (pick_n[c] * pick_n[n]))
            )
    roles = np.bincount(role_picks.ravel(), minlength=k * 5).reshape(k, 5).astype(float)
    played = pick_n > 0
    share = np.zeros_like(roles)
    share[played] = roles[played] / pick_n[played, None]
    overlap = share @ share.T
    np.fill_diagonal(overlap, 0.0)
    for c in np.flatnonzero(played).tolist():
        for n in np.flatnonzero(overlap[c] >= MIN_OVERLAP).tolist():
            out.append((c, n, "substitute", float(overlap[c, n]), float(overlap[c, n]), None))
    return out


def build_graph(t: GameTable, pm: pd.DataFrame, window: int, min_games: int) -> pd.DataFrame:
    kept = kept_patches(pm)
    index = {p: i for i, p in enumerate(kept)}
    kept_idx = np.array([index.get(p, -1) for p in t.patch])
    cut_of = pm.set_index("patch")["cut"]
    events = pair_events(t)
    frames = []
    for i in range(1, len(kept)):
        lo = max(0, i - window)
        mask = (kept_idx >= lo) & (kept_idx < i) & (t.dt < np.datetime64(cut_of[kept[i]]))
        rows = edges_for_window(t, mask, events, min_games)
        df = pd.DataFrame(rows, columns=["c", "n", "relation", "weight", "share", "lift"])
        df.insert(0, "patch", kept[i])
        df["window_first"], df["window_last"] = kept[lo], kept[i - 1]
        df["n_games_window"] = int(mask.sum())
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)
    names = np.array(t.champions)
    g.insert(1, "champion", names[g.pop("c").to_numpy()])
    g.insert(2, "neighbour", names[g.pop("n").to_numpy()])
    return g


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--window", type=int, default=6, help="kept patches before P in the window")
    ap.add_argument("--min-games", type=int, default=5, help="minimum games for partner and counter edges")


def run(args: argparse.Namespace) -> None:
    pm = load_patch_map(args.data / "patch_map.csv")
    t = build(pd.read_parquet(args.data / "games.parquet"), sorted(champion_release_dates().index))
    g = build_graph(t, pm, args.window, args.min_games)
    sample = g.sample(min(len(g), 5000), random_state=0)
    TypeAdapter(list[RelationEdge]).validate_python(
        sample.astype(object).where(sample.notna(), None).to_dict("records")
    )
    name = "relation_graph.parquet" if args.window == 6 else f"relation_graph_w{args.window}.parquet"
    out = args.data / name
    g.to_parquet(out, index=False)
    print(
        f"{out}: {len(g)} edges over {g['patch'].nunique()} patches; {g['relation'].value_counts().to_dict()}"
    )
