"""Re-check every number quoted in Person 2's docs against the data on disk.

Usage (from the repo root):
  uv run python docs/checks/verify_claims.py [--json PATH]

The script is offline. It reads data/processed, data/private, data/raw/pro, data/results and
data/predictions, and never writes to data/. Each claim has an expected value copied from a doc and a
function that recomputes it. Status is PASS, FAIL or MANUAL (needs a web page; the record names the
screenshot or the missing source). The exit code is 1 if any claim fails.

Test labels (data/private/labels_test.parquet) are read only for their keys (patch, champion): no
test-set presence, delta or flag value is read, printed or written.
"""

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from patch2meta.eval import bootstrap, ceiling, metrics
from patch2meta.eval.load import load_labels
from patch2meta.pro import cargo
from patch2meta.pro.fearless import THRESHOLD, series_fearless
from patch2meta.pro.names import NOT_A_CHAMPION
from patch2meta.pro.patches import PATCH_ID, client_version, kept_patches, load_patch_map, patch_key, split_of

GUIDE = "docs/guide/patch2meta-guide.html"
PLAN = "docs/plans/person-2-implementation-plan.md"
CHECKS = "docs/checks/2026-11-18-manual-checks.md"
IMG = "docs/checks/img/"
PROCESSED = Path("data/processed")
RAW = Path("data/raw/pro")
RESULTS = Path("data/results")
CHAMP_LISTS = ["picks_blue", "picks_red", "bans_blue", "bans_red"]
STUDY = ("train", "dev", "test")


@dataclass
class Claim:
    id: str
    doc: str
    locator: str
    claim: str
    expected: object
    actual: Callable[[], object] | None  # None: needs a web page or a source that is not downloaded
    how: str
    tol: float = 0.0


# ---------------------------------------------------------------- data, loaded once


@cache
def games() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "games.parquet")


@cache
def pm() -> pd.DataFrame:
    return load_patch_map()


@cache
def pmi() -> pd.DataFrame:
    return pm().set_index("patch")


@cache
def labels() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "labels.parquet")


@cache
def labels_test() -> pd.DataFrame:
    """Keys of the test labels only; no value column is ever loaded."""
    return pd.read_parquet("data/private/labels_test.parquet", columns=["patch", "champion"])


@cache
def all_labels() -> pd.DataFrame:
    return pd.concat([labels(), labels_test()], ignore_index=True)


@cache
def inputs() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "prepatch_stats.parquet")


@cache
def graph() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "relation_graph.parquet")


@cache
def draft() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "draft.parquet")


@cache
def gptilt_raw() -> pd.DataFrame:
    g = pd.read_parquet(RAW / "gptilt_games.parquet")
    g["datetime_utc"] = pd.to_datetime(g["datetime_utc"])
    return g


@cache
def cargo_raw() -> pd.DataFrame:
    return pd.read_parquet(RAW / "cargo_games_2026.parquet")


@cache
def draft_raw() -> pd.DataFrame:
    parts = [
        pd.read_parquet(p).assign(year=int(p.stem)) for p in sorted((RAW / "cargo_draft").glob("*.parquet"))
    ]
    return pd.concat(parts, ignore_index=True)


@cache
def champions() -> pd.Series:
    c = pd.read_parquet(RAW / "cargo_champions.parquet")
    return pd.Series(pd.to_datetime(c["ReleaseDate"]).to_numpy(), index=c["Name"]).sort_index()


def result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def split_list(value: object) -> list[str]:
    return [x.strip() for x in value.split(",")] if isinstance(value, str) and value else []


def r(x, nd: int = 3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def study_patches() -> pd.DataFrame:
    p = pm()
    return p[p["kept"] & p["split"].isin(STUDY)]


def label_row(patch: str, champ: str, cols: list[str], frame: pd.DataFrame | None = None) -> list:
    d = labels() if frame is None else frame
    row = d[(d["patch"] == patch) & (d["champion"] == champ)].iloc[0]
    return [r(row[c]) if isinstance(row[c], float) else row[c] for c in cols]


def n_present(g: pd.DataFrame, champ: str) -> int:
    """Games in which the champion is picked or banned."""
    return int(sum(any(champ in list(g[c].iloc[i]) for c in CHAMP_LISTS) for i in range(len(g))))


# ---------------------------------------------------------------- patches and games


def dropped_patches() -> dict:
    p = pm()
    d = p[~p["kept"] & (p["n_pro_games"] > 0)]
    return dict(zip(d["patch"], d["n_pro_games"].astype(int)))


def patch_map_row(patch: str) -> list:
    row = pmi().loc[patch]
    since = row["riot_patches_since_prev"]
    return [
        row["wiki_title"],
        row["client_version"],
        str(row["release_date"].date()),
        str(row["cut"].date()),
        int(row["n_pro_games"]),
        bool(row["kept"]),
        row["prev_patch"] if isinstance(row["prev_patch"], str) else None,
        since if isinstance(since, str) else None,
        row["split"],
    ]


def overlap_2026() -> int:
    """Patches 26.01 to 26.18 whose game count differs between gptilt and Cargo."""
    gp = gptilt_raw()
    gp = gp[gp["datetime_utc"] >= "2026-01-01"].groupby("patch").size()
    cg = cargo_raw().groupby("Patch").size()
    lo, hi = patch_key("26.01"), patch_key("26.18")
    ps = [p for p in cg.index if PATCH_ID.match(p) and lo <= patch_key(p) <= hi]
    return int(sum(gp.get(p, 0) != cg[p] for p in ps)), len(ps)


def no_patch() -> dict:
    gp = gptilt_raw()
    gp = gp[(gp["datetime_utc"] >= "2019-01-01") & (gp["datetime_utc"] < "2026-01-01")]
    empty = gp[gp["patch"].isna() | (gp["patch"] == "")]
    c = cargo_raw()
    n25 = gp["datetime_utc"].dt.year.eq(2025)
    e25 = empty[empty["datetime_utc"].dt.year == 2025]
    return {
        "total": len(empty) + int((c["Patch"].isna() | (c["Patch"] == "")).sum()),
        "2025": len(e25),
        "2025_share": r(len(e25) / n25.sum(), 2),
        "2025_ldl": int(e25["tournament"].str.startswith("LDL").sum()),
        "2026_cargo": int((c["Patch"].isna() | (c["Patch"] == "")).sum()),
    }


def tier1() -> list[int]:
    ct = pd.read_parquet(RAW / "cargo_tournaments.parquet").drop_duplicates("OverviewPage")
    level = games()["tournament"].map(ct.set_index("OverviewPage")["TournamentLevel"])
    n = games()[level == "Primary"].groupby("patch").size()
    k = [p for p in n[n >= 100].index if split_of(p) in STUDY]
    return [len(k), sum(split_of(p) == "test" for p in k)]


def wiki_dates() -> list:
    w = pd.read_csv(RAW / "wiki_release_dates.csv", dtype={"patch": str})
    w = w[w["patch"].str.match(PATCH_ID.pattern)]
    dated = w.dropna(subset=["release_date"])
    order = sorted(dated["patch"], key=patch_key)
    in_order = bool(
        pd.to_datetime(dated.set_index("patch").loc[order, "release_date"]).is_monotonic_increasing
    )
    return [len(dated), len(w), sorted(w.loc[w["release_date"].isna(), "patch"]), in_order]


# ---------------------------------------------------------------- cleaning


def none_ban_share_range() -> list:
    """Yearly share of games with a "None" ban, over the games the pipeline uses (a non-empty patch)."""
    gp = gptilt_raw()
    gp = gp[(gp["datetime_utc"] >= "2019-01-01") & (gp["datetime_utc"] < "2026-01-01")]
    gp = gp[gp["patch"].notna() & (gp["patch"] != "")]
    has = [("None" in list(b)) or ("None" in list(x)) for b, x in zip(gp["bans_blue"], gp["bans_red"])]
    by_year = pd.Series(has).groupby(gp["datetime_utc"].dt.year.to_numpy()).mean()
    c = cargo_raw()
    c = c[c["Patch"].notna() & (c["Patch"] != "")]
    by_year[2026] = np.mean(
        [
            ("None" in split_list(b)) or ("None" in split_list(x))
            for b, x in zip(c["Team1Bans"], c["Team2Bans"])
        ]
    )
    return [r(by_year.min()), r(by_year.max())]


def missing_data() -> list:
    gp = gptilt_raw()
    gp = gp[(gp["datetime_utc"] >= "2019-01-01") & (gp["datetime_utc"] < "2026-01-01")]
    slots = [sum(list(gp[c].iloc[i]).count("Missing Data") for c in CHAMP_LISTS) for i in range(len(gp))]
    hit = gp[np.array(slots) > 0]
    c = cargo_raw()
    cargo_hits = sum(
        any("Missing Data" in split_list(v) for v in row)
        for row in zip(c["Team1Picks"], c["Team2Picks"], c["Team1Bans"], c["Team2Bans"])
    )
    return [len(hit) + cargo_hits, int(sum(slots)), sorted(hit["patch"].unique())]


def double_listed() -> int:
    n = 0
    for row in zip(*(games()[c] for c in CHAMP_LISTS)):
        xs = [x for lst in row for x in lst if x not in NOT_A_CHAMPION]
        n += len(xs) != len(set(xs))
    return n


def none_presence_2620() -> float:
    """Occurrences of "None" per game on 26.20 in the source the pipeline now uses (Cargo)."""
    c = cargo_raw()
    c = c[c["Patch"] == "26.20"]
    occ = sum(
        split_list(v).count("None")
        for col in ["Team1Picks", "Team2Picks", "Team1Bans", "Team2Bans"]
        for v in c[col]
    )
    return r(occ / len(c), 2)


def none_presence_2620_gptilt() -> float:
    g = gptilt_raw()
    g = g[g["patch"] == "26.20"]
    occ = sum(list(v).count("None") for col in CHAMP_LISTS for v in g[col])
    return r(occ / len(g), 2)


# ---------------------------------------------------------------- labels and inputs


def release_patch(day: pd.Timestamp) -> str:
    p = pm().dropna(subset=["release_date"])
    p = p[p["release_date"] <= day]
    return max(p["patch"], key=patch_key)


def recent_champions() -> pd.Series:
    rel = champions()
    return rel[rel >= "2019-01-01"].sort_values()


@cache
def mid_patch_releases() -> pd.DataFrame:
    rows = []
    for champ, day in recent_champions().items():
        p = release_patch(day)
        gap = (day - pmi().at[p, "release_date"]).days
        if gap >= 1:
            rows.append((champ, p, gap, n_present(games()[games()["patch"] == p], champ)))
    return pd.DataFrame(rows, columns=["champion", "patch", "days", "games"]).set_index("champion")


def release_splits() -> list:
    rec = recent_champions()
    sp = pd.Series({c: split_of(release_patch(d)) for c, d in rec.items()})
    train = [c for c in rec.index if sp[c] == "train"]
    return [
        int((sp == "train").sum()),
        int((sp == "dev").sum()),
        int((sp == "test").sum()),
        train[0],
        train[-1],
        sorted(sp[sp == "dev"].index),
        sorted(sp[sp == "test"].index),
    ]


def presence_path(champ: str, n: int) -> list:
    d = labels()[(labels()["champion"] == champ) & (labels()["presence"] > 0)].copy()
    d = d.iloc[sorted(range(len(d)), key=lambda i: patch_key(d["patch"].iloc[i]))].head(n)
    return [[p, r(x)] for p, x in zip(d["patch"], d["presence"])]


def new_champion_counts() -> list:
    """Flagged rows in labels.parquet (train and dev); Locke's test rows are checked in the test run."""
    nc = labels()[labels()["new_champion"]]
    per = nc.groupby("champion").size()
    return [nc["champion"].nunique(), len(nc), sorted(per.unique().tolist())]


def locke() -> list:
    """Locke's patches_since_debut, from the inputs only."""
    i = inputs()
    psd = i[(i["champion"] == "Locke") & i["patch"].isin(["26.13", "26.14", "26.15"])]
    psd = dict(zip(psd["patch"], psd["patches_since_debut"].astype(int)))
    return [psd["26.13"], psd["26.14"], psd["26.15"]]


def debut_percentiles() -> dict:
    """Median within-patch percentile of |Δpresence| on a new champion's k-th pro patch, train and dev only.

    A k-th patch in the test split (Zaahen's third and fourth) has no label here and is skipped. The
    percentile of |prev_delta_1| on k = 3 comes from the inputs, which cover every split.
    """
    kept = kept_patches(pm())
    pos = {p: i for i, p in enumerate(kept)}
    a = labels().copy()
    a["pct"] = a.groupby("patch")["delta_presence"].transform(lambda s: s.abs().rank(pct=True))
    inp = inputs().copy()
    inp["pct1"] = inp.groupby("patch")["prev_delta_1"].transform(lambda s: s.abs().rank(pct=True))
    a = a.set_index(["patch", "champion"])
    inp = inp.set_index(["patch", "champion"])
    first = a[a["new_champion"]].reset_index().groupby("champion")["patch"].agg(lambda s: min(s, key=pos.get))
    out: dict = {f"k{k}": [] for k in range(1, 5)}
    out["k3_prev_delta_1"] = []
    for champ, p0 in first.items():
        for k in range(1, 5):
            p = kept[pos[p0] + k - 1]
            if (p, champ) in a.index:
                out[f"k{k}"].append(a.at[(p, champ), "pct"])
            if k == 3:
                out["k3_prev_delta_1"].append(inp.at[(p, champ), "pct1"])
    med = {key: r(np.median(v), 2) for key, v in out.items()}
    med["n_champions"] = len(first)
    med["n_k4"] = len(out["k4"])
    return med


def dev_spearman_recomputed() -> dict:
    """Dev mean Spearman of two baselines from labels and inputs, with an own rank correlation.

    It does not read data/results/ or call the eval code, so it checks the scorer, not a copy of its output.
    """
    lab = labels()[(labels()["patch"].map(pmi()["split"]) == "dev") & ~labels()["new_champion"]]
    m = lab.merge(inputs(), on=["patch", "champion"], suffixes=("", "_in"))
    preds = {"mean_reversion": -m["prev_presence_in"], "last_delta": m["prev_delta_1"].fillna(0.0)}
    out = {}
    for name, pred in preds.items():
        vals = []
        for _, d in m.assign(pred=pred.to_numpy()).groupby("patch"):
            x, y = d["pred"].rank(), d["delta_presence"].rank()
            vals.append(0.0 if x.nunique() < 2 or y.nunique() < 2 else float(np.corrcoef(x, y)[0, 1]))
        out[name] = r(np.mean(vals))
    return out


def team1_name_mismatch_by_year() -> dict:
    """Draft rows (2019 to 2025) whose Team1 is not the blue team of games.parquet, and of those the true
    side swaps (Team1 is the red team); blank Team1 rows are left out."""
    d = draft_raw()
    d = d[d["year"] <= 2025].drop_duplicates("GameId")
    m = d.merge(games()[["game_id", "team_blue", "team_red"]], left_on="GameId", right_on="game_id")
    named = m["Team1"].notna() & (m["Team1"] != "")
    differ = named & (m["Team1"] != m["team_blue"])
    swap = differ & (m["Team1"] == m["team_red"])
    return {
        "differ": {int(y): int(v) for y, v in differ.groupby(m["year"]).sum().items()},
        "side_swap": {int(y): int(v) for y, v in swap.groupby(m["year"]).sum().items()},
    }


def cut_effect() -> dict:
    out = {}
    for sp in ("train", "dev"):
        lab = load_labels([sp])
        m = lab.merge(
            inputs()[["patch", "champion", "prev_presence"]], on=["patch", "champion"], suffixes=("", "_in")
        )
        m = m[~m["new_champion"]]
        cut = metrics.patch_spearman(m.assign(pred_delta_presence=-m["prev_presence_in"]))[0].mean()
        full = metrics.patch_spearman(m.assign(pred_delta_presence=-m["prev_presence"]))[0].mean()
        out[sp] = [r(cut), r(full)]
    return out


def prev_share_by_split() -> dict:
    i = inputs().groupby("patch").agg(prev=("prev_patch", "first"), n=("prev_n_games", "first"))
    share = i["n"] / i["prev"].map(pmi()["n_pro_games"])
    sp = i.index.map(pmi()["split"])
    return {s: r(share[sp == s].median(), 2) for s in STUDY}


def gptilt_only_patches() -> list:
    n = gptilt_raw()[gptilt_raw()["datetime_utc"] >= "2019-01-01"].groupby("patch").size()
    n = n[[PATCH_ID.match(p) is not None for p in n.index]]
    n = n[(n >= 100) & np.array([split_of(p) in STUDY for p in n.index])]
    return [len(n), {s: sum(split_of(p) == s for p in n.index) for s in STUDY}, int(n.sum())]


def thin_train_patches() -> list:
    i = inputs().groupby("patch")["prev_n_games"].first()
    tr = i[i.index.map(pmi()["split"]) == "train"]
    return [len(tr), int((tr < 100).sum()), int((tr < 50).sum()), int(tr.min())]


# ---------------------------------------------------------------- Fearless


@cache
def fearless_parts():
    g = games().sort_values("datetime_utc", kind="stable").copy()
    g["k"] = g.groupby("match_id").cumcount() + 1
    g["n"] = g.groupby("match_id")["game_id"].transform("size")
    tour = g.drop_duplicates("match_id").set_index("match_id")["tournament"]
    hard = series_fearless(g)
    nolast = series_fearless(g[g["k"] < g["n"]])
    multi = g[g["n"] >= 2]
    cols = ["match_id", "team", "picks"]
    long = pd.concat(
        [
            multi[["match_id", "team_blue", "picks_blue"]].set_axis(cols, axis=1),
            multi[["match_id", "team_red", "picks_red"]].set_axis(cols, axis=1),
        ]
    ).explode("picks")
    soft = (long.groupby(["match_id", "team", "picks"]).size().groupby(level="match_id").max() <= 1).astype(
        float
    )

    def per_tour(s: pd.Series) -> pd.DataFrame:
        return s.groupby(tour.reindex(s.index)).agg(["sum", "size", "mean"])

    t = pd.concat({"hard": per_tour(hard), "nolast": per_tour(nolast), "soft": per_tour(soft)}, axis=1)
    a = t[("hard", "mean")] >= THRESHOLD
    b = a | ((t[("nolast", "mean")] >= THRESHOLD) & (t[("nolast", "size")] >= 3))
    c = b | ((t[("soft", "mean")] >= THRESHOLD) & (t[("soft", "size")] >= 5))
    return g, t, {"A": a, "B": b, "C": c}


def rule_flags(rule: str) -> pd.Series:
    g, _, rules = fearless_parts()
    return g["tournament"].map(rules[rule]).fillna(False).astype(bool)


def patch_shares(rule: str) -> pd.Series:
    g = fearless_parts()[0]
    kept = set(kept_patches(pm()))
    on = g["patch"].isin(kept)
    return rule_flags(rule)[on].groupby(g.loc[on, "patch"]).mean()


def impact(rule: str) -> dict:
    share = patch_shares(rule)
    sp = share.index.map(pmi()["split"])
    out = {"games": int(rule_flags(rule).sum())}
    for s in STUDY:
        out[s] = f"{int((share[sp == s] >= 0.5).sum())}/{int((sp == s).sum())}"
    if rule != "A":
        moved = (share - patch_shares("A")).abs() > 0.05
        out["moved"] = int(moved.sum())
        out["moved_by_split"] = {
            s: int((moved & (sp == s)).sum()) for s in STUDY if (moved & (sp == s)).any()
        }
    return out


def extension(rule: str, base: str) -> dict:
    g, _, rules = fearless_parts()
    added = rules[rule] & ~rules[base]
    names = added[added].index
    in_added = g["tournament"].isin(names)
    return {
        "tournaments": len(names),
        "games": int(in_added.sum()),
        "games_after_2024": int((in_added & (g["datetime_utc"].dt.year >= 2025)).sum()),
    }


def tournament_row(name: str) -> str:
    g, t, _ = fearless_parts()
    row = t.loc[name]
    flag = "Fearless" if bool(g.loc[g["tournament"] == name, "fearless"].iloc[0]) else "Standard"

    def part(key: str) -> str:
        return (
            f"{int(row[(key, 'sum')])}/{int(row[(key, 'size')])}" if pd.notna(row[(key, "size")]) else "0/0"
        )

    return f"{flag}; hard {part('hard')}; soft {part('soft')}; without last {part('nolast')}"


def fearless_by_year() -> dict:
    g = games()
    year = g["datetime_utc"].dt.year
    series = series_fearless(g)
    series_year = g.drop_duplicates("match_id").set_index("match_id")["datetime_utc"].dt.year
    return {
        "series_no_repeat": {
            int(y): r(v) for y, v in series.groupby(series_year.reindex(series.index)).mean().items()
        },
        "games_flagged": {int(y): r(v) for y, v in g["fearless"].groupby(year).mean().items()},
    }


def demacia_2524() -> list:
    g = games()[games()["patch"] == "25.24"]
    demacia = int((g["tournament"] == "Demacia Cup 2025").sum())
    return [demacia, len(g), r(patch_shares("A")["25.24"]), r(patch_shares("B")["25.24"], 2)]


# ---------------------------------------------------------------- relation graph


def caitlyn_2620() -> pd.DataFrame:
    e = graph()
    return e[(e["patch"] == "26.20") & (e["champion"] == "Caitlyn")]


def top_ties() -> list:
    s = graph().query("relation == 'substitute'")[["patch", "champion", "weight"]].copy()
    s["w"] = s["weight"].round(3)
    top = s.groupby(["patch", "champion"])["w"].transform("max")
    ties = s[s["w"] == top].groupby(["patch", "champion"]).size()
    return [round(float((ties > 15).mean()), 3), int(ties.max())]


def red_first_by_year() -> dict:
    d = pd.read_parquet(PROCESSED / "draft.parquet")
    d = d[(d["order"] == 1) & d["game_id"].isin(set(d.loc[d["champion"].notna(), "game_id"]))]  # real drafts
    g = pd.read_parquet(PROCESSED / "games.parquet", columns=["game_id", "datetime_utc"])
    d = d.merge(g, on="game_id")
    return d.groupby(d["datetime_utc"].dt.year)["side"].apply(lambda s: int((s == "red").sum())).to_dict()


def graph_row(neighbour: str, relation: str) -> list:
    row = caitlyn_2620().query("neighbour == @neighbour and relation == @relation").iloc[0]
    w = r(row["weight"]) if relation == "substitute" else int(row["weight"])
    return [
        w,
        r(row["share"]),
        r(row["lift"], 2),
        row["window_first"],
        row["window_last"],
        int(row["n_games_window"]),
    ]


def top(relation: str, n: int, by: str = "weight") -> list:
    d = caitlyn_2620()[caitlyn_2620()["relation"] == relation].sort_values(by, ascending=False).head(n)
    if by == "lift":
        return [[x, r(lift, 1), int(w)] for x, lift, w in zip(d["neighbour"], d["lift"], d["weight"])]
    return [[x, int(w)] for x, w in zip(d["neighbour"], d["weight"])]


def graph_rules() -> list:
    e = graph()
    pc = e[e["relation"].isin(["partner", "counter"])]
    sub = e[e["relation"] == "substitute"]
    kept = kept_patches(pm())
    pos = {p: i for i, p in enumerate(kept)}
    w = e.drop_duplicates("patch")
    span = {
        pos[last] - pos[f] + 1
        for p, f, last in zip(w["patch"], w["window_first"], w["window_last"])
        if pos[p] >= 6
    }
    before = all(pos[last] < pos[p] for p, last in zip(w["patch"], w["window_last"]))
    return [int(pc["weight"].min()), r(sub["weight"].min(), 2), sorted(span), before]


# ---------------------------------------------------------------- draft order


def draft_rows(game_id: str, orders: list[int]) -> list:
    d = draft()
    d = d[(d["game_id"] == game_id) & d["order"].isin(orders)].sort_values("order")
    return [
        [int(o), t, bool(f), s, a, c]
        for o, t, f, s, a, c in d[["order", "team", "drafts_first", "side", "action", "champion"]].itertuples(
            index=False
        )
    ]


def switched_by_year() -> dict:
    d = draft_raw()
    s = pd.to_numeric(d["SwitchedSides"], errors="coerce").eq(1)
    return {"2025": int(s[d["year"] == 2025].sum()), "2026": int(s[d["year"] == 2026].sum())}


def red_first_drafters() -> dict:
    d = draft()
    first = d[d["drafts_first"] & (d["order"] == 1)]
    year = first["game_id"].map(games().set_index("game_id")["datetime_utc"].dt.year)
    return {"2025": int(((first["side"] == "red") & (year == 2025)).sum())}


def team1_mismatch_2026() -> list:
    d = draft_raw()
    d = d[d["year"] == 2026]
    m = d.merge(cargo_raw()[["GameId", "Team1"]], on="GameId", suffixes=("", "_sg"))
    named = m["Team1"].notna() & (m["Team1"] != "")
    differ = named & (m["Team1"] != m["Team1_sg"])
    switched = pd.to_numeric(m["SwitchedSides"], errors="coerce").eq(1)
    return [int(differ.sum()), bool((differ == switched).all())]


def worlds_final_draft() -> list:
    d = draft_raw()
    row = d[d["GameId"] == "2026 Season World Championship/Main Event_Finals_1_1"].iloc[0]
    return [row["Team1Ban1"], row["Team2Ban1"], row["Team1Pick1"]]


# ---------------------------------------------------------------- eval


def dev_result(model: str) -> dict:
    res = result(f"{model}_dev.json")
    sp = res["spearman"]["all"]
    return {
        "mean": r(sp["mean"]),
        "ci": [r(x) for x in sp["ci"]] if sp["ci"] else None,
        "n_patches": sp["n_patches"],
        "n_constant_patches": sp["n_constant_patches"],
    }


def last_delta_json() -> dict:
    res = result("last_delta_dev.json")
    return {
        **dev_result("last_delta"),
        "missing_rows": res["missing_rows"],
        "extra_rows": res["extra_rows"],
        "risers": r(res["top10_riser_precision"]["mean"]),
        "fallers": r(res["top10_faller_precision"]["mean"]),
        "mae": r(res["mae"]["mean"]),
        "wmae": r(res["weighted_mae"]["mean"]),
    }


def compare_json() -> list:
    res = result("compare_mean_reversion_vs_last_delta_all_dev.json")
    return [r(res["mean_difference"]), r(res["ci"][0]), r(res["ci"][1])]


def n_tests() -> int:
    return sum(
        line.startswith("def test_")
        for f in Path("tests").glob("test_*.py")
        for line in f.read_text().splitlines()
    )


def caitlyn_q() -> float:
    old, new = [50, 90, 130, 170, 210], [50, 95, 140, 185, 230]
    return r(np.mean([n / o - 1 for o, n in zip(old, new)]))


# ---------------------------------------------------------------- single rows and small summaries


def example_row() -> list:
    row = games().set_index("game_id").loc["VCS/2026 Season/Summer Season_Week 3_3_2"]
    return [row["patch"], list(row["picks_blue"]), list(row["bans_blue"])]


def worlds_final_game1() -> list:
    row = games().set_index("game_id").loc["2026 Season World Championship/Main Event_Finals_1_1"]
    return [
        row["datetime_utc"].strftime("%Y-%m-%d %H:%M"),
        row["patch"],
        row["team_blue"],
        row["team_red"],
        row["winner"],
    ]


def dev_rows() -> list:
    d = load_labels(["dev"])
    return [len(d), d["patch"].nunique()]


def keys_2520() -> list:
    d = labels().query("patch == '25.20'")
    return [len(d), int(((d["presence"] == 0) & (d["prev_presence"] == 0)).sum())]


def mid_patch_summary() -> list:
    d = mid_patch_releases().query("games > 0")
    return [len(d), int(d["days"].min()), int(d["days"].max())]


def movers_2520() -> dict:
    d = labels().query("patch == '25.20'")[["champion", "delta_presence"]]
    return {
        "fallers": [[c, r(x)] for c, x in d.nsmallest(4, "delta_presence").to_numpy()],
        "risers": [[c, r(x)] for c, x in d.nlargest(4, "delta_presence").to_numpy()],
    }


def lpl_split3_raw() -> list:
    x = cargo_raw()[cargo_raw()["OverviewPage"] == "LPL/2026 Season/Split 3"]
    return [len(x), int((x["Patch"].isna() | (x["Patch"] == "")).sum())]


# ---------------------------------------------------------------- the claims

G, P, C = GUIDE, PLAN, CHECKS
GP = f"{GUIDE}, {PLAN}"
GPC = f"{GUIDE}, {PLAN}, {CHECKS}"

CLAIMS: list[Claim] = [
    # patches
    Claim(
        "patches.study",
        GP,
        "Overview card '173'; plan 3",
        "173 pro patches with 100+ games in the study",
        173,
        lambda: len(study_patches()),
        "kept patches in patch_map.csv with split train, dev or test",
    ),
    Claim(
        "patches.splits",
        GP,
        "Overview '130 / 24 / 19'; plan 7.3 '154 up to 2025'",
        "Train, dev and test patch counts",
        {"train": 130, "dev": 24, "test": 19},
        lambda: {s: int((study_patches()["split"] == s).sum()) for s in STUDY},
        "count of study patches by split",
    ),
    Claim(
        "patches.games",
        G,
        "Overview card '108,183'; step 3 proof",
        "Pro games on the study patches",
        108183,
        lambda: int(study_patches()["n_pro_games"].sum()),
        "sum of n_pro_games over the study patches",
    ),
    Claim(
        "patches.map",
        GP,
        "6.6 'The file has 194 patches, 174 of them kept'; plan 13",
        "Patch map rows, kept patches, and one kept patch outside the study (26.22)",
        [194, 174, ["26.22"]],
        lambda: [len(pm()), int(pm()["kept"].sum()), sorted(pm().query("kept and split == 'none'")["patch"])],
        "rows, kept rows and kept rows with split none in patch_map.csv",
    ),
    Claim(
        "patches.range",
        G,
        "3 scope 'Patches 9.1 to 26.20'",
        "First and last study patch",
        ["9.1", "26.20"],
        lambda: [min(study_patches()["patch"], key=patch_key), max(study_patches()["patch"], key=patch_key)],
        "min and max study patch by patch_key",
    ),
    Claim(
        "patches.n_games",
        GPC,
        "3 '26.21 has only 25'; 6.6; step 1; plan 3",
        "Games on 26.19, 26.20, 26.21 and 26.22",
        {"26.19": 45, "26.20": 531, "26.21": 25, "26.22": 158},
        lambda: {p: int(pmi().at[p, "n_pro_games"]) for p in ["26.19", "26.20", "26.21", "26.22"]},
        "n_pro_games in patch_map.csv",
    ),
    Claim(
        "patches.dropped",
        G,
        "Step 3 proof 'dropped 2019+'",
        "Patches with games but under 100, and their counts",
        {
            "9.9": 81,
            "9.18": 24,
            "9.20": 47,
            "9.21": 89,
            "9.24b": 46,
            "10.16b": 16,
            "10.17": 11,
            "10.23": 96,
            "10.24": 36,
            "10.25": 33,
            "10.25b": 65,
            "12.17": 44,
            "12.22": 46,
            "13.23": 32,
            "14.24": 18,
            "26.19": 45,
            "26.21": 25,
        },
        dropped_patches,
        "patch_map.csv rows with kept False and n_pro_games > 0",
    ),
    Claim(
        "patches.label_patches",
        G,
        "Step 3 '9.1 has no P', so labels cover 172 patches'",
        "Label patches: 129 train, 24 dev, 19 test",
        {"train": 129, "dev": 24, "test": 19},
        lambda: {
            s: int(all_labels()["patch"].map(pmi()["split"]).eq(s).groupby(all_labels()["patch"]).any().sum())
            for s in STUDY
        },
        "distinct patches in labels.parquet and labels_test.parquet by split",
    ),
    Claim(
        "patches.row_2620",
        G,
        "6.6 patch map table, row 26.20",
        "26.20 patch map row",
        ["V26.20", "16.20", "2026-10-07", "2026-10-06", 531, True, "26.18", "26.19;26.20", "test"],
        lambda: patch_map_row("26.20"),
        "patch_map.csv row (riot_url is empty)",
    ),
    Claim(
        "patches.row_2619",
        G,
        "6.6 patch map table, row 26.19",
        "26.19 patch map row",
        ["V26.19", "16.19", "2026-09-23", "2026-09-22", 45, False, None, None, "none"],
        lambda: patch_map_row("26.19"),
        "patch_map.csv row",
    ),
    Claim(
        "patches.row_25S11",
        G,
        "6.6 patch map table, row 25.S1.1",
        "25.S1.1 patch map row",
        ["V25.S1.1", "15.1", "2025-01-09", "2025-01-08", 315, True, "14.23", "14.24;25.S1.1", "dev"],
        lambda: patch_map_row("25.S1.1"),
        "patch_map.csv row",
    ),
    Claim(
        "patches.cut_2621",
        GP,
        "Step 3 'notes 20 Oct, release 21 Oct'; plan 7.4",
        "26.21 release date and cut",
        ["2026-10-21", "2026-10-20"],
        lambda: [str(pmi().at["26.21", "release_date"].date()), str(pmi().at["26.21", "cut"].date())],
        "release_date and cut in patch_map.csv",
    ),
    Claim(
        "patches.client_version",
        GP,
        "Step 3 table '16.21'; plan 7.4",
        "Client version of 26.21 and 25.S1.1",
        ["16.21", "15.1"],
        lambda: [client_version("26.21"), client_version("25.S1.1")],
        "patches.client_version()",
    ),
    Claim(
        "patches.cut_2520",
        G,
        "6.4b '271 of the 452 games on 25.19'",
        "25.20 release, cut, and P' games before the cut",
        ["2025-10-08", "2025-10-07", 271, 452],
        lambda: [
            str(pmi().at["25.20", "release_date"].date()),
            str(pmi().at["25.20", "cut"].date()),
            int(inputs().query("patch == '25.20'")["prev_n_games"].iloc[0]),
            int(pmi().at["25.19", "n_pro_games"]),
        ],
        "patch_map.csv and prev_n_games in prepatch_stats.parquet",
    ),
    Claim(
        "patches.median_2024",
        GP,
        "11 report fixes 'Median games per patch since 2024 is 590'; plan 3",
        "Median games per study patch from 14.1 on",
        590,
        lambda: int(
            study_patches()[[patch_key(p) >= patch_key("14.1") for p in study_patches()["patch"]]][
                "n_pro_games"
            ].median()
        ),
        "median n_pro_games over study patches with patch_key >= 14.1",
    ),
    Claim(
        "patches.gptilt_only",
        P,
        "Plan 3 'gptilt patches with >= 100 games'",
        "gptilt alone: 173 patches, 108,022 games",
        [173, {"train": 130, "dev": 24, "test": 19}, 108022],
        gptilt_only_patches,
        "raw gptilt games from 2019, patches with 100+ games in a study split",
    ),
    Claim(
        "patches.tier1",
        GP,
        "Plan 2 'Tier-1 only leaves 101 patches and 13 test patches'; guide 12",
        "Study patches with 100+ games from Primary tournaments only, and how many are test",
        [101, 13],
        tier1,
        "Cargo Tournaments.TournamentLevel == Primary, then the 100-game rule",
    ),
    Claim(
        "patches.wiki_dates",
        P,
        "Plan 3 'LoL Wiki infobox gives 192 of 194 patch IDs, in order'",
        "Patch IDs with a release date, all patch IDs, the missing ones, and whether dates increase",
        [192, 194, ["10.16b", "10.25b"], True],
        wiki_dates,
        "wiki_release_dates.csv rows whose patch matches PATCH_ID",
    ),
    # games
    Claim(
        "games.gptilt_raw",
        GP,
        "Plan 3 'gptilt games rows'; guide step 1 '138,402 rows'",
        "gptilt games: total, from 2019, in 2026, last game",
        [138402, 110082, 12102, "2026-10-25 00:34"],
        lambda: [
            len(gptilt_raw()),
            int((gptilt_raw()["datetime_utc"] >= "2019-01-01").sum()),
            int((gptilt_raw()["datetime_utc"] >= "2026-01-01").sum()),
            gptilt_raw()["datetime_utc"].max().strftime("%Y-%m-%d %H:%M"),
        ],
        "row counts and max datetime_utc of gptilt_games.parquet",
    ),
    Claim(
        "games.table",
        G,
        "Step 1 'games.parquet ... with 109,095 games'",
        "games.parquet rows and rows by source",
        [109095, {"gptilt": 96835, "cargo": 12260}],
        lambda: [len(games()), {k: int(v) for k, v in games()["source"].value_counts().items()}],
        "len(games.parquet) and source counts",
    ),
    Claim(
        "games.gptilt_2620",
        GP,
        "Step 1 'only 370 of the 531 games on 26.20'; plan 3",
        "gptilt games on 26.20",
        370,
        lambda: int((gptilt_raw()["patch"] == "26.20").sum()),
        "raw gptilt rows with patch 26.20",
    ),
    Claim(
        "games.overlap",
        GP,
        "Step 1 'On 26.01 to 26.18 the two sources have identical per-patch counts'",
        "Patches 26.01 to 26.18 whose counts differ between gptilt and Cargo, out of all",
        [0, 18],
        overlap_2026,
        "per-patch counts of raw gptilt 2026 rows vs raw Cargo ScoreboardGames",
    ),
    Claim(
        "games.no_patch",
        GP,
        "Step 1 'Games with an empty patch are dropped (1,325 in total, mostly 2025 LDL)'; plan 3",
        "Games with an empty patch",
        {"total": 1325, "2025": 1136, "2025_share": 0.08, "2025_ldl": 901, "2026_cargo": 180},
        no_patch,
        "raw gptilt 2019-2025 and raw Cargo 2026 rows with an empty patch",
    ),
    Claim(
        "games.cargo_2026",
        P,
        "Plan 3 'Cargo ScoreboardGames from 2026-01-01: 12,440 rows'",
        "Cargo 2026 ScoreboardGames rows",
        12440,
        lambda: len(cargo_raw()),
        "len(cargo_games_2026.parquet)",
    ),
    Claim(
        "games.gptilt_2025",
        P,
        "Plan 3 'gptilt 2025 vs Cargo 2025 ... (14,117 games)'",
        "gptilt 2025 games",
        14117,
        lambda: int(gptilt_raw()["datetime_utc"].dt.year.eq(2025).sum()),
        "raw gptilt rows dated 2025 (the Cargo 2025 side is in claim games.cargo_2025)",
    ),
    Claim(
        "games.cargo_2025",
        P,
        "Plan 3 'Per-patch counts identical'",
        "gptilt 2025 per-patch counts equal Cargo 2025",
        "identical",
        None,
        "needs Cargo ScoreboardGames for 2025, which is not in data/raw/pro",
    ),
    Claim(
        "games.example_row",
        G,
        "Step 1 '# one real row'",
        "VCS Summer Week 3 game 3_2: patch, blue picks and bans",
        [
            "26.09",
            ["Renekton", "Pantheon", "Twisted Fate", "Yunara", "Milio"],
            ["Varus", "Nautilus", "Annie", "Naafiri", "Aurora"],
        ],
        example_row,
        "games.parquet row",
    ),
    # cleaning
    Claim(
        "clean.none_bans",
        GP,
        "Step 2 table '3.1% to 5.1% of games each year'; plan 3",
        "Lowest and highest yearly share of games with a 'None' ban, 2019 to 2026",
        [0.031, 0.051],
        none_ban_share_range,
        "raw gptilt 2019-2025 and raw Cargo 2026, share of games with 'None' in either ban list",
    ),
    Claim(
        "clean.missing_data",
        GP,
        "Step 2 table '1 game (10 slots, 25.11)'; plan 3",
        "'Missing Data' games, slots, patches",
        [1, 10, ["25.11"]],
        missing_data,
        "raw gptilt 2019-2025 and raw Cargo 2026",
    ),
    Claim(
        "clean.double",
        GP,
        "Step 2 table 'A champion listed twice ... 45 games'; plan 3",
        "Games that list one champion twice across picks and bans",
        45,
        double_listed,
        "games.parquet, after dropping NOT_A_CHAMPION",
    ),
    Claim(
        "clean.none_2620",
        G,
        "Step 2 'a champion named \"None\" reaches 195% presence on 26.20 (Cargo rows)'",
        "Uncleaned 'None' presence on 26.20 (occurrences per game)",
        1.95,
        none_presence_2620,
        "raw Cargo 26.20 rows (the 2026 source), count of 'None' slots / games",
    ),
    Claim(
        "clean.none_2620_gptilt",
        G,
        "Step 2 'reaches 138% presence on 26.20'",
        "The same figure on the gptilt snapshot (the 2026 source before 17 Nov)",
        1.38,
        none_presence_2620_gptilt,
        "raw gptilt 26.20 rows, count of 'None' slots / games",
    ),
    Claim(
        "clean.champions",
        GP,
        "Step 2 '173 champions in Cargo Champions'",
        "Champions in Cargo Champions",
        173,
        lambda: len(champions()),
        "len(cargo_champions.parquet)",
    ),
    # labels
    Claim(
        "labels.rows",
        GP,
        "Overview '27,624: 24,349 train and dev, 3,275 test'; step 8; plan 7.6",
        "Label rows: total, train and dev, test",
        [27624, 24349, 3275],
        lambda: [len(all_labels()), len(labels()), len(labels_test())],
        "row counts of the two label files",
    ),
    Claim(
        "labels.inputs",
        GP,
        "Step 8 '27,797 (all splits and 26.22)'; plan 7.6",
        "prepatch_stats rows, patches, and keys = label keys + 173 rows of 26.22",
        [27797, 173, True, 173],
        lambda: [
            len(inputs()),
            inputs()["patch"].nunique(),
            set(map(tuple, inputs().query("patch != '26.22'")[["patch", "champion"]].to_numpy()))
            == set(map(tuple, all_labels()[["patch", "champion"]].to_numpy())),
            int((inputs()["patch"] == "26.22").sum()),
        ],
        "prepatch_stats.parquet keys compared with both label files",
    ),
    Claim(
        "labels.dev_rows",
        G,
        "6.5 'dev has 4,092 rows over 24 patches'",
        "Dev rows and patches",
        [4092, 24],
        dev_rows,
        "load_labels(['dev'])",
    ),
    Claim(
        "labels.row_caitlyn",
        G,
        "6.4 table, Caitlyn on 25.20",
        "n_games, presence, prev, delta, fearless, changed, new",
        [425, 0.266, 0.126, 0.140, 0.920, False, False],
        lambda: [
            int(labels().query("patch == '25.20'")["n_games"].iloc[0]),
            *label_row("25.20", "Caitlyn", ["presence", "prev_presence", "delta_presence", "fearless_share"]),
            *[bool(x) for x in label_row("25.20", "Caitlyn", ["changed", "new_champion"])],
        ],
        "labels.parquet row (25.20, Caitlyn)",
    ),
    Claim(
        "labels.row_yunara",
        G,
        "6.4 table, Yunara on 25.20",
        "presence, prev_presence, delta_presence",
        [0.468, 0.343, 0.125],
        lambda: label_row("25.20", "Yunara", ["presence", "prev_presence", "delta_presence"]),
        "labels.parquet row (25.20, Yunara)",
    ),
    Claim(
        "labels.row_rammus",
        G,
        "6.4 table, Rammus on 25.20",
        "presence, prev_presence, delta_presence",
        [0.0, 0.0, 0.0],
        lambda: label_row("25.20", "Rammus", ["presence", "prev_presence", "delta_presence"]),
        "labels.parquet row (25.20, Rammus)",
    ),
    Claim(
        "labels.2520_keys",
        G,
        "6.4 '25.20 has 171 rows, and 4 of them ... drafted on neither patch'",
        "Rows on 25.20 and rows with presence 0 on both patches",
        [171, 4],
        keys_2520,
        "labels.parquet rows on 25.20",
    ),
    Claim(
        "labels.undrafted",
        G,
        "Step 4 'they are 3.4% of rows'",
        "Share of train and dev rows drafted on neither patch",
        0.034,
        lambda: r(((labels()["presence"] == 0) & (labels()["prev_presence"] == 0)).mean()),
        "labels.parquet, presence == 0 and prev_presence == 0",
    ),
    Claim(
        "labels.mid_patch",
        GP,
        "6.4 'Renata Glasc on 12.4, 320 games'; step 4 'Qiyana on 9.13, 323'; plan 3",
        "Games with a pick or ban of a champion on its release patch",
        {"Renata Glasc": 320, "Qiyana": 323, "Yuumi": 160},
        lambda: {c: int(mid_patch_releases().at[c, "games"]) for c in ["Renata Glasc", "Qiyana", "Yuumi"]},
        "games.parquet on the wiki patch live at the champion's release date",
    ),
    Claim(
        "labels.mid_patch_count",
        P,
        "Plan 3 '12 champions went live 1 to 7 days after their patch's release'",
        "Champions released mid-patch and drafted on that patch; min and max days after the patch",
        [12, 1, 7],
        mid_patch_summary,
        "release date vs the release date of the patch then live; drafted = picked or banned on it",
    ),
    Claim(
        "labels.renata_key",
        G,
        "6.4 'Champions released mid-patch (Renata Glasc on 12.4) are included'",
        "Renata Glasc has a label row on 12.4",
        True,
        lambda: bool(((labels()["patch"] == "12.4") & (labels()["champion"] == "Renata Glasc")).any()),
        "labels.parquet",
    ),
    Claim(
        "labels.movers_2520",
        G,
        "Step 4 'largest movers on dev patch 25.20'",
        "Four largest fallers and risers on 25.20",
        {
            "fallers": [["Rumble", -0.179], ["Azir", -0.135], ["Jarvan IV", -0.105], ["Wukong", -0.101]],
            "risers": [["Pantheon", 0.176], ["Caitlyn", 0.140], ["Yunara", 0.125], ["Qiyana", 0.122]],
        },
        movers_2520,
        "labels.parquet on 25.20, sorted by delta_presence",
    ),
    Claim(
        "inputs.row_caitlyn",
        G,
        "6.4b table, Caitlyn on 25.20",
        "prev_n_games, presence, pick, ban, win, role, delta_1, delta_2, fearless share, since debut",
        [271, 0.125, 0.074, 0.052, 0.600, "bot", 0.075, 0.008, 0.849, None],
        lambda: label_row(
            "25.20",
            "Caitlyn",
            [
                "prev_n_games",
                "prev_presence",
                "prev_pick_rate",
                "prev_ban_rate",
                "prev_win_rate",
                "prev_main_role",
                "prev_delta_1",
                "prev_delta_2",
                "prev_fearless_share",
                "patches_since_debut",
            ],
            inputs(),
        ),
        "prepatch_stats.parquet row (25.20, Caitlyn)",
    ),
    Claim(
        "inputs.row_yunara",
        G,
        "6.4b table, Yunara on 25.20",
        "same columns as Caitlyn",
        [271, 0.347, 0.188, 0.159, 0.529, "bot", 0.114, -0.383, 0.849, 5.0],
        lambda: label_row(
            "25.20",
            "Yunara",
            [
                "prev_n_games",
                "prev_presence",
                "prev_pick_rate",
                "prev_ban_rate",
                "prev_win_rate",
                "prev_main_role",
                "prev_delta_1",
                "prev_delta_2",
                "prev_fearless_share",
                "patches_since_debut",
            ],
            inputs(),
        ),
        "prepatch_stats.parquet row (25.20, Yunara)",
    ),
    Claim(
        "inputs.prev_share",
        GP,
        "Step 5 'median share of P' games played before cut(P)'; plan 3",
        "Median share of P' games before the cut, by split",
        {"train": 0.30, "dev": 0.61, "test": 0.57},
        prev_share_by_split,
        "prev_n_games / n_pro_games of P', median over patches",
    ),
    Claim(
        "inputs.thin_train",
        GP,
        "Plan 3 '43 of 129 train patches keep fewer than 100 games, 13 fewer than 50, minimum 10'",
        "Train patches, under 100, under 50, minimum prev_n_games",
        [129, 43, 13, 10],
        thin_train_patches,
        "prev_n_games per train patch in prepatch_stats.parquet",
    ),
    Claim(
        "inputs.cut_effect",
        GP,
        "Step 5 'the cut keeps the signal'; plan 3 'Date-cut inputs keep the signal'",
        "Mean-reversion mean Spearman: [cut inputs, all P' games] for train and dev",
        {"train": [0.212, 0.202], "dev": [0.152, 0.149]},
        cut_effect,
        "harness Spearman of -prev_presence (input vs label) against delta_presence, new-champion rows left out",
    ),
    # new champions
    Claim(
        "newchamp.released",
        GP,
        "9 '31 champions from January 2019 to June 2026: 27 ..., 3 ..., 1'; plan 3",
        "Champions released since 2019 by split of their release patch",
        [27, 3, 1, "Sylas", "Ambessa", ["Mel", "Yunara", "Zaahen"], ["Locke"]],
        release_splits,
        "Cargo Champions.ReleaseDate >= 2019-01-01, wiki patch live on that date",
    ),
    Claim(
        "newchamp.locke_date",
        P,
        "Plan 3 'Locke 2026-06-24'",
        "Locke release date",
        "2026-06-24",
        lambda: str(champions()["Locke"].date()),
        "Cargo Champions.ReleaseDate",
    ),
    Claim(
        "newchamp.rows",
        GP,
        "Step 11 'The 30 champions released from 2019 to 2025 ... 60 in train and dev'; plan 7.6",
        "Champions with the flag, flagged rows, rows per champion (train and dev)",
        [30, 60, [2]],
        new_champion_counts,
        "new_champion == True in labels.parquet; Locke's test rows are checked in the test run",
    ),
    Claim(
        "newchamp.locke",
        P,
        "Plan 7.6 'Locke has patches_since_debut 0, 1, 2 on 26.13, 26.14, 26.15'",
        "Locke's patches_since_debut on 26.13 to 26.15",
        [0, 1, 2],
        locke,
        "prepatch_stats.parquet (inputs only)",
    ),
    Claim(
        "newchamp.mel",
        G,
        "9 table, Mel",
        "Mel presence from the first pro patch",
        [["25.S1.2", 0.029], ["25.S1.3", 0.484], ["25.04", 0.109], ["25.05", 0.060]],
        lambda: presence_path("Mel", 4),
        "labels.parquet, Mel rows with presence > 0, in patch order",
    ),
    Claim(
        "newchamp.yunara",
        G,
        "9 table, Yunara",
        "Yunara presence from the first pro patch",
        [["25.15", 0.520], ["25.16", 0.706], ["25.17", 0.616], ["25.18", 0.233]],
        lambda: presence_path("Yunara", 4),
        "labels.parquet, Yunara rows with presence > 0, in patch order",
    ),
    Claim(
        "newchamp.zaahen",
        G,
        "9 table, Zaahen",
        "Zaahen presence on dev patches",
        [["25.23", 0.028], ["25.24", 0.417]],
        lambda: presence_path("Zaahen", 2),
        "labels.parquet (dev only)",
    ),
    Claim(
        "newchamp.aurora",
        G,
        "9 table, Aurora",
        "Aurora presence from the first pro patch",
        [["14.15", 0.515], ["14.16", 0.332], ["14.17", 0.380], ["14.18", 0.588]],
        lambda: presence_path("Aurora", 4),
        "labels.parquet, Aurora rows with presence > 0, in patch order",
    ),
    Claim(
        "newchamp.jumps",
        G,
        "9 'Mel's +0.455 jump on 25.S1.3'; 'Yunara was released on 25.14'; '(0.706, +0.186)'",
        "Mel delta on 25.S1.3, Yunara release patch, Yunara presence and delta on 25.16",
        [0.455, "25.14", 0.706, 0.186],
        lambda: [
            label_row("25.S1.3", "Mel", ["delta_presence"])[0],
            release_patch(champions()["Yunara"]),
            *label_row("25.16", "Yunara", ["presence", "delta_presence"]),
        ],
        "labels.parquet and the wiki patch live on Yunara's release date",
    ),
    Claim(
        "newchamp.percentiles",
        GP,
        "Plan 3 'Median within-patch percentile of |Δpresence|'; guide 9 '94th percentile'",
        "Median percentile of |Δpresence| on pro patch k = 1..4 (train and dev), of |prev_delta_1| on k = 3",
        {
            "k1": 0.78,
            "k2": 0.95,
            "k3": 0.76,
            "k4": 0.85,
            "k3_prev_delta_1": 0.94,
            "n_champions": 30,
            "n_k4": 29,
        },
        debut_percentiles,
        "rank(pct=True) of |value| within each patch, median over champions; k-th patches in test are skipped",
    ),
    # Fearless
    Claim(
        "fearless.series_by_year",
        GP,
        "Step 6 'multi-game series with no repeated pick, by year'; plan 3, 7.5",
        "Share of multi-game series with no repeated pick, by year",
        {
            2019: 0.011,
            2020: 0.013,
            2021: 0.011,
            2022: 0.025,
            2023: 0.037,
            2024: 0.175,
            2025: 0.948,
            2026: 0.998,
        },
        lambda: fearless_by_year()["series_no_repeat"],
        "series_fearless() over games.parquet, by first-game year",
    ),
    Claim(
        "fearless.games_by_year",
        G,
        "Step 6 'share of games flagged Fearless (tournament level, threshold 0.9)'",
        "Share of games flagged Fearless, by year",
        {2019: 0.0, 2020: 0.0, 2021: 0.0, 2022: 0.0, 2023: 0.008, 2024: 0.097, 2025: 0.811, 2026: 0.940},
        lambda: fearless_by_year()["games_flagged"],
        "mean of games.parquet `fearless` by year",
    ),
    Claim(
        "fearless.label_share",
        G,
        "6.4 fearless_share 0.920; 6.4b prev_fearless_share 0.849",
        "fearless_share on 25.20 and prev_fearless_share for 25.20",
        [0.920, 0.849],
        lambda: [
            label_row("25.20", "Caitlyn", ["fearless_share"])[0],
            label_row("25.20", "Caitlyn", ["prev_fearless_share"], inputs())[0],
        ],
        "labels.parquet and prepatch_stats.parquet",
    ),
    Claim(
        "fearless.games_flagged",
        GPC,
        "Checks 'Effect of fixing them' table; plan 3; guide 12",
        "Impact of the current rule A: games flagged, patches with share >= 0.5 by split",
        {"games": 23577, "train": "2/130", "dev": "23/24", "test": "19/19"},
        lambda: impact("A"),
        "games.parquet `fearless` share per kept patch, rule A recomputed",
    ),
    Claim(
        "fearless.rule_b",
        C,
        "Checks impact table 'Plus reset decider'",
        "Rule B impact",
        {
            "games": 23969,
            "train": "2/130",
            "dev": "24/24",
            "test": "19/19",
            "moved": 5,
            "moved_by_split": {"train": 4, "dev": 1},
        },
        lambda: impact("B"),
        "rule A or (hard share without the last game >= 0.9 over >= 3 series)",
    ),
    Claim(
        "fearless.rule_c",
        GPC,
        "Checks impact table 'Plus reset decider and soft'; guide 12 '33 patches'; plan 13",
        "Rule C impact",
        {
            "games": 26872,
            "train": "4/130",
            "dev": "24/24",
            "test": "19/19",
            "moved": 33,
            "moved_by_split": {"train": 32, "dev": 1},
        },
        lambda: impact("C"),
        "rule B or (soft share >= 0.9 over >= 5 series)",
    ),
    Claim(
        "fearless.reset_decider",
        GPC,
        "Checks gap 1 '5 tournaments and 392 games'; guide step 6; plan 3",
        "Tournaments and games added by the reset-decider case",
        {"tournaments": 5, "games": 392},
        lambda: {k: v for k, v in extension("B", "A").items() if k != "games_after_2024"},
        "tournaments flagged by rule B and not by rule A",
    ),
    Claim(
        "fearless.soft",
        GPC,
        "Checks gap 2 '29 tournaments (2,903 games, 15 of them after 2024)'; guide step 6",
        "Tournaments and games added by soft Fearless, and games dated 2025 or later",
        {"tournaments": 29, "games": 2903, "games_after_2024": 15},
        lambda: extension("C", "B"),
        "tournaments flagged by rule C and not by rule B",
    ),
    Claim(
        "fearless.2524",
        GPC,
        "Checks gap 1 '98 of its 132 games are Demacia Cup 2025 ... 0.167 to about 0.91'",
        "Demacia Cup games on 25.24, games on 25.24, share under rule A and rule B",
        [98, 132, 0.167, 0.91],
        demacia_2524,
        "games.parquet on 25.24",
    ),
]

SPOT = [
    (
        "01",
        "lck-cl-2024-summer",
        "LCK CL/2024 Season/Summer Season",
        "Fearless; hard 90/90; soft 90/90; without last 43/43",
    ),
    ("02", "first-stand-2026", "2026 First Stand", "Fearless; hard 13/13; soft 13/13; without last 13/13"),
    (
        "03",
        "prime-pokal-2024",
        "Prime Pokal 2024/Main Event",
        "Fearless; hard 14/15; soft 14/15; without last 6/6",
    ),
    (
        "04",
        "asia-masters-2025",
        "2025 Asia Masters/Finals",
        "Fearless; hard 24/24; soft 24/24; without last 14/14",
    ),
    (
        "05",
        "emea-masters-2025-summer",
        "EMEA Masters/2025 Season/Summer Main Event",
        "Fearless; hard 41/41; soft 41/41; without last 26/26",
    ),
    (
        "06",
        "lck-2024-summer",
        "LCK/2024 Season/Summer Season",
        "Standard; hard 0/90; soft 10/90; without last 0/29",
    ),
    (
        "07",
        "ldl-2024-grand-finals",
        "LDL/2024 Season/Grand Finals",
        "Standard; hard 12/18; soft 12/18; without last 18/18",
    ),
    ("08", "demacia-cup-2025", "Demacia Cup 2025", "Standard; hard 26/38; soft 27/38; without last 17/17"),
    (
        "09",
        "ldl-2022-summer",
        "LDL/2022 Season/Summer Season",
        "Standard; hard 13/112; soft 112/112; without last 9/39",
    ),
    (
        "10",
        "nacl-2024-spring",
        "North American Challengers League/2024 Season/Spring Season",
        "Standard; hard 8/45; soft 45/45; without last 4/22",
    ),
]
for num, slug, name, expected in SPOT:
    CLAIMS.append(
        Claim(
            f"fearless.spot_{num}",
            C,
            f"Checks 2 table row {int(num)}",
            f"Flag and series counts for {name}",
            expected,
            lambda name=name: tournament_row(name),
            "series counts from games.parquet (hard, soft, hard without each series' last game)",
        )
    )
    CLAIMS.append(
        Claim(
            f"fearless.page_{num}",
            C,
            f"Checks 2 table row {int(num)}, 'Leaguepedia page says'",
            f"Leaguepedia page text on the draft mode of {name}",
            "see screenshot",
            None,
            f"{IMG}fearless-{num}-{slug}.png",
        )
    )

SIDES = [
    ("1", "worlds-main-event", "2026 Season World Championship/Main Event", 76),
    ("2", "lck-rounds-1-2", "LCK/2026 Season/Rounds 1-2", 204),
    ("3", "lpl-split-3", "LPL/2026 Season/Split 3", 165),
    ("4", "lec-versus-playoffs", "LEC/2026 Season/Versus Playoffs", 39),
    ("5", "vcs-summer", "VCS/2026 Season/Summer Season", 34),
]
for num, slug, name, n in SIDES:
    CLAIMS.append(
        Claim(
            f"sides.data_{num}",
            C,
            f"Checks 1 table row {num}",
            f"Games of {name} in games.parquet",
            n,
            lambda name=name: int((games()["tournament"] == name).sum()),
            "games.parquet rows of the tournament",
        )
    )
    CLAIMS.append(
        Claim(
            f"sides.page_{num}",
            GPC,
            f"Checks 1 table row {num}; '514 of 514 comparable games match'",
            f"Blue and red teams on the Match History page of {name} equal team_blue and team_red",
            "see screenshot",
            None,
            f"{IMG}side-{num}-{slug}.png",
        )
    )

CLAIMS += [
    Claim(
        "sides.lpl_no_patch",
        C,
        "Checks 1 'LPL 2026 Split 3 ... 2 games with an empty patch'",
        "LPL 2026 Split 3 rows in raw Cargo and those with an empty patch",
        [167, 2],
        lpl_split3_raw,
        "raw Cargo ScoreboardGames",
    ),
    Claim(
        "sides.final_g1",
        C,
        "Checks 1 'scoreboard of the Worlds 2026 final, game 1'",
        "Date, patch, blue, red and winner of the Worlds 2026 final game 1",
        ["2026-11-14 19:53", "26.20", "Gen.G", "Bilibili Gaming", "Gen.G"],
        worlds_final_game1,
        "games.parquet row",
    ),
    Claim(
        "sides.final_g1_page",
        C,
        "Checks 1 'Gen.G is in the left column ... side-blue'",
        "Gen.G is the blue column on the Worlds final game 1 scoreboard",
        "see screenshot",
        None,
        f"{IMG}side-1-worlds-final-g1.png",
    ),
    # relation graph
    Claim(
        "graph.size",
        GP,
        "6.6 '1.83 million edges over 173 forecast patches'; plan 13; plan 7.7 'about 1.8 million'",
        "Edges (millions) and forecast patches",
        [1.83, 173],
        lambda: [r(len(graph()) / 1e6, 2), graph()["patch"].nunique()],
        "len and distinct patches of relation_graph",
    ),
    Claim(
        "graph.rules",
        G,
        "Step 9 table 'weight >= 5', 'overlap >= 0.5'; 6.6 '6 kept patches before P'",
        "Smallest partner/counter weight, smallest substitute overlap, window spans, window ends before P",
        [5, 0.5, [6], True],
        graph_rules,
        "relation_graph.parquet, window span counted in kept patches",
    ),
    Claim(
        "graph.bard",
        G,
        "6.6 graph rows, Bard",
        "Caitlyn-Bard partner on 26.20: weight, share, lift, window",
        [237, 0.417, 5.27, "26.13", "26.18", 4190],
        lambda: graph_row("Bard", "partner"),
        "relation_graph.parquet",
    ),
    Claim(
        "graph.karma",
        G,
        "6.6 graph rows, Karma",
        "Caitlyn-Karma partner on 26.20",
        [134, 0.236, 3.65, "26.13", "26.18", 4190],
        lambda: graph_row("Karma", "partner"),
        "relation_graph.parquet",
    ),
    Claim(
        "graph.ezreal",
        G,
        "6.6 graph rows, Ezreal",
        "Caitlyn-Ezreal counter on 26.20",
        [105, 0.185, 1.49, "26.13", "26.18", 4190],
        lambda: graph_row("Ezreal", "counter"),
        "relation_graph.parquet",
    ),
    Claim(
        "graph.aphelios",
        G,
        "6.6 graph rows, Aphelios",
        "Caitlyn-Aphelios substitute on 26.20",
        [0.996, 0.996, None, "26.13", "26.18", 4190],
        lambda: graph_row("Aphelios", "substitute"),
        "relation_graph.parquet",
    ),
    Claim(
        "graph.top",
        GP,
        "Step 9 proof; plan 7.7 checks",
        "Caitlyn's top 3 partners and counters on 26.20",
        {
            "partner": [["Bard", 237], ["Karma", 134], ["Ryze", 78]],
            "counter": [["Ezreal", 105], ["Lucian", 83], ["Jhin", 79]],
        },
        lambda: {"partner": top("partner", 3), "counter": top("counter", 3)},
        "relation_graph.parquet sorted by weight",
    ),
    Claim(
        "graph.lift",
        GP,
        "Step 9 'highest lift: Elise 8.3 (13 games), Lux 7.7 (38 games)'; plan 7.7",
        "Caitlyn's two highest-lift partners on 26.20",
        [["Elise", 8.3, 13], ["Lux", 7.7, 38]],
        lambda: top("partner", 2, by="lift"),
        "relation_graph.parquet sorted by lift",
    ),
    Claim(
        "graph.ties",
        GP,
        "Step 9 'Caitlyn on 26.20 has 13 bot laners tied at 0.996'; plan 7.7",
        "Caitlyn substitutes on 26.20 at overlap 0.996",
        13,
        lambda: int((caitlyn_2620().query("relation == 'substitute'")["weight"].round(3) == 0.996).sum()),
        "relation_graph.parquet, substitute weight rounded to 0.996",
    ),
    Claim(
        "graph.tie_share",
        GP,
        "Step 9 'more than 15 substitutes tie at the top overlap for 9% of champion-patch pairs (up to 45)'",
        "[share of (patch, champion) pairs with more than 15 substitutes tied at the top overlap, largest tie]",
        [0.09, 45],
        top_ties,
        "relation_graph.parquet substitutes, weight rounded to 3 decimals, ties at each pair's maximum",
        tol=0.005,
    ),
    Claim(
        "graph.tie_names",
        G,
        "Step 9 'substitutes ... (Aphelios, Twitch, Xayah, ...)'",
        "Aphelios, Twitch and Xayah are at 0.996",
        True,
        lambda: (
            {"Aphelios", "Twitch", "Xayah"}
            <= set(
                caitlyn_2620()
                .query("relation == 'substitute'")
                .loc[lambda d: d["weight"].round(3) == 0.996, "neighbour"]
            )
        ),
        "relation_graph.parquet",
    ),
    # draft order
    Claim(
        "draft.example",
        G,
        "6.6 draft order rows",
        "Worlds 2025 final game 1, draft orders 1, 2, 7, 8",
        [
            [1, "KT Rolster", True, "blue", "ban", "Bard"],
            [2, "T1", False, "red", "ban", "Azir"],
            [7, "KT Rolster", True, "blue", "pick", "Wukong"],
            [8, "T1", False, "red", "pick", "Taliyah"],
        ],
        lambda: draft_rows("2025 Season World Championship/Main Event_Finals_1_1", [1, 2, 7, 8]),
        "draft.parquet",
    ),
    Claim(
        "draft.switched",
        GP,
        "6.6 'No 2025 game has a red-side first drafter, but 3,090 games in 2026 do'; plan 3, 7.8",
        "PicksAndBansS7 games with SwitchedSides = 1",
        {"2025": 0, "2026": 3090},
        switched_by_year,
        "raw cargo_draft, SwitchedSides == 1 by year",
    ),
    Claim(
        "draft.red_first",
        G,
        "6.6 'No 2025 game has a red-side first drafter, but 3,090 games in 2026 do'",
        "2025 games whose first drafter is on red in draft.parquet",
        {"2025": 0},
        red_first_drafters,
        "draft.parquet, order 1 rows with side red in 2025",
    ),
    Claim(
        "draft.team1_2026",
        P,
        "Plan 3 'Differs in 3,090 of 12,440 games in 2026, all with SwitchedSides=1'",
        "2026 games where draft Team1 differs from ScoreboardGames Team1, and whether exactly those are switched",
        [3090, True],
        team1_mismatch_2026,
        "raw cargo_draft 2026 joined to raw ScoreboardGames 2026 on GameId; blank draft Team1 left out",
    ),
    Claim(
        "draft.team1_before",
        P,
        "Plan 3 'before 2026 ... in 3 to 107 games a year, true side swaps in 0 to 49'",
        "Draft rows whose Team1 is not games.parquet's blue team, and the true side swaps, by year",
        {
            "differ": {2019: 3, 2020: 55, 2021: 19, 2022: 27, 2023: 89, 2024: 77, 2025: 107},
            "side_swap": {2019: 2, 2020: 49, 2021: 15, 2022: 10, 2023: 26, 2024: 0, 2025: 0},
        },
        team1_name_mismatch_by_year,
        "raw cargo_draft 2019-2025 joined to games.parquet on GameId; blank Team1 left out",
    ),
    Claim(
        "draft.rows",
        P,
        "Plan 7.1 step 4 'The pull returned 87,818 rows for 2019 to 2026'",
        "Draft-order rows downloaded (2019 to 2026)",
        87818,
        lambda: len(draft_raw()),
        "rows of raw cargo_draft/*.parquet",
    ),
    Claim(
        "draft.worlds_2026",
        G,
        "Step 10 '# result'",
        "Worlds 2026 final game 1: Team1Ban1, Team2Ban1, Team1Pick1",
        ["Nocturne", "Qiyana", "Lee Sin"],
        worlds_final_draft,
        "raw cargo_draft 2026",
    ),
    Claim(
        "cargo.client",
        G,
        "Step 10 'waits 8 seconds ... waits 2 minutes and retries, up to 10 times'; 'at most 500 rows'",
        "Cargo delay (s), rate-limit wait (s), rate-limit retries, page size",
        [8.0, 120.0, 10, 500],
        lambda: [cargo.DELAY_S, cargo.RATE_LIMIT_WAIT_S, cargo.RATE_LIMIT_RETRIES, cargo.PAGE],
        "constants in src/patch2meta/pro/cargo.py",
    ),
    Claim(
        "draft.red_first_by_year",
        C,
        "Section 1 'Among drafts that list champions, the first drafter is on the red side in 3,054 of 11,108 games in 2026, in 2 games in 2019 and in none from 2020 to 2025'",
        "Games whose first drafter is on red, by year",
        {2019: 2, 2020: 0, 2021: 0, 2022: 0, 2023: 0, 2024: 0, 2025: 0, 2026: 3054},
        red_first_by_year,
        "draft.parquet order 1 rows with side red, joined to games.parquet for the year",
    ),
    Claim(
        "cargo.plan_retries",
        P,
        "Plan 7.1 step 2 'on ratelimited waits 2 minutes before retrying (up to 10 retries)'",
        "Rate-limit retries",
        10,
        lambda: cargo.RATE_LIMIT_RETRIES,
        "RATE_LIMIT_RETRIES in src/patch2meta/pro/cargo.py",
    ),
    # eval
    Claim(
        "eval.last_delta",
        G,
        "6.6 'Eval results' JSON",
        "Dev results of last_delta",
        {
            "mean": -0.146,
            "ci": [-0.206, -0.082],
            "n_patches": 24,
            "n_constant_patches": 0,
            "missing_rows": 0,
            "extra_rows": 0,
            "risers": 0.167,
            "fallers": 0.154,
            "mae": 0.044,
            "wmae": 0.023,
        },
        last_delta_json,
        "data/results/last_delta_dev.json",
    ),
    Claim(
        "eval.no_change",
        GP,
        "8.3 '0 (24 constant patches, counted)'; plan 3",
        "Dev result of no_change",
        {"mean": 0.0, "ci": [0.0, 0.0], "n_patches": 24, "n_constant_patches": 24},
        lambda: dev_result("no_change"),
        "data/results/no_change_dev.json",
    ),
    Claim(
        "eval.mean_reversion",
        GP,
        "8.3 '+0.152 (0.115 to 0.190)'; plan 3",
        "Dev mean Spearman of mean_reversion and CI",
        [0.152, [0.115, 0.190]],
        lambda: [dev_result("mean_reversion")["mean"], dev_result("mean_reversion")["ci"]],
        "data/results/mean_reversion_dev.json",
    ),
    Claim(
        "eval.dev_recomputed",
        GP,
        "8.3 dev Spearman of mean_reversion and last_delta; plan 3",
        "Dev mean Spearman recomputed from labels and inputs, without data/results/ or the eval code",
        {"mean_reversion": 0.152, "last_delta": -0.146},
        dev_spearman_recomputed,
        "own average-rank correlation per dev patch, new-champion rows left out, constant patches score 0",
    ),
    Claim(
        "eval.paired",
        G,
        "8.3 '+0.298 (0.224 to 0.370)'",
        "Mean reversion minus last delta on dev, with CI",
        [0.298, 0.224, 0.370],
        compare_json,
        "data/results/compare_mean_reversion_vs_last_delta_all_dev.json",
    ),
    Claim(
        "eval.ceiling",
        GP,
        "8.4 'On dev it is 0.895 (all champions)'; '200 times'; plan 3, 13",
        "Dev noise ceiling (all) and resamples",
        [0.895, 200],
        lambda: [
            r(result("noise_ceiling_dev.json")["mean"]["all"]),
            result("noise_ceiling_dev.json")["n_boot"],
        ],
        "data/results/noise_ceiling_dev.json",
    ),
    Claim(
        "eval.constants",
        G,
        "8.2 table; 8.2 'bootstrapping patches 2,000 times'; 8.4 '200 times'",
        "Bootstrap draws, ceiling resamples, top-k, variance floor, minimum rows per patch",
        [2000, 200, 10, 1e-4, 3],
        lambda: [bootstrap.N_BOOT, ceiling.N_BOOT, metrics.TOP_K, metrics.VAR_FLOOR, metrics.MIN_ROWS],
        "constants in src/patch2meta/eval/",
    ),
    Claim(
        "eval.tests",
        GP,
        "Step 0 '48 tests on toy data'; plan 13 '48 unit tests'",
        "Unit tests",
        48,
        n_tests,
        "count of `def test_` in tests/test_*.py",
    ),
    # other numbers in the guide
    Claim(
        "notes.caitlyn_q",
        G,
        "6.2 'The Q value 0.063 is the mean over ranks of new/old - 1'",
        "Mean of new/old - 1 over the five Q ranks",
        0.063,
        caitlyn_q,
        "arithmetic on the quoted numbers",
    ),
    Claim(
        "notes.wayback",
        G,
        "10 'Wayback Machine calendar ... 6 captures from 21 to 27 Oct 2026'",
        "Wayback captures of the 26.21 notes page",
        "6 captures, 21 to 27 Oct 2026",
        None,
        "web page (Wayback calendar screenshot in the guide); not pro data",
    ),
]


# ---------------------------------------------------------------- running


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def same(expected, actual, tol: float) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(same(expected[k], actual[k], tol) for k in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(same(e, a, tol) for e, a in zip(expected, actual))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tol + 1e-9
    return expected == actual


def run(claims: list[Claim]) -> list[dict]:
    out = []
    for c in claims:
        expected = plain(c.expected)
        if c.actual is None:
            actual, status = None, "MANUAL"
        else:
            try:
                actual = plain(c.actual())
                status = "PASS" if same(expected, actual, c.tol) else "FAIL"
            except (KeyError, IndexError, ValueError, FileNotFoundError) as e:  # a failed check, not a crash
                actual, status = f"error: {e!r}", "FAIL"
        out.append(
            {
                "id": c.id,
                "doc": c.doc,
                "locator": c.locator,
                "claim": c.claim,
                "expected": expected,
                "actual": actual,
                "status": status,
                "how": c.how,
                "tol": c.tol,
            }
        )
    return out


def short(x, width: int = 48) -> str:
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s if len(s) <= width else s[: width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, help="write the results as a JSON list")
    args = ap.parse_args()
    rows = run(CLAIMS)
    for row in rows:
        print(f"{row['status']:<6}  {row['id']:<26}  expected {short(row['expected'])}")
        if row["status"] != "PASS":
            print(f"{'':<34}actual   {short(row['actual']) if row['actual'] is not None else row['how']}")
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    print(f"\n{len(rows)} claims: {counts}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"wrote {args.json}")
    sys.exit(1 if counts.get("FAIL") else 0)


if __name__ == "__main__":
    main()
