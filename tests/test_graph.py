import pandas as pd
from conftest import game

from patch2meta.pro.counts import build
from patch2meta.pro.graph import build_graph
from patch2meta.pro.patches import build_patch_map, patch_key


def graph(toy_games, toy_wiki, toy_release, window=6, min_games=1):
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    pm["cut"] = pd.to_datetime(pm["cut"])
    t = build(toy_games, list(toy_release.index))
    return build_graph(t, pm, window, min_games)


def test_window_never_reaches_the_forecast_patch(toy_games, toy_wiki, toy_release):
    g = graph(toy_games, toy_wiki, toy_release)
    assert (g["window_last"].map(patch_key) < g["patch"].map(patch_key)).all()


def test_games_after_the_cut_are_left_out(toy_games, toy_wiki, toy_release):
    g = graph(toy_games, toy_wiki, toy_release)
    # For 9.2 the window is 9.1 before 23 Jan: g1 and g2. g4 (L top with B) comes later.
    e = g[g["patch"] == "9.2"]
    assert e["n_games_window"].iloc[0] == 2
    assert e[(e["champion"] == "L") & (e["relation"] == "partner")].empty
    ab = e[(e["champion"] == "A") & (e["neighbour"] == "B") & (e["relation"] == "partner")]
    assert ab["weight"].item() == 2
    assert ab["share"].item() == 1.0
    af = e[(e["champion"] == "A") & (e["neighbour"] == "F") & (e["relation"] == "counter")]
    assert af["weight"].item() == 2


def test_substitutes_share_a_role(toy_games, toy_wiki, toy_release):
    g = graph(toy_games, toy_wiki, toy_release)
    e = g[(g["patch"] == "9.6") & (g["champion"] == "N") & (g["relation"] == "substitute")]
    # N plays top only; A, F and L are the only other champions picked in the top slot
    assert set(e["neighbour"]) == {"A", "F", "L"}
    assert e.set_index("neighbour").loc["F", "weight"] == 1.0


def test_window_keeps_only_the_last_kept_patches(toy_games, toy_wiki, toy_release):
    g = graph(toy_games, toy_wiki, toy_release, window=2)
    e = g[g["patch"] == "9.6"]
    assert (e["window_first"].iloc[0], e["window_last"].iloc[0]) == ("9.4", "9.5")
    assert e["n_games_window"].iloc[0] == 4  # g7 and g8 on 9.4, g9 and g10 on 9.5


def test_lift_corrects_for_popularity(toy_games, toy_wiki, toy_release):
    g = graph(toy_games, toy_wiki, toy_release)
    ab = g.query("patch == '9.2' and champion == 'A' and neighbour == 'B' and relation == 'partner'")
    # 2 games give 4 team drafts; A and B are picked twice each, always together: 2 * 4 / (2 * 2)
    assert ab["lift"].item() == 2.0


def test_games_on_the_forecast_patch_never_enter_its_window(toy_games, toy_wiki, toy_release):
    early = game("early", "mE", "T", "2019-01-20", "9.2", "KBCDE", "FGHIJ")  # a 9.2 game before cut(9.2)
    g = graph(pd.concat([toy_games, pd.DataFrame([early])], ignore_index=True), toy_wiki, toy_release)
    assert g.loc[g["patch"] == "9.2", "n_games_window"].iloc[0] == 2


def test_a_champion_on_both_teams_is_not_its_own_neighbour(toy_games, toy_wiki, toy_release):
    mirror = [game(f"m{i}", f"mm{i}", "T", "2019-01-12", "9.1", "ABCDE", "AGHIJ") for i in range(3)]
    g = graph(pd.concat([toy_games, pd.DataFrame(mirror)], ignore_index=True), toy_wiki, toy_release)
    assert (g["champion"] != g["neighbour"]).all()
