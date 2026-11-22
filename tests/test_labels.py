"""Labels and pre-patch statistics on the toy games in conftest.py."""

import argparse
import json

import pandas as pd
import pytest
from conftest import CHAMPS, game

from patch2meta.pro import labels as labels_mod
from patch2meta.pro.counts import build
from patch2meta.pro.labels import build_rows, changed_pairs
from patch2meta.pro.names import canonical
from patch2meta.pro.patches import build_patch_map


@pytest.fixture
def rows(toy_games, toy_wiki, toy_release):
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    pm["release_date"] = pd.to_datetime(pm["release_date"])
    pm["cut"] = pd.to_datetime(pm["cut"])
    t = build(toy_games, list(toy_release.index))
    labels, inputs = build_rows(t, pm, toy_release, changed={("9.3", "A")})
    return labels.set_index(["patch", "champion"]), inputs.set_index(["patch", "champion"])


def test_keys_cover_every_released_champion(rows):
    labels, inputs = rows
    assert labels.index.equals(inputs.index)
    # N goes live on 10 Feb, during 9.3 (7 to 21 Feb): a key from 9.3 on, not on 9.2
    assert ("9.2", "N") not in labels.index
    assert ("9.3", "N") in labels.index
    assert inputs.loc[("9.3", "N"), "patches_since_debut"] == 0
    # K is never picked or banned on 9.3 or 9.4 but still has a row with delta 0
    assert labels.loc[("9.4", "K"), "delta_presence"] == 0.0


def test_target_uses_all_games_and_counts_a_champion_once_per_game(rows):
    labels, _ = rows
    # 9.2: A picked in g3 and g3b (2 of 2 games); P' = 9.1 with g1, g2, g4: A picked in g1, g2 -> 2/3
    assert labels.loc[("9.2", "A"), "presence"] == 1.0
    assert labels.loc[("9.2", "A"), "prev_presence"] == pytest.approx(2 / 3)
    # 9.3: A picked in g5, picked and banned in g6 -> presence 1.0, not 1.5
    assert labels.loc[("9.3", "A"), "presence"] == 1.0
    assert labels.loc[("9.3", "A"), "changed"]
    assert not labels.loc[("9.3", "B"), "changed"]


def test_inputs_ignore_games_after_the_cut(rows):
    labels, inputs = rows
    # g4 (9.1, 26 Jan) is after cut(9.2) = 23 Jan: the input sees only g1, g2
    assert inputs.loc[("9.2", "L"), "prev_n_games"] == 2
    assert inputs.loc[("9.2", "L"), "prev_presence"] == 0.5  # banned in g1 only
    assert labels.loc[("9.2", "L"), "prev_presence"] == pytest.approx(2 / 3)  # g1 ban + g4 pick
    assert inputs.loc[("9.2", "A"), "prev_win_rate"] == 1.0
    assert inputs.loc[("9.2", "A"), "prev_main_role"] == "top"
    assert inputs.loc[("9.2", "K"), "prev_ban_rate"] == 1.0


def test_previous_deltas(rows):
    _, inputs = rows
    # On 9.4: P' = 9.3 (A presence 1.0), P'' = 9.2 (A 1.0), P''' = 9.1 before cut(9.4) (A in g1, g2, g4 -> 1.0)
    assert inputs.loc[("9.4", "A"), "prev_delta_1"] == 0.0
    assert (
        pd.isna(inputs.loc[("9.2", "A"), "prev_delta_1"]) or inputs.loc[("9.2", "A"), "prev_delta_1"] is None
    )
    # On 9.3: L is absent on 9.2 and on 9.1 (before cut(9.3), so g4 counts) it is in g1 and g4
    assert inputs.loc[("9.3", "L"), "prev_delta_1"] == pytest.approx(0.0 - 2 / 3)
    # On 9.4: prev_delta_2 = L on P'' = 9.2 (absent) minus L on P''' = 9.1 (g1 and g4 of 3 games)
    assert inputs.loc[("9.4", "L"), "prev_delta_2"] == pytest.approx(0.0 - 2 / 3)


def test_a_game_exactly_at_the_cut_is_not_an_input(toy_games, toy_wiki, toy_release):
    at_cut = game("atcut", "mc", "T", "2019-01-23 00:00", "9.1", "LBCDE", "FGHIJ")  # cut(9.2) = 23 Jan 00:00
    g = pd.concat([toy_games, pd.DataFrame([at_cut])], ignore_index=True)
    pm = build_patch_map(g, toy_wiki, min_games=2)
    pm["release_date"], pm["cut"] = pd.to_datetime(pm["release_date"]), pd.to_datetime(pm["cut"])
    _, inputs = build_rows(build(g, list(toy_release.index)), pm, toy_release, changed=None)
    assert inputs.set_index(["patch", "champion"]).loc[("9.2", "L"), "prev_n_games"] == 2


def test_run_writes_test_rows_only_to_the_private_file(
    tmp_path, monkeypatch, toy_games, toy_wiki, toy_release
):
    monkeypatch.chdir(tmp_path)
    data, private = tmp_path / "data/processed", tmp_path / "data/private"
    data.mkdir(parents=True)
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    pm.loc[pm["patch"].isin(["9.5", "9.6"]), "split"] = "test"
    pm.to_csv(data / "patch_map.csv", index=False)
    toy_games.to_parquet(data / "games.parquet", index=False)
    champions = pd.DataFrame({"Name": toy_release.index, "ReleaseDate": toy_release.dt.strftime("%Y-%m-%d")})
    monkeypatch.setattr(labels_mod, "CHAMPIONS", tmp_path / "champions.parquet")
    champions.to_parquet(tmp_path / "champions.parquet", index=False)
    labels_mod.run(argparse.Namespace(data=data, private=private, allow_unknown_blocks=False))
    shared = pd.read_parquet(data / "labels.parquet")
    held = pd.read_parquet(private / "labels_test.parquet")
    assert set(shared["patch"]) == {"9.2", "9.3", "9.4"}
    assert set(held["patch"]) == {"9.5", "9.6"}
    assert len(pd.read_parquet(data / "prepatch_stats.parquet")) == len(shared) + len(held)


def write_changes(path, blocks):
    path.write_text("".join(json.dumps(b) + "\n" for b in blocks))


def block(patch: str, name: str, kind: str = "champion", gameplay: bool = True) -> dict:
    return {"patch": patch, "kind": kind, "name": name, "gameplay": gameplay}


def test_changed_uses_every_riot_patch_since_p_prime(tmp_path, monkeypatch, toy_games, toy_wiki):
    monkeypatch.setattr(labels_mod, "CHANGES", tmp_path / "changes.jsonl")
    write_changes(tmp_path / "changes.jsonl", [
        block("9.5b", "A"),  # 9.5b is not kept, so it counts for 9.6
        block("9.6", "Infinity Edge", kind="item"),
    ])  # fmt: skip
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    assert changed_pairs(pm, known=frozenset(CHAMPS)) == {("9.6", "A")}


def test_changed_counts_only_gameplay_blocks(tmp_path, monkeypatch, toy_games, toy_wiki):
    monkeypatch.setattr(labels_mod, "CHANGES", tmp_path / "changes.jsonl")
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    write_changes(tmp_path / "changes.jsonl", [block("9.6", "A"), block("9.6", "B", gameplay=False)])
    assert changed_pairs(pm, known=frozenset(CHAMPS)) == {("9.6", "A")}
    write_changes(tmp_path / "changes.jsonl", [{"patch": "9.6", "kind": "champion", "name": "A"}])
    with pytest.raises(ValueError, match="no `gameplay` field"):
        changed_pairs(pm, known=frozenset(CHAMPS))


def test_unknown_block_names_and_patches_stop_the_build(tmp_path, monkeypatch, toy_games, toy_wiki):
    monkeypatch.setattr(labels_mod, "CHANGES", tmp_path / "changes.jsonl")
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    write_changes(tmp_path / "changes.jsonl", [block("9.6", "Zed")])
    with pytest.raises(ValueError, match="not champions"):
        changed_pairs(pm, known=frozenset(CHAMPS))
    write_changes(tmp_path / "changes.jsonl", [block("V9.6", "A")])
    with pytest.raises(ValueError, match="not in the patch map"):
        changed_pairs(pm, known=frozenset(CHAMPS))
    assert changed_pairs(pm, allow_unknown=True, known=frozenset(CHAMPS)) == set()


def test_patch_note_form_names_map_to_champions():
    known = frozenset({"Kled", "Gnar"})
    assert [canonical(n, known) for n in ("Kled & Skaarl", "Mega Gnar")] == ["Kled", "Gnar"]


def test_new_champion_flag_and_debut_count(rows):
    labels, inputs = rows
    # N debuts on 9.4: flagged on 9.4 and 9.5, not on 9.6
    assert labels.loc[("9.4", "N"), "new_champion"]
    assert labels.loc[("9.5", "N"), "new_champion"]
    assert not labels.loc[("9.6", "N"), "new_champion"]
    assert [inputs.loc[(p, "N"), "patches_since_debut"] for p in ["9.4", "9.5", "9.6"]] == [0, 1, 2]
    # Old champions get None
    assert pd.isna(inputs.loc[("9.4", "A"), "patches_since_debut"])
    assert not labels.loc[("9.4", "A"), "new_champion"]
