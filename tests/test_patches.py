import pandas as pd

from patch2meta.pro.patches import build_patch_map, client_version, cut_of, patch_key, split_of


def test_patch_key_order():
    ids = ["10.1", "9.24b", "9.2", "25.04", "25.S1.3", "25.S1.1", "14.24", "26.01", "9.24"]
    assert sorted(ids, key=patch_key) == [
        "9.2", "9.24", "9.24b", "10.1", "14.24", "25.S1.1", "25.S1.3", "25.04", "26.01",
    ]  # fmt: skip


def test_split_and_client_version():
    assert [split_of(p) for p in ["9.1", "14.24", "25.S1.1", "26.01", "26.20", "26.21"]] == [
        "train", "train", "dev", "test", "test", "none",
    ]  # fmt: skip
    assert client_version("26.21") == "16.21"
    assert client_version("26.01") == "16.1"
    assert client_version("25.S1.2") == "15.2"
    assert client_version("13.7") == "13.7"


def test_cut_is_midnight_the_day_before_release():
    assert cut_of(pd.Timestamp("2026-10-21")) == pd.Timestamp("2026-10-20 00:00")


def test_patch_map_chain(toy_games, toy_wiki):
    pm = build_patch_map(toy_games, toy_wiki, min_games=2).set_index("patch")
    assert pm.loc["9.5b", "kept"] == False
    assert pm.at["9.6", "prev_patch"] == "9.5"
    assert pm.at["9.6", "riot_patches_since_prev"] == "9.5b;9.6"
    assert pd.isna(pm.at["9.1", "prev_patch"])
    assert pm.at["9.2", "cut"] == pd.Timestamp("2019-01-23")
