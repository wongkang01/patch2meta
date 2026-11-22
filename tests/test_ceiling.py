"""Noise ceiling on the toy games in conftest.py."""

import pandas as pd

from patch2meta.eval.ceiling import noise_ceiling
from patch2meta.pro import labels as labels_mod
from patch2meta.pro.counts import build
from patch2meta.pro.labels import build_rows
from patch2meta.pro.patches import build_patch_map


def test_all_ceiling_does_not_depend_on_changed_flags(
    tmp_path, monkeypatch, toy_games, toy_wiki, toy_release
):
    champions = pd.DataFrame({"Name": toy_release.index, "ReleaseDate": toy_release.dt.strftime("%Y-%m-%d")})
    champions.to_parquet(tmp_path / "champions.parquet", index=False)
    monkeypatch.setattr(labels_mod, "CHAMPIONS", tmp_path / "champions.parquet")
    toy_games.to_parquet(tmp_path / "games.parquet", index=False)
    pm = build_patch_map(toy_games, toy_wiki, min_games=2)
    pm["release_date"] = pd.to_datetime(pm["release_date"])
    pm["cut"] = pd.to_datetime(pm["cut"])
    labels, _ = build_rows(build(toy_games, list(toy_release.index)), pm, toy_release, changed=None)

    def all_ceiling() -> dict:
        result = noise_ceiling(labels, "dev", n_boot=50, games_path=tmp_path / "games.parquet", save=False)
        return {patch: v["all"] for patch, v in result["per_patch"].items()}

    before = all_ceiling()
    labels["changed"] = labels.index % 3 == 0
    assert any(v is not None for v in before.values())
    assert all_ceiling() == before
