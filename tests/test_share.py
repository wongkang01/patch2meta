"""The hand-off folder: no 2026 or 26.22 row before the freeze, and never a test label."""

import pandas as pd

from patch2meta.pro.share import SPLIT_FILES, build_share


def test_share_holds_back_test_rows_until_the_freeze(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    pd.DataFrame({
        "patch": ["25.24", "26.01", "26.22"], "release_date": ["2025-12-03", "2026-01-08", "2026-11-04"],
        "cut": ["2025-12-02", "2026-01-07", "2026-11-03"], "kept": True, "split": ["dev", "test", "none"],
    }).to_csv(data / "patch_map.csv", index=False)  # fmt: skip
    for name in SPLIT_FILES:
        pd.DataFrame({"patch": ["25.24", "26.01", "26.22"], "champion": "A"}).to_parquet(data / name)

    build_share(data, tmp_path / "before")
    for name in SPLIT_FILES:
        assert set(pd.read_parquet(tmp_path / "before" / name)["patch"]) == {"25.24"}
    assert (tmp_path / "before/patch_map.csv").read_bytes() == (data / "patch_map.csv").read_bytes()

    build_share(data, tmp_path / "after", include_test=True)
    assert set(pd.read_parquet(tmp_path / "after/prepatch_stats.parquet")["patch"]) == {
        "25.24",
        "26.01",
        "26.22",
    }
    assert set(pd.read_parquet(tmp_path / "after/labels.parquet")["patch"]) == {"25.24"}
