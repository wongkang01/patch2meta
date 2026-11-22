import pandas as pd

from patch2meta.eval.strata import build_strata
from patch2meta.pro.draft import long_draft

KNOWN = frozenset({"Caitlyn", "Bard", "Lux"})


def test_block_to_row_rule():
    blocks = pd.DataFrame(
        [
            ("9.2:champion:Caitlyn", "9.2", "champion", "Caitlyn"),
            ("9.3:champion:Caitlyn", "9.3", "champion", "Caitlyn"),
            ("9.3:champion:Bard", "9.3", "champion", "Bard"),
            ("9.3:item:Infinity Edge", "9.3", "item", "Infinity Edge"),
            ("9.2:system:Dragon", "9.2", "system", "Dragon"),
        ],
        columns=["block_id", "patch", "kind", "name"],
    )
    labels = pd.DataFrame(
        [
            ("9.2:champion:Caitlyn", "mechanic", "nerf"),
            ("9.3:champion:Caitlyn", "number_tweak", "buff"),
            ("9.3:champion:Bard", "rework", "buff"),
            ("9.3:item:Infinity Edge", "item", "nerf"),
            ("9.2:system:Dragon", "map_objective", "shift"),
        ],
        columns=["block_id", "change_type", "intent"],
    )
    pm = pd.DataFrame({"patch": ["9.3", "9.4"], "riot_patches_since_prev": ["9.2;9.3", "9.4"]})
    rows, patches = build_strata(blocks, labels, pm, known=KNOWN)
    rows = rows.set_index(["patch", "champion"])
    assert rows.loc[("9.3", "Caitlyn"), "change_type"] == "mechanic"  # mechanic beats number_tweak
    assert rows.loc[("9.3", "Caitlyn"), "intent"] == "mixed"  # nerf and buff disagree
    assert rows.loc[("9.3", "Caitlyn"), "n_blocks"] == 2
    assert rows.loc[("9.3", "Bard"), "change_type"] == "rework"
    p = patches.set_index("patch")
    assert p.loc["9.3"].to_dict() == {"has_item": True, "has_rune": False, "has_map_objective": True}
    assert not p.loc["9.4"].any()


def test_draft_order_follows_the_first_team_not_the_side():
    raw = pd.DataFrame(
        [
            {
                "GameId": "g1",
                "Team1": "Red Team",  # drafts first while playing red (SwitchedSides)
                "Team2": "Blue Team",
                **{f"Team1Ban{i}": "Lux" for i in range(1, 6)},
                **{f"Team2Ban{i}": "None" for i in range(1, 6)},
                **{f"Team1Pick{i}": "Caitlyn" for i in range(1, 6)},
                **{f"Team2Pick{i}": "Bard" for i in range(1, 6)},
            }
        ]
    )
    games = pd.DataFrame({"game_id": ["g1"], "team_blue": ["Blue Team"], "team_red": ["Red Team"]})
    d = long_draft(raw, games, known=KNOWN).set_index("order")
    assert (d.loc[1, "team"], d.loc[1, "side"], d.loc[1, "action"]) == ("Red Team", "red", "ban")
    assert pd.isna(d.loc[2, "champion"])  # skipped ban
    assert (d.loc[7, "team"], d.loc[7, "action"]) == ("Red Team", "pick")
    assert list(d.loc[[8, 9], "team"]) == ["Blue Team", "Blue Team"]
    assert d.loc[13, "team"] == "Blue Team" and d.loc[20, "team"] == "Blue Team"
