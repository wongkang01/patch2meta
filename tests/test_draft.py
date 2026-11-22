"""Draft order: a game that Cargo returns twice keeps its filled copy, not the empty one."""

import pandas as pd

from patch2meta.pro.draft import long_draft

KNOWN = frozenset({"Caitlyn", "Bard", "Lux"})


def test_a_duplicated_game_keeps_its_filled_row():
    slots = {f"Team{t}{a}{i}": "" for t in (1, 2) for a in ("Ban", "Pick") for i in range(1, 6)}
    empty = {"GameId": "g1", "Team1": "Blue Team", "Team2": "Red Team", **slots}
    filled = (
        empty
        | {f"Team1Pick{i}": "Caitlyn" for i in range(1, 6)}
        | {f"Team2Pick{i}": "Bard" for i in range(1, 6)}
    )
    games = pd.DataFrame({"game_id": ["g1"], "team_blue": ["Blue Team"], "team_red": ["Red Team"]})
    d = long_draft(pd.DataFrame([empty | {k: None for k in slots}, filled]), games, known=KNOWN)
    assert len(d) == 20
    assert d["champion"].notna().sum() == 10  # the 10 picks of the filled copy
