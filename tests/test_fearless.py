import pandas as pd
from conftest import game

from patch2meta.pro.fearless import game_fearless, series_fearless


def test_series_and_tournament_flags():
    rows = [
        # Fearless tournament: two series, no repeated pick inside either
        game("f1", "s1", "Fearless Cup", "2025-01-01", "25.S1.1", "ABCDE", "FGHIJ"),
        game("f2", "s1", "Fearless Cup", "2025-01-01", "25.S1.1", "KLNAB", "CDEFG")
        | {"picks_blue": ["K", "L", "N", "M1", "M2"], "picks_red": ["M3", "M4", "M5", "M6", "M7"]},
        game("f3", "s2", "Fearless Cup", "2025-01-02", "25.S1.1", "ABCDE", "FGHIJ"),
        game("f4", "s2", "Fearless Cup", "2025-01-02", "25.S1.1", "KLNAB", "CDEFG")
        | {"picks_blue": ["K", "L", "N", "M1", "M2"], "picks_red": ["M3", "M4", "M5", "M6", "M7"]},
        # Standard tournament: A is picked in both games of the series
        game("n1", "s3", "Standard Cup", "2025-01-01", "25.S1.1", "ABCDE", "FGHIJ"),
        game("n2", "s3", "Standard Cup", "2025-01-01", "25.S1.1", "AKLNM", "FGHIJ"),
        # Bo1-only tournament
        game("b1", "s4", "Bo1 League", "2025-01-01", "25.S1.1", "ABCDE", "FGHIJ"),
    ]
    g = pd.DataFrame(rows)
    series = series_fearless(g)
    assert series.to_dict() == {"s1": 1.0, "s2": 1.0, "s3": 0.0}
    flags = dict(zip(g["game_id"], game_fearless(g)))
    assert flags == {"f1": True, "f2": True, "f3": True, "f4": True, "n1": False, "n2": False, "b1": False}
