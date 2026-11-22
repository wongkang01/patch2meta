"""The four build steps, their folder options, and one argument parser for any run of them.

`--data` holds the shared outputs and `--private` the test labels. The defaults are the team's files;
`patch2meta.pro variants` points both at one folder per open data choice.
"""

import argparse
import importlib
from pathlib import Path

DATA = Path("data/processed")
PRIVATE = Path("data/private")
STEPS = {
    "games": "patch2meta.pro.games",
    "patches": "patch2meta.pro.patches",
    "labels": "patch2meta.pro.labels",
    "graph": "patch2meta.pro.graph",
}


def parser(prog: str, names: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument(
        "--data", type=Path, default=DATA, help="folder for shared outputs (default: %(default)s)"
    )
    ap.add_argument(
        "--private", type=Path, default=PRIVATE, help="folder for test labels (default: %(default)s)"
    )
    for name in names:
        add_args = getattr(importlib.import_module(STEPS[name]), "add_args", None)
        if add_args is not None:
            add_args(ap)
    return ap


def run(names: list[str], args: argparse.Namespace) -> None:
    for name in names:
        print(f"== {name}")
        importlib.import_module(STEPS[name]).run(args)
