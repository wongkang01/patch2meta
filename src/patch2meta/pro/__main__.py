"""Pro data pipeline. Usage: uv run python -m patch2meta.pro <command> [options]

Commands, in pipeline order:
  download   raw files into data/raw/pro/ (see pro/download.py)
  games      data/processed/games.parquet
  patches    data/processed/patch_map.csv
  labels     labels.parquet, data/private/labels_test.parquet, prepatch_stats.parquet
  graph      relation_graph.parquet
  draft      draft.parquet
  build      games, patches, labels and graph in one run
  variants   one build per open data choice, compared with the default build (see pro/variants.py)
  share      data/share/, the hand-off folder for teammates (train and dev rows until the freeze)

games, patches, labels, graph and build take --data DIR and --private DIR, plus the options of their
steps; run `uv run python -m patch2meta.pro build --help` for the list.
"""

import importlib
import sys

from patch2meta.pro import steps

COMMANDS = {
    "download": "patch2meta.pro.download",
    "draft": "patch2meta.pro.draft",
    "variants": "patch2meta.pro.variants",
    "share": "patch2meta.pro.share",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in [*steps.STEPS, *COMMANDS, "build"]:
        print(__doc__)
        sys.exit(2)
    command, rest = sys.argv[1], sys.argv[2:]
    if command in COMMANDS:
        importlib.import_module(COMMANDS[command]).main(rest)
        return
    names = list(steps.STEPS) if command == "build" else [command]
    steps.run(names, steps.parser(f"patch2meta.pro {command}", names).parse_args(rest))


if __name__ == "__main__":
    main()
