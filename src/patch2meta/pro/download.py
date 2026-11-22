"""Raw downloads for the pro data, all saved under data/raw/pro/ and never edited afterwards.

Parts, in the order `all` runs them:
  gptilt       gptilt_{games,matches,tournaments}.parquet from Hugging Face (Leaguepedia snapshot)
  games2026    cargo_games_2026.parquet: Leaguepedia ScoreboardGames from 2026-01-01
  champions    cargo_champions.parquet: Leaguepedia Champions (Name, ReleaseDate)
  tournaments  cargo_tournaments.parquet: Leaguepedia Tournaments from 2019
  wiki         wiki_release_dates.csv: LoL Wiki patch chain from V9.1 with release dates
  draft        cargo_draft/<year>.parquet: Leaguepedia PicksAndBansS7 in draft order (about 90 minutes)

Every file gets an entry in manifest.json (source, pull time in UTC, rows, sha256).
Files already on disk are skipped unless --force is given.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq

from patch2meta.notes.fetch import infobox_field, parse_release
from patch2meta.pro.cargo import HEADERS, cargo_query

RAW = Path("data/raw/pro")
GPTILT_URL = "https://huggingface.co/api/datasets/gptilt/lol-esports-matches/parquet/{config}/train/0.parquet"
WIKI_API = "https://wiki.leagueoflegends.com/en-us/api.php"
WIKI_DELAY_S = 1.5
WIKI_START = "V9.1"
GAMES_FROM = "2019-01-01"
CARGO_FROM = "2026-01-01"
DRAFT_YEARS = range(2019, 2027)
PARTS = ["gptilt", "games2026", "champions", "tournaments", "wiki", "draft"]

# Aliases avoid underscores: Cargo turns underscores in result keys into spaces.
GAME_FIELDS = {
    "GameId": "SG.GameId",
    "MatchId": "SG.MatchId",
    "OverviewPage": "SG.OverviewPage",
    "Tournament": "SG.Tournament",
    "Team1": "SG.Team1",
    "Team2": "SG.Team2",
    "WinTeam": "SG.WinTeam",
    "DateTimeUTC": "SG.DateTime_UTC",
    "Patch": "SG.Patch",
    "Team1Picks": "SG.Team1Picks",
    "Team2Picks": "SG.Team2Picks",
    "Team1Bans": "SG.Team1Bans",
    "Team2Bans": "SG.Team2Bans",
    "NGameInMatch": "SG.N_GameInMatch",
}
DRAFT_FIELDS = {
    "GameId": "PB.GameId",
    "MatchId": "PB.MatchId",
    "OverviewPage": "PB.OverviewPage",
    "DateTimeUTC": "SG.DateTime_UTC",
    "Team1": "PB.Team1",
    "Team2": "PB.Team2",
    "SwitchedSides": "PB.SwitchedSides",
    "Team1PicksByRoleOrder": "PB.Team1PicksByRoleOrder",
    "Team2PicksByRoleOrder": "PB.Team2PicksByRoleOrder",
    **{f"Team{t}{a}{i}": f"PB.Team{t}{a}{i}" for t in (1, 2) for a in ("Ban", "Pick") for i in range(1, 6)},
}
TOURNAMENT_FIELDS = {
    "OverviewPage": "T.OverviewPage",
    "Name": "T.Name",
    "League": "T.League",
    "Region": "T.Region",
    "TournamentLevel": "T.TournamentLevel",
    "IsOfficial": "T.IsOfficial",
    "Year": "T.Year",
    "DateStart": "T.DateStart",
    "DateEnd": "T.Date",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, source: str, rows: int) -> None:
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[str(path.relative_to(RAW))] = {
        "source": source,
        "pulled_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": rows,
        "sha256": sha256(path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_rows(rows: list[dict], path: Path, source: str) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    record(path, source, len(df))
    print(f"  {path}: {len(df)} rows")


def download_gptilt(force: bool) -> None:
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        for config in ("games", "matches", "tournaments"):
            path = RAW / f"gptilt_{config}.parquet"
            if path.exists() and not force:
                continue
            url = GPTILT_URL.format(config=config)
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            record(path, url, pq.ParquetFile(path).metadata.num_rows)
            print(f"  {path}: downloaded")


def download_games2026(client: httpx.Client, force: bool) -> None:
    path = RAW / "cargo_games_2026.parquet"
    if path.exists() and not force:
        return
    where = f"SG.DateTime_UTC >= '{CARGO_FROM}'"
    rows = cargo_query(client, "ScoreboardGames=SG", GAME_FIELDS, where=where, order_by="SG.GameId")
    write_rows(rows, path, f"cargo ScoreboardGames where {where}")


def download_champions(client: httpx.Client, force: bool) -> None:
    path = RAW / "cargo_champions.parquet"
    if path.exists() and not force:
        return
    fields = {"Name": "C.Name", "ReleaseDate": "C.ReleaseDate"}
    rows = cargo_query(client, "Champions=C", fields, order_by="C.Name")
    write_rows(rows, path, "cargo Champions")


def download_tournaments(client: httpx.Client, force: bool) -> None:
    path = RAW / "cargo_tournaments.parquet"
    if path.exists() and not force:
        return
    where = "T.Year >= 2019"
    rows = cargo_query(client, "Tournaments=T", TOURNAMENT_FIELDS, where=where, order_by="T.OverviewPage")
    write_rows(rows, path, f"cargo Tournaments where {where}")


def download_draft(client: httpx.Client, force: bool) -> None:
    for year in DRAFT_YEARS:
        path = RAW / "cargo_draft" / f"{year}.parquet"
        if path.exists() and not force:
            continue
        where = f"SG.DateTime_UTC >= '{year}-01-01' AND SG.DateTime_UTC < '{year + 1}-01-01'"
        print(f"  draft {year} ...")
        rows = cargo_query(
            client,
            "PicksAndBansS7=PB,ScoreboardGames=SG",
            DRAFT_FIELDS,
            where=where,
            join_on="PB.GameId=SG.GameId",
            order_by="SG.GameId",
        )
        write_rows(rows, path, f"cargo PicksAndBansS7 joined ScoreboardGames where {where}")


def wiki_infobox(client: httpx.Client, title: str) -> dict | None:
    """Release date and chain links from section 0 of a LoL Wiki version page, or None if missing."""
    r = client.get(
        WIKI_API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": 0,
            "format": "json",
            "redirects": 1,
        },
    )
    r.raise_for_status()
    data = r.json()
    time.sleep(WIKI_DELAY_S)
    if "error" in data:
        return None
    wikitext = data["parse"]["wikitext"]["*"]
    canonical = data["parse"]["title"]
    release_raw = infobox_field(wikitext, "Release")
    try:
        release = parse_release(release_raw) if release_raw else None
    except ValueError:
        release = None
    redirected = canonical != title
    return {
        "wiki_title": title,
        "patch": title.removeprefix("V"),
        "release_date": None if redirected else release,
        "prev": None if redirected else infobox_field(wikitext, "Prev"),
        "next": None if redirected else infobox_field(wikitext, "Next"),
        "redirect_to": canonical if redirected else None,
    }


def game_patch_ids() -> set[str]:
    g = pd.read_parquet(RAW / "gptilt_games.parquet", columns=["datetime_utc", "patch"])
    ids = set(g.loc[(g["datetime_utc"] >= GAMES_FROM) & (g["patch"] != ""), "patch"])
    c = RAW / "cargo_games_2026.parquet"
    if c.exists():
        ids |= set(pd.read_parquet(c, columns=["Patch"])["Patch"].dropna()) - {""}
    return ids


def download_wiki(force: bool) -> None:
    path = RAW / "wiki_release_dates.csv"
    if path.exists() and not force:
        return
    rows: list[dict] = []
    seen: set[str] = set()
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        title: str | None = WIKI_START
        while title and title not in seen:
            seen.add(title)
            info = wiki_infobox(client, title)
            if info is None or info["release_date"] is None and not info["patch"][:1].isdigit():
                print(f"  {title}: not a patch page, chain stops")
                break
            info["chain_order"] = len(rows)
            rows.append(info)
            title = info["next"]
        for patch in sorted(game_patch_ids() - {r["patch"] for r in rows}):
            info = wiki_infobox(client, "V" + patch)
            rows.append(
                (info or {"wiki_title": "V" + patch, "patch": patch, "release_date": None})
                | {"chain_order": None}
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    record(path, f"{WIKI_API} section 0 of V<patch>, chain from {WIKI_START}", len(df))
    print(f"  {path}: {len(df)} patches, {df['release_date'].notna().sum()} with a release date")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="patch2meta.pro download")
    ap.add_argument("parts", nargs="*", default=["all"], help=f"any of {PARTS}, or all")
    ap.add_argument("--force", action="store_true", help="download files already on disk again")
    args = ap.parse_args(argv)
    parts = PARTS if "all" in args.parts else args.parts
    RAW.mkdir(parents=True, exist_ok=True)
    if "gptilt" in parts:
        print("gptilt")
        download_gptilt(args.force)
    with httpx.Client(headers=HEADERS) as client:
        for name, fn in (
            ("games2026", download_games2026),
            ("champions", download_champions),
            ("tournaments", download_tournaments),
        ):
            if name in parts:
                print(name)
                fn(client, args.force)
    if "wiki" in parts:
        print("wiki")
        download_wiki(args.force)
    if "draft" in parts:
        print("draft")
        with httpx.Client(headers=HEADERS) as client:
            download_draft(client, args.force)
