"""Download raw patch notes for every League of Legends patch from 9.1 onward.

Walks the LoL Wiki's patch chain (each version page links Prev/Next) and saves, per patch:
  data/raw/wiki/<V9.22>.wikitext   change lines in templated wikitext (no Riot rationale)
  data/raw/riot/<V9.22>.html       official notes page with the rationale prose: the live en-us
                                   Riot page (still up back to 9.1; the en-sg index stops at 12.17),
                                   or, if that is gone, the first Wayback capture after release
  data/raw/patches.jsonl           one PatchSource row per patch

Usage: uv run python -m patch2meta.notes.fetch [--start V9.1] [--stop V26.21] [--force]
Reruns skip files already on disk unless --force is given.
"""

import argparse
import re
import time
from datetime import date, datetime
from pathlib import Path

import httpx

from patch2meta.schema import PatchSource

WIKI_API = "https://wiki.leagueoflegends.com/en-us/api.php"
CDX_API = "https://web.archive.org/cdx/search/cdx"
HEADERS = {"User-Agent": "patch2meta-research/0.1 (university course project)"}
RAW = Path("data/raw")
DELAY_S = 1.0  # be polite to the wiki and Wayback


def infobox_field(wikitext: str, name: str) -> str | None:
    m = re.search(rf"^\|\s*{name}\s*=\s*(.*)$", wikitext, re.MULTILINE)
    return m.group(1).strip() if m and m.group(1).strip() else None


def parse_release(raw: str) -> date:
    # "November {{NumberSup|6}}, 2019" -> 2019-11-06
    text = re.sub(r"\{\{NumberSup\|(\d+)\}\}", r"\1", raw)
    text = re.sub(r"<[^>]+>|\{\{[^}]*\}\}", "", text).strip()
    return datetime.strptime(text, "%B %d, %Y").date()


def fetch_wikitext(client: httpx.Client, title: str) -> tuple[str, str] | None:
    """Return (canonical title, wikitext), or None if the page does not exist."""
    r = client.get(
        WIKI_API,
        params={"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1},
    )
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        return None
    return data["parse"]["title"], data["parse"]["wikitext"]["*"]


def fetch_live(client: httpx.Client, url: str) -> str | None:
    r = client.get(url, follow_redirects=True)
    # A removed article can redirect to an index page that also returns 200.
    ok = r.status_code == 200 and "notes" in r.url.path
    return r.text if ok else None


def fetch_wayback(client: httpx.Client, url: str, release: date) -> tuple[str, str] | None:
    """First 200 capture of `url` (any host variant) on or after `release`: (timestamp, html)."""
    path = re.sub(r"^https?://[^/]+", "", url)
    captures = []
    for host in ("na.leagueoflegends.com", "www.leagueoflegends.com"):
        r = client.get(
            CDX_API,
            params={
                "url": host + path,
                "from": release.strftime("%Y%m%d"),
                "filter": "statuscode:200",
                "limit": 1,
                "output": "json",
            },
            timeout=90,
        )
        rows = r.json() if r.status_code == 200 and r.text.strip() else []
        captures += [(row[1], row[2]) for row in rows[1:]]
        time.sleep(DELAY_S)
    if not captures:
        return None
    ts, original = min(captures)
    r = client.get(f"https://web.archive.org/web/{ts}id_/{original}", follow_redirects=True, timeout=90)
    return (ts, r.text) if r.status_code == 200 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="V9.1")
    ap.add_argument("--stop", default=None, help="last wiki title to fetch (inclusive)")
    ap.add_argument("--force", action="store_true", help="refetch files already on disk")
    args = ap.parse_args()

    (RAW / "wiki").mkdir(parents=True, exist_ok=True)
    (RAW / "riot").mkdir(parents=True, exist_ok=True)
    manifest = RAW / "patches.jsonl"
    done = {}
    if manifest.exists() and not args.force:
        for line in manifest.read_text().splitlines():
            row = PatchSource.model_validate_json(line)
            done[row.wiki_title] = row

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        title: str | None = args.start
        while title:
            got = fetch_wikitext(client, title)
            time.sleep(DELAY_S)
            if got is None:
                print(f"{title}: no wiki page, stopping")
                break
            title, wikitext = got
            release_raw = infobox_field(wikitext, "Release")
            if release_raw is None:  # the chain ends at a non-patch page such as VPBE
                print(f"{title}: not a released patch, stopping")
                break
            (RAW / "wiki" / f"{title}.wikitext").write_text(wikitext)

            if title in done and (RAW / "riot" / f"{title}.html").exists():
                row = done[title]
            else:
                release = parse_release(release_raw)
                related = infobox_field(wikitext, "Related") or ""
                m = re.search(r"https?://\S+", related)
                riot_url = m.group(0) if m else None
                if riot_url:  # some early-2019 pages link a Wayback copy; unwrap to the Riot URL
                    riot_url = re.sub(r"^https?://web\.archive\.org/web/[^/]+/", "", riot_url)
                source, snapshot, html = "missing", None, None
                if riot_url:
                    html = fetch_live(client, riot_url)
                    source = "live" if html else source
                    if html is None:
                        got_wb = fetch_wayback(client, riot_url, release)
                        if got_wb:
                            snapshot, html = got_wb
                            source = "wayback"
                if html:
                    (RAW / "riot" / f"{title}.html").write_text(html)
                row = PatchSource(
                    wiki_title=title,
                    release_date=release,
                    riot_url=riot_url,
                    riot_source=source,
                    wayback_timestamp=snapshot,
                )
                done[title] = row
                manifest.write_text("".join(r.model_dump_json() + "\n" for r in done.values()))
                time.sleep(DELAY_S)
            print(f"{title}: {row.release_date} riot={row.riot_source} {row.wayback_timestamp or ''}")

            if title == args.stop:
                break
            title = infobox_field(wikitext, "Next")


if __name__ == "__main__":
    main()
