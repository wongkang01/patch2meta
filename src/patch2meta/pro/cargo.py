"""A small client for Leaguepedia's Cargo API (https://lol.fandom.com/api.php?action=cargoquery).

Every field is aliased so result keys are stable (unaliased keys turn underscores into spaces).
Results come back 500 rows at a time; the client pages with `offset` and waits between requests.
Leaguepedia answers a rate limit with HTTP 200 and an `error` key, so the body is always checked.
"""

import time

import httpx

API = "https://lol.fandom.com/api.php"
HEADERS = {"User-Agent": "patch2meta-research/0.1 (university course project)"}
PAGE = 500
DELAY_S = 8.0
RATE_LIMIT_WAIT_S = 120.0
RATE_LIMIT_RETRIES = 10  # a rate limit can last 6 to 8 minutes; give up after about 20
RETRIES = 3  # network and HTTP errors


class CargoError(RuntimeError):
    pass


def _get(client: httpx.Client, params: dict) -> list[dict]:
    errors = limits = 0
    while True:
        try:
            r = client.get(API, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            errors += 1
            if errors > RETRIES:
                raise CargoError(f"request failed: {e!r}") from e
            time.sleep(DELAY_S * 2**errors)
            continue
        if "error" in data:
            code = data["error"].get("code")
            if code == "ratelimited" and limits < RATE_LIMIT_RETRIES:
                limits += 1
                print(
                    f"  cargo: rate limited, waiting {RATE_LIMIT_WAIT_S:.0f} s ({limits}/{RATE_LIMIT_RETRIES})"
                )
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
            raise CargoError(f"{code}: {data['error'].get('info')}")
        return [
            {k: v for k, v in row["title"].items() if not k.endswith("__precision")}
            for row in data.get("cargoquery", [])
        ]


def cargo_query(
    client: httpx.Client,
    tables: str,
    fields: dict[str, str],
    where: str | None = None,
    join_on: str | None = None,
    order_by: str | None = None,
    group_by: str | None = None,
    max_rows: int | None = None,
) -> list[dict]:
    """Run one Cargo query and return every row.

    `fields` maps an alias to its expression, e.g. {"GameId": "SG.GameId"}.
    `order_by` should name a unique field so paging is stable.
    """
    params = {
        "action": "cargoquery",
        "format": "json",
        "tables": tables,
        "fields": ",".join(f"{expr}={alias}" for alias, expr in fields.items()),
        "limit": PAGE,
    }
    for key, value in (
        ("where", where),
        ("join_on", join_on),
        ("order_by", order_by),
        ("group_by", group_by),
    ):
        if value:
            params[key] = value
    rows: list[dict] = []
    offset = 0
    while True:
        page = _get(client, {**params, "offset": offset})
        rows += page
        time.sleep(DELAY_S)
        if len(page) < PAGE or (max_rows is not None and len(rows) >= max_rows):
            return rows
        offset += PAGE
