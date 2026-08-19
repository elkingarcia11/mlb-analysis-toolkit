"""
Fetch expanded MLB player and team hitting/pitching stats from the mlb.com
stats backend (bdfed stitch API), then export to CSVs.

Timeframes (no sitCodes):
  - ytd
  - last_7   (timeframe=-6 on mlb.com)
  - last_15  (timeframe=-14)
  - last_30  (timeframe=-29)

YTD splits only: every situation code from statsapi /api/v1/situationCodes
that applies to the requested entity/group (batting, pitching, and/or team).

Usage:
  python data_fetcher.py
  python data_fetcher.py --season 2026
  python data_fetcher.py --timeframes ytd last_7 --groups hitting --no-splits

Writes to data/raw/YYYY-MM-DD/ by default. Existing CSVs are left intact
(never clobbered); re-runs skip files that already exist for that day.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from http_utils import default_ssl_context
from paths import raw_day_dir

BDFED_PLAYER = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
BDFED_TEAM = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/team"
SITUATION_CODES_URL = "https://statsapi.mlb.com/api/v1/situationCodes"

USER_AGENT = "mlb-analysis-toolkit/1.0 (+local data fetcher)"
REQUEST_PAUSE_S = 0.15
MAX_RETRIES = 4

# mlb.com timeframe=-N means inclusive window of N+1 calendar days ending today.
TIMEFRAMES: dict[str, int | None] = {
    "ytd": None,
    "last_7": 6,
    "last_15": 14,
    "last_30": 29,
}

META_COLS = [
    "date_fetched",
    "season",
    "entity",
    "stat_group",
    "timeframe",
    "split_code",
    "split_description",
    "split_menu",
]


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60, context=default_ssl_context()) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}") from last_err


def resolve_season(season: int | None = None, as_of: date | None = None) -> int:
    if season is not None:
        return season
    as_of = as_of or date.today()
    # MLB season year follows the calendar year of the summer months.
    return as_of.year if as_of.month >= 3 else as_of.year - 1


def load_situation_codes() -> list[dict[str, Any]]:
    data = _http_get_json(SITUATION_CODES_URL)
    if not isinstance(data, list):
        raise RuntimeError("Unexpected situationCodes payload")
    return data


def applicable_splits(codes: Iterable[dict[str, Any]], entity: str, group: str) -> list[dict[str, Any]]:
    """Return sitCodes that apply to this entity/group."""
    out: list[dict[str, Any]] = []
    for code in codes:
        batting = bool(code.get("batting"))
        pitching = bool(code.get("pitching"))
        team = bool(code.get("team"))
        if entity == "player":
            if group == "hitting" and batting:
                out.append(code)
            elif group == "pitching" and pitching:
                out.append(code)
        elif entity == "team":
            if not team:
                continue
            if group == "hitting" and (batting or not pitching):
                # batting splits + team-only situational filters
                if batting or (not batting and not pitching):
                    out.append(code)
            elif group == "pitching" and (pitching or not batting):
                if pitching or (not batting and not pitching):
                    out.append(code)
    # de-dupe by code preserving order
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in out:
        c = str(item["code"])
        if c in seen:
            continue
        seen.add(c)
        unique.append(item)
    return unique


def _date_range_for_timeframe(timeframe: str, as_of: date) -> tuple[date, date] | None:
    days_back = TIMEFRAMES[timeframe]
    if days_back is None:
        return None
    return as_of - timedelta(days=days_back), as_of


def fetch_expanded_stats(
    entity: str,
    group: str,
    season: int,
    *,
    timeframe: str = "ytd",
    sit_code: str | None = None,
    as_of: date | None = None,
    game_type: str = "R",
    page_size: int = 5000,
) -> list[dict[str, Any]]:
    """
    Fetch all rows for one entity/group/timeframe or YTD sitCode combo.

    Rolling windows (last_7/15/30) cannot be combined with sitCodes.
    """
    if entity not in {"player", "team"}:
        raise ValueError(f"entity must be player|team, got {entity!r}")
    if group not in {"hitting", "pitching"}:
        raise ValueError(f"group must be hitting|pitching, got {group!r}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}")
    if sit_code and timeframe != "ytd":
        raise ValueError("sitCodes/splits are only supported for timeframe=ytd")

    as_of = as_of or date.today()
    endpoint = BDFED_PLAYER if entity == "player" else BDFED_TEAM

    params: dict[str, Any] = {
        "stitch_env": "prod",
        "season": season,
        "sportId": 1,
        "group": group,
        "gameType": game_type,
        "offset": 0,
        "limit": page_size,
    }
    if entity == "player":
        params["playerPool"] = "ALL"

    date_range = _date_range_for_timeframe(timeframe, as_of)
    if date_range is None:
        params["stats"] = "season"
    else:
        start, end = date_range
        params["stats"] = "byDateRange"
        params["startDate"] = start.isoformat()
        params["endDate"] = end.isoformat()

    if sit_code:
        params["sitCodes"] = sit_code

    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while True:
        params["offset"] = offset
        payload = _http_get_json(endpoint, params)
        batch = payload.get("stats") or []
        if total is None:
            total = int(payload.get("totalSplits") or 0)
        rows.extend(batch)
        offset += len(batch)
        if not batch or (total is not None and offset >= total) or len(batch) < page_size:
            break
        time.sleep(REQUEST_PAUSE_S)

    time.sleep(REQUEST_PAUSE_S)
    return rows


def _annotate_rows(
    rows: list[dict[str, Any]],
    *,
    date_fetched: str,
    season: int,
    entity: str,
    group: str,
    timeframe: str,
    split: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    split_code = "" if split is None else str(split.get("code", ""))
    split_description = "" if split is None else str(split.get("description", ""))
    split_menu = "" if split is None else str(split.get("navigationMenu", ""))
    annotated: list[dict[str, Any]] = []
    for row in rows:
        annotated.append(
            {
                "date_fetched": date_fetched,
                "season": season,
                "entity": entity,
                "stat_group": group,
                "timeframe": timeframe,
                "split_code": split_code,
                "split_description": split_description,
                "split_menu": split_menu,
                **row,
            }
        )
    return annotated


def fetch_requested(
    *,
    season: int | None = None,
    entities: Iterable[str] = ("player", "team"),
    groups: Iterable[str] = ("hitting", "pitching"),
    timeframes: Iterable[str] = tuple(TIMEFRAMES),
    include_splits: bool = True,
    split_codes: Iterable[str] | None = None,
    as_of: date | None = None,
    game_type: str = "R",
    on_progress: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch all requested slices and return {"player": [...], "team": [...]}.
    """
    as_of = as_of or date.today()
    season = resolve_season(season, as_of)
    # Date only (YYYY-MM-DD) so odds/results can join on the same calendar day.
    date_fetched = as_of.isoformat()

    entities = tuple(entities)
    groups = tuple(groups)
    timeframes = tuple(timeframes)
    requested_split_codes = {
        str(code).strip() for code in (split_codes or ()) if str(code).strip()
    }

    sit_codes = load_situation_codes() if include_splits or requested_split_codes else []
    buckets: dict[str, list[dict[str, Any]]] = {"player": [], "team": []}

    jobs: list[tuple[str, str, str, dict[str, Any] | None]] = []
    for entity in entities:
        for group in groups:
            for timeframe in timeframes:
                jobs.append((entity, group, timeframe, None))
            if include_splits or requested_split_codes:
                if requested_split_codes and (entity != "player" or group != "hitting"):
                    continue
                for split in applicable_splits(sit_codes, entity, group):
                    if requested_split_codes and str(split.get("code") or "") not in requested_split_codes:
                        continue
                    jobs.append((entity, group, "ytd", split))

    total_jobs = len(jobs)
    for idx, (entity, group, timeframe, split) in enumerate(jobs, start=1):
        label = f"{entity}/{group}/{timeframe}" + (
            f"/split={split['code']}" if split else ""
        )
        if on_progress:
            on_progress(idx, total_jobs, label)
        try:
            rows = fetch_expanded_stats(
                entity,
                group,
                season,
                timeframe=timeframe,
                sit_code=None if split is None else str(split["code"]),
                as_of=as_of,
                game_type=game_type,
            )
        except Exception as err:  # keep going so one bad sitCode does not abort the run
            if on_progress:
                on_progress(idx, total_jobs, f"{label} FAILED: {err}")
            continue
        buckets[entity].extend(
            _annotate_rows(
                rows,
                date_fetched=date_fetched,
                season=season,
                entity=entity,
                group=group,
                timeframe=timeframe if split is None else "ytd",
                split=split,
            )
        )

    return buckets


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for col in META_COLS:
        keys.append(col)
        seen.add(col)
    extras: set[str] = set()
    for row in rows:
        extras.update(row.keys())
    for col in sorted(extras):
        if col not in seen:
            keys.append(col)
            seen.add(col)
    return keys


def write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    """
    Write rows to path if it does not already exist.

    Returns "wrote" or "skipped" (existing file left intact — never clobbered).
    """
    if path.exists():
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return "wrote"


def export_csvs(
    data: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    *,
    player_name: str = "players.csv",
    team_name: str = "teams.csv",
) -> dict[str, tuple[Path, str, int]]:
    """
    Write player/team CSVs under out_dir without overwriting history.

    Returns {label: (path, status, row_count)} where status is wrote|skipped.
    """
    results: dict[str, tuple[Path, str, int]] = {}
    for label, name, key in (
        ("player", player_name, "player"),
        ("team", team_name, "team"),
    ):
        rows = data.get(key, [])
        path = out_dir / name
        status = write_csv(path, rows)
        results[label] = (path, status, len(rows))
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=None, help="MLB season year (default: current)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="Anchor date YYYY-MM-DD for rolling windows")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="CSV output directory (default: <data-dir>/raw/YYYY-MM-DD from --as-of/today)",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        choices=["player", "team"],
        default=["player", "team"],
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["hitting", "pitching"],
        default=["hitting", "pitching"],
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=list(TIMEFRAMES),
        default=list(TIMEFRAMES),
    )
    parser.add_argument("--no-splits", action="store_true", help="Skip YTD sitCode splits")
    parser.add_argument("--game-type", default="R", help="gameType filter (default R)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    as_of = args.as_of or date.today()
    out_dir = args.out_dir or raw_day_dir(as_of, args.data_dir)

    def progress(i: int, n: int, label: str) -> None:
        if not args.quiet:
            print(f"[{i}/{n}] {label}", file=sys.stderr)

    data = fetch_requested(
        season=args.season,
        entities=args.entities,
        groups=args.groups,
        timeframes=args.timeframes,
        include_splits=not args.no_splits,
        as_of=as_of,
        game_type=args.game_type,
        on_progress=progress,
    )
    exported = export_csvs(data, out_dir)
    for label in ("player", "team"):
        path, status, n = exported[label]
        if status == "skipped":
            print(f"Skipped {label} CSV (already exists, left intact) -> {path}")
        else:
            print(f"Wrote {n} {label} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
