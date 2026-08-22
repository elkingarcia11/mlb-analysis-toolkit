"""
Fetch expanded MLB player and team hitting/pitching stats, then export CSVs.

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

Implementation notes
--------------------
- The bdfed stats backend accepts only a single sitCodes value per request,
  so fetching N split codes previously required N HTTP round trips.  The
  StatsAPI /api/v1/stats endpoint with stats=statSplits accepts *multiple*
  situation codes in one request and returns one row per split in its
  ``splits`` array.  We batch the applicable codes into a few multi-sitCodes
  requests instead of one request per code.
- Rolling windows (last_7/15/30) cannot be combined with sitCodes.
- Independent entity/group/timeframe jobs run concurrently through a bounded
  ThreadPoolExecutor.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from http_utils import DEFAULT_USER_AGENT, fetch_page_with_retry
from paths import raw_day_dir

BDFED_PLAYER = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
BDFED_TEAM = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/team"
STATSAPI_STATS = "https://statsapi.mlb.com/api/v1/stats"
STATSAPI_TEAMS = "https://statsapi.mlb.com/api/v1/teams"
SITUATION_CODES_URL = "https://statsapi.mlb.com/api/v1/situationCodes"

REQUEST_PAUSE_S = 0.15
MAX_RETRIES = 4
# Split codes are grouped into multi-sitCode requests.  A single monster URL
# is fragile, so chunk the applicable codes into sets of this size.
SIT_CODES_PER_REQUEST = 20
# Small politeness pause between split-batch pages while threads interleave.
THREAD_PAUSE_S = 0.02

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

# Nested identity keys inside a StatsAPI split row; each maps to flat columns.
_IDENTITY_KEYS = ("player", "team", "league", "sport", "position")


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """Thin wrapper retained for compatibility with existing callers."""
    return fetch_page_with_retry(
        url, params, headers={"User-Agent": DEFAULT_USER_AGENT}, max_retries=MAX_RETRIES
    )


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


def applicable_splits(
    codes: Iterable[dict[str, Any]], entity: str, group: str
) -> list[dict[str, Any]]:
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


def _date_range_for_timeframe(
    timeframe: str, as_of: date
) -> tuple[date, date] | None:
    days_back = TIMEFRAMES[timeframe]
    if days_back is None:
        return None
    return as_of - timedelta(days=days_back), as_of


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def _base_params(
    *,
    entity: str,
    group: str,
    season: int,
    timeframe: str,
    as_of: date,
    game_type: str,
    page_size: int,
) -> dict[str, Any]:
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
    return params


def _fetch_bdfed_base(
    *,
    entity: str,
    group: str,
    season: int,
    timeframe: str,
    as_of: date,
    game_type: str,
    page_size: int = 5000,
) -> list[dict[str, Any]]:
    """Fetch one entity/group/timeframe slice from the bdfed backend."""
    endpoint = BDFED_PLAYER if entity == "player" else BDFED_TEAM
    params = _base_params(
        entity=entity,
        group=group,
        season=season,
        timeframe=timeframe,
        as_of=as_of,
        game_type=game_type,
        page_size=page_size,
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params["offset"] = offset
        payload = fetch_page_with_retry(
            endpoint,
            params,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            max_retries=MAX_RETRIES,
        )
        batch = payload.get("stats") or []
        rows.extend(batch)
        offset += len(batch)
        total = int(payload.get("totalSplits") or 0) or len(batch)
        if not batch or offset >= total or len(batch) < page_size:
            break
        time.sleep(REQUEST_PAUSE_S)
    time.sleep(REQUEST_PAUSE_S)
    return rows


def _flatten_split_row(row: dict[str, Any], *, entity: str) -> dict[str, Any]:
    """
    Convert a StatsAPI statSplits row (nested identity dicts + stat object)
    into the flat shape the rest of the toolkit consumes.
    """
    split_obj = row.get("split") if isinstance(row.get("split"), dict) else {}
    out: dict[str, Any] = {
        "split_code": str(split_obj.get("code") or ""),
        "split_description": str(split_obj.get("description") or ""),
        "type": row.get("type") or entity,
        "rank": row.get("rank"),
        "season": row.get("season"),
        "gameType": row.get("gameType"),
        "numTeams": row.get("numTeams"),
    }
    for key, value in row.items():
        if key in _IDENTITY_KEYS or key in {"split", "stat", "rank", "type", "season", "gameType", "numTeams"}:
            continue
        if isinstance(value, dict):
            out[key] = value
        else:
            out[key] = value
    # identity dicts -> flat columns
    pos_obj = row.get("position")
    if isinstance(pos_obj, dict):
        pos = pos_obj.get("abbreviation") or pos_obj.get("name") or ""
        out["position"] = pos
        out["positionAbbrev"] = pos
        out["primaryPositionAbbrev"] = pos
    for key in ("player", "team", "league", "sport"):
        obj = row.get(key)
        if not isinstance(obj, dict):
            continue
        prefix = key
        for k, v in obj.items():
            if k == "id":
                out[f"{prefix}Id"] = v
            elif k == "fullName":
                out["playerFullName"] = v
                out["playerName"] = v
            elif k == "name":
                if prefix in {"team", "league", "sport"}:
                    out[f"{prefix}Name"] = v
                else:
                    out["playerName"] = v
            elif k == "abbreviation":
                out[f"{prefix}Abbrev"] = v
            elif k == "firstName":
                out["playerFirstName"] = v
            elif k == "lastName":
                out["playerLastName"] = v
            elif k == "link":
                out[f"{prefix}Link"] = v
            else:
                out[f"{prefix}{k.capitalize()}"] = v
    if "teamAbbrev" not in out and "teamName" in out:
        out["teamAbbrev"] = str(out.get("teamName") or "").upper()[:3]
    # stat object -> flat stat columns
    stat_obj = row.get("stat")
    if isinstance(stat_obj, dict):
        for k, v in stat_obj.items():
            if k not in out:
                out[k] = v
    return out


def _team_ids(season: int) -> list[int]:
    """Current MLB team ids for the season (used for team-level splits)."""
    payload = fetch_page_with_retry(
        STATSAPI_TEAMS,
        {"sportId": 1, "season": season},
        headers={"User-Agent": DEFAULT_USER_AGENT},
        max_retries=MAX_RETRIES,
    )
    return [int(t["id"]) for t in payload.get("teams") or [] if t.get("id")]


def _fetch_split_rows(
    *,
    entity: str,
    group: str,
    season: int,
    sit_codes: Iterable[str],
    as_of: date,
    game_type: str,
    page_size: int = 5000,
) -> list[dict[str, Any]]:
    """
    Fetch batched statSplits rows for an entity/group.

    Player splits hit the global endpoint once per code chunk (player rows).
    Team splits hit the per-team endpoint for each team (team rows), batching
    all split codes together so we avoid one request per sitCode.
    """
    codes = [c.strip() for c in sit_codes if c.strip()]
    rows: list[dict[str, Any]] = []

    if entity == "player":
        targets: list[int | None] = [None]
    else:
        if not codes:
            return []
        targets = _team_ids(season)

    for target in targets:
        if entity == "player":
            endpoint = STATSAPI_STATS
            base_params: dict[str, Any] = {
                "stats": "statSplits",
                "group": group,
                "season": season,
                "sportId": 1,
                "gameType": game_type,
                "playerPool": "ALL",
            }
        else:
            # Team-level statSplits are served by /api/v1/teams/{id}/stats.
            # All requested split codes are batched into one request.
            endpoint = f"{STATSAPI_TEAMS}/{target}/stats"
            base_params = {
                "stats": "statSplits",
                "group": group,
                "season": season,
                "sportId": 1,
                "gameType": game_type,
            }

        for chunk in _chunked(codes, SIT_CODES_PER_REQUEST):
            params = dict(base_params)
            params["sitCodes"] = ",".join(chunk)
            offset = 0
            while True:
                params["offset"] = offset
                params["limit"] = page_size
                payload = fetch_page_with_retry(
                    endpoint,
                    params,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                    max_retries=MAX_RETRIES,
                )
                stat_block = (payload.get("stats") or [{}])[0]
                page_rows = stat_block.get("splits") or []
                for split_row in page_rows:
                    rows.append(_flatten_split_row(split_row, entity=entity))
                total = int(stat_block.get("totalSplits")
                            or 0) or offset + len(page_rows)
                offset += len(page_rows)
                if not page_rows or offset >= total:
                    break
                time.sleep(THREAD_PAUSE_S)
    return rows


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
    Fetch all rows for one entity/group/timeframe or YTD sitCode set.

    When ``sit_code`` contains a comma-separated list, that whole set is
    requested from the statSplits endpoint in manageable chunks — one round
    trip per chunk instead of one per sitCode.  Rows are flattened into the
    standard flat CSV shape (split_code/split_description included).
    """
    if entity not in {"player", "team"}:
        raise ValueError(f"entity must be player|team, got {entity!r}")
    if group not in {"hitting", "pitching"}:
        raise ValueError(f"group must be hitting|pitching, got {group!r}")
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(TIMEFRAMES)}")
    if sit_code and timeframe != "ytd":
        raise ValueError(
            "sitCodes/splits are only supported for timeframe=ytd")

    as_of = as_of or date.today()
    if not sit_code:
        return _fetch_bdfed_base(
            entity=entity,
            group=group,
            season=season,
            timeframe=timeframe,
            as_of=as_of,
            game_type=game_type,
            page_size=page_size,
        )
    return _fetch_split_rows(
        entity=entity,
        group=group,
        season=season,
        sit_codes=sit_code.split(","),
        as_of=as_of,
        game_type=game_type,
        page_size=page_size,
    )


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
    split_description = "" if split is None else str(
        split.get("description", ""))
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
    max_workers: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all requested slices; returns {"player": [...], "team": [...]}."""
    as_of = as_of or date.today()
    season = resolve_season(season, as_of)
    date_fetched = as_of.isoformat()

    entities = tuple(entities)
    groups = tuple(groups)
    timeframes = tuple(timeframes)
    requested_split_codes = {
        str(code).strip() for code in (split_codes or ()) if str(code).strip()
    }

    sit_codes = (
        load_situation_codes() if include_splits or requested_split_codes else []
    )

    # Job spec: (entity, group,, timeframe, split_set_or_None)
    # split_set: list of sit code dicts -> one batched split fetch covering
    # all codes at once.
    jobs: list[tuple[str, str, str, list[dict[str, Any]] | None]] = []
    for entity in entities:
        for group in groups:
            for timeframe in timeframes:
                jobs.append((entity, group, timeframe, None))
            if (include_splits or requested_split_codes) and (
                not requested_split_codes or (
                    entity == "player" and group == "hitting")
            ):
                applicable = [
                    s
                    for s in applicable_splits(sit_codes, entity, group)
                    if not requested_split_codes
                    or str(s.get("code") or "") in requested_split_codes
                ]
                if applicable:
                    jobs.append((entity, group, "ytd", applicable))

    total_jobs = len(jobs)
    buckets: dict[str, list[dict[str, Any]]] = {"player": [], "team": []}

    def job_label(
        entity: str, group: str, timeframe: str, split_set: list[dict[str, Any]] | None
    ) -> str:
        if split_set is None:
            return f"{entity}/{group}/{timeframe}"
        return f"{entity}/{group}/ytd/splits({len(split_set)} batched)"

    def run_job(spec: tuple[str, str, str, list[dict[str, Any]] | None], idx: int):
        entity, group, timeframe, split_set = spec
        label = job_label(entity, group, timeframe, split_set)
        if on_progress:
            on_progress(idx, total_jobs, label)
        error: str | None = None
        try:
            if split_set is None:
                rows = fetch_expanded_stats(
                    entity,
                    group,
                    season,
                    timeframe=timeframe,
                    as_of=as_of,
                    game_type=game_type,
                )
            else:
                codes = [str(s["code"]) for s in split_set]
                rows = fetch_expanded_stats(
                    entity,
                    group,
                    season,
                    timeframe="ytd",
                    sit_code=",".join(codes),
                    as_of=as_of,
                    game_type=game_type,
                )
        except Exception as err:
            if on_progress:
                on_progress(idx, total_jobs, f"{label} FAILED: {err}")
            return entity, [], label, str(err)
        return entity, rows, label, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            (idx, spec, pool.submit(run_job, spec, idx))
            for idx, spec in enumerate(jobs, start=1)
        ]
        # Process in submission order so output order is stable across runs.
        collected: list[tuple[Any, Any]] = []
        for idx, spec, fut in futures:
            entity, rows, label, error = fut.result()
            if error is not None:
                continue
            collected.append((spec, rows))

    for spec, rows in collected:
        entity, group, timeframe, split_set = spec
        if split_set is None:
            buckets[entity].extend(
                _annotate_rows(
                    rows,
                    date_fetched=date_fetched,
                    season=season,
                    entity=entity,
                    group=group,
                    timeframe=timeframe,
                    split=None,
                )
            )
            continue
        # Split rows are already flattened with split_code/split_description;
        # fill the remaining meta via the sit code table object.
        by_code = {str(s.get("code") or ""): s for s in split_set}
        for row in rows:
            code = str(row.get("split_code") or "")
            desc_obj = by_code.get(code) or {}
            row["split_code"] = code
            if not row.get("split_description"):
                row["split_description"] = str(
                    desc_obj.get("description") or "")
            row["split_menu"] = str(desc_obj.get("navigationMenu") or "")
            row["timeframe"] = "ytd"
        buckets[entity].extend(
            _annotate_rows(
                rows,
                date_fetched=date_fetched,
                season=season,
                entity=entity,
                group=group,
                timeframe="ytd",
                split=None,
            )
        )

    # Dedup identical split rows (parallel jobs could overlap codes).
    for entity in buckets:
        seen: set[tuple[Any, ...]] = set()
        dedup: list[dict[str, Any]] = []
        for row in buckets[entity]:
            key = (
                str(row.get("split_code") or ""),
                str(row.get("timeframe") or ""),
                str(row.get("season") or ""),
                str(row.get("playerId") or row.get("teamId") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(row)
        buckets[entity] = dedup

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
    """Write rows to path if it does not already exist (never clobbers)."""
    if path.exists():
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore")
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
    """Write player/team CSVs under out_dir without overwriting history."""
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--season", type=int, default=None,
                        help="MLB season year (default: current)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="Anchor date YYYY-MM-DD for rolling windows")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory (default: data)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="CSV output directory (default: <data-dir>/raw/YYYY-MM-DD)")
    parser.add_argument("--entities", nargs="+",
                        choices=["player", "team"], default=["player", "team"])
    parser.add_argument("--groups", nargs="+",
                        choices=["hitting", "pitching"], default=["hitting", "pitching"])
    parser.add_argument("--timeframes", nargs="+",
                        choices=list(TIMEFRAMES), default=list(TIMEFRAMES))
    parser.add_argument("--no-splits", action="store_true",
                        help="Skip YTD sitCode splits")
    parser.add_argument("--game-type", default="R",
                        help="gameType filter (default R)")
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
            print(
                f"Skipped {label} CSV (already exists, left intact) -> {path}")
        else:
            print(f"Wrote {n} {label} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
