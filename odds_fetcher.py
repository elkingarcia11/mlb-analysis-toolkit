"""
Fetch today's MLB moneyline, run line, and total odds and write them onto
teams.csv and players.csv rows whose date_fetched matches that calendar day.

Designed to run in the morning before games begin (same day as data_fetcher):
capture the current pre-game DraftKings markets from the ESPN scoreboard and
stamp them onto that day's date_fetched snapshot. results_fetcher settles
those markets on later days.

Source: ESPN scoreboard DraftKings markets (no API key required). ESPN's
"close" field is the current line before first pitch; "open" is opening only.

Usage:
  python odds_fetcher.py
  python odds_fetcher.py --data-dir data --as-of 2026-08-11
  python odds_fetcher.py --dry-run

Default CSVs live under data/raw/YYYY-MM-DD/ (same day as --as-of / today).
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
from datetime import date, datetime
from pathlib import Path
from typing import Any

from http_utils import default_ssl_context
from paths import resolve_day_csvs

ESPN_SCOREBOARD = "https://site.web.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
MAX_RETRIES = 4

# Team-perspective market columns written onto teams.csv.
ODDS_COLUMNS = (
    "moneyline",
    "run_line",
    "run_line_odds",
    "total",
    "total_over_odds",
    "total_under_odds",
    "odds_opponent",
    "odds_home_away",
)

# ESPN abbrev -> MLB / stats abbreviation aliases used in teams.csv
ABBREV_ALIASES: dict[str, set[str]] = {
    "ARI": {"ARI", "AZ"},
    "AZ": {"ARI", "AZ"},
    "CHW": {"CHW", "CWS", "CHA"},
    "CWS": {"CHW", "CWS", "CHA"},
    "WSH": {"WSH", "WAS", "WSN"},
    "WAS": {"WSH", "WAS", "WSN"},
    "ATH": {"ATH", "OAK"},
    "OAK": {"ATH", "OAK"},
    "SD": {"SD", "SDP"},
    "SF": {"SF", "SFG"},
    "TB": {"TB", "TBR", "TBD"},
    "KC": {"KC", "KCR"},
    "NYM": {"NYM", "NYN"},
    "NYY": {"NYY", "NYA"},
    "LAD": {"LAD", "LAN"},
    "LAA": {"LAA", "ANA", "CAL"},
    "CHC": {"CHC", "CHN"},
}


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.espn.com",
            "Referer": "https://www.espn.com/",
        },
    )
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60, context=default_ssl_context()) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}") from last_err


def strip_date_fetched(value: str | None) -> str:
    """Normalize date_fetched to YYYY-MM-DD (drop any time / timezone)."""
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        if "T" in text:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text[:10]


def _close_odds(block: dict[str, Any] | None, side: str) -> tuple[str, str]:
    """Return (line, american_odds) from a moneyline/pointSpread/total side block."""
    if not block:
        return "", ""
    side_data = block.get(side) or {}
    close = side_data.get("close") or {}
    line = str(close.get("line") or "").strip()
    odds = str(close.get("odds") or "").strip()
    return line, odds


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def _abbrev_keys(abbr: str) -> set[str]:
    abbr = abbr.strip().upper()
    keys = {abbr}
    keys.update(ABBREV_ALIASES.get(abbr, set()))
    return keys


def fetch_odds_for_date(as_of: date | None = None) -> dict[str, dict[str, str]]:
    """
    Fetch current pre-game run line / total / moneyline for MLB matchups on as_of.

    Intended for a morning run before first pitch. Returns a mapping keyed by
    team abbrev aliases and by normalized team name. If there are no games
    (or markets not posted yet), returns {}.
    """
    as_of = as_of or date.today()
    payload = _http_get_json(
        ESPN_SCOREBOARD,
        {"dates": as_of.strftime("%Y%m%d")},
    )
    events = payload.get("events") or []
    if not events:
        return {}

    by_key: dict[str, dict[str, str]] = {}

    for event in events:
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0]
        odds_list = comp.get("odds") or []
        if not odds_list:
            continue
        odds = odds_list[0]
        competitors = {
            c.get("homeAway"): c
            for c in (comp.get("competitors") or [])
            if c.get("homeAway") in {"home", "away"}
        }
        if "home" not in competitors or "away" not in competitors:
            continue

        home = competitors["home"]["team"]
        away = competitors["away"]["team"]
        home_abbr = str(home.get("abbreviation") or "").strip().upper()
        away_abbr = str(away.get("abbreviation") or "").strip().upper()
        home_name = str(home.get("displayName") or "")
        away_name = str(away.get("displayName") or "")

        ml_home = _close_odds(odds.get("moneyline"), "home")[1]
        ml_away = _close_odds(odds.get("moneyline"), "away")[1]
        rl_home_line, rl_home_odds = _close_odds(odds.get("pointSpread"), "home")
        rl_away_line, rl_away_odds = _close_odds(odds.get("pointSpread"), "away")
        if not rl_home_line and odds.get("spread") is not None:
            spread = float(odds["spread"])
            rl_home_line = f"{spread:+g}"
            rl_away_line = f"{-spread:+g}"
        over_line, over_odds = _close_odds(odds.get("total"), "over")
        under_line, under_odds = _close_odds(odds.get("total"), "under")
        total_line = ""
        if over_line:
            total_line = over_line.lstrip("oO").strip()
        elif under_line:
            total_line = under_line.lstrip("uU").strip()
        elif odds.get("overUnder") is not None:
            total_line = str(odds["overUnder"])

        home_row = {
            "moneyline": ml_home,
            "run_line": rl_home_line,
            "run_line_odds": rl_home_odds,
            "total": total_line,
            "total_over_odds": over_odds,
            "total_under_odds": under_odds,
            "odds_opponent": away_name,
            "odds_home_away": "home",
        }
        away_row = {
            "moneyline": ml_away,
            "run_line": rl_away_line,
            "run_line_odds": rl_away_odds,
            "total": total_line,
            "total_over_odds": over_odds,
            "total_under_odds": under_odds,
            "odds_opponent": home_name,
            "odds_home_away": "away",
        }

        for abbr, name, row in (
            (home_abbr, home_name, home_row),
            (away_abbr, away_name, away_row),
        ):
            for key in _abbrev_keys(abbr):
                by_key[f"abbr:{key}"] = row
            if name:
                by_key[f"name:{_normalize_name(name)}"] = row
            short = str((home if abbr == home_abbr else away).get("name") or "").strip()
            if short:
                by_key[f"short:{_normalize_name(short)}"] = row

    return by_key


def _lookup_team_odds(row: dict[str, str], odds_by_key: dict[str, dict[str, str]]) -> dict[str, str] | None:
    abbr = str(row.get("teamAbbrev") or "").strip().upper()
    for key in _abbrev_keys(abbr):
        hit = odds_by_key.get(f"abbr:{key}")
        if hit:
            return hit
    for field in ("teamName", "teamShortName", "shortName"):
        name = str(row.get(field) or "").strip()
        if not name:
            continue
        hit = odds_by_key.get(f"name:{_normalize_name(name)}")
        if hit:
            return hit
        hit = odds_by_key.get(f"short:{_normalize_name(name)}")
        if hit:
            return hit
    return None


def populate_csv(
    path: Path,
    *,
    odds_by_key: dict[str, dict[str, str]],
    as_of: date,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Strip date_fetched to YYYY-MM-DD and fill odds columns for rows matching as_of."""
    as_of_str = as_of.isoformat()
    if not path.exists():
        raise FileNotFoundError(f"csv not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise RuntimeError(f"empty or headerless csv: {path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for col in ODDS_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    matched_teams: set[str] = set()
    updated_rows = 0
    date_normed = 0

    for row in rows:
        original = row.get("date_fetched", "")
        normalized = strip_date_fetched(original)
        if normalized != original:
            date_normed += 1
        row["date_fetched"] = normalized

        for col in ODDS_COLUMNS:
            row.setdefault(col, "")

        if normalized != as_of_str or not odds_by_key:
            continue

        odds = _lookup_team_odds(row, odds_by_key)
        if not odds:
            continue

        for col in ODDS_COLUMNS:
            row[col] = odds.get(col, "")
        updated_rows += 1
        label = row.get("teamAbbrev") or row.get("teamName") or "?"
        matched_teams.add(str(label))

    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return {
        "rows_updated": updated_rows,
        "teams_matched": sorted(matched_teams),
        "date_fetched_normalized": date_normed,
        "path": str(path),
    }


def populate_teams_csv(
    teams_csv: Path,
    *,
    as_of: date | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Back-compat wrapper: fill odds on a single teams.csv."""
    as_of = as_of or date.today()
    odds_by_key = fetch_odds_for_date(as_of)
    part = populate_csv(teams_csv, odds_by_key=odds_by_key, as_of=as_of, dry_run=dry_run)
    matchup_ids = {
        (v.get("odds_opponent"), v.get("odds_home_away"), v.get("moneyline"), v.get("total"))
        for v in odds_by_key.values()
    }
    return {
        "as_of": as_of.isoformat(),
        "matchups_with_odds": max(len(matchup_ids) // 2, 0),
        "teams_matched": part["teams_matched"],
        "rows_updated": part["rows_updated"],
        "date_fetched_normalized": part["date_fetched_normalized"],
        "games_today": bool(odds_by_key),
        "dry_run": dry_run,
        "path": part["path"],
    }


def populate_odds(
    *,
    teams_csv: Path,
    players_csv: Path | None = None,
    as_of: date | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch odds once and write onto teams.csv and optionally players.csv."""
    as_of = as_of or date.today()
    odds_by_key = fetch_odds_for_date(as_of)
    matchup_ids = {
        (v.get("odds_opponent"), v.get("odds_home_away"), v.get("moneyline"), v.get("total"))
        for v in odds_by_key.values()
    }
    summary: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "matchups_with_odds": max(len(matchup_ids) // 2, 0),
        "games_today": bool(odds_by_key),
        "dry_run": dry_run,
        "teams": None,
        "players": None,
    }
    if teams_csv.exists():
        summary["teams"] = populate_csv(
            teams_csv, odds_by_key=odds_by_key, as_of=as_of, dry_run=dry_run
        )
    if players_csv is not None and players_csv.exists():
        summary["players"] = populate_csv(
            players_csv, odds_by_key=odds_by_key, as_of=as_of, dry_run=dry_run
        )
    if summary["teams"] is None and summary["players"] is None:
        raise FileNotFoundError(f"csv not found: {teams_csv}")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data); CSVs resolved under raw/YYYY-MM-DD",
    )
    parser.add_argument(
        "--teams-csv",
        type=Path,
        default=None,
        help="Override path to teams.csv (default: <data-dir>/raw/<as-of>/teams.csv)",
    )
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=None,
        help="Override path to players.csv (default: <data-dir>/raw/<as-of>/players.csv)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Calendar day to fetch odds for / match on date_fetched (default: today)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report without writing CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    as_of = args.as_of or date.today()
    teams_csv, players_csv = resolve_day_csvs(
        data_dir=args.data_dir,
        as_of=as_of,
        teams_csv=args.teams_csv,
        players_csv=args.players_csv,
    )
    try:
        summary = populate_odds(
            teams_csv=teams_csv,
            players_csv=players_csv,
            as_of=as_of,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    prefix = "Dry-run: " if args.dry_run else ""
    if not summary["games_today"]:
        print(f"{prefix}No MLB odds found for {summary['as_of']} (no games or markets not posted yet).")
        return 0

    print(
        f"{prefix}{summary['as_of']}: {summary['matchups_with_odds']} matchups with odds."
    )
    for label in ("teams", "players"):
        part = summary.get(label)
        if not part:
            print(f"  {label}: (file missing, skipped)")
            continue
        print(
            f"  {label}: updated {part['rows_updated']} rows "
            f"across {len(part['teams_matched'])} teams -> {part['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
