"""
Fetch MLB schedule spine (gamePk + home/away + probable pitchers) for a day.

Writes data/raw/YYYY-MM-DD/games.csv — the join key for odds, results, and
later team_game / player_game panels.

Unlike teams/players snapshots, games.csv is rewritten on each run so status
and scores stay current after finals.

Usage:
  python schedule_fetcher.py
  python schedule_fetcher.py --as-of 2026-08-09
  python schedule_fetcher.py --data-dir data --dry-run
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
from datetime import date
from pathlib import Path
from typing import Any

from http_utils import default_ssl_context
from paths import raw_day_dir

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
USER_AGENT = "mlb-analysis-toolkit/1.0 (+schedule fetcher)"
MAX_RETRIES = 4
REQUEST_PAUSE_S = 0.15

GAME_COLUMNS = (
    "game_date",
    "gamePk",
    "game_datetime",
    "season",
    "game_type",
    "status",
    "abstract_state",
    "day_night",
    "doubleheader",
    "game_number",
    "venue_id",
    "venue_name",
    "home_team_id",
    "home_team_abbr",
    "home_team_name",
    "away_team_id",
    "away_team_abbr",
    "away_team_name",
    "home_probable_pitcher_id",
    "home_probable_pitcher_name",
    "away_probable_pitcher_id",
    "away_probable_pitcher_name",
    "home_score",
    "away_score",
    "home_win",
)


def _http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
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


def _side_team(entry: dict[str, Any]) -> dict[str, Any]:
    team = entry.get("team") or {}
    pitcher = entry.get("probablePitcher") or {}
    score = entry.get("score")
    is_winner = entry.get("isWinner")
    return {
        "team_id": team.get("id") if team.get("id") is not None else "",
        "team_abbr": str(team.get("abbreviation") or "").strip().upper(),
        "team_name": str(team.get("name") or ""),
        "probable_pitcher_id": pitcher.get("id") if pitcher.get("id") is not None else "",
        "probable_pitcher_name": str(pitcher.get("fullName") or ""),
        "score": "" if score is None else score,
        "is_winner": "" if is_winner is None else ("1" if is_winner else "0"),
    }


def _normalize_game(game: dict[str, Any], fallback_date: date) -> dict[str, str]:
    teams = game.get("teams") or {}
    home = _side_team(teams.get("home") or {})
    away = _side_team(teams.get("away") or {})
    status = game.get("status") or {}
    venue = game.get("venue") or {}
    official = str(game.get("officialDate") or fallback_date.isoformat())
    game_dt = str(game.get("gameDate") or "")
    return {
        "game_date": official,
        "gamePk": str(game.get("gamePk") or ""),
        "game_datetime": game_dt,
        "season": str(game.get("season") or ""),
        "game_type": str(game.get("gameType") or ""),
        "status": str(status.get("detailedState") or ""),
        "abstract_state": str(status.get("abstractGameState") or ""),
        "day_night": str(game.get("dayNight") or "").strip().lower(),
        "doubleheader": str(game.get("doubleHeader") or "N"),
        "game_number": str(game.get("gameNumber") or "1"),
        "venue_id": str(venue.get("id") or ""),
        "venue_name": str(venue.get("name") or ""),
        "home_team_id": str(home["team_id"]),
        "home_team_abbr": home["team_abbr"],
        "home_team_name": home["team_name"],
        "away_team_id": str(away["team_id"]),
        "away_team_abbr": away["team_abbr"],
        "away_team_name": away["team_name"],
        "home_probable_pitcher_id": str(home["probable_pitcher_id"]),
        "home_probable_pitcher_name": home["probable_pitcher_name"],
        "away_probable_pitcher_id": str(away["probable_pitcher_id"]),
        "away_probable_pitcher_name": away["probable_pitcher_name"],
        "home_score": str(home["score"]),
        "away_score": str(away["score"]),
        "home_win": str(home["is_winner"]),
    }


def fetch_games_for_date(game_date: date) -> list[dict[str, str]]:
    """Return one spine row per gamePk for the calendar day."""
    payload = _http_get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "date": game_date.isoformat(),
            "gameTypes": "R,F,D,L,W,C",
            "hydrate": "probablePitcher,team",
        },
    )
    rows: list[dict[str, str]] = []
    for day in payload.get("dates") or []:
        games = list(day.get("games") or [])
        games.sort(key=lambda g: (str(g.get("gameDate") or ""), int(g.get("gamePk") or 0)))
        for game in games:
            row = _normalize_game(game, game_date)
            if row["gamePk"]:
                rows.append(row)
    return rows


def write_games_csv(path: Path, rows: list[dict[str, str]], *, dry_run: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(GAME_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def populate_schedule(
    *,
    as_of: date | None = None,
    data_dir: Path = Path("data"),
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    as_of = as_of or date.today()
    day_dir = out_dir or raw_day_dir(as_of, data_dir)
    rows = fetch_games_for_date(as_of)
    path = day_dir / "games.csv"
    write_games_csv(path, rows, dry_run=dry_run)
    finals = sum(1 for r in rows if r.get("abstract_state") == "Final")
    return {
        "as_of": as_of.isoformat(),
        "path": str(path),
        "games": len(rows),
        "finals": finals,
        "dry_run": dry_run,
        "gamePks": [r["gamePk"] for r in rows],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Output folder (default: <data-dir>/raw/<as-of>)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Calendar day for the slate (default: today)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report without writing CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = populate_schedule(
        as_of=args.as_of,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )
    prefix = "Dry-run: " if args.dry_run else ""
    if summary["games"] == 0:
        print(f"{prefix}No MLB games found for {summary['as_of']}.")
        return 0
    print(
        f"{prefix}{summary['as_of']}: {summary['games']} games "
        f"({summary['finals']} final) -> {summary['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
