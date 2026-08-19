"""
Fetch MLB box scores for a day's games and stamp player-level labels.

Writes:
  data/raw/YYYY-MM-DD/boxscores.csv          — one row per player appearance
  data/panels/YYYY-MM-DD/player_game.csv     — fills label_* columns when present

Team-level runs / win labels already come from games.csv via the aligner.

Usage:
  python boxscore_fetcher.py --as-of 2026-08-09
  python boxscore_fetcher.py --as-of 2026-08-09 --dry-run
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
from paths import panels_day_dir, raw_day_dir

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
USER_AGENT = "mlb-analysis-toolkit/1.0 (+boxscore fetcher)"
MAX_RETRIES = 4
REQUEST_PAUSE_S = 0.15

BOXSCORE_COLUMNS = (
    "game_date",
    "gamePk",
    "team_id",
    "team_side",
    "player_id",
    "player_name",
    "batting_order",
    "is_batter",
    "is_pitcher",
    "hits",
    "home_runs",
    "strikeouts",
    "at_bats",
    "runs",
    "rbi",
    "walks",
    "innings_pitched",
    "pitcher_strikeouts",
    "earned_runs",
    "hits_allowed",
    "home_runs_allowed",
    "walks_allowed",
    "pitches_thrown",
)

# player_game label columns filled from box scores.
LABEL_MAP = {
    "label_hits": "hits",
    "label_home_runs": "home_runs",
    "label_strikeouts": "strikeouts",
    "label_at_bats": "at_bats",
    "label_runs": "runs",
    "label_rbi": "rbi",
    "label_innings_pitched": "innings_pitched",
    "label_pitcher_strikeouts": "pitcher_strikeouts",
    "label_earned_runs": "earned_runs",
}


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


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _num(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_boxscore_side(
    *,
    game_pk: str,
    game_date: str,
    side: str,
    block: dict[str, Any],
) -> list[dict[str, str]]:
    team = block.get("team") or {}
    team_id = str(team.get("id") or "")
    batters = {str(x) for x in (block.get("batters") or [])}
    pitchers = {str(x) for x in (block.get("pitchers") or [])}
    players = block.get("players") or {}
    rows: list[dict[str, str]] = []

    for key, player in players.items():
        person = player.get("person") or {}
        player_id = str(person.get("id") or key.replace("ID", ""))
        if not player_id:
            continue
        stats = player.get("stats") or {}
        batting = stats.get("batting") or {}
        pitching = stats.get("pitching") or {}
        is_batter = player_id in batters or bool(batting.get("plateAppearances") or batting.get("atBats"))
        is_pitcher = player_id in pitchers or bool(pitching.get("inningsPitched"))
        if not is_batter and not is_pitcher:
            continue
        batting_order = player.get("battingOrder")
        rows.append(
            {
                "game_date": game_date,
                "gamePk": game_pk,
                "team_id": team_id,
                "team_side": side,
                "player_id": player_id,
                "player_name": str(person.get("fullName") or ""),
                "batting_order": "" if batting_order is None else str(batting_order),
                "is_batter": "1" if is_batter else "0",
                "is_pitcher": "1" if is_pitcher else "0",
                "hits": _num(batting.get("hits")) if is_batter else "",
                "home_runs": _num(batting.get("homeRuns")) if is_batter else "",
                "strikeouts": _num(batting.get("strikeOuts")) if is_batter else "",
                "at_bats": _num(batting.get("atBats")) if is_batter else "",
                "runs": _num(batting.get("runs")) if is_batter else "",
                "rbi": _num(batting.get("rbi")) if is_batter else "",
                "walks": _num(batting.get("baseOnBalls")) if is_batter else "",
                "innings_pitched": _num(pitching.get("inningsPitched")) if is_pitcher else "",
                "pitcher_strikeouts": _num(pitching.get("strikeOuts")) if is_pitcher else "",
                "earned_runs": _num(pitching.get("earnedRuns")) if is_pitcher else "",
                "hits_allowed": _num(pitching.get("hits")) if is_pitcher else "",
                "home_runs_allowed": _num(pitching.get("homeRuns")) if is_pitcher else "",
                "walks_allowed": _num(pitching.get("baseOnBalls")) if is_pitcher else "",
                "pitches_thrown": _num(pitching.get("pitchesThrown")) if is_pitcher else "",
            }
        )
    return rows


def fetch_boxscore_rows(game_pk: str, game_date: str = "") -> list[dict[str, str]]:
    payload = _http_get_json(BOXSCORE_URL.format(game_pk=game_pk))
    teams = payload.get("teams") or {}
    rows: list[dict[str, str]] = []
    for side in ("away", "home"):
        block = teams.get(side) or {}
        if not block:
            continue
        rows.extend(
            _parse_boxscore_side(
                game_pk=str(game_pk),
                game_date=game_date,
                side=side,
                block=block,
            )
        )
    return rows


def fetch_day_boxscores(games: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for game in games:
        game_pk = str(game.get("gamePk") or "").strip()
        if not game_pk:
            continue
        abstract = str(game.get("abstract_state") or "").strip()
        status = str(game.get("status") or "").strip()
        if abstract != "Final" and status not in {"Final", "Game Over", "Completed Early"}:
            continue
        game_date = str(game.get("game_date") or "")
        rows.extend(fetch_boxscore_rows(game_pk, game_date))
        time.sleep(REQUEST_PAUSE_S)
    return rows


def stamp_player_game_labels(
    player_game_path: Path,
    box_rows: list[dict[str, str]],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not player_game_path.exists():
        return {
            "path": str(player_game_path),
            "rows_updated": 0,
            "missing_panel": True,
        }

    rows = _load_csv(player_game_path)
    if not rows:
        return {"path": str(player_game_path), "rows_updated": 0, "missing_panel": False}

    by_key: dict[tuple[str, str], dict[str, str]] = {
        (str(r.get("gamePk") or ""), str(r.get("player_id") or "")): r for r in box_rows
    }

    fieldnames = list(rows[0].keys())
    for col in LABEL_MAP:
        if col not in fieldnames:
            fieldnames.append(col)
    if "label_appeared" not in fieldnames:
        fieldnames.append("label_appeared")

    updated = 0
    for row in rows:
        for col in LABEL_MAP:
            row.setdefault(col, "")
        row.setdefault("label_appeared", "")
        key = (str(row.get("gamePk") or ""), str(row.get("player_id") or ""))
        box = by_key.get(key)
        if not box:
            continue
        for label_col, box_col in LABEL_MAP.items():
            row[label_col] = box.get(box_col, "")
        row["label_appeared"] = "1"
        updated += 1

    if not dry_run:
        _write_csv(player_game_path, rows, fieldnames)

    return {
        "path": str(player_game_path),
        "rows_updated": updated,
        "missing_panel": False,
        "panel_rows": len(rows),
    }


def populate_boxscores(
    *,
    as_of: date | None = None,
    data_dir: Path = Path("data"),
    raw_dir: Path | None = None,
    panels_dir: Path | None = None,
    games_csv: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    as_of = as_of or date.today()
    day_raw = raw_dir or raw_day_dir(as_of, data_dir)
    day_panels = panels_dir or panels_day_dir(as_of, data_dir)
    games_path = games_csv or (day_raw / "games.csv")
    games = _load_csv(games_path)
    if not games:
        raise FileNotFoundError(f"no games found at {games_path}")

    box_rows = fetch_day_boxscores(games)
    box_path = day_raw / "boxscores.csv"
    if not dry_run:
        _write_csv(box_path, box_rows, list(BOXSCORE_COLUMNS))

    stamp = stamp_player_game_labels(
        day_panels / "player_game.csv",
        box_rows,
        dry_run=dry_run,
    )
    return {
        "as_of": as_of.isoformat(),
        "games": len(games),
        "boxscore_rows": len(box_rows),
        "boxscores_path": str(box_path),
        "players_labeled": len({(r["gamePk"], r["player_id"]) for r in box_rows}),
        "player_game": stamp,
        "dry_run": dry_run,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--panels-dir", type=Path, default=None)
    parser.add_argument("--games-csv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = populate_boxscores(
            as_of=args.as_of,
            data_dir=args.data_dir,
            raw_dir=args.raw_dir,
            panels_dir=args.panels_dir,
            games_csv=args.games_csv,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    prefix = "Dry-run: " if args.dry_run else ""
    print(
        f"{prefix}{summary['as_of']}: {summary['boxscore_rows']} boxscore player rows "
        f"across {summary['games']} games -> {summary['boxscores_path']}"
    )
    pg = summary["player_game"]
    if pg.get("missing_panel"):
        print(f"  player_game panel missing (skipped labels): {pg['path']}")
    else:
        print(
            f"  player_game: labeled {pg['rows_updated']} / {pg.get('panel_rows', '?')} rows "
            f"-> {pg['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
