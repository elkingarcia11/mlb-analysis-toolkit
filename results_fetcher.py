"""
Backfill game W/L, box-score labels, and settle betting odds for prior
date_fetched days.

Designed to run the day after data_fetcher (+ odds_fetcher): for every unique
date_fetched strictly before today (or --as-of), fetch that day's MLB finals
and write onto teams.csv / players.csv:

  results              — game win/loss that day (W/L; W,L for doubleheaders)
  team_runs            — runs scored that day (comma-joined for doubleheaders)
  opp_runs             — opponent runs that day
  final_score          — team-opp label (e.g. 7-1; comma-joined for DH)
  moneyline_result     — W/L/P if moneyline odds were posted
  run_line_result      — W/L/P if run line was posted
  total_over_result    — W/L/P if total was posted
  total_under_result   — W/L/P if total was posted

Usage:
  python results_fetcher.py
  python results_fetcher.py --data-dir data
  python results_fetcher.py --as-of 2026-08-11 --dry-run
  python results_fetcher.py --force

Default layout: settle every data/raw/YYYY-MM-DD/ folder with date < as_of.
Pass --teams-csv/--players-csv to target a single pair of files instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

from http_utils import default_ssl_context
from paths import iter_raw_day_dirs

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
USER_AGENT = "mlb-analysis-toolkit/1.0 (+results fetcher)"
MAX_RETRIES = 4
REQUEST_PAUSE_S = 0.15

# Game result + box-score labels for the date_fetched slate
# (always filled when a final exists).
GAME_RESULT_COLUMN = "results"
BOX_SCORE_COLUMNS = (
    "team_runs",
    "opp_runs",
    "final_score",
)

# Betting-settlement columns written next to odds from odds_fetcher.
BET_RESULT_COLUMNS = (
    "moneyline_result",
    "run_line_result",
    "total_over_result",
    "total_under_result",
)

RESULT_COLUMNS = (GAME_RESULT_COLUMN,) + BOX_SCORE_COLUMNS + BET_RESULT_COLUMNS
GAME_LABEL_COLUMNS = (GAME_RESULT_COLUMN,) + BOX_SCORE_COLUMNS

# Odds columns required to settle the matching market.
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

FINAL_STATES = {"Final", "Game Over", "Completed Early"}


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


def _parse_date_safe(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().replace(".", "").split())


def _parse_line(value: str | None) -> float | None:
    """Parse a run line / total like '-1.5', '+1.5', 'o8.5', '8.5'."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = re.sub(r"^[ou]\s*", "", text)
    text = text.replace("½", ".5")
    try:
        return float(text)
    except ValueError:
        return None


def settle_moneyline(team_score: int, opp_score: int) -> str:
    if team_score > opp_score:
        return "W"
    if team_score < opp_score:
        return "L"
    return "P"


def settle_run_line(team_score: int, opp_score: int, line: float) -> str:
    margin = (team_score - opp_score) + line
    if margin > 0:
        return "W"
    if margin < 0:
        return "L"
    return "P"


def settle_total(team_score: int, opp_score: int, total: float, side: str) -> str:
    runs = team_score + opp_score
    if side == "over":
        if runs > total:
            return "W"
        if runs < total:
            return "L"
        return "P"
    if runs < total:
        return "W"
    if runs > total:
        return "L"
    return "P"


def fetch_scores_for_date(game_date: date) -> dict[int, list[dict[str, Any]]]:
    """
    Return teamId -> list of final game score dicts for that calendar day.

    Each entry: team_score, opp_score, opp_name, opp_id, home_away.
    Doubleheaders yield multiple entries in chronological order.
    """
    payload = _http_get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "date": game_date.isoformat(),
            "gameTypes": "R,F,D,L,W,C",
        },
    )
    by_team: dict[int, list[dict[str, Any]]] = {}

    for day in payload.get("dates") or []:
        games = list(day.get("games") or [])
        games.sort(key=lambda g: (str(g.get("gameDate") or ""), int(g.get("gamePk") or 0)))
        for game in games:
            status = (game.get("status") or {}).get("detailedState") or ""
            if status not in FINAL_STATES:
                continue
            teams = game.get("teams") or {}
            sides: dict[str, dict[str, Any]] = {}
            for side in ("away", "home"):
                entry = teams.get(side) or {}
                team = entry.get("team") or {}
                team_id = team.get("id")
                try:
                    score = int(entry.get("score"))
                except (TypeError, ValueError):
                    score = None
                if team_id is None or score is None:
                    continue
                sides[side] = {
                    "team_id": int(team_id),
                    "name": str(team.get("name") or ""),
                    "score": score,
                }
            if "away" not in sides or "home" not in sides:
                continue
            for side, other in (("away", "home"), ("home", "away")):
                own = sides[side]
                opp = sides[other]
                by_team.setdefault(own["team_id"], []).append(
                    {
                        "team_score": own["score"],
                        "opp_score": opp["score"],
                        "opp_name": opp["name"],
                        "opp_id": opp["team_id"],
                        "home_away": side,
                    }
                )

    return by_team


def _pick_game(
    games: list[dict[str, Any]],
    row: dict[str, str],
) -> dict[str, Any] | None:
    """Prefer the final matching odds_opponent / home_away; else first final."""
    if not games:
        return None
    opponent = _normalize_name(str(row.get("odds_opponent") or ""))
    home_away = str(row.get("odds_home_away") or "").strip().lower()
    if opponent:
        for game in games:
            if _normalize_name(str(game.get("opp_name") or "")) == opponent:
                return game
            # allow short-name containment ("Guardians" vs "Cleveland Guardians")
            opp_norm = _normalize_name(str(game.get("opp_name") or ""))
            if opponent in opp_norm or opp_norm in opponent:
                return game
    if home_away in {"home", "away"}:
        for game in games:
            if game.get("home_away") == home_away:
                return game
    return games[0]


def _selected_games(games: list[dict[str, Any]], row: dict[str, str]) -> list[dict[str, Any]]:
    """
    Games whose labels attach to this row.

    Prefer the final matching odds_opponent / home_away when odds context
    exists; otherwise return all finals that day (doubleheaders).
    """
    if not games:
        return []
    if _row_has_odds(row) or str(row.get("odds_opponent") or "").strip():
        game = _pick_game(games, row)
        return [game] if game else []
    return list(games)


def game_result_string(games: list[dict[str, Any]], row: dict[str, str]) -> str:
    """
    W/L for the date_fetched game. Prefer the game matching odds_opponent;
    otherwise join all finals that day (doubleheaders → 'W,L').
    """
    selected = _selected_games(games, row)
    if not selected:
        return ""
    return ",".join(
        settle_moneyline(int(g["team_score"]), int(g["opp_score"])) for g in selected
    )


def box_score_labels(games: list[dict[str, Any]], row: dict[str, str]) -> dict[str, str]:
    """
    team_runs / opp_runs / final_score for the selected game(s).
    Doubleheaders without odds context → comma-joined values.
    """
    selected = _selected_games(games, row)
    if not selected:
        return {col: "" for col in BOX_SCORE_COLUMNS}
    team_runs = [str(int(g["team_score"])) for g in selected]
    opp_runs = [str(int(g["opp_score"])) for g in selected]
    finals = [f"{t}-{o}" for t, o in zip(team_runs, opp_runs)]
    return {
        "team_runs": ",".join(team_runs),
        "opp_runs": ",".join(opp_runs),
        "final_score": ",".join(finals),
    }


def settle_row(row: dict[str, str], game: dict[str, Any]) -> dict[str, str]:
    """Compute bet-settlement columns for one odds row against one final score."""
    team_score = int(game["team_score"])
    opp_score = int(game["opp_score"])
    out = {col: "" for col in BET_RESULT_COLUMNS}

    if str(row.get("moneyline") or "").strip():
        out["moneyline_result"] = settle_moneyline(team_score, opp_score)

    line = _parse_line(row.get("run_line"))
    if line is not None:
        out["run_line_result"] = settle_run_line(team_score, opp_score, line)

    total = _parse_line(row.get("total"))
    if total is not None:
        out["total_over_result"] = settle_total(team_score, opp_score, total, "over")
        out["total_under_result"] = settle_total(team_score, opp_score, total, "under")

    return out


def _row_has_odds(row: dict[str, str]) -> bool:
    return any(str(row.get(col) or "").strip() for col in ("moneyline", "run_line", "total"))


def _row_needs_update(row: dict[str, str], *, force: bool) -> bool:
    """True if game labels or bet settlement still need filling."""
    if force:
        return True
    if any(not str(row.get(c) or "").strip() for c in GAME_LABEL_COLUMNS):
        return True
    if _row_has_odds(row) and any(not str(row.get(c) or "").strip() for c in BET_RESULT_COLUMNS):
        return True
    return False


def dates_needing_results(
    rows: list[dict[str, str]],
    *,
    before: date,
    force: bool,
) -> set[date]:
    needed: set[date] = set()
    for row in rows:
        day = _parse_date_safe(strip_date_fetched(row.get("date_fetched")))
        if day is None or day >= before:
            continue
        if _row_needs_update(row, force=force):
            needed.add(day)
    return needed


def _ensure_columns(fieldnames: list[str], columns: tuple[str, ...]) -> list[str]:
    out = list(fieldnames)
    for col in columns:
        if col not in out:
            out.append(col)
    return out


def populate_csv(
    path: Path,
    *,
    scores_by_date: dict[date, dict[int, list[dict[str, Any]]]],
    before: date,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Normalize date_fetched; fill game W/L and bet-settlement columns."""
    if not path.exists():
        raise FileNotFoundError(f"csv not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise RuntimeError(f"empty or headerless csv: {path}")
        fieldnames = _ensure_columns(list(reader.fieldnames), ODDS_COLUMNS + RESULT_COLUMNS)
        # Keep game result + box-score labels near date_fetched when newly added.
        if "date_fetched" in fieldnames:
            ordered = [c for c in GAME_LABEL_COLUMNS if c in fieldnames]
            fieldnames = [c for c in fieldnames if c not in ordered]
            idx = fieldnames.index("date_fetched") + 1
            fieldnames = fieldnames[:idx] + ordered + fieldnames[idx:]
        rows = list(reader)

    updated_rows = 0
    skipped_no_final = 0
    date_normed = 0
    teams_touched: set[str] = set()

    for row in rows:
        original = row.get("date_fetched", "")
        normalized = strip_date_fetched(original)
        if normalized != original:
            date_normed += 1
        row["date_fetched"] = normalized

        for col in ODDS_COLUMNS + RESULT_COLUMNS:
            row.setdefault(col, "")

        day = _parse_date_safe(normalized)
        if day is None or day >= before:
            continue
        if not _row_needs_update(row, force=force):
            continue

        team_raw = str(row.get("teamId") or "").strip()
        if not team_raw:
            continue
        try:
            team_id = int(float(team_raw))
        except ValueError:
            continue

        games = (scores_by_date.get(day) or {}).get(team_id) or []
        if not games:
            if force:
                for col in RESULT_COLUMNS:
                    row[col] = ""
            skipped_no_final += 1
            continue

        # Game W/L + box-score labels for that date_fetched day (independent of odds).
        if force or any(not str(row.get(c) or "").strip() for c in GAME_LABEL_COLUMNS):
            if force or not str(row.get(GAME_RESULT_COLUMN) or "").strip():
                row[GAME_RESULT_COLUMN] = game_result_string(games, row)
            if force or any(not str(row.get(c) or "").strip() for c in BOX_SCORE_COLUMNS):
                for col, value in box_score_labels(games, row).items():
                    row[col] = value

        # Bet settlement only when odds were posted on the row.
        if _row_has_odds(row) and (
            force or any(not str(row.get(c) or "").strip() for c in BET_RESULT_COLUMNS)
        ):
            game = _pick_game(games, row)
            if game:
                settled = settle_row(row, game)
                for col, value in settled.items():
                    row[col] = value

        updated_rows += 1
        teams_touched.add(str(row.get("teamAbbrev") or row.get("teamName") or team_id))

    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    return {
        "path": str(path),
        "rows_updated": updated_rows,
        "rows_no_final": skipped_no_final,
        "date_fetched_normalized": date_normed,
        "teams_touched": sorted(teams_touched),
    }


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def populate_results(
    *,
    teams_csv: Path,
    players_csv: Path,
    as_of: date | None = None,
    force: bool = False,
    dry_run: bool = False,
    scores_by_date: dict[date, dict[int, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """
    For date_fetched < as_of, write game W/L (`results`), box-score labels
    (`team_runs` / `opp_runs` / `final_score`), and settle any posted betting
    markets onto both teams and players CSVs.
    """
    as_of = as_of or date.today()

    team_rows = _load_rows(teams_csv)
    player_rows = _load_rows(players_csv)
    if not team_rows and not player_rows:
        missing = [p for p in (teams_csv, players_csv) if not p.exists()]
        raise FileNotFoundError(
            "no teams/players csv found: " + ", ".join(str(p) for p in missing)
        )

    needed = dates_needing_results(team_rows, before=as_of, force=force) | dates_needing_results(
        player_rows, before=as_of, force=force
    )

    if scores_by_date is None:
        scores_by_date = {}
        for day in sorted(needed):
            scores_by_date[day] = fetch_scores_for_date(day)
            time.sleep(REQUEST_PAUSE_S)
    else:
        # Fetch any missing days the caller did not pre-load.
        for day in sorted(needed):
            if day in scores_by_date:
                continue
            scores_by_date[day] = fetch_scores_for_date(day)
            time.sleep(REQUEST_PAUSE_S)

    summaries: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "dates_fetched": [d.isoformat() for d in sorted(needed)],
        "teams_with_finals": {
            d.isoformat(): len(v)
            for d, v in sorted(scores_by_date.items())
            if d in needed
        },
        "force": force,
        "dry_run": dry_run,
        "teams": None,
        "players": None,
        "day_dir": str(teams_csv.parent),
    }

    if teams_csv.exists():
        summaries["teams"] = populate_csv(
            teams_csv,
            scores_by_date=scores_by_date,
            before=as_of,
            force=force,
            dry_run=dry_run,
        )
    if players_csv.exists():
        summaries["players"] = populate_csv(
            players_csv,
            scores_by_date=scores_by_date,
            before=as_of,
            force=force,
            dry_run=dry_run,
        )
    return summaries


def populate_results_data_dir(
    *,
    data_dir: Path,
    as_of: date | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Settle every data/raw/YYYY-MM-DD folder (or a legacy flat data_dir)
    with calendar day < as_of. Legacy flat dirs always scan row date_fetched.
    """
    as_of = as_of or date.today()
    raw_root = data_dir / "raw"
    if raw_root.is_dir():
        day_dirs = [(day, path) for day, path in iter_raw_day_dirs(data_dir) if day < as_of]
    else:
        # smoke / legacy: CSVs live directly under data_dir
        day_dirs = list(iter_raw_day_dirs(data_dir))

    if not day_dirs:
        return {
            "as_of": as_of.isoformat(),
            "dates_fetched": [],
            "force": force,
            "dry_run": dry_run,
            "days": [],
            "teams": None,
            "players": None,
            "note": f"no raw day folders under {raw_root} with date < {as_of.isoformat()}",
        }

    needed: set[date] = set()
    for _day, path in day_dirs:
        for name in ("teams.csv", "players.csv"):
            needed |= dates_needing_results(
                _load_rows(path / name), before=as_of, force=force
            )

    scores_by_date: dict[date, dict[int, list[dict[str, Any]]]] = {}
    for day in sorted(needed):
        scores_by_date[day] = fetch_scores_for_date(day)
        time.sleep(REQUEST_PAUSE_S)

    days: list[dict[str, Any]] = []
    for _day, path in day_dirs:
        teams_csv = path / "teams.csv"
        players_csv = path / "players.csv"
        if not teams_csv.exists() and not players_csv.exists():
            continue
        days.append(
            populate_results(
                teams_csv=teams_csv,
                players_csv=players_csv,
                as_of=as_of,
                force=force,
                dry_run=dry_run,
                scores_by_date=scores_by_date,
            )
        )

    all_dates = sorted({d for part in days for d in part.get("dates_fetched", [])})
    return {
        "as_of": as_of.isoformat(),
        "dates_fetched": all_dates,
        "force": force,
        "dry_run": dry_run,
        "days": days,
        "teams": None,
        "players": None,
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
        help="Root data directory (default: data); settles each raw/YYYY-MM-DD folder",
    )
    parser.add_argument(
        "--teams-csv",
        type=Path,
        default=None,
        help="Override: settle only this teams.csv (pair with --players-csv)",
    )
    parser.add_argument(
        "--players-csv",
        type=Path,
        default=None,
        help="Override: settle only this players.csv (pair with --teams-csv)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Treat as 'today'; settle only date_fetched before it (default: today)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing non-empty settlement values",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report without writing CSV")
    return parser.parse_args(argv)


def _print_day_summary(prefix: str, summary: dict[str, Any]) -> None:
    dates = summary["dates_fetched"]
    day_label = summary.get("day_dir") or ""
    if day_label:
        print(f"{prefix}  day={day_label}")
    if not dates:
        print(f"{prefix}    (no rows needed settlement)")
        return
    print(f"{prefix}    settled date_fetched: {', '.join(dates)}")
    for label in ("teams", "players"):
        part = summary.get(label)
        if not part:
            print(f"{prefix}    {label}: (file missing, skipped)")
            continue
        print(
            f"{prefix}    {label}: updated {part['rows_updated']} rows "
            f"({len(part['teams_touched'])} teams; "
            f"{part['rows_no_final']} awaiting final / off-day) -> {part['path']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.teams_csv is not None or args.players_csv is not None:
            teams_csv = args.teams_csv or (args.data_dir / "teams.csv")
            players_csv = args.players_csv or (args.data_dir / "players.csv")
            summary = populate_results(
                teams_csv=teams_csv,
                players_csv=players_csv,
                as_of=args.as_of,
                force=args.force,
                dry_run=args.dry_run,
            )
            day_summaries = [summary]
        else:
            summary = populate_results_data_dir(
                data_dir=args.data_dir,
                as_of=args.as_of,
                force=args.force,
                dry_run=args.dry_run,
            )
            day_summaries = summary.get("days") or []
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    prefix = "Dry-run: " if args.dry_run else ""
    dates = summary["dates_fetched"]
    if not dates:
        print(
            f"{prefix}No prior date_fetched rows need game/odds results "
            f"(as_of={summary['as_of']})."
        )
        return 0

    print(
        f"{prefix}as_of={summary['as_of']}: filled results for "
        f"{len(dates)} date(s): {', '.join(dates)}"
    )
    for day_summary in day_summaries:
        _print_day_summary(prefix, day_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
