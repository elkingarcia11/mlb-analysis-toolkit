"""
Align raw daily snapshots + games spine into ML panels.

Reads:
  data/raw/YYYY-MM-DD/{teams,players,games}.csv

Writes:
  data/panels/YYYY-MM-DD/team_game.csv
  data/panels/YYYY-MM-DD/player_game.csv

team_game: one row per (gamePk, team_id)
player_game: one row per (gamePk, player_id) for players on teams in that
slate (probable pitchers flagged). Base features only (empty split_code),
pivoted across timeframe x stat_group.

Usage:
  python aligner.py --as-of 2026-08-09
  python aligner.py --raw-dir data/raw/2026-08-09 --out-dir data/panels/2026-08-09
  python aligner.py --as-of 2026-08-10 --raw-dir data/smoke --games-csv data/raw/2026-08-09/games.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from paths import panels_day_dir, raw_day_dir
from hands import ensure_hands, platoon_advantage
from park_factors import FACTOR_COLUMNS, load_park_factors

# Carried through from odds_fetcher / results_fetcher when present on raw rows.
CONTEXT_COLUMNS = (
    "moneyline",
    "run_line",
    "run_line_odds",
    "total",
    "total_over_odds",
    "total_under_odds",
    "odds_opponent",
    "odds_home_away",
    "results",
    "team_runs",
    "opp_runs",
    "final_score",
    "moneyline_result",
    "run_line_result",
    "total_over_result",
    "total_under_result",
)

# Columns that are identity / meta — never pivoted into features.
_NON_FEATURE_EXACT = {
    "date_fetched",
    "season",
    "entity",
    "stat_group",
    "timeframe",
    "split_code",
    "split_description",
    "split_menu",
    "rank",
    "type",
    "year",
    *CONTEXT_COLUMNS,
}

_NON_FEATURE_PREFIXES = (
    "player",
    "team",
    "league",
    "position",
    "primaryPosition",
)


def _is_feature_column(name: str) -> bool:
    if name in _NON_FEATURE_EXACT:
        return False
    if name.endswith("Id") or name.endswith("Name") or name.endswith("Abbrev"):
        return False
    for prefix in _NON_FEATURE_PREFIXES:
        if name == prefix or name.startswith(prefix):
            return False
    return True


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


def _fieldnames(rows: Iterable[dict[str, Any]], preferred: Iterable[str] = ()) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for col in preferred:
        if col not in seen:
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


def _base_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep overall windows only (no sitCode splits)."""
    return [r for r in rows if not str(r.get("split_code") or "").strip()]


def pivot_features(
    rows: list[dict[str, str]],
    *,
    id_field: str,
) -> dict[str, dict[str, str]]:
    """
    Pivot base-stat rows keyed by entity id.

    Feature names: {stat_group}_{timeframe}_{stat} e.g. hitting_ytd_ops.
    """
    out: dict[str, dict[str, str]] = {}
    for row in _base_rows(rows):
        entity_id = str(row.get(id_field) or "").strip()
        if not entity_id:
            continue
        group = str(row.get("stat_group") or "stats").strip() or "stats"
        timeframe = str(row.get("timeframe") or "ytd").strip() or "ytd"
        bucket = out.setdefault(entity_id, {})
        for key, value in row.items():
            if not _is_feature_column(key):
                continue
            if value is None or str(value).strip() == "":
                continue
            bucket[f"{group}_{timeframe}_{key}"] = str(value)
    return out


def pivot_split_features(
    rows: list[dict[str, str]],
    *,
    id_field: str,
) -> dict[str, dict[str, dict[str, str]]]:
    """Pivot non-empty split rows as entity -> split_code -> stat features."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        entity_id = str(row.get(id_field) or "").strip()
        split_code = str(row.get("split_code") or "").strip()
        if not entity_id or not split_code or row.get("stat_group") != "hitting":
            continue
        bucket = out.setdefault(entity_id, {}).setdefault(split_code, {})
        for key, value in row.items():
            if not _is_feature_column(key):
                continue
            if value is None or str(value).strip() == "":
                continue
            bucket[key] = str(value)
    return out


def _weekday_split_code(game_date: str) -> str:
    try:
        day = date.fromisoformat(str(game_date)[:10]).weekday()
    except ValueError:
        return ""
    return ("dmo", "dtu", "dwe", "dth", "dfr", "dsa", "dsu")[day]


def context_by_team(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Pick first non-empty odds/results context per teamId."""
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        team_id = str(row.get("teamId") or "").strip()
        if not team_id:
            continue
        bucket = out.setdefault(team_id, {c: "" for c in CONTEXT_COLUMNS})
        for col in CONTEXT_COLUMNS:
            if bucket.get(col):
                continue
            val = str(row.get(col) or "").strip()
            if val:
                bucket[col] = val
    return out


def _game_side_labels(game: dict[str, str], side: str) -> dict[str, str]:
    """Derive W/L and runs for one side from games.csv scores when final."""
    home_score = str(game.get("home_score") or "").strip()
    away_score = str(game.get("away_score") or "").strip()
    if home_score == "" or away_score == "":
        return {
            "label_team_runs": "",
            "label_opp_runs": "",
            "label_total_runs": "",
            "label_run_diff": "",
            "label_win": "",
            "label_final_score": "",
        }
    try:
        hs, aws = int(home_score), int(away_score)
    except ValueError:
        return {
            "label_team_runs": "",
            "label_opp_runs": "",
            "label_total_runs": "",
            "label_run_diff": "",
            "label_win": "",
            "label_final_score": "",
        }
    if side == "home":
        team_runs, opp_runs = hs, aws
    else:
        team_runs, opp_runs = aws, hs
    if team_runs > opp_runs:
        win = "1"
    elif team_runs < opp_runs:
        win = "0"
    else:
        win = ""
    return {
        "label_team_runs": str(team_runs),
        "label_opp_runs": str(opp_runs),
        "label_total_runs": str(team_runs + opp_runs),
        "label_run_diff": str(team_runs - opp_runs),
        "label_win": win,
        "label_final_score": f"{team_runs}-{opp_runs}",
    }


def build_team_game(
    games: list[dict[str, str]],
    team_features: dict[str, dict[str, str]],
    team_context: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in games:
        game_pk = str(game.get("gamePk") or "").strip()
        if not game_pk:
            continue
        for side, team_key, opp_key, pitcher_key in (
            (
                "home",
                "home_team_id",
                "away_team_id",
                "home_probable_pitcher_id",
            ),
            (
                "away",
                "away_team_id",
                "home_team_id",
                "away_probable_pitcher_id",
            ),
        ):
            team_id = str(game.get(team_key) or "").strip()
            opp_id = str(game.get(opp_key) or "").strip()
            if not team_id:
                continue
            row: dict[str, Any] = {
                "game_date": game.get("game_date", ""),
                "gamePk": game_pk,
                "game_datetime": game.get("game_datetime", ""),
                "season": game.get("season", ""),
                "status": game.get("status", ""),
                "abstract_state": game.get("abstract_state", ""),
                "venue_id": game.get("venue_id", ""),
                "venue_name": game.get("venue_name", ""),
                "team_id": team_id,
                "team_abbr": game.get(f"{side}_team_abbr", ""),
                "team_name": game.get(f"{side}_team_name", ""),
                "opponent_team_id": opp_id,
                "opponent_team_abbr": game.get(
                    "away_team_abbr" if side == "home" else "home_team_abbr", ""
                ),
                "opponent_team_name": game.get(
                    "away_team_name" if side == "home" else "home_team_name", ""
                ),
                "is_home": "1" if side == "home" else "0",
                "probable_pitcher_id": game.get(pitcher_key, ""),
                "probable_pitcher_name": game.get(
                    "home_probable_pitcher_name"
                    if side == "home"
                    else "away_probable_pitcher_name",
                    "",
                ),
            }
            row.update(_game_side_labels(game, side))
            ctx = team_context.get(team_id) or {}
            for col in CONTEXT_COLUMNS:
                row[f"ctx_{col}"] = ctx.get(col, "")
            # Own team features + opponent mirrors.
            own = team_features.get(team_id) or {}
            opp = team_features.get(opp_id) or {}
            for key, value in own.items():
                row[f"team_{key}"] = value
            for key, value in opp.items():
                row[f"opp_{key}"] = value
            rows.append(row)
    return rows


def _game_hour_utc(game_datetime: str) -> str:
    text = str(game_datetime or "").strip()
    if "T" not in text:
        return ""
    try:
        # 2026-08-14T23:15:00Z or with offset
        time_part = text.split("T", 1)[1]
        return str(int(time_part[0:2]))
    except (IndexError, ValueError):
        return ""


def _is_day_game(game: dict[str, str]) -> str:
    dn = str(game.get("day_night") or "").strip().lower()
    if dn == "day":
        return "1"
    if dn == "night":
        return "0"
    hour = _game_hour_utc(str(game.get("game_datetime") or ""))
    if hour == "":
        return ""
    # Rough UTC heuristic when schedule omitted dayNight.
    h = int(hour)
    return "1" if 16 <= h < 23 else "0"


def build_player_game(
    games: list[dict[str, str]],
    player_rows: list[dict[str, str]],
    player_features: dict[str, dict[str, str]],
    *,
    hands: dict[str, dict[str, str]] | None = None,
    split_features: dict[str, dict[str, dict[str, str]]] | None = None,
    park_factors: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    One row per player on a team that has a game that day.

    Adds matchup context for hitters:
      - opposing probable pitcher pitching stats (opp_pitcher_*)
      - batter bat side / pitcher throw hand / platoon flag
      - venue_id, is_day_game, game_hour_utc
    """
    hands = hands or {}
    split_features = split_features or {}
    park_factors = park_factors or {}

    # team_id -> list of game sides
    slate: dict[str, list[tuple[dict[str, str], str]]] = {}
    probable: dict[str, set[str]] = {}  # team_id -> pitcher ids
    for game in games:
        for side, team_key, pitcher_key in (
            ("home", "home_team_id", "home_probable_pitcher_id"),
            ("away", "away_team_id", "away_probable_pitcher_id"),
        ):
            team_id = str(game.get(team_key) or "").strip()
            if not team_id:
                continue
            slate.setdefault(team_id, []).append((game, side))
            pid = str(game.get(pitcher_key) or "").strip()
            if pid:
                probable.setdefault(team_id, set()).add(pid)

    # First identity glimpse per playerId from raw rows.
    identity: dict[str, dict[str, str]] = {}
    for row in player_rows:
        pid = str(row.get("playerId") or "").strip()
        if not pid or pid in identity:
            continue
        identity[pid] = {
            "player_id": pid,
            "player_name": str(
                row.get("playerFullName")
                or row.get("playerName")
                or row.get("playerUseName")
                or ""
            ),
            "team_id": str(row.get("teamId") or "").strip(),
            "team_abbr": str(row.get("teamAbbrev") or "").strip(),
            "position": str(
                row.get("primaryPositionAbbrev") or row.get("positionAbbrev") or ""
            ),
        }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for pid, meta in identity.items():
        team_id = meta["team_id"]
        if team_id not in slate:
            continue
        feats = player_features.get(pid) or {}
        batter_hands = hands.get(pid) or {}
        bat_side = str(batter_hands.get("bat_side") or "").upper()
        for game, side in slate[team_id]:
            game_pk = str(game.get("gamePk") or "").strip()
            if not game_pk:
                continue
            key = (game_pk, pid)
            if key in seen:
                continue
            seen.add(key)
            opp_side = "away" if side == "home" else "home"
            opp_pitcher_id = str(
                game.get(
                    "away_probable_pitcher_id"
                    if side == "home"
                    else "home_probable_pitcher_id",
                    "",
                )
                or ""
            ).strip()
            opp_pitcher_name = str(
                game.get(
                    "away_probable_pitcher_name"
                    if side == "home"
                    else "home_probable_pitcher_name",
                    "",
                )
                or ""
            )
            opp_hands = hands.get(opp_pitcher_id) or {}
            pitch_hand = str(opp_hands.get("pitch_hand") or "").upper()
            venue_id = str(game.get("venue_id") or "")
            park = park_factors.get(venue_id) or {}

            row: dict[str, Any] = {
                "game_date": game.get("game_date", ""),
                "gamePk": game_pk,
                "game_datetime": game.get("game_datetime", ""),
                "season": game.get("season", ""),
                "status": game.get("status", ""),
                "abstract_state": game.get("abstract_state", ""),
                "day_night": game.get("day_night", ""),
                "venue_id": venue_id,
                "venue_name": game.get("venue_name", ""),
                "player_id": pid,
                "player_name": meta["player_name"],
                "team_id": team_id,
                "team_abbr": meta["team_abbr"] or game.get(f"{side}_team_abbr", ""),
                "opponent_team_id": game.get(f"{opp_side}_team_id", ""),
                "opponent_team_abbr": game.get(f"{opp_side}_team_abbr", ""),
                "is_home": "1" if side == "home" else "0",
                "position": meta["position"],
                "is_probable_pitcher": (
                    "1" if pid in probable.get(team_id, set()) else "0"
                ),
                "is_day_game": _is_day_game(game),
                "game_hour_utc": _game_hour_utc(str(game.get("game_datetime") or "")),
                "batter_bat_side": bat_side,
                "batter_bats_R": "1" if bat_side == "R" else ("0" if bat_side else ""),
                "batter_bats_L": "1" if bat_side == "L" else ("0" if bat_side else ""),
                "batter_bats_S": "1" if bat_side == "S" else ("0" if bat_side else ""),
                "opp_probable_pitcher_id": opp_pitcher_id,
                "opp_probable_pitcher_name": opp_pitcher_name,
                "opp_pitch_hand": pitch_hand,
                "opp_throws_R": "1" if pitch_hand == "R" else ("0" if pitch_hand else ""),
                "opp_throws_L": "1" if pitch_hand == "L" else ("0" if pitch_hand else ""),
                "platoon_advantage": platoon_advantage(bat_side, pitch_hand),
                # Filled by box-score step later.
                "label_hits": "",
                "label_home_runs": "",
                "label_strikeouts": "",
                "label_at_bats": "",
                "label_innings_pitched": "",
                "label_pitcher_strikeouts": "",
            }
            for key_f, value in feats.items():
                row[f"player_{key_f}"] = value

            # Savant indices use 100 as neutral; normalize to 1.00.
            for factor_name in FACTOR_COLUMNS:
                raw_factor = str(park.get(factor_name) or "").strip()
                if not raw_factor:
                    continue
                try:
                    row[f"park_factor_{factor_name.removeprefix('index_')}"] = str(
                        float(raw_factor) / 100.0
                    )
                except ValueError:
                    continue

            # Opposing probable pitcher pitching windows only.
            opp_feats = player_features.get(opp_pitcher_id) or {}
            for key_f, value in opp_feats.items():
                if key_f.startswith("pitching_"):
                    row[f"opp_pitcher_{key_f}"] = value

            batter_splits = split_features.get(pid) or {}
            hand_code = "vl" if pitch_hand == "L" else ("vr" if pitch_hand == "R" else "")
            if hand_code:
                for key_f, value in (batter_splits.get(hand_code) or {}).items():
                    row[f"batter_vs_hand_{key_f}"] = value
            weekday_code = _weekday_split_code(str(game.get("game_date") or ""))
            if weekday_code:
                for key_f, value in (batter_splits.get(weekday_code) or {}).items():
                    row[f"batter_weekday_{key_f}"] = value
            rows.append(row)
    return rows


def align_day(
    *,
    raw_dir: Path,
    out_dir: Path,
    games_csv: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    teams_path = raw_dir / "teams.csv"
    players_path = raw_dir / "players.csv"
    games_path = games_csv or (raw_dir / "games.csv")

    teams = _load_csv(teams_path)
    players = _load_csv(players_path)
    games = _load_csv(games_path)

    if not games:
        raise FileNotFoundError(f"no games found at {games_path}")

    team_features = pivot_features(teams, id_field="teamId")
    player_features = pivot_features(players, id_field="playerId")
    player_split_features = pivot_split_features(players, id_field="playerId")
    team_context = context_by_team(teams)

    # Hands for batters on the slate + probable pitchers.
    hand_ids: set[str] = set()
    for row in players:
        pid = str(row.get("playerId") or "").strip()
        if pid:
            hand_ids.add(pid)
    for game in games:
        for key in ("home_probable_pitcher_id", "away_probable_pitcher_id"):
            pid = str(game.get(key) or "").strip()
            if pid:
                hand_ids.add(pid)
    # Infer data_dir from raw_dir (…/data/raw/YYYY-MM-DD -> …/data)
    data_dir = raw_dir.parent.parent if raw_dir.parent.name == "raw" else Path("data")
    hands = ensure_hands(hand_ids, data_dir=data_dir) if hand_ids else {}
    seasons = {
        int(str(game.get("season") or "0"))
        for game in games
        if str(game.get("season") or "").isdigit()
    }
    season = max(seasons) if seasons else date.today().year
    try:
        parks = load_park_factors(season, data_dir=data_dir)
    except Exception:
        parks = {}

    team_game = build_team_game(games, team_features, team_context)
    player_game = build_player_game(
        games,
        players,
        player_features,
        hands=hands,
        split_features=player_split_features,
        park_factors=parks,
    )

    team_path = out_dir / "team_game.csv"
    player_path = out_dir / "player_game.csv"

    team_fields = _fieldnames(
        team_game,
        preferred=[
            "game_date",
            "gamePk",
            "team_id",
            "team_abbr",
            "opponent_team_id",
            "is_home",
            "label_win",
            "label_team_runs",
            "label_opp_runs",
            "label_total_runs",
            "label_run_diff",
        ],
    )
    player_fields = _fieldnames(
        player_game,
        preferred=[
            "game_date",
            "gamePk",
            "player_id",
            "player_name",
            "team_id",
            "is_home",
            "is_probable_pitcher",
            "venue_id",
            "is_day_game",
            "game_hour_utc",
            "batter_bat_side",
            "opp_probable_pitcher_id",
            "opp_pitch_hand",
            "platoon_advantage",
            "label_hits",
            "label_home_runs",
            "label_strikeouts",
        ],
    )

    if not dry_run:
        _write_csv(team_path, team_game, team_fields)
        _write_csv(player_path, player_game, player_fields)

    return {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "games": len(games),
        "team_game_rows": len(team_game),
        "player_game_rows": len(player_game),
        "team_feature_entities": len(team_features),
        "player_feature_entities": len(player_features),
        "team_game_path": str(team_path),
        "player_game_path": str(player_path),
        "dry_run": dry_run,
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
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="Calendar day (default: today). Resolves raw/ and panels/ folders.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override folder containing teams.csv / players.csv",
    )
    parser.add_argument(
        "--games-csv",
        type=Path,
        default=None,
        help="Override path to games.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override panels output folder",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    as_of = args.as_of or date.today()
    raw_dir = args.raw_dir or raw_day_dir(as_of, args.data_dir)
    out_dir = args.out_dir or panels_day_dir(as_of, args.data_dir)
    try:
        summary = align_day(
            raw_dir=raw_dir,
            out_dir=out_dir,
            games_csv=args.games_csv,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    prefix = "Dry-run: " if args.dry_run else ""
    print(
        f"{prefix}aligned {summary['games']} games -> "
        f"{summary['team_game_rows']} team_game rows, "
        f"{summary['player_game_rows']} player_game rows"
    )
    print(f"  raw: {summary['raw_dir']}")
    print(f"  team_game: {summary['team_game_path']}")
    print(f"  player_game: {summary['player_game_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
