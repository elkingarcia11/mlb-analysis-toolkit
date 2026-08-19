"""
Detect MLB season phase from schedule (+ light calendar bias).

Phases:
  regular     — gameType R on/near as_of
  postseason  — gameType F/D/L/W/C on/near as_of
  offseason   — no slate (or deep winter empty calendar)

Usage:
  python season_phase.py
  python season_phase.py --as-of 2026-01-15
  python season_phase.py --as-of 2026-10-20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any, Iterable, Optional, Sequence, Set

from http_utils import default_ssl_context
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
USER_AGENT = "mlb-analysis-toolkit/1.0 (+season phase)"
MAX_RETRIES = 3

REGULAR_TYPES = frozenset({"R"})
POSTSEASON_TYPES = frozenset({"F", "D", "L", "W", "C"})  # WC, DS, LCS, WS, etc.
# Spring (S) / exhibition (E) are treated like no-slate for predict purposes.
IGNORE_TYPES = frozenset({"S", "E", "A"})  # A = All-Star

PHASE_REGULAR = "regular"
PHASE_POSTSEASON = "postseason"
PHASE_OFFSEASON = "offseason"
PHASES = (PHASE_REGULAR, PHASE_POSTSEASON, PHASE_OFFSEASON)


def _http_get_json(url: str, params: Optional[dict] = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=45, context=default_ssl_context()) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(min(2 ** attempt, 6))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} tries: {url}") from last_err


def _game_types_for_date(game_date: date) -> Set[str]:
    payload = _http_get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "date": game_date.isoformat(),
            "gameTypes": "R,F,D,L,W,C,S,E,A",
        },
    )
    types: Set[str] = set()
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            gt = str(game.get("gameType") or "").strip().upper()
            if gt:
                types.add(gt)
    return types


def _calendar_bias(as_of: date) -> Optional[str]:
    """Soft prior when the schedule window is empty."""
    # Deep winter: almost certainly offseason.
    if as_of.month in {12, 1, 2}:
        return PHASE_OFFSEASON
    return None


def collect_game_types(
    as_of: date,
    *,
    lookback_days: int = 1,
    lookahead_days: int = 3,
) -> Set[str]:
    """Union of game types in [as_of - lookback, as_of + lookahead]."""
    types: Set[str] = set()
    for offset in range(-lookback_days, lookahead_days + 1):
        day = as_of + timedelta(days=offset)
        try:
            types |= _game_types_for_date(day)
        except RuntimeError:
            continue
        time.sleep(0.05)
    return types


def detect_phase(
    as_of: Optional[date] = None,
    *,
    override: Optional[str] = None,
    lookback_days: int = 1,
    lookahead_days: int = 3,
    game_types: Optional[Iterable[str]] = None,
) -> dict:
    """
    Return {phase, as_of, game_types, reason, stats_season, slate_season}.

    Detection order:
      1. explicit override
      2. postseason types in window
      3. regular types in window
      4. calendar bias if empty / only spring/exhibition
      5. offseason
    """
    as_of = as_of or date.today()
    if override:
        phase = override.strip().lower()
        if phase not in PHASES:
            raise ValueError(f"override must be one of {PHASES}, got {override!r}")
        types = set(game_types or [])
        reason = f"override={phase}"
    else:
        types = set(game_types) if game_types is not None else collect_game_types(
            as_of, lookback_days=lookback_days, lookahead_days=lookahead_days
        )
        meaningful = types - IGNORE_TYPES
        if meaningful & POSTSEASON_TYPES:
            phase = PHASE_POSTSEASON
            reason = "schedule has postseason gameType(s)"
        elif meaningful & REGULAR_TYPES:
            phase = PHASE_REGULAR
            reason = "schedule has regular-season gameType R"
        else:
            bias = _calendar_bias(as_of)
            if bias:
                phase = bias
                reason = "empty/non-R slate + calendar bias"
            else:
                phase = PHASE_OFFSEASON
                reason = "no regular/postseason games in look window"

    return {
        "as_of": as_of.isoformat(),
        "phase": phase,
        "game_types": sorted(types),
        "reason": reason,
        "stats_season": stats_season(as_of, phase=phase),
        "slate_season": slate_season(as_of, phase=phase),
        # Rolling windows are noisy with no games; prefer YTD-only snapshots.
        "prefer_ytd_only": phase == PHASE_OFFSEASON,
        "skip_slate_steps": phase == PHASE_OFFSEASON,
    }


def stats_season(as_of: Optional[date] = None, *, phase: Optional[str] = None) -> int:
    """
    Season year to use when pulling cumulative / rolling stats.

    Mar–Dec: calendar year. Jan–Feb: previous calendar year (last completed
    season). Offseason uses the same rule so winter pulls the year that just
    finished (or is finishing).
    """
    as_of = as_of or date.today()
    if as_of.month >= 3:
        return as_of.year
    return as_of.year - 1


def slate_season(as_of: Optional[date] = None, *, phase: Optional[str] = None) -> int:
    """
    Season year associated with 'today's slate' folder labeling.

    Same as stats_season for now; kept separate so Jan–Feb upcoming-season
    experiments can diverge later without touching stats pulls.
    """
    return stats_season(as_of, phase=phase)


def workflow_step_policy(
    phase: str,
    *,
    games_today: Optional[int] = None,
    morning_run: bool = True,
) -> dict:
    """
    Which workflow steps make sense for this phase / slate.

    Default assumes a ~10am morning cron:
      - settle yesterday + backfill labels
      - refresh today's stats/odds/slate/predict when games exist
      - skip today's boxscores (games not played yet)

    Offseason / off day: skip slate scoring steps.
    """
    if phase == PHASE_OFFSEASON:
        return {
            "results": True,
            "backfill": True,
            "stats": True,
            "schedule": False,
            "odds": False,
            "align": False,
            "boxscore_today": False,
            "train": True,
            "predict": False,
        }
    policy = {
        "results": True,
        "backfill": True,
        "stats": True,
        "schedule": True,
        "odds": True,
        "align": True,
        # Morning run: finals are yesterday's job via backfill.
        "boxscore_today": not morning_run,
        "train": True,
        "predict": True,
    }
    if games_today == 0:
        policy["odds"] = False
        policy["align"] = False
        policy["boxscore_today"] = False
        policy["predict"] = False
    return policy


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--phase",
        choices=list(PHASES),
        default=None,
        help="Override auto-detect",
    )
    parser.add_argument("--lookback-days", type=int, default=1)
    parser.add_argument("--lookahead-days", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    info = detect_phase(
        args.as_of,
        override=args.phase,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
    )
    print(json.dumps(info, indent=2))
    policy = workflow_step_policy(info["phase"])
    skipped = [k for k, v in policy.items() if not v]
    if skipped:
        print(f"workflow skips: {', '.join(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
