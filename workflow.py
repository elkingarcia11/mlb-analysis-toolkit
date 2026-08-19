"""
Morning MLB analysis workflow (~10am daily cron, year-round).

Designed to run every morning around 10:00 local time:

  1. Detect season phase (regular / postseason / offseason)
  2. Settle yesterday's W/L + bet results
  3. Backfill yesterday's schedule / panels / boxscore labels
  4. Fetch today's pre-game stats (skip if already snapped)
  5. Fetch today's slate + odds (skip on off days / offseason)
  6. Align today's panels and predict
  7. Train only when labeled panels are newer than saved models

Today's boxscores are skipped by default (games haven't been played at 10am).
Use --include-today-boxscores only for afternoon/evening replays.

Cron example (10:00 every day):
  0 10 * * * cd /path/to/mlb-analysis-toolkit && python3 workflow.py >> logs/workflow.log 2>&1
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from paths import iter_raw_day_dirs, panels_day_dir, raw_day_dir
from season_phase import PHASES, detect_phase, workflow_step_policy


StepFn = Callable[[], Optional[Dict[str, Any]]]
WEEKDAY_SPLIT_CODES = ("dmo", "dtu", "dwe", "dth", "dfr", "dsa", "dsu")


def _print_step(name: str) -> None:
    print(f"\n==> {name}")


def _safe_call(name: str, fn: StepFn) -> dict:
    _print_step(name)
    try:
        result = fn() or {}
        result.setdefault("ok", True)
        result.setdefault("step", name)
        return result
    except Exception as err:
        print(f"FAILED: {err}", file=sys.stderr)
        traceback.print_exc(limit=2)
        return {"ok": False, "step": name, "error": str(err)}


def _skip_step(name: str, reason: str) -> dict:
    _print_step(f"{name} (skipped)")
    print(f"  {reason}")
    return {"ok": True, "step": name, "skipped": True, "reason": reason}


def _day_stats_exist(data_dir: Path, as_of: date) -> bool:
    day = raw_day_dir(as_of, data_dir)
    return (day / "teams.csv").exists() and (day / "players.csv").exists()


def _count_games_csv(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sum(1 for r in rows if str(r.get("gamePk") or "").strip())


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


FINAL_STATES = {"Final", "Game Over", "Completed Early"}


def _unsettled_games(games_csv: Path) -> int:
    """Games on a past slate that still lack a final state or final score."""
    unsettled = 0
    for row in _read_csv(games_csv):
        if not str(row.get("gamePk") or "").strip():
            continue
        final = (
            str(row.get("abstract_state") or "").strip() == "Final"
            or str(row.get("status") or "").strip() in FINAL_STATES
        )
        scored = bool(str(row.get("home_score") or "").strip()) and bool(
            str(row.get("away_score") or "").strip()
        )
        if not final or not scored:
            unsettled += 1
    return unsettled


def _unlabeled_rows(panel_csv: Path, label_col: str) -> int:
    return sum(
        1 for row in _read_csv(panel_csv) if not str(row.get(label_col) or "").strip()
    )


def _boxscore_games(box_csv: Path) -> int:
    return len(
        {
            str(row.get("gamePk") or "")
            for row in _read_csv(box_csv)
            if str(row.get("gamePk") or "").strip()
        }
    )


def _final_games(games_csv: Path) -> int:
    return sum(
        1
        for row in _read_csv(games_csv)
        if str(row.get("gamePk") or "").strip()
        and (
            str(row.get("abstract_state") or "").strip() == "Final"
            or str(row.get("status") or "").strip() in FINAL_STATES
        )
    )


def _latest_mtime(paths: List[Path]) -> Optional[float]:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else None


def labeled_panels_newer_than_models(data_dir: Path, model_dir: Path) -> dict:
    """
    True when any panel CSV is newer than the newest model artifact, or when
    models are missing while panels exist.
    """
    panel_files: List[Path] = []
    panels_root = data_dir / "panels"
    if panels_root.is_dir():
        for day_dir in panels_root.iterdir():
            if not day_dir.is_dir():
                continue
            for name in ("team_game.csv", "player_game.csv"):
                panel_files.append(day_dir / name)

    model_files = list(model_dir.glob("*.pkl")) if model_dir.is_dir() else []
    panel_mtime = _latest_mtime(panel_files)
    model_mtime = _latest_mtime(model_files)

    if panel_mtime is None:
        return {
            "needed": False,
            "reason": "no panels on disk",
            "panel_mtime": None,
            "model_mtime": model_mtime,
        }
    if model_mtime is None:
        return {
            "needed": True,
            "reason": "no saved models yet",
            "panel_mtime": panel_mtime,
            "model_mtime": None,
        }
    if panel_mtime > model_mtime:
        return {
            "needed": True,
            "reason": "labeled panels newer than models",
            "panel_mtime": panel_mtime,
            "model_mtime": model_mtime,
        }
    return {
        "needed": False,
        "reason": "models up to date vs panels",
        "panel_mtime": panel_mtime,
        "model_mtime": model_mtime,
    }


def backfill_prior_days(
    *,
    data_dir: Path,
    as_of: date,
    force_boxscores: bool = False,
) -> dict[str, Any]:
    """Ensure prior raw days have games.csv, panels, and boxscore labels."""
    import aligner
    import boxscore_fetcher
    import schedule_fetcher

    raw_root = data_dir / "raw"
    day_dirs = [(d, p) for d, p in iter_raw_day_dirs(data_dir) if d < as_of]
    yesterday = as_of - timedelta(days=1)
    ydir = raw_day_dir(yesterday, data_dir)
    if not any(d == yesterday for d, _ in day_dirs):
        day_dirs.append((yesterday, ydir))
    day_dirs = sorted({d: p for d, p in day_dirs}.items(), key=lambda x: x[0])

    summaries: list[dict[str, Any]] = []
    for day, path in day_dirs:
        entry: dict[str, Any] = {"day": day.isoformat()}
        games_csv = path / "games.csv"
        teams_csv = path / "teams.csv"

        # A morning cron writes games.csv while the slate is still "Preview",
        # so past days need a re-pull to pick up final states and scores.
        if not games_csv.exists() or _unsettled_games(games_csv):
            entry["schedule"] = schedule_fetcher.populate_schedule(
                as_of=day, data_dir=data_dir, out_dir=path
            )

        panels = panels_day_dir(day, data_dir)
        team_panel = panels / "team_game.csv"
        player_panel = panels / "player_game.csv"
        need_align = (
            teams_csv.exists()
            and games_csv.exists()
            and _count_games_csv(games_csv) > 0
            and (
                not team_panel.exists()
                or not player_panel.exists()
                # panels built pre-game carry empty team labels
                or _unlabeled_rows(team_panel, "label_win")
            )
        )
        if need_align:
            entry["align"] = aligner.align_day(
                raw_dir=path,
                out_dir=panels,
                games_csv=games_csv if games_csv.exists() else None,
            )

        box_path = path / "boxscores.csv"
        if (
            games_csv.exists()
            and _count_games_csv(games_csv) > 0
            and (
                force_boxscores
                or not box_path.exists()
                or _boxscore_games(box_path) < _final_games(games_csv)
            )
        ):
            entry["boxscores"] = boxscore_fetcher.populate_boxscores(
                as_of=day,
                data_dir=data_dir,
                raw_dir=path,
                panels_dir=panels,
                games_csv=games_csv,
            )
        summaries.append(entry)

    return {
        "days": len(summaries),
        "as_of": as_of.isoformat(),
        "raw_root": str(raw_root),
        "details": summaries,
    }


def run_workflow(
    *,
    as_of: Optional[date] = None,
    data_dir: Path = Path("data"),
    skip_fetch: bool = False,
    skip_train: bool = False,
    skip_predict: bool = False,
    skip_backfill: bool = False,
    only: Optional[str] = None,
    targets: Optional[List[str]] = None,
    no_splits: bool = False,
    season_phase_override: Optional[str] = None,
    force_stats: bool = False,
    force_train: bool = False,
    include_today_boxscores: bool = False,
) -> dict:
    as_of = as_of or date.today()
    morning_run = not include_today_boxscores

    phase_info = detect_phase(as_of, override=season_phase_override)
    policy = workflow_step_policy(phase_info["phase"], morning_run=morning_run)
    print(
        f"morning_run=10am_profile phase={phase_info['phase']} "
        f"stats_season={phase_info['stats_season']} "
        f"({phase_info['reason']}; types={phase_info['game_types'] or '[]'})"
    )

    results: dict = {
        "as_of": as_of.isoformat(),
        "phase": phase_info,
        "games_today": None,
        "mode": "morning_10am",
        "steps": [],
    }

    def add(step: dict) -> None:
        results["steps"].append(step)

    def allowed(step_key: str) -> bool:
        if only:
            if step_key == "boxscore_today":
                if only != "align":
                    return False
            elif step_key != only:
                return False
        return bool(policy.get(step_key, True))

    steps_wanted = {
        "results",
        "backfill",
        "stats",
        "schedule",
        "odds",
        "align",
        "train",
        "predict",
    }
    if only:
        steps_wanted = {only}

    # 1) Settle prior bet/game results.
    if "results" in steps_wanted and not skip_fetch:
        if not allowed("results"):
            add(_skip_step("results_fetcher", f"phase={phase_info['phase']}"))
        else:
            import results_fetcher

            add(
                _safe_call(
                    "results_fetcher",
                    lambda: results_fetcher.populate_results_data_dir(
                        data_dir=data_dir, as_of=as_of
                    ),
                )
            )

    # 2) Backfill prior days.
    if "backfill" in steps_wanted and not skip_backfill and not skip_fetch:
        if not allowed("backfill"):
            add(_skip_step("backfill_prior_days", f"phase={phase_info['phase']}"))
        else:
            add(
                _safe_call(
                    "backfill_prior_days",
                    lambda: backfill_prior_days(data_dir=data_dir, as_of=as_of),
                )
            )

    # 3) Today's stats — skip network when today's snapshot already exists.
    if "stats" in steps_wanted and not skip_fetch:
        if not allowed("stats"):
            add(_skip_step("data_fetcher", f"phase={phase_info['phase']}"))
        elif _day_stats_exist(data_dir, as_of) and not force_stats:
            add(
                _skip_step(
                    "data_fetcher",
                    f"snapshot already exists at {raw_day_dir(as_of, data_dir)}",
                )
            )
        else:
            import data_fetcher

            def _stats() -> dict:
                timeframes = ("ytd",) if phase_info.get("prefer_ytd_only") else None
                matchup_split_codes = (
                    "vl",
                    "vr",
                    WEEKDAY_SPLIT_CODES[as_of.weekday()],
                )
                kwargs = {
                    "as_of": as_of,
                    "season": phase_info["stats_season"],
                    # Daily cron fetches only the matchup splits needed by the
                    # models. --no-splits suppresses the large all-splits job.
                    "include_splits": False,
                    "split_codes": matchup_split_codes,
                }
                if timeframes is not None:
                    kwargs["timeframes"] = timeframes
                data = data_fetcher.fetch_requested(**kwargs)
                exported = data_fetcher.export_csvs(data, raw_day_dir(as_of, data_dir))
                return {
                    label: {"path": str(path), "status": status, "rows": n}
                    for label, (path, status, n) in exported.items()
                }

            add(_safe_call("data_fetcher", _stats))

    # 4) Today's schedule — establishes games_today for idle off days.
    games_today: Optional[int] = None
    if "schedule" in steps_wanted and not skip_fetch:
        if not allowed("schedule"):
            games_today = 0
            results["games_today"] = 0
            add(
                _skip_step(
                    "schedule_fetcher",
                    f"phase={phase_info['phase']}: no slate to fetch",
                )
            )
        else:
            import schedule_fetcher

            def _schedule() -> dict:
                return schedule_fetcher.populate_schedule(
                    as_of=as_of, data_dir=data_dir
                )

            sched = _safe_call("schedule_fetcher", _schedule)
            add(sched)
            if sched.get("ok", True) and not sched.get("skipped"):
                games_today = int(sched.get("games") or 0)
            else:
                games_today = _count_games_csv(
                    raw_day_dir(as_of, data_dir) / "games.csv"
                )
            results["games_today"] = games_today
            # Tighten policy for mid-season off days.
            policy = workflow_step_policy(
                phase_info["phase"],
                games_today=games_today,
                morning_run=morning_run,
            )
            if games_today == 0:
                print("  off day / empty slate — skipping odds, align, predict")
    elif allowed("schedule") and (data_dir / "raw" / as_of.isoformat() / "games.csv").exists():
        games_today = _count_games_csv(raw_day_dir(as_of, data_dir) / "games.csv")
        results["games_today"] = games_today
        policy = workflow_step_policy(
            phase_info["phase"],
            games_today=games_today,
            morning_run=morning_run,
        )

    # 5) Odds
    if "odds" in steps_wanted and not skip_fetch:
        if not allowed("odds"):
            reason = (
                "no games today"
                if games_today == 0
                else f"phase={phase_info['phase']}: no markets expected"
            )
            add(_skip_step("odds_fetcher", reason))
        else:
            import odds_fetcher
            from paths import resolve_day_csvs

            teams_csv, players_csv = resolve_day_csvs(data_dir=data_dir, as_of=as_of)
            add(
                _safe_call(
                    "odds_fetcher",
                    lambda: odds_fetcher.populate_odds(
                        teams_csv=teams_csv,
                        players_csv=players_csv,
                        as_of=as_of,
                    ),
                )
            )

    # 6) Align + optional same-day boxscores
    if "align" in steps_wanted and not skip_fetch:
        if not allowed("align"):
            reason = (
                "no games today"
                if games_today == 0
                else f"phase={phase_info['phase']}: no games to align"
            )
            add(_skip_step("aligner", reason))
        else:
            import aligner

            add(
                _safe_call(
                    "aligner",
                    lambda: aligner.align_day(
                        raw_dir=raw_day_dir(as_of, data_dir),
                        out_dir=panels_day_dir(as_of, data_dir),
                    ),
                )
            )

        if not allowed("boxscore_today"):
            reason = (
                "no games today"
                if games_today == 0
                else "morning run: today's boxscores deferred (use --include-today-boxscores)"
            )
            add(_skip_step("boxscore_fetcher(today)", reason))
        elif allowed("align"):
            import boxscore_fetcher

            add(
                _safe_call(
                    "boxscore_fetcher(today)",
                    lambda: boxscore_fetcher.populate_boxscores(
                        as_of=as_of, data_dir=data_dir
                    ),
                )
            )

    # 7) Train only when panels are newer than models (unless --force-train).
    if "train" in steps_wanted and not skip_train:
        if not allowed("train"):
            add(_skip_step("train", f"phase={phase_info['phase']}"))
        else:
            model_dir = data_dir / "models"
            freshness = labeled_panels_newer_than_models(data_dir, model_dir)
            if not force_train and not freshness["needed"]:
                add(_skip_step("train", freshness["reason"]))
            else:
                import train_predict

                def _train() -> dict:
                    wanted = targets or list(train_predict.TARGETS)
                    out = []
                    for target in wanted:
                        try:
                            out.append(
                                train_predict.train_target(
                                    target,
                                    data_dir=data_dir,
                                    model_dir=model_dir,
                                )
                            )
                        except Exception as err:
                            out.append(
                                {"target": target, "ok": False, "error": str(err)}
                            )
                    out_meta = {"targets": out, "freshness": freshness}
                    return out_meta

                add(_safe_call("train", _train))

    # 8) Predict when there is a slate.
    if "predict" in steps_wanted and not skip_predict:
        if not allowed("predict"):
            reason = (
                "no games today"
                if games_today == 0
                else f"phase={phase_info['phase']}: no slate to score"
            )
            add(_skip_step("predict", reason))
        else:
            import train_predict

            def _predict() -> dict:
                wanted = targets or list(train_predict.TARGETS)
                out = []
                for target in wanted:
                    try:
                        out.append(
                            train_predict.predict_target(
                                target,
                                as_of=as_of,
                                data_dir=data_dir,
                                model_dir=data_dir / "models",
                                pred_dir=data_dir / "predictions",
                            )
                        )
                    except Exception as err:
                        out.append({"target": target, "ok": False, "error": str(err)})
                return {"targets": out}

            add(_safe_call("predict", _predict))

    results["ok"] = all(s.get("ok", True) for s in results["steps"])
    results["policy"] = policy
    return results


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--season-phase",
        choices=list(PHASES),
        default=None,
        help="Override auto-detected phase (regular|postseason|offseason)",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument(
        "--force-stats",
        action="store_true",
        help="Re-fetch stats even if today's snapshot exists",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if models are newer than panels",
    )
    parser.add_argument(
        "--include-today-boxscores",
        action="store_true",
        help="Also stamp today's boxscores (for afternoon/evening runs; off by default at 10am)",
    )
    parser.add_argument(
        "--only",
        choices=[
            "results",
            "backfill",
            "stats",
            "schedule",
            "odds",
            "align",
            "train",
            "predict",
        ],
        default=None,
    )
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--no-splits", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_workflow(
        as_of=args.as_of,
        data_dir=args.data_dir,
        skip_fetch=args.skip_fetch,
        skip_train=args.skip_train,
        skip_predict=args.skip_predict,
        skip_backfill=args.skip_backfill,
        only=args.only,
        targets=args.target,
        no_splits=args.no_splits,
        season_phase_override=args.season_phase,
        force_stats=args.force_stats,
        force_train=args.force_train,
        include_today_boxscores=args.include_today_boxscores,
    )
    print("\n==> done")
    phase = (summary.get("phase") or {}).get("phase", "?")
    games = summary.get("games_today")
    games_bit = f" games_today={games}" if games is not None else ""
    print(
        f"as_of={summary['as_of']} phase={phase}{games_bit} "
        f"ok={summary['ok']} steps={len(summary['steps'])}"
    )
    for step in summary["steps"]:
        if step.get("skipped"):
            status = "skip"
        else:
            status = "ok" if step.get("ok", True) else "FAIL"
        detail = step.get("error") or step.get("reason") or ""
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {step.get('step')}{suffix}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
