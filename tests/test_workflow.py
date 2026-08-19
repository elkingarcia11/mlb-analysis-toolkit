import csv
import time
from datetime import date
from pathlib import Path

import workflow as wf


def test_safe_call_success_and_failure():
    ok = wf._safe_call("ok_step", lambda: {"value": 1})
    assert ok["ok"] is True
    assert ok["step"] == "ok_step"
    assert ok["value"] == 1

    bad = wf._safe_call("bad_step", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert bad["ok"] is False
    assert "boom" in bad["error"]


def test_skip_step():
    out = wf._skip_step("align", "no games today")
    assert out["skipped"] is True
    assert out["ok"] is True
    assert out["reason"] == "no games today"


def test_day_stats_exist(tmp_path: Path):
    as_of = date(2026, 8, 9)
    day = tmp_path / "raw" / "2026-08-09"
    day.mkdir(parents=True)
    assert wf._day_stats_exist(tmp_path, as_of) is False
    (day / "teams.csv").write_text("x\n", encoding="utf-8")
    (day / "players.csv").write_text("x\n", encoding="utf-8")
    assert wf._day_stats_exist(tmp_path, as_of) is True


def test_count_games_csv(tmp_path: Path):
    path = tmp_path / "games.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["gamePk", "status"])
        writer.writeheader()
        writer.writerow({"gamePk": "1", "status": "Final"})
        writer.writerow({"gamePk": "", "status": "Final"})
        writer.writerow({"gamePk": "2", "status": "Scheduled"})
    assert wf._count_games_csv(path) == 2
    assert wf._count_games_csv(tmp_path / "missing.csv") == 0


def test_labeled_panels_newer_than_models(tmp_path: Path):
    data_dir = tmp_path / "data"
    model_dir = data_dir / "models"
    panels = data_dir / "panels" / "2026-08-09"
    panels.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    # No panels → not needed
    empty = wf.labeled_panels_newer_than_models(tmp_path / "empty", model_dir)
    assert empty["needed"] is False

    panel = panels / "team_game.csv"
    panel.write_text("gamePk\n1\n", encoding="utf-8")
    # Panels but no models
    needed = wf.labeled_panels_newer_than_models(data_dir, model_dir)
    assert needed["needed"] is True
    assert "no saved models" in needed["reason"]

    model = model_dir / "team_win.pkl"
    model.write_bytes(b"fake")
    # Make panel newer than model
    time.sleep(0.05)
    panel.write_text("gamePk\n1\n2\n", encoding="utf-8")
    newer = wf.labeled_panels_newer_than_models(data_dir, model_dir)
    assert newer["needed"] is True

    # Touch model so it is newest
    time.sleep(0.05)
    model.write_bytes(b"fake2")
    current = wf.labeled_panels_newer_than_models(data_dir, model_dir)
    assert current["needed"] is False
