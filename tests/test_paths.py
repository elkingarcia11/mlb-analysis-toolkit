from datetime import date
from pathlib import Path

from paths import (
    iter_raw_day_dirs,
    panels_day_dir,
    raw_day_dir,
    resolve_day_csvs,
    resolve_day_dir,
)


def test_raw_and_panels_day_dir():
    as_of = date(2026, 8, 9)
    root = Path("data")
    assert raw_day_dir(as_of, root) == Path("data/raw/2026-08-09")
    assert panels_day_dir(as_of, root) == Path("data/panels/2026-08-09")


def test_resolve_day_dir_prefers_legacy_flat(tmp_path: Path):
    as_of = date(2026, 8, 9)
    (tmp_path / "teams.csv").write_text("teamId\n1\n", encoding="utf-8")
    assert resolve_day_dir(tmp_path, as_of) == tmp_path


def test_resolve_day_dir_uses_dated_raw(tmp_path: Path):
    as_of = date(2026, 8, 9)
    assert resolve_day_dir(tmp_path, as_of) == tmp_path / "raw" / "2026-08-09"


def test_resolve_day_csvs_default_and_override(tmp_path: Path):
    as_of = date(2026, 8, 9)
    teams, players = resolve_day_csvs(data_dir=tmp_path, as_of=as_of)
    assert teams == tmp_path / "raw" / "2026-08-09" / "teams.csv"
    assert players.name == "players.csv"

    override = tmp_path / "custom_teams.csv"
    teams2, _ = resolve_day_csvs(
        data_dir=tmp_path, as_of=as_of, teams_csv=override
    )
    assert teams2 == override


def test_iter_raw_day_dirs_skips_non_iso(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "2026-08-09").mkdir(parents=True)
    (raw / "2026-08-10").mkdir()
    (raw / "notes").mkdir()
    (raw / "readme.txt").write_text("x", encoding="utf-8")

    days = iter_raw_day_dirs(tmp_path)
    assert [d for d, _ in days] == [date(2026, 8, 9), date(2026, 8, 10)]


def test_iter_raw_day_dirs_legacy_flat(tmp_path: Path):
    (tmp_path / "players.csv").write_text("playerId\n1\n", encoding="utf-8")
    days = iter_raw_day_dirs(tmp_path)
    assert len(days) == 1
    assert days[0][1] == tmp_path
