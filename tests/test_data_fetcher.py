from datetime import date
from pathlib import Path

import pytest

import data_fetcher as df


def test_resolve_season():
    assert df.resolve_season(2024) == 2024
    assert df.resolve_season(None, date(2026, 8, 10)) == 2026
    assert df.resolve_season(None, date(2026, 2, 1)) == 2025


def test_date_range_for_timeframe():
    as_of = date(2026, 8, 10)
    assert df._date_range_for_timeframe("ytd", as_of) is None
    start, end = df._date_range_for_timeframe("last_7", as_of)
    assert end == as_of
    assert start == date(2026, 8, 4)  # timeframe=-6 → 7 calendar days


def test_applicable_splits_player_and_team():
    codes = [
        {"code": "h", "description": "Home", "batting": True, "pitching": True, "team": True},
        {"code": "vl", "description": "vs LHP", "batting": True, "pitching": False, "team": True},
        {"code": "rp", "description": "as RP", "batting": False, "pitching": True, "team": False},
        {"code": "h", "description": "Home dup", "batting": True, "pitching": True, "team": True},
    ]
    hitting = df.applicable_splits(codes, "player", "hitting")
    assert [c["code"] for c in hitting] == ["h", "vl"]

    pitching = df.applicable_splits(codes, "player", "pitching")
    assert [c["code"] for c in pitching] == ["h", "rp"]

    team_hit = df.applicable_splits(codes, "team", "hitting")
    assert "h" in {c["code"] for c in team_hit}
    assert "rp" not in {c["code"] for c in team_hit}


def test_annotate_rows_adds_meta():
    rows = [{"playerId": 1, "avg": ".300"}]
    out = df._annotate_rows(
        rows,
        date_fetched="2026-08-09",
        season=2026,
        entity="player",
        group="hitting",
        timeframe="ytd",
        split={"code": "h", "description": "Home", "navigationMenu": "Home"},
    )
    assert out[0]["split_code"] == "h"
    assert out[0]["entity"] == "player"
    assert out[0]["avg"] == ".300"


def test_write_csv_never_clobbers(tmp_path: Path):
    path = tmp_path / "teams.csv"
    first = df.write_csv(path, [{"date_fetched": "2026-08-09", "teamId": 1, "ops": ".700"}])
    second = df.write_csv(path, [{"date_fetched": "2026-08-09", "teamId": 99, "ops": ".800"}])
    assert first == "wrote"
    assert second == "skipped"
    text = path.read_text(encoding="utf-8")
    assert ",1" in text or text.rstrip().endswith(",1")
    assert "99" not in text
    assert ".800" not in text


def test_export_csvs(tmp_path: Path):
    data = {
        "player": [{"date_fetched": "2026-08-09", "playerId": 1}],
        "team": [{"date_fetched": "2026-08-09", "teamId": 10}],
    }
    results = df.export_csvs(data, tmp_path)
    assert results["player"][1] == "wrote"
    assert results["team"][2] == 1
    assert (tmp_path / "players.csv").exists()
    assert (tmp_path / "teams.csv").exists()


def test_fetch_expanded_stats_rejects_bad_args():
    with pytest.raises(ValueError):
        df.fetch_expanded_stats("fan", "hitting", 2026)
    with pytest.raises(ValueError):
        df.fetch_expanded_stats("player", "fielding", 2026)
    with pytest.raises(ValueError):
        df.fetch_expanded_stats("player", "hitting", 2026, timeframe="last_7", sit_code="h")
