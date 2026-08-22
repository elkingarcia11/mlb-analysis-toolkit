"""Tests for data_fetcher split batching and flattening."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

import data_fetcher as df


class _FakeResp:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


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
        {"code": "h", "description": "Home",
            "batting": True, "pitching": True, "team": True},
        {"code": "vl", "description": "vs LHP",
            "batting": True, "pitching": False, "team": True},
        {"code": "rp", "description": "as RP",
            "batting": False, "pitching": True, "team": False},
        {"code": "h", "description": "Home dup",
            "batting": True, "pitching": True, "team": True},
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
    first = df.write_csv(
        path, [{"date_fetched": "2026-08-09", "teamId": 1, "ops": ".700"}])
    second = df.write_csv(
        path, [{"date_fetched": "2026-08-09", "teamId": 99, "ops": ".800"}])
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
        df.fetch_expanded_stats("player", "hitting",
                                2026, timeframe="last_7", sit_code="h")


def test_flatten_split_row_maps_identity_and_stats():
    row = {
        "season": "2026",
        "gameType": "R",
        "numTeams": 2,
        "rank": 1,
        "split": {"code": "h", "description": "Home Games", "sortOrder": 1},
        "player": {"id": 650333, "fullName": "Luis Arraez", "firstName": "Luis", "lastName": "Arraez", "link": "/api/v1/people/650333"},
        "team": {"id": 143, "name": "Philadelphia Phillies", "link": "/api/v1/teams/143"},
        "position": {"abbreviation": "2B", "name": "Second Base"},
        "stat": {"avg": ".322", "ops": ".799", "homeRuns": 6},
    }
    flat = df._flatten_split_row(row, entity="player")
    assert flat["split_code"] == "h"
    assert flat["split_description"] == "Home Games"
    assert flat["playerId"] == 650333
    assert flat["playerFullName"] == "Luis Arraez"
    assert flat["playerName"] == "Luis Arraez"
    assert flat["teamId"] == 143
    assert flat["teamName"] == "Philadelphia Phillies"
    assert flat["teamAbbrev"] == "PHI"
    assert flat["position"] == "2B"
    assert flat["avg"] == ".322"
    assert flat["homeRuns"] == 6


def test_fetch_split_rows_batches_all_codes_in_one_request():
    payload = {
        "stats": [
            {
                "splits": [
                    {
                        "season": "2026",
                        "split": {"code": "h", "description": "Home"},
                        "player": {"id": 1, "fullName": "A"},
                        "team": {"id": 10, "name": "Team"},
                        "stat": {"avg": ".300"},
                    },
                    {
                        "season": "2026",
                        "split": {"code": "a", "description": "Away"},
                        "player": {"id": 2, "fullName": "B"},
                        "team": {"id": 10, "name": "Team"},
                        "stat": {"avg": ".250"},
                    },
                ],
                "totalSplits": 2,
            }
        ]
    }
    with patch("data_fetcher.fetch_page_with_retry", return_value=payload) as mock_fetch:
        rows = df._fetch_split_rows(
            entity="player",
            group="hitting",
            season=2026,
            sit_codes=["h", "a"],
            as_of=date(2026, 8, 10),
            game_type="R",
            page_size=5000,
        )
    assert len(rows) == 2
    assert {r["split_code"] for r in rows} == {"h", "a"}
    assert [r["playerId"] for r in rows] == [1, 2]
    mock_fetch.assert_called_once()


def test_fetch_split_rows_chunks_codes_across_requests():
    codes = [f"c{i}" for i in range(45)]  # > SIT_CODES_PER_REQUEST (20)
    calls: list[tuple] = []

    def fake_fetch(endpoint, params, **kwargs):
        calls.append((endpoint, params.get("sitCodes").split(",")))
        return {
            "stats": [
                {
                    "splits": [
                        {
                            "season": "2026",
                            "split": {"code": "x", "description": "X"},
                            "player": {"id": 1, "fullName": "A"},
                            "team": {"id": 10, "name": "Team"},
                            "stat": {},
                        }
                    ],
                    "totalSplits": 1,
                }
            ]
        }

    with patch("data_fetcher.fetch_page_with_retry", side_effect=fake_fetch) as mock_fetch:
        rows = df._fetch_split_rows(
            entity="player",
            group="hitting",
            season=2026,
            sit_codes=codes,
            as_of=date(2026, 8, 10),
            game_type="R",
            page_size=5000,
        )
    assert len(calls) == 3  # ceil(45/20)
    sizes = [len(c[1]) for c in calls]
    assert sizes == [20, 20, 5]
