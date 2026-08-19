import csv
from pathlib import Path

import boxscore_fetcher as bf


def test_num():
    assert bf._num(None) == ""
    assert bf._num(3) == "3"
    assert bf._num(5.1) == "5.1"


def test_parse_boxscore_side_batters_and_pitchers():
    block = {
        "team": {"id": 147},
        "batters": [1],
        "pitchers": [2],
        "players": {
            "ID1": {
                "person": {"id": 1, "fullName": "Batter One"},
                "battingOrder": "101",
                "stats": {
                    "batting": {
                        "hits": 2,
                        "homeRuns": 1,
                        "strikeOuts": 0,
                        "atBats": 4,
                        "runs": 1,
                        "rbi": 2,
                        "baseOnBalls": 0,
                        "plateAppearances": 4,
                    },
                    "pitching": {},
                },
            },
            "ID2": {
                "person": {"id": 2, "fullName": "Pitcher Two"},
                "stats": {
                    "batting": {},
                    "pitching": {
                        "inningsPitched": "6.0",
                        "strikeOuts": 8,
                        "earnedRuns": 1,
                        "hits": 4,
                        "homeRuns": 0,
                        "baseOnBalls": 1,
                        "pitchesThrown": 95,
                    },
                },
            },
            "ID3": {
                "person": {"id": 3, "fullName": "Bench"},
                "stats": {"batting": {}, "pitching": {}},
            },
        },
    }
    rows = bf._parse_boxscore_side(
        game_pk="1001",
        game_date="2026-08-09",
        side="home",
        block=block,
    )
    assert len(rows) == 2
    by_id = {r["player_id"]: r for r in rows}
    assert by_id["1"]["hits"] == "2"
    assert by_id["1"]["is_batter"] == "1"
    assert by_id["2"]["pitcher_strikeouts"] == "8"
    assert by_id["2"]["is_pitcher"] == "1"


def test_stamp_player_game_labels(tmp_path: Path):
    panel = tmp_path / "player_game.csv"
    with panel.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["gamePk", "player_id", "player_name", "label_hits"]
        )
        writer.writeheader()
        writer.writerow(
            {"gamePk": "1001", "player_id": "1", "player_name": "A", "label_hits": ""}
        )
        writer.writerow(
            {"gamePk": "1001", "player_id": "2", "player_name": "B", "label_hits": ""}
        )

    box_rows = [
        {
            "gamePk": "1001",
            "player_id": "1",
            "hits": "3",
            "home_runs": "1",
            "strikeouts": "0",
            "at_bats": "4",
            "innings_pitched": "",
            "pitcher_strikeouts": "",
            "earned_runs": "",
            "hits_allowed": "",
            "home_runs_allowed": "",
            "walks_allowed": "",
            "pitches_thrown": "",
            "runs": "1",
            "rbi": "2",
            "walks": "0",
        }
    ]
    summary = bf.stamp_player_game_labels(panel, box_rows)
    assert summary["rows_updated"] == 1

    with panel.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    a = next(r for r in rows if r["player_id"] == "1")
    b = next(r for r in rows if r["player_id"] == "2")
    assert a["label_hits"] == "3"
    assert a["label_appeared"] == "1"
    assert not b.get("label_hits") or b["label_hits"] == ""


def test_stamp_missing_panel(tmp_path: Path):
    summary = bf.stamp_player_game_labels(tmp_path / "missing.csv", [])
    assert summary["missing_panel"] is True
    assert summary["rows_updated"] == 0
