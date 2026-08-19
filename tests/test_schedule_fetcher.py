from datetime import date
from pathlib import Path
from unittest.mock import patch

import schedule_fetcher as sf


def test_side_team_normalizes_winner_and_score():
    entry = {
        "team": {"id": 147, "abbreviation": "nyy", "name": "New York Yankees"},
        "probablePitcher": {"id": 543, "fullName": "Gerrit Cole"},
        "score": 5,
        "isWinner": True,
    }
    out = sf._side_team(entry)
    assert out["team_id"] == 147
    assert out["team_abbr"] == "NYY"
    assert out["probable_pitcher_id"] == 543
    assert out["score"] == 5
    assert out["is_winner"] == "1"


def test_normalize_game_builds_spine_row():
    game = {
        "gamePk": 777,
        "officialDate": "2026-08-09",
        "gameDate": "2026-08-09T23:05:00Z",
        "season": "2026",
        "gameType": "R",
        "doubleHeader": "N",
        "gameNumber": 1,
        "status": {"detailedState": "Scheduled", "abstractGameState": "Preview"},
        "dayNight": "night",
        "venue": {"id": 3313, "name": "Yankee Stadium"},
        "teams": {
            "home": {
                "team": {"id": 147, "abbreviation": "NYY", "name": "New York Yankees"},
                "probablePitcher": {"id": 1, "fullName": "Starter A"},
            },
            "away": {
                "team": {"id": 111, "abbreviation": "BOS", "name": "Boston Red Sox"},
                "probablePitcher": {"id": 2, "fullName": "Starter B"},
            },
        },
    }
    row = sf._normalize_game(game, date(2026, 8, 9))
    assert row["gamePk"] == "777"
    assert row["home_team_id"] == "147"
    assert row["away_team_abbr"] == "BOS"
    assert row["home_probable_pitcher_name"] == "Starter A"
    assert row["status"] == "Scheduled"
    assert row["day_night"] == "night"


def test_fetch_games_for_date_uses_schedule_payload():
    payload = {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2026-08-09T17:00:00Z",
                        "officialDate": "2026-08-09",
                        "season": "2026",
                        "gameType": "R",
                        "status": {"detailedState": "Final", "abstractGameState": "Final"},
                        "venue": {"id": 1, "name": "Park"},
                        "teams": {
                            "home": {
                                "team": {
                                    "id": 10,
                                    "abbreviation": "NYY",
                                    "name": "Yankees",
                                },
                                "score": 4,
                                "isWinner": True,
                            },
                            "away": {
                                "team": {
                                    "id": 20,
                                    "abbreviation": "BOS",
                                    "name": "Red Sox",
                                },
                                "score": 2,
                                "isWinner": False,
                            },
                        },
                    }
                ]
            }
        ]
    }
    with patch.object(sf, "_http_get_json", return_value=payload):
        rows = sf.fetch_games_for_date(date(2026, 8, 9))
    assert len(rows) == 1
    assert rows[0]["home_score"] == "4"
    assert rows[0]["home_win"] == "1"


def test_write_games_csv(tmp_path: Path):
    path = tmp_path / "games.csv"
    rows = [
        {
            "game_date": "2026-08-09",
            "gamePk": "1",
            "game_datetime": "",
            "season": "2026",
            "game_type": "R",
            "status": "Final",
            "abstract_state": "Final",
            "doubleheader": "N",
            "game_number": "1",
            "venue_id": "1",
            "venue_name": "Park",
            "home_team_id": "10",
            "home_team_abbr": "NYY",
            "home_team_name": "Yankees",
            "away_team_id": "20",
            "away_team_abbr": "BOS",
            "away_team_name": "Red Sox",
            "home_probable_pitcher_id": "",
            "home_probable_pitcher_name": "",
            "away_probable_pitcher_id": "",
            "away_probable_pitcher_name": "",
            "home_score": "4",
            "away_score": "2",
            "home_win": "1",
        }
    ]
    sf.write_games_csv(path, rows)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "gamePk" in text
    assert "NYY" in text
