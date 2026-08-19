import csv
from datetime import date
from pathlib import Path

import results_fetcher as rf


def test_strip_date_fetched_variants():
    assert rf.strip_date_fetched("2026-08-09") == "2026-08-09"
    assert rf.strip_date_fetched("2026-08-09T14:30:00Z") == "2026-08-09"
    assert rf.strip_date_fetched("") == ""
    assert rf.strip_date_fetched(None) == ""


def test_parse_line():
    assert rf._parse_line("-1.5") == -1.5
    assert rf._parse_line("+1.5") == 1.5
    assert rf._parse_line("o8.5") == 8.5
    assert rf._parse_line("u8½") == 8.5
    assert rf._parse_line("") is None
    assert rf._parse_line("abc") is None


def test_settle_moneyline():
    assert rf.settle_moneyline(5, 3) == "W"
    assert rf.settle_moneyline(2, 4) == "L"
    assert rf.settle_moneyline(3, 3) == "P"


def test_settle_run_line():
    # Favorite -1.5 covers a 5-3 win (margin 2 + (-1.5) = 0.5)
    assert rf.settle_run_line(5, 3, -1.5) == "W"
    # Favorite -1.5 loses a 4-3 win (margin 1 + (-1.5) = -0.5)
    assert rf.settle_run_line(4, 3, -1.5) == "L"
    # Push when margin + line == 0
    assert rf.settle_run_line(5, 3, -2.0) == "P"


def test_settle_total_over_under():
    assert rf.settle_total(5, 4, 8.5, "over") == "W"
    assert rf.settle_total(5, 3, 8.5, "over") == "L"
    assert rf.settle_total(4, 4, 8.0, "over") == "P"
    assert rf.settle_total(5, 3, 8.5, "under") == "W"
    assert rf.settle_total(5, 4, 8.5, "under") == "L"


def test_settle_row_fills_bet_columns():
    row = {"moneyline": "-120", "run_line": "-1.5", "total": "8.5"}
    game = {"team_score": 5, "opp_score": 3}
    out = rf.settle_row(row, game)
    assert out["moneyline_result"] == "W"
    assert out["run_line_result"] == "W"
    assert out["total_over_result"] == "L"
    assert out["total_under_result"] == "W"


def test_pick_game_prefers_opponent():
    games = [
        {"opp_name": "Boston Red Sox", "home_away": "away", "team_score": 1, "opp_score": 2},
        {"opp_name": "New York Yankees", "home_away": "home", "team_score": 4, "opp_score": 3},
    ]
    picked = rf._pick_game(games, {"odds_opponent": "Yankees", "odds_home_away": "home"})
    assert picked["opp_name"] == "New York Yankees"


def test_game_result_and_box_labels_doubleheader():
    games = [
        {"team_score": 5, "opp_score": 3, "opp_name": "A", "home_away": "home"},
        {"team_score": 1, "opp_score": 2, "opp_name": "B", "home_away": "away"},
    ]
    row = {}  # no odds context → all games
    assert rf.game_result_string(games, row) == "W,L"
    labels = rf.box_score_labels(games, row)
    assert labels["team_runs"] == "5,1"
    assert labels["final_score"] == "5-3,1-2"


def test_dates_needing_results():
    rows = [
        {"date_fetched": "2026-08-08", "results": "", "team_runs": ""},
        {"date_fetched": "2026-08-09", "results": "", "team_runs": ""},  # not before
        {
            "date_fetched": "2026-08-07",
            "results": "W",
            "team_runs": "5",
            "opp_runs": "3",
            "final_score": "5-3",
            "moneyline": "-110",
            "moneyline_result": "W",
            "run_line_result": "W",
            "total_over_result": "L",
            "total_under_result": "W",
        },
    ]
    needed = rf.dates_needing_results(rows, before=date(2026, 8, 9), force=False)
    assert needed == {date(2026, 8, 8)}


def test_populate_csv_settles_and_labels(tmp_path: Path):
    path = tmp_path / "teams.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "date_fetched",
                "teamId",
                "teamAbbrev",
                "moneyline",
                "run_line",
                "total",
                "odds_opponent",
                "odds_home_away",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date_fetched": "2026-08-08T12:00:00Z",
                "teamId": "147",
                "teamAbbrev": "NYY",
                "moneyline": "-130",
                "run_line": "-1.5",
                "total": "8.5",
                "odds_opponent": "Boston Red Sox",
                "odds_home_away": "home",
            }
        )

    scores = {
        date(2026, 8, 8): {
            147: [
                {
                    "team_score": 5,
                    "opp_score": 3,
                    "opp_name": "Boston Red Sox",
                    "opp_id": 111,
                    "home_away": "home",
                }
            ]
        }
    }
    summary = rf.populate_csv(
        path,
        scores_by_date=scores,
        before=date(2026, 8, 9),
        force=False,
        dry_run=False,
    )
    assert summary["rows_updated"] == 1

    with path.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["date_fetched"] == "2026-08-08"
    assert row["results"] == "W"
    assert row["team_runs"] == "5"
    assert row["moneyline_result"] == "W"
    assert row["run_line_result"] == "W"
