import csv
from datetime import date
from pathlib import Path
from unittest.mock import patch

import odds_fetcher as of


def test_close_odds():
    block = {
        "home": {"close": {"line": "-1.5", "odds": "-110"}},
        "away": {"close": {"line": "+1.5", "odds": "-110"}},
    }
    assert of._close_odds(block, "home") == ("-1.5", "-110")
    assert of._close_odds(None, "home") == ("", "")
    assert of._close_odds({}, "home") == ("", "")


def test_abbrev_keys_aliases():
    keys = of._abbrev_keys("AZ")
    assert "AZ" in keys
    assert "ARI" in keys


def test_lookup_team_odds_by_abbr_and_name():
    odds = {
        "abbr:NYY": {"moneyline": "-120", "odds_home_away": "home"},
        "name:new york mets": {"moneyline": "+110", "odds_home_away": "away"},
    }
    assert of._lookup_team_odds({"teamAbbrev": "NYY"}, odds)["moneyline"] == "-120"
    assert of._lookup_team_odds({"teamAbbrev": "", "teamName": "New York Mets"}, odds)[
        "moneyline"
    ] == "+110"
    assert of._lookup_team_odds({"teamAbbrev": "BOS"}, odds) is None


def test_fetch_odds_for_date_parses_event():
    payload = {
        "events": [
            {
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {
                                    "abbreviation": "NYY",
                                    "displayName": "New York Yankees",
                                    "name": "Yankees",
                                },
                            },
                            {
                                "homeAway": "away",
                                "team": {
                                    "abbreviation": "BOS",
                                    "displayName": "Boston Red Sox",
                                    "name": "Red Sox",
                                },
                            },
                        ],
                        "odds": [
                            {
                                "moneyline": {
                                    "home": {"close": {"odds": "-140"}},
                                    "away": {"close": {"odds": "+120"}},
                                },
                                "pointSpread": {
                                    "home": {"close": {"line": "-1.5", "odds": "-110"}},
                                    "away": {"close": {"line": "+1.5", "odds": "-110"}},
                                },
                                "total": {
                                    "over": {"close": {"line": "o8.5", "odds": "-105"}},
                                    "under": {"close": {"line": "u8.5", "odds": "-115"}},
                                },
                            }
                        ],
                    }
                ]
            }
        ]
    }
    with patch.object(of, "_http_get_json", return_value=payload):
        by_key = of.fetch_odds_for_date(date(2026, 8, 9))

    home = by_key["abbr:NYY"]
    assert home["moneyline"] == "-140"
    assert home["run_line"] == "-1.5"
    assert home["total"] == "8.5"
    assert home["odds_opponent"] == "Boston Red Sox"
    assert home["odds_home_away"] == "home"
    assert by_key["name:boston red sox"]["odds_home_away"] == "away"


def test_populate_csv_stamps_matching_day(tmp_path: Path):
    path = tmp_path / "teams.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["date_fetched", "teamId", "teamAbbrev", "teamName"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "date_fetched": "2026-08-09",
                "teamId": "147",
                "teamAbbrev": "NYY",
                "teamName": "New York Yankees",
            }
        )
        writer.writerow(
            {
                "date_fetched": "2026-08-08",
                "teamId": "111",
                "teamAbbrev": "BOS",
                "teamName": "Boston Red Sox",
            }
        )

    odds_by_key = {
        "abbr:NYY": {
            "moneyline": "-130",
            "run_line": "-1.5",
            "run_line_odds": "-110",
            "total": "8.5",
            "total_over_odds": "-105",
            "total_under_odds": "-115",
            "odds_opponent": "Boston Red Sox",
            "odds_home_away": "home",
        }
    }
    summary = of.populate_csv(
        path,
        odds_by_key=odds_by_key,
        as_of=date(2026, 8, 9),
        dry_run=False,
    )
    assert summary["rows_updated"] == 1

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    yankees = next(r for r in rows if r["teamAbbrev"] == "NYY")
    bos = next(r for r in rows if r["teamAbbrev"] == "BOS")
    assert yankees["moneyline"] == "-130"
    assert not bos.get("moneyline")
