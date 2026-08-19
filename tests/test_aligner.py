from pathlib import Path

import aligner as al


SAMPLE_GAME = {
    "game_date": "2026-08-09",
    "gamePk": "1001",
    "game_datetime": "2026-08-09T23:05:00Z",
    "season": "2026",
    "status": "Final",
    "abstract_state": "Final",
    "venue_id": "1",
    "venue_name": "Yankee Stadium",
    "home_team_id": "147",
    "home_team_abbr": "NYY",
    "home_team_name": "New York Yankees",
    "away_team_id": "111",
    "away_team_abbr": "BOS",
    "away_team_name": "Boston Red Sox",
    "home_probable_pitcher_id": "543",
    "home_probable_pitcher_name": "Gerrit Cole",
    "away_probable_pitcher_id": "321",
    "away_probable_pitcher_name": "Chris Sale",
    "home_score": "5",
    "away_score": "3",
}


def test_is_feature_column():
    assert al._is_feature_column("ops") is True
    assert al._is_feature_column("teamId") is False
    assert al._is_feature_column("playerName") is False
    assert al._is_feature_column("moneyline") is False


def test_base_rows_drops_splits():
    rows = [
        {"playerId": "1", "split_code": "", "ops": ".800"},
        {"playerId": "1", "split_code": "h", "ops": ".900"},
    ]
    assert len(al._base_rows(rows)) == 1


def test_pivot_features():
    rows = [
        {
            "teamId": "147",
            "stat_group": "hitting",
            "timeframe": "ytd",
            "split_code": "",
            "ops": ".780",
            "avg": ".265",
            "teamName": "Yankees",
        },
        {
            "teamId": "147",
            "stat_group": "hitting",
            "timeframe": "last_7",
            "split_code": "",
            "ops": ".810",
        },
    ]
    pivoted = al.pivot_features(rows, id_field="teamId")
    assert pivoted["147"]["hitting_ytd_ops"] == ".780"
    assert pivoted["147"]["hitting_last_7_ops"] == ".810"
    assert "hitting_ytd_teamName" not in pivoted["147"]


def test_pivot_split_features():
    rows = [
        {
            "playerId": "9",
            "stat_group": "hitting",
            "split_code": "vl",
            "avg": ".310",
            "ops": ".900",
        },
        {
            "playerId": "9",
            "stat_group": "hitting",
            "split_code": "dfr",
            "avg": ".280",
        },
    ]
    out = al.pivot_split_features(rows, id_field="playerId")
    assert out["9"]["vl"]["ops"] == ".900"
    assert out["9"]["dfr"]["avg"] == ".280"
    assert al._weekday_split_code("2026-08-14") == "dfr"


def test_context_by_team_keeps_first_nonempty():
    rows = [
        {"teamId": "147", "moneyline": "-130", "total": ""},
        {"teamId": "147", "moneyline": "-999", "total": "8.5"},
    ]
    ctx = al.context_by_team(rows)
    assert ctx["147"]["moneyline"] == "-130"
    assert ctx["147"]["total"] == "8.5"


def test_game_side_labels():
    home = al._game_side_labels(SAMPLE_GAME, "home")
    away = al._game_side_labels(SAMPLE_GAME, "away")
    assert home["label_team_runs"] == "5"
    assert home["label_win"] == "1"
    assert away["label_win"] == "0"
    assert home["label_total_runs"] == "8"
    assert home["label_final_score"] == "5-3"


def test_build_team_game():
    features = {
        "147": {"hitting_ytd_ops": ".780"},
        "111": {"hitting_ytd_ops": ".720"},
    }
    context = {"147": {"moneyline": "-140", **{c: "" for c in al.CONTEXT_COLUMNS if c != "moneyline"}}}
    rows = al.build_team_game([SAMPLE_GAME], features, context)
    assert len(rows) == 2
    home = next(r for r in rows if r["is_home"] == "1")
    assert home["team_id"] == "147"
    assert home["team_hitting_ytd_ops"] == ".780"
    assert home["opp_hitting_ytd_ops"] == ".720"
    assert home["ctx_moneyline"] == "-140"
    assert home["label_win"] == "1"


def test_build_player_game_flags_probable():
    player_rows = [
        {
            "playerId": "543",
            "playerFullName": "Gerrit Cole",
            "teamId": "147",
            "teamAbbrev": "NYY",
            "primaryPositionAbbrev": "P",
        },
        {
            "playerId": "999",
            "playerFullName": "Batter X",
            "teamId": "147",
            "teamAbbrev": "NYY",
            "primaryPositionAbbrev": "CF",
        },
        {
            "playerId": "1111",
            "playerFullName": "Other Team",
            "teamId": "999",
            "teamAbbrev": "XYZ",
        },
    ]
    feats = {
        "543": {"pitching_ytd_era": "3.10"},
        "999": {"hitting_ytd_ops": ".850"},
        "321": {"pitching_ytd_era": "2.80", "pitching_ytd_strikeOuts": "180", "hitting_ytd_avg": ".100"},
    }
    hands = {
        "999": {"bat_side": "L", "pitch_hand": "R"},
        "321": {"bat_side": "L", "pitch_hand": "L"},
    }
    split_features = {
        "999": {
            "vl": {"avg": ".333", "ops": ".950"},
            "dsu": {"avg": ".290"},
        }
    }
    parks = {
        "1": {
            "index_runs": "105",
            "index_hits": "103",
            "index_hr": "110",
        }
    }
    rows = al.build_player_game(
        [SAMPLE_GAME],
        player_rows,
        feats,
        hands=hands,
        split_features=split_features,
        park_factors=parks,
    )
    by_id = {r["player_id"]: r for r in rows}
    assert "1111" not in by_id
    assert by_id["543"]["is_probable_pitcher"] == "1"
    assert by_id["999"]["is_probable_pitcher"] == "0"
    assert by_id["543"]["player_pitching_ytd_era"] == "3.10"
    batter = by_id["999"]
    assert batter["opp_probable_pitcher_id"] == "321"
    assert batter["opp_pitcher_pitching_ytd_era"] == "2.80"
    assert batter["opp_pitcher_pitching_ytd_strikeOuts"] == "180"
    assert "opp_pitcher_hitting_ytd_avg" not in batter
    assert batter["batter_bat_side"] == "L"
    assert batter["opp_pitch_hand"] == "L"
    assert batter["platoon_advantage"] == "0"
    assert batter["venue_id"] == "1"
    assert batter["park_factor_runs"] == "1.05"
    assert batter["park_factor_hits"] == "1.03"
    assert batter["park_factor_hr"] == "1.1"
    assert batter["batter_vs_hand_avg"] == ".333"
    assert batter["batter_weekday_avg"] == ".290"
    assert batter["game_hour_utc"] == "23"
    assert batter["is_day_game"] in {"0", "1"}


def test_platoon_and_day_helpers():
    from hands import platoon_advantage

    assert platoon_advantage("L", "R") == "1"
    assert platoon_advantage("R", "R") == "0"
    assert platoon_advantage("S", "R") == ""
    assert al._game_hour_utc("2026-08-09T23:05:00Z") == "23"
    assert al._is_day_game({"day_night": "day"}) == "1"
    assert al._is_day_game({"day_night": "night"}) == "0"


def test_align_day_writes_panels(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "panels"
    raw.mkdir()
    # Minimal CSVs
    (raw / "teams.csv").write_text(
        "teamId,stat_group,timeframe,split_code,ops,moneyline\n"
        "147,hitting,ytd,,.780,-130\n"
        "111,hitting,ytd,,.720,\n",
        encoding="utf-8",
    )
    (raw / "players.csv").write_text(
        "playerId,playerFullName,teamId,teamAbbrev,stat_group,timeframe,split_code,avg\n"
        "999,Batter X,147,NYY,hitting,ytd,,.300\n",
        encoding="utf-8",
    )
    headers = ",".join(
        [
            "game_date",
            "gamePk",
            "game_datetime",
            "season",
            "status",
            "abstract_state",
            "venue_id",
            "venue_name",
            "home_team_id",
            "home_team_abbr",
            "home_team_name",
            "away_team_id",
            "away_team_abbr",
            "away_team_name",
            "home_probable_pitcher_id",
            "home_probable_pitcher_name",
            "away_probable_pitcher_id",
            "away_probable_pitcher_name",
            "home_score",
            "away_score",
        ]
    )
    values = ",".join(
        [
            SAMPLE_GAME[k]
            for k in [
                "game_date",
                "gamePk",
                "game_datetime",
                "season",
                "status",
                "abstract_state",
                "venue_id",
                "venue_name",
                "home_team_id",
                "home_team_abbr",
                "home_team_name",
                "away_team_id",
                "away_team_abbr",
                "away_team_name",
                "home_probable_pitcher_id",
                "home_probable_pitcher_name",
                "away_probable_pitcher_id",
                "away_probable_pitcher_name",
                "home_score",
                "away_score",
            ]
        ]
    )
    (raw / "games.csv").write_text(f"{headers}\n{values}\n", encoding="utf-8")

    summary = al.align_day(raw_dir=raw, out_dir=out)
    assert (out / "team_game.csv").exists()
    assert (out / "player_game.csv").exists()
    assert summary["team_game_rows"] == 2
    assert summary["player_game_rows"] >= 1
