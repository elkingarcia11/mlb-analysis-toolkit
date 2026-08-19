from datetime import date
from pathlib import Path

import pandas as pd

import train_predict as tp


def test_is_feature_col():
    assert tp._is_feature_col("is_home", tp.FEATURE_PREFIXES_TEAM) is True
    assert tp._is_feature_col("team_hitting_ytd_ops", tp.FEATURE_PREFIXES_TEAM) is True
    assert tp._is_feature_col("label_win", tp.FEATURE_PREFIXES_TEAM) is False
    assert tp._is_feature_col("player_hitting_ytd_avg", tp.FEATURE_PREFIXES_PLAYER) is True
    assert tp._is_feature_col(
        "opp_pitcher_pitching_ytd_era", tp.FEATURE_PREFIXES_PLAYER
    ) is True
    assert tp._is_feature_col(
        "platoon_advantage",
        tp.FEATURE_PREFIXES_PLAYER,
        exact=tp.FEATURE_EXACT_PLAYER,
    ) is True
    assert tp._is_feature_col(
        "venue_id",
        tp.FEATURE_PREFIXES_PLAYER,
        exact=tp.FEATURE_EXACT_PLAYER,
    ) is True
    assert tp._is_feature_col(
        "park_factor_hr", tp.FEATURE_PREFIXES_PLAYER
    ) is True
    assert tp._is_feature_col(
        "batter_vs_hand_ops", tp.FEATURE_PREFIXES_PLAYER
    ) is True
    assert tp._is_feature_col(
        "batter_weekday_avg", tp.FEATURE_PREFIXES_PLAYER
    ) is True


def test_labeled_xy_drops_empty_and_constant_features():
    df = pd.DataFrame(
        {
            "label_win": ["1", "0", "1"],
            "team_hitting_ytd_ops": [".700", ".800", ".750"],
            "team_hitting_ytd_avg": ["", "", ""],
            "is_home": ["1", "1", "1"],
        }
    )
    spec = tp.TARGETS["team_win"]
    X, _, columns = tp.labeled_xy(df, spec)
    assert columns == ["team_hitting_ytd_ops"]
    assert list(X.columns) == columns


def test_team_home_only_filter():
    df = pd.DataFrame({"is_home": ["1", "0", "1"], "x": [1, 2, 3]})
    out = tp._team_home_only(df)
    assert list(out["x"]) == [1, 3]


def test_player_batter_frame_prefers_labeled_ab():
    df = pd.DataFrame(
        {
            "label_hits": ["1", "0", "1"],
            "label_appeared": ["1", "1", "0"],
            "label_at_bats": ["4", "0", "3"],
            "player_id": ["a", "b", "c"],
        }
    )
    out = tp._player_batter_frame(df)
    assert list(out["player_id"]) == ["a"]


def test_player_batter_frame_keeps_unlabeled_slate_when_some_games_final():
    df = pd.DataFrame(
        {
            "label_hits": ["2", "", ""],
            "label_appeared": ["1", "", ""],
            "label_at_bats": ["4", "", ""],
            "player_id": ["finished", "upcoming_a", "upcoming_b"],
        }
    )
    out = tp._player_batter_frame(df)
    assert list(out["player_id"]) == ["finished", "upcoming_a", "upcoming_b"]


def test_player_pitcher_frame_falls_back_to_probable():
    df = pd.DataFrame(
        {
            "label_innings_pitched": ["", "", ""],
            "is_probable_pitcher": ["0", "1", "0"],
            "player_id": ["a", "b", "c"],
        }
    )
    out = tp._player_pitcher_frame(df)
    assert list(out["player_id"]) == ["b"]


def test_feature_matrix_and_labeled_xy():
    df = pd.DataFrame(
        {
            "is_home": ["1", "1", "0"],
            "team_hitting_ytd_ops": [".780", ".720", "bad"],
            "label_win": ["1", "0", ""],
            "ctx_moneyline": ["-110", "-105", ""],
        }
    )
    X, y, cols = tp.labeled_xy(df, tp.TARGETS["team_win"])
    assert len(y) == 2
    assert "team_hitting_ytd_ops" in cols
    assert "is_home" not in cols
    assert list(y.astype(int)) == [1, 0]
    assert X.shape[0] == 2


def test_list_panel_days_and_load(tmp_path: Path):
    panels = tmp_path / "panels"
    (panels / "2026-08-08").mkdir(parents=True)
    (panels / "2026-08-09").mkdir()
    (panels / "not-a-date").mkdir()
    (panels / "2026-08-09" / "team_game.csv").write_text(
        "gamePk,is_home,label_win,team_hitting_ytd_ops\n"
        "1,1,1,.780\n",
        encoding="utf-8",
    )
    days = tp.list_panel_days(tmp_path)
    assert days == [date(2026, 8, 8), date(2026, 8, 9)]

    loaded = tp.load_panels(data_dir=tmp_path, panel="team_game", as_of=date(2026, 8, 9))
    assert len(loaded) == 1
    assert loaded.iloc[0]["gamePk"] == "1"

    through = tp.load_panels(
        data_dir=tmp_path, panel="team_game", through=date(2026, 8, 8)
    )
    assert through.empty
