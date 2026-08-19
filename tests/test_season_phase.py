from datetime import date

import pytest

import season_phase as sp


def test_stats_season_calendar_rules():
    assert sp.stats_season(date(2026, 8, 10)) == 2026
    assert sp.stats_season(date(2026, 3, 1)) == 2026
    assert sp.stats_season(date(2026, 2, 28)) == 2025
    assert sp.stats_season(date(2026, 1, 15)) == 2025


def test_slate_season_matches_stats_season():
    as_of = date(2026, 1, 10)
    assert sp.slate_season(as_of) == sp.stats_season(as_of)


def test_calendar_bias_winter_only():
    assert sp._calendar_bias(date(2026, 1, 5)) == sp.PHASE_OFFSEASON
    assert sp._calendar_bias(date(2026, 12, 20)) == sp.PHASE_OFFSEASON
    assert sp._calendar_bias(date(2026, 7, 4)) is None


def test_detect_phase_override():
    info = sp.detect_phase(date(2026, 8, 10), override="postseason", game_types=["R"])
    assert info["phase"] == "postseason"
    assert info["reason"].startswith("override=")
    assert info["skip_slate_steps"] is False


def test_detect_phase_invalid_override():
    with pytest.raises(ValueError):
        sp.detect_phase(date(2026, 8, 10), override="spring")


def test_detect_phase_from_injected_game_types():
    post = sp.detect_phase(date(2026, 10, 15), game_types=["D", "S"])
    assert post["phase"] == sp.PHASE_POSTSEASON

    regular = sp.detect_phase(date(2026, 6, 1), game_types=["R"])
    assert regular["phase"] == sp.PHASE_REGULAR

    spring_only = sp.detect_phase(date(2026, 1, 20), game_types=["S"])
    assert spring_only["phase"] == sp.PHASE_OFFSEASON
    assert spring_only["prefer_ytd_only"] is True
    assert spring_only["skip_slate_steps"] is True


def test_workflow_step_policy_offseason():
    policy = sp.workflow_step_policy(sp.PHASE_OFFSEASON)
    assert policy["schedule"] is False
    assert policy["odds"] is False
    assert policy["predict"] is False
    assert policy["train"] is True
    assert policy["stats"] is True


def test_workflow_step_policy_regular_morning():
    policy = sp.workflow_step_policy(sp.PHASE_REGULAR, morning_run=True)
    assert policy["boxscore_today"] is False
    assert policy["predict"] is True
    assert policy["schedule"] is True


def test_workflow_step_policy_zero_games():
    policy = sp.workflow_step_policy(sp.PHASE_REGULAR, games_today=0)
    assert policy["odds"] is False
    assert policy["align"] is False
    assert policy["predict"] is False
    assert policy["stats"] is True
