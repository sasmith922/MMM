"""Unit tests for scripts/build_features_v2.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_v2_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_features_v2.py"
    spec = importlib.util.spec_from_file_location("build_features_v2_script", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Unable to load build_features_v2.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_team_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "kaggle_team_id": [1, 2, 3],
            "seed": [1, 8, 9],
            "win_pct": [0.80, 0.55, 0.50],
            "conference_wins": [10, 8, 7],
            "conference_losses": [2, 4, 5],
            "away_wins": [5, 3, 2],
            "away_losses": [2, 4, 5],
            "neutral_wins": [3, 2, 2],
            "neutral_losses": [1, 2, 2],
            "avg_margin": [12.0, 3.0, 1.0],
            "pre_tourney_adjoe": [117.0, 111.0, 109.0],
            "pre_tourney_adjde": [95.0, 101.0, 103.0],
            "pre_tourney_tempo": [68.0, 70.0, 69.0],
            "top25_wins": [6, 2, 1],
            "top50_wins": [10, 5, 4],
            "wins": [24, 18, 16],
            "sos_elo": [1540.0, 1505.0, 1490.0],
            "resume_delta": [2.4, 0.8, -0.2],
            "experience": [2.3, 1.9, 1.8],
            "net_rating": [22.0, 10.0, 6.0],
            "elo_pre_tourney": [1760.0, 1650.0, 1600.0],
            "elo_rank": [5, 28, 42],
            "bad_losses_100plus": [0, 1, 2],
            "nonconf_wins": [9, 6, 5],
            "nonconf_losses": [1, 3, 4],
        }
    )


def _make_games_boxscores() -> pd.DataFrame:
    rows = []
    # Team 1 regular season (10 games)
    for i in range(10):
        rows.append(
            {
                "season": 2022,
                "season_type": "regular",
                "daynum": 10 + i,
                "team_id": 1,
                "team_score": 80 + i,
                "opponent_team_score": 70 + (i % 3),
                "team_winner": 1 if i < 8 else 0,
                "team_home_away": "away" if i % 2 == 0 else "neutral",
                "team_fga": 60,
                "team_or": 10,
                "team_to": 12,
                "team_fta": 18,
                "opponent_fga": 58,
                "opponent_or": 9,
                "opponent_to": 13,
                "opponent_fta": 16,
            }
        )

    # Team 2 regular season (8 games)
    for i in range(8):
        rows.append(
            {
                "season": 2022,
                "season_type": "regular",
                "daynum": 20 + i,
                "team_id": 2,
                "team_score": 72 + i,
                "opponent_team_score": 71 + (i % 4),
                "team_winner": 1 if i < 5 else 0,
                "team_home_away": "away" if i % 3 == 0 else "neutral",
                "team_fga": 57,
                "team_or": 9,
                "team_to": 11,
                "team_fta": 17,
                "opponent_fga": 59,
                "opponent_or": 10,
                "opponent_to": 12,
                "opponent_fta": 18,
            }
        )

    # Team 3 regular season (8 games)
    for i in range(8):
        rows.append(
            {
                "season": 2022,
                "season_type": "regular",
                "daynum": 30 + i,
                "team_id": 3,
                "team_score": 69 + i,
                "opponent_team_score": 70 + (i % 3),
                "team_winner": 1 if i < 4 else 0,
                "team_home_away": "neutral" if i % 2 == 0 else "away",
                "team_fga": 56,
                "team_or": 8,
                "team_to": 12,
                "team_fta": 16,
                "opponent_fga": 58,
                "opponent_or": 9,
                "opponent_to": 11,
                "opponent_fta": 17,
            }
        )

    # Postseason row should be filtered out by season_type.
    rows.append(
        {
            "season": 2022,
            "season_type": "postseason",
            "daynum": 150,
            "team_id": 1,
            "team_score": 50,
            "opponent_team_score": 80,
            "team_winner": 0,
            "team_home_away": "neutral",
            "team_fga": 55,
            "team_or": 7,
            "team_to": 14,
            "team_fta": 12,
            "opponent_fga": 57,
            "opponent_or": 10,
            "opponent_to": 10,
            "opponent_fta": 19,
        }
    )
    return pd.DataFrame(rows)


def _make_tourney_matchups() -> pd.DataFrame:
    # mirrored rows for one game plus another game
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022, 2022],
            "daynum": [136, 136, 138, 138],
            "team_a_id": [1, 2, 1, 3],
            "team_b_id": [2, 1, 3, 1],
            "target": [1, 0, 1, 0],
            "round_num_guess": [1, 1, 2, 2],
            "region": ["W", "W", "W", "W"],
            "game_id": ["g1", "g1", "g2", "g2"],
        }
    )


def test_build_team_season_features_stable_schema_and_uniques() -> None:
    module = _load_v2_script_module()
    features = module.build_team_season_features(
        _make_team_profiles(),
        _make_games_boxscores(),
        _make_tourney_matchups(),
    )
    assert list(features.columns) == module.TEAM_FEATURE_COLUMNS
    assert features[["season", "team_id"]].duplicated().sum() == 0
    assert len(features) == 3


def test_build_team_season_features_recent_form_uses_pre_tourney_games() -> None:
    module = _load_v2_script_module()
    features = module.build_team_season_features(
        _make_team_profiles(),
        _make_games_boxscores(),
        _make_tourney_matchups(),
    )
    team1 = features[features["team_id"] == 1].iloc[0]
    # last 5 among 10 regular games: wins in first 8, losses in last 2 -> last 5 has 3 wins
    assert team1["last_5_win_pct"] == 0.6
    # postseason loss should not be counted
    assert team1["overall_win_pct"] == 0.80


def test_build_tournament_matchups_stable_schema_and_no_dupes() -> None:
    module = _load_v2_script_module()
    features = module.build_team_season_features(
        _make_team_profiles(),
        _make_games_boxscores(),
        _make_tourney_matchups(),
    )
    matchups = module.build_tournament_matchups(features, _make_tourney_matchups())
    assert list(matchups.columns) == module.MATCHUP_COLUMNS
    assert matchups[["season", "team1_id", "team2_id"]].duplicated().sum() == 0
    assert set(matchups["label"].dropna().unique()).issubset({0, 1})
    assert len(matchups) == 2

