"""Tests for central modeling dataframe assembly and rolling model training."""

from __future__ import annotations

import pandas as pd

from madness_model.backtest_models import run_backtest
from madness_model.build_model_dataset import build_modeling_dataframe
from madness_model.model_utils import get_train_test_split
from madness_model.train_models import train_single_model


def _sample_team_profiles() -> pd.DataFrame:
    rows = []
    for season in [2018, 2019, 2020, 2021, 2022]:
        rows.extend(
            [
                {
                    "season": season,
                    "team_id": 1,
                    "seed": 1,
                    "win_pct": 0.88,
                    "points_per_game": 80,
                    "points_allowed_per_game": 62,
                    "average_margin": 18,
                    "fg_pct": 0.49,
                    "three_pct": 0.38,
                    "ft_pct": 0.76,
                    "rebounds_per_game": 38,
                    "assists_per_game": 16,
                    "turnovers_per_game": 11,
                    "steals_per_game": 7,
                    "blocks_per_game": 4,
                    "offensive_efficiency": 118,
                    "defensive_efficiency": 92,
                    "net_efficiency": 26,
                    "sos": 8.0,
                    "last10_win_pct": 1.0,
                    "neutral_win_pct": 0.9,
                    "elo_pre_tourney": 1800,
                    "conference": "A",
                },
                {
                    "season": season,
                    "team_id": 2,
                    "seed": 8,
                    "win_pct": 0.62,
                    "points_per_game": 72,
                    "points_allowed_per_game": 68,
                    "average_margin": 4,
                    "fg_pct": 0.45,
                    "three_pct": 0.33,
                    "ft_pct": 0.72,
                    "rebounds_per_game": 34,
                    "assists_per_game": 13,
                    "turnovers_per_game": 13,
                    "steals_per_game": 6,
                    "blocks_per_game": 3,
                    "offensive_efficiency": 107,
                    "defensive_efficiency": 101,
                    "net_efficiency": 6,
                    "sos": 4.0,
                    "last10_win_pct": 0.6,
                    "neutral_win_pct": 0.55,
                    "elo_pre_tourney": 1630,
                    "conference": "A",
                },
                {
                    "season": season,
                    "team_id": 3,
                    "seed": 5,
                    "win_pct": 0.70,
                    "points_per_game": 75,
                    "points_allowed_per_game": 66,
                    "average_margin": 9,
                    "fg_pct": 0.47,
                    "three_pct": 0.35,
                    "ft_pct": 0.74,
                    "rebounds_per_game": 36,
                    "assists_per_game": 14,
                    "turnovers_per_game": 12,
                    "steals_per_game": 6.5,
                    "blocks_per_game": 3.5,
                    "offensive_efficiency": 111,
                    "defensive_efficiency": 97,
                    "net_efficiency": 14,
                    "sos": 6.0,
                    "last10_win_pct": 0.8,
                    "neutral_win_pct": 0.7,
                    "elo_pre_tourney": 1700,
                    "conference": "B",
                },
            ]
        )
    return pd.DataFrame(rows)


def _sample_matchups() -> pd.DataFrame:
    rows = []
    for season in [2018, 2019, 2020, 2021, 2022]:
        rows.extend(
            [
                {
                    "season": season,
                    "round": "R64",
                    "region": "East",
                    "teamA_id": 1,
                    "teamB_id": 2,
                    "target": 1,
                },
                {
                    "season": season,
                    "round": "R64",
                    "region": "West",
                    "teamA_id": 3,
                    "teamB_id": 2,
                    "target": 1,
                },
                {
                    "season": season,
                    "round": "R32",
                    "region": "East",
                    "teamA_id": 2,
                    "teamB_id": 1,
                    "target": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _sample_games_boxscores() -> pd.DataFrame:
    rows = []
    for season in [2018, 2019, 2020, 2021, 2022]:
        rows.extend(
            [
                {"season": season, "team_id": 1, "pace": 70.0, "is_tourney": False},
                {"season": season, "team_id": 1, "pace": 72.0, "is_tourney": False},
                {"season": season, "team_id": 2, "pace": 66.0, "is_tourney": False},
                {"season": season, "team_id": 3, "pace": 68.0, "is_tourney": False},
            ]
        )
    return pd.DataFrame(rows)


def test_build_modeling_dataframe_assembles_sources_and_engineers_features() -> None:
    model_df = build_modeling_dataframe(
        team_profiles_df=_sample_team_profiles(),
        tourney_matchups_df=_sample_matchups(),
        games_boxscores_df=_sample_games_boxscores(),
    )

    assert "teamA_win_pct" in model_df.columns
    assert "teamB_win_pct" in model_df.columns
    assert "teamA_box_pace" in model_df.columns
    assert "teamB_box_pace" in model_df.columns
    assert "seed_diff" in model_df.columns
    assert "win_pct_diff" in model_df.columns
    assert "off_eff_diff" in model_df.columns
    assert "same_conference_flag" in model_df.columns
    assert "upset_bucket" in model_df.columns

    first = model_df.iloc[0]
    assert first["seed_diff"] == -7
    assert first["same_conference_flag"] == 1


def test_season_split_and_train_single_model_use_rolling_logic() -> None:
    model_df = build_modeling_dataframe(
        team_profiles_df=_sample_team_profiles(),
        tourney_matchups_df=_sample_matchups(),
        games_boxscores_df=_sample_games_boxscores(),
    )

    split = get_train_test_split(model_df, test_season=2022, feature_cols=["seed_diff"])
    assert split["train_df"]["season"].max() == 2021
    assert set(split["test_df"]["season"].unique()) == {2022}

    result = train_single_model(
        modeling_df=model_df,
        model_name="seed_only_logistic",
        test_season=2022,
        random_state=42,
    )

    assert set(result["predictions_df"].columns) >= {
        "season",
        "round",
        "teamA_id",
        "teamB_id",
        "target",
        "pred_prob",
        "pred_class",
        "model_name",
    }
    assert result["metrics"]["test_season"] == 2022
    assert result["metrics"]["n_train"] == len(model_df[model_df["season"] < 2022])


def test_run_backtest_collects_metrics_and_predictions_without_saving() -> None:
    model_df = build_modeling_dataframe(
        team_profiles_df=_sample_team_profiles(),
        tourney_matchups_df=_sample_matchups(),
        games_boxscores_df=_sample_games_boxscores(),
    )

    results = run_backtest(
        modeling_df=model_df,
        model_names=["seed_only_logistic", "random_forest"],
        test_seasons=[2021, 2022],
        min_train_seasons=2,
        random_state=42,
        save_outputs=False,
    )

    metrics_df = results["metrics_by_season"]
    preds_df = results["predictions"]

    assert len(metrics_df) == 4  # 2 models x 2 seasons
    assert sorted(metrics_df["model_name"].unique().tolist()) == [
        "random_forest",
        "seed_only_logistic",
    ]
    assert set(preds_df["season"].unique()) == {2021, 2022}
