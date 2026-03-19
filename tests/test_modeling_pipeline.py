"""Tests for matchup-model dataset assembly and season-heldout training pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from madness_model.backtest_runner import run_backtest
from madness_model.build_model_dataset import _coerce_team_profiles, build_modeling_dataframe
from madness_model.evaluate_models import compute_metrics
from madness_model.feature_config import get_feature_columns
from madness_model.train_models import (
    get_available_test_seasons,
    get_train_test_split,
    train_single_model_for_season,
)


def _sample_team_profiles() -> pd.DataFrame:
    rows = []
    for season in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
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
    for season in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
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
    for season in [2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        rows.extend(
            [
                {"season": season, "team_id": 1, "pace": 70.0, "is_tourney": False},
                {"season": season, "team_id": 1, "pace": 72.0, "is_tourney": False},
                {"season": season, "team_id": 2, "pace": 66.0, "is_tourney": False},
                {"season": season, "team_id": 3, "pace": 68.0, "is_tourney": False},
            ]
        )
    return pd.DataFrame(rows)


def _build_sample_model_df() -> pd.DataFrame:
    return build_modeling_dataframe(
        team_profiles_df=_sample_team_profiles(),
        tourney_matchups_df=_sample_matchups(),
        games_boxscores_df=_sample_games_boxscores(),
    )


def test_build_modeling_dataframe_assembles_sources_and_engineers_features() -> None:
    model_df = _build_sample_model_df()

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


def test_get_train_test_split_uses_season_heldout_logic() -> None:
    model_df = _build_sample_model_df()
    feature_cols = get_feature_columns(model_df, "logistic_regression", strict=True)

    _, _, _, _, train_df, test_df = get_train_test_split(
        model_df,
        test_season=2023,
        feature_cols=feature_cols,
    )

    assert train_df["season"].max() == 2022
    assert set(test_df["season"].unique()) == {2023}


def test_train_single_model_for_season_returns_predictions_and_metrics() -> None:
    model_df = _build_sample_model_df()

    result = train_single_model_for_season(
        modeling_df=model_df,
        model_name="logistic_regression",
        test_season=2023,
        random_state=42,
        save_model_artifact=False,
    )

    assert set(result["predictions"].columns) >= {
        "season",
        "round",
        "teamA_id",
        "teamB_id",
        "target",
        "pred_prob",
        "pred_class",
        "model_name",
        "test_season",
    }
    assert result["metrics"]["test_season"] == 2023
    assert result["metrics"]["n_train"] == len(model_df[model_df["season"] < 2023])


def test_get_available_test_seasons_respects_min_train_seasons() -> None:
    model_df = _build_sample_model_df()
    assert get_available_test_seasons(model_df, min_train_seasons=5) == [2022, 2023]


def test_run_backtest_collects_metrics_predictions_and_summary() -> None:
    model_df = _build_sample_model_df()

    results = run_backtest(
        modeling_df=model_df,
        model_names=["logistic_regression", "random_forest"],
        test_seasons=[2022, 2023],
        min_train_seasons=2,
        random_state=42,
        save_outputs=False,
    )

    metrics_df = results["metrics"]
    preds_df = results["predictions"]
    summary_df = results["summary"]

    assert len(metrics_df) == 4  # 2 models x 2 seasons
    assert sorted(metrics_df["model_name"].unique().tolist()) == [
        "logistic_regression",
        "random_forest",
    ]
    assert set(preds_df["season"].unique()) == {2022, 2023}
    assert set(summary_df["model_name"].unique()) == {
        "logistic_regression",
        "random_forest",
    }


def test_compute_metrics_handles_single_class_roc_auc_edge_case() -> None:
    metrics = compute_metrics(
        y_true=np.array([1, 1, 1]),
        y_prob=np.array([0.8, 0.7, 0.9]),
        y_pred=np.array([1, 1, 1]),
    )

    assert np.isnan(metrics["roc_auc"])
    assert metrics["n_samples"] == 3


def test_coerce_team_profiles_dedupes_by_preferred_completeness(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    team_profiles = pd.DataFrame(
        [
            {
                "season": 2023,
                "team_id": 10,
                "seed": np.nan,
                "elo_pre_tourney": np.nan,
                "team_name": "Alpha",
                "canonical_team_name": np.nan,
                "win_pct": 0.70,
            },
            {
                "season": 2023,
                "team_id": 10,
                "seed": 7,
                "elo_pre_tourney": np.nan,
                "team_name": np.nan,
                "canonical_team_name": np.nan,
                "win_pct": np.nan,
            },
            {
                "season": 2023,
                "team_id": 10,
                "seed": 7,
                "elo_pre_tourney": 1650,
                "team_name": "Alpha Wildcats",
                "canonical_team_name": "alpha-wildcats",
                "win_pct": 0.71,
            },
            {
                "season": 2023,
                "team_id": 20,
                "seed": 12,
                "elo_pre_tourney": 1500,
                "team_name": "Beta",
                "canonical_team_name": "beta",
                "win_pct": 0.55,
            },
        ]
    )

    deduped = _coerce_team_profiles(team_profiles)

    assert len(deduped) == 2
    chosen = deduped.loc[(deduped["season"] == 2023) & (deduped["team_id"] == 10)].iloc[0]
    assert chosen["seed"] == 7
    assert chosen["elo_pre_tourney"] == 1650
    assert chosen["team_name"] == "Alpha Wildcats"
    assert chosen["canonical_team_name"] == "alpha-wildcats"

    audit_path = tmp_path / "outputs" / "reports" / "team_profiles_duplicates_audit.csv"
    assert audit_path.exists()
    audit_df = pd.read_csv(audit_path)
    assert set(audit_df["team_id"]) == {10}
