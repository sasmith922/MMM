"""Focused tests for new model-training utility modules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from madness_model.feature_config import get_feature_columns
from madness_model.load_processed_data import (
    load_games_boxscores,
    load_team_profiles,
    load_tourney_matchups,
)
from madness_model.model_utils import build_model, model_supports_predict_proba, save_model


def test_load_processed_data_validates_required_columns(tmp_path: Path) -> None:
    team_profiles = tmp_path / "team_profiles.csv"
    games_boxscores = tmp_path / "games_boxscores.csv"
    tourney_matchups = tmp_path / "tourney_matchups.csv"

    pd.DataFrame([{"season": 2023, "team_id": 1, "win_pct": 0.8}]).to_csv(team_profiles, index=False)
    pd.DataFrame([{"season": 2023, "team_id": 1, "pace": 70.0}]).to_csv(games_boxscores, index=False)
    pd.DataFrame(
        [{"season": 2023, "teamA_id": 1, "teamB_id": 2, "target": 1}]
    ).to_csv(tourney_matchups, index=False)

    assert not load_team_profiles(path=team_profiles).empty
    assert not load_games_boxscores(path=games_boxscores).empty
    assert not load_tourney_matchups(path=tourney_matchups).empty


def test_load_tourney_matchups_raises_on_missing_required_columns(tmp_path: Path) -> None:
    bad_matchups = tmp_path / "bad_matchups.csv"
    pd.DataFrame([{"season": 2023, "teamA_id": 1, "teamB_id": 2}]).to_csv(bad_matchups, index=False)

    with pytest.raises(KeyError):
        load_tourney_matchups(path=bad_matchups)


def test_feature_config_get_feature_columns_strict_and_non_strict() -> None:
    df = pd.DataFrame({"seed_diff": [1], "win_pct_diff": [0.1]})

    assert get_feature_columns(df, "logistic_regression", strict=False) == [
        "seed_diff",
        "win_pct_diff",
    ]

    with pytest.raises(KeyError):
        get_feature_columns(df, "logistic_regression", strict=True)


def test_model_utils_build_and_save_model(tmp_path: Path) -> None:
    model = build_model("logistic_regression", random_state=42)
    assert model_supports_predict_proba(model)

    model_path = tmp_path / "model.joblib"
    saved = save_model(model, model_path)
    assert saved.exists()
