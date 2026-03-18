"""Model training utilities for rolling-season matchup prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from madness_model.evaluate_models import compute_metrics
from madness_model.feature_config import TARGET_COL, get_feature_columns
from madness_model.model_utils import (
    build_model,
    build_train_test_matrices,
    model_supports_predict_proba,
    save_model,
)
from madness_model.paths import MODELS_DIR


def get_train_test_split(
    modeling_df: pd.DataFrame,
    test_season: int,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split data with strict season holdout: train ``< Y`` and test ``== Y``."""

    required = ["season", target_col, *feature_cols]
    missing = [column for column in required if column not in modeling_df.columns]
    if missing:
        raise KeyError(f"modeling_df missing required split columns: {missing}")

    train_df = modeling_df[modeling_df["season"] < test_season].copy()
    test_df = modeling_df[modeling_df["season"] == test_season].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found for test_season={test_season}.")
    if test_df.empty:
        raise ValueError(f"No test rows found for test_season={test_season}.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(int).copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(int).copy()

    return X_train, y_train, X_test, y_test, train_df, test_df


def get_available_test_seasons(
    modeling_df: pd.DataFrame,
    min_train_seasons: int = 5,
) -> list[int]:
    """Return seasons that have enough prior unique seasons for training."""

    if "season" not in modeling_df.columns:
        raise KeyError("modeling_df must include a 'season' column.")

    seasons = sorted(modeling_df["season"].dropna().astype(int).unique().tolist())
    available: list[int] = []

    for season in seasons:
        prior_seasons = [prior for prior in seasons if prior < season]
        if len(prior_seasons) >= min_train_seasons:
            available.append(season)

    return available


def _predict_probabilities(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    if not model_supports_predict_proba(model):
        raise ValueError(f"Model {type(model).__name__} does not support predict_proba().")
    return np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)


def train_single_model_for_season(
    modeling_df: pd.DataFrame,
    model_name: str,
    test_season: int,
    random_state: int = 42,
    strict_features: bool = True,
    save_model_artifact: bool = True,
) -> dict[str, Any]:
    """Train one model on seasons before ``test_season`` and evaluate held-out season.

    TODO: Add post-hoc probability calibration per season/model.
    """

    feature_cols = get_feature_columns(modeling_df, model_name, strict=strict_features)
    _X_train, _y_train, _X_test, _y_test, train_df, test_df = get_train_test_split(
        modeling_df,
        test_season=test_season,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
    )

    matrices = build_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
    )

    model = build_model(model_name=model_name, random_state=random_state)
    model.fit(matrices["X_train"], matrices["y_train"])

    y_prob = _predict_probabilities(model, matrices["X_test"])
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model_name": model_name,
        "test_season": int(test_season),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **compute_metrics(matrices["y_test"], y_prob, y_pred),
    }

    prediction_cols = [column for column in ["season", "round", "teamA_id", "teamB_id", TARGET_COL] if column in test_df.columns]
    predictions_df = test_df[prediction_cols].copy()
    predictions_df["pred_prob"] = y_prob
    predictions_df["pred_class"] = y_pred
    predictions_df["model_name"] = model_name
    predictions_df["test_season"] = int(test_season)

    if save_model_artifact:
        artifact_path = Path(MODELS_DIR) / f"{model_name}_{test_season}.joblib"
        save_model(model, artifact_path)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "predictions": predictions_df,
        "metrics": metrics,
    }


# Backward-compat wrapper for previous API.
def train_single_model(
    modeling_df: pd.DataFrame,
    model_name: str,
    test_season: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compatibility wrapper around ``train_single_model_for_season``."""

    result = train_single_model_for_season(
        modeling_df=modeling_df,
        model_name=model_name,
        test_season=test_season,
        random_state=random_state,
        strict_features=False,
        save_model_artifact=False,
    )
    return {
        "model": result["model"],
        "feature_cols": result["feature_cols"],
        "predictions_df": result["predictions"],
        "metrics": result["metrics"],
    }
