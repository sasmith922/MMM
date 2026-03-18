"""Shared utilities for season-based splits, feature matrix prep, and metrics."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def _encode_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode feature frame with deterministic numeric output."""
    frame = df.copy()

    for col in frame.columns:
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype(int)

    cat_cols = frame.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        frame[cat_cols] = frame[cat_cols].fillna("MISSING").astype(str)

    encoded = pd.get_dummies(frame, dummy_na=False)
    return encoded


def _fill_missing_with_train_statistics(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN values using train medians for stable train/test behavior."""
    train_filled = train_df.copy()
    other_filled = other_df.copy()

    train_medians = train_filled.median(axis=0, numeric_only=True)
    train_filled = train_filled.fillna(train_medians).fillna(0.0)
    other_filled = other_filled.fillna(train_medians).fillna(0.0)

    return train_filled, other_filled


def get_train_test_split(
    df: pd.DataFrame,
    test_season: int,
    feature_cols: List[str],
    target_col: str = "target",
) -> dict:
    """Split modeling dataframe into rolling train/test subsets by season."""
    required_cols = ["season", *feature_cols, target_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for split: {missing}")

    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    if train_df.empty:
        raise ValueError(f"No training rows for test_season={test_season}.")
    if test_df.empty:
        raise ValueError(f"No test rows for test_season={test_season}.")

    return {
        "train_df": train_df,
        "test_df": test_df,
    }


def build_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
) -> dict:
    """Build aligned numeric matrices for model training and testing."""
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    X_train = _encode_feature_frame(X_train_raw)
    X_test = _encode_feature_frame(X_test_raw)

    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0.0)
    X_train, X_test = _fill_missing_with_train_statistics(X_train, X_test)

    y_train = train_df[target_col].astype(int).values
    y_test = test_df[target_col].astype(int).values

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_feature_columns": X_train.columns.tolist(),
    }


def build_inference_matrix(
    inference_df: pd.DataFrame,
    feature_cols: List[str],
    model_feature_columns: List[str],
) -> pd.DataFrame:
    """Build an aligned inference matrix that matches training feature schema."""
    X_infer = _encode_feature_frame(inference_df[feature_cols])
    X_infer = X_infer.reindex(columns=model_feature_columns, fill_value=0.0)
    X_infer = X_infer.fillna(0.0)
    return X_infer


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute standard binary classification metrics with safe fallbacks."""
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = np.nan

    return metrics
