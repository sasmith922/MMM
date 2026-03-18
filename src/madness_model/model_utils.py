"""Model factory and shared matrix utilities for matchup prediction models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROB_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Model factory and model I/O
# ---------------------------------------------------------------------------


def build_model(model_name: str, random_state: int = 42) -> Any:
    """Build a model instance by name.

    TODO: Add probability calibration wrappers after baseline backtests.
    TODO: Add hyperparameter tuning once baseline leaderboard is stable.
    """

    if model_name in {"logistic_regression", "logistic_baseline", "seed_only_logistic"}:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xgboost is required for model_name='xgboost'.") from exc

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )

    if model_name == "neural_net":
        # TODO: Improve NN normalization/scaling strategy after baseline runs.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Supported: logistic_regression, random_forest, xgboost, neural_net"
    )


def model_supports_predict_proba(model: Any) -> bool:
    """Return ``True`` when a model implements ``predict_proba``."""

    return hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba"))


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a fitted model artifact with joblib."""

    model_path = Path(path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path) -> Any:
    """Load a model artifact saved with :func:`save_model`."""

    model_path = Path(path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Legacy matrix utilities used by predict_matchups
# ---------------------------------------------------------------------------


def _encode_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode feature frame with deterministic numeric output."""

    frame = df.copy()

    for column in frame.columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype(int)

    categorical_cols = frame.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if categorical_cols:
        frame[categorical_cols] = frame[categorical_cols].fillna("MISSING").astype(str)

    return pd.get_dummies(frame, dummy_na=False)


def _fill_missing_with_train_statistics(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaNs using train medians for stable train/test behavior."""

    train_filled = train_df.copy()
    other_filled = other_df.copy()

    train_medians = train_filled.median(axis=0, numeric_only=True)
    train_filled = train_filled.fillna(train_medians).fillna(0.0)
    other_filled = other_filled.fillna(train_medians).fillna(0.0)

    return train_filled, other_filled


def build_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
) -> dict[str, Any]:
    """Build aligned numeric matrices for model training and testing."""

    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    X_train = _encode_feature_frame(X_train_raw)
    X_test = _encode_feature_frame(X_test_raw)

    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0.0)
    X_train, X_test = _fill_missing_with_train_statistics(X_train, X_test)

    y_train = train_df[target_col].astype(int).to_numpy()
    y_test = test_df[target_col].astype(int).to_numpy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_feature_columns": X_train.columns.tolist(),
    }


def build_inference_matrix(
    inference_df: pd.DataFrame,
    feature_cols: list[str],
    model_feature_columns: list[str],
) -> pd.DataFrame:
    """Build an aligned inference matrix that matches training feature schema."""

    X_infer = _encode_feature_frame(inference_df[feature_cols])
    X_infer = X_infer.reindex(columns=model_feature_columns, fill_value=0.0)
    X_infer = X_infer.fillna(0.0)
    return X_infer


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics with safe fallbacks."""

    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y_prob = np.clip(y_prob, PROB_EPSILON, 1 - PROB_EPSILON)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics
