"""Model training utilities for rolling-season matchup prediction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from madness_model.feature_config import TARGET_COLUMN, get_model_feature_columns
from madness_model.model_utils import (
    build_train_test_matrices,
    calculate_classification_metrics,
    get_train_test_split,
)


def _build_model(model_name: str, random_state: int = 42) -> Any:
    if model_name in {"seed_only_logistic", "logistic_baseline"}:
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
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
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Supported: seed_only_logistic, logistic_baseline, random_forest, xgboost"
    )


def train_single_model(
    modeling_df: pd.DataFrame,
    model_name: str,
    test_season: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a single model using all seasons prior to test_season and score test season."""
    feature_cols = get_model_feature_columns(
        modeling_df_columns=modeling_df.columns.tolist(),
        model_name=model_name,
        strict=False,
    )

    split = get_train_test_split(
        modeling_df,
        test_season=test_season,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
    )
    train_df = split["train_df"]
    test_df = split["test_df"]

    matrices = build_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
    )

    model = _build_model(model_name, random_state=random_state)
    model.fit(matrices["X_train"], matrices["y_train"])

    pred_prob = model.predict_proba(matrices["X_test"])[:, 1]
    pred_class = (pred_prob >= 0.5).astype(int)

    metrics = {
        "test_season": int(test_season),
        "model_name": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **calculate_classification_metrics(matrices["y_test"], pred_prob),
    }

    prediction_cols = ["season", "teamA_id", "teamB_id", TARGET_COLUMN]
    if "round" in test_df.columns:
        prediction_cols.insert(1, "round")

    predictions_df = test_df[prediction_cols].copy()
    predictions_df["pred_prob"] = pred_prob
    predictions_df["pred_class"] = pred_class
    predictions_df["model_name"] = model_name

    return {
        "model": model,
        "feature_cols": feature_cols,
        "model_feature_cols": matrices["model_feature_columns"],
        "predictions_df": predictions_df,
        "metrics": metrics,
    }
