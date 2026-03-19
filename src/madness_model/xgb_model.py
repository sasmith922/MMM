"""
xgb_model.py
------------
XGBoost training and inference for NCAA matchup win probability prediction.

Uses the same feature interface as :mod:`madness_model.baseline_model` so
that training scripts can swap models with minimal changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from madness_model.config import XGB_CONFIG, XGB_MODEL_PATH
from madness_model.baseline_model import NON_FEATURE_COLS, get_feature_cols


def build_model() -> XGBClassifier:
    """Construct an XGBClassifier from :data:`~madness_model.config.XGB_CONFIG`.

    Returns
    -------
    xgboost.XGBClassifier
        Unfitted classifier.
    """
    return XGBClassifier(
        n_estimators=XGB_CONFIG.n_estimators,
        max_depth=XGB_CONFIG.max_depth,
        learning_rate=XGB_CONFIG.learning_rate,
        subsample=XGB_CONFIG.subsample,
        colsample_bytree=XGB_CONFIG.colsample_bytree,
        eval_metric=XGB_CONFIG.eval_metric,
        random_state=XGB_CONFIG.random_state,
        # TODO: enable early stopping with an eval set
    )


def train(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    eval_df: Optional[pd.DataFrame] = None,
) -> XGBClassifier:
    """Fit the XGBoost classifier on a labelled matchup DataFrame.

    Parameters
    ----------
    train_df:
        Labelled matchup DataFrame with a ``label`` column.
    feature_cols:
        Feature columns to use.  Inferred automatically when ``None``.
    eval_df:
        Optional validation DataFrame for early stopping / monitoring.

    Returns
    -------
    xgboost.XGBClassifier
        Fitted classifier.
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    model = build_model()

    eval_set = None
    if eval_df is not None:
        X_eval = eval_df[feature_cols].values
        y_eval = eval_df["label"].values
        eval_set = [(X_eval, y_eval)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )
    return model


def predict_proba(
    model: XGBClassifier,
    matchup_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Predict win probabilities for Team A.

    Parameters
    ----------
    model:
        Fitted XGBClassifier from :func:`train`.
    matchup_df:
        Matchup DataFrame (with or without ``label``).
    feature_cols:
        Feature columns used during training.  Inferred when ``None``.

    Returns
    -------
    np.ndarray, shape (n_matchups,)
        Probability that Team A wins each matchup.
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(matchup_df)
    X = matchup_df[feature_cols].values
    return model.predict_proba(X)[:, 1]


def get_feature_importance(
    model: XGBClassifier,
    feature_cols: List[str],
) -> pd.Series:
    """Return feature importances as a sorted pandas Series.

    Parameters
    ----------
    model:
        Fitted XGBClassifier.
    feature_cols:
        Ordered list of feature column names used during training.

    Returns
    -------
    pd.Series
        Feature importances indexed by feature name, sorted descending.
    """
    importance = model.feature_importances_
    return pd.Series(importance, index=feature_cols).sort_values(ascending=False)


def save_model(model: XGBClassifier, path: Path = XGB_MODEL_PATH) -> None:
    """Save the XGBoost model in its native JSON format.

    Parameters
    ----------
    model:
        Fitted XGBClassifier.
    path:
        Destination file path (should end in ``.json``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path = XGB_MODEL_PATH) -> XGBClassifier:
    """Load a previously saved XGBoost model from JSON.

    Parameters
    ----------
    path:
        Path to the ``.json`` model file.

    Returns
    -------
    xgboost.XGBClassifier
        Loaded classifier (must call fit before using feature names).
    """
    model = XGBClassifier()
    model.load_model(str(path))
    return model
