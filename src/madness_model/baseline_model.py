"""
baseline_model.py
-----------------
Logistic regression baseline model for predicting NCAA matchup win
probabilities.

This module wraps scikit-learn's LogisticRegression in a thin, project-
consistent interface so it can be swapped for other models without changing
the training scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from madness_model.config import BASELINE_MODEL_PATH, LOGREG_CONFIG


# Columns that are not model features (identifiers / label)
NON_FEATURE_COLS: List[str] = ["season", "team_a_id", "team_b_id", "label"]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns in a matchup DataFrame.

    Parameters
    ----------
    df:
        Matchup DataFrame produced by :mod:`madness_model.build_matchups`.

    Returns
    -------
    list of str
        Column names that are model inputs.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_pipeline() -> Pipeline:
    """Construct a scikit-learn Pipeline with StandardScaler + LogisticRegression.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for training.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=LOGREG_CONFIG.C,
                    max_iter=LOGREG_CONFIG.max_iter,
                    solver=LOGREG_CONFIG.solver,
                    random_state=LOGREG_CONFIG.random_state,
                ),
            ),
        ]
    )


def train(
    train_df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Pipeline:
    """Fit the baseline logistic regression pipeline.

    Parameters
    ----------
    train_df:
        Labelled matchup DataFrame with a ``label`` column.
    feature_cols:
        Explicit list of feature columns.  Inferred automatically when
        ``None``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df)

    X = train_df[feature_cols].values
    y = train_df["label"].values

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    return pipeline


def predict_proba(
    pipeline: Pipeline,
    matchup_df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> np.ndarray:
    """Predict win probabilities for Team A in each matchup row.

    Parameters
    ----------
    pipeline:
        Fitted pipeline from :func:`train`.
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
    return pipeline.predict_proba(X)[:, 1]


def save_model(pipeline: Pipeline, path: Path = BASELINE_MODEL_PATH) -> None:
    """Persist the fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline:
        Fitted pipeline to save.
    path:
        Destination file path.  Parent directory is created if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: Path = BASELINE_MODEL_PATH) -> Pipeline:
    """Load a previously saved baseline pipeline from disk.

    Parameters
    ----------
    path:
        Path to the serialised pipeline file.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline.
    """
    return joblib.load(path)
