"""
calibrate.py
------------
Probability calibration helpers.

After training a model on historical data, its raw predicted probabilities
may not reflect true win likelihoods (e.g., a model may be over-confident).
Calibration techniques like Platt scaling and isotonic regression address
this.

This module wraps scikit-learn's CalibratedClassifierCV and exposes a
project-consistent save/load interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from madness_model.config import CALIBRATED_MODEL_PATH
from madness_model.baseline_model import get_feature_cols


def calibrate_model(
    base_model,
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wrap *base_model* in a calibrated classifier and fit it.

    Parameters
    ----------
    base_model:
        A fitted scikit-learn-compatible classifier (e.g., XGBClassifier
        wrapped to expose ``predict_proba``).
    train_df:
        Labelled matchup DataFrame used for calibration fitting.
    feature_cols:
        Feature columns.  Inferred automatically when ``None``.
    method:
        Calibration method: ``"isotonic"`` or ``"sigmoid"`` (Platt scaling).
    cv:
        Number of cross-validation folds, or ``"prefit"`` if the base model
        is already fitted and should not be refitted.

    Returns
    -------
    sklearn.calibration.CalibratedClassifierCV
        Fitted calibrated model.
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df)

    X = train_df[feature_cols].values
    y = train_df["label"].values

    calibrated = CalibratedClassifierCV(base_model, method=method, cv=cv)
    calibrated.fit(X, y)
    return calibrated


def get_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve data (fraction of positives vs mean predicted prob).

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities for the positive class.
    n_bins:
        Number of bins for the calibration curve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(fraction_of_positives, mean_predicted_value)`` arrays.
    """
    # TODO: add confidence intervals via bootstrap
    return calibration_curve(y_true, y_prob, n_bins=n_bins)


def save_model(model: CalibratedClassifierCV, path: Path = CALIBRATED_MODEL_PATH) -> None:
    """Persist a calibrated model to disk.

    Parameters
    ----------
    model:
        Fitted calibrated classifier.
    path:
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = CALIBRATED_MODEL_PATH) -> CalibratedClassifierCV:
    """Load a calibrated model from disk.

    Parameters
    ----------
    path:
        Path to the serialised calibrated model file.

    Returns
    -------
    sklearn.calibration.CalibratedClassifierCV
    """
    return joblib.load(path)
