"""Evaluation helpers for model backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Compute classification metrics with robust edge-case handling."""

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    y_prob_clipped = np.clip(y_prob_arr, 1e-6, 1 - 1e-6)

    metrics: dict[str, float | int] = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "log_loss": float(log_loss(y_true_arr, y_prob_clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_clipped)),
    }

    if np.unique(y_true_arr).size < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_clipped))

    return metrics


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize model metrics across backtest seasons."""

    if metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "mean_accuracy",
                "std_accuracy",
                "mean_log_loss",
                "std_log_loss",
                "mean_brier_score",
                "std_brier_score",
                "mean_roc_auc",
                "std_roc_auc",
            ]
        )

    summary = (
        metrics_df.groupby("model_name", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_brier_score=("brier_score", "mean"),
            std_brier_score=("brier_score", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
        )
        .sort_values("mean_log_loss", ascending=True)
        .reset_index(drop=True)
    )

    return summary
