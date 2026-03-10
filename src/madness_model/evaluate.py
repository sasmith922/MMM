"""
evaluate.py
-----------
Model evaluation metrics for binary win-probability predictions.

Computes standard metrics used for probability-based classifiers:
- Accuracy
- Log loss (cross-entropy)
- Brier score (mean squared probability error)
- AUC-ROC

All functions accept numpy arrays or pandas Series interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass
class EvaluationReport:
    """Container for all evaluation metrics.

    Attributes
    ----------
    accuracy:
        Fraction of correctly predicted winners.
    log_loss:
        Binary cross-entropy loss (lower is better).
    brier_score:
        Mean squared error of predicted probabilities (lower is better).
    auc_roc:
        Area under the ROC curve (higher is better).
    n_samples:
        Number of evaluation samples.
    """

    accuracy: float
    log_loss: float
    brier_score: float
    auc_roc: float
    n_samples: int

    def to_dict(self) -> dict:
        """Serialise the report to a plain dictionary."""
        return {
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "auc_roc": self.auc_roc,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"EvaluationReport("
            f"n={self.n_samples}, "
            f"acc={self.accuracy:.4f}, "
            f"log_loss={self.log_loss:.4f}, "
            f"brier={self.brier_score:.4f}, "
            f"auc={self.auc_roc:.4f})"
        )


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> EvaluationReport:
    """Compute all evaluation metrics at once.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class (Team A wins).

    Returns
    -------
    EvaluationReport
        Dataclass containing all metric values.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    y_pred = (y_prob_arr >= 0.5).astype(int)

    return EvaluationReport(
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        log_loss=float(log_loss(y_true_arr, y_prob_arr)),
        brier_score=float(brier_score_loss(y_true_arr, y_prob_arr)),
        auc_roc=float(roc_auc_score(y_true_arr, y_prob_arr)),
        n_samples=len(y_true_arr),
    )


def print_report(report: EvaluationReport) -> None:  # pragma: no cover
    """Pretty-print an :class:`EvaluationReport`.

    Parameters
    ----------
    report:
        Report to display.
    """
    print("=" * 40)
    print("  Model Evaluation Report")
    print("=" * 40)
    print(f"  Samples    : {report.n_samples}")
    print(f"  Accuracy   : {report.accuracy:.4f}")
    print(f"  Log Loss   : {report.log_loss:.4f}")
    print(f"  Brier Score: {report.brier_score:.4f}")
    print(f"  AUC-ROC    : {report.auc_roc:.4f}")
    print("=" * 40)
