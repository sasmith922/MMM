"""
visualize.py
------------
Plotting helpers for the March Madness prediction pipeline.

All functions return a ``matplotlib.figure.Figure`` so callers can either
display it interactively or save it to :data:`~madness_model.paths.FIGURES_DIR`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from madness_model.paths import FIGURES_DIR


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a reliability (calibration) curve.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted win probabilities.
    model_name:
        Label used in the plot legend.
    n_bins:
        Number of probability bins.
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from madness_model.calibrate import get_calibration_curve

    frac_pos, mean_pred = get_calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    # TODO: add histogram of prediction distribution as a subplot

    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    importances:
        pd.Series indexed by feature name, sorted descending (as returned by
        :func:`~madness_model.xgb_model.get_feature_importance`).
    top_n:
        Number of top features to display.
    save_path:
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))
    top[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_team_champion_odds(
    simulation_df: pd.DataFrame,
    team_names: Optional[dict] = None,
    top_n: int = 16,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a bar chart of each team's estimated championship probability.

    Parameters
    ----------
    simulation_df:
        Output of
        :func:`~madness_model.simulate_bracket.monte_carlo_simulation`.
        Must contain columns ``team_id`` and ``champion_pct``.
    team_names:
        Optional mapping ``{team_id: "Team Name"}`` for axis labels.
    top_n:
        Number of top teams to display.
    save_path:
        If provided, save the figure here.

    Returns
    -------
    matplotlib.figure.Figure
    """
    top = simulation_df.head(top_n).copy()

    if team_names:
        top["label"] = top["team_id"].map(team_names).fillna(top["team_id"].astype(str))
    else:
        top["label"] = top["team_id"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["label"], top["champion_pct"] * 100, color="steelblue")
    ax.set_ylabel("Champion probability (%)")
    ax.set_title(f"Top {top_n} Teams by Championship Probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    # TODO: add error bars using standard deviation across simulations

    return fig
