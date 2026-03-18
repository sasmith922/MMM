"""Rolling-season backtest runner for matchup prediction models."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from madness_model.paths import PREDICTIONS_DIR, REPORTS_DIR
from madness_model.train_models import train_single_model


def _infer_test_seasons(modeling_df: pd.DataFrame, min_train_seasons: int) -> List[int]:
    seasons = sorted(modeling_df["season"].dropna().astype(int).unique().tolist())
    valid = []
    for season in seasons:
        prior_count = sum(prev < season for prev in seasons)
        if prior_count >= min_train_seasons:
            valid.append(season)
    return valid


def run_backtest(
    modeling_df: pd.DataFrame,
    model_names: list[str],
    test_seasons: list[int] | None = None,
    min_train_seasons: int = 5,
    random_state: int = 42,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Run rolling season backtests across one or more model types.

    Parameters
    ----------
    modeling_df:
        Matchup-level modeling dataframe with a ``season`` column.
    model_names:
        List of model names to evaluate per season.
    test_seasons:
        Explicit seasons to evaluate. If None, valid seasons are inferred based
        on ``min_train_seasons``.
    min_train_seasons:
        Minimum number of unique historical seasons required before a season is
        eligible as a test season when inferring ``test_seasons``.
    random_state:
        Random seed passed into each model training call.
    save_outputs:
        Whether to persist CSV outputs under ``outputs/reports`` and
        ``outputs/predictions``.

    Returns
    -------
    dict
        Dictionary with keys:
        ``metrics_by_season`` (per-model, per-season metrics),
        ``model_summary`` (aggregated averages by model),
        and ``predictions`` (row-level held-out predictions).
    """
    if "season" not in modeling_df.columns:
        raise KeyError("modeling_df must include 'season'.")

    if test_seasons is None:
        test_seasons = _infer_test_seasons(modeling_df, min_train_seasons=min_train_seasons)

    if not test_seasons:
        raise ValueError("No valid test seasons available for backtest.")

    metrics_rows = []
    prediction_frames = []

    for season in sorted(test_seasons):
        for model_name in model_names:
            result = train_single_model(
                modeling_df=modeling_df,
                model_name=model_name,
                test_season=season,
                random_state=random_state,
            )
            metrics_rows.append(result["metrics"])
            prediction_frames.append(result["predictions_df"])

    metrics_by_season_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    summary = (
        metrics_by_season_df.groupby("model_name", as_index=False)
        .agg(
            n_test_seasons=("test_season", "nunique"),
            avg_log_loss=("log_loss", "mean"),
            avg_brier_score=("brier_score", "mean"),
            avg_auc=("auc", "mean"),
            avg_accuracy=("accuracy", "mean"),
            total_train_rows=("n_train", "sum"),
            total_test_rows=("n_test", "sum"),
        )
        .sort_values("avg_log_loss")
        .reset_index(drop=True)
    )

    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        metrics_by_season_df.to_csv(
            REPORTS_DIR / "model_metrics_by_season.csv",
            index=False,
        )
        summary.to_csv(REPORTS_DIR / "model_summary.csv", index=False)
        predictions_df.to_csv(
            PREDICTIONS_DIR / "model_predictions_by_season.csv",
            index=False,
        )

    return {
        "metrics_by_season": metrics_by_season_df,
        "model_summary": summary,
        "predictions": predictions_df,
    }
