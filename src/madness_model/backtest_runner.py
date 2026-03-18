"""Run season-heldout backtests across one or more model types."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from madness_model.evaluate_models import summarize_metrics
from madness_model.paths import PREDICTIONS_DIR, REPORTS_DIR
from madness_model.train_models import get_available_test_seasons, train_single_model_for_season

LOGGER = logging.getLogger(__name__)


def run_backtest(
    modeling_df: pd.DataFrame,
    model_names: list[str],
    test_seasons: list[int] | None = None,
    min_train_seasons: int = 5,
    random_state: int = 42,
    save_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run rolling season backtests and return predictions/metrics/summary.

    TODO: Add side-by-side feature-importance exports for tree models.
    TODO: Add bracket integration endpoint after matchup model selection.
    """

    if test_seasons is None:
        test_seasons = get_available_test_seasons(
            modeling_df,
            min_train_seasons=min_train_seasons,
        )
    else:
        test_seasons = sorted(set(int(season) for season in test_seasons))

    if not test_seasons:
        raise ValueError("No valid test seasons available for backtesting.")

    LOGGER.info(
        "Starting backtest with models=%s over test_seasons=%s (rows=%s).",
        model_names,
        test_seasons,
        len(modeling_df),
    )

    predictions_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for season in sorted(test_seasons):
        for model_name in model_names:
            LOGGER.info("Running model=%s on held-out season=%s", model_name, season)
            result = train_single_model_for_season(
                modeling_df=modeling_df,
                model_name=model_name,
                test_season=season,
                random_state=random_state,
                strict_features=False,
                save_model_artifact=save_outputs,
            )
            predictions_frames.append(result["predictions"])
            metrics_rows.append(result["metrics"])

    predictions_df = pd.concat(predictions_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = summarize_metrics(metrics_df)

    if save_outputs:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        predictions_path = Path(PREDICTIONS_DIR) / "model_predictions_by_season.csv"
        metrics_path = Path(REPORTS_DIR) / "model_metrics_by_season.csv"
        summary_path = Path(REPORTS_DIR) / "model_summary.csv"

        predictions_df.to_csv(predictions_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info(
            "Saved backtest outputs: predictions=%s metrics=%s summary=%s",
            predictions_path,
            metrics_path,
            summary_path,
        )

    return {
        "predictions": predictions_df,
        "metrics": metrics_df,
        "summary": summary_df,
    }
