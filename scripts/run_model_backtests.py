"""Run end-to-end season-heldout model backtests from processed CSV inputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from madness_model.backtest_runner import run_backtest
from madness_model.build_model_dataset import build_modeling_dataframe
from madness_model.load_processed_data import load_all_processed_data
from madness_model.paths import MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR
from madness_model.train_models import get_available_test_seasons

DEFAULT_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "neural_net",
]
TEST_SEASONS_OVERRIDE: list[int] | None = None
MIN_TRAIN_SEASONS = 5
RANDOM_STATE = 42


def _get_best_model_by_metric(
    summary_df,
    metric: str,
    *,
    ascending: bool,
) -> tuple[str, float] | None:
    """Return (model_name, metric_value) for the best model on one summary metric.

    Expects ``summary_df`` to include ``model_name`` and the requested ``metric``
    column. Returns ``None`` when the metric column is unavailable or has only NaN.
    """
    if metric not in summary_df.columns:
        return None
    ranking = summary_df[["model_name", metric]].dropna()
    if ranking.empty:
        return None
    ordered = ranking.sort_values(metric, ascending=ascending)
    best = ordered.iloc[0]
    return str(best["model_name"]), float(best[metric])


def main() -> None:
    """Load data, build matchup modeling table, and run model backtests."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = load_all_processed_data()

    model_df = build_modeling_dataframe(
        team_profiles_df=data["team_profiles"],
        tourney_matchups_df=data["tourney_matchups"],
        games_boxscores_df=data["games_boxscores"],
    )

    test_seasons = (
        sorted(TEST_SEASONS_OVERRIDE)
        if TEST_SEASONS_OVERRIDE is not None
        else get_available_test_seasons(model_df, min_train_seasons=MIN_TRAIN_SEASONS)
    )
    if not test_seasons:
        raise ValueError("No valid test seasons found for model backtests.")

    print(f"Modeling dataframe shape: {model_df.shape}")
    print(f"Seasons being evaluated: {test_seasons}")

    results = run_backtest(
        modeling_df=model_df,
        model_names=DEFAULT_MODELS,
        test_seasons=test_seasons,
        min_train_seasons=MIN_TRAIN_SEASONS,
        random_state=RANDOM_STATE,
        save_outputs=True,
    )

    metrics_df = results["metrics"]
    summary_df = results["summary"]

    print("\nMetrics by model/season:")
    print(
        metrics_df[
            ["model_name", "test_season", "n_train", "n_test", "accuracy", "log_loss", "brier_score", "roc_auc"]
        ].round(4).to_string(index=False)
    )

    print("\nSummary metrics across seasons:")
    print(summary_df.round(4).to_string(index=False))

    best_log_loss = _get_best_model_by_metric(summary_df, "mean_log_loss", ascending=True)
    best_brier = _get_best_model_by_metric(summary_df, "mean_brier_score", ascending=True)
    best_accuracy = _get_best_model_by_metric(summary_df, "mean_accuracy", ascending=False)
    best_roc_auc = _get_best_model_by_metric(summary_df, "mean_roc_auc", ascending=False)

    print("\nTop-performing models:")
    if best_log_loss:
        print(f"  Lowest mean log_loss: {best_log_loss[0]} ({best_log_loss[1]:.4f})")
    if best_brier:
        print(f"  Lowest mean brier_score: {best_brier[0]} ({best_brier[1]:.4f})")
    if best_accuracy:
        print(f"  Highest mean accuracy: {best_accuracy[0]} ({best_accuracy[1]:.4f})")
    if best_roc_auc:
        print(f"  Highest mean roc_auc: {best_roc_auc[0]} ({best_roc_auc[1]:.4f})")

    print("\nSaved outputs:")
    print(f"  Predictions: {PREDICTIONS_DIR / 'model_predictions_by_season.csv'}")
    print(f"  Metrics: {REPORTS_DIR / 'model_metrics_by_season.csv'}")
    print(f"  Summary: {REPORTS_DIR / 'model_summary.csv'}")
    print(f"  Model artifacts directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
