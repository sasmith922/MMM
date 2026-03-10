"""
evaluate.py
-----------
Evaluate a trained model on held-out tournament seasons and write a report.

Usage
-----
    python scripts/evaluate.py [--model {baseline,xgb}] [--val-seasons 2022 2023]

Outputs
-------
- Console report (accuracy, log loss, Brier score, AUC-ROC)
- outputs/reports/evaluation_report.json
- outputs/figures/calibration_curve.png
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from madness_model import baseline_model, evaluate, xgb_model
from madness_model.baseline_model import get_feature_cols
from madness_model.config import VALIDATION_SEASONS
from madness_model.paths import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from madness_model.visualize import plot_calibration_curve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model",
        choices=["baseline", "xgb"],
        default="xgb",
        help="Which saved model to evaluate.",
    )
    parser.add_argument(
        "--val-seasons",
        nargs="+",
        type=int,
        default=VALIDATION_SEASONS,
        help="Seasons to evaluate on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matchups_path = PROCESSED_DATA_DIR / "matchups.parquet"
    if not matchups_path.exists():
        raise FileNotFoundError(
            f"Matchups not found at {matchups_path}. "
            "Run scripts/build_matchups.py first."
        )

    matchups = pd.read_parquet(matchups_path)
    val_df = matchups[matchups["season"].isin(args.val_seasons)].reset_index(drop=True)
    feature_cols = get_feature_cols(val_df)

    if args.model == "baseline":
        model = baseline_model.load_model()
        y_prob = baseline_model.predict_proba(model, val_df, feature_cols)
    else:
        model = xgb_model.load_model()
        y_prob = xgb_model.predict_proba(model, val_df, feature_cols)

    report = evaluate.evaluate(val_df["label"].values, y_prob)
    evaluate.print_report(report)

    # Save JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    log.info("Report saved to %s", report_path)

    # Save calibration curve
    fig = plot_calibration_curve(
        val_df["label"].values,
        y_prob,
        model_name=args.model,
        save_path=FIGURES_DIR / "calibration_curve.png",
    )
    close_figure(fig)

    # TODO: per-season breakdown of metrics


def close_figure(fig) -> None:
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
