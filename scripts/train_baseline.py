"""
train_baseline.py
-----------------
Train and save the logistic regression baseline model.

Usage
-----
    python scripts/train_baseline.py [--val-seasons 2022 2023]

Pipeline
--------
1. Load the matchup dataset from data/processed/.
2. Split into train / validation by season.
3. Train the logistic regression pipeline.
4. Evaluate on the validation set and print a report.
5. Save the fitted model to models/.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from madness_model import baseline_model, evaluate
from madness_model.config import VALIDATION_SEASONS
from madness_model.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the logistic regression baseline.")
    parser.add_argument(
        "--val-seasons",
        nargs="+",
        type=int,
        default=VALIDATION_SEASONS,
        help="Seasons to use for validation (default: config.VALIDATION_SEASONS).",
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
    log.info("Loaded matchups: %d rows", len(matchups))

    val_mask = matchups["season"].isin(args.val_seasons)
    train_df = matchups[~val_mask].reset_index(drop=True)
    val_df = matchups[val_mask].reset_index(drop=True)

    log.info("Train rows: %d | Val rows: %d", len(train_df), len(val_df))

    feature_cols = baseline_model.get_feature_cols(train_df)
    pipeline = baseline_model.train(train_df, feature_cols)

    # Evaluate
    y_prob = baseline_model.predict_proba(pipeline, val_df, feature_cols)
    report = evaluate.evaluate(val_df["label"].values, y_prob)
    evaluate.print_report(report)

    # Save
    baseline_model.save_model(pipeline)
    log.info("Model saved.")

    # TODO: log metrics to MLflow or W&B


if __name__ == "__main__":
    main()
