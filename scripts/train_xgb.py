"""
train_xgb.py
------------
Train and save the XGBoost model.

Usage
-----
    python scripts/train_xgb.py [--val-seasons 2022 2023]

Pipeline
--------
1. Load the matchup dataset from data/processed/.
2. Split into train / validation by season.
3. Train the XGBoost classifier.
4. Evaluate on the validation set.
5. Save the fitted model to models/.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from madness_model import evaluate, xgb_model
from madness_model.baseline_model import get_feature_cols
from madness_model.config import VALIDATION_SEASONS
from madness_model.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost model.")
    parser.add_argument(
        "--val-seasons",
        nargs="+",
        type=int,
        default=VALIDATION_SEASONS,
        help="Seasons to hold out for validation.",
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

    feature_cols = get_feature_cols(train_df)
    model = xgb_model.train(train_df, feature_cols, eval_df=val_df)

    # Evaluate
    y_prob = xgb_model.predict_proba(model, val_df, feature_cols)
    report = evaluate.evaluate(val_df["label"].values, y_prob)
    evaluate.print_report(report)

    # Feature importance
    importance = xgb_model.get_feature_importance(model, feature_cols)
    log.info("Top features:\n%s", importance.head(10).to_string())

    # Save
    xgb_model.save_model(model)
    log.info("Model saved.")

    # TODO: calibrate model after training (see scripts/calibrate.py)
    # TODO: log experiment to MLflow / W&B


if __name__ == "__main__":
    main()
