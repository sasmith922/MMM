"""
predict_bracket.py
------------------
Generate win-probability predictions for every possible matchup in the
prediction season bracket and write them to outputs/predictions/.

Usage
-----
    python scripts/predict_bracket.py [--season 2025] [--model {baseline,xgb}]

Outputs
-------
- outputs/predictions/bracket_predictions.csv
  Columns: season, team_a_id, team_b_id, win_prob_a
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations

import pandas as pd

from madness_model import baseline_model, xgb_model
from madness_model.baseline_model import get_feature_cols
from madness_model.build_matchups import build_prediction_matchups
from madness_model.config import PREDICTION_SEASON
from madness_model.paths import PREDICTIONS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict bracket matchup probabilities.")
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON)
    parser.add_argument(
        "--model",
        choices=["baseline", "xgb"],
        default="xgb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    season = args.season

    # Load features for prediction season
    features_path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Team features not found at {features_path}. "
            "Run scripts/build_features.py first."
        )

    features = pd.read_parquet(features_path).set_index(["season", "team_id"])
    season_features = features.loc[season] if season in features.index.get_level_values("season") else None
    if season_features is None or len(season_features) == 0:
        raise ValueError(f"No features found for season {season}.")

    team_ids = season_features.index.tolist()
    log.info("Teams in season %d: %d", season, len(team_ids))

    # Build all pairwise prediction matchups
    matchup_pairs = list(combinations(team_ids, 2))
    log.info("Matchup pairs to predict: %d", len(matchup_pairs))

    matchup_df = build_prediction_matchups(season, matchup_pairs, features)
    feature_cols = get_feature_cols(matchup_df)

    # Load model and predict
    if args.model == "baseline":
        model = baseline_model.load_model()
        y_prob = baseline_model.predict_proba(model, matchup_df, feature_cols)
    else:
        model = xgb_model.load_model()
        y_prob = xgb_model.predict_proba(model, matchup_df, feature_cols)

    matchup_df["win_prob_a"] = y_prob

    # Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / "bracket_predictions.csv"
    matchup_df[["season", "team_a_id", "team_b_id", "win_prob_a"]].to_csv(out_path, index=False)
    log.info("Saved %d predictions to %s", len(matchup_df), out_path)


if __name__ == "__main__":
    main()
