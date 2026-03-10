"""
build_matchups.py
-----------------
Build the labelled matchup dataset from team features and historical
tournament results, then write it to data/processed/.

Usage
-----
    python scripts/build_matchups.py

Pipeline
--------
1. Load cleaned team features from data/processed/.
2. Load and clean tournament results.
3. Build matchup rows (feature differentials + labels).
4. Write to data/processed/matchups.parquet.
"""

from __future__ import annotations

import logging

import pandas as pd

from madness_model.build_matchups import build_matchups_from_results
from madness_model.clean_data import clean_game_results
from madness_model.config import TRAINING_SEASONS, VALIDATION_SEASONS
from madness_model.load_data import load_tourney_results
from madness_model.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    # Load pre-built team features
    features_path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Team features not found at {features_path}. "
            "Run scripts/build_features.py first."
        )

    features = pd.read_parquet(features_path).set_index(["season", "team_id"])
    log.info("Loaded team features: %s", features.shape)

    # Load and clean tournament results
    tourney_raw = load_tourney_results()
    tourney = clean_game_results(tourney_raw)
    all_seasons = sorted(set(TRAINING_SEASONS + VALIDATION_SEASONS))
    tourney = tourney[tourney["season"].isin(all_seasons)]
    log.info("Tournament games: %d rows", len(tourney))

    # Build matchups
    matchups = build_matchups_from_results(tourney, features)
    log.info("Matchup rows: %d", len(matchups))

    # Save
    out_path = PROCESSED_DATA_DIR / "matchups.parquet"
    matchups.to_parquet(out_path, index=False)
    log.info("Saved matchups to %s", out_path)

    # TODO: also save a train/validation split


if __name__ == "__main__":
    main()
