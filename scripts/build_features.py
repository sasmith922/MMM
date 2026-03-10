"""
build_features.py
-----------------
Build season-end team features from raw data and write to data/processed/.

Usage
-----
    python scripts/build_features.py [--seasons 2003 2004 ...]

Pipeline
--------
1. Load raw regular-season game results and team metadata.
2. Clean the raw DataFrames.
3. Compute team features (win %, avg point differential, etc.).
4. (TODO) Compute Elo ratings and merge in.
5. Write the feature table to data/processed/team_features.parquet.
"""

from __future__ import annotations

import argparse
import logging

from madness_model.clean_data import clean_game_results, clean_teams, filter_seasons
from madness_model.config import TRAINING_SEASONS, VALIDATION_SEASONS
from madness_model.build_team_features import build_team_features
from madness_model.load_data import load_regular_season, load_teams
from madness_model.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    all_seasons = sorted(set(TRAINING_SEASONS + VALIDATION_SEASONS))
    parser = argparse.ArgumentParser(description="Build season-end team features.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=all_seasons,
        help="Seasons to include (default: all training + validation seasons).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Building features for seasons: %s", args.seasons)

    # Load raw data
    teams_raw = load_teams()
    games_raw = load_regular_season()

    # Clean
    teams = clean_teams(teams_raw)
    games = clean_game_results(games_raw)
    games = filter_seasons(games, args.seasons)

    log.info("Games loaded: %d rows across %d seasons.", len(games), games["season"].nunique())

    # Build features
    features = build_team_features(games)
    log.info("Feature table shape: %s", features.shape)

    # Write output
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "team_features.parquet"
    features.reset_index().to_parquet(out_path, index=False)
    log.info("Saved team features to %s", out_path)

    # TODO: compute and merge Elo ratings


if __name__ == "__main__":
    main()
