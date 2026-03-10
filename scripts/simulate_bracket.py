"""
simulate_bracket.py
-------------------
Run Monte Carlo bracket simulations and output team-level odds.

Usage
-----
    python scripts/simulate_bracket.py [--season 2025] [--n-sims 10000]

The script reads the tournament field (team IDs in bracket order) from
data/processed/bracket_field_<season>.csv if it exists, otherwise it
falls back to using all teams for which predictions were generated.

Outputs
-------
- outputs/predictions/simulation_results.csv
- outputs/figures/champion_odds.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from madness_model import baseline_model, xgb_model
from madness_model.baseline_model import get_feature_cols
from madness_model.config import NUM_SIMULATIONS, PREDICTION_SEASON
from madness_model.paths import FIGURES_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from madness_model.simulate_bracket import build_predict_fn, monte_carlo_simulation
from madness_model.visualize import plot_team_champion_odds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the March Madness bracket.")
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON)
    parser.add_argument("--n-sims", type=int, default=NUM_SIMULATIONS)
    parser.add_argument(
        "--model",
        choices=["baseline", "xgb"],
        default="xgb",
    )
    return parser.parse_args()


def load_bracket_field(season: int) -> list[int] | None:
    """Attempt to load the ordered 64-team bracket field for a given season.

    Parameters
    ----------
    season:
        Target season year.

    Returns
    -------
    list[int] or None
        Ordered team ID list if the file exists, else ``None``.
    """
    field_path = PROCESSED_DATA_DIR / f"bracket_field_{season}.csv"
    if not field_path.exists():
        return None
    df = pd.read_csv(field_path)
    return df["team_id"].tolist()


def main() -> None:
    args = parse_args()
    season = args.season

    # Load features
    features_path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Team features not found at {features_path}. "
            "Run scripts/build_features.py first."
        )

    features = pd.read_parquet(features_path).set_index(["season", "team_id"])

    # Determine bracket field
    field = load_bracket_field(season)
    if field is None:
        season_features = features.xs(season, level="season") if season in features.index.get_level_values("season") else None
        if season_features is None:
            raise ValueError(f"No features for season {season}.")
        field = season_features.index.tolist()
        log.warning(
            "No bracket_field_%d.csv found; using all %d teams with features.",
            season,
            len(field),
        )

    log.info("Bracket field: %d teams", len(field))

    # Load model
    if args.model == "baseline":
        model = baseline_model.load_model()
        matchup_cols_df = pd.read_parquet(PROCESSED_DATA_DIR / "matchups.parquet")
        feature_cols = get_feature_cols(matchup_cols_df)
    else:
        model = xgb_model.load_model()
        matchup_cols_df = pd.read_parquet(PROCESSED_DATA_DIR / "matchups.parquet")
        feature_cols = get_feature_cols(matchup_cols_df)

    predict_fn = build_predict_fn(model, season, features, feature_cols)

    # Run simulations
    log.info("Running %d simulations…", args.n_sims)
    results = monte_carlo_simulation(field, predict_fn, n_simulations=args.n_sims)
    log.info("Top 5 championship odds:\n%s", results.head(5).to_string(index=False))

    # Save results
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / "simulation_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved simulation results to %s", out_path)

    # Plot
    fig = plot_team_champion_odds(
        results,
        save_path=FIGURES_DIR / "champion_odds.png",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
