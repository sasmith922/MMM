"""Run end-to-end season-heldout model backtests from processed CSV inputs."""

from __future__ import annotations

from madness_model.backtest_runner import run_backtest
from madness_model.build_model_dataset import build_modeling_dataframe
from madness_model.load_processed_data import load_all_processed_data


def main() -> None:
    """Load data, build matchup modeling table, and run model backtests."""

    data = load_all_processed_data()

    model_df = build_modeling_dataframe(
        team_profiles_df=data["team_profiles"],
        tourney_matchups_df=data["tourney_matchups"],
        games_boxscores_df=data["games_boxscores"],
    )

    results = run_backtest(
        modeling_df=model_df,
        model_names=[
            "logistic_regression",
            "random_forest",
            "xgboost",
            "neural_net",
        ],
        test_seasons=None,
        random_state=42,
        save_outputs=True,
    )

    print(results["summary"])


if __name__ == "__main__":
    main()
