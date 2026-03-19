"""Focused tests for bracket-focused evaluation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_brackets.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_brackets_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Unable to load evaluate_brackets.py for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_inputs_round_and_season_normalization(tmp_path) -> None:
    module = _load_script_module()

    predictions_path = tmp_path / "preds.csv"
    actuals_path = tmp_path / "actuals.csv"

    pd.DataFrame(
        {
            "test_season": [2025],
            "team_a_id": [1],
            "team_b_id": [2],
            "pred_prob": [0.65],
            "model_name": ["logistic_regression"],
        }
    ).to_csv(predictions_path, index=False)

    pd.DataFrame(
        {
            "season": [2025],
            "team_a_id": [1],
            "team_b_id": [2],
            "round": [1],
            "target": [1],
        }
    ).to_csv(actuals_path, index=False)

    predictions_df, actuals_df = module.load_inputs(predictions_path, actuals_path)

    assert list(predictions_df["season"]) == [2025]
    assert list(actuals_df["round_name"]) == ["R64"]
    assert list(actuals_df["actual_winner"]) == [1]


def test_evaluate_outputs_expected_bracket_reports(tmp_path) -> None:
    module = _load_script_module()

    predictions_path = tmp_path / "preds.csv"
    actuals_path = tmp_path / "actuals.csv"
    output_dir = tmp_path / "outputs" / "bracket_reports"

    actual_games = [
        {"season": 2025, "team_a_id": 1, "team_b_id": 4, "round": "R64", "target": 1},
        {"season": 2025, "team_a_id": 2, "team_b_id": 3, "round": "R64", "target": 1},
        {"season": 2025, "team_a_id": 1, "team_b_id": 2, "round": "CHAMP", "target": 1},
    ]
    pd.DataFrame(actual_games).to_csv(actuals_path, index=False)

    pred_games = [
        {"season": 2025, "teamA_id": 1, "teamB_id": 4, "pred_prob": 0.8, "model_name": "logistic_regression"},
        {"season": 2025, "teamA_id": 2, "teamB_id": 3, "pred_prob": 0.7, "model_name": "logistic_regression"},
        {"season": 2025, "teamA_id": 1, "teamB_id": 2, "pred_prob": 0.75, "model_name": "logistic_regression"},
    ]
    pd.DataFrame(pred_games).to_csv(predictions_path, index=False)

    results = module.evaluate(
        predictions_path=predictions_path,
        actuals_path=actuals_path,
        output_dir=output_dir,
        n_simulations=20,
        random_state=123,
    )

    assert set(results.keys()) == {
        "bracket_metrics_by_season",
        "bracket_summary_by_model",
        "champion_probs_by_season",
        "round_reach_probs_by_season",
        "bracket_simulation_summary",
    }
    assert (output_dir / "bracket_metrics_by_season.csv").exists()
    assert (output_dir / "bracket_summary_by_model.csv").exists()
    assert (output_dir / "champion_probs_by_season.csv").exists()
    assert (output_dir / "round_reach_probs_by_season.csv").exists()
    assert (output_dir / "bracket_simulation_summary.csv").exists()

    metrics_df = results["bracket_metrics_by_season"]
    assert len(metrics_df) == 1
    assert metrics_df.loc[0, "model_name"] == "logistic_regression"
    assert metrics_df.loc[0, "champion_correct"] in {0, 1}
