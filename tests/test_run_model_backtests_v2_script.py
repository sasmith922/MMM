"""Focused tests for the v2 backtest runner script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_model_backtests_v2.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_model_backtests_v2_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Unable to load run_model_backtests_v2.py for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_v2_inputs_normalizes_columns(tmp_path) -> None:
    module = _load_script_module()

    team_features_path = tmp_path / "team_features_v2.csv"
    matchups_path = tmp_path / "tournament_matchups_v2.csv"

    pd.DataFrame(
        {
            "season": [2020, 2020],
            "kaggle_team_id": [1001, 1002],
            "seed": [1, 16],
        }
    ).to_csv(team_features_path, index=False)

    pd.DataFrame(
        {
            "season": [2020],
            "team_a_id": [1001],
            "team_b_id": [1002],
            "label": [1],
        }
    ).to_csv(matchups_path, index=False)

    team_features, matchups = module.load_v2_inputs(team_features_path, matchups_path)

    assert "team_id" in team_features.columns
    assert {"teamA_id", "teamB_id", "target"}.issubset(set(matchups.columns))
    assert team_features["team_id"].dtype.kind in {"i", "u"}
    assert set(matchups["target"].unique()) == {1}


def test_run_v2_backtests_writes_v2_outputs(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    fake_modeling_df = pd.DataFrame(
        {
            "season": [2019, 2020, 2021],
            "teamA_id": [1, 3, 5],
            "teamB_id": [2, 4, 6],
            "target": [1, 0, 1],
            "seed_diff": [1.0, -1.0, 0.0],
        }
    )

    monkeypatch.setattr(module, "load_v2_inputs", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(module, "build_modeling_dataframe", lambda **_kwargs: fake_modeling_df)
    monkeypatch.setattr(module, "get_available_test_seasons", lambda *_args, **_kwargs: [2021])

    def _fake_train_and_score_one_model(**kwargs):
        season = kwargs["test_season"]
        model_name = kwargs["model_name"]
        return {
            "metrics": {
                "model_name": model_name,
                "test_season": season,
                "n_train": 2,
                "n_test": 1,
                "n_samples": 1,
                "accuracy": 1.0,
                "log_loss": 0.1,
                "brier_score": 0.01,
                "roc_auc": 1.0,
            },
            "predictions": pd.DataFrame(
                {
                    "season": [season],
                    "teamA_id": [5],
                    "teamB_id": [6],
                    "target": [1],
                    "pred_prob": [0.9],
                    "pred_class": [1],
                    "model_name": [model_name],
                    "test_season": [season],
                }
            ),
        }

    monkeypatch.setattr(module, "_train_and_score_one_model", _fake_train_and_score_one_model)

    predictions_dir = tmp_path / "outputs" / "predictions_v2"
    reports_dir = tmp_path / "outputs" / "reports_v2"

    results = module.run_v2_backtests(
        team_features_path=tmp_path / "unused_team_features.csv",
        matchups_path=tmp_path / "unused_matchups.csv",
        predictions_dir=predictions_dir,
        reports_dir=reports_dir,
        models_dir=tmp_path / "models" / "v2",
        model_names=["logistic_regression", "xgboost"],
        min_train_seasons=1,
        random_state=42,
        save_model_artifacts=False,
    )

    assert (predictions_dir / "model_predictions_by_season_v2.csv").exists()
    assert (reports_dir / "model_metrics_by_season_v2.csv").exists()
    assert (reports_dir / "model_summary_v2.csv").exists()
    assert set(results.keys()) == {"predictions", "metrics", "summary"}
    assert sorted(results["metrics"]["model_name"].tolist()) == ["logistic_regression", "xgboost"]
