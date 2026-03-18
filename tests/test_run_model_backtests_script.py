"""Tests for the model backtest runner script output formatting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_model_backtests.py"
    spec = importlib.util.spec_from_file_location("run_model_backtests_script", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Unable to load run_model_backtests.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_prints_readable_summary_and_top_models(monkeypatch, capsys) -> None:
    module = _load_script_module()

    monkeypatch.setattr(
        module,
        "load_all_processed_data",
        lambda: {
            "team_profiles": pd.DataFrame(),
            "games_boxscores": pd.DataFrame(),
            "tourney_matchups": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        module,
        "build_modeling_dataframe",
        lambda **_: pd.DataFrame({"season": [2022, 2023], "target": [1, 0]}),
    )
    monkeypatch.setattr(module, "get_available_test_seasons", lambda *_args, **_kwargs: [2023])

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "test_season": 2023,
                "n_train": 10,
                "n_test": 2,
                "accuracy": 0.5,
                "log_loss": 0.69,
                "brier_score": 0.25,
                "roc_auc": 0.50,
            },
            {
                "model_name": "xgboost",
                "test_season": 2023,
                "n_train": 10,
                "n_test": 2,
                "accuracy": 1.0,
                "log_loss": 0.20,
                "brier_score": 0.08,
                "roc_auc": 1.00,
            },
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "model_name": "xgboost",
                "mean_accuracy": 1.0,
                "std_accuracy": 0.0,
                "mean_log_loss": 0.20,
                "std_log_loss": 0.0,
                "mean_brier_score": 0.08,
                "std_brier_score": 0.0,
                "mean_roc_auc": 1.00,
                "std_roc_auc": 0.0,
            },
            {
                "model_name": "logistic_regression",
                "mean_accuracy": 0.5,
                "std_accuracy": 0.0,
                "mean_log_loss": 0.69,
                "std_log_loss": 0.0,
                "mean_brier_score": 0.25,
                "std_brier_score": 0.0,
                "mean_roc_auc": 0.50,
                "std_roc_auc": 0.0,
            },
        ]
    )

    def _fake_run_backtest(*, test_seasons, **kwargs):
        assert test_seasons == [2023]
        assert kwargs["save_outputs"] is True
        return {
            "predictions": pd.DataFrame(),
            "metrics": metrics_df,
            "summary": summary_df,
        }

    monkeypatch.setattr(module, "run_backtest", _fake_run_backtest)

    module.main()

    out = capsys.readouterr().out
    assert "Modeling dataframe shape:" in out
    assert "Seasons being evaluated: [2023]" in out
    assert "Metrics by model/season:" in out
    assert "Summary metrics across seasons:" in out
    assert "Lowest mean log_loss: xgboost (0.2000)" in out
    assert "Lowest mean brier_score: xgboost (0.0800)" in out
    assert "Highest mean accuracy: xgboost (1.0000)" in out
    assert "Highest mean roc_auc: xgboost (1.0000)" in out
