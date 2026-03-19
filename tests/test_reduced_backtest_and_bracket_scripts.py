"""Focused tests for reduced backtest and 2026 bracket prediction scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKTEST_SCRIPT_PATH = ROOT / "scripts" / "run_backtest_reduced.py"
BRACKET_SCRIPT_PATH = ROOT / "scripts" / "predict_bracket_2026_reduced.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sample_team_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "team_name": ["Alpha", "Beta", "Gamma", "Delta"],
            "team_name_norm": ["alpha", "beta", "gamma", "delta"],
            "seed": ["W01", "W16", "X08", "X09"],
            "win_pct": [0.82, 0.56, 0.68, 0.66],
            "pre_tourney_adjoe": [118.0, 104.0, 112.0, 111.0],
            "pre_tourney_adjde": [95.0, 107.0, 101.0, 102.0],
            "pre_tourney_adjem": [23.0, -3.0, 11.0, 9.0],
        }
    )


def _sample_historical_matchups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2025, 2025],
            "label": [1, 0, 1, 0, 1],
            "seed_diff": [-15, 2, -4, 3, -2],
            "adj_o_diff": [14.0, -2.0, 7.0, -1.0, 2.0],
            "adj_d_diff": [-8.0, 2.0, -3.0, 1.0, -1.0],
            "net_rating_diff": [22.0, -4.0, 10.0, -2.0, 3.0],
            "win_pct_diff": [0.25, -0.08, 0.1, -0.05, 0.07],
            "round": ["R64", "R64", "R64", "R64", "R64"],
            "region": ["W", "X", "Y", "Z", "W"],
            "teamA_name": ["A", "B", "C", "D", "E"],
            "teamB_name": ["F", "G", "H", "I", "J"],
        }
    )


def test_run_backtest_reduced_writes_outputs(tmp_path, monkeypatch) -> None:
    module = _load_module(BACKTEST_SCRIPT_PATH, "run_backtest_reduced_script_test")

    historical_path = tmp_path / "historical.csv"
    _sample_historical_matchups().to_csv(historical_path, index=False)

    from madness_model.model_utils import build_model as _build_model

    def _fake_build_model(model_name: str, random_state: int = 42):
        assert model_name in {"xgboost", "logistic_regression"}
        return _build_model("logistic_regression", random_state=random_state)

    monkeypatch.setattr(module, "build_model", _fake_build_model)

    preds_path, metrics_path = module.run_backtest_reduced(
        test_year=2025,
        historical_matchups_path=historical_path,
        model_name="xgboost",
        metrics_output_path=tmp_path / "outputs" / "predictions" / "backtest_metrics_reduced.csv",
        feature_list_output_path=tmp_path / "outputs" / "reports" / "features_2026_reduced_used.txt",
    )

    assert preds_path.exists()
    assert metrics_path.exists()

    preds = pd.read_csv(preds_path)
    assert {"pred_prob", "pred_pick", "test_year", "model_name"}.issubset(preds.columns)
    assert preds["pred_prob"].between(0.0, 1.0).all()

    metrics = pd.read_csv(metrics_path)
    assert {"accuracy", "log_loss", "brier_score", "feature_count"}.issubset(metrics.columns)
    assert (metrics["test_year"] == 2025).any()


def test_predict_bracket_2026_reduced_writes_breakdown_and_summary(tmp_path, monkeypatch) -> None:
    module = _load_module(BRACKET_SCRIPT_PATH, "predict_bracket_2026_reduced_script_test")

    team_features_path = tmp_path / "team_features_2026_reduced.csv"
    matchups_path = tmp_path / "tourney_matchups_2026.csv"
    feature_list_path = tmp_path / "features_2026_reduced_used.txt"
    model_path = tmp_path / "xgboost_2026_reduced.joblib"
    breakdown_path = tmp_path / "outputs" / "predictions" / "bracket_breakdown_2026_reduced.csv"
    summary_path = tmp_path / "outputs" / "predictions" / "bracket_summary_2026_reduced.txt"

    _sample_team_features().to_csv(team_features_path, index=False)
    pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": ["R64", "R64"],
            "region": ["West", "East"],
            "teamA_name_norm": ["alpha", "gamma"],
            "teamB_name_norm": ["beta", "delta"],
            "teamA_seed_num": [1, 8],
            "teamB_seed_num": [16, 9],
        }
    ).to_csv(matchups_path, index=False)
    feature_list_path.write_text(
        "seed_diff\nadj_o_diff\nadj_d_diff\nnet_rating_diff\nwin_pct_diff\n",
        encoding="utf-8",
    )

    class _FakeModel:
        def predict_proba(self, X):
            import numpy as np

            n = len(X)
            return np.column_stack([np.full(n, 0.35), np.full(n, 0.65)])

    monkeypatch.setattr(module, "load_model", lambda _path: _FakeModel())
    model_path.write_text("stub", encoding="utf-8")

    out_breakdown, out_summary = module.predict_bracket_2026_reduced(
        team_features_2026_path=team_features_path,
        tourney_matchups_2026_path=matchups_path,
        model_path=model_path,
        feature_list_path=feature_list_path,
        breakdown_output_path=breakdown_path,
        summary_output_path=summary_path,
        train_if_missing=False,
    )

    assert out_breakdown.exists()
    assert out_summary.exists()
    breakdown = pd.read_csv(out_breakdown)
    assert {"round", "region", "team_1", "team_2", "predicted_winner", "predicted_win_probability", "next_round_slot"}.issubset(
        breakdown.columns
    )
    assert (breakdown["round"] == "R64").any()

    summary_text = out_summary.read_text(encoding="utf-8")
    assert "Round of 64 winners:" in summary_text
    assert "Champion:" in summary_text
