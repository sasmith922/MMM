"""Focused tests for 2026 reduced data/model pipeline scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT_PATH = ROOT / "scripts" / "build_2026_dataset.py"
TRAIN_SCRIPT_PATH = ROOT / "scripts" / "train_predict_2026_reduced.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_2026_reduced_features() -> pd.DataFrame:
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


def test_build_2026_dataset_writes_cleaned_features_and_matchups(tmp_path: Path) -> None:
    module = _load_module(BUILD_SCRIPT_PATH, "build_2026_dataset_script_test")

    features_path = tmp_path / "team_features_2026_reduced.csv"
    matchups_path = tmp_path / "tourney_matchups_2026.csv"
    _sample_2026_reduced_features().to_csv(features_path, index=False)

    cleaned_df, matchups_df = module.build_2026_reduced_dataset(
        features_path=features_path,
        matchups_path=matchups_path,
    )

    assert features_path.exists()
    assert matchups_path.exists()
    assert len(cleaned_df) == 4
    assert {"teamA_name_norm", "teamB_name_norm", "teamA_seed_num", "teamB_seed_num"}.issubset(
        set(matchups_df.columns)
    )
    assert len(matchups_df) == 2  # W01-W16 and X08-X09


def test_build_2026_dataset_raises_on_duplicate_team_name_norm(tmp_path: Path) -> None:
    module = _load_module(BUILD_SCRIPT_PATH, "build_2026_dataset_script_test_dupes")

    features_path = tmp_path / "team_features_2026_reduced.csv"
    dupes = _sample_2026_reduced_features()
    dupes.loc[1, "team_name_norm"] = "alpha"
    dupes.to_csv(features_path, index=False)

    try:
        module.build_2026_reduced_dataset(
            features_path=features_path,
            matchups_path=tmp_path / "tourney_matchups_2026.csv",
        )
    except ValueError as exc:
        assert "Duplicate (season, team_name_norm)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for duplicate team_name_norm.")


def test_train_predict_2026_reduced_writes_predictions(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(TRAIN_SCRIPT_PATH, "train_predict_2026_reduced_script_test")

    historical_path = tmp_path / "historical_matchups_v2.csv"
    team_features_2026_path = tmp_path / "team_features_2026_reduced.csv"
    matchups_2026_path = tmp_path / "tourney_matchups_2026.csv"
    model_path = tmp_path / "models" / "xgboost_2026_reduced.joblib"
    predictions_path = tmp_path / "outputs" / "predictions" / "bracket_predictions_2026_reduced.csv"

    pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2025],
            "label": [1, 0, 1, 0],
            "seed_diff": [-15, 2, -4, 3],
            "adj_o_diff": [14.0, -2.0, 7.0, -1.0],
            "adj_d_diff": [-8.0, 2.0, -3.0, 1.0],
            "net_rating_diff": [22.0, -4.0, 10.0, -2.0],
        }
    ).to_csv(historical_path, index=False)

    _sample_2026_reduced_features().to_csv(team_features_2026_path, index=False)
    pd.DataFrame(
        {
            "season": [2026, 2026],
            "round": ["R64", "R64"],
            "region": ["W", "X"],
            "teamA_name_norm": ["alpha", "gamma"],
            "teamB_name_norm": ["beta", "delta"],
        }
    ).to_csv(matchups_2026_path, index=False)

    # Keep test fast: use logistic regression estimator in place of xgboost.
    from madness_model.model_utils import build_model as _build_model

    captured_model_name: list[str] = []

    def _fake_build_model(model_name: str, random_state: int = 42):
        captured_model_name.append(model_name)
        return _build_model("logistic_regression", random_state=random_state)

    monkeypatch.setattr(module, "build_model", _fake_build_model)

    out_model_path, out_predictions_path = module.train_and_predict_2026_reduced(
        historical_matchups_path=historical_path,
        team_features_2026_path=team_features_2026_path,
        tourney_matchups_2026_path=matchups_2026_path,
        model_output_path=model_path,
        predictions_output_path=predictions_path,
    )

    assert out_model_path.exists()
    assert out_predictions_path.exists()

    preds = pd.read_csv(out_predictions_path)
    assert len(preds) == 2
    assert "win_prob_a" in preds.columns
    assert preds["win_prob_a"].between(0.0, 1.0).all()
    assert captured_model_name == ["xgboost"]
