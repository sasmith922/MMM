"""Run season-heldout model backtests using v2 processed tournament datasets.

This script intentionally keeps baseline outputs untouched by writing everything
under v2-specific directories.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from madness_model.build_model_dataset import build_modeling_dataframe
from madness_model.evaluate_models import compute_metrics, summarize_metrics
from madness_model.feature_config import TARGET_COL, get_feature_columns
from madness_model.model_utils import build_model, build_train_test_matrices, save_model
from madness_model.train_models import get_available_test_seasons

LOGGER = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "neural_net",
]

DEFAULT_TEAM_FEATURES_PATH = ROOT_DIR / "data" / "processed_v2" / "team_features_v2.csv"
DEFAULT_MATCHUPS_PATH = ROOT_DIR / "data" / "processed_v2" / "tournament_matchups_v2.csv"
DEFAULT_PREDICTIONS_DIR = ROOT_DIR / "outputs" / "predictions_v2"
DEFAULT_REPORTS_DIR = ROOT_DIR / "outputs" / "reports_v2"
DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "v2"


def _normalize_team_features_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"kaggle_team_id": "team_id"}
    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(columns=existing)


def _normalize_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "team_a_id": "teamA_id",
        "team_b_id": "teamB_id",
        "team_a_seed": "teamA_seed",
        "team_b_seed": "teamB_seed",
        "team_a_name": "teamA_name",
        "team_b_name": "teamB_name",
        "round_num_guess": "round",
        "label": TARGET_COL,
        "y": TARGET_COL,
    }
    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(columns=existing)


def _validate_required_columns(df: pd.DataFrame, *, name: str, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns {missing}. Available: {list(df.columns)}")


def load_v2_inputs(team_features_path: Path, matchups_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate v2 modeling inputs."""
    if not team_features_path.exists():
        raise FileNotFoundError(f"Missing team features file: {team_features_path}")
    if not matchups_path.exists():
        raise FileNotFoundError(f"Missing tournament matchups file: {matchups_path}")

    team_features = _normalize_team_features_columns(pd.read_csv(team_features_path))
    matchups = _normalize_matchup_columns(pd.read_csv(matchups_path))

    _validate_required_columns(team_features, name="team_features_v2", required=["season", "team_id"])
    _validate_required_columns(
        matchups,
        name="tournament_matchups_v2",
        required=["season", "teamA_id", "teamB_id", TARGET_COL],
    )

    team_features["season"] = pd.to_numeric(team_features["season"], errors="coerce")
    team_features["team_id"] = pd.to_numeric(team_features["team_id"], errors="coerce")
    matchups["season"] = pd.to_numeric(matchups["season"], errors="coerce")
    matchups["teamA_id"] = pd.to_numeric(matchups["teamA_id"], errors="coerce")
    matchups["teamB_id"] = pd.to_numeric(matchups["teamB_id"], errors="coerce")
    matchups[TARGET_COL] = pd.to_numeric(matchups[TARGET_COL], errors="coerce")

    team_features = team_features.dropna(subset=["season", "team_id"]).copy()
    matchups = matchups.dropna(subset=["season", "teamA_id", "teamB_id", TARGET_COL]).copy()

    team_features["season"] = team_features["season"].astype(int)
    team_features["team_id"] = team_features["team_id"].astype(int)
    matchups["season"] = matchups["season"].astype(int)
    matchups["teamA_id"] = matchups["teamA_id"].astype(int)
    matchups["teamB_id"] = matchups["teamB_id"].astype(int)
    matchups[TARGET_COL] = matchups[TARGET_COL].astype(int)

    if not set(matchups[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COL} must be binary 0/1 in tournament_matchups_v2.csv")

    print(f"Loaded team features rows: {len(team_features):,}")
    print(f"Loaded tournament matchups rows: {len(matchups):,}")
    print(
        "Team feature seasons coverage: "
        f"{team_features['season'].min()}-{team_features['season'].max()}"
    )
    print(
        "Tournament matchup seasons coverage: "
        f"{matchups['season'].min()}-{matchups['season'].max()}"
    )

    return team_features, matchups


def _train_and_score_one_model(
    *,
    modeling_df: pd.DataFrame,
    model_name: str,
    test_season: int,
    random_state: int,
    models_dir: Path,
    save_model_artifacts: bool,
) -> dict[str, Any] | None:
    """Train one model on prior seasons and evaluate one held-out season."""
    feature_cols = get_feature_columns(modeling_df, model_name, strict=False)

    train_df = modeling_df[modeling_df["season"] < test_season].copy()
    test_df = modeling_df[modeling_df["season"] == test_season].copy()

    if train_df.empty:
        print(f"[skip] {model_name} season={test_season}: no prior-season training rows")
        return None
    if test_df.empty:
        print(f"[skip] {model_name} season={test_season}: no test rows")
        return None

    if train_df["season"].max() >= test_season:
        raise RuntimeError("Detected potential leakage: training includes test/future season rows.")

    if train_df[TARGET_COL].nunique() < 2:
        print(f"[skip] {model_name} season={test_season}: training labels are single-class")
        return None
    if test_df[TARGET_COL].nunique() < 2:
        print(f"[warn] {model_name} season={test_season}: test labels single-class; roc_auc may be NaN")

    matrices = build_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
    )

    model = build_model(model_name=model_name, random_state=random_state)
    model.fit(matrices["X_train"], matrices["y_train"])

    y_prob = model.predict_proba(matrices["X_test"])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model_name": model_name,
        "test_season": int(test_season),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **compute_metrics(matrices["y_test"], y_prob, y_pred),
    }

    prediction_cols = [
        col
        for col in ["season", "round", "region", "teamA_id", "teamB_id", TARGET_COL]
        if col in test_df.columns
    ]
    predictions_df = test_df[prediction_cols].copy()
    predictions_df["pred_prob"] = y_prob
    predictions_df["pred_class"] = y_pred
    predictions_df["model_name"] = model_name
    predictions_df["test_season"] = int(test_season)

    if save_model_artifacts:
        model_path = models_dir / f"{model_name}_{test_season}.joblib"
        save_model(model, model_path)

    return {
        "metrics": metrics,
        "predictions": predictions_df,
    }


def run_v2_backtests(
    *,
    team_features_path: Path,
    matchups_path: Path,
    predictions_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    model_names: list[str],
    min_train_seasons: int,
    random_state: int,
    save_model_artifacts: bool,
) -> dict[str, pd.DataFrame]:
    """Run season-by-season walk-forward backtests for v2 datasets."""
    team_features, matchups = load_v2_inputs(team_features_path, matchups_path)

    modeling_df = build_modeling_dataframe(
        team_profiles_df=team_features,
        tourney_matchups_df=matchups,
        games_boxscores_df=None,
        strict=False,
    )

    if modeling_df.empty:
        raise ValueError("Modeling dataframe is empty after building v2 features.")

    print(f"Modeling dataframe shape: {modeling_df.shape}")
    print(
        "Modeling dataframe season coverage: "
        f"{int(modeling_df['season'].min())}-{int(modeling_df['season'].max())}"
    )

    test_seasons = get_available_test_seasons(modeling_df, min_train_seasons=min_train_seasons)
    if not test_seasons:
        raise ValueError("No valid v2 test seasons found. Try lowering --min-train-seasons.")
    print(f"Seasons being evaluated: {test_seasons}")

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for season in test_seasons:
        for model_name in model_names:
            print(f"Training model={model_name} for held-out season={season}")
            result = _train_and_score_one_model(
                modeling_df=modeling_df,
                model_name=model_name,
                test_season=season,
                random_state=random_state,
                models_dir=models_dir,
                save_model_artifacts=save_model_artifacts,
            )
            if result is None:
                continue
            metrics_rows.append(result["metrics"])
            prediction_frames.append(result["predictions"])

    if not metrics_rows or not prediction_frames:
        raise RuntimeError("No successful model-season runs were produced for v2 backtests.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["model_name", "test_season"]).reset_index(drop=True)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["model_name", "test_season", "teamA_id", "teamB_id"])
        .reset_index(drop=True)
    )
    summary_df = summarize_metrics(metrics_df).sort_values("model_name").reset_index(drop=True)

    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if save_model_artifacts:
        models_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = predictions_dir / "model_predictions_by_season_v2.csv"
    metrics_path = reports_dir / "model_metrics_by_season_v2.csv"
    summary_path = reports_dir / "model_summary_v2.csv"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved v2 outputs:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Summary: {summary_path}")
    if save_model_artifacts:
        print(f"  Model artifacts directory: {models_dir}")

    return {
        "predictions": predictions_df,
        "metrics": metrics_df,
        "summary": summary_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model backtests on processed v2 datasets.")
    parser.add_argument("--team-features-path", type=Path, default=DEFAULT_TEAM_FEATURES_PATH)
    parser.add_argument("--matchups-path", type=Path, default=DEFAULT_MATCHUPS_PATH)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--min-train-seasons", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Disable saving model artifacts to the v2 model directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    run_v2_backtests(
        team_features_path=args.team_features_path,
        matchups_path=args.matchups_path,
        predictions_dir=args.predictions_dir,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        model_names=args.models,
        min_train_seasons=args.min_train_seasons,
        random_state=args.random_state,
        save_model_artifacts=not args.no_save_models,
    )


if __name__ == "__main__":
    main()
