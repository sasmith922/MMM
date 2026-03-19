"""
train_predict_2026_reduced.py
-----------------------------
Train a dedicated reduced-feature 2026 model using overlapping historical
reduced columns, then generate 2026 matchup probabilities.

Usage
-----
    python scripts/train_predict_2026_reduced.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.features_2026_reduced import (
    FEATURE_LIST_2026_REDUCED_PATH,
    HISTORICAL_MATCHUPS_PROCESSED_PATH,
    HISTORICAL_MATCHUPS_V2_PATH,
    MIN_OVERLAP_FEATURE_COUNT,
    MODEL_ARTIFACT_2026_PATH,
    PREDICTIONS_2026_PATH,
    TEAM_FEATURES_2026_REDUCED_PATH,
    TOURNEY_MATCHUPS_2026_PATH,
    ensure_parent,
    parse_seed_number,
    resolve_first_existing_path,
)
from madness_model.model_utils import build_model, build_train_test_matrices, save_model

# We intentionally train on strict overlap to avoid assumptions from full-feature eras.
DERIVABLE_DIFF_FEATURES_2026 = [
    "seed_diff",
    "adj_o_diff",
    "adj_d_diff",
    "net_rating_diff",
    "win_pct_diff",
]

TEAM_FEATURE_TO_2026_COLUMNS = {
    "seed_diff": ("seed_num", "seed_num"),
    "adj_o_diff": ("pre_tourney_adjoe", "pre_tourney_adjoe"),
    "adj_d_diff": ("pre_tourney_adjde", "pre_tourney_adjde"),
    "net_rating_diff": ("pre_tourney_adjem", "pre_tourney_adjem"),
    "win_pct_diff": ("win_pct", "win_pct"),
}

def _build_2026_prediction_dataset(
    team_features_2026: pd.DataFrame,
    tourney_matchups_2026: pd.DataFrame,
) -> pd.DataFrame:
    def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    required_matchup_cols = ["season", "teamA_name_norm", "teamB_name_norm"]
    missing_matchup_cols = [col for col in required_matchup_cols if col not in tourney_matchups_2026.columns]
    if missing_matchup_cols:
        raise KeyError(f"2026 matchup file missing required columns: {missing_matchup_cols}")

    if "team_name_norm" not in team_features_2026.columns:
        raise KeyError("2026 team features missing required column: team_name_norm")

    feat = team_features_2026.copy()
    if "seed_num" not in feat.columns:
        feat["seed_num"] = feat["seed"].map(parse_seed_number)
    a_cols = {column: f"teamA_{column}" for column in feat.columns if column != "team_name_norm"}
    a_cols["team_name_norm"] = "teamA_name_norm"
    b_cols = {column: f"teamB_{column}" for column in feat.columns if column != "team_name_norm"}
    b_cols["team_name_norm"] = "teamB_name_norm"
    merged = (
        tourney_matchups_2026.copy()
        .merge(feat.rename(columns=a_cols), on="teamA_name_norm", how="left", validate="many_to_one")
        .merge(feat.rename(columns=b_cols), on="teamB_name_norm", how="left", validate="many_to_one")
    )

    for diff_name, (a_source, b_source) in TEAM_FEATURE_TO_2026_COLUMNS.items():
        col_a = _find_first_existing_column(
            merged, [f"teamA_{a_source}", f"teamA_{a_source}_x", f"teamA_{a_source}_y"]
        )
        col_b = _find_first_existing_column(
            merged, [f"teamB_{b_source}", f"teamB_{b_source}_x", f"teamB_{b_source}_y"]
        )
        if col_a and col_b:
            merged[diff_name] = pd.to_numeric(merged[col_a], errors="coerce") - pd.to_numeric(
                merged[col_b], errors="coerce"
            )

    return merged


def _select_overlap_feature_columns(train_df: pd.DataFrame, prediction_df: pd.DataFrame) -> list[str]:
    overlap = [
        column
        for column in DERIVABLE_DIFF_FEATURES_2026
        if column in train_df.columns and column in prediction_df.columns
    ]
    if len(overlap) < MIN_OVERLAP_FEATURE_COUNT:
        raise ValueError(
            "Insufficient overlapping reduced feature columns found between historical data and 2026 prediction set. "
            f"Need >= {MIN_OVERLAP_FEATURE_COUNT}, found {len(overlap)}: {overlap}"
        )
    return overlap


def _write_feature_list(path: Path, feature_cols: list[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train reduced-feature model and predict 2026 R64 matchups."
    )
    parser.add_argument("--historical-matchups-path", type=Path, default=None)
    parser.add_argument("--team-features-2026-path", type=Path, default=TEAM_FEATURES_2026_REDUCED_PATH)
    parser.add_argument("--tourney-matchups-2026-path", type=Path, default=TOURNEY_MATCHUPS_2026_PATH)
    parser.add_argument("--model-output-path", type=Path, default=MODEL_ARTIFACT_2026_PATH)
    parser.add_argument("--predictions-output-path", type=Path, default=PREDICTIONS_2026_PATH)
    parser.add_argument("--feature-list-output-path", type=Path, default=FEATURE_LIST_2026_REDUCED_PATH)
    return parser.parse_args()


def train_and_predict_2026_reduced(
    *,
    historical_matchups_path: Path | None = None,
    team_features_2026_path: Path = TEAM_FEATURES_2026_REDUCED_PATH,
    tourney_matchups_2026_path: Path = TOURNEY_MATCHUPS_2026_PATH,
    model_output_path: Path = MODEL_ARTIFACT_2026_PATH,
    predictions_output_path: Path = PREDICTIONS_2026_PATH,
    feature_list_output_path: Path = FEATURE_LIST_2026_REDUCED_PATH,
) -> tuple[Path, Path, Path]:
    if historical_matchups_path is None:
        historical_matchups_path = resolve_first_existing_path(
            [HISTORICAL_MATCHUPS_V2_PATH, HISTORICAL_MATCHUPS_PROCESSED_PATH],
            purpose="historical matchup dataset for reduced training",
        )
    elif not historical_matchups_path.exists():
        raise FileNotFoundError(f"Missing historical matchup dataset: {historical_matchups_path}.")
    if not team_features_2026_path.exists():
        raise FileNotFoundError(
            f"Missing 2026 reduced team features: {team_features_2026_path}. "
            "Build/clean them first with scripts/build_2026_dataset.py."
        )
    if not tourney_matchups_2026_path.exists():
        raise FileNotFoundError(
            f"Missing 2026 matchup file: {tourney_matchups_2026_path}. "
            "Build it first with scripts/build_2026_dataset.py."
        )

    historical = pd.read_csv(historical_matchups_path)
    if "label" not in historical.columns:
        raise KeyError(f"Historical matchups must include 'label': {historical_matchups_path}")

    team_features_2026 = pd.read_csv(team_features_2026_path)
    matchups_2026 = pd.read_csv(tourney_matchups_2026_path)
    pred_df = _build_2026_prediction_dataset(team_features_2026, matchups_2026)

    feature_cols = _select_overlap_feature_columns(historical, pred_df)
    print(f"[train] using reduced overlap features: {feature_cols}")

    historical = historical.dropna(subset=["label"]).copy()
    train_df = historical[feature_cols + ["label"]].copy()
    pred_matrix = pred_df[feature_cols].copy()

    train_median = train_df[feature_cols].median(numeric_only=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(train_median).fillna(0.0)
    pred_matrix[feature_cols] = pred_matrix[feature_cols].fillna(train_median).fillna(0.0)

    # `build_train_test_matrices` returns the encoded training feature schema we need
    # for inference alignment. A one-row stub keeps this utility path centralized.
    test_stub = train_df.iloc[:1].copy()
    matrices = build_train_test_matrices(
        train_df=train_df,
        test_df=test_stub,
        feature_cols=feature_cols,
        target_col="label",
    )

    model = build_model("xgboost", random_state=42)
    model.fit(matrices["X_train"], matrices["y_train"])

    X_pred = pd.get_dummies(pred_matrix, dummy_na=False)
    X_pred = X_pred.reindex(columns=matrices["model_feature_columns"], fill_value=0.0).fillna(0.0)
    pred_df["win_prob_a"] = np.asarray(model.predict_proba(X_pred)[:, 1], dtype=float)

    ensure_parent(model_output_path)
    ensure_parent(predictions_output_path)
    save_model(model, model_output_path)
    _write_feature_list(feature_list_output_path, feature_cols)

    save_cols = [
        "season",
        "round",
        "region",
        "teamA_name",
        "teamB_name",
        "teamA_seed",
        "teamB_seed",
        "win_prob_a",
        *feature_cols,
    ]
    final_cols = [column for column in save_cols if column in pred_df.columns]
    pred_df[final_cols].to_csv(predictions_output_path, index=False)

    print(f"[save] model artifact: {model_output_path}")
    print(f"[save] reduced feature list: {feature_list_output_path}")
    print(f"[save] 2026 reduced predictions: {predictions_output_path} ({len(pred_df)} rows)")
    return model_output_path, predictions_output_path, feature_list_output_path


def main() -> None:
    args = parse_args()
    train_and_predict_2026_reduced(
        historical_matchups_path=args.historical_matchups_path,
        team_features_2026_path=args.team_features_2026_path,
        tourney_matchups_2026_path=args.tourney_matchups_2026_path,
        model_output_path=args.model_output_path,
        predictions_output_path=args.predictions_output_path,
        feature_list_output_path=args.feature_list_output_path,
    )


if __name__ == "__main__":
    main()
