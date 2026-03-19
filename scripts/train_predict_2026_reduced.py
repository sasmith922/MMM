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
    HISTORICAL_MATCHUPS_V2_PATH,
    MODEL_ARTIFACT_2026_PATH,
    PREDICTIONS_2026_PATH,
    TEAM_FEATURES_2026_REDUCED_PATH,
    TOURNEY_MATCHUPS_2026_PATH,
    ensure_parent,
    parse_seed_number,
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
    required_matchup_cols = ["season", "teamA_name_norm", "teamB_name_norm"]
    missing_matchup_cols = [col for col in required_matchup_cols if col not in tourney_matchups_2026.columns]
    if missing_matchup_cols:
        raise KeyError(f"2026 matchup file missing required columns: {missing_matchup_cols}")

    if "team_name_norm" not in team_features_2026.columns:
        raise KeyError("2026 team features missing required column: team_name_norm")

    feat = team_features_2026.copy()
    if "seed_num" not in feat.columns:
        feat["seed_num"] = feat["seed"].map(parse_seed_number)

    a_cols = {
        "team_name_norm": "teamA_name_norm",
        "team_name": "teamA_name",
        "seed": "teamA_seed",
        "seed_num": "teamA_seed_num",
    }
    b_cols = {
        "team_name_norm": "teamB_name_norm",
        "team_name": "teamB_name",
        "seed": "teamB_seed",
        "seed_num": "teamB_seed_num",
    }
    merged = (
        tourney_matchups_2026.copy()
        .merge(feat.rename(columns=a_cols), on="teamA_name_norm", how="left", validate="many_to_one")
        .merge(feat.rename(columns=b_cols), on="teamB_name_norm", how="left", validate="many_to_one")
    )

    for diff_name, (a_source, b_source) in TEAM_FEATURE_TO_2026_COLUMNS.items():
        col_a = f"teamA_{a_source}"
        col_b = f"teamB_{b_source}"
        if col_a in merged.columns and col_b in merged.columns:
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
    if not overlap:
        raise ValueError(
            "No overlapping reduced feature columns found between historical matchups and 2026 prediction set."
        )
    return overlap


def train_and_predict_2026_reduced(
    *,
    historical_matchups_path: Path = HISTORICAL_MATCHUPS_V2_PATH,
    team_features_2026_path: Path = TEAM_FEATURES_2026_REDUCED_PATH,
    tourney_matchups_2026_path: Path = TOURNEY_MATCHUPS_2026_PATH,
    model_output_path: Path = MODEL_ARTIFACT_2026_PATH,
    predictions_output_path: Path = PREDICTIONS_2026_PATH,
) -> tuple[Path, Path]:
    if not historical_matchups_path.exists():
        raise FileNotFoundError(
            f"Missing historical reduced matchups: {historical_matchups_path}. "
            "Build it first with scripts/build_features_v2.py."
        )
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
    print(f"[save] 2026 reduced predictions: {predictions_output_path} ({len(pred_df)} rows)")
    return model_output_path, predictions_output_path


def main() -> None:
    train_and_predict_2026_reduced()


if __name__ == "__main__":
    main()
