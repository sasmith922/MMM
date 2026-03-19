"""
run_backtest_reduced.py
-----------------------
Run a reduced-feature held-out backtest for one target year.

Usage
-----
    python scripts/run_backtest_reduced.py --test-year 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.features_2026_reduced import (
    BACKTEST_METRICS_REDUCED_PATH,
    FEATURE_LIST_2026_REDUCED_PATH,
    HISTORICAL_MATCHUPS_PROCESSED_PATH,
    HISTORICAL_MATCHUPS_V2_PATH,
    MIN_OVERLAP_FEATURE_COUNT,
    ensure_parent,
    resolve_first_existing_path,
)
from madness_model.model_utils import build_model, build_train_test_matrices

DERIVABLE_DIFF_FEATURES_2026 = [
    "seed_diff",
    "adj_o_diff",
    "adj_d_diff",
    "net_rating_diff",
    "win_pct_diff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out reduced-feature model backtest for one season.")
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--historical-matchups-path", type=Path, default=None)
    parser.add_argument("--model-name", choices=["xgboost", "logistic_regression"], default="xgboost")
    parser.add_argument("--metrics-output-path", type=Path, default=BACKTEST_METRICS_REDUCED_PATH)
    parser.add_argument("--feature-list-output-path", type=Path, default=FEATURE_LIST_2026_REDUCED_PATH)
    return parser.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"y": "label", "target": "label"}
    existing = {old: new for old, new in rename_map.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=existing)


def _select_overlap_feature_columns(df: pd.DataFrame) -> list[str]:
    overlap = [column for column in DERIVABLE_DIFF_FEATURES_2026 if column in df.columns]
    if len(overlap) < MIN_OVERLAP_FEATURE_COUNT:
        raise ValueError(
            "Insufficient reduced overlap feature columns in historical matchups. "
            f"Need >= {MIN_OVERLAP_FEATURE_COUNT}, found {len(overlap)}: {overlap}"
        )
    return overlap


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return ROC-AUC when labels are binary; NaN for single-class test folds."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return clipped-probability log loss; NaN for single-class test folds."""
    eps = 1e-6
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, clipped))


def run_backtest_reduced(
    *,
    test_year: int,
    historical_matchups_path: Path | None = None,
    model_name: str = "xgboost",
    metrics_output_path: Path = BACKTEST_METRICS_REDUCED_PATH,
    feature_list_output_path: Path = FEATURE_LIST_2026_REDUCED_PATH,
) -> tuple[Path, Path]:
    if historical_matchups_path is None:
        historical_matchups_path = resolve_first_existing_path(
            [HISTORICAL_MATCHUPS_V2_PATH, HISTORICAL_MATCHUPS_PROCESSED_PATH],
            purpose="historical matchup dataset for reduced backtest",
        )
    if not historical_matchups_path.exists():
        raise FileNotFoundError(
            f"Missing historical matchup dataset: {historical_matchups_path}. "
            "Build it with scripts/build_features_v2.py or pass --historical-matchups-path."
        )

    historical = _normalize_columns(pd.read_csv(historical_matchups_path))
    required = ["season", "label"]
    missing_required = [column for column in required if column not in historical.columns]
    if missing_required:
        raise KeyError(f"Historical dataset missing required columns: {missing_required}")

    historical = historical.dropna(subset=["season", "label"]).copy()
    historical["season"] = pd.to_numeric(historical["season"], errors="coerce")
    historical["label"] = pd.to_numeric(historical["label"], errors="coerce")
    historical = historical.dropna(subset=["season", "label"]).copy()
    historical["season"] = historical["season"].astype(int)
    historical["label"] = historical["label"].astype(int)

    feature_cols = _select_overlap_feature_columns(historical)
    ensure_parent(feature_list_output_path)
    feature_list_output_path.write_text("\n".join(feature_cols) + "\n", encoding="utf-8")

    train_df = historical[historical["season"] < test_year].copy()
    test_df = historical[historical["season"] == test_year].copy()
    if train_df.empty:
        raise ValueError(f"No training rows found for test year {test_year}.")
    if test_df.empty:
        raise ValueError(f"No held-out rows found for test year {test_year}.")
    if train_df["label"].nunique() < 2:
        raise ValueError("Training rows contain a single class; cannot fit classifier.")

    train_median = train_df[feature_cols].median(numeric_only=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(train_median).fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(train_median).fillna(0.0)

    matrices = build_train_test_matrices(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col="label",
    )
    model = build_model(model_name=model_name, random_state=42)
    model.fit(matrices["X_train"], matrices["y_train"])

    y_prob = np.asarray(model.predict_proba(matrices["X_test"])[:, 1], dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = matrices["y_test"]

    preds_out_path = ROOT_DIR / "outputs" / "predictions" / f"backtest_{test_year}_reduced.csv"
    ensure_parent(preds_out_path)
    prediction_cols = [
        column
        for column in ["season", "round", "region", "teamA_name", "teamB_name", "teamA_seed", "teamB_seed", "label"]
        if column in test_df.columns
    ]
    preds_out = test_df[prediction_cols].copy()
    preds_out["pred_prob"] = y_prob
    preds_out["pred_pick"] = y_pred
    preds_out["model_name"] = model_name
    preds_out["test_year"] = int(test_year)
    preds_out.to_csv(preds_out_path, index=False)

    metrics_row = pd.DataFrame(
        [
            {
                "test_year": int(test_year),
                "model_name": model_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "log_loss": _safe_log_loss(y_true, y_prob),
                "brier_score": float(brier_score_loss(y_true, y_prob)),
                "roc_auc": _safe_roc_auc(y_true, y_prob),
                "feature_count": len(feature_cols),
                "features_used": ",".join(feature_cols),
            }
        ]
    )

    ensure_parent(metrics_output_path)
    if metrics_output_path.exists():
        existing = pd.read_csv(metrics_output_path)
        existing = existing[~((existing["test_year"] == test_year) & (existing["model_name"] == model_name))]
        metrics_out = pd.concat([existing, metrics_row], ignore_index=True)
    else:
        metrics_out = metrics_row
    metrics_out = metrics_out.sort_values(["test_year", "model_name"]).reset_index(drop=True)
    metrics_out.to_csv(metrics_output_path, index=False)

    print(f"[backtest] test_year={test_year} model={model_name} n_train={len(train_df)} n_test={len(test_df)}")
    print(
        "[metrics] "
        f"accuracy={metrics_row.iloc[0]['accuracy']:.4f} "
        f"log_loss={metrics_row.iloc[0]['log_loss']:.4f} "
        f"brier={metrics_row.iloc[0]['brier_score']:.4f} "
        f"roc_auc={metrics_row.iloc[0]['roc_auc']:.4f}"
    )
    print(f"[save] feature list: {feature_list_output_path}")
    print(f"[save] backtest predictions: {preds_out_path}")
    print(f"[save] backtest metrics: {metrics_output_path}")
    return preds_out_path, metrics_output_path


def main() -> None:
    args = parse_args()
    run_backtest_reduced(
        test_year=args.test_year,
        historical_matchups_path=args.historical_matchups_path,
        model_name=args.model_name,
        metrics_output_path=args.metrics_output_path,
        feature_list_output_path=args.feature_list_output_path,
    )


if __name__ == "__main__":
    main()
