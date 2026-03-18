"""Helpers for scoring matchup rows with trained models."""

from __future__ import annotations

from typing import Any, List

import pandas as pd

from madness_model.model_utils import build_inference_matrix


def predict_matchups(
    model: Any,
    matchup_df: pd.DataFrame,
    *,
    feature_cols: List[str],
    model_feature_cols: List[str],
    model_name: str,
) -> pd.DataFrame:
    """Predict matchup win probabilities for Team A using a fitted model."""
    X_infer = build_inference_matrix(
        inference_df=matchup_df,
        feature_cols=feature_cols,
        model_feature_columns=model_feature_cols,
    )

    pred_prob = model.predict_proba(X_infer)[:, 1]
    pred_class = (pred_prob >= 0.5).astype(int)

    output_cols = [c for c in ["season", "round", "teamA_id", "teamB_id", "target"] if c in matchup_df.columns]
    preds = matchup_df[output_cols].copy()
    preds["pred_prob"] = pred_prob
    preds["pred_class"] = pred_class
    preds["model_name"] = model_name

    return preds
