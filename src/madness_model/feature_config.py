"""Centralized feature definitions for matchup model training."""

from __future__ import annotations

import warnings

import pandas as pd

TARGET_COL = "target"
TARGET_COLUMN = TARGET_COL  # Backward-compat alias.

METADATA_COLS = [
    "season",
    "round",
    "teamA_id",
    "teamB_id",
    TARGET_COL,
]

SEED_ONLY_FEATURES = ["seed_diff"]

LOGISTIC_FEATURES = [
    "seed_diff",
    "win_pct_diff",
    "margin_diff",
    "ppg_diff",
    "opp_ppg_diff",
    "off_eff_diff",
    "def_eff_diff",
    "elo_diff",
]

DIFF_FEATURES = [
    "seed_diff",
    "win_pct_diff",
    "ppg_diff",
    "opp_ppg_diff",
    "margin_diff",
    "fg_pct_diff",
    "three_pct_diff",
    "ft_pct_diff",
    "reb_diff",
    "ast_diff",
    "tov_diff",
    "stl_diff",
    "blk_diff",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "sos_diff",
    "last10_diff",
    "neutral_win_pct_diff",
    "elo_diff",
]

CONTEXT_FEATURES = [
    "same_conference_flag",
    "round",
    "seed_bucket",
    "upset_bucket",
    "region",
]

TREE_RAW_FEATURES = [
    "teamA_seed",
    "teamB_seed",
    "teamA_win_pct",
    "teamB_win_pct",
    "teamA_points_per_game",
    "teamB_points_per_game",
    "teamA_points_allowed_per_game",
    "teamB_points_allowed_per_game",
    "teamA_average_margin",
    "teamB_average_margin",
    "teamA_offensive_efficiency",
    "teamB_offensive_efficiency",
    "teamA_defensive_efficiency",
    "teamB_defensive_efficiency",
    "teamA_net_efficiency",
    "teamB_net_efficiency",
    "teamA_sos",
    "teamB_sos",
    "teamA_last10_win_pct",
    "teamB_last10_win_pct",
    "teamA_neutral_win_pct",
    "teamB_neutral_win_pct",
    "teamA_elo_pre_tourney",
    "teamB_elo_pre_tourney",
]

TREE_FEATURES = [*DIFF_FEATURES, *TREE_RAW_FEATURES, *CONTEXT_FEATURES]
NEURAL_NET_FEATURES = TREE_FEATURES.copy()

MODEL_FEATURES: dict[str, list[str]] = {
    "logistic_regression": LOGISTIC_FEATURES,
    "random_forest": TREE_FEATURES,
    "xgboost": TREE_FEATURES,
    "neural_net": NEURAL_NET_FEATURES,
    # Backward-compat aliases for prior code/tests.
    "seed_only_logistic": SEED_ONLY_FEATURES,
    "logistic_baseline": LOGISTIC_FEATURES,
}


def validate_feature_columns_exist(
    df: pd.DataFrame,
    feature_cols: list[str],
    strict: bool = True,
) -> list[str]:
    """Validate feature columns exist in a dataframe.

    Returns the usable feature list. If strict=False, missing features are warned
    and dropped; if strict=True, missing features raise a ``KeyError``.
    """

    missing = [column for column in feature_cols if column not in df.columns]
    if missing:
        message = f"Missing feature columns: {missing}"
        if strict:
            raise KeyError(message)
        warnings.warn(f"{message}. Dropping missing optional features.", stacklevel=2)

    usable = [column for column in feature_cols if column in df.columns]
    if not usable:
        raise ValueError("No usable feature columns found for model input.")
    return usable


def get_feature_columns(df: pd.DataFrame, model_name: str, strict: bool = True) -> list[str]:
    """Get validated feature columns for a model name."""

    if model_name not in MODEL_FEATURES:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. Supported: {sorted(MODEL_FEATURES)}"
        )
    return validate_feature_columns_exist(df, MODEL_FEATURES[model_name], strict=strict)


# Backward-compat alias used by older modules.
def get_model_feature_columns(
    modeling_df_columns: list[str],
    model_name: str,
    *,
    strict: bool = True,
) -> list[str]:
    """Compatibility wrapper around :func:`get_feature_columns`."""

    return get_feature_columns(pd.DataFrame(columns=modeling_df_columns), model_name, strict=strict)
