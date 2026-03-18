"""Centralized feature definitions and model feature-set selection."""

from __future__ import annotations

import warnings
from typing import Dict, List

# Canonical binary supervised label column (1 = Team A wins, 0 = Team B wins).
TARGET_COLUMN = "target"

TEAM_ID_COLUMNS: List[str] = ["teamA_id", "teamB_id"]
METADATA_COLUMNS: List[str] = ["season", "round", "region", *TEAM_ID_COLUMNS]

# Base (un-prefixed) team-season columns expected from team feature sources.
RAW_TEAM_FEATURE_COLUMNS: List[str] = [
    "seed",
    "wins",
    "losses",
    "win_pct",
    "points_per_game",
    "points_allowed_per_game",
    "average_margin",
    "fg_pct",
    "three_pct",
    "ft_pct",
    "rebounds_per_game",
    "assists_per_game",
    "turnovers_per_game",
    "steals_per_game",
    "blocks_per_game",
    "offensive_efficiency",
    "defensive_efficiency",
    "net_efficiency",
    "sos",
    "last10_win_pct",
    "neutral_win_pct",
    "elo_pre_tourney",
    "conference",
]

DIFF_FEATURE_COLUMNS: List[str] = [
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
    "win_pct_ratio",
    "off_eff_ratio",
    "net_eff_ratio",
    "elo_ratio",
]

CONTEXT_FEATURE_COLUMNS: List[str] = [
    "same_conference_flag",
    "round",
    "region",
    "seed_bucket",
    "upset_bucket",
]

SEED_ONLY_FEATURES: List[str] = ["seed_diff"]

LOGISTIC_BASELINE_FEATURES: List[str] = [
    "seed_diff",
    "win_pct_diff",
    "margin_diff",
    "ppg_diff",
    "opp_ppg_diff",
    "off_eff_diff",
    "def_eff_diff",
    "elo_diff",
]

TREE_MODEL_FEATURES: List[str] = [
    *DIFF_FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
    "teamA_seed",
    "teamB_seed",
    "teamA_win_pct",
    "teamB_win_pct",
    "teamA_offensive_efficiency",
    "teamB_offensive_efficiency",
    "teamA_defensive_efficiency",
    "teamB_defensive_efficiency",
    "teamA_net_efficiency",
    "teamB_net_efficiency",
    "teamA_elo_pre_tourney",
    "teamB_elo_pre_tourney",
]

EXPANDED_MODEL_FEATURES: List[str] = [
    *TREE_MODEL_FEATURES,
    "teamA_points_per_game",
    "teamB_points_per_game",
    "teamA_points_allowed_per_game",
    "teamB_points_allowed_per_game",
    "teamA_average_margin",
    "teamB_average_margin",
    "teamA_sos",
    "teamB_sos",
    "teamA_last10_win_pct",
    "teamB_last10_win_pct",
]

MODEL_FEATURE_SETS: Dict[str, List[str]] = {
    "seed_only_logistic": SEED_ONLY_FEATURES,
    "logistic_baseline": LOGISTIC_BASELINE_FEATURES,
    "random_forest": TREE_MODEL_FEATURES,
    "xgboost": EXPANDED_MODEL_FEATURES,
}


def resolve_feature_columns(
    available_columns: List[str],
    desired_columns: List[str],
    *,
    strict: bool = False,
    feature_set_name: str | None = None,
) -> List[str]:
    """Resolve the final usable feature columns from a desired feature set."""
    available = set(available_columns)
    resolved = [col for col in desired_columns if col in available]
    missing = [col for col in desired_columns if col not in available]

    if missing:
        prefix = f"Feature set '{feature_set_name}'" if feature_set_name else "Feature set"
        message = f"{prefix} is missing {len(missing)} features: {missing}"
        if strict:
            raise KeyError(message)
        warnings.warn(f"{message}. Dropping missing features.", stacklevel=2)

    if not resolved:
        raise ValueError(
            f"No usable features resolved for set '{feature_set_name or 'unknown'}'."
        )

    return resolved


def get_model_feature_columns(
    modeling_df_columns: List[str],
    model_name: str,
    *,
    strict: bool = False,
) -> List[str]:
    """Return the resolved feature list for a model name."""
    if model_name not in MODEL_FEATURE_SETS:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Supported: {sorted(MODEL_FEATURE_SETS)}"
        )

    return resolve_feature_columns(
        available_columns=modeling_df_columns,
        desired_columns=MODEL_FEATURE_SETS[model_name],
        strict=strict,
        feature_set_name=model_name,
    )
