"""Central feature assembly for matchup-level modeling datasets."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd

from madness_model.feature_config import TARGET_COLUMN


REQUIRED_MATCHUP_COLUMNS = ["season", "teamA_id", "teamB_id"]
REQUIRED_TEAM_PROFILE_COLUMNS = ["season", "team_id"]

# Explicit mapping for canonical engineered diff names.
DIFF_COLUMN_MAPPING: Dict[str, str] = {
    "seed_diff": "seed",
    "win_pct_diff": "win_pct",
    "ppg_diff": "points_per_game",
    "opp_ppg_diff": "points_allowed_per_game",
    "margin_diff": "average_margin",
    "fg_pct_diff": "fg_pct",
    "three_pct_diff": "three_pct",
    "ft_pct_diff": "ft_pct",
    "reb_diff": "rebounds_per_game",
    "ast_diff": "assists_per_game",
    "tov_diff": "turnovers_per_game",
    "stl_diff": "steals_per_game",
    "blk_diff": "blocks_per_game",
    "off_eff_diff": "offensive_efficiency",
    "def_eff_diff": "defensive_efficiency",
    "net_eff_diff": "net_efficiency",
    "sos_diff": "sos",
    "last10_diff": "last10_win_pct",
    "neutral_win_pct_diff": "neutral_win_pct",
    "elo_diff": "elo_pre_tourney",
}

BOX_FEATURE_PREFIX = "box_"
SEED_BUCKET_BINS = [-np.inf, 2, 5, np.inf]
SEED_BUCKET_LABELS = [0, 1, 2]  # 0=close seed matchup, 1=moderate gap, 2=large gap


def _require_columns(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing required columns: {missing}")


def _coerce_team_profiles(team_profiles_df: pd.DataFrame) -> pd.DataFrame:
    team_profiles = team_profiles_df.copy()

    rename_map = {}
    if "teamID" in team_profiles.columns and "team_id" not in team_profiles.columns:
        rename_map["teamID"] = "team_id"
    if "Season" in team_profiles.columns and "season" not in team_profiles.columns:
        rename_map["Season"] = "season"
    if rename_map:
        team_profiles = team_profiles.rename(columns=rename_map)

    _require_columns(team_profiles, REQUIRED_TEAM_PROFILE_COLUMNS, "team_profiles_df")

    team_profiles["season"] = team_profiles["season"].astype(int)
    team_profiles["team_id"] = team_profiles["team_id"].astype(int)

    return team_profiles


def _filter_regular_season_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Try to filter out tournament/postseason rows before aggregation."""
    filtered = games_df.copy()

    if "is_tourney" in filtered.columns:
        filtered = filtered[~filtered["is_tourney"].fillna(False)]
    if "is_postseason" in filtered.columns:
        filtered = filtered[~filtered["is_postseason"].fillna(False)]
    if "season_phase" in filtered.columns:
        phase = filtered["season_phase"].astype(str).str.lower()
        filtered = filtered[~phase.str.contains("tourn|post", regex=True, na=False)]

    return filtered


def _derive_games_boxscore_features(
    games_boxscores_df: pd.DataFrame,
    existing_team_feature_columns: list[str],
) -> pd.DataFrame:
    """Build season/team supplemental features from processed game boxscores."""
    games = games_boxscores_df.copy()

    team_col_candidates = ["team_id", "TeamID", "teamID"]
    team_col = next((c for c in team_col_candidates if c in games.columns), None)
    if team_col is None:
        warnings.warn(
            "games_boxscores_df has no team_id-like column; skipping boxscore enrichment.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["season", "team_id"])

    if team_col != "team_id":
        games = games.rename(columns={team_col: "team_id"})

    _require_columns(games, ["season", "team_id"], "games_boxscores_df")

    games = _filter_regular_season_games(games)

    candidate_numeric_cols = [
        c
        for c in games.select_dtypes(include=[np.number]).columns
        if c
        not in {
            "season",
            "team_id",
            "opponent_team_id",
            "opp_team_id",
            "game_id",
        }
    ]

    if not candidate_numeric_cols:
        return games[["season", "team_id"]].drop_duplicates().reset_index(drop=True)

    # Only keep truly supplemental columns not already in team_profiles.
    supplemental_numeric_cols = [
        c for c in candidate_numeric_cols if c not in set(existing_team_feature_columns)
    ]

    if not supplemental_numeric_cols:
        return games[["season", "team_id"]].drop_duplicates().reset_index(drop=True)

    agg_df = (
        games.groupby(["season", "team_id"], as_index=False)[supplemental_numeric_cols]
        .mean(numeric_only=True)
        # Prefix supplemental aggregate features to distinguish them from
        # team_profiles columns that are already season-level summaries.
        .rename(columns={c: f"{BOX_FEATURE_PREFIX}{c}" for c in supplemental_numeric_cols})
    )

    return agg_df


def _merge_team_features(
    matchups_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    team_col_in_matchups: str,
    team_prefix: str,
) -> pd.DataFrame:
    """Merge a team-season feature table onto a matchup table for one side."""
    side_features = features_df.rename(columns={"team_id": team_col_in_matchups})

    rename_cols = {
        c: f"{team_prefix}_{c}"
        for c in side_features.columns
        if c not in {"season", team_col_in_matchups}
    }
    side_features = side_features.rename(columns=rename_cols)

    merged = matchups_df.merge(
        side_features,
        on=["season", team_col_in_matchups],
        how="left",
        validate="many_to_one",
    )
    return merged


def _validate_profile_joins(modeling_df: pd.DataFrame) -> None:
    required_pairs = [
        ("teamA_seed", "teamA_id"),
        ("teamB_seed", "teamB_id"),
    ]
    for feature_col, team_col in required_pairs:
        if feature_col in modeling_df.columns:
            missing_rows = modeling_df[modeling_df[feature_col].isna()]
            if not missing_rows.empty:
                sample = missing_rows[["season", team_col]].head(5).to_dict("records")
                raise ValueError(
                    "Critical team-profile join produced missing rows. "
                    f"Column '{feature_col}' has nulls; sample keys: {sample}"
                )


def _build_diff_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    for diff_col, base_col in DIFF_COLUMN_MAPPING.items():
        teamA_col = f"teamA_{base_col}"
        teamB_col = f"teamB_{base_col}"
        if teamA_col in modeling_df.columns and teamB_col in modeling_df.columns:
            modeling_df[diff_col] = modeling_df[teamA_col] - modeling_df[teamB_col]

    # Generic fallback: generate diffs for any common numeric teamA/teamB pair not yet covered.
    teamA_numeric = {
        col.removeprefix("teamA_"): col
        for col in modeling_df.columns
        if col.startswith("teamA_") and pd.api.types.is_numeric_dtype(modeling_df[col])
    }
    for base_col, teamA_col in teamA_numeric.items():
        teamB_col = f"teamB_{base_col}"
        if teamB_col not in modeling_df.columns:
            continue
        generic_diff_name = f"{base_col}_diff"
        if generic_diff_name in modeling_df.columns:
            continue
        modeling_df[generic_diff_name] = modeling_df[teamA_col] - modeling_df[teamB_col]

    return modeling_df


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_safe = denominator.replace(0, np.nan)
    ratio = numerator / denominator_safe
    return ratio.replace([np.inf, -np.inf], np.nan)


def _build_ratio_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    ratio_map = {
        "win_pct_ratio": "win_pct",
        "off_eff_ratio": "offensive_efficiency",
        "net_eff_ratio": "net_efficiency",
        "elo_ratio": "elo_pre_tourney",
    }
    for ratio_col, base_col in ratio_map.items():
        teamA_col = f"teamA_{base_col}"
        teamB_col = f"teamB_{base_col}"
        if teamA_col in modeling_df.columns and teamB_col in modeling_df.columns:
            modeling_df[ratio_col] = _safe_ratio(modeling_df[teamA_col], modeling_df[teamB_col])
    return modeling_df


def _build_context_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    if (
        "same_conference_flag" not in modeling_df.columns
        and "teamA_conference" in modeling_df.columns
        and "teamB_conference" in modeling_df.columns
    ):
        modeling_df["same_conference_flag"] = (
            modeling_df["teamA_conference"].astype(str)
            == modeling_df["teamB_conference"].astype(str)
        ).astype(int)

    if "seed_diff" in modeling_df.columns:
        abs_seed_diff = modeling_df["seed_diff"].abs()
        modeling_df["seed_bucket"] = pd.cut(
            abs_seed_diff,
            bins=SEED_BUCKET_BINS,
            labels=SEED_BUCKET_LABELS,
        ).astype(float)

    if "teamA_seed" in modeling_df.columns and "teamB_seed" in modeling_df.columns:
        modeling_df["upset_bucket"] = (modeling_df["teamA_seed"] > modeling_df["teamB_seed"]).astype(int)

    return modeling_df


def build_modeling_dataframe(
    team_profiles_df: pd.DataFrame,
    tourney_matchups_df: pd.DataFrame,
    games_boxscores_df: pd.DataFrame | None = None,
    use_raw_team_features: bool = True,
    use_diff_features: bool = True,
    use_context_features: bool = True,
) -> pd.DataFrame:
    """Build the final matchup-level modeling dataframe.

    Parameters
    ----------
    team_profiles_df:
        Team-season feature table with one row per ``(season, team_id)``.
    tourney_matchups_df:
        Supervised tournament matchup base table containing at least
        ``season``, ``teamA_id``, ``teamB_id`` and optionally ``target`` and
        matchup metadata columns.
    games_boxscores_df:
        Optional processed games/boxscores source used to derive supplemental
        season-level team features not already in ``team_profiles_df``.
    use_raw_team_features:
        Whether to retain prefixed ``teamA_`` and ``teamB_`` raw columns.
    use_diff_features:
        Whether to create engineered difference/ratio matchup features.
    use_context_features:
        Whether to create context features (conference, seed buckets, etc.).

    Returns
    -------
    pd.DataFrame
        Matchup-level modeling dataframe ready for season-based train/test
        splitting and model training.

    Leakage note
    ------------
    This function assumes upstream processed tables already contain pre-tournament
    frozen season features. We do not recompute any tournament-inclusive season
    statistics. When aggregating games_boxscores, we apply best-effort filtering
    hooks for postseason/tournament rows when explicit indicators are present.
    """
    team_profiles = _coerce_team_profiles(team_profiles_df)

    matchups = tourney_matchups_df.copy()
    _require_columns(matchups, REQUIRED_MATCHUP_COLUMNS, "tourney_matchups_df")

    matchups["season"] = matchups["season"].astype(int)
    matchups["teamA_id"] = matchups["teamA_id"].astype(int)
    matchups["teamB_id"] = matchups["teamB_id"].astype(int)

    modeling_df = matchups.copy()

    # 1) Join team profile features for Team A and Team B.
    modeling_df = _merge_team_features(
        modeling_df,
        team_profiles,
        team_col_in_matchups="teamA_id",
        team_prefix="teamA",
    )
    modeling_df = _merge_team_features(
        modeling_df,
        team_profiles,
        team_col_in_matchups="teamB_id",
        team_prefix="teamB",
    )

    _validate_profile_joins(modeling_df)

    # 2) Optional enrichment from games boxscores.
    if games_boxscores_df is not None and not games_boxscores_df.empty:
        team_feature_cols = [
            c for c in team_profiles.columns if c not in {"season", "team_id"}
        ]
        supplemental = _derive_games_boxscore_features(games_boxscores_df, team_feature_cols)
        if not supplemental.empty:
            modeling_df = _merge_team_features(
                modeling_df,
                supplemental,
                team_col_in_matchups="teamA_id",
                team_prefix="teamA",
            )
            modeling_df = _merge_team_features(
                modeling_df,
                supplemental,
                team_col_in_matchups="teamB_id",
                team_prefix="teamB",
            )

    # 3) Engineered matchup features.
    if use_diff_features:
        modeling_df = _build_diff_features(modeling_df)
        modeling_df = _build_ratio_features(modeling_df)

    if use_context_features:
        modeling_df = _build_context_features(modeling_df)

    if not use_raw_team_features:
        raw_cols = [
            c
            for c in modeling_df.columns
            if c.startswith("teamA_") or c.startswith("teamB_")
        ]
        modeling_df = modeling_df.drop(columns=raw_cols)

    # Keep metadata + features; preserve target/metadata columns if present.
    preferred_front = [
        "season",
        "round",
        "region",
        "teamA_id",
        "teamB_id",
        TARGET_COLUMN,
    ]
    front_cols = [c for c in preferred_front if c in modeling_df.columns]
    other_cols = [c for c in modeling_df.columns if c not in front_cols]

    return modeling_df[front_cols + other_cols]
