"""Central feature assembly for matchup-level modeling datasets."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from madness_model.feature_config import TARGET_COL

REQUIRED_MATCHUP_COLUMNS = ["season", "teamA_id", "teamB_id", TARGET_COL]
REQUIRED_TEAM_PROFILE_COLUMNS = ["season", "team_id"]

DIFF_COLUMN_MAPPING: dict[str, str] = {
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
TEAM_PROFILES_DUPLICATES_AUDIT_PATH = Path("outputs/reports/team_profiles_duplicates_audit.csv")


def _require_columns(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing required columns: {missing}")


def _coerce_team_profiles(team_profiles: pd.DataFrame) -> pd.DataFrame:
    team_profiles = team_profiles.copy()

    if "team_id" not in team_profiles.columns and "kaggle_team_id" in team_profiles.columns:
        team_profiles = team_profiles.rename(columns={"kaggle_team_id": "team_id"})

    team_profiles["season"] = pd.to_numeric(team_profiles["season"], errors="coerce")
    team_profiles["team_id"] = pd.to_numeric(team_profiles["team_id"], errors="coerce")
    bad = team_profiles["season"].isna() | team_profiles["team_id"].isna()
    if bad.any():
        print(f"Dropping {bad.sum()} team_profiles rows with missing season/team_id")
        cols_to_show = [
            c for c in ["season", "team_name", "canonical_team_name", "conference", "seed"]
            if c in team_profiles.columns
        ]
        if cols_to_show:
            print(team_profiles.loc[bad, cols_to_show].head(20).to_string(index=False))
        team_profiles = team_profiles.loc[~bad].copy()

    team_profiles["season"] = team_profiles["season"].astype(int)
    team_profiles["team_id"] = team_profiles["team_id"].astype(int)
    return _dedupe_team_profiles(team_profiles)

def _dedupe_team_profiles(team_profiles: pd.DataFrame) -> pd.DataFrame:
    deduped = team_profiles.copy()
    duplicate_mask = deduped.duplicated(subset=["season", "team_id"], keep=False)
    if not duplicate_mask.any():
        return deduped

    duplicate_rows = int(duplicate_mask.sum())
    print(f"Deduplicating {duplicate_rows} team_profiles rows across repeated (season, team_id) keys")

    duplicates_audit = deduped.loc[duplicate_mask].copy()
    audit_path = TEAM_PROFILES_DUPLICATES_AUDIT_PATH
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_audit.to_csv(audit_path, index=False)

    deduped["_non_null_count"] = deduped.notna().sum(axis=1)
    helper_column_mapping = {
        "_has_seed": "seed",
        "_has_elo_pre_tourney": "elo_pre_tourney",
        "_has_team_name": "team_name",
        "_has_canonical_team_name": "canonical_team_name",
    }
    for helper_col, source_col in helper_column_mapping.items():
        deduped[helper_col] = deduped[source_col].notna().astype(int) if source_col in deduped.columns else 0

    sort_columns = [
        "season",
        "team_id",
        "_has_seed",
        "_has_elo_pre_tourney",
        "_has_team_name",
        "_has_canonical_team_name",
        "_non_null_count",
    ]
    ascending = [True, True, False, False, False, False, False]
    deduped = deduped.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    deduped = deduped.drop_duplicates(subset=["season", "team_id"], keep="first")
    return deduped.drop(
        columns=[
            "_non_null_count",
            "_has_seed",
            "_has_elo_pre_tourney",
            "_has_team_name",
            "_has_canonical_team_name",
        ],
        errors="ignore",
    )


def _validate_unique_team_profiles(team_profiles: pd.DataFrame, *, strict: bool) -> None:
    duplicated = team_profiles.duplicated(subset=["season", "team_id"], keep=False)
    if not duplicated.any():
        return

    sample = team_profiles.loc[duplicated, ["season", "team_id"]].head(10).to_dict("records")
    message = (
        "team_profiles_df must be unique on (season, team_id). "
        f"Found duplicates like: {sample}"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _validate_matchup_joins(
    matchups: pd.DataFrame,
    team_profiles: pd.DataFrame,
    *,
    strict: bool,
) -> None:
    profile_keys = team_profiles[["season", "team_id"]].drop_duplicates()

    missing_teamA_df = (
        matchups[["season", "teamA_id"]]
        .rename(columns={"teamA_id": "team_id"})
        .merge(profile_keys, on=["season", "team_id"], how="left", indicator=True)
        .loc[lambda df: df["_merge"] == "left_only", ["season", "team_id"]]
        .drop_duplicates()
    )
    missing_teamB_df = (
        matchups[["season", "teamB_id"]]
        .rename(columns={"teamB_id": "team_id"})
        .merge(profile_keys, on=["season", "team_id"], how="left", indicator=True)
        .loc[lambda df: df["_merge"] == "left_only", ["season", "team_id"]]
        .drop_duplicates()
    )

    missing_teamA = list(missing_teamA_df.head(5).itertuples(index=False, name=None))
    missing_teamB = list(missing_teamB_df.head(5).itertuples(index=False, name=None))

    if not missing_teamA and not missing_teamB:
        return

    message = (
        "Tournament matchup rows are missing team profile joins. "
        f"Missing teamA keys sample: {missing_teamA} | "
        f"Missing teamB keys sample: {missing_teamB}"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _filter_regular_season_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort filtering to avoid using tournament/postseason games."""

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
    """Build supplemental team-season features from games boxscores."""

    games = games_boxscores_df.copy()

    team_col_candidates = ["team_id", "TeamID", "teamID"]
    team_col = next((col for col in team_col_candidates if col in games.columns), None)
    if team_col is None:
        warnings.warn(
            "games_boxscores_df has no team_id-like column; skipping supplemental joins.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["season", "team_id"])

    if team_col != "team_id":
        games = games.rename(columns={team_col: "team_id"})

    _require_columns(games, ["season", "team_id"], "games_boxscores_df")

    games = _filter_regular_season_games(games)

    candidate_numeric_cols = [
        column
        for column in games.select_dtypes(include=[np.number]).columns
        if column not in {"season", "team_id", "opponent_team_id", "opp_team_id", "game_id"}
    ]

    supplemental_numeric_cols = [
        column for column in candidate_numeric_cols if column not in set(existing_team_feature_columns)
    ]

    if not supplemental_numeric_cols:
        return pd.DataFrame(columns=["season", "team_id"])

    return (
        games.groupby(["season", "team_id"], as_index=False)[supplemental_numeric_cols]
        .mean(numeric_only=True)
        .rename(columns={column: f"{BOX_FEATURE_PREFIX}{column}" for column in supplemental_numeric_cols})
    )


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
        column: f"{team_prefix}_{column}"
        for column in side_features.columns
        if column not in {"season", team_col_in_matchups}
    }
    side_features = side_features.rename(columns=rename_cols)

    return matchups_df.merge(
        side_features,
        on=["season", team_col_in_matchups],
        how="left",
        validate="many_to_one",
    )


def _build_diff_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    for diff_col, base_col in DIFF_COLUMN_MAPPING.items():
        teamA_col = f"teamA_{base_col}"
        teamB_col = f"teamB_{base_col}"
        if teamA_col in modeling_df.columns and teamB_col in modeling_df.columns:
            modeling_df[diff_col] = modeling_df[teamA_col] - modeling_df[teamB_col]
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
            bins=[-np.inf, 2, 5, np.inf],
            labels=[0, 1, 2],
        ).astype(float)

    if "teamA_seed" in modeling_df.columns and "teamB_seed" in modeling_df.columns:
        modeling_df["upset_bucket"] = (modeling_df["teamA_seed"] > modeling_df["teamB_seed"]).astype(int)

    return modeling_df


def build_modeling_dataframe(
    team_profiles_df: pd.DataFrame,
    tourney_matchups_df: pd.DataFrame,
    games_boxscores_df: pd.DataFrame | None = None,
    include_raw_team_features: bool = True,
    include_diff_features: bool = True,
    include_context_features: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Build the final matchup-level modeling dataframe.

    TODO: Add richer matchup feature importance diagnostics.
    TODO: Integrate bracket-simulation feature handoff for downstream runs.
    """

    team_profiles = _coerce_team_profiles(team_profiles_df)
    _validate_unique_team_profiles(team_profiles, strict=strict)

    matchups = tourney_matchups_df.copy()
    _require_columns(matchups, REQUIRED_MATCHUP_COLUMNS, "tourney_matchups_df")

    matchups["season"] = matchups["season"].astype(int)
    matchups["teamA_id"] = matchups["teamA_id"].astype(int)
    matchups["teamB_id"] = matchups["teamB_id"].astype(int)

    _validate_matchup_joins(matchups, team_profiles, strict=strict)

    modeling_df = _merge_team_features(
        matchups,
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

    if games_boxscores_df is not None and not games_boxscores_df.empty:
        team_feature_cols = [column for column in team_profiles.columns if column not in {"season", "team_id"}]
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

    if include_diff_features:
        modeling_df = _build_diff_features(modeling_df)

    if include_context_features:
        modeling_df = _build_context_features(modeling_df)

    if not include_raw_team_features:
        raw_cols = [
            column
            for column in modeling_df.columns
            if column.startswith("teamA_") or column.startswith("teamB_")
        ]
        modeling_df = modeling_df.drop(columns=raw_cols)

    metadata_priority = [
        "season",
        "round",
        "teamA_id",
        "teamB_id",
        TARGET_COL,
        "teamA_seed",
        "teamB_seed",
        "region",
    ]
    front_cols = [column for column in metadata_priority if column in modeling_df.columns]
    other_cols = [column for column in modeling_df.columns if column not in front_cols]

    # Never silently drop matchup rows.
    if len(modeling_df) != len(matchups):
        raise RuntimeError(
            "Row count changed during modeling dataframe assembly, which indicates an unexpected drop/duplication."
        )

    return modeling_df[front_cols + other_cols]
