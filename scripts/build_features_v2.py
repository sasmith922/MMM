"""
build_features_v2.py
--------------------
Build a no-leakage v2 feature-engineering dataset in separate output folders.

This script intentionally leaves the baseline pipeline untouched and writes only:

- data/processed_v2/team_features_v2.csv
- data/processed_v2/tournament_matchups_v2.csv

Usage
-----
    python scripts/build_features_v2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from madness_model.load_processed_data import load_all_processed_data
from madness_model.paths import PROJECT_ROOT


PROCESSED_V2_DIR = PROJECT_ROOT / "data" / "processed_v2"
REPORTS_V2_DIR = PROJECT_ROOT / "outputs" / "reports_v2"
TEAM_FEATURES_V2_PATH = PROCESSED_V2_DIR / "team_features_v2.csv"
TOURNEY_MATCHUPS_V2_PATH = PROCESSED_V2_DIR / "tournament_matchups_v2.csv"


TEAM_FEATURE_COLUMNS = [
    "season",
    "team_id",
    "seed",
    "overall_win_pct",
    "conference_win_pct",
    "neutral_win_pct",
    "away_win_pct",
    "last_5_win_pct",
    "last_10_win_pct",
    "avg_margin",
    "std_margin",
    "avg_off_eff",
    "avg_def_eff",
    "std_off_eff",
    "std_def_eff",
    "top25_win_pct",
    "top50_win_pct",
    "close_game_win_pct",
    "blowout_win_pct",
    "sos",
    "sor",
    "tempo",
    "experience",
    "net_rating",
    "elo_pre_tourney",
    "elo_rank",
    "resume_delta",
    "bad_loss_rate",
    "nonconf_win_pct",
    "games_played",
]


MATCHUP_COLUMNS = [
    "season",
    "game_id",
    "round",
    "region",
    "team1_id",
    "team2_id",
    "label",
    "seed_diff",
    "seed_abs_diff",
    "adj_o_diff",
    "adj_o_abs_diff",
    "adj_d_diff",
    "adj_d_abs_diff",
    "net_rating_diff",
    "net_rating_abs_diff",
    "sos_diff",
    "sor_diff",
    "tempo_diff",
    "experience_diff",
    "top25_win_diff",
    "top50_win_diff",
    "recent_form_diff",
    "volatility_diff",
    "neutral_win_diff",
    "close_game_diff",
    "blowout_diff",
    "seed_x_net_interaction",
    "upset_risk_score",
]


def _safe_div(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    """Safe division that returns NaN for divide-by-zero rows."""
    num = pd.Series(numerator)
    den = pd.Series(denominator)
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out


def _first_existing(columns: Iterable[str], options: list[str]) -> str | None:
    """Return first matching column from options, else None."""
    col_set = set(columns)
    for option in options:
        if option in col_set:
            return option
    return None


def _normalize_team_profiles(team_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key team profile columns used by v2 feature engineering."""
    df = team_profiles_df.copy()
    if "team_id" not in df.columns and "kaggle_team_id" in df.columns:
        df = df.rename(columns={"kaggle_team_id": "team_id"})

    profile_rename_map = {
        "win_pct": "profile_win_pct",
        "pre_tourney_adjoe": "profile_adjoe",
        "pre_tourney_adjde": "profile_adjde",
        "pre_tourney_tempo": "profile_tempo",
        "seed": "profile_seed",
        "sos_elo": "profile_sos",
        "resume_delta": "profile_sor",
    }
    for old_col, new_col in profile_rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})

    return df


def _compute_possessions(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Compute approximate possessions from box-score columns.

    Formula: FGA - OR + TO + 0.475 * FTA
    """
    fga_col = _first_existing(df.columns, [f"{prefix}_fga", "field_goals_attempted"])
    or_col = _first_existing(df.columns, [f"{prefix}_or", "offensive_rebounds"])
    to_col = _first_existing(df.columns, [f"{prefix}_to", "turnovers"])
    fta_col = _first_existing(df.columns, [f"{prefix}_fta", "free_throws_attempted"])
    if not all([fga_col, or_col, to_col, fta_col]):
        return pd.Series(np.nan, index=df.index, dtype=float)

    possessions = (
        pd.to_numeric(df[fga_col], errors="coerce")
        - pd.to_numeric(df[or_col], errors="coerce")
        + pd.to_numeric(df[to_col], errors="coerce")
        + (0.475 * pd.to_numeric(df[fta_col], errors="coerce"))
    )
    return possessions


def _build_regular_season_team_games(
    games_boxscores_df: pd.DataFrame,
    tourney_matchups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a regular-season team-game table used for season feature aggregation.

    Leakage guard:
    - Uses regular-season rows only by season_type when available.
    - If day-number columns exist in both game tables, also removes games
      at/after first tournament day for each season.
    """
    games = games_boxscores_df.copy()

    if "team_id" not in games.columns:
        raise KeyError("games_boxscores is missing required column: team_id")

    if "season_type" in games.columns:
        season_type = games["season_type"].astype(str).str.lower()
        is_tourney_like = season_type.str.contains("post|tournament|ncaa", na=False)
        games = games[~is_tourney_like].copy()

    # Optional hard leakage cutoff by day number if both datasets support it.
    game_day_col = _first_existing(games.columns, ["daynum", "day_num"])
    tourney_day_col = _first_existing(tourney_matchups_df.columns, ["daynum", "day_num"])
    if game_day_col and tourney_day_col and "season" in tourney_matchups_df.columns:
        min_tourney_day = (
            tourney_matchups_df.groupby("season")[tourney_day_col]
            .min()
            .rename("min_tourney_day")
            .reset_index()
        )
        pre_join_rows = len(games)
        games = games.merge(min_tourney_day, on="season", how="left")
        games = games[
            games["min_tourney_day"].isna()
            | (pd.to_numeric(games[game_day_col], errors="coerce") < games["min_tourney_day"])
        ].drop(columns=["min_tourney_day"])
        print(f"[leakage-check] Removed {pre_join_rows - len(games)} potential post-tournament rows by day cutoff.")

    # Build ordering column used for recent-form windows.
    if game_day_col:
        games["sort_key"] = pd.to_numeric(games[game_day_col], errors="coerce")
    elif "game_date" in games.columns:
        games["sort_key"] = pd.to_datetime(games["game_date"], errors="coerce")
    else:
        games["sort_key"] = np.arange(len(games))

    # Standardize win/margin/home-away flags.
    if "win" not in games.columns and "team_winner" in games.columns:
        games["win"] = pd.to_numeric(games["team_winner"], errors="coerce")
    games["win"] = pd.to_numeric(games.get("win", np.nan), errors="coerce")

    if "score_margin" not in games.columns:
        score_col = _first_existing(games.columns, ["team_score", "score"])
        opp_score_col = _first_existing(games.columns, ["opponent_team_score", "opp_score"])
        if score_col and opp_score_col:
            games["score_margin"] = pd.to_numeric(games[score_col], errors="coerce") - pd.to_numeric(
                games[opp_score_col], errors="coerce"
            )
        else:
            games["score_margin"] = np.nan
    games["score_margin"] = pd.to_numeric(games["score_margin"], errors="coerce")

    if "is_away" not in games.columns and "team_home_away" in games.columns:
        loc = games["team_home_away"].astype(str).str.lower()
        games["is_away"] = (loc == "away").astype(int)
    if "is_neutral" not in games.columns and "team_home_away" in games.columns:
        loc = games["team_home_away"].astype(str).str.lower()
        games["is_neutral"] = (loc == "neutral").astype(int)

    games["is_away"] = pd.to_numeric(games.get("is_away", 0), errors="coerce").fillna(0)
    games["is_neutral"] = pd.to_numeric(games.get("is_neutral", 0), errors="coerce").fillna(0)

    # Build per-game off/def efficiencies (if enough box-score columns exist).
    team_points_col = _first_existing(games.columns, ["team_score", "score"])
    opp_points_col = _first_existing(games.columns, ["opponent_team_score", "opp_score"])
    team_poss = _compute_possessions(games, "team")
    opp_poss = _compute_possessions(games, "opponent")
    poss = np.nanmean(np.vstack([team_poss, opp_poss]), axis=0)

    if team_points_col:
        games["off_eff"] = 100.0 * _safe_div(pd.to_numeric(games[team_points_col], errors="coerce"), poss)
    else:
        games["off_eff"] = np.nan

    if opp_points_col:
        games["def_eff"] = 100.0 * _safe_div(pd.to_numeric(games[opp_points_col], errors="coerce"), poss)
    else:
        games["def_eff"] = np.nan

    return games


def _window_win_pct(group: pd.DataFrame, window: int) -> float:
    """Win percentage over last N games for one team-season group."""
    if group.empty:
        return np.nan
    ordered = group.sort_values("sort_key")
    last_n = ordered.tail(window)
    return float(pd.to_numeric(last_n["win"], errors="coerce").mean())


def build_team_season_features(
    team_profiles_df: pd.DataFrame,
    games_boxscores_df: pd.DataFrame,
    tourney_matchups_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per (season, team_id) using pre-tournament information only."""
    profiles = _normalize_team_profiles(team_profiles_df)
    regular_games = _build_regular_season_team_games(games_boxscores_df, tourney_matchups_df)

    required_cols = {"season", "team_id"}
    missing_required = [c for c in required_cols if c not in profiles.columns]
    if missing_required:
        raise KeyError(f"team_profiles is missing required columns: {missing_required}")

    by_team = regular_games.groupby(["season", "team_id"], dropna=False)
    agg = by_team.agg(
        games_played=("win", "count"),
        game_win_pct=("win", "mean"),
        avg_margin_game=("score_margin", "mean"),
        std_margin=("score_margin", "std"),
        avg_off_eff_game=("off_eff", "mean"),
        avg_def_eff_game=("def_eff", "mean"),
        std_off_eff=("off_eff", "std"),
        std_def_eff=("def_eff", "std"),
    ).reset_index()

    # Location-specific win percentages.
    away = regular_games[regular_games["is_away"] == 1]
    neutral = regular_games[regular_games["is_neutral"] == 1]
    away_pct = (
        away.groupby(["season", "team_id"], dropna=False)["win"]
        .mean()
        .rename("away_win_pct")
        .reset_index()
    )
    neutral_pct = (
        neutral.groupby(["season", "team_id"], dropna=False)["win"]
        .mean()
        .rename("neutral_win_pct")
        .reset_index()
    )

    # Recent form.
    recent_5 = by_team.apply(_window_win_pct, window=5).rename("last_5_win_pct").reset_index()
    recent_10 = by_team.apply(_window_win_pct, window=10).rename("last_10_win_pct").reset_index()

    # Outcome-bucket win percentages.
    close_games = regular_games[regular_games["score_margin"].abs() <= 5]
    blowout_games = regular_games[regular_games["score_margin"] >= 15]
    close_pct = (
        close_games.groupby(["season", "team_id"], dropna=False)["win"]
        .mean()
        .rename("close_game_win_pct")
        .reset_index()
    )
    blowout_pct = (
        blowout_games.groupby(["season", "team_id"], dropna=False)["win"]
        .mean()
        .rename("blowout_win_pct")
        .reset_index()
    )

    feature_df = profiles.copy()
    feature_df = feature_df.merge(agg, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(away_pct, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(neutral_pct, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(recent_5, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(recent_10, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(close_pct, on=["season", "team_id"], how="left")
    feature_df = feature_df.merge(blowout_pct, on=["season", "team_id"], how="left")

    # Use profile-level stats where available; otherwise backfill from game-level aggregates.
    feature_df["seed"] = pd.to_numeric(feature_df.get("profile_seed"), errors="coerce")
    feature_df["overall_win_pct"] = pd.to_numeric(feature_df.get("profile_win_pct"), errors="coerce").combine_first(
        pd.to_numeric(feature_df.get("game_win_pct"), errors="coerce")
    )
    feature_df["conference_win_pct"] = _safe_div(
        pd.to_numeric(feature_df.get("conference_wins"), errors="coerce"),
        pd.to_numeric(feature_df.get("conference_wins"), errors="coerce")
        + pd.to_numeric(feature_df.get("conference_losses"), errors="coerce"),
    )
    feature_df["neutral_win_pct"] = pd.to_numeric(feature_df.get("neutral_win_pct"), errors="coerce").combine_first(
        _safe_div(
            pd.to_numeric(feature_df.get("neutral_wins"), errors="coerce"),
            pd.to_numeric(feature_df.get("neutral_wins"), errors="coerce")
            + pd.to_numeric(feature_df.get("neutral_losses"), errors="coerce"),
        )
    )
    feature_df["away_win_pct"] = pd.to_numeric(feature_df.get("away_win_pct"), errors="coerce").combine_first(
        _safe_div(
            pd.to_numeric(feature_df.get("away_wins"), errors="coerce"),
            pd.to_numeric(feature_df.get("away_wins"), errors="coerce")
            + pd.to_numeric(feature_df.get("away_losses"), errors="coerce"),
        )
    )
    feature_df["avg_margin"] = pd.to_numeric(feature_df.get("avg_margin"), errors="coerce").combine_first(
        pd.to_numeric(feature_df.get("avg_margin_game"), errors="coerce")
    )
    feature_df["avg_off_eff"] = pd.to_numeric(feature_df.get("profile_adjoe"), errors="coerce").combine_first(
        pd.to_numeric(feature_df.get("avg_off_eff_game"), errors="coerce")
    )
    feature_df["avg_def_eff"] = pd.to_numeric(feature_df.get("profile_adjde"), errors="coerce").combine_first(
        pd.to_numeric(feature_df.get("avg_def_eff_game"), errors="coerce")
    )

    wins = pd.to_numeric(feature_df.get("wins"), errors="coerce")
    feature_df["top25_win_pct"] = _safe_div(pd.to_numeric(feature_df.get("top25_wins"), errors="coerce"), wins)
    feature_df["top50_win_pct"] = _safe_div(pd.to_numeric(feature_df.get("top50_wins"), errors="coerce"), wins)
    feature_df["sos"] = pd.to_numeric(feature_df.get("profile_sos"), errors="coerce")
    feature_df["sor"] = pd.to_numeric(feature_df.get("profile_sor"), errors="coerce")
    feature_df["tempo"] = pd.to_numeric(feature_df.get("profile_tempo"), errors="coerce")
    feature_df["experience"] = pd.to_numeric(feature_df.get("experience"), errors="coerce")
    feature_df["net_rating"] = pd.to_numeric(feature_df.get("net_rating"), errors="coerce")
    feature_df["elo_pre_tourney"] = pd.to_numeric(feature_df.get("elo_pre_tourney"), errors="coerce")
    feature_df["elo_rank"] = pd.to_numeric(feature_df.get("elo_rank"), errors="coerce")
    feature_df["resume_delta"] = pd.to_numeric(feature_df.get("resume_delta"), errors="coerce")
    feature_df["bad_loss_rate"] = _safe_div(
        pd.to_numeric(feature_df.get("bad_losses_100plus"), errors="coerce"),
        pd.to_numeric(feature_df.get("games_played"), errors="coerce"),
    )
    feature_df["nonconf_win_pct"] = _safe_div(
        pd.to_numeric(feature_df.get("nonconf_wins"), errors="coerce"),
        pd.to_numeric(feature_df.get("nonconf_wins"), errors="coerce")
        + pd.to_numeric(feature_df.get("nonconf_losses"), errors="coerce"),
    )

    # Stable output schema.
    for col in TEAM_FEATURE_COLUMNS:
        if col not in feature_df.columns:
            feature_df[col] = np.nan
    feature_df = feature_df[TEAM_FEATURE_COLUMNS].copy()
    feature_df = feature_df.drop_duplicates(subset=["season", "team_id"], keep="first").reset_index(drop=True)

    return feature_df


def _canonical_tourney_games(tourney_matchups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize tournament games to one row per game.

    The processed dataset usually contains mirrored rows (winner and loser
    perspectives). We select winner rows and then sort teams deterministically.
    """
    df = tourney_matchups_df.copy()
    if "target" in df.columns:
        winner_rows = df[df["target"] == 1].copy()
    else:
        winner_rows = df.copy()

    if "team_a_id" not in winner_rows.columns or "team_b_id" not in winner_rows.columns:
        raise KeyError("tourney_matchups is missing team_a_id/team_b_id columns.")

    winner_rows["winner_id"] = pd.to_numeric(winner_rows["team_a_id"], errors="coerce")
    winner_rows["loser_id"] = pd.to_numeric(winner_rows["team_b_id"], errors="coerce")
    winner_rows["team1_id"] = winner_rows[["winner_id", "loser_id"]].min(axis=1).astype("Int64")
    winner_rows["team2_id"] = winner_rows[["winner_id", "loser_id"]].max(axis=1).astype("Int64")
    winner_rows["label"] = (winner_rows["winner_id"] == winner_rows["team1_id"]).astype(int)

    # Standard metadata columns.
    if "game_id" not in winner_rows.columns:
        winner_rows["game_id"] = (
            winner_rows["season"].astype(str)
            + "_"
            + winner_rows.get("daynum", pd.Series(index=winner_rows.index, dtype=object)).astype(str)
            + "_"
            + winner_rows["team1_id"].astype(str)
            + "_"
            + winner_rows["team2_id"].astype(str)
        )

    if "round" not in winner_rows.columns:
        winner_rows["round"] = winner_rows.get("round_num_guess", np.nan)
    if "region" not in winner_rows.columns:
        winner_rows["region"] = np.nan

    dedupe_cols = ["season", "team1_id", "team2_id"]
    winner_rows = winner_rows.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
    return winner_rows


def build_tournament_matchups(team_features_df: pd.DataFrame, tourney_matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per tournament matchup with v2 differential features."""
    games = _canonical_tourney_games(tourney_matchups_df)

    feat = team_features_df.copy()
    team1_cols = {c: f"team1_{c}" for c in feat.columns if c not in ["season", "team_id"]}
    team2_cols = {c: f"team2_{c}" for c in feat.columns if c not in ["season", "team_id"]}

    games = games.merge(
        feat.rename(columns={"team_id": "team1_id", **team1_cols}),
        on=["season", "team1_id"],
        how="left",
    )
    games = games.merge(
        feat.rename(columns={"team_id": "team2_id", **team2_cols}),
        on=["season", "team2_id"],
        how="left",
    )

    def diff(col: str) -> pd.Series:
        return pd.to_numeric(games.get(f"team1_{col}"), errors="coerce") - pd.to_numeric(
            games.get(f"team2_{col}"), errors="coerce"
        )

    games["seed_diff"] = diff("seed")
    games["seed_abs_diff"] = games["seed_diff"].abs()
    games["adj_o_diff"] = diff("avg_off_eff")
    games["adj_o_abs_diff"] = games["adj_o_diff"].abs()
    games["adj_d_diff"] = diff("avg_def_eff")
    games["adj_d_abs_diff"] = games["adj_d_diff"].abs()
    games["net_rating_diff"] = diff("net_rating")
    games["net_rating_abs_diff"] = games["net_rating_diff"].abs()
    games["sos_diff"] = diff("sos")
    games["sor_diff"] = diff("sor")
    games["tempo_diff"] = diff("tempo")
    games["experience_diff"] = diff("experience")
    games["top25_win_diff"] = diff("top25_win_pct")
    games["top50_win_diff"] = diff("top50_win_pct")
    games["recent_form_diff"] = diff("last_10_win_pct")
    games["volatility_diff"] = diff("std_margin")
    games["neutral_win_diff"] = diff("neutral_win_pct")
    games["close_game_diff"] = diff("close_game_win_pct")
    games["blowout_diff"] = diff("blowout_win_pct")

    # Simple interaction / upset features.
    games["seed_x_net_interaction"] = games["seed_diff"] * games["net_rating_diff"]
    games["upset_risk_score"] = games["seed_abs_diff"] * games["volatility_diff"].abs()

    # Stable schema.
    for col in MATCHUP_COLUMNS:
        if col not in games.columns:
            games[col] = np.nan

    matchups = games[MATCHUP_COLUMNS].copy()
    matchups = matchups.drop_duplicates(subset=["season", "team1_id", "team2_id"], keep="first").reset_index(
        drop=True
    )
    return matchups


def _print_missingness_summary(df: pd.DataFrame, name: str) -> None:
    """Print concise missingness summary for debugging data quality."""
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    print(f"\n[{name}] shape={df.shape}")
    print(f"[{name}] top missingness (%):")
    print(missing_pct.head(15).to_string())


def _run_validations(
    team_features_df: pd.DataFrame,
    tournament_matchups_df: pd.DataFrame,
    regular_games_df: pd.DataFrame,
    tourney_matchups_df: pd.DataFrame,
) -> None:
    """Run safety checks for leakage, duplicates, and key missingness."""
    print("\n=== V2 DATA VALIDATION ===")

    # Leakage check by day cutoff where available.
    game_day_col = _first_existing(regular_games_df.columns, ["daynum", "day_num"])
    tourney_day_col = _first_existing(tourney_matchups_df.columns, ["daynum", "day_num"])
    if game_day_col and tourney_day_col:
        min_tourney_day = tourney_matchups_df.groupby("season")[tourney_day_col].min()
        check_df = regular_games_df.merge(min_tourney_day.rename("min_tourney_day"), on="season", how="left")
        leaked_rows = check_df[
            check_df["min_tourney_day"].notna()
            & (pd.to_numeric(check_df[game_day_col], errors="coerce") >= check_df["min_tourney_day"])
        ]
        print(f"[leakage-check] post-tournament rows in regular input: {len(leaked_rows)}")
    else:
        print("[leakage-check] skipped day-based check (missing day columns).")

    team_dupes = team_features_df.duplicated(subset=["season", "team_id"]).sum()
    matchup_dupes = tournament_matchups_df.duplicated(subset=["season", "team1_id", "team2_id"]).sum()
    print(f"[dup-check] duplicate season-team rows: {team_dupes}")
    print(f"[dup-check] duplicate matchup rows: {matchup_dupes}")

    key_team_cols = ["seed", "overall_win_pct", "avg_off_eff", "avg_def_eff"]
    key_matchup_cols = ["seed_diff", "adj_o_diff", "adj_d_diff", "label"]
    print("[missing-check] team key feature missing counts:")
    print(team_features_df[key_team_cols].isna().sum().to_string())
    print("[missing-check] matchup key feature missing counts:")
    print(tournament_matchups_df[key_matchup_cols].isna().sum().to_string())

    _print_missingness_summary(team_features_df, "team_features_v2")
    _print_missingness_summary(tournament_matchups_df, "tournament_matchups_v2")

    print("\n[sample] team_features_v2 preview:")
    print(team_features_df.head(5).to_string(index=False))
    print("\n[sample] tournament_matchups_v2 preview:")
    print(tournament_matchups_df.head(5).to_string(index=False))


def _ensure_v2_directories() -> None:
    """Create v2 output directories if they do not exist."""
    PROCESSED_V2_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_V2_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Build and save v2 team features and tournament matchup datasets."""
    _ensure_v2_directories()
    data = load_all_processed_data()

    team_profiles = data["team_profiles"]
    games_boxscores = data["games_boxscores"]
    tourney_matchups = data["tourney_matchups"]

    regular_games = _build_regular_season_team_games(games_boxscores, tourney_matchups)
    team_features_v2 = build_team_season_features(team_profiles, games_boxscores, tourney_matchups)
    tournament_matchups_v2 = build_tournament_matchups(team_features_v2, tourney_matchups)

    _run_validations(team_features_v2, tournament_matchups_v2, regular_games, tourney_matchups)

    team_features_v2.to_csv(TEAM_FEATURES_V2_PATH, index=False)
    tournament_matchups_v2.to_csv(TOURNEY_MATCHUPS_V2_PATH, index=False)

    print(f"\nSaved: {TEAM_FEATURES_V2_PATH}")
    print(f"Saved: {TOURNEY_MATCHUPS_V2_PATH}")


if __name__ == "__main__":
    main()
