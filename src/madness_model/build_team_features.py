"""
build_team_features.py
----------------------
Aggregate regular-season statistics into a single row of season-end team
features.  These features become the inputs to the matchup model.

Typical features include:
- Win percentage
- Average point differential
- Strength of schedule (TODO)
- Elo rating at end of season (computed separately in elo.py)
"""

from __future__ import annotations

import pandas as pd


def compute_win_pct(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team win percentage for each season.

    Parameters
    ----------
    games:
        Cleaned regular-season game results with columns
        ``season``, ``w_team_id``, ``l_team_id``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season``, ``team_id``, ``wins``, ``losses``,
        ``win_pct``.
    """
    # Count wins
    wins = (
        games.groupby(["season", "w_team_id"])
        .size()
        .reset_index(name="wins")
        .rename(columns={"w_team_id": "team_id"})
    )
    # Count losses
    losses = (
        games.groupby(["season", "l_team_id"])
        .size()
        .reset_index(name="losses")
        .rename(columns={"l_team_id": "team_id"})
    )
    df = wins.merge(losses, on=["season", "team_id"], how="outer").fillna(0)
    df["wins"] = df["wins"].astype(int)
    df["losses"] = df["losses"].astype(int)
    df["win_pct"] = df["wins"] / (df["wins"] + df["losses"])
    return df


def compute_avg_point_diff(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team average scoring margin for each season.

    Parameters
    ----------
    games:
        Cleaned regular-season game results with columns
        ``season``, ``w_team_id``, ``l_team_id``, ``w_score``, ``l_score``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season``, ``team_id``, ``avg_point_diff``.
    """
    # Margin from the winner's perspective
    w = games[["season", "w_team_id", "w_score", "l_score"]].copy()
    w = w.rename(columns={"w_team_id": "team_id"})
    w["margin"] = w["w_score"] - w["l_score"]

    # Margin from the loser's perspective (negative)
    l = games[["season", "l_team_id", "w_score", "l_score"]].copy()
    l = l.rename(columns={"l_team_id": "team_id"})
    l["margin"] = l["l_score"] - l["w_score"]

    combined = pd.concat([w[["season", "team_id", "margin"]], l[["season", "team_id", "margin"]]])
    avg_diff = (
        combined.groupby(["season", "team_id"])["margin"]
        .mean()
        .reset_index(name="avg_point_diff")
    )
    return avg_diff


def build_team_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build a full season-end feature table for all teams.

    Combines win percentage, average point differential, and
    (TODO) other features into one DataFrame.

    Parameters
    ----------
    games:
        Cleaned regular-season game results.

    Returns
    -------
    pd.DataFrame
        One row per (season, team_id) with all computed features.
        Index: ``season``, ``team_id``.
    """
    win_pct = compute_win_pct(games)
    avg_diff = compute_avg_point_diff(games)

    features = win_pct.merge(avg_diff, on=["season", "team_id"], how="left")

    # TODO: merge in Elo ratings from elo.py
    # TODO: merge in advanced stats (offensive/defensive efficiency)
    # TODO: merge in strength-of-schedule metric

    return features.set_index(["season", "team_id"])
