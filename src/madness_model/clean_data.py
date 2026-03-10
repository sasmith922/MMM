"""
clean_data.py
-------------
Functions for cleaning and validating raw team, game, and seed data.

Cleaning steps typically include:
- Dropping duplicates and null-heavy rows
- Standardising column names
- Parsing seed strings into numeric values
- Filtering to valid seasons
"""

from __future__ import annotations

from typing import List

import pandas as pd


def clean_teams(teams: pd.DataFrame) -> pd.DataFrame:
    """Clean the teams DataFrame.

    Parameters
    ----------
    teams:
        Raw output of :func:`~madness_model.load_data.load_teams`.

    Returns
    -------
    pd.DataFrame
        Deduplicated, null-free team mapping.
    """
    # TODO: standardise column names to snake_case
    df = teams.drop_duplicates().dropna(subset=["team_id"])
    df["team_id"] = df["team_id"].astype(int)
    return df.reset_index(drop=True)


def clean_game_results(games: pd.DataFrame) -> pd.DataFrame:
    """Clean a game-results DataFrame (regular season or tournament).

    Parameters
    ----------
    games:
        Raw output of :func:`~madness_model.load_data.load_regular_season`
        or :func:`~madness_model.load_data.load_tourney_results`.

    Returns
    -------
    pd.DataFrame
        Cleaned game results with consistent dtypes.
    """
    df = games.copy()
    # TODO: handle overtime columns gracefully when absent
    df = df.dropna(subset=["season", "w_team_id", "l_team_id"])
    int_cols = ["season", "w_team_id", "l_team_id", "w_score", "l_score"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def parse_seed(seed_str: str) -> int:
    """Extract the integer seed number from a seed string such as ``"W01"``.

    Parameters
    ----------
    seed_str:
        Raw seed string, e.g. ``"W01"``, ``"X16a"``.

    Returns
    -------
    int
        Numeric seed value (1–16).
    """
    # Strip region letter prefix and optional play-in suffix (a/b)
    numeric = "".join(ch for ch in seed_str if ch.isdigit())
    # TODO: handle edge cases like empty strings
    return int(numeric)


def clean_seeds(seeds: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the seeds DataFrame.

    Converts seed strings (e.g. ``"W01"``) to integer seed numbers and
    extracts the region character.

    Parameters
    ----------
    seeds:
        Raw output of :func:`~madness_model.load_data.load_seeds`.

    Returns
    -------
    pd.DataFrame
        Columns: ``season``, ``team_id``, ``seed_str``, ``seed`` (int),
        ``region`` (str).
    """
    df = seeds.copy()
    df = df.dropna(subset=["season", "seed", "team_id"])
    df = df.rename(columns={"seed": "seed_str"})
    df["seed"] = df["seed_str"].apply(parse_seed)
    df["region"] = df["seed_str"].str[0]
    df["team_id"] = df["team_id"].astype(int)
    df["season"] = df["season"].astype(int)
    return df.reset_index(drop=True)


def filter_seasons(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Keep only rows belonging to the specified seasons.

    Parameters
    ----------
    df:
        Any DataFrame that contains a ``season`` column.
    seasons:
        List of integer season years to retain.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[df["season"].isin(seasons)].reset_index(drop=True)
