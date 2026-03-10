"""
load_data.py
------------
Functions for loading raw CSV and Parquet data files into pandas DataFrames.

Expected raw files (place in data/raw/):
- teams.csv          : team ID ↔ name mapping
- seasons.csv        : season metadata (year, region dates, etc.)
- regular_season.csv : regular-season game results
- tourney_results.csv: tournament game results
- seeds.csv          : tournament seedings per team per season
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from madness_model.paths import RAW_DATA_DIR


def load_teams(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the team ID ↔ name mapping.

    Parameters
    ----------
    path:
        Override the default file location.  Defaults to
        ``data/raw/teams.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``team_id`` (int), ``team_name`` (str).
    """
    file = path or RAW_DATA_DIR / "teams.csv"
    # TODO: validate expected columns after loading
    return pd.read_csv(file)


def load_seasons(path: Optional[Path] = None) -> pd.DataFrame:
    """Load season metadata.

    Parameters
    ----------
    path:
        Override the default file location.  Defaults to
        ``data/raw/seasons.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season`` (int), plus regional/date metadata.
    """
    file = path or RAW_DATA_DIR / "seasons.csv"
    # TODO: parse date columns as datetime objects
    return pd.read_csv(file)


def load_regular_season(path: Optional[Path] = None) -> pd.DataFrame:
    """Load regular-season game results.

    Parameters
    ----------
    path:
        Override the default file location.  Defaults to
        ``data/raw/regular_season.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season``, ``day_num``, ``w_team_id``, ``w_score``,
        ``l_team_id``, ``l_score``, ``w_loc``, ``num_ot``.
    """
    file = path or RAW_DATA_DIR / "regular_season.csv"
    # TODO: add dtype mapping for efficiency on large files
    return pd.read_csv(file)


def load_tourney_results(path: Optional[Path] = None) -> pd.DataFrame:
    """Load NCAA tournament game results.

    Parameters
    ----------
    path:
        Override the default file location.  Defaults to
        ``data/raw/tourney_results.csv``.

    Returns
    -------
    pd.DataFrame
        Same schema as :func:`load_regular_season`.
    """
    file = path or RAW_DATA_DIR / "tourney_results.csv"
    return pd.read_csv(file)


def load_seeds(path: Optional[Path] = None) -> pd.DataFrame:
    """Load tournament seedings.

    Parameters
    ----------
    path:
        Override the default file location.  Defaults to
        ``data/raw/seeds.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season`` (int), ``seed`` (str, e.g. ``"W01"``),
        ``team_id`` (int).
    """
    file = path or RAW_DATA_DIR / "seeds.csv"
    return pd.read_csv(file)


def load_parquet(path: Path) -> pd.DataFrame:
    """Generic helper to load any Parquet file.

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
    """
    # TODO: support reading from cloud storage (S3, GCS) via fsspec
    return pd.read_parquet(path)
