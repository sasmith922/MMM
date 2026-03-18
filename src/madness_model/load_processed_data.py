"""Helpers for loading processed CSV feature sources for modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from madness_model.paths import PROCESSED_DATA_DIR


DEFAULT_PROCESSED_FILES = {
    "team_profiles": "team_profiles.csv",
    "games_boxscores": "games_boxscores.csv",
    "tourney_matchups": "tourney_matchups.csv",
}


def _load_processed_csv(
    file_name: str,
    *,
    processed_dir: Path = PROCESSED_DATA_DIR,
    required: bool = True,
) -> pd.DataFrame | None:
    """Load a processed CSV by file name from the processed data directory."""
    path = processed_dir / file_name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required processed CSV not found: {path}")
        return None
    return pd.read_csv(path)


def load_team_profiles(*, required: bool = True) -> pd.DataFrame | None:
    """Load the team season profile table."""
    return _load_processed_csv(DEFAULT_PROCESSED_FILES["team_profiles"], required=required)


def load_games_boxscores(*, required: bool = False) -> pd.DataFrame | None:
    """Load the processed games/boxscore table used for feature enrichment."""
    return _load_processed_csv(DEFAULT_PROCESSED_FILES["games_boxscores"], required=required)


def load_tourney_matchups(*, required: bool = True) -> pd.DataFrame | None:
    """Load the supervised historical tournament matchup table."""
    return _load_processed_csv(DEFAULT_PROCESSED_FILES["tourney_matchups"], required=required)


def load_all_processed_data(
    *,
    include_additional_csvs: bool = True,
) -> Dict[str, pd.DataFrame | Dict[str, pd.DataFrame] | None]:
    """Load all primary processed feature sources.

    Returns a dict with required modular sources:
    - ``team_profiles``
    - ``games_boxscores``
    - ``tourney_matchups``

    If ``include_additional_csvs`` is True, also returns ``additional_sources``
    containing any extra processed CSVs keyed by stem for future extensibility.
    """
    data: Dict[str, pd.DataFrame | Dict[str, pd.DataFrame] | None] = {
        "team_profiles": load_team_profiles(required=True),
        "games_boxscores": load_games_boxscores(required=False),
        "tourney_matchups": load_tourney_matchups(required=True),
    }

    if include_additional_csvs:
        known = set(DEFAULT_PROCESSED_FILES.values())
        additional_sources: Dict[str, pd.DataFrame] = {}
        if PROCESSED_DATA_DIR.exists():
            for path in sorted(PROCESSED_DATA_DIR.glob("*.csv")):
                if path.name in known:
                    continue
                additional_sources[path.stem] = pd.read_csv(path)
        data["additional_sources"] = additional_sources

    return data
