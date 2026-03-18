"""Helpers for loading processed CSV feature sources for modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from madness_model.paths import PROCESSED_DATA_DIR


DEFAULT_PROCESSED_FILES: Mapping[str, str] = {
    "team_profiles": "team_profiles.csv",
    "games_boxscores": "games_boxscores.csv",
    "tourney_matchups": "tourney_matchups.csv",
}

REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "team_profiles": ("season", "team_id"),
    "games_boxscores": ("season",),
    "tourney_matchups": ("season", "teamA_id", "teamB_id", "target"),
}


def _resolve_path(path: str | Path | None, default_file_name: str) -> Path:
    """Resolve an explicit file path or the default processed-data file path."""
    if path is None:
        return (PROCESSED_DATA_DIR / default_file_name).resolve()
    return Path(path).resolve()


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    required_columns: tuple[str, ...],
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"{dataset_name} is missing required columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def _load_dataset(
    dataset_name: str,
    default_file_name: str,
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate one processed dataset CSV."""
    file_path = _resolve_path(path, default_file_name)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required processed dataset '{dataset_name}' not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    _validate_required_columns(
        df,
        dataset_name=dataset_name,
        required_columns=REQUIRED_COLUMNS[dataset_name],
    )

    print(f"Loaded {dataset_name}: {len(df):,} rows from {file_path}")
    return df


def load_team_profiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``team_profiles.csv`` with required schema validation."""
    return _load_dataset("team_profiles", DEFAULT_PROCESSED_FILES["team_profiles"], path=path)


def load_games_boxscores(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``games_boxscores.csv`` with required schema validation."""
    return _load_dataset(
        "games_boxscores",
        DEFAULT_PROCESSED_FILES["games_boxscores"],
        path=path,
    )


def load_tourney_matchups(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``tourney_matchups.csv`` with required schema validation."""
    return _load_dataset(
        "tourney_matchups",
        DEFAULT_PROCESSED_FILES["tourney_matchups"],
        path=path,
    )


def load_all_processed_data() -> dict[str, pd.DataFrame]:
    """Load all required processed modeling data sources."""
    return {
        "team_profiles": load_team_profiles(),
        "games_boxscores": load_games_boxscores(),
        "tourney_matchups": load_tourney_matchups(),
    }
