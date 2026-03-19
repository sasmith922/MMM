from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

DEFAULT_PROCESSED_FILES = {
    "team_profiles": PROCESSED_DIR / "team_profiles.csv",
    "games_boxscores": PROCESSED_DIR / "games_boxscores.csv",
    "tourney_matchups": PROCESSED_DIR / "tourney_matchups.csv",
}

REQUIRED_COLUMNS = {
    "team_profiles": ["season", "team_id"],
    "games_boxscores": ["season"],
    "tourney_matchups": ["season", "teamA_id", "teamB_id", "target"],
}


def _validate_required_columns(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    required_columns: list[str],
) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"{dataset_name} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _normalize_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if "team_id" not in df.columns and "kaggle_team_id" in df.columns:
        df = df.rename(columns={"kaggle_team_id": "team_id"})
    return df


def _normalize_tourney_matchups(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "team_a_id": "teamA_id",
        "team_b_id": "teamB_id",
        "team_a_name": "teamA_name",
        "team_b_name": "teamB_name",
        "team_a_seed": "teamA_seed",
        "team_b_seed": "teamB_seed",
        "team_a_seed_num": "teamA_seed_num",
        "team_b_seed_num": "teamB_seed_num",
        "round_num_guess": "round",
    }
    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def _read_csv(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    print(f"Loading {dataset_name} from: {csv_path}")
    print(f"Exists: {csv_path.exists()}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def load_team_profiles(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DEFAULT_PROCESSED_FILES["team_profiles"]
    df = _read_csv(csv_path, "team_profiles")
    df = _normalize_team_profiles(df)
    _validate_required_columns(
        df,
        dataset_name="team_profiles",
        required_columns=REQUIRED_COLUMNS["team_profiles"],
    )
    return df


def load_games_boxscores(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DEFAULT_PROCESSED_FILES["games_boxscores"]
    df = _read_csv(csv_path, "games_boxscores")
    _validate_required_columns(
        df,
        dataset_name="games_boxscores",
        required_columns=REQUIRED_COLUMNS["games_boxscores"],
    )
    return df


def load_tourney_matchups(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DEFAULT_PROCESSED_FILES["tourney_matchups"]
    df = _read_csv(csv_path, "tourney_matchups")
    df = _normalize_tourney_matchups(df)
    _validate_required_columns(
        df,
        dataset_name="tourney_matchups",
        required_columns=REQUIRED_COLUMNS["tourney_matchups"],
    )
    return df


def load_all_processed_data() -> Mapping[str, pd.DataFrame]:
    return {
        "team_profiles": load_team_profiles(),
        "games_boxscores": load_games_boxscores(),
        "tourney_matchups": load_tourney_matchups(),
    }