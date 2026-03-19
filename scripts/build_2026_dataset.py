"""
build_2026_dataset.py
---------------------
Prepare and validate a dedicated 2026 reduced-feature dataset and
first-round tournament matchup file.

Usage
-----
    python scripts/build_2026_dataset.py
"""

from __future__ import annotations

import re
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.features_2026_reduced import (
    CRITICAL_NULL_CHECK_COLUMNS,
    REQUIRED_ID_COLUMNS,
    REQUIRED_MODEL_COLUMNS,
    SEASON_2026,
    TEAM_FEATURES_2026_REDUCED_PATH,
    TOURNAMENT_SEED_COLUMNS,
    TOURNEY_MATCHUPS_2026_PATH,
    ensure_parent,
)

ROUND64_SEED_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def normalize_team_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("st.", "st")
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_seed_number(seed_value: object) -> float:
    if pd.isna(seed_value):
        return float("nan")
    match = re.search(r"(\d+)", str(seed_value))
    if match is None:
        return float("nan")
    return float(match.group(1))


def _parse_region(seed_value: object) -> str | None:
    if pd.isna(seed_value):
        return None
    match = re.match(r"([A-Za-z]+)", str(seed_value).strip())
    if match is None:
        return None
    return match.group(1).upper()


def _require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing required columns: {missing}")


def _normalize_reduced_team_features(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    _require_columns(normalized, REQUIRED_ID_COLUMNS, "team_features_2026_reduced")
    _require_columns(normalized, REQUIRED_MODEL_COLUMNS, "team_features_2026_reduced")

    normalized["season"] = pd.to_numeric(normalized["season"], errors="coerce")
    normalized = normalized[normalized["season"] == SEASON_2026].copy()
    if normalized.empty:
        raise ValueError(f"No rows found for season={SEASON_2026}.")

    normalized["team_name"] = normalized["team_name"].astype(str).str.strip()
    missing_team_names = normalized["team_name"].eq("") | normalized["team_name"].str.lower().eq("nan")
    if missing_team_names.any():
        raise ValueError(
            "Missing team_name values detected in 2026 reduced features. "
            f"Rows: {int(missing_team_names.sum())}"
        )

    team_name_norm_missing = (
        normalized["team_name_norm"].isna()
        | normalized["team_name_norm"].astype(str).str.strip().eq("")
        | normalized["team_name_norm"].astype(str).str.lower().eq("nan")
    )
    if team_name_norm_missing.any():
        normalized.loc[team_name_norm_missing, "team_name_norm"] = (
            normalized.loc[team_name_norm_missing, "team_name"].map(normalize_team_name)
        )
        print(f"[normalize] filled {int(team_name_norm_missing.sum())} missing team_name_norm values from team_name")

    duplicate_mask = normalized.duplicated(subset=["season", "team_name_norm"], keep=False)
    if duplicate_mask.any():
        dupes = normalized.loc[duplicate_mask, ["season", "team_name", "team_name_norm"]]
        raise ValueError(
            "Duplicate (season, team_name_norm) rows found in 2026 reduced features:\n"
            + dupes.head(10).to_string(index=False)
        )

    for column in TOURNAMENT_SEED_COLUMNS:
        if column in normalized.columns and normalized[column].isna().any():
            raise ValueError(
                "Missing seeds found for 2026 tournament teams. "
                f"Column={column}, missing={int(normalized[column].isna().sum())}"
            )

    for column in CRITICAL_NULL_CHECK_COLUMNS:
        if column in normalized.columns:
            missing_count = int(normalized[column].isna().sum())
            if missing_count > 0:
                raise ValueError(f"Critical nulls found in column '{column}': {missing_count}")

    return normalized.reset_index(drop=True)


def _build_round64_matchups(team_features_2026: pd.DataFrame) -> pd.DataFrame:
    teams = team_features_2026.copy()
    teams["seed_num"] = teams["seed"].map(_parse_seed_number)
    teams["region"] = teams["seed"].map(_parse_region)

    if teams["seed_num"].isna().any():
        raise ValueError(
            "Unable to parse numeric seed values for all teams. "
            f"Unparsed rows: {int(teams['seed_num'].isna().sum())}"
        )

    if teams["region"].notna().sum() == 0:
        teams["region"] = "ALL"
        print("[matchups] seed region prefix not present; falling back to pooled seed pairing.")

    rows: list[dict[str, object]] = []
    for region, region_df in teams.groupby("region", dropna=False):
        region_df = region_df.sort_values("team_name_norm")
        for seed_a, seed_b in ROUND64_SEED_PAIRS:
            a_df = region_df[region_df["seed_num"] == seed_a].sort_values("team_name_norm")
            b_df = region_df[region_df["seed_num"] == seed_b].sort_values("team_name_norm")
            if a_df.empty or b_df.empty:
                continue

            pair_count = min(len(a_df), len(b_df))
            for idx in range(pair_count):
                team_a = a_df.iloc[idx]
                team_b = b_df.iloc[idx]
                rows.append(
                    {
                        "season": int(SEASON_2026),
                        "round": "R64",
                        "region": region,
                        "teamA_name": team_a["team_name"],
                        "teamB_name": team_b["team_name"],
                        "teamA_name_norm": team_a["team_name_norm"],
                        "teamB_name_norm": team_b["team_name_norm"],
                        "teamA_seed": team_a["seed"],
                        "teamB_seed": team_b["seed"],
                        "teamA_seed_num": int(team_a["seed_num"]),
                        "teamB_seed_num": int(team_b["seed_num"]),
                    }
                )

    matchups = pd.DataFrame(rows)
    if matchups.empty:
        raise ValueError("No 2026 round-of-64 matchups could be built from available seeds.")

    matchups = matchups.drop_duplicates(
        subset=["season", "teamA_name_norm", "teamB_name_norm"],
        keep="first",
    ).reset_index(drop=True)
    return matchups


def build_2026_reduced_dataset(
    *,
    features_path: Path = TEAM_FEATURES_2026_REDUCED_PATH,
    matchups_path: Path = TOURNEY_MATCHUPS_2026_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing required 2026 reduced feature file: {features_path}. "
            "Generate it first (e.g., from cbb26 + Torvik merge)."
        )

    raw = pd.read_csv(features_path)
    cleaned = _normalize_reduced_team_features(raw)
    matchups = _build_round64_matchups(cleaned)

    ensure_parent(features_path)
    ensure_parent(matchups_path)
    cleaned.to_csv(features_path, index=False)
    matchups.to_csv(matchups_path, index=False)

    print(f"[save] cleaned 2026 reduced team features: {features_path} ({len(cleaned)} rows)")
    print(f"[save] 2026 tournament matchups: {matchups_path} ({len(matchups)} rows)")
    return cleaned, matchups


def main() -> None:
    build_2026_reduced_dataset()


if __name__ == "__main__":
    main()
