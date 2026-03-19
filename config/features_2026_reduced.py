"""2026 reduced-feature configuration and validation policy."""

from __future__ import annotations

from pathlib import Path

from madness_model.paths import DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

SEASON_2026 = 2026

# Inputs/outputs for the dedicated 2026 reduced workflow.
TEAM_FEATURES_2026_REDUCED_PATH = PROCESSED_DATA_DIR / "team_features_2026_reduced.csv"
TOURNEY_MATCHUPS_2026_PATH = PROCESSED_DATA_DIR / "tourney_matchups_2026.csv"
MODELING_DATASET_2026_PATH = PROCESSED_DATA_DIR / "modeling_dataset_2026_reduced.csv"
MODEL_ARTIFACT_2026_PATH = PROJECT_ROOT / "models" / "xgboost_2026_reduced.joblib"
PREDICTIONS_2026_PATH = PROJECT_ROOT / "outputs" / "predictions" / "bracket_predictions_2026_reduced.csv"

# Optional/auxiliary data files used by fallback and training modes.
CBB26_RAW_PATH = DATA_DIR / "raw" / "cbb26.csv"
HISTORICAL_TEAM_FEATURES_V2_PATH = PROJECT_ROOT / "data" / "processed_v2" / "team_features_v2.csv"
HISTORICAL_MATCHUPS_V2_PATH = PROJECT_ROOT / "data" / "processed_v2" / "tournament_matchups_v2.csv"
FEATURE_NOTES_2026_PATH = PROJECT_ROOT / "outputs" / "reports" / "2026_reduced_feature_notes.md"

# Required identity columns that must exist in 2026 reduced team features.
REQUIRED_ID_COLUMNS = [
    "season",
    "team_name",
    "team_name_norm",
]

# Required model columns for the reduced 2026 modeling/training flow.
REQUIRED_MODEL_COLUMNS = [
    "seed",
    "win_pct",
    "pre_tourney_adjoe",
    "pre_tourney_adjde",
]

# Optional, but checked for data-quality visibility in logs.
TOURNAMENT_SEED_COLUMNS = ["seed"]

# Strict null checks for core modeling fields.
CRITICAL_NULL_CHECK_COLUMNS = [
    "team_name",
    "team_name_norm",
    "seed",
    "win_pct",
    "pre_tourney_adjoe",
    "pre_tourney_adjde",
]

# Preferred reduced features for modeling. Use intersection with available columns.
PREFERRED_2026_REDUCED_FEATURES = [
    "seed",
    "win_pct",
    "pre_tourney_adjoe",
    "pre_tourney_adjde",
    "pre_tourney_adjem",
    "pre_tourney_barthag",
    "pre_tourney_wab",
    "pre_tourney_adjtempo",
    "efgpct",
    "opp_efgpct",
    "topct",
    "opp_topct",
    "orpct",
    "opp_orpct",
    "ftrate",
    "opp_ftrate",
    "fg2pct",
    "oppfg2pct",
    "fg3pct",
    "oppfg3pct",
]


def ensure_parent(path: Path) -> None:
    """Create parent directory for output paths."""
    path.parent.mkdir(parents=True, exist_ok=True)

