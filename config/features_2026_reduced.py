"""2026 reduced-feature configuration and validation policy."""

from __future__ import annotations

from pathlib import Path
import re

from madness_model.paths import DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

SEASON_2026 = 2026

# Inputs/outputs for the dedicated 2026 reduced workflow.
TEAM_FEATURES_2026_REDUCED_PATH = PROCESSED_DATA_DIR / "team_features_2026_reduced.csv"
TOURNEY_MATCHUPS_2026_PATH = PROCESSED_DATA_DIR / "tourney_matchups_2026.csv"
MODELING_DATASET_2026_PATH = PROCESSED_DATA_DIR / "modeling_dataset_2026_reduced.csv"
MODEL_ARTIFACT_2026_PATH = PROJECT_ROOT / "models" / "xgboost_2026_reduced.joblib"
PREDICTIONS_2026_PATH = PROJECT_ROOT / "outputs" / "predictions" / "bracket_predictions_2026_reduced.csv"
BRACKET_BREAKDOWN_2026_PATH = PROJECT_ROOT / "outputs" / "predictions" / "bracket_breakdown_2026_reduced.csv"
BRACKET_SUMMARY_2026_PATH = PROJECT_ROOT / "outputs" / "predictions" / "bracket_summary_2026_reduced.txt"
BACKTEST_METRICS_REDUCED_PATH = PROJECT_ROOT / "outputs" / "predictions" / "backtest_metrics_reduced.csv"
FEATURE_LIST_2026_REDUCED_PATH = PROJECT_ROOT / "outputs" / "reports" / "features_2026_reduced_used.txt"

# Optional/auxiliary data files used by fallback and training modes.
CBB26_RAW_PATH = DATA_DIR / "raw" / "cbb26.csv"
HISTORICAL_TEAM_FEATURES_V2_PATH = PROJECT_ROOT / "data" / "processed_v2" / "team_features_v2.csv"
HISTORICAL_MATCHUPS_V2_PATH = PROJECT_ROOT / "data" / "processed_v2" / "tournament_matchups_v2.csv"
HISTORICAL_MATCHUPS_PROCESSED_PATH = PROCESSED_DATA_DIR / "tourney_matchups.csv"
FEATURE_NOTES_2026_PATH = PROJECT_ROOT / "outputs" / "reports" / "2026_reduced_feature_notes.md"

# Minimum viable overlap to avoid accidentally fitting on almost-empty signal.
MIN_OVERLAP_FEATURE_COUNT = 3

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


def resolve_first_existing_path(candidates: list[Path], *, purpose: str) -> Path:
    """Return first existing candidate path; raise FileNotFoundError when none exist."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find required {purpose}. Checked:\n{candidate_text}\n"
        "Please build the required dataset(s) or pass an explicit path argument."
    )


def parse_seed_number(seed_value: object) -> float:
    """Parse integer seed number from values like 'W01', 'X16', or '12'."""
    if seed_value is None:
        return float("nan")
    if isinstance(seed_value, float):
        try:
            if seed_value != seed_value:  # NaN
                return float("nan")
        except Exception:
            return float("nan")
    match = re.search(r"(\d+)", str(seed_value))
    if match is None:
        return float("nan")
    return float(match.group(1))


def parse_seed_region(seed_value: object) -> str | None:
    """Parse region prefix from seeded values like 'W01'."""
    if seed_value is None:
        return None
    match = re.match(r"([A-Za-z]+)", str(seed_value).strip())
    if match is None:
        return None
    return match.group(1).upper()
