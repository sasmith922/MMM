"""
config.py
---------
Central configuration constants for the March Madness prediction pipeline.

All tunable knobs — seeds, season splits, model hyperparameters, and output
paths — live here so that the rest of the codebase never contains magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from madness_model.paths import MODELS_DIR, OUTPUTS_DIR

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Season configuration
# ---------------------------------------------------------------------------

# Seasons used to train the model (inclusive on both ends).
TRAINING_SEASONS: List[int] = list(range(2003, 2022))

# Seasons used to tune / validate hyper-parameters.
VALIDATION_SEASONS: List[int] = [2022, 2023]

# The season for which we generate bracket predictions.
PREDICTION_SEASON: int = 2025

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

# Number of Monte Carlo bracket simulations to run.
NUM_SIMULATIONS: int = 10_000

# ---------------------------------------------------------------------------
# Model output paths
# ---------------------------------------------------------------------------

BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_logreg.pkl"
XGB_MODEL_PATH: Path = MODELS_DIR / "xgb_model.json"
CALIBRATED_MODEL_PATH: Path = MODELS_DIR / "calibrated_model.pkl"

# ---------------------------------------------------------------------------
# Data paths (resolved at import time via paths.py)
# ---------------------------------------------------------------------------

PREDICTIONS_OUTPUT_PATH: Path = OUTPUTS_DIR / "predictions" / "bracket_predictions.csv"
REPORT_OUTPUT_PATH: Path = OUTPUTS_DIR / "reports" / "evaluation_report.json"

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters (starter values; tune via cross-validation)
# ---------------------------------------------------------------------------


@dataclass
class XGBConfig:
    """Hyper-parameter container for the XGBoost model."""

    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    use_label_encoder: bool = False
    eval_metric: str = "logloss"
    random_state: int = field(default_factory=lambda: RANDOM_SEED)

    # TODO: Add early-stopping rounds once a validation set is wired in.


@dataclass
class LogRegConfig:
    """Hyper-parameter container for the logistic regression baseline."""

    C: float = 1.0
    max_iter: int = 1_000
    solver: str = "lbfgs"
    random_state: int = field(default_factory=lambda: RANDOM_SEED)


XGB_CONFIG = XGBConfig()
LOGREG_CONFIG = LogRegConfig()
