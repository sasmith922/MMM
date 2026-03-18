"""
madness_model
=============
Reusable library for the March Madness bracket prediction pipeline.

Modules
-------
config           Central constants and hyperparameters.
paths            Central pathlib.Path objects for project directories.
load_data        Load raw CSV/Parquet data files.
load_processed_data Load processed CSV modeling sources.
clean_data       Clean and validate team, game, and seed data.
build_team_features  Aggregate season-end team features.
build_matchups   Build Team A vs Team B matchup rows for modelling.
build_model_dataset Assemble final matchup-level modeling dataframe.
feature_config   Central model feature lists and feature sets.
model_utils      Season split and metric utilities for model pipelines.
train_models     Multi-model training for rolling-season evaluation.
backtest_models  Rolling backtest runner across multiple models.
predict_matchups Score matchup rows with trained models.
elo              Elo rating system for NCAA teams.
baseline_model   Logistic regression baseline model.
xgb_model        XGBoost training and inference.
calibrate        Probability calibration helpers.
evaluate         Evaluation metrics (accuracy, log-loss, Brier score).
bracket          NCAA bracket data structures and graph representation.
simulate_bracket Deterministic and Monte Carlo bracket simulation.
visualize        Plots for calibration, feature importance, and team odds.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("madness-model")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "config",
    "paths",
    "load_data",
    "load_processed_data",
    "clean_data",
    "build_team_features",
    "build_matchups",
    "build_model_dataset",
    "feature_config",
    "model_utils",
    "train_models",
    "backtest_models",
    "predict_matchups",
    "elo",
    "baseline_model",
    "xgb_model",
    "calibrate",
    "evaluate",
    "bracket",
    "simulate_bracket",
    "visualize",
]
