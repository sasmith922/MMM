# March Madness Model (MMM)

A machine learning pipeline that predicts NCAA tournament matchup win probabilities
using tabular team-season features, then simulates the full 64-team bracket.

---

## Project Purpose

This project builds a data-driven bracket prediction system for the NCAA Men's
Basketball Tournament.  It ingests historical regular-season and tournament data,
engineers team-level features, trains probability models, calibrates their outputs,
and runs Monte Carlo bracket simulations to estimate each team's championship odds.

---

## Folder Structure

```
march-madness-model/
├── data/
│   ├── raw/            ← Downloaded source files (ignored by git)
│   ├── interim/        ← Intermediate cleaned/merged files
│   └── processed/      ← Final feature tables and matchup datasets
├── models/             ← Saved model artefacts (.pkl, .json)
├── notebooks/          ← Jupyter notebooks for EDA only
├── outputs/
│   ├── figures/        ← Plots (calibration curves, feature importance, …)
│   ├── predictions/    ← Bracket probability CSVs
│   └── reports/        ← Evaluation metric reports (JSON)
├── scripts/            ← Runnable entry-point scripts
├── src/
│   └── madness_model/  ← Reusable project library
└── tests/              ← pytest unit tests
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sasmith922/MMM.git
cd MMM
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
# or, using requirements.txt directly:
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
pytest
```

---

## Expected Workflow

```
Raw data
  │  scripts/download_data.py
  ▼
data/raw/
  │  scripts/build_features.py
  ▼
data/processed/team_features.parquet
  │  scripts/build_matchups.py
  ▼
data/processed/matchups.parquet
  │  scripts/train_baseline.py  /  scripts/train_xgb.py
  ▼
models/baseline_logreg.pkl  /  models/xgb_model.json
  │  scripts/evaluate.py
  ▼
outputs/reports/evaluation_report.json
outputs/figures/calibration_curve.png
  │  scripts/predict_bracket.py
  ▼
outputs/predictions/bracket_predictions.csv
  │  scripts/simulate_bracket.py
  ▼
outputs/predictions/simulation_results.csv
outputs/figures/champion_odds.png
```

---

## Key Modules (`src/madness_model/`)

| Module | Purpose |
|---|---|
| `config.py` | Central constants (seed, seasons, paths, hyperparameters) |
| `paths.py` | `pathlib.Path` objects for every project directory |
| `load_data.py` | Load CSV / Parquet raw files |
| `clean_data.py` | Validate and standardise raw DataFrames |
| `build_team_features.py` | Aggregate season-end team statistics |
| `build_matchups.py` | Construct Team A vs Team B feature-differential rows |
| `elo.py` | Elo rating system for NCAA teams |
| `baseline_model.py` | Logistic regression baseline (scikit-learn Pipeline) |
| `xgb_model.py` | XGBoost training and inference |
| `calibrate.py` | Probability calibration (isotonic / Platt scaling) |
| `evaluate.py` | Accuracy, log loss, Brier score, AUC-ROC |
| `simulate_bracket.py` | Deterministic and Monte Carlo bracket simulation |
| `visualize.py` | Calibration curve, feature importance, champion odds plots |

---

## Running Tests

```bash
pytest -v
```

---

## Data Sources

Place raw data files in `data/raw/` before running the pipeline.  The
project is designed to work with the
[March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2024)
Kaggle dataset.  See `scripts/download_data.py` for download instructions.
