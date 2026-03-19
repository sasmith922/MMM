# 2026 Reduced Feature Notes

This repository includes a dedicated 2026 reduced pipeline that is intentionally separate from the historical full-feature assumptions.

## Included 2026 feature groups

- Core team identity:
  - `season`, `team_name`, `team_name_norm`
- Tournament seed:
  - `seed` (plus parsed `seed_num` and optional region prefix)
- Pre-tournament strength:
  - `pre_tourney_adjoe`, `pre_tourney_adjde`, `pre_tourney_adjem`
- Basic results:
  - `win_pct`
- Optional reduced stats if present in input:
  - `pre_tourney_adjtempo`, `pre_tourney_barthag`, `pre_tourney_wab`
  - `efgpct`, `opp_efgpct`, `topct`, `opp_topct`, `orpct`, `opp_orpct`
  - `ftrate`, `opp_ftrate`, `fg2pct`, `oppfg2pct`, `fg3pct`, `oppfg3pct`
  - Torvik-prefixed merged columns

## Excluded on purpose for 2026 reduced

- Custom top-25/top-50 and bad-loss feature groups from older pipelines (unless already present in the reduced file)
- Resume-delta or other handcrafted feature groups not guaranteed in the 2026 reduced input
- ESPN IDs, Kaggle IDs, or any ID system not present in `team_features_2026_reduced.csv`
- Legacy full-feature assumptions from baseline/v2 paths when missing in 2026 reduced data

## Validation checks (`scripts/build_2026_dataset.py`)

- Duplicate key check on `(season, team_name_norm)`
- Missing `team_name` and missing/blank `team_name_norm`
- Missing seeds for tournament teams
- Critical null checks for:
  - `team_name`, `team_name_norm`, `seed`, `win_pct`, `pre_tourney_adjoe`, `pre_tourney_adjde`

## Reduced 2026 commands

Use these scripts for the dedicated reduced-feature workflow (separate from full-feature pipelines):

```bash
# 1) Build/validate 2026 reduced data + first-round matchup file
python scripts/build_2026_dataset.py

# 2) Run held-out backtest on a prior year (train on seasons < YEAR)
python scripts/run_backtest_reduced.py --test-year 2025

# 3) Train reduced model and write 2026 matchup probabilities + feature list
python scripts/train_predict_2026_reduced.py

# 4) Generate full bracket-style 2026 breakdown + readable summary
python scripts/predict_bracket_2026_reduced.py
```

Default outputs:

- `outputs/reports/features_2026_reduced_used.txt`
- `outputs/predictions/backtest_<YEAR>_reduced.csv`
- `outputs/predictions/backtest_metrics_reduced.csv`
- `outputs/predictions/bracket_predictions_2026_reduced.csv`
- `outputs/predictions/bracket_breakdown_2026_reduced.csv`
- `outputs/predictions/bracket_summary_2026_reduced.txt`
