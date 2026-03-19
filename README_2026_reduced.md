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

