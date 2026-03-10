"""
build_matchups.py
-----------------
Construct Team A vs Team B matchup rows from team features and tournament
results.  Each row represents one potential or actual tournament matchup and
contains the feature *difference* between the two teams plus a binary label
indicating whether Team A won.

The resulting DataFrame is the direct input to the logistic regression
and XGBoost models.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def build_matchup_row(
    season: int,
    team_a_id: int,
    team_b_id: int,
    features: pd.DataFrame,
    label: Optional[int] = None,
) -> dict:
    """Build a single matchup feature row as a dictionary.

    Feature values are computed as ``team_a_feature - team_b_feature`` for
    every numeric feature column so that the model sees a signed differential.

    Parameters
    ----------
    season:
        The season year for this matchup.
    team_a_id:
        Team A's integer ID.
    team_b_id:
        Team B's integer ID.
    features:
        Season-end team features indexed by ``(season, team_id)`` as
        returned by :func:`~madness_model.build_team_features.build_team_features`.
    label:
        1 if Team A wins, 0 if Team B wins.  Pass ``None`` for prediction
        rows where the outcome is unknown.

    Returns
    -------
    dict
        Dictionary with keys: ``season``, ``team_a_id``, ``team_b_id``,
        feature differentials, and optionally ``label``.
    """
    try:
        feat_a = features.loc[(season, team_a_id)]
        feat_b = features.loc[(season, team_b_id)]
    except KeyError as exc:
        raise KeyError(
            f"Features not found for season={season}, team(s) missing: {exc}"
        ) from exc

    numeric_cols: List[str] = features.select_dtypes("number").columns.tolist()
    row: dict = {
        "season": season,
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
    }
    for col in numeric_cols:
        row[f"diff_{col}"] = float(feat_a[col]) - float(feat_b[col])

    if label is not None:
        row["label"] = int(label)

    return row


def build_matchups_from_results(
    tourney_results: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a labelled matchup dataset from historical tournament results.

    For every tournament game, two rows are generated: one with the winner
    as Team A (label=1) and one with the loser as Team A (label=0).  This
    ensures the model does not learn a positional bias.

    Parameters
    ----------
    tourney_results:
        Cleaned tournament results with columns ``season``, ``w_team_id``,
        ``l_team_id``.
    features:
        Season-end team features indexed by ``(season, team_id)``.

    Returns
    -------
    pd.DataFrame
        Matchup rows ready for model training.
    """
    rows = []
    for _, game in tourney_results.iterrows():
        season = int(game["season"])
        w_id = int(game["w_team_id"])
        l_id = int(game["l_team_id"])

        # Skip if either team is missing features
        if (season, w_id) not in features.index or (season, l_id) not in features.index:
            # TODO: log skipped rows rather than silently dropping
            continue

        # Winner as Team A
        rows.append(build_matchup_row(season, w_id, l_id, features, label=1))
        # Loser as Team A
        rows.append(build_matchup_row(season, l_id, w_id, features, label=0))

    return pd.DataFrame(rows)


def build_prediction_matchups(
    season: int,
    matchup_pairs: List[tuple[int, int]],
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build unlabelled matchup rows for bracket prediction.

    Parameters
    ----------
    season:
        The season to predict.
    matchup_pairs:
        List of ``(team_a_id, team_b_id)`` tuples representing potential
        first-round (or any-round) matchups.
    features:
        Season-end team features indexed by ``(season, team_id)``.

    Returns
    -------
    pd.DataFrame
        Matchup rows without a ``label`` column, ready for inference.
    """
    rows = [
        build_matchup_row(season, a_id, b_id, features)
        for a_id, b_id in matchup_pairs
        if (season, a_id) in features.index and (season, b_id) in features.index
    ]
    # TODO: warn about any matchup_pairs that were skipped due to missing features
    return pd.DataFrame(rows)
