"""
elo.py
------
Elo rating system for NCAA basketball teams.

Elo ratings are updated after every regular-season game and can be used as
a powerful feature in the matchup model.  The implementation follows the
standard Elo formula with a configurable K-factor and optional margin-of-
victory multiplier.

References
----------
- Elo, A. (1978). The Rating of Chessplayers, Past and Present.
- FiveThirtyEight March Madness Elo methodology.
"""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd


# Default starting Elo for a team with no history.
DEFAULT_ELO: float = 1500.0

# K-factor controls how quickly ratings change after each game.
DEFAULT_K: float = 20.0

# At the start of each new season, ratings revert toward the mean by this factor.
SEASON_REVERSION_FACTOR: float = 0.5


def expected_score(elo_a: float, elo_b: float) -> float:
    """Compute the expected win probability for Team A given Elo ratings.

    Parameters
    ----------
    elo_a:
        Elo rating of Team A.
    elo_b:
        Elo rating of Team B.

    Returns
    -------
    float
        Probability in (0, 1) that Team A wins.
    """
    return 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))


def update_elos(
    elo_a: float,
    elo_b: float,
    score_a: int,
    score_b: int,
    k: float = DEFAULT_K,
) -> tuple[float, float]:
    """Update Elo ratings after a single game.

    Parameters
    ----------
    elo_a:
        Pre-game Elo rating of Team A.
    elo_b:
        Pre-game Elo rating of Team B.
    score_a:
        Points scored by Team A.
    score_b:
        Points scored by Team B.
    k:
        K-factor (step size).

    Returns
    -------
    tuple[float, float]
        Updated ``(elo_a, elo_b)`` ratings.
    """
    actual_a = 1.0 if score_a > score_b else 0.0
    expected_a = expected_score(elo_a, elo_b)

    # TODO: implement margin-of-victory multiplier for more accurate ratings
    delta = k * (actual_a - expected_a)
    return elo_a + delta, elo_b - delta


def revert_to_mean(elo: float, mean: float = DEFAULT_ELO, factor: float = SEASON_REVERSION_FACTOR) -> float:
    """Apply mean-reversion to an Elo rating at the start of a new season.

    Parameters
    ----------
    elo:
        End-of-season Elo rating.
    mean:
        Population mean Elo (default 1500).
    factor:
        Fraction to revert toward the mean.  0 = no reversion, 1 = full reset.

    Returns
    -------
    float
        Reverted Elo rating.
    """
    return elo + factor * (mean - elo)


def compute_elo_ratings(games: pd.DataFrame) -> Dict[tuple[int, int], float]:
    """Compute end-of-season Elo ratings for all teams across all seasons.

    Processes games in chronological order.  Between seasons, ratings revert
    toward the population mean.

    Parameters
    ----------
    games:
        Cleaned regular-season game results with columns
        ``season``, ``day_num``, ``w_team_id``, ``l_team_id``,
        ``w_score``, ``l_score``.

    Returns
    -------
    dict
        Mapping of ``(season, team_id)`` → end-of-season Elo rating.
    """
    ratings: Dict[int, float] = {}
    season_end_ratings: Dict[tuple[int, int], float] = {}

    # Process games in season + day order
    sorted_games = games.sort_values(["season", "day_num"])
    prev_season: int | None = None

    for _, game in sorted_games.iterrows():
        season = int(game["season"])
        w_id = int(game["w_team_id"])
        l_id = int(game["l_team_id"])

        # Apply mean reversion at season boundary
        if prev_season is not None and season != prev_season:
            ratings = {
                tid: revert_to_mean(r)
                for tid, r in ratings.items()
            }

        prev_season = season

        # Initialise unseen teams
        if w_id not in ratings:
            ratings[w_id] = DEFAULT_ELO
        if l_id not in ratings:
            ratings[l_id] = DEFAULT_ELO

        # Update ratings
        ratings[w_id], ratings[l_id] = update_elos(
            ratings[w_id],
            ratings[l_id],
            int(game["w_score"]),
            int(game["l_score"]),
        )

    # Record end-of-season ratings
    if prev_season is not None:
        for tid, r in ratings.items():
            season_end_ratings[(prev_season, tid)] = r

    # TODO: also store ratings at end of *each* season, not just the last
    return season_end_ratings


def elo_to_dataframe(elo_ratings: Dict[tuple[int, int], float]) -> pd.DataFrame:
    """Convert the Elo ratings dict to a tidy DataFrame.

    Parameters
    ----------
    elo_ratings:
        Mapping from ``(season, team_id)`` to Elo rating.

    Returns
    -------
    pd.DataFrame
        Columns: ``season``, ``team_id``, ``elo``.
    """
    records = [
        {"season": season, "team_id": team_id, "elo": elo}
        for (season, team_id), elo in elo_ratings.items()
    ]
    return pd.DataFrame(records)
