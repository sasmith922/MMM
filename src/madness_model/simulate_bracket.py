"""
simulate_bracket.py
-------------------
Deterministic and Monte Carlo NCAA bracket simulation.

The bracket is represented as a list of rounds, where each round contains
pairs of team IDs.  The model assigns a win probability to each matchup,
and the simulation either:

1. **Deterministic**: always advances the team with probability > 0.5.
2. **Monte Carlo**: samples outcomes stochastically, runs N simulations,
   and aggregates each team's probability of reaching each round.

Bracket format expected
-----------------------
A 64-team field is represented as a flat list of 64 team IDs ordered so
that adjacent pairs are first-round matchups (indices 0&1, 2&3, …).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from madness_model.config import NUM_SIMULATIONS, RANDOM_SEED

# Type alias: a predict function takes (team_a_id, team_b_id) and returns P(A wins)
PredictFn = Callable[[int, int], float]


def simulate_game(
    team_a: int,
    team_b: int,
    predict_fn: PredictFn,
    deterministic: bool = False,
    rng: random.Random | None = None,
) -> int:
    """Simulate a single game and return the winning team's ID.

    Parameters
    ----------
    team_a:
        ID of Team A.
    team_b:
        ID of Team B.
    predict_fn:
        Callable that returns P(team_a wins).
    deterministic:
        If ``True``, the team with higher probability always wins.
    rng:
        Optional seeded :class:`random.Random` instance for reproducibility.

    Returns
    -------
    int
        ID of the winning team.
    """
    prob_a = predict_fn(team_a, team_b)
    if deterministic:
        return team_a if prob_a >= 0.5 else team_b
    r = (rng or random).random()
    return team_a if r < prob_a else team_b


def simulate_bracket(
    field: List[int],
    predict_fn: PredictFn,
    deterministic: bool = False,
    rng: random.Random | None = None,
) -> Dict[int, int]:
    """Simulate a full single-elimination bracket.

    Parameters
    ----------
    field:
        Ordered list of team IDs (length must be a power of 2).
        Adjacent pairs are first-round matchups.
    predict_fn:
        Callable ``(team_a_id, team_b_id) -> float`` returning P(A wins).
    deterministic:
        If ``True``, always advance the favourite.
    rng:
        Optional seeded RNG for Monte Carlo reproducibility.

    Returns
    -------
    dict
        Mapping of ``team_id → round_reached`` (1 = first round exit,
        2 = second round, …, 7 = champion for a 64-team bracket).
    """
    n = len(field)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Field size must be a power of 2, got {n}.")

    round_reached: Dict[int, int] = {team: 1 for team in field}
    current_round = list(field)
    round_number = 1

    while len(current_round) > 1:
        round_number += 1
        next_round = []
        for i in range(0, len(current_round), 2):
            winner = simulate_game(
                current_round[i],
                current_round[i + 1],
                predict_fn,
                deterministic=deterministic,
                rng=rng,
            )
            round_reached[winner] = round_number
            next_round.append(winner)
        current_round = next_round

    return round_reached


def monte_carlo_simulation(
    field: List[int],
    predict_fn: PredictFn,
    n_simulations: int = NUM_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run N Monte Carlo bracket simulations and aggregate results.

    Parameters
    ----------
    field:
        Ordered list of team IDs.
    predict_fn:
        Callable returning P(team_a wins).
    n_simulations:
        Number of independent simulations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``team_id``, ``avg_round``, ``champion_pct``,
        ``final_four_pct``, ``elite_eight_pct``.
        One row per team, sorted by ``champion_pct`` descending.
    """
    rng = random.Random(seed)
    total_rounds = int(np.log2(len(field))) + 1  # rounds 1..total_rounds
    champion_round = total_rounds

    # Accumulate round counts
    round_totals: Dict[int, List[int]] = {team: [] for team in field}

    for _ in range(n_simulations):
        result = simulate_bracket(field, predict_fn, deterministic=False, rng=rng)
        for team, rd in result.items():
            round_totals[team].append(rd)

    records = []
    for team, rounds in round_totals.items():
        arr = np.array(rounds)
        records.append(
            {
                "team_id": team,
                "avg_round": float(arr.mean()),
                "champion_pct": float((arr == champion_round).mean()),
                "final_four_pct": float((arr >= champion_round - 1).mean()),
                "elite_eight_pct": float((arr >= champion_round - 2).mean()),
            }
        )

    df = pd.DataFrame(records).sort_values("champion_pct", ascending=False)
    return df.reset_index(drop=True)


def build_predict_fn(
    model,
    season: int,
    features: pd.DataFrame,
    feature_cols: List[str],
) -> PredictFn:
    """Construct a predict function closure for a fitted model.

    Parameters
    ----------
    model:
        Any fitted model with a ``predict_proba`` method.
    season:
        Season year used to look up features.
    features:
        Team feature DataFrame indexed by ``(season, team_id)``.
    feature_cols:
        Ordered feature column names.

    Returns
    -------
    PredictFn
        Callable ``(team_a_id, team_b_id) -> float``.
    """
    # TODO: cache matchup rows to avoid redundant DataFrame construction
    from madness_model.build_matchups import build_matchup_row

    def predict_fn(team_a: int, team_b: int) -> float:
        row = build_matchup_row(season, team_a, team_b, features)
        df = pd.DataFrame([row])
        X = df[feature_cols].values
        return float(model.predict_proba(X)[0, 1])

    return predict_fn
