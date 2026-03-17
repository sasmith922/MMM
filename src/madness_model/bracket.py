"""
bracket.py
----------
Data classes and constants for the NCAA tournament bracket structure.

The bracket is modelled as a directed acyclic graph of games
(:class:`BracketGame` nodes).  Each game knows which slots or
previous-game winners feed into it, and which game its winner
advances to next.

Typical usage
-------------
>>> games = load_bracket_structure(2025)
>>> state = build_initial_bracket(2025, teams_df, seeds_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS: list[str] = ["East", "West", "South", "Midwest"]

ROUND_NAMES: list[str] = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]

# Ordered index for round comparison (higher = further in tournament)
ROUND_ORDER: dict[str, int] = {name: idx for idx, name in enumerate(ROUND_NAMES)}

# Default mapping from Kaggle single-letter region codes → full region names.
# These letters vary by year; callers may supply their own mapping.
DEFAULT_REGION_MAP: dict[str, str] = {
    "W": "West",
    "X": "East",
    "Y": "South",
    "Z": "Midwest",
}

# Standard Round-of-64 seed matchups, in bracket order.
# Adjacent pairs of R64 games feed into the same R32 game.
# Order: (1v16), (8v9), (5v12), (4v13), (6v11), (3v14), (7v10), (2v15)
R64_SEED_PAIRS: list[tuple[int, int]] = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]

# Final Four regional pairings (which two regions meet in each semi-final)
FINAL_FOUR_PAIRS: list[tuple[str, str]] = [
    ("East", "West"),     # F4_1
    ("South", "Midwest"), # F4_2
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BracketGame:
    """A single game node in the bracket graph.

    Attributes
    ----------
    game_id:
        Unique identifier, e.g. ``"R64_East_1"`` or ``"CHAMP"``.
    round_name:
        Round label: ``"R64"``, ``"R32"``, ``"S16"``, ``"E8"``,
        ``"F4"``, or ``"CHAMP"``.
    region:
        Regional bracket (``"East"``, ``"West"``, ``"South"``,
        ``"Midwest"``) or ``None`` for cross-regional games.
    slot_label:
        Human-readable label, e.g. ``"East 1v16"``.
    left_source:
        Source of the left-side team.  Either an initial seed slot such
        as ``"East_S1"`` or a previous-game winner reference like
        ``"WINNER_R64_East_1"``.
    right_source:
        Source of the right-side team, same format as *left_source*.
    next_game_id:
        ``game_id`` of the game this winner advances to, or ``None``
        for the championship game.
    next_slot:
        ``"left"`` or ``"right"`` indicating which slot the winner fills
        in *next_game_id*.  ``None`` for the championship.
    """

    game_id: str
    round_name: str
    region: str | None
    slot_label: str
    left_source: str
    right_source: str
    next_game_id: str | None
    next_slot: str | None


@dataclass
class BracketState:
    """Snapshot of a tournament bracket before simulation begins.

    Attributes
    ----------
    games:
        Complete ordered list of all :class:`BracketGame` nodes in the
        bracket, from R64 through the championship.
    initial_slots:
        Mapping of seed-slot name → team ID for the starting field,
        e.g. ``{"East_S1": 1234, "East_S2": 5678, ...}``.
    """

    games: list[BracketGame]
    initial_slots: dict[str, int]


@dataclass
class GamePrediction:
    """Model output for a single predicted game.

    Attributes
    ----------
    game_id:
        Identifier of the game being predicted.
    team_a_id:
        ID of Team A (left source).
    team_b_id:
        ID of Team B (right source).
    prob_a_wins:
        Model-predicted probability that Team A wins (0–1).
    predicted_winner_id:
        Predicted winner (``team_a_id`` if ``prob_a_wins >= 0.5``,
        else ``team_b_id``).
    """

    game_id: str
    team_a_id: int
    team_b_id: int
    prob_a_wins: float
    predicted_winner_id: int


@dataclass
class SimulationResult:
    """Result of a single full-bracket simulation.

    Attributes
    ----------
    game_results:
        Mapping of ``game_id → winner_team_id`` for every game played.
    champion_id:
        Team ID of the simulated champion, or ``None`` if the bracket
        could not be completed.
    team_round_reached:
        Mapping of ``team_id → round_name`` for the deepest round each
        team participated in (both winners and losers of that round).
    """

    game_results: dict[str, int]
    champion_id: int | None
    team_round_reached: dict[int, str]


@dataclass
class AggregateSimulationResult:
    """Aggregated statistics from many bracket simulations.

    Attributes
    ----------
    n_sims:
        Number of simulations run.
    champion_probs:
        ``{team_id: probability_of_winning_championship}``.
    round_probs:
        ``{team_id: {round_name: probability_of_reaching_that_round_or_further}}``.
    most_common_bracket:
        :class:`SimulationResult` for the bracket whose champion
        appeared most frequently.  ``None`` if no simulations ran.
    game_win_probs:
        ``{game_id: {team_id: fraction_of_sims_this_team_won}}``.
        Useful for per-game probability tables.
    """

    n_sims: int
    champion_probs: dict[int, float]
    round_probs: dict[int, dict[str, float]]
    most_common_bracket: SimulationResult | None
    game_win_probs: dict[str, dict[int, float]]


@dataclass
class ModelBundle:
    """Container for everything needed to make game predictions.

    Attributes
    ----------
    model:
        Any fitted classifier that exposes ``predict_proba(X) -> array``.
    features:
        Team-feature DataFrame indexed by ``(season, team_id)`` with one
        row per team per season.
    feature_cols:
        Ordered list of column names used as model inputs.
    """

    model: Any
    features: pd.DataFrame
    feature_cols: list[str]
