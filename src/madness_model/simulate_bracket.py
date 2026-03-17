"""
simulate_bracket.py
-------------------
Deterministic and Monte Carlo NCAA bracket simulation.

Two simulation layers are provided:

**Legacy flat-list API** (``simulate_game``, ``simulate_bracket``,
``monte_carlo_simulation``, ``build_predict_fn``)
    The bracket is a flat list of 64 team IDs where adjacent pairs are
    first-round matchups.  Simple but does not encode the real bracket graph.

**Graph-based API** (``load_bracket_structure``, ``build_initial_bracket``,
``predict_game``, ``simulate_single_bracket``, ``simulate_many_brackets``,
``build_most_likely_bracket``)
    The bracket is a directed acyclic graph of :class:`~madness_model.bracket.BracketGame`
    nodes.  Winners advance to the exact correct next slot, mirroring how
    the real NCAA tournament works.

Bracket format (legacy)
-----------------------
A 64-team field is represented as a flat list of 64 team IDs ordered so
that adjacent pairs are first-round matchups (indices 0&1, 2&3, …).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from madness_model.config import NUM_SIMULATIONS, RANDOM_SEED

# Type alias: a predict function takes (team_a_id, team_b_id) and returns P(A wins)
PredictFn = Callable[[int, int], float]

SAFE_PROB_EPS = 1e-6


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


# ===========================================================================
# Graph-based bracket simulation API
# ===========================================================================

def load_bracket_structure(season: int) -> list:  # list[BracketGame]
    """Build the standard 64-team NCAA tournament bracket as a game graph.

    Returns a complete list of :class:`~madness_model.bracket.BracketGame`
    objects (63 games total) ordered from the Round of 64 through the
    Championship.  Each game encodes its left/right team sources and which
    game and slot the winner advances to.

    The ``season`` parameter is accepted for future extension (e.g. adding
    First Four play-in games for seasons ≥ 2011) but the base 64-team
    structure is the same for all modern seasons.

    Parameters
    ----------
    season:
        Tournament season year (currently unused but reserved for
        First Four / play-in support).

    Returns
    -------
    list[BracketGame]
        All 63 tournament games in round order.
    """
    from madness_model.bracket import (
        BracketGame,
        FINAL_FOUR_PAIRS,
        R64_SEED_PAIRS,
        REGIONS,
    )

    games: list[BracketGame] = []

    # ------------------------------------------------------------------
    # Regional games (R64 → R32 → S16 → E8) for each of the 4 regions
    # ------------------------------------------------------------------
    # Final Four slot for each region winner
    f4_slots: dict[str, tuple[str, str]] = {}
    for f4_idx, (region_a, region_b) in enumerate(FINAL_FOUR_PAIRS, start=1):
        f4_game_id = f"F4_{f4_idx}"
        f4_slots[region_a] = (f4_game_id, "left")
        f4_slots[region_b] = (f4_game_id, "right")

    for region in REGIONS:
        # ---- Round of 64 (8 games per region) -------------------------
        for game_n, (seed_a, seed_b) in enumerate(R64_SEED_PAIRS, start=1):
            game_id = f"R64_{region}_{game_n}"
            # Games 1&2 feed R32 game 1, games 3&4 feed R32 game 2, etc.
            r32_n = (game_n + 1) // 2
            next_game_id = f"R32_{region}_{r32_n}"
            next_slot = "left" if game_n % 2 == 1 else "right"

            games.append(
                BracketGame(
                    game_id=game_id,
                    round_name="R64",
                    region=region,
                    slot_label=f"{region} {seed_a}v{seed_b}",
                    left_source=f"{region}_S{seed_a}",
                    right_source=f"{region}_S{seed_b}",
                    next_game_id=next_game_id,
                    next_slot=next_slot,
                )
            )

        # ---- Round of 32 (4 games per region) -------------------------
        for game_n in range(1, 5):
            game_id = f"R32_{region}_{game_n}"
            s16_n = (game_n + 1) // 2
            next_game_id = f"S16_{region}_{s16_n}"
            next_slot = "left" if game_n % 2 == 1 else "right"

            games.append(
                BracketGame(
                    game_id=game_id,
                    round_name="R32",
                    region=region,
                    slot_label=f"{region} R32 game {game_n}",
                    left_source=f"WINNER_R64_{region}_{2 * game_n - 1}",
                    right_source=f"WINNER_R64_{region}_{2 * game_n}",
                    next_game_id=next_game_id,
                    next_slot=next_slot,
                )
            )

        # ---- Sweet 16 (2 games per region) ----------------------------
        for game_n in range(1, 3):
            game_id = f"S16_{region}_{game_n}"
            next_game_id = f"E8_{region}"
            next_slot = "left" if game_n == 1 else "right"

            games.append(
                BracketGame(
                    game_id=game_id,
                    round_name="S16",
                    region=region,
                    slot_label=f"{region} Sweet 16 game {game_n}",
                    left_source=f"WINNER_R32_{region}_{2 * game_n - 1}",
                    right_source=f"WINNER_R32_{region}_{2 * game_n}",
                    next_game_id=next_game_id,
                    next_slot=next_slot,
                )
            )

        # ---- Elite 8 (1 game per region) ------------------------------
        e8_next_game_id, e8_next_slot = f4_slots[region]
        games.append(
            BracketGame(
                game_id=f"E8_{region}",
                round_name="E8",
                region=region,
                slot_label=f"{region} Elite 8",
                left_source=f"WINNER_S16_{region}_1",
                right_source=f"WINNER_S16_{region}_2",
                next_game_id=e8_next_game_id,
                next_slot=e8_next_slot,
            )
        )

    # ------------------------------------------------------------------
    # Final Four (2 games)
    # ------------------------------------------------------------------
    for f4_idx, (region_a, region_b) in enumerate(FINAL_FOUR_PAIRS, start=1):
        f4_game_id = f"F4_{f4_idx}"
        games.append(
            BracketGame(
                game_id=f4_game_id,
                round_name="F4",
                region=None,
                slot_label=f"Final Four {f4_idx} ({region_a} vs {region_b})",
                left_source=f"WINNER_E8_{region_a}",
                right_source=f"WINNER_E8_{region_b}",
                next_game_id="CHAMP",
                next_slot="left" if f4_idx == 1 else "right",
            )
        )

    # ------------------------------------------------------------------
    # Championship (1 game)
    # ------------------------------------------------------------------
    f4_game_ids = [f"F4_{i}" for i in range(1, len(FINAL_FOUR_PAIRS) + 1)]
    games.append(
        BracketGame(
            game_id="CHAMP",
            round_name="CHAMP",
            region=None,
            slot_label="National Championship",
            left_source=f"WINNER_{f4_game_ids[0]}",
            right_source=f"WINNER_{f4_game_ids[1]}",
            next_game_id=None,
            next_slot=None,
        )
    )

    return games


def build_initial_bracket(
    season: int,
    teams_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    region_map: dict | None = None,
) -> object:  # BracketState
    """Construct a :class:`~madness_model.bracket.BracketState` for a season.

    Reads the seedings for *season* from *seeds_df* and maps each seed slot
    (e.g. ``"East_S1"``) to the corresponding team ID.

    Parameters
    ----------
    season:
        Tournament season year.
    teams_df:
        Teams reference DataFrame (columns include ``team_id``).  Currently
        used for validation; pass an empty DataFrame if unavailable.
    seeds_df:
        Seed data for the tournament field.  Expected columns:

        * ``season`` – integer year
        * ``team_id`` – integer team identifier
        * ``seed`` – integer seed (1–16)
        * ``region`` – region identifier.  May be a single letter code
          (``"W"``, ``"X"``, ``"Y"``, ``"Z"``) **or** a full name
          (``"East"``, ``"West"``, ``"South"``, ``"Midwest"``).

    region_map:
        Optional mapping from region letter code to full region name,
        e.g. ``{"W": "West", "X": "East"}``.  Defaults to
        :data:`~madness_model.bracket.DEFAULT_REGION_MAP`.

    Returns
    -------
    BracketState
        Bracket structure plus the initial seed-slot → team-ID mapping.
    """
    from madness_model.bracket import BracketState, DEFAULT_REGION_MAP, REGIONS

    effective_map: dict[str, str] = region_map if region_map is not None else DEFAULT_REGION_MAP

    games = load_bracket_structure(season)

    season_seeds = seeds_df[seeds_df["season"] == season].copy()

    initial_slots: dict[str, int] = {}
    for _, row in season_seeds.iterrows():
        region_raw = str(row["region"])
        # Accept either a full name or a letter code
        if region_raw in REGIONS:
            region_full = region_raw
        else:
            region_full = effective_map.get(region_raw)
            if region_full is None:
                continue  # skip unknown region codes

        seed = int(row["seed"])
        team_id = int(row["team_id"])
        slot_name = f"{region_full}_S{seed}"
        initial_slots[slot_name] = team_id

    return BracketState(games=games, initial_slots=initial_slots)


def predict_game(
    team_a_id: int,
    team_b_id: int,
    season: int,
    model_bundle: object,  # ModelBundle
    game_id: str = "",
) -> object:  # GamePrediction
    """Predict the outcome of a single matchup.

    Parameters
    ----------
    team_a_id:
        ID of Team A.
    team_b_id:
        ID of Team B.
    season:
        Tournament season year used for feature lookup.
    model_bundle:
        :class:`~madness_model.bracket.ModelBundle` with a fitted model,
        team features, and feature column list.
    game_id:
        Optional identifier for the game being predicted (informational).

    Returns
    -------
    GamePrediction
        Predicted probability and winner for the matchup.
    """
    from madness_model.bracket import GamePrediction
    from madness_model.build_matchups import build_matchup_row

    row = build_matchup_row(season, team_a_id, team_b_id, model_bundle.features)
    X = pd.DataFrame([row])[model_bundle.feature_cols].values
    prob_a_wins = float(model_bundle.model.predict_proba(X)[0, 1])
    if np.isnan(prob_a_wins):
        raise ValueError("Model returned NaN probability.")
    if prob_a_wins < 0.0 or prob_a_wins > 1.0:
        raise ValueError(
            f"Model returned invalid probability {prob_a_wins}; expected [0, 1]."
        )
    prob_a_wins = float(np.clip(prob_a_wins, SAFE_PROB_EPS, 1.0 - SAFE_PROB_EPS))
    predicted_winner_id = team_a_id if prob_a_wins >= 0.5 else team_b_id

    return GamePrediction(
        game_id=game_id,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        prob_a_wins=prob_a_wins,
        predicted_winner_id=predicted_winner_id,
    )


def simulate_single_bracket(
    bracket_state: object,  # BracketState
    season: int,
    model_bundle: object,  # ModelBundle
    mode: str = "deterministic",
    random_state: int | None = None,
) -> object:  # SimulationResult
    """Simulate a complete bracket once using the bracket graph.

    Advances winners through the correct next game slot exactly as the
    real NCAA tournament bracket works.

    Parameters
    ----------
    bracket_state:
        :class:`~madness_model.bracket.BracketState` with the bracket
        structure and initial seed-to-team assignments.
    season:
        Tournament season year.
    model_bundle:
        :class:`~madness_model.bracket.ModelBundle` for game predictions.
    mode:
        ``"deterministic"`` – always advance the higher-probability team.
        ``"stochastic"`` – sample winners according to predicted probabilities.
    random_state:
        Integer seed for reproducibility in stochastic mode.

    Returns
    -------
    SimulationResult
        Per-game winners, champion ID, and each team's deepest round.
        ``SimulationResult.champion_id`` is ``None`` only if the
        championship game could not be played due to unresolved bracket
        slots (i.e. an incomplete *bracket_state*); callers should treat
        a ``None`` champion as an error indicator.
    """
    from madness_model.bracket import ROUND_NAMES, SimulationResult
    from madness_model.build_matchups import build_matchup_row

    _validate_bracket_state(bracket_state)
    deterministic = mode == "deterministic"
    rng = random.Random(random_state)
    team_seed_map = _build_team_seed_map(bracket_state.initial_slots)

    # Slots dict: maps slot names to resolved team IDs.
    # Starts with the initial seed assignments; winner slots are added
    # as games complete.
    slots: dict[str, int] = dict(bracket_state.initial_slots)

    game_results: dict[str, int] = {}
    team_round_reached: dict[int, str] = {}

    for round_name in ROUND_NAMES:
        round_games = [g for g in bracket_state.games if g.round_name == round_name]

        for game in round_games:
            team_a = slots.get(game.left_source)
            team_b = slots.get(game.right_source)

            if team_a is None or team_b is None:
                def _source_hint(source: str, resolved: int | None) -> str:
                    if resolved is not None:
                        return f"{source!r} -> {resolved}"
                    if source.startswith("WINNER_"):
                        return f"{source!r} unresolved (upstream winner not available)"
                    return f"{source!r} unresolved (missing initial slot assignment)"

                raise ValueError(
                    f"Game {game.game_id!r} cannot start: unresolved participant slot(s) "
                    f"{_source_hint(game.left_source, team_a)}, "
                    f"{_source_hint(game.right_source, team_b)}."
                )

            # Get win probability for team_a
            row = build_matchup_row(season, team_a, team_b, model_bundle.features)
            X = pd.DataFrame([row])[model_bundle.feature_cols].values
            prob_a = float(model_bundle.model.predict_proba(X)[0, 1])
            if np.isnan(prob_a):
                raise ValueError(f"Model returned NaN probability for game {game.game_id}.")
            if prob_a < 0.0 or prob_a > 1.0:
                raise ValueError(
                    f"Model returned invalid probability {prob_a} for game {game.game_id}; "
                    "expected [0, 1]."
                )
            prob_a = float(np.clip(prob_a, SAFE_PROB_EPS, 1.0 - SAFE_PROB_EPS))

            if deterministic:
                if prob_a > 0.5:
                    winner = team_a
                elif prob_a < 0.5:
                    winner = team_b
                else:
                    winner = _break_deterministic_tie(
                        team_a=team_a,
                        team_b=team_b,
                        team_seed_map=team_seed_map,
                        season=season,
                        features=model_bundle.features,
                    )
            else:
                winner = team_a if rng.random() < prob_a else team_b

            loser = team_b if winner == team_a else team_a

            # Record the game result and expose the winner slot for downstream games
            game_results[game.game_id] = winner
            slots[f"WINNER_{game.game_id}"] = winner

            # Both participants "reached" this round (winner will be updated further)
            team_round_reached[winner] = round_name
            team_round_reached[loser] = round_name

    champion_id: int | None = game_results.get("CHAMP")

    return SimulationResult(
        game_results=game_results,
        champion_id=champion_id,
        team_round_reached=team_round_reached,
    )


def simulate_many_brackets(
    bracket_state: object,  # BracketState
    season: int,
    model_bundle: object,  # ModelBundle
    n_sims: int = NUM_SIMULATIONS,
    random_state: int | None = None,
) -> object:  # AggregateSimulationResult
    """Run many stochastic simulations and aggregate bracket statistics.

    Parameters
    ----------
    bracket_state:
        :class:`~madness_model.bracket.BracketState` with the bracket
        structure and initial seed assignments.
    season:
        Tournament season year.
    model_bundle:
        :class:`~madness_model.bracket.ModelBundle` for game predictions.
    n_sims:
        Number of independent Monte Carlo simulations to run.
    random_state:
        Master integer seed.  Each simulation receives a derived seed so
        that results are fully reproducible.

    Returns
    -------
    AggregateSimulationResult
        Champion probabilities, round-reaching probabilities, per-game
        win fractions, and the most common bracket.
    """
    from madness_model.bracket import AggregateSimulationResult, ROUND_NAMES, ROUND_ORDER

    master_rng = random.Random(random_state)

    # Collect all results so we can find the most common bracket
    all_results = []
    champion_counts: Counter = Counter()
    game_win_counts: dict[str, Counter] = defaultdict(Counter)
    round_reach_counts: dict[int, Counter] = defaultdict(Counter)

    for _ in range(n_sims):
        seed = master_rng.randint(0, 2**31 - 1)
        result = simulate_single_bracket(
            bracket_state, season, model_bundle, mode="stochastic", random_state=seed
        )
        all_results.append(result)

        if result.champion_id is not None:
            champion_counts[result.champion_id] += 1

        for game_id, winner in result.game_results.items():
            game_win_counts[game_id][winner] += 1

        for team, round_name in result.team_round_reached.items():
            round_reach_counts[team][round_name] += 1

    # --- Champion probabilities -------------------------------------------
    champion_probs: dict[int, float] = {
        team: count / n_sims for team, count in champion_counts.items()
    }

    # --- Round-reaching probabilities -------------------------------------
    # P(team reaches round R) = fraction of sims where team's deepest round
    # is R or further along the bracket.
    all_teams = set(round_reach_counts.keys())
    round_probs: dict[int, dict[str, float]] = {}
    for team in all_teams:
        team_counts = round_reach_counts[team]
        # deepest_round_index for this team in each sim
        team_round_probs: dict[str, float] = {}
        for rn in ROUND_NAMES:
            rn_idx = ROUND_ORDER[rn]
            # Count sims where team's deepest round index >= rn_idx
            count = sum(
                c
                for reached_round, c in team_counts.items()
                if ROUND_ORDER.get(reached_round, -1) >= rn_idx
            )
            team_round_probs[rn] = count / n_sims
        round_probs[team] = team_round_probs

    # --- Per-game win fractions -------------------------------------------
    game_win_probs: dict[str, dict[int, float]] = {
        game_id: {team: count / n_sims for team, count in counts.items()}
        for game_id, counts in game_win_counts.items()
    }

    # --- Most common bracket (by most frequent champion) ------------------
    most_common_bracket = None
    if all_results and champion_counts:
        most_common_champion = champion_counts.most_common(1)[0][0]
        most_common_bracket = next(
            r for r in all_results if r.champion_id == most_common_champion
        )

    return AggregateSimulationResult(
        n_sims=n_sims,
        champion_probs=champion_probs,
        round_probs=round_probs,
        most_common_bracket=most_common_bracket,
        game_win_probs=game_win_probs,
    )


def build_most_likely_bracket(
    bracket_state: object,  # BracketState
    season: int,
    model_bundle: object,  # ModelBundle
) -> object:  # SimulationResult
    """Build the fully deterministic "most likely" bracket.

    At every game, the team with the higher predicted win probability
    advances.  Equivalent to calling :func:`simulate_single_bracket`
    with ``mode="deterministic"``.

    Parameters
    ----------
    bracket_state:
        :class:`~madness_model.bracket.BracketState` with the bracket
        structure and initial seed assignments.
    season:
        Tournament season year.
    model_bundle:
        :class:`~madness_model.bracket.ModelBundle` for game predictions.

    Returns
    -------
    SimulationResult
        The predicted bracket with every game decided deterministically.
    """
    return simulate_single_bracket(
        bracket_state, season, model_bundle, mode="deterministic"
    )


def _build_team_seed_map(initial_slots: dict[str, int]) -> dict[int, int]:
    team_seed_map: dict[int, int] = {}
    for slot_name, team_id in initial_slots.items():
        if "_S" not in slot_name:
            continue
        seed = int(slot_name.split("_S", 1)[1])
        if team_id in team_seed_map:
            raise ValueError(f"Duplicate team assignment detected for team_id={team_id}.")
        team_seed_map[team_id] = seed
    return team_seed_map


def _pick_elo_column(features: pd.DataFrame) -> str | None:
    for col in ("elo", "elo_post", "elo_pre", "elo_rating"):
        if col in features.columns:
            return col
    return None


def _break_deterministic_tie(
    team_a: int,
    team_b: int,
    team_seed_map: dict[int, int],
    season: int,
    features: pd.DataFrame,
) -> int:
    seed_a = team_seed_map.get(team_a)
    seed_b = team_seed_map.get(team_b)
    if seed_a is not None and seed_b is not None and seed_a != seed_b:
        return team_a if seed_a < seed_b else team_b

    elo_col = _pick_elo_column(features)
    if elo_col is not None:
        try:
            elo_a = float(features.loc[(season, team_a), elo_col])
            elo_b = float(features.loc[(season, team_b), elo_col])
            if abs(elo_a - elo_b) > 1e-9:
                return team_a if elo_a > elo_b else team_b
        except KeyError:
            pass

    return team_a if team_a < team_b else team_b


def _validate_bracket_state(bracket_state: object) -> None:
    from madness_model.bracket import ROUND_NAMES

    games = list(bracket_state.games)
    game_ids = {g.game_id for g in games}
    if len(game_ids) != len(games):
        raise ValueError("Bracket contains duplicate game IDs.")

    if "CHAMP" not in game_ids:
        raise ValueError("Bracket must contain CHAMP game.")

    for g in games:
        if g.round_name not in ROUND_NAMES:
            raise ValueError(f"Game {g.game_id} has invalid round {g.round_name!r}.")
        if not g.left_source or not g.right_source:
            raise ValueError(f"Game {g.game_id} must have exactly two input sources.")
        if g.game_id == "CHAMP":
            if g.next_game_id is not None:
                raise ValueError("Championship game cannot have next_game_id.")
        else:
            if g.next_game_id is None or g.next_game_id not in game_ids:
                raise ValueError(
                    f"Game {g.game_id} has invalid next_game_id={g.next_game_id!r}."
                )

        for source in (g.left_source, g.right_source):
            if source.startswith("WINNER_"):
                ref_id = source[len("WINNER_") :]
                if ref_id not in game_ids:
                    raise ValueError(
                        f"Game {g.game_id} has invalid source reference {source!r}."
                    )
            elif source not in bracket_state.initial_slots:
                raise ValueError(
                    f"Game {g.game_id} has unresolved initial source slot {source!r}."
                )

    indegree: dict[str, int] = {g.game_id: 0 for g in games}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for g in games:
        if g.next_game_id is not None:
            outgoing[g.game_id].append(g.next_game_id)
            indegree[g.next_game_id] += 1

    queue = [gid for gid, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nxt in outgoing[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited != len(games):
        raise ValueError("Bracket graph contains a cycle.")
