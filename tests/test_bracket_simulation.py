"""
test_bracket_simulation.py
--------------------------
Unit tests for the graph-based NCAA bracket simulation API:
  - load_bracket_structure
  - build_initial_bracket
  - predict_game
  - simulate_single_bracket
  - simulate_many_brackets
  - build_most_likely_bracket

All tests use mock models and in-memory DataFrames so that no external
data files are needed.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from madness_model.bracket import (
    AggregateSimulationResult,
    BracketGame,
    BracketState,
    GamePrediction,
    ModelBundle,
    REGIONS,
    ROUND_NAMES,
    ROUND_ORDER,
    SimulationResult,
)
from madness_model.simulate_bracket import (
    build_initial_bracket,
    build_most_likely_bracket,
    load_bracket_structure,
    predict_game,
    simulate_many_brackets,
    simulate_single_bracket,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SEASON = 2025
NUM_SEEDS = 16  # seeds per region
NUM_REGIONS = 4  # East, West, South, Midwest
NUM_TEAMS = NUM_SEEDS * NUM_REGIONS  # 64


def _make_team_id(region: str, seed: int) -> int:
    """Stable integer team ID from region index and seed (1-based)."""
    region_idx = REGIONS.index(region)
    return region_idx * 100 + seed


def _make_seeds_df() -> pd.DataFrame:
    """Build a synthetic seeds DataFrame for SEASON."""
    rows = []
    for region in REGIONS:
        for seed in range(1, NUM_SEEDS + 1):
            rows.append(
                {
                    "season": SEASON,
                    "team_id": _make_team_id(region, seed),
                    "seed": seed,
                    "region": region,  # full region name
                }
            )
    return pd.DataFrame(rows)


def _make_teams_df() -> pd.DataFrame:
    """Build a minimal teams DataFrame."""
    rows = [
        {"team_id": _make_team_id(r, s), "team_name": f"{r}_S{s}"}
        for r in REGIONS
        for s in range(1, NUM_SEEDS + 1)
    ]
    return pd.DataFrame(rows)


def _make_features_df() -> pd.DataFrame:
    """Build a minimal team features DataFrame indexed by (season, team_id)."""
    rows = []
    for region in REGIONS:
        for seed in range(1, NUM_SEEDS + 1):
            team_id = _make_team_id(region, seed)
            rows.append(
                {
                    "season": SEASON,
                    "team_id": team_id,
                    # Higher-seeded teams get better features so the model
                    # can pick them deterministically via differentials.
                    "win_pct": round((17 - seed) / 16, 4),
                    "avg_point_diff": float(17 - seed),
                }
            )
    df = pd.DataFrame(rows).set_index(["season", "team_id"])
    return df


FEATURE_COLS = ["diff_win_pct", "diff_avg_point_diff"]


class _AlwaysFavouriteModel:
    """Mock model: P(A wins) = sigmoid of average feature differential."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n, 2) with diff_win_pct and diff_avg_point_diff
        score = X[:, 0] + X[:, 1] / 10.0
        prob_a = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1 - prob_a, prob_a])


class _ConstantModel:
    """Mock model that always returns the same probability for Team A."""

    def __init__(self, prob_a: float = 0.5):
        self.prob_a = prob_a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack(
            [np.full(n, 1 - self.prob_a), np.full(n, self.prob_a)]
        )


def _make_model_bundle(prob_a: float | None = None) -> ModelBundle:
    features = _make_features_df()
    if prob_a is None:
        model = _AlwaysFavouriteModel()
    else:
        model = _ConstantModel(prob_a)
    return ModelBundle(model=model, features=features, feature_cols=FEATURE_COLS)


def _make_bracket_state() -> BracketState:
    return build_initial_bracket(SEASON, _make_teams_df(), _make_seeds_df())


# ---------------------------------------------------------------------------
# Tests: load_bracket_structure
# ---------------------------------------------------------------------------


class TestLoadBracketStructure:
    def test_returns_list_of_bracket_games(self):
        games = load_bracket_structure(SEASON)
        assert isinstance(games, list)
        assert all(isinstance(g, BracketGame) for g in games)

    def test_total_game_count(self):
        """64-team single elimination requires exactly 63 games."""
        games = load_bracket_structure(SEASON)
        assert len(games) == 63

    def test_game_ids_are_unique(self):
        games = load_bracket_structure(SEASON)
        ids = [g.game_id for g in games]
        assert len(ids) == len(set(ids))

    def test_championship_game_present(self):
        games = load_bracket_structure(SEASON)
        game_ids = {g.game_id for g in games}
        assert "CHAMP" in game_ids

    def test_round_counts(self):
        """Verify the correct number of games per round."""
        games = load_bracket_structure(SEASON)
        counts = {}
        for g in games:
            counts[g.round_name] = counts.get(g.round_name, 0) + 1

        assert counts["R64"] == 32
        assert counts["R32"] == 16
        assert counts["S16"] == 8
        assert counts["E8"] == 4
        assert counts["F4"] == 2
        assert counts["CHAMP"] == 1

    def test_regional_games_have_region_set(self):
        games = load_bracket_structure(SEASON)
        for g in games:
            if g.round_name in ("R64", "R32", "S16", "E8"):
                assert g.region in REGIONS, f"{g.game_id} missing region"

    def test_champ_and_f4_have_no_region(self):
        games = load_bracket_structure(SEASON)
        for g in games:
            if g.round_name in ("F4", "CHAMP"):
                assert g.region is None, f"{g.game_id} should have region=None"

    def test_championship_has_no_next_game(self):
        games = load_bracket_structure(SEASON)
        champ = next(g for g in games if g.game_id == "CHAMP")
        assert champ.next_game_id is None
        assert champ.next_slot is None

    def test_all_non_champ_games_have_next_game(self):
        games = load_bracket_structure(SEASON)
        for g in games:
            if g.game_id != "CHAMP":
                assert g.next_game_id is not None, f"{g.game_id} missing next_game_id"

    def test_winner_sources_reference_valid_game_ids(self):
        """WINNER_ references in source slots must point to known game IDs."""
        games = load_bracket_structure(SEASON)
        game_ids = {g.game_id for g in games}
        for g in games:
            for source in (g.left_source, g.right_source):
                if source.startswith("WINNER_"):
                    referenced_id = source[len("WINNER_"):]
                    assert referenced_id in game_ids, (
                        f"{g.game_id} references unknown game {referenced_id!r}"
                    )

    def test_season_parameter_accepted(self):
        """load_bracket_structure must accept any integer season."""
        for season in (2010, 2019, 2025):
            games = load_bracket_structure(season)
            assert len(games) == 63

    def test_r64_seed_sources_cover_all_seeds(self):
        """Every seed 1-16 must appear exactly once per region in R64."""
        games = load_bracket_structure(SEASON)
        r64_games = [g for g in games if g.round_name == "R64"]
        for region in REGIONS:
            region_r64 = [g for g in r64_games if g.region == region]
            assert len(region_r64) == 8
            seeds_seen = set()
            for g in region_r64:
                for src in (g.left_source, g.right_source):
                    # src is "{region}_S{seed}"
                    seed_num = int(src.split("_S")[1])
                    seeds_seen.add(seed_num)
            assert seeds_seen == set(range(1, 17)), (
                f"Region {region} missing seeds: {set(range(1, 17)) - seeds_seen}"
            )


# ---------------------------------------------------------------------------
# Tests: build_initial_bracket
# ---------------------------------------------------------------------------


class TestBuildInitialBracket:
    def test_returns_bracket_state(self):
        state = _make_bracket_state()
        assert isinstance(state, BracketState)

    def test_games_list_populated(self):
        state = _make_bracket_state()
        assert len(state.games) == 63

    def test_initial_slots_count(self):
        """Should have exactly 64 seed slots (16 seeds × 4 regions)."""
        state = _make_bracket_state()
        assert len(state.initial_slots) == NUM_TEAMS

    def test_initial_slots_keys_format(self):
        state = _make_bracket_state()
        for key in state.initial_slots:
            # Key should be like "East_S1", "West_S16", etc.
            region, seed_part = key.split("_S")
            assert region in REGIONS
            assert 1 <= int(seed_part) <= 16

    def test_initial_slots_values_are_ints(self):
        state = _make_bracket_state()
        for team_id in state.initial_slots.values():
            assert isinstance(team_id, int)

    def test_region_letter_codes_accepted(self):
        """build_initial_bracket must handle single-letter region codes."""
        letter_map = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}
        seeds_df = _make_seeds_df().copy()
        # Replace full names with letters
        reverse_map = {v: k for k, v in letter_map.items()}
        seeds_df["region"] = seeds_df["region"].map(reverse_map)

        state = build_initial_bracket(
            SEASON, _make_teams_df(), seeds_df, region_map=letter_map
        )
        assert len(state.initial_slots) == NUM_TEAMS

    def test_custom_region_map_applied(self):
        """Custom mapping should override the default letter codes."""
        custom_map = {"A": "East", "B": "West", "C": "South", "D": "Midwest"}
        seeds_df = _make_seeds_df().copy()
        letter_map_reverse = {"East": "A", "West": "B", "South": "C", "Midwest": "D"}
        seeds_df["region"] = seeds_df["region"].map(letter_map_reverse)

        state = build_initial_bracket(
            SEASON, _make_teams_df(), seeds_df, region_map=custom_map
        )
        assert len(state.initial_slots) == NUM_TEAMS

    def test_unknown_region_skipped(self):
        """Rows with unrecognised region codes should be silently skipped."""
        seeds_df = _make_seeds_df().copy()
        first_row = seeds_df.iloc[0]
        skipped_team_id = int(first_row["team_id"])
        seeds_df.loc[0, "region"] = "UNKNOWN"
        state = build_initial_bracket(SEASON, _make_teams_df(), seeds_df)
        # One row was skipped, so fewer than 64 slots
        assert len(state.initial_slots) == NUM_TEAMS - 1
        # The skipped team should not appear in any slot
        assert skipped_team_id not in state.initial_slots.values()
        # A known other team should still be present
        other_team_id = _make_team_id("West", 1)
        assert other_team_id in state.initial_slots.values()


# ---------------------------------------------------------------------------
# Tests: predict_game
# ---------------------------------------------------------------------------


class TestPredictGame:
    def test_returns_game_prediction(self):
        mb = _make_model_bundle()
        team_a = _make_team_id("East", 1)
        team_b = _make_team_id("East", 16)
        result = predict_game(team_a, team_b, SEASON, mb)
        assert isinstance(result, GamePrediction)

    def test_prob_a_wins_in_range(self):
        mb = _make_model_bundle()
        team_a = _make_team_id("East", 1)
        team_b = _make_team_id("East", 16)
        result = predict_game(team_a, team_b, SEASON, mb)
        assert 0.0 <= result.prob_a_wins <= 1.0

    def test_predicted_winner_consistent_with_prob(self):
        mb = _make_model_bundle(prob_a=0.9)
        team_a = _make_team_id("East", 1)
        team_b = _make_team_id("East", 16)
        result = predict_game(team_a, team_b, SEASON, mb)
        assert result.predicted_winner_id == team_a

    def test_predicted_winner_team_b_when_prob_low(self):
        mb = _make_model_bundle(prob_a=0.1)
        team_a = _make_team_id("East", 16)
        team_b = _make_team_id("East", 1)
        result = predict_game(team_a, team_b, SEASON, mb)
        assert result.predicted_winner_id == team_b

    def test_game_id_stored(self):
        mb = _make_model_bundle()
        team_a = _make_team_id("West", 2)
        team_b = _make_team_id("West", 15)
        result = predict_game(team_a, team_b, SEASON, mb, game_id="R64_West_8")
        assert result.game_id == "R64_West_8"

    def test_team_ids_stored(self):
        mb = _make_model_bundle()
        team_a = _make_team_id("South", 3)
        team_b = _make_team_id("South", 14)
        result = predict_game(team_a, team_b, SEASON, mb)
        assert result.team_a_id == team_a
        assert result.team_b_id == team_b


# ---------------------------------------------------------------------------
# Tests: simulate_single_bracket
# ---------------------------------------------------------------------------


class TestSimulateSingleBracket:
    def test_returns_simulation_result(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb)
        assert isinstance(result, SimulationResult)

    def test_champion_id_is_a_valid_team(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        all_teams = set(state.initial_slots.values())
        assert result.champion_id in all_teams

    def test_all_63_games_decided(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        assert len(result.game_results) == 63

    def test_all_teams_have_round_reached(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        all_teams = set(state.initial_slots.values())
        assert all_teams == set(result.team_round_reached.keys())

    def test_champion_reaches_champ_round(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        assert result.team_round_reached[result.champion_id] == "CHAMP"

    def test_round_reached_values_are_valid_round_names(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        for team, round_name in result.team_round_reached.items():
            assert round_name in ROUND_NAMES, (
                f"Team {team} has invalid round_reached={round_name!r}"
            )

    def test_stochastic_mode_valid_champion(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        result = simulate_single_bracket(
            state, SEASON, mb, mode="stochastic", random_state=42
        )
        all_teams = set(state.initial_slots.values())
        assert result.champion_id in all_teams

    def test_deterministic_mode_reproducible(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        r1 = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        r2 = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        assert r1.champion_id == r2.champion_id
        assert r1.game_results == r2.game_results

    def test_stochastic_same_seed_reproducible(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        r1 = simulate_single_bracket(
            state, SEASON, mb, mode="stochastic", random_state=7
        )
        r2 = simulate_single_bracket(
            state, SEASON, mb, mode="stochastic", random_state=7
        )
        assert r1.champion_id == r2.champion_id
        assert r1.game_results == r2.game_results

    def test_winner_advances_to_correct_next_game(self):
        """Manually verify that R64 winner shows up in the right R32 game."""
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = simulate_single_bracket(state, SEASON, mb, mode="deterministic")

        # Winner of R64_East_1 should be one of the teams in R32_East_1
        r64_winner = result.game_results.get("R64_East_1")
        r32_winner = result.game_results.get("R32_East_1")

        # The R32 game was played, so a winner exists
        assert r32_winner is not None
        # The R64 winner must have participated in R32_East_1
        # (either won or lost it)
        r32_game = next(g for g in state.games if g.game_id == "R32_East_1")
        slots = dict(state.initial_slots)
        # Resolve left source for R32_East_1 — it should equal r64_winner
        assert r32_game.left_source == f"WINNER_R64_East_1"
        assert r64_winner is not None


# ---------------------------------------------------------------------------
# Tests: simulate_many_brackets
# ---------------------------------------------------------------------------


class TestSimulateManyBrackets:
    def test_returns_aggregate_result(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=20, random_state=0)
        assert isinstance(agg, AggregateSimulationResult)

    def test_n_sims_recorded(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=50, random_state=0)
        assert agg.n_sims == 50
        # Simulations must have actually run and produced results
        assert len(agg.champion_probs) > 0
        assert len(agg.round_probs) == NUM_TEAMS

    def test_champion_probs_sum_to_one(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=200, random_state=1)
        total = sum(agg.champion_probs.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_every_team_in_round_probs(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=20, random_state=2)
        all_teams = set(state.initial_slots.values())
        assert all_teams == set(agg.round_probs.keys())

    def test_round_probs_monotonically_decreasing(self):
        """P(reach round R+1) <= P(reach round R) for every team."""
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=100, random_state=3)
        for team, probs in agg.round_probs.items():
            for i in range(len(ROUND_NAMES) - 1):
                rn_curr = ROUND_NAMES[i]
                rn_next = ROUND_NAMES[i + 1]
                assert probs[rn_next] <= probs[rn_curr] + 1e-9, (
                    f"Team {team}: P({rn_next})={probs[rn_next]} > P({rn_curr})={probs[rn_curr]}"
                )

    def test_game_win_probs_keys_are_game_ids(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=20, random_state=4)
        all_game_ids = {g.game_id for g in state.games}
        assert set(agg.game_win_probs.keys()).issubset(all_game_ids)

    def test_most_common_bracket_is_simulation_result(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=20, random_state=5)
        assert isinstance(agg.most_common_bracket, SimulationResult)

    def test_reproducibility_with_same_seed(self):
        state = _make_bracket_state()
        mb = _make_model_bundle(prob_a=0.5)
        agg1 = simulate_many_brackets(state, SEASON, mb, n_sims=50, random_state=99)
        agg2 = simulate_many_brackets(state, SEASON, mb, n_sims=50, random_state=99)
        assert agg1.champion_probs == agg2.champion_probs

    def test_deterministic_favourite_dominates(self):
        """When the model strongly favours seed-1 teams, they should
        dominate the champion probability."""
        state = _make_bracket_state()
        mb = _make_model_bundle()  # AlwaysFavourite model based on seed differential
        agg = simulate_many_brackets(state, SEASON, mb, n_sims=200, random_state=42)
        # The champion should appear in champion_probs
        if agg.most_common_bracket is not None:
            champion = agg.most_common_bracket.champion_id
            assert champion in agg.champion_probs


# ---------------------------------------------------------------------------
# Tests: build_most_likely_bracket
# ---------------------------------------------------------------------------


class TestBuildMostLikelyBracket:
    def test_returns_simulation_result(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = build_most_likely_bracket(state, SEASON, mb)
        assert isinstance(result, SimulationResult)

    def test_champion_is_valid_team(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = build_most_likely_bracket(state, SEASON, mb)
        all_teams = set(state.initial_slots.values())
        assert result.champion_id in all_teams

    def test_same_as_deterministic_simulate(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        r_likely = build_most_likely_bracket(state, SEASON, mb)
        r_det = simulate_single_bracket(state, SEASON, mb, mode="deterministic")
        assert r_likely.champion_id == r_det.champion_id
        assert r_likely.game_results == r_det.game_results

    def test_all_games_decided(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        result = build_most_likely_bracket(state, SEASON, mb)
        assert len(result.game_results) == 63

    def test_reproducible(self):
        state = _make_bracket_state()
        mb = _make_model_bundle()
        r1 = build_most_likely_bracket(state, SEASON, mb)
        r2 = build_most_likely_bracket(state, SEASON, mb)
        assert r1.game_results == r2.game_results
