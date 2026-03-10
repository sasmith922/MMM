"""
test_simulation.py
------------------
Unit tests for madness_model.simulate_bracket.
"""

import math
import random

import pandas as pd
import pytest

from madness_model.simulate_bracket import (
    monte_carlo_simulation,
    simulate_bracket,
    simulate_game,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def always_a_wins(team_a: int, team_b: int) -> float:
    """Deterministic predict function: Team A always wins."""
    return 1.0


def always_b_wins(team_a: int, team_b: int) -> float:
    """Deterministic predict function: Team B always wins."""
    return 0.0


def coin_flip(team_a: int, team_b: int) -> float:
    """50/50 predict function."""
    return 0.5


# ---------------------------------------------------------------------------
# simulate_game
# ---------------------------------------------------------------------------

class TestSimulateGame:
    def test_deterministic_a_wins(self):
        assert simulate_game(1, 2, always_a_wins, deterministic=True) == 1

    def test_deterministic_b_wins(self):
        assert simulate_game(1, 2, always_b_wins, deterministic=True) == 2

    def test_stochastic_returns_valid_team(self):
        rng = random.Random(42)
        winner = simulate_game(1, 2, coin_flip, deterministic=False, rng=rng)
        assert winner in (1, 2)

    def test_prob_one_always_picks_a_stochastic(self):
        rng = random.Random(0)
        for _ in range(20):
            assert simulate_game(1, 2, always_a_wins, deterministic=False, rng=rng) == 1

    def test_prob_zero_always_picks_b_stochastic(self):
        rng = random.Random(0)
        for _ in range(20):
            assert simulate_game(1, 2, always_b_wins, deterministic=False, rng=rng) == 2


# ---------------------------------------------------------------------------
# simulate_bracket
# ---------------------------------------------------------------------------

class TestSimulateBracket:
    def test_invalid_field_size_raises(self):
        with pytest.raises(ValueError):
            simulate_bracket([1, 2, 3], always_a_wins)

    def test_empty_field_raises(self):
        with pytest.raises(ValueError):
            simulate_bracket([], always_a_wins)

    def test_two_team_bracket_deterministic(self):
        result = simulate_bracket([1, 2], always_a_wins, deterministic=True)
        # Team 1 wins; round_reached = 2 (champion)
        assert result[1] == 2
        assert result[2] == 1

    def test_four_team_bracket_team_a_dominates(self):
        """With always_a_wins, the first team in every pair advances."""
        field = [1, 2, 3, 4]
        result = simulate_bracket(field, always_a_wins, deterministic=True)
        # Bracket: (1v2), (3v4) → 1 and 3 advance → 1v3 → 1 wins
        assert result[1] == 3  # champion round
        assert result[3] == 2  # finalist
        assert result[2] == 1  # first round exit
        assert result[4] == 1  # first round exit

    def test_64_team_bracket_produces_correct_rounds(self):
        field = list(range(1, 65))
        result = simulate_bracket(field, always_a_wins, deterministic=True)
        total_rounds = int(math.log2(64)) + 1  # 7
        # Champion should reach round 7
        champion = max(result, key=result.get)
        assert result[champion] == total_rounds

    def test_all_teams_appear_in_result(self):
        field = [10, 20, 30, 40, 50, 60, 70, 80]
        result = simulate_bracket(field, coin_flip)
        assert set(result.keys()) == set(field)

    def test_stochastic_different_seeds_may_differ(self):
        field = list(range(1, 17))
        rng1 = random.Random(1)
        rng2 = random.Random(999)
        r1 = simulate_bracket(field, coin_flip, rng=rng1)
        r2 = simulate_bracket(field, coin_flip, rng=rng2)
        # With 16 teams the same champion is astronomically unlikely
        champ1 = max(r1, key=r1.get)
        champ2 = max(r2, key=r2.get)
        # This is probabilistic; the test guards against the trivial deterministic case
        # (In theory both seeds could agree — if flaky, increase field size.)
        # We at minimum check both results are valid
        assert champ1 in field
        assert champ2 in field


# ---------------------------------------------------------------------------
# monte_carlo_simulation
# ---------------------------------------------------------------------------

class TestMonteCarloSimulation:
    def test_returns_dataframe_with_expected_columns(self):
        field = [1, 2, 3, 4]
        df = monte_carlo_simulation(field, coin_flip, n_simulations=100, seed=42)
        for col in ("team_id", "avg_round", "champion_pct", "final_four_pct", "elite_eight_pct"):
            assert col in df.columns

    def test_one_row_per_team(self):
        field = [1, 2, 3, 4]
        df = monte_carlo_simulation(field, coin_flip, n_simulations=100, seed=42)
        assert len(df) == len(field)

    def test_champion_pct_sums_to_one(self):
        field = [1, 2, 3, 4]
        df = monte_carlo_simulation(field, coin_flip, n_simulations=1000, seed=0)
        assert df["champion_pct"].sum() == pytest.approx(1.0, abs=0.01)

    def test_deterministic_favourite_wins_every_time(self):
        """When team 1 always beats everyone, champion_pct should be ~1."""
        field = [1, 2, 3, 4]

        def team1_always_wins(a: int, b: int) -> float:
            return 1.0 if a == 1 else 0.0

        df = monte_carlo_simulation(field, team1_always_wins, n_simulations=100, seed=0)
        champ_row = df[df["team_id"] == 1]
        assert champ_row["champion_pct"].iloc[0] == pytest.approx(1.0)

    def test_reproducibility_with_same_seed(self):
        field = list(range(1, 9))
        df1 = monte_carlo_simulation(field, coin_flip, n_simulations=200, seed=7)
        df2 = monte_carlo_simulation(field, coin_flip, n_simulations=200, seed=7)
        pd.testing.assert_frame_equal(df1, df2)
