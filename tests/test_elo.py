"""
test_elo.py
-----------
Unit tests for the Elo rating system in madness_model.elo.
"""

import pytest

from madness_model.elo import (
    DEFAULT_ELO,
    expected_score,
    revert_to_mean,
    update_elos,
)


class TestExpectedScore:
    def test_equal_ratings_returns_half(self):
        assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_higher_rating_gives_higher_probability(self):
        prob = expected_score(1600.0, 1400.0)
        assert prob > 0.5

    def test_lower_rating_gives_lower_probability(self):
        prob = expected_score(1400.0, 1600.0)
        assert prob < 0.5

    def test_result_is_between_0_and_1(self):
        for elo_a, elo_b in [(500, 3000), (3000, 500), (1500, 1500)]:
            prob = expected_score(elo_a, elo_b)
            assert 0.0 < prob < 1.0

    def test_symmetric_probabilities_sum_to_one(self):
        p_a = expected_score(1600.0, 1400.0)
        p_b = expected_score(1400.0, 1600.0)
        assert p_a + p_b == pytest.approx(1.0)


class TestUpdateElos:
    def test_winner_gains_rating(self):
        elo_a, elo_b = update_elos(1500.0, 1500.0, score_a=80, score_b=70)
        assert elo_a > 1500.0

    def test_loser_loses_rating(self):
        elo_a, elo_b = update_elos(1500.0, 1500.0, score_a=80, score_b=70)
        assert elo_b < 1500.0

    def test_ratings_sum_preserved(self):
        """Total rating mass is conserved after a game."""
        elo_a_pre, elo_b_pre = 1500.0, 1500.0
        elo_a_post, elo_b_post = update_elos(elo_a_pre, elo_b_pre, 80, 70)
        assert elo_a_post + elo_b_post == pytest.approx(elo_a_pre + elo_b_pre)

    def test_upset_gives_large_update(self):
        """A big underdog winning should produce a larger rating jump."""
        # Team A is a big underdog; both score such that A wins
        elo_a_pre, elo_b_pre = 1300.0, 1700.0
        elo_a_post, _ = update_elos(elo_a_pre, elo_b_pre, 65, 60, k=20.0)
        gain_upset = elo_a_post - elo_a_pre

        # Expected gain if ratings were equal
        elo_a_eq_post, _ = update_elos(1500.0, 1500.0, 65, 60, k=20.0)
        gain_even = elo_a_eq_post - 1500.0

        assert gain_upset > gain_even

    def test_custom_k_factor_scales_update(self):
        elo_a_k20, _ = update_elos(1500.0, 1500.0, 80, 70, k=20.0)
        elo_a_k40, _ = update_elos(1500.0, 1500.0, 80, 70, k=40.0)
        assert (elo_a_k40 - 1500.0) == pytest.approx(2 * (elo_a_k20 - 1500.0))


class TestRevertToMean:
    def test_no_reversion_when_factor_zero(self):
        assert revert_to_mean(1700.0, mean=1500.0, factor=0.0) == pytest.approx(1700.0)

    def test_full_reversion_when_factor_one(self):
        assert revert_to_mean(1700.0, mean=1500.0, factor=1.0) == pytest.approx(1500.0)

    def test_partial_reversion(self):
        reverted = revert_to_mean(1700.0, mean=1500.0, factor=0.5)
        assert reverted == pytest.approx(1600.0)

    def test_below_mean_reverts_upward(self):
        reverted = revert_to_mean(1300.0, mean=1500.0, factor=0.5)
        assert reverted == pytest.approx(1400.0)

    def test_at_mean_no_change(self):
        assert revert_to_mean(DEFAULT_ELO, mean=DEFAULT_ELO) == pytest.approx(DEFAULT_ELO)
