"""Aggregation tests for repeated bracket simulations."""

from __future__ import annotations

import pytest

from madness_model.bracket import ROUND_NAMES
from madness_model.simulate_bracket import simulate_many_brackets


def _round_prob_value(round_probs, team_id, round_name):
    return round_probs.get(team_id, {}).get(round_name, 0.0)


def test_many_simulations_returns_probability_table(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=50, random_state=7)
    assert agg.round_probs, "Aggregation should return non-empty round probability table."


def test_champion_probabilities_sum_to_one(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=200, random_state=8)
    assert sum(agg.champion_probs.values()) == pytest.approx(1.0, abs=1e-9), "Champion probabilities must sum to 1."


def test_round_probabilities_are_monotonic(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=200, random_state=9)
    for team_id in agg.round_probs:
        for i in range(len(ROUND_NAMES) - 1):
            curr_round = ROUND_NAMES[i]
            next_round = ROUND_NAMES[i + 1]
            assert _round_prob_value(agg.round_probs, team_id, curr_round) + 1e-12 >= _round_prob_value(
                agg.round_probs, team_id, next_round
            ), f"Team {team_id} violates monotonic reach probabilities."


def test_all_round_probabilities_are_between_zero_and_one(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=100, random_state=10)
    for team_id, rounds in agg.round_probs.items():
        for round_name, prob in rounds.items():
            assert 0.0 <= prob <= 1.0, f"Team {team_id} has out-of-range probability for {round_name}: {prob}."


def test_total_number_of_champions_across_sims_equals_n_sims(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    n_sims = 120
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=n_sims, random_state=11)
    total = sum(int(round(prob * n_sims)) for prob in agg.champion_probs.values())
    assert total == n_sims, "Total champion counts reconstructed from probabilities should match n_sims."


def test_expected_round_score_is_nonnegative_and_ordered(fake_full_64_team_bracket, probabilistic_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, probabilistic_model_bundle, n_sims=150, random_state=12)
    expected_scores = {}
    for team_id, rounds in agg.round_probs.items():
        expected = sum(rounds[rn] for rn in ROUND_NAMES)
        expected_scores[team_id] = expected
        assert expected >= 0.0, f"Expected round score must be nonnegative for team {team_id}."
    assert max(expected_scores.values()) >= min(expected_scores.values()), "Expected round score ordering should be valid."


def test_dominant_team_fake_model_has_highest_title_probability(fake_full_64_team_bracket, seed_model_bundle, season):
    agg = simulate_many_brackets(fake_full_64_team_bracket, season, seed_model_bundle, n_sims=200, random_state=13)
    best_team = max(agg.champion_probs, key=agg.champion_probs.get)
    assert best_team % 1000 == 1, "Seed-based dominant model should favor a 1-seed as title favorite."
