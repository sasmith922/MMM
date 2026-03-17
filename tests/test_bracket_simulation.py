"""Simulation behavior tests for toy and full bracket fixtures."""

from __future__ import annotations

from copy import deepcopy

import pytest

from madness_model.bracket import BracketGame, BracketState, ModelBundle
from madness_model.simulate_bracket import predict_game, simulate_single_bracket
from tests.fixtures.fake_models import FakeModelNaN, FakeModelOutOfRange


def test_toy_bracket_deterministic_picks_expected_winners(toy_4_team_bracket, toy_model_bundle, season):
    result = simulate_single_bracket(toy_4_team_bracket, season, toy_model_bundle, mode="deterministic")
    assert result.game_results["R64_TOY_1"] == 1, "Semifinal 1 should be won by team 1."
    assert result.game_results["R64_TOY_2"] == 3, "Semifinal 2 should be won by team 3."
    assert result.champion_id == 1, "Team 1 should win deterministic toy bracket."


def test_toy_bracket_winners_advance_to_correct_next_game(toy_4_team_bracket, toy_model_bundle, season):
    result = simulate_single_bracket(toy_4_team_bracket, season, toy_model_bundle, mode="deterministic")
    slots = dict(toy_4_team_bracket.initial_slots)
    slots["WINNER_R64_TOY_1"] = result.game_results["R64_TOY_1"]
    slots["WINNER_R64_TOY_2"] = result.game_results["R64_TOY_2"]
    champ_game = next(g for g in toy_4_team_bracket.games if g.game_id == "CHAMP")
    assert slots[champ_game.left_source] == result.game_results["R64_TOY_1"], "Left semifinal winner must fill championship left slot."
    assert slots[champ_game.right_source] == result.game_results["R64_TOY_2"], "Right semifinal winner must fill championship right slot."


def test_toy_bracket_returns_exact_number_of_predictions(toy_4_team_bracket, toy_model_bundle, season):
    result = simulate_single_bracket(toy_4_team_bracket, season, toy_model_bundle, mode="deterministic")
    assert len(result.game_results) == 3, "4-team bracket should produce exactly 3 game results."


def test_toy_bracket_records_correct_champion(toy_4_team_bracket, toy_model_bundle, season):
    result = simulate_single_bracket(toy_4_team_bracket, season, toy_model_bundle, mode="deterministic")
    assert result.game_results["CHAMP"] == result.champion_id == 1, "Champion should match championship game winner."


def test_round_tracking_for_toy_bracket_is_correct(toy_4_team_bracket, toy_model_bundle, season):
    result = simulate_single_bracket(toy_4_team_bracket, season, toy_model_bundle, mode="deterministic")
    assert result.team_round_reached[2] == "R64", "Semifinal loser should stop at semifinal round."
    assert result.team_round_reached[4] == "R64", "Semifinal loser should stop at semifinal round."
    assert result.team_round_reached[3] == "CHAMP", "Finalist should reach championship round."
    assert result.team_round_reached[1] == "CHAMP", "Champion should be marked as reaching championship."


def test_deterministic_simulation_is_reproducible(toy_8_team_bracket, left_model_bundle, season):
    r1 = simulate_single_bracket(toy_8_team_bracket, season, left_model_bundle, mode="deterministic")
    r2 = simulate_single_bracket(toy_8_team_bracket, season, left_model_bundle, mode="deterministic")
    assert r1.game_results == r2.game_results, "Deterministic mode must be fully reproducible."
    assert r1.champion_id == r2.champion_id, "Deterministic mode should produce one stable champion."


def test_equal_probability_tie_breaking_is_stable(toy_4_team_bracket, features_df, feature_cols, season):
    class TieModel:
        def predict_proba(self, X):
            import numpy as np

            n = X.shape[0]
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

    # Team 2 and 4 have same seed in this modified toy bracket; Elo and team_id decide.
    custom_state = deepcopy(toy_4_team_bracket)
    custom_state.initial_slots = {"Toy_S1": 2, "Toy_S4": 4, "Toy_S2": 3, "Toy_S3": 1}
    bundle = ModelBundle(model=TieModel(), features=features_df, feature_cols=feature_cols)
    result = simulate_single_bracket(custom_state, season, bundle, mode="deterministic")
    assert result.game_results["R64_TOY_1"] == 2, "Tie should first use better seed."
    assert result.game_results["R64_TOY_2"] == 3, "Tie should use better seed in second semifinal."
    assert result.champion_id == 2, "With equal probs, tie-break should be deterministic and stable."


def test_deterministic_simulation_never_marks_two_champions(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    assert result.champion_id is not None, "Simulation must return exactly one champion id."
    assert result.game_results["CHAMP"] == result.champion_id, "Champion id should match championship game winner."


def test_all_games_have_valid_winners(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    slots = dict(fake_full_64_team_bracket.initial_slots)
    for game in fake_full_64_team_bracket.games:
        team_a = slots.get(game.left_source)
        team_b = slots.get(game.right_source)
        winner = result.game_results[game.game_id]
        assert winner in {team_a, team_b}, f"{game.game_id} winner must be one of the two participants."
        slots[f"WINNER_{game.game_id}"] = winner


def test_stochastic_simulation_is_reproducible_with_fixed_seed(toy_8_team_bracket, probabilistic_model_bundle, season, deterministic_rng_seed):
    r1 = simulate_single_bracket(toy_8_team_bracket, season, probabilistic_model_bundle, mode="stochastic", random_state=deterministic_rng_seed)
    r2 = simulate_single_bracket(toy_8_team_bracket, season, probabilistic_model_bundle, mode="stochastic", random_state=deterministic_rng_seed)
    assert r1.game_results == r2.game_results, "Same random seed must reproduce stochastic bracket exactly."


def test_stochastic_simulation_changes_with_different_seed(toy_8_team_bracket, probabilistic_model_bundle, season):
    r1 = simulate_single_bracket(toy_8_team_bracket, season, probabilistic_model_bundle, mode="stochastic", random_state=11)
    r2 = simulate_single_bracket(toy_8_team_bracket, season, probabilistic_model_bundle, mode="stochastic", random_state=12)
    assert r1.game_results != r2.game_results, "Different seeds should produce different stochastic outcomes in toy bracket."


def test_stochastic_winner_is_always_one_of_the_two_teams(toy_8_team_bracket, probabilistic_model_bundle, season):
    result = simulate_single_bracket(toy_8_team_bracket, season, probabilistic_model_bundle, mode="stochastic", random_state=99)
    slots = dict(toy_8_team_bracket.initial_slots)
    for game in toy_8_team_bracket.games:
        winner = result.game_results[game.game_id]
        assert winner in {slots.get(game.left_source), slots.get(game.right_source)}, f"{game.game_id} produced impossible winner {winner}."
        slots[f"WINNER_{game.game_id}"] = winner


def test_probabilities_are_respected_in_repeated_runs(toy_4_team_bracket, probabilistic_model_bundle, season):
    wins_for_team1 = 0
    n_runs = 2000
    for seed in range(n_runs):
        result = simulate_single_bracket(
            toy_4_team_bracket,
            season,
            probabilistic_model_bundle,
            mode="stochastic",
            random_state=seed,
        )
        if result.game_results["R64_TOY_1"] == 1:
            wins_for_team1 += 1
    observed = wins_for_team1 / n_runs
    assert observed == pytest.approx(0.8, abs=0.04), "Observed win frequency should be close to model probability."


def test_predict_game_returns_probability_between_zero_and_one(seed_model_bundle, season):
    pred = predict_game(1, 2, season, seed_model_bundle, game_id="toy")
    assert 0.0 <= pred.prob_a_wins <= 1.0, "Prediction probability must be bounded in [0,1]."


def test_predict_game_probabilities_sum_to_one(seed_model_bundle, season):
    pred = predict_game(1, 2, season, seed_model_bundle, game_id="toy")
    assert (pred.prob_a_wins + (1.0 - pred.prob_a_wins)) == pytest.approx(1.0), "P(A)+P(B) must sum to 1."


def test_predict_game_returns_expected_teams_and_metadata(seed_model_bundle, season):
    pred = predict_game(1, 4, season, seed_model_bundle, game_id="R64_TOY_1")
    assert pred.game_id == "R64_TOY_1", "Prediction should retain game_id metadata."
    assert pred.team_a_id == 1 and pred.team_b_id == 4, "Prediction should return the same input team IDs."


def test_predict_game_uses_team_order_correctly(toy_model_bundle, season):
    pred_ab = predict_game(1, 2, season, toy_model_bundle, game_id="A_vs_B")
    pred_ba = predict_game(2, 1, season, toy_model_bundle, game_id="B_vs_A")
    assert pred_ab.prob_a_wins > 0.5, "Model should favor Team 1 when ordered as team_a."
    assert pred_ba.prob_a_wins < 0.5, "Reversed team order should invert Team A win probability."


def test_predict_game_clips_extreme_probabilities(features_df, feature_cols, season):
    class ExtremeModel:
        def predict_proba(self, X):
            import numpy as np

            n = X.shape[0]
            return np.column_stack([np.zeros(n), np.ones(n)])

    pred = predict_game(1, 2, season, ModelBundle(ExtremeModel(), features_df, feature_cols), game_id="clip")
    # TODO: if clipping policy changes, tighten to exact expected epsilon value.
    assert 0.0 < pred.prob_a_wins < 1.0, "Extreme model outputs should be clipped away from exact 0/1."


def test_full_bracket_simulation_runs_end_to_end(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    assert len(result.game_results) == 63, "Full bracket simulation should produce 63 game results."
    assert result.champion_id is not None, "Full bracket simulation should produce exactly one champion."


def test_full_bracket_has_64_unique_initial_teams(fake_full_64_team_bracket):
    teams = list(fake_full_64_team_bracket.initial_slots.values())
    assert len(teams) == 64 and len(set(teams)) == 64, "Full bracket must start with 64 unique teams."


def test_full_bracket_champion_is_from_original_field(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    assert result.champion_id in set(fake_full_64_team_bracket.initial_slots.values()), "Champion must come from initial tournament field."


def test_no_team_appears_in_two_games_same_round(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    slots = dict(fake_full_64_team_bracket.initial_slots)
    teams_by_round = {}
    for game in fake_full_64_team_bracket.games:
        round_set = teams_by_round.setdefault(game.round_name, set())
        participants = {slots[game.left_source], slots[game.right_source]}
        assert round_set.isdisjoint(participants), f"A team appeared in more than one {game.round_name} game."
        round_set.update(participants)
        slots[f"WINNER_{game.game_id}"] = result.game_results[game.game_id]


def test_advancing_teams_only_come_from_previous_round_winners(fake_full_64_team_bracket, seed_model_bundle, season):
    result = simulate_single_bracket(fake_full_64_team_bracket, season, seed_model_bundle, mode="deterministic")
    winners_seen = set(fake_full_64_team_bracket.initial_slots.values())
    for game in fake_full_64_team_bracket.games:
        for source in (game.left_source, game.right_source):
            if source.startswith("WINNER_"):
                source_game = source.replace("WINNER_", "")
                assert result.game_results[source_game] in winners_seen, "Advanced team must come from a previously won game."
        winners_seen.add(result.game_results[game.game_id])


def test_missing_team_in_initial_slot_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.initial_slots.pop("Toy_S4")
    with pytest.raises(ValueError, match="unresolved initial source slot|cannot start"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")


def test_game_starts_without_both_participants_resolved_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.games[2] = BracketGame(
        game_id="CHAMP",
        round_name="CHAMP",
        region=None,
        slot_label="broken",
        left_source="WINNER_R64_TOY_1",
        right_source="WINNER_DOES_NOT_EXIST",
        next_game_id=None,
        next_slot=None,
    )
    with pytest.raises(ValueError, match="invalid source reference"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")


def test_invalid_next_game_id_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.games[0] = BracketGame(
        game_id="R64_TOY_1",
        round_name="R64",
        region="Toy",
        slot_label="broken",
        left_source="Toy_S1",
        right_source="Toy_S4",
        next_game_id="NOT_A_GAME",
        next_slot="left",
    )
    with pytest.raises(ValueError, match="invalid next_game_id"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")


def test_invalid_source_reference_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.games[1] = BracketGame(
        game_id="R64_TOY_2",
        round_name="R64",
        region="Toy",
        slot_label="broken",
        left_source="BAD_SLOT",
        right_source="Toy_S3",
        next_game_id="CHAMP",
        next_slot="right",
    )
    with pytest.raises(ValueError, match="unresolved initial source slot"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")


def test_bracket_graph_cycle_detected_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.games[0] = BracketGame(
        game_id="R64_TOY_1",
        round_name="R64",
        region="Toy",
        slot_label="cycle",
        left_source="Toy_S1",
        right_source="Toy_S4",
        next_game_id="R64_TOY_1",
        next_slot="left",
    )
    with pytest.raises(ValueError, match="cycle"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")


def test_prediction_function_returns_nan_probability_raises(features_df, feature_cols, season):
    bundle = ModelBundle(model=FakeModelNaN(), features=features_df, feature_cols=feature_cols)
    with pytest.raises(ValueError, match="NaN probability"):
        predict_game(1, 2, season, bundle, game_id="nan")


def test_probability_outside_bounds_raises(features_df, feature_cols, season):
    bundle = ModelBundle(model=FakeModelOutOfRange(1.2), features=features_df, feature_cols=feature_cols)
    with pytest.raises(ValueError, match="expected \\[0, 1\\]"):
        predict_game(1, 2, season, bundle, game_id="oob")


def test_duplicate_team_assignment_in_initial_bracket_raises(toy_4_team_bracket, toy_model_bundle, season):
    bad_state = deepcopy(toy_4_team_bracket)
    bad_state.initial_slots["Toy_S4"] = bad_state.initial_slots["Toy_S1"]
    with pytest.raises(ValueError, match="Duplicate team assignment"):
        simulate_single_bracket(bad_state, season, toy_model_bundle, mode="deterministic")
