"""Bracket graph structure and validity tests."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from madness_model.bracket import ROUND_NAMES, REGIONS
from madness_model.simulate_bracket import load_bracket_structure


def _topological_order(games):
    game_map = {g.game_id: g for g in games}
    indegree = {g.game_id: 0 for g in games}
    outgoing = defaultdict(list)

    for g in games:
        if g.next_game_id:
            outgoing[g.game_id].append(g.next_game_id)
            indegree[g.next_game_id] += 1

    queue = deque([gid for gid, deg in indegree.items() if deg == 0])
    ordered = []
    while queue:
        gid = queue.popleft()
        ordered.append(gid)
        for nxt in outgoing[gid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return ordered


def test_full_bracket_has_expected_number_of_games(season):
    games = load_bracket_structure(season)
    assert len(games) == 63, "A 64-team single-elimination bracket must have 63 games."


def test_each_game_has_valid_round_name(season):
    games = load_bracket_structure(season)
    for game in games:
        assert game.round_name in ROUND_NAMES, f"{game.game_id} has invalid round {game.round_name!r}."


def test_non_final_games_advance_somewhere(season):
    games = load_bracket_structure(season)
    for game in games:
        if game.game_id != "CHAMP":
            assert game.next_game_id, f"{game.game_id} must advance to a next game."


def test_final_game_has_no_next_game(season):
    games = load_bracket_structure(season)
    final = next(g for g in games if g.game_id == "CHAMP")
    assert final.next_game_id is None, "Championship must not advance to any game."


def test_initial_round_sources_are_seed_slots(season):
    games = load_bracket_structure(season)
    r64_games = [g for g in games if g.round_name == "R64"]
    for game in r64_games:
        assert not game.left_source.startswith("WINNER_"), f"{game.game_id} left source should be a seed slot."
        assert not game.right_source.startswith("WINNER_"), f"{game.game_id} right source should be a seed slot."


def test_later_round_sources_are_winner_references(season):
    games = load_bracket_structure(season)
    for game in games:
        if game.round_name != "R64":
            assert game.left_source.startswith("WINNER_"), f"{game.game_id} left source should reference prior winner."
            assert game.right_source.startswith("WINNER_"), f"{game.game_id} right source should reference prior winner."


def test_games_form_acyclic_graph(season):
    games = load_bracket_structure(season)
    ordered = _topological_order(games)
    assert len(ordered) == len(games), "Bracket graph contains a cycle."


def test_topological_sort_contains_all_games_once(season):
    games = load_bracket_structure(season)
    ordered = _topological_order(games)
    assert len(ordered) == len(set(ordered)) == len(games), "Topological order must include each game once."


def test_each_game_accepts_exactly_two_inputs(season):
    games = load_bracket_structure(season)
    for game in games:
        assert game.left_source, f"{game.game_id} missing left input source."
        assert game.right_source, f"{game.game_id} missing right input source."


def test_each_region_has_expected_round_counts(season):
    games = load_bracket_structure(season)
    counts_by_region = {region: Counter() for region in REGIONS}
    overall = Counter(g.round_name for g in games)

    for game in games:
        if game.region in counts_by_region:
            counts_by_region[game.region][game.round_name] += 1

    for region, counts in counts_by_region.items():
        assert counts["R64"] == 8, f"{region} should have 8 R64 games."
        assert counts["R32"] == 4, f"{region} should have 4 R32 games."
        assert counts["S16"] == 2, f"{region} should have 2 S16 games."
        assert counts["E8"] == 1, f"{region} should have 1 E8 game."

    assert overall["F4"] == 2, "Bracket should have 2 Final Four games."
    assert overall["CHAMP"] == 1, "Bracket should have 1 championship game."
