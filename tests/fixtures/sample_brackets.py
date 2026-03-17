"""Handcrafted bracket fixtures used by simulator tests."""

from __future__ import annotations

import pandas as pd

from madness_model.bracket import BracketGame, BracketState, REGIONS
from madness_model.simulate_bracket import build_initial_bracket, load_bracket_structure

SEASON = 2025


def make_toy_4_team_bracket() -> BracketState:
    """4-team bracket: two semifinals and one championship."""
    games = [
        BracketGame(
            game_id="R64_TOY_1",
            round_name="R64",
            region="Toy",
            slot_label="Toy semifinal 1",
            left_source="Toy_S1",
            right_source="Toy_S4",
            next_game_id="CHAMP",
            next_slot="left",
        ),
        BracketGame(
            game_id="R64_TOY_2",
            round_name="R64",
            region="Toy",
            slot_label="Toy semifinal 2",
            left_source="Toy_S2",
            right_source="Toy_S3",
            next_game_id="CHAMP",
            next_slot="right",
        ),
        BracketGame(
            game_id="CHAMP",
            round_name="CHAMP",
            region=None,
            slot_label="Toy championship",
            left_source="WINNER_R64_TOY_1",
            right_source="WINNER_R64_TOY_2",
            next_game_id=None,
            next_slot=None,
        ),
    ]
    return BracketState(
        games=games,
        initial_slots={"Toy_S1": 1, "Toy_S4": 2, "Toy_S2": 3, "Toy_S3": 4},
    )


def make_toy_8_team_bracket() -> BracketState:
    """8-team bracket with quarterfinals, semifinals, and championship."""
    games = [
        BracketGame("R64_TOY_1", "R64", "Toy", "QF1", "Toy_S1", "Toy_S8", "R32_TOY_1", "left"),
        BracketGame("R64_TOY_2", "R64", "Toy", "QF2", "Toy_S4", "Toy_S5", "R32_TOY_1", "right"),
        BracketGame("R64_TOY_3", "R64", "Toy", "QF3", "Toy_S2", "Toy_S7", "R32_TOY_2", "left"),
        BracketGame("R64_TOY_4", "R64", "Toy", "QF4", "Toy_S3", "Toy_S6", "R32_TOY_2", "right"),
        BracketGame(
            "R32_TOY_1",
            "R32",
            "Toy",
            "SF1",
            "WINNER_R64_TOY_1",
            "WINNER_R64_TOY_2",
            "CHAMP",
            "left",
        ),
        BracketGame(
            "R32_TOY_2",
            "R32",
            "Toy",
            "SF2",
            "WINNER_R64_TOY_3",
            "WINNER_R64_TOY_4",
            "CHAMP",
            "right",
        ),
        BracketGame(
            "CHAMP",
            "CHAMP",
            None,
            "Toy championship",
            "WINNER_R32_TOY_1",
            "WINNER_R32_TOY_2",
            None,
            None,
        ),
    ]
    initial_slots = {
        "Toy_S1": 1,
        "Toy_S8": 8,
        "Toy_S4": 4,
        "Toy_S5": 5,
        "Toy_S2": 2,
        "Toy_S7": 7,
        "Toy_S3": 3,
        "Toy_S6": 6,
    }
    return BracketState(games=games, initial_slots=initial_slots)


def make_fake_teams_df() -> pd.DataFrame:
    """Synthetic 64-team table with IDs stable by region/seed."""
    rows = []
    for region_idx, region in enumerate(REGIONS):
        for seed in range(1, 17):
            team_id = (region_idx + 1) * 1000 + seed
            rows.append({"team_id": team_id, "team_name": f"{region}_S{seed}"})
    return pd.DataFrame(rows)


def make_fake_seeds_df(season: int = SEASON) -> pd.DataFrame:
    """Synthetic seed assignments for a complete 64-team field."""
    rows = []
    for region_idx, region in enumerate(REGIONS):
        for seed in range(1, 17):
            team_id = (region_idx + 1) * 1000 + seed
            rows.append(
                {"season": season, "team_id": team_id, "seed": seed, "region": region}
            )
    return pd.DataFrame(rows)


def make_fake_full_64_team_bracket(season: int = SEASON) -> BracketState:
    """Graph-based full tournament state from synthetic seeds."""
    teams_df = make_fake_teams_df()
    seeds_df = make_fake_seeds_df(season)
    return build_initial_bracket(season, teams_df, seeds_df)


def get_full_structure_games(season: int = SEASON) -> list[BracketGame]:
    """Convenience function for full-bracket structure-only tests."""
    return load_bracket_structure(season)
