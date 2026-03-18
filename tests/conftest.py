"""Shared pytest fixtures for bracket simulator tests."""

from __future__ import annotations

import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from madness_model.bracket import ModelBundle
from tests.fixtures.fake_models import (
    FakeModelAlwaysLeftTeam,
    FakeModelFixedProbs,
    FakeModelProbabilistic,
    FakeModelSeedBased,
)
from tests.fixtures.sample_brackets import (
    SEASON,
    make_fake_full_64_team_bracket,
    make_fake_seeds_df,
    make_fake_teams_df,
    make_toy_4_team_bracket,
    make_toy_8_team_bracket,
)

TOY_TEAM_SEED_MAP = {1: 1, 2: 4, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8}


@pytest.fixture
def season() -> int:
    """Stable synthetic season used by tests."""
    return SEASON


@pytest.fixture
def deterministic_rng_seed() -> int:
    """Shared fixed random seed for reproducible stochastic tests."""
    return 12345


@pytest.fixture
def toy_4_team_bracket():
    """Minimal 4-team bracket fixture."""
    return make_toy_4_team_bracket()


@pytest.fixture
def toy_8_team_bracket():
    """8-team toy bracket fixture."""
    return make_toy_8_team_bracket()


@pytest.fixture
def fake_full_64_team_bracket(season):
    """Synthetic full 64-team bracket state fixture."""
    return make_fake_full_64_team_bracket(season)


@pytest.fixture
def fake_team_slots(toy_4_team_bracket):
    """Convenience fixture exposing toy initial slot assignments."""
    return dict(toy_4_team_bracket.initial_slots)


@pytest.fixture
def features_df(fake_full_64_team_bracket, season) -> pd.DataFrame:
    """Synthetic feature table indexed by (season, team_id)."""
    all_team_ids = set(fake_full_64_team_bracket.initial_slots.values()) | {1, 2, 3, 4, 5, 6, 7, 8}
    rows = []
    for team_id in sorted(all_team_ids):
        seed = (team_id % 1000) if team_id >= 1000 else TOY_TEAM_SEED_MAP[team_id]
        rows.append(
            {
                "season": season,
                "team_id": team_id,
                "team_id_feature": float(team_id),
                "team_id_sq_feature": float(team_id * team_id),
                "seed": float(seed),
                "elo": float(2000 - team_id),
            }
        )
    return pd.DataFrame(rows).set_index(["season", "team_id"])


@pytest.fixture
def feature_cols() -> list[str]:
    """Feature columns consumed by fake models."""
    return ["diff_team_id_feature", "diff_team_id_sq_feature", "diff_seed", "diff_elo"]


@pytest.fixture
def fake_model_fixed_probs(feature_cols):
    """Model fixture with explicit ordered-matchup probabilities."""
    probs = {
        (1, 2): 0.90,
        (3, 4): 0.80,
        (1, 3): 0.70,
        (1, 4): 0.95,
        (2, 3): 0.40,
        (2, 4): 0.60,
    }
    return FakeModelFixedProbs(
        matchup_probs=probs,
        team_id_diff_col=feature_cols.index("diff_team_id_feature"),
        team_sq_diff_col=feature_cols.index("diff_team_id_sq_feature"),
    )


@pytest.fixture
def fake_model_seed_based(feature_cols):
    """Model fixture driven only by seed differential."""
    return FakeModelSeedBased(diff_seed_col=feature_cols.index("diff_seed"))


@pytest.fixture
def fake_model_always_left_team():
    """Model fixture that always favors Team A."""
    return FakeModelAlwaysLeftTeam(prob_a=0.75)


@pytest.fixture
def fake_model_probabilistic():
    """Model fixture with fixed stochastic probability."""
    return FakeModelProbabilistic(prob_a=0.8)


@pytest.fixture
def toy_model_bundle(fake_model_fixed_probs, features_df, feature_cols):
    """Model bundle for exact toy-bracket expectations."""
    return ModelBundle(
        model=fake_model_fixed_probs, features=features_df, feature_cols=feature_cols
    )


@pytest.fixture
def seed_model_bundle(fake_model_seed_based, features_df, feature_cols):
    """Model bundle using seed-based probabilities."""
    return ModelBundle(
        model=fake_model_seed_based, features=features_df, feature_cols=feature_cols
    )


@pytest.fixture
def left_model_bundle(fake_model_always_left_team, features_df, feature_cols):
    """Model bundle always favoring Team A."""
    return ModelBundle(
        model=fake_model_always_left_team, features=features_df, feature_cols=feature_cols
    )


@pytest.fixture
def probabilistic_model_bundle(fake_model_probabilistic, features_df, feature_cols):
    """Model bundle with fixed P(team_a_wins)=0.8."""
    return ModelBundle(
        model=fake_model_probabilistic, features=features_df, feature_cols=feature_cols
    )


@pytest.fixture
def fake_teams_df():
    """Synthetic teams dataframe fixture."""
    return make_fake_teams_df()


@pytest.fixture
def fake_seeds_df(season):
    """Synthetic seeds dataframe fixture."""
    return make_fake_seeds_df(season)
