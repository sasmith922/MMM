"""
test_features.py
----------------
Unit tests for madness_model.build_team_features and madness_model.clean_data.
"""

import pandas as pd
import pytest

from madness_model.build_team_features import (
    build_team_features,
    compute_avg_point_diff,
    compute_win_pct,
)
from madness_model.clean_data import (
    clean_game_results,
    clean_seeds,
    clean_teams,
    filter_seasons,
    parse_seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_games() -> pd.DataFrame:
    """Minimal regular-season game records."""
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022, 2022],
            "day_num": [10, 15, 20, 25],
            "w_team_id": [1, 1, 2, 3],
            "l_team_id": [2, 3, 3, 2],
            "w_score": [80, 75, 70, 65],
            "l_score": [70, 65, 60, 55],
        }
    )


@pytest.fixture()
def sample_seeds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "seed": ["W01", "X16", "Y08a"],
            "team_id": [1, 2, 3],
        }
    )


@pytest.fixture()
def sample_teams() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3, 1],  # duplicate
            "team_name": ["Alpha", "Beta", "Gamma", "Alpha"],
        }
    )


# ---------------------------------------------------------------------------
# parse_seed
# ---------------------------------------------------------------------------

class TestParseSeed:
    @pytest.mark.parametrize(
        "seed_str, expected",
        [
            ("W01", 1),
            ("X16", 16),
            ("Y08a", 8),
            ("Z11b", 11),
        ],
    )
    def test_various_formats(self, seed_str, expected):
        assert parse_seed(seed_str) == expected


# ---------------------------------------------------------------------------
# clean_seeds
# ---------------------------------------------------------------------------

class TestCleanSeeds:
    def test_seed_column_is_integer(self, sample_seeds):
        df = clean_seeds(sample_seeds)
        assert df["seed"].dtype == int

    def test_region_column_extracted(self, sample_seeds):
        df = clean_seeds(sample_seeds)
        assert set(df["region"]) == {"W", "X", "Y"}

    def test_original_seed_preserved_as_seed_str(self, sample_seeds):
        df = clean_seeds(sample_seeds)
        assert "seed_str" in df.columns

    def test_play_in_seed_parsed_correctly(self, sample_seeds):
        df = clean_seeds(sample_seeds)
        play_in_row = df[df["seed_str"] == "Y08a"]
        assert play_in_row["seed"].iloc[0] == 8


# ---------------------------------------------------------------------------
# clean_teams
# ---------------------------------------------------------------------------

class TestCleanTeams:
    def test_duplicates_removed(self, sample_teams):
        df = clean_teams(sample_teams)
        assert df["team_id"].nunique() == df["team_id"].count()

    def test_team_id_is_integer(self, sample_teams):
        df = clean_teams(sample_teams)
        assert df["team_id"].dtype == int


# ---------------------------------------------------------------------------
# clean_game_results
# ---------------------------------------------------------------------------

class TestCleanGameResults:
    def test_returns_dataframe(self, sample_games):
        df = clean_game_results(sample_games)
        assert isinstance(df, pd.DataFrame)

    def test_integer_columns(self, sample_games):
        df = clean_game_results(sample_games)
        for col in ["season", "w_team_id", "l_team_id", "w_score", "l_score"]:
            assert df[col].dtype == int

    def test_drops_null_rows(self):
        games = pd.DataFrame(
            {
                "season": [2022, None],
                "w_team_id": [1, 2],
                "l_team_id": [2, 3],
                "w_score": [80, 75],
                "l_score": [70, 65],
            }
        )
        df = clean_game_results(games)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# filter_seasons
# ---------------------------------------------------------------------------

class TestFilterSeasons:
    def test_filters_correctly(self, sample_games):
        df = filter_seasons(sample_games, [2022])
        assert set(df["season"]) == {2022}

    def test_empty_when_no_match(self, sample_games):
        df = filter_seasons(sample_games, [1999])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# compute_win_pct
# ---------------------------------------------------------------------------

class TestComputeWinPct:
    def test_win_pct_between_0_and_1(self, sample_games):
        df = compute_win_pct(sample_games)
        assert (df["win_pct"] >= 0).all() and (df["win_pct"] <= 1).all()

    def test_team1_wins_most(self, sample_games):
        """Team 1 wins 2 out of 2 games → win_pct = 1.0."""
        df = compute_win_pct(sample_games)
        team1 = df[df["team_id"] == 1]
        assert team1["win_pct"].iloc[0] == pytest.approx(1.0)

    def test_team3_loses_most(self, sample_games):
        """Team 3 wins 1 out of 3 games."""
        df = compute_win_pct(sample_games)
        team3 = df[df["team_id"] == 3]
        assert team3["win_pct"].iloc[0] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# compute_avg_point_diff
# ---------------------------------------------------------------------------

class TestComputeAvgPointDiff:
    def test_returns_expected_columns(self, sample_games):
        df = compute_avg_point_diff(sample_games)
        assert "avg_point_diff" in df.columns
        assert "team_id" in df.columns

    def test_team1_positive_margin(self, sample_games):
        df = compute_avg_point_diff(sample_games)
        team1 = df[df["team_id"] == 1]
        assert team1["avg_point_diff"].iloc[0] > 0

    def test_team3_negative_margin(self, sample_games):
        """Team 3 loses all 3 games → negative average margin."""
        df = compute_avg_point_diff(sample_games)
        team3 = df[df["team_id"] == 3]
        # Team 3 is the loser in all games they appear in except one where they beat team 2
        # Recalculate: Team 3 loses to 1 (−10), loses to 1 (−10), loses to 2 (−10), beats 2 (+10)
        # avg = (−10 −10 −10 +10)/4 = −5
        assert team3["avg_point_diff"].iloc[0] < 0


# ---------------------------------------------------------------------------
# build_team_features
# ---------------------------------------------------------------------------

class TestBuildTeamFeatures:
    def test_indexed_by_season_team(self, sample_games):
        features = build_team_features(sample_games)
        assert features.index.names == ["season", "team_id"]

    def test_all_teams_present(self, sample_games):
        features = build_team_features(sample_games)
        team_ids = features.index.get_level_values("team_id")
        assert set(team_ids) == {1, 2, 3}

    def test_win_pct_column_present(self, sample_games):
        features = build_team_features(sample_games)
        assert "win_pct" in features.columns

    def test_avg_point_diff_column_present(self, sample_games):
        features = build_team_features(sample_games)
        assert "avg_point_diff" in features.columns
