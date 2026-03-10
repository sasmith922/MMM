"""
test_build_matchups.py
----------------------
Unit tests for madness_model.build_matchups.
"""

import pandas as pd
import pytest

from madness_model.build_matchups import (
    build_matchup_row,
    build_matchups_from_results,
    build_prediction_matchups,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_features() -> pd.DataFrame:
    """A minimal (season, team_id)-indexed feature DataFrame."""
    data = {
        "season": [2022, 2022, 2022],
        "team_id": [1, 2, 3],
        "win_pct": [0.8, 0.5, 0.3],
        "avg_point_diff": [10.0, 0.0, -5.0],
    }
    df = pd.DataFrame(data).set_index(["season", "team_id"])
    return df


@pytest.fixture()
def simple_tourney_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022],
            "w_team_id": [1, 2],
            "l_team_id": [2, 3],
        }
    )


# ---------------------------------------------------------------------------
# build_matchup_row
# ---------------------------------------------------------------------------

class TestBuildMatchupRow:
    def test_returns_dict_with_expected_keys(self, simple_features):
        row = build_matchup_row(2022, 1, 2, simple_features, label=1)
        assert "season" in row
        assert "team_a_id" in row
        assert "team_b_id" in row
        assert "label" in row

    def test_feature_differentials_are_correct(self, simple_features):
        row = build_matchup_row(2022, 1, 2, simple_features)
        # Team 1 win_pct=0.8, Team 2 win_pct=0.5 → diff = 0.3
        assert row["diff_win_pct"] == pytest.approx(0.3)
        # Team 1 avg_point_diff=10, Team 2=0 → diff = 10
        assert row["diff_avg_point_diff"] == pytest.approx(10.0)

    def test_antisymmetric_differentials(self, simple_features):
        row_ab = build_matchup_row(2022, 1, 2, simple_features)
        row_ba = build_matchup_row(2022, 2, 1, simple_features)
        assert row_ab["diff_win_pct"] == pytest.approx(-row_ba["diff_win_pct"])

    def test_label_none_omitted(self, simple_features):
        row = build_matchup_row(2022, 1, 2, simple_features, label=None)
        assert "label" not in row

    def test_label_included_when_provided(self, simple_features):
        row = build_matchup_row(2022, 1, 2, simple_features, label=1)
        assert row["label"] == 1

    def test_raises_for_missing_team(self, simple_features):
        with pytest.raises(KeyError):
            build_matchup_row(2022, 1, 99, simple_features)

    def test_raises_for_missing_season(self, simple_features):
        with pytest.raises(KeyError):
            build_matchup_row(2099, 1, 2, simple_features)


# ---------------------------------------------------------------------------
# build_matchups_from_results
# ---------------------------------------------------------------------------

class TestBuildMatchupsFromResults:
    def test_returns_dataframe(self, simple_tourney_results, simple_features):
        df = build_matchups_from_results(simple_tourney_results, simple_features)
        assert isinstance(df, pd.DataFrame)

    def test_two_rows_per_game(self, simple_tourney_results, simple_features):
        # 2 games → 4 rows (winner-as-A and loser-as-A for each game, but
        # game 2 (2 vs 3) might get fewer if a team is missing)
        df = build_matchups_from_results(simple_tourney_results, simple_features)
        # Both games have all teams in features → expect 4 rows
        assert len(df) == 4

    def test_labels_are_binary(self, simple_tourney_results, simple_features):
        df = build_matchups_from_results(simple_tourney_results, simple_features)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_skips_missing_teams(self, simple_features):
        """Games where a team has no features should be silently skipped."""
        results = pd.DataFrame(
            {"season": [2022], "w_team_id": [1], "l_team_id": [99]}
        )
        df = build_matchups_from_results(results, simple_features)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# build_prediction_matchups
# ---------------------------------------------------------------------------

class TestBuildPredictionMatchups:
    def test_no_label_column(self, simple_features):
        pairs = [(1, 2), (1, 3)]
        df = build_prediction_matchups(2022, pairs, simple_features)
        assert "label" not in df.columns

    def test_correct_number_of_rows(self, simple_features):
        pairs = [(1, 2), (1, 3), (2, 3)]
        df = build_prediction_matchups(2022, pairs, simple_features)
        assert len(df) == 3

    def test_skips_pairs_with_missing_features(self, simple_features):
        pairs = [(1, 99), (2, 3)]
        df = build_prediction_matchups(2022, pairs, simple_features)
        assert len(df) == 1
