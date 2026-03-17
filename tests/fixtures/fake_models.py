"""Simple fake model classes for deterministic simulator testing."""

from __future__ import annotations

import numpy as np


class FakeModelFixedProbs:
    """Returns lookup-based probabilities for exact ordered team matchups."""

    def __init__(
        self,
        matchup_probs: dict[tuple[int, int], float],
        default_prob: float = 0.5,
        team_id_diff_col: int = 0,
        team_sq_diff_col: int = 1,
    ) -> None:
        self.matchup_probs = matchup_probs
        self.default_prob = default_prob
        self.team_id_diff_col = team_id_diff_col
        self.team_sq_diff_col = team_sq_diff_col

    def _recover_ordered_pair(self, diff_id: float, diff_sq: float) -> tuple[int, int] | None:
        if diff_id == 0:
            return None
        sum_id = diff_sq / diff_id
        team_a = int(round((diff_id + sum_id) / 2))
        team_b = int(round((sum_id - diff_id) / 2))
        return team_a, team_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for row in X:
            pair = self._recover_ordered_pair(
                float(row[self.team_id_diff_col]), float(row[self.team_sq_diff_col])
            )
            if pair is None:
                prob_a = self.default_prob
            elif pair in self.matchup_probs:
                prob_a = self.matchup_probs[pair]
            elif (pair[1], pair[0]) in self.matchup_probs:
                prob_a = 1.0 - self.matchup_probs[(pair[1], pair[0])]
            else:
                prob_a = self.default_prob
            probs.append(prob_a)
        probs_arr = np.array(probs, dtype=float)
        return np.column_stack([1.0 - probs_arr, probs_arr])


class FakeModelSeedBased:
    """Higher probability for Team A when Team A has a better (lower) seed."""

    def __init__(self, diff_seed_col: int = 2) -> None:
        self.diff_seed_col = diff_seed_col

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # diff_seed = seed_a - seed_b; negative means team_a has better seed.
        diff_seed = X[:, self.diff_seed_col]
        prob_a = 1.0 / (1.0 + np.exp(diff_seed))
        return np.column_stack([1.0 - prob_a, prob_a])


class FakeModelAlwaysLeftTeam:
    """Predicts Team A with fixed confidence."""

    def __init__(self, prob_a: float = 0.75) -> None:
        self.prob_a = prob_a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1.0 - self.prob_a), np.full(n, self.prob_a)])


class FakeModelProbabilistic:
    """Fixed-probability Bernoulli model used for repeated stochastic tests."""

    def __init__(self, prob_a: float) -> None:
        self.prob_a = prob_a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1.0 - self.prob_a), np.full(n, self.prob_a)])


class FakeModelNaN:
    """Returns NaN to test probability validation."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        probs = np.full(n, np.nan)
        return np.column_stack([1.0 - probs, probs])


class FakeModelOutOfRange:
    """Returns a fixed out-of-range probability."""

    def __init__(self, prob_a: float) -> None:
        self.prob_a = prob_a

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return np.column_stack([np.full(n, 1.0 - self.prob_a), np.full(n, self.prob_a)])
