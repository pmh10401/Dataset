from __future__ import annotations

import unittest

import numpy as np

from experiments.exp403_408.classical_committee import (
    grade_frozen_hard,
    score_classical_committee,
)
from experiments.exp403_408.contracts import (
    CommitteeScores,
    SEED,
    TrainEligibility,
    train_eligibility,
)


class SpyLOF:
    """Reject accidental novelty scoring of a row used in the current fit."""

    def __init__(self) -> None:
        self.fit_rows: set[tuple[float, ...]] = set()
        self.scored_fit_rows = False
        self.fit_calls = 0
        self.score_calls = 0

    def fit(self, values: np.ndarray) -> SpyLOF:
        self.fit_calls += 1
        self.fit_rows = {tuple(row) for row in np.asarray(values, dtype=float)}
        return self

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        self.score_calls += 1
        rows = {tuple(row) for row in np.asarray(values, dtype=float)}
        self.scored_fit_rows |= bool(rows & self.fit_rows)
        return -np.arange(len(values), dtype=float)


def train5() -> np.ndarray:
    return np.column_stack((np.arange(5, dtype=float), np.arange(5, dtype=float) ** 2))


def train20() -> np.ndarray:
    return np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) ** 2))


def train_rows(n_rows: int) -> np.ndarray:
    values = np.arange(n_rows, dtype=float)
    return np.column_stack((values, values ** 2))


def test2() -> np.ndarray:
    return np.asarray([[100.0, 10000.0], [101.0, 10201.0]])


class ClassicalCommitteeTests(unittest.TestCase):
    def test_small_train_uses_only_robust_distance(self) -> None:
        result = score_classical_committee(
            train5(), test2(), train_eligibility(5), SEED
        )

        self.assertEqual(result.available_models, ("shrinkage_mahalanobis",))
        self.assertEqual(set(result.scores_by_alpha), {"0.005", "0.01", "0.02", "0.05"})
        self.assertEqual(set(result.confidence_by_alpha), set(result.scores_by_alpha))

    def test_lof_scores_only_unseen_queries(self) -> None:
        spy = SpyLOF()

        score_classical_committee(
            train20(),
            test2(),
            train_eligibility(20),
            SEED,
            lof_factory=lambda: spy,
        )

        self.assertGreater(spy.fit_calls, 1)
        self.assertEqual(spy.score_calls, spy.fit_calls)
        self.assertFalse(spy.scored_fit_rows)

    def test_committee_returns_one_shared_feature_evidence_grade(self) -> None:
        result = score_classical_committee(
            train20(), test2(), train_eligibility(20), SEED
        )

        self.assertEqual(
            result.available_models,
            (
                "shrinkage_mahalanobis",
                "isolation_forest",
                "one_class_svm",
                "local_outlier_factor",
            ),
        )
        self.assertTrue(all(len(values) == len(test2()) for values in result.scores_by_alpha.values()))
        self.assertTrue(all(len(values) == len(test2()) for values in result.confidence_by_alpha.values()))

    def test_committee_never_changes_hard_indices(self) -> None:
        committee = CommitteeScores(
            scores_by_alpha={"0.005": (0.7, 0.8)},
            available_models=("shrinkage_mahalanobis",),
            confidence_by_alpha={"0.005": ("recheck", "high")},
        )

        result = grade_frozen_hard(frozen_hard={1, 4}, committee=committee)

        self.assertEqual(result.hard_indices, frozenset({1, 4}))
        self.assertEqual(result.scores_by_alpha, committee.scores_by_alpha)

    def test_abstained_train_has_no_models_but_grades_every_frozen_hard(self) -> None:
        committee = score_classical_committee(
            train5()[:4], test2(), train_eligibility(4), SEED
        )
        result = grade_frozen_hard(frozen_hard={4, 1}, committee=committee)

        self.assertEqual(result.available_models, ())
        self.assertEqual(result.scores_by_alpha, {})
        self.assertEqual(
            result.confidence_by_alpha,
            {
                "0.005": ("abstain", "abstain"),
                "0.01": ("abstain", "abstain"),
                "0.02": ("abstain", "abstain"),
                "0.05": ("abstain", "abstain"),
            },
        )
        self.assertEqual(result.hard_indices, frozenset({1, 4}))

    def test_rejects_forged_eligibility_for_every_train_boundary(self) -> None:
        forged_cases = (
            (4, train_eligibility(5)),
            (5, TrainEligibility("robust_distance", True, True, False)),
            (19, TrainEligibility("robust_distance", True, True, False)),
            (20, train_eligibility(5)),
            (50, train_eligibility(20)),
        )

        for n_train, forged in forged_cases:
            with self.subTest(n_train=n_train), self.assertRaisesRegex(ValueError, "eligibility"):
                score_classical_committee(train_rows(n_train), test2(), forged, SEED)


if __name__ == "__main__":
    unittest.main()
