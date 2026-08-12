from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.config import CLASSICAL_METHOD_IDS
from experiments.paper_public_baselines.methods.classical import (
    crossfit_scores,
    score_classical,
)


class ClassicalMethodTests(unittest.TestCase):
    def test_all_classical_scores_are_finite_and_point_upward(self) -> None:
        rng = np.random.default_rng(7)
        train = rng.normal(0.0, 0.05, size=(30, 4))
        query = np.asarray([[0.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]])

        for method_id in CLASSICAL_METHOD_IDS:
            with self.subTest(method_id=method_id):
                first = score_classical(method_id, train, query, seed=7)
                second = score_classical(method_id, train, query, seed=7)
                self.assertEqual(first.shape, (2,))
                self.assertTrue(np.isfinite(first).all())
                np.testing.assert_array_equal(first, second)
                self.assertGreater(first[1], first[0])

    def test_crossfit_produces_aligned_oof_and_test_scores(self) -> None:
        train = np.arange(48, dtype=float).reshape(12, 4)
        test = np.asarray([[0.0, 1.0, 2.0, 3.0], [99.0, 99.0, 99.0, 99.0]])

        result = crossfit_scores("knn_distance", train, test, seed=11)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.reason, None)
        self.assertEqual(result.oof_scores.shape, (12,))
        self.assertEqual(result.test_scores.shape, (2,))
        self.assertTrue(np.isfinite(result.oof_scores).all())
        self.assertTrue(np.isfinite(result.test_scores).all())

    def test_lof_abstains_instead_of_substituting_another_detector(self) -> None:
        result = crossfit_scores(
            "lof_novelty", np.zeros((1, 2)), np.zeros((2, 2)), seed=7
        )

        self.assertEqual(result.status, "abstain")
        self.assertEqual(result.reason, "insufficient_train_rows")
        self.assertEqual(result.oof_scores.size, 0)
        self.assertEqual(result.test_scores.size, 0)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            score_classical("invented", np.zeros((2, 2)), np.zeros((1, 2)), 7)


if __name__ == "__main__":
    unittest.main()
