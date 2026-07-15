from __future__ import annotations

import unittest
import numpy as np

from experiments.exp169_170.oob_core import (
    OOBSplit,
    balanced_oob_plan,
    robust_model_scale,
    aggregate_oob_scores,
)


class TestExp169170OOBCore(unittest.TestCase):
    def test_balanced_oob_plan_is_deterministic_and_covers_every_train_row(self):
        first = balanced_oob_plan(7)
        second = balanced_oob_plan(7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        counts = np.zeros(7, dtype=int)
        for split in first:
            counts[list(split.holdout_indices)] += 1
            self.assertEqual(len(split.holdout_indices), 3)
            self.assertEqual(len(split.fit_indices), 4)
            self.assertFalse(set(split.fit_indices) & set(split.holdout_indices))
        self.assertGreaterEqual(counts.min(), 1)

    def test_train_and_test_aggregate_the_same_component_count(self):
        plan = balanced_oob_plan(5)
        train_components = np.arange(32 * 5, dtype=float).reshape(32, 5)
        test_components = np.arange(32 * 3, dtype=float).reshape(32, 3)
        train, test, q_d = aggregate_oob_scores(plan, train_components, test_components)

        expected_q_d = min(sum(i in s.holdout_indices for s in plan) for i in range(5))
        self.assertEqual(q_d, expected_q_d)
        self.assertEqual(train.shape, (5,))
        self.assertEqual(test.shape, (3,))

    def test_robust_model_scale_prevents_heavy_tail_amplification(self):
        # When MAD is very small, we should fallback to std or epsilon
        scores = np.array([1.0, 1.0, 1.0, 1.0, 2.0]) # median=1.0, MAD=0.0
        center, scale = robust_model_scale(scores)
        self.assertEqual(center, 1.0)
        std = np.std(scores)
        self.assertGreater(scale, 0.0)
        self.assertAlmostEqual(scale, max(0.0, 0.0, std, 1e-9))

    def test_non_finite_inputs_raise_value_error(self):
        plan = balanced_oob_plan(5)
        train_components = np.ones((32, 5))
        test_components = np.ones((32, 3))

        # Test nan in train
        train_components[0, 0] = np.nan
        with self.assertRaises(ValueError):
            aggregate_oob_scores(plan, train_components, test_components)

        train_components[0, 0] = 1.0
        test_components[0, 0] = np.inf
        with self.assertRaises(ValueError):
            aggregate_oob_scores(plan, train_components, test_components)

    def test_aggregate_rejects_non_matrix_components(self):
        plan = balanced_oob_plan(5)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            aggregate_oob_scores(plan, np.zeros(32), np.zeros((32, 3)))

    def test_aggregate_rejects_invalid_model_index_contract(self):
        plan = balanced_oob_plan(5)
        broken = [
            OOBSplit(1, plan[0].fit_indices, plan[0].holdout_indices),
            *plan[1:],
        ]
        with self.assertRaisesRegex(ValueError, "model indices"):
            aggregate_oob_scores(broken, np.zeros((32, 5)), np.zeros((32, 3)))


if __name__ == "__main__":
    unittest.main()
