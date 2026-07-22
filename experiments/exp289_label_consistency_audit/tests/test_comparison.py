from __future__ import annotations

import unittest

import numpy as np

from experiments.exp289_label_consistency_audit.comparison import compare_pair


class PairComparisonTest(unittest.TestCase):
    def test_raw_exact_has_highest_priority(self) -> None:
        row = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        result = compare_pair(row, row.copy(), row, row.copy())
        self.assertEqual(result.tier, "raw_exact")
        self.assertTrue(result.raw_exact)

    def test_aligned_exact_is_distinct_from_raw_exact(self) -> None:
        left_raw = np.asarray([0.0, 2.0], dtype=np.float32)
        right_raw = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        aligned = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        result = compare_pair(left_raw, right_raw, aligned, aligned.copy())
        self.assertEqual(result.tier, "aligned_exact")
        self.assertFalse(result.raw_exact)

    def test_scale_and_shift_equivalent_rows_are_normalized_equivalent(self) -> None:
        left = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        right = left * 3.0 + 7.0
        result = compare_pair(left, right, left, right)
        self.assertEqual(result.tier, "normalized_equivalent")
        self.assertLessEqual(result.normalized_rmse, 1e-6)

    def test_near_duplicate_uses_preregistered_joint_limits(self) -> None:
        left = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
        right = left.copy()
        right[50] += np.float32(0.02)
        result = compare_pair(left, right, left, right)
        self.assertEqual(result.tier, "near_duplicate")
        self.assertLessEqual(result.normalized_rmse, 0.01)
        self.assertLessEqual(result.correlation_distance, 1e-4)

    def test_different_rows_are_distinct(self) -> None:
        left = np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        right = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        result = compare_pair(left, right, left, right)
        self.assertEqual(result.tier, "distinct")

    def test_different_constant_rows_are_not_normalized_equivalent(self) -> None:
        left = np.ones(8, dtype=np.float32)
        right = np.full(8, 2.0, dtype=np.float32)
        result = compare_pair(left, right, left, right)
        self.assertEqual(result.tier, "distinct")


if __name__ == "__main__":
    unittest.main()
