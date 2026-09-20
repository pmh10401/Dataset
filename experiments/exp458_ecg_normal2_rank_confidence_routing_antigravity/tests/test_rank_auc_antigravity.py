from __future__ import annotations

import math
import unittest

import numpy as np


def direct_auc_antigravity(normal: np.ndarray, anomaly: np.ndarray) -> float:
    wins = 0.0
    for anomaly_value in anomaly:
        for normal_value in normal:
            if anomaly_value > normal_value:
                wins += 1.0
            elif anomaly_value == normal_value:
                wins += 0.5
    return wins / float(normal.size * anomaly.size)


class RankAucAntigravityTest(unittest.TestCase):
    def test_auc_counts_wins_and_half_ties_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
            mann_whitney_auc_antigravity,
        )

        self.assertEqual(
            mann_whitney_auc_antigravity(
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([1.0, 2.0], dtype=np.float64),
            ),
            0.875,
        )
        self.assertEqual(
            mann_whitney_auc_antigravity(
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
            ),
            0.125,
        )
        self.assertEqual(
            mann_whitney_auc_antigravity(
                np.array([1.0, 1.0], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
            ),
            0.5,
        )

    def test_auc_rejects_non_exact_real_finite_nonempty_vectors_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
            mann_whitney_auc_antigravity,
        )

        valid = np.array([0.0, 1.0], dtype=np.float64)
        invalid_vectors = (
            np.array([], dtype=np.float64),
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([True, False], dtype=np.bool_),
            np.array([1 + 0j], dtype=np.complex128),
            np.array(["1.0"], dtype=np.str_),
            np.array([object()], dtype=np.object_),
            np.array([0.0, math.nan], dtype=np.float64),
            np.array([0.0, math.inf], dtype=np.float64),
        )
        for vector in invalid_vectors:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                mann_whitney_auc_antigravity(vector, valid)
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                mann_whitney_auc_antigravity(valid, vector)

    def test_chunked_auc_matches_direct_pairwise_formula_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
            mann_whitney_auc_antigravity,
        )

        normal = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        anomaly = np.array([0.0, 0.5, 0.75, 2.0, 3.0], dtype=np.float64)

        self.assertEqual(
            mann_whitney_auc_antigravity(normal, anomaly),
            direct_auc_antigravity(normal, anomaly),
        )

    def test_integer_auc_preserves_exact_comparisons_past_float53_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
            mann_whitney_auc_antigravity,
        )

        normal = np.array([2**53], dtype=np.int64)
        anomaly = np.array([2**53 + 1], dtype=np.int64)
        self.assertEqual(mann_whitney_auc_antigravity(normal, anomaly), 1.0)

    def test_integer_chunked_auc_matches_direct_pairwise_formula_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
            mann_whitney_auc_antigravity,
        )

        normal = np.array([2**53 - 1, 2**53, 2**53 + 2, 2**53 + 4], dtype=np.int64)
        anomaly = np.array([2**53, 2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.int64)
        self.assertEqual(
            mann_whitney_auc_antigravity(normal, anomaly),
            direct_auc_antigravity(normal, anomaly),
        )


if __name__ == "__main__":
    unittest.main()
