from __future__ import annotations

import unittest

import numpy as np

from experiments.exp226_235_hard_alert_quality.exp226_train_risk import (
    conditional_risk_lanes,
    fold_local_pvalues,
    restore_fold_concatenated,
)
from experiments.exp226_235_hard_alert_quality.exp227_independent_confirmation import (
    confirmation_tiers,
)
from experiments.exp226_235_hard_alert_quality.exp228_margin_confidence import (
    conformal_margin,
    normalized_score_margin,
)
from experiments.exp226_235_hard_alert_quality.exp229_correlation_aware_agreement import (
    correlation_adjusted_weights,
    spearman_matrix,
)


class Exp226Tests(unittest.TestCase):
    def test_restore_fold_concatenated_returns_original_order(self) -> None:
        plan = [
            (np.asarray([1, 3]), np.asarray([0, 2])),
            (np.asarray([0, 2]), np.asarray([1, 3])),
        ]
        actual = restore_fold_concatenated(
            np.asarray([0.1, 0.3, 0.2, 0.4]), plan, 4
        )
        np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])

    def test_restore_rejects_duplicate_query_indices(self) -> None:
        plan = [
            (np.asarray([2]), np.asarray([0, 1])),
            (np.asarray([0]), np.asarray([1, 2])),
        ]
        with self.assertRaisesRegex(ValueError, "exactly once"):
            restore_fold_concatenated(np.arange(4), plan, 3)

    def test_fold_local_pvalues_excludes_query_rows(self) -> None:
        plan = [
            (np.asarray([0, 1]), np.asarray([2, 3])),
            (np.asarray([2, 3]), np.asarray([0, 1])),
        ]
        actual = fold_local_pvalues(np.asarray([1.0, 2.0, 100.0, 200.0]), plan)
        np.testing.assert_allclose(actual, [1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0])

    def test_conditional_lanes_partition_shape_candidates(self) -> None:
        lanes = conditional_risk_lanes(
            shape_p=np.asarray([0.001, 0.001, 0.001, 0.5]),
            temporal_p=np.asarray([0.001, 0.001, 0.5, 0.001]),
            canonical_p=np.asarray([0.001, 0.5, 0.5, 0.001]),
            shape_alpha=0.005,
            confirmation_alpha=0.01,
        )
        self.assertEqual(lanes["dual"], {0})
        self.assertEqual(lanes["single"], {1})
        self.assertEqual(lanes["unconfirmed"], {2})


class Exp227Tests(unittest.TestCase):
    def test_confirmation_tiers_partition_baseline_hard(self) -> None:
        tiers = confirmation_tiers({1, 2, 3}, {1, 2}, {1, 3})
        self.assertEqual(tiers.dual, frozenset({1}))
        self.assertEqual(tiers.single, frozenset({2, 3}))
        self.assertEqual(tiers.unconfirmed, frozenset())


class Exp228Tests(unittest.TestCase):
    def test_normalized_score_margin(self) -> None:
        actual = normalized_score_margin(np.asarray([1.0, 2.0, 3.0]), 2.0)
        np.testing.assert_allclose(actual, [-0.5, 0.0, 0.5])

    def test_conformal_margin_is_positive_inside_alpha(self) -> None:
        actual = conformal_margin(np.asarray([0.001, 0.01, 0.02]), 0.01)
        np.testing.assert_allclose(actual, [0.9, 0.0, -1.0])


class Exp229Tests(unittest.TestCase):
    def test_spearman_matrix_uses_train_vectors(self) -> None:
        matrix = spearman_matrix(
            {
                "a": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "b": np.asarray([4.0, 3.0, 2.0, 1.0]),
                "c": np.asarray([1.0, 3.0, 2.0, 4.0]),
            }
        )
        self.assertAlmostEqual(matrix["a"]["b"], -1.0)
        self.assertAlmostEqual(matrix["a"]["a"], 1.0)

    def test_correlation_weights_penalize_duplicate_sources(self) -> None:
        weights = correlation_adjusted_weights(
            {
                "a": {"a": 1.0, "b": 1.0, "c": 0.0},
                "b": {"a": 1.0, "b": 1.0, "c": 0.0},
                "c": {"a": 0.0, "b": 0.0, "c": 1.0},
            }
        )
        self.assertLess(weights["a"], weights["c"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
