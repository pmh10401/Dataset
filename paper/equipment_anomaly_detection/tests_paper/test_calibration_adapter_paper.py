"""Contracts for continuous evidence and route-preserving paper scores."""
from __future__ import annotations

import unittest

import numpy as np

from paper.equipment_anomaly_detection.calibration_paper import (
    leaf_evidence,
    mark_missing_evidence,
    upper_tail_pvalues,
)
from paper.equipment_anomaly_detection.score_adapter_paper import (
    combine_evidence,
    route_preserving_scores,
    strict_native_pass,
)


class CalibrationAdapterTests(unittest.TestCase):
    def test_empirical_p_equality_passes(self) -> None:
        leaf = leaf_evidence(np.arange(99.0), np.asarray([100.0]), alpha=0.01)
        self.assertEqual(leaf.pvalue.tolist(), [0.01])
        self.assertEqual(leaf.native_pass.tolist(), [True])
        self.assertGreaterEqual(leaf.margin[0], 0.0)

    def test_strict_raw_equality_fails(self) -> None:
        self.assertFalse(strict_native_pass(np.asarray([3.0]), np.asarray([3.0]))[0])

    def test_route_score_reproduces_alerts_at_the_band_boundary(self) -> None:
        scores = route_preserving_scores(np.asarray([0.0, 0.9]), np.asarray([False, True]))
        np.testing.assert_array_equal(scores >= 0.5, np.asarray([False, True]))
        np.testing.assert_array_equal(scores, np.asarray([0.0, 0.95]))

    def test_all_valid_strengths_preserve_routes(self) -> None:
        strength = np.asarray([0.0, 0.1, 0.5, np.nextafter(1.0, 0.0)])
        alerts = np.asarray([False, True, False, True])
        np.testing.assert_array_equal(route_preserving_scores(strength, alerts) >= 0.5, alerts)

    def test_strength_equal_to_one_is_outside_the_evidence_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "strength"):
            route_preserving_scores(np.asarray([1.0]), np.asarray([False]))

    def test_upper_tail_uses_exact_finite_sample_correction(self) -> None:
        pvalue = upper_tail_pvalues(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 4.0]))
        np.testing.assert_array_equal(pvalue, np.asarray([0.75, 0.25]))

    def test_leaf_evidence_is_immutable_and_has_a_false_missing_sentinel(self) -> None:
        leaf = leaf_evidence(np.asarray([1.0, 2.0]), np.asarray([3.0]), alpha=1.0)
        self.assertEqual(leaf.evidence_missing.tolist(), [False])
        with self.assertRaises(ValueError):
            leaf.strength[0] = 0.0

    def test_empty_query_produces_aligned_empty_evidence(self) -> None:
        leaf = leaf_evidence(np.asarray([1.0]), np.asarray([], dtype=np.float64), alpha=0.5)
        self.assertEqual(leaf.pvalue.dtype, np.dtype(np.float64))
        self.assertEqual(leaf.pvalue.size, 0)
        self.assertEqual(leaf.native_pass.dtype, np.dtype(bool))
        self.assertEqual(leaf.evidence_missing.size, 0)

    def test_alpha_accepts_one_and_rejects_invalid_values(self) -> None:
        self.assertTrue(leaf_evidence(np.asarray([1.0]), np.asarray([1.0]), alpha=1.0).native_pass[0])
        for alpha in (0.0, -0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                leaf_evidence(np.asarray([1.0]), np.asarray([1.0]), alpha=alpha)

    def test_combine_evidence_uses_boolean_strength_aggregations(self) -> None:
        left = np.asarray([0.1, 0.8, 0.4])
        right = np.asarray([0.6, 0.2, 0.9])
        third = np.asarray([0.5, 0.7, 0.3])
        np.testing.assert_array_equal(combine_evidence((left, right, third), "and"), np.asarray([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(combine_evidence((left, right, third), "or"), np.asarray([0.6, 0.8, 0.9]))
        np.testing.assert_array_equal(combine_evidence((left, right, third), "k_of_n", k=2), np.asarray([0.5, 0.7, 0.4]))

    def test_missing_evidence_uses_the_explicit_neutral_sentinel(self) -> None:
        leaf = leaf_evidence(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), alpha=0.5)
        marked = mark_missing_evidence(leaf, np.asarray([False, True]))
        self.assertEqual(marked.pvalue.tolist(), [1.0, 0.5])
        self.assertEqual(marked.margin.tolist(), [np.log10(0.5), 0.0])
        self.assertEqual(marked.strength.tolist(), [0.0, 0.5])
        self.assertEqual(marked.native_pass.tolist(), [False, False])
        self.assertEqual(marked.evidence_missing.tolist(), [False, True])

    def test_missing_mask_overrides_stale_strength_before_banding(self) -> None:
        scores = route_preserving_scores(
            np.asarray([0.9, 0.1]), np.asarray([False, True]), np.asarray([True, True])
        )
        np.testing.assert_array_equal(scores, np.asarray([0.25, 0.75]))
        np.testing.assert_array_equal(scores >= 0.5, np.asarray([False, True]))

    def test_invalid_shapes_finiteness_ranges_and_masks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            upper_tail_pvalues(np.asarray([]), np.asarray([1.0]))
        with self.assertRaisesRegex(ValueError, "reference"):
            upper_tail_pvalues(np.asarray([[1.0]]), np.asarray([1.0]))
        with self.assertRaisesRegex(ValueError, "query"):
            upper_tail_pvalues(np.asarray([1.0]), np.asarray([float("nan")]))
        with self.assertRaisesRegex(ValueError, "aligned"):
            strict_native_pass(np.asarray([1.0]), np.asarray([0.0, 1.0]))
        with self.assertRaisesRegex(ValueError, "raw_scores"):
            strict_native_pass(np.asarray([True]), np.asarray([0.0]))
        with self.assertRaisesRegex(ValueError, "aligned"):
            combine_evidence((np.asarray([0.1]), np.asarray([0.2, 0.3])), "and")
        with self.assertRaisesRegex(ValueError, "mask"):
            route_preserving_scores(np.asarray([0.1]), np.asarray([True]), np.asarray([1]))
        with self.assertRaisesRegex(ValueError, "mask"):
            route_preserving_scores(np.asarray([0.1]), np.asarray([True]), np.asarray([True, False]))
        with self.assertRaisesRegex(ValueError, "strength"):
            route_preserving_scores(np.asarray([float("nan")]), np.asarray([True]))
        with self.assertRaisesRegex(ValueError, "strength"):
            combine_evidence((np.asarray([0.1]), np.asarray([1.0])), "or")
        with self.assertRaisesRegex(ValueError, "k"):
            combine_evidence((np.asarray([0.1]),), "k_of_n", k=2)


if __name__ == "__main__":
    unittest.main()
