from __future__ import annotations

import unittest

import numpy as np

from experiments.exp317_318_train_reliability_fusion.reliability import (
    audit_source_reliability,
    deterministic_reference_audit_splits,
    empirical_upper_pvalues,
    reliability_weight,
    wilson_upper_one_sided,
)


class ReliabilityTests(unittest.TestCase):
    def test_reference_and_audit_are_disjoint_complete_and_deterministic(self):
        first = deterministic_reference_audit_splits(25, seed=20260722, splits=5)
        second = deterministic_reference_audit_splits(25, seed=20260722, splits=5)
        self.assertEqual(first, second)
        for reference, audit in first:
            self.assertTrue(set(reference).isdisjoint(audit))
            self.assertEqual(sorted(reference + audit), list(range(25)))

    def test_empirical_upper_pvalues_include_finite_sample_correction(self):
        actual = empirical_upper_pvalues([1.0, 2.0, 3.0], [0.0, 4.0])
        np.testing.assert_allclose(actual, [1.0, 0.25])

    def test_wilson_upper_bound_is_larger_for_more_false_alerts(self):
        self.assertGreater(
            wilson_upper_one_sided(5, 20),
            wilson_upper_one_sided(0, 20),
        )

    def test_reliability_weight_decreases_as_normal_risk_increases(self):
        self.assertGreater(
            reliability_weight(0.05, n_audit_min=20),
            reliability_weight(0.25, n_audit_min=20),
        )

    def test_audit_abstains_below_twenty_scores(self):
        actual = audit_source_reliability(np.arange(19.0), alpha=0.05, seed=20260722)
        self.assertFalse(actual.available)
        self.assertEqual(actual.reason, "insufficient_train_reference")


if __name__ == "__main__":
    unittest.main()
