from __future__ import annotations

import unittest

from experiments.exp276_277_shadow_confidence.policy import confidence_record, normalized_tail_margin


class ConfidencePolicyTests(unittest.TestCase):
    def test_margin_is_zero_at_threshold_and_one_at_attainable_floor(self) -> None:
        self.assertEqual(normalized_tail_margin(0.005, 999), 0.0)
        self.assertAlmostEqual(normalized_tail_margin(0.001, 999), 1.0)

    def test_guards_require_two_low_risk_groups(self) -> None:
        row = confidence_record([0.8, 0.7], [0.004, 0.003])
        self.assertTrue(row["guard_all_deep"])
        self.assertTrue(row["guard_median_deep"])
        self.assertTrue(row["guard_consistent_deep"])
        self.assertFalse(confidence_record([0.9], [0.001])["guard_all_deep"])
        self.assertFalse(confidence_record([0.9, 0.9], [0.001, 0.006])["guard_all_deep"])

    def test_prediction_features_are_label_free(self) -> None:
        row = confidence_record([0.2, 0.8], [0.003, 0.004])
        self.assertTrue({"tp", "fp", "f1", "y_test"}.isdisjoint(row))


if __name__ == "__main__":
    unittest.main()
