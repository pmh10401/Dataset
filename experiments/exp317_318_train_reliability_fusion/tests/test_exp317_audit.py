from __future__ import annotations

import unittest

from experiments.exp317_318_train_reliability_fusion.exp317_audit import audit_one_source
from experiments.exp317_318_train_reliability_fusion.source_loader import validate_source_dataset_sets


class Exp317AuditTests(unittest.TestCase):
    def test_exp317_abstains_below_twenty_scores(self):
        row = audit_one_source("D", "multiscale_patch_discord", 0.05, list(range(19)))
        self.assertFalse(row["available"])
        self.assertEqual(row["abstention_reason"], "insufficient_train_reference")
        self.assertEqual(row["weight"], 0.0)

    def test_exp317_available_row_contains_no_evaluation_fields(self):
        row = audit_one_source("D", "multiscale_patch_discord", 0.05, list(range(40)))
        self.assertTrue(row["available"])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertNotIn("tp", row)
        self.assertNotIn("precision", row)

    def test_source_loader_rejects_coverage_mismatch(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_source_dataset_sets(
                {"discord": {"A"}, "svdd": {"A", "B"}},
                frozen_datasets={"A", "B"}, expected=2,
            )

    def test_source_loader_accepts_equal_frozen_coverage(self):
        validate_source_dataset_sets(
            {"discord": {"A", "B"}, "svdd": {"A", "B"}},
            frozen_datasets={"A", "B"}, expected=2,
        )


if __name__ == "__main__":
    unittest.main()
