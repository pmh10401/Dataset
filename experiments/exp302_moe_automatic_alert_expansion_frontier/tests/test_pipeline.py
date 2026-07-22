from __future__ import annotations

import unittest

import numpy as np

from experiments.exp302_moe_automatic_alert_expansion_frontier.pipeline import (
    build_prediction_rows,
    evaluate_prediction_rows,
)
from experiments.exp302_moe_automatic_alert_expansion_frontier.tests.test_routing import (
    GATES,
    gate_row,
)


class ExpansionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_rows = [
            gate_row(GATES[0], {4: 2, 7: 0}),
            gate_row(GATES[1], {4: 2, 7: 1}),
            gate_row(GATES[2], {4: 2, 7: 0}),
        ]

    def test_prediction_is_label_free_and_preserves_source_lanes(self) -> None:
        rows = build_prediction_rows(self.source_rows, expected_datasets=1)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["automatic_hard_indices"], [1])
        self.assertEqual(row["standard_review_indices"], [4, 7])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertFalse(row["automatic_policy_changed"])
        for forbidden in ("tp", "fp", "precision", "f1", "y_test"):
            self.assertNotIn(forbidden, row)

    def test_evaluation_reports_baseline_and_incremental_expansion(self) -> None:
        predictions = build_prediction_rows(self.source_rows, expected_datasets=1)
        results = evaluate_prediction_rows(
            predictions,
            labels_by_dataset={"demo": np.asarray([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
            scopes=("all_1117",),
        )
        by_policy = {row["expansion_policy"]: row for row in results}

        self.assertEqual(by_policy["baseline_no_expansion"]["automatic_tp"], 1)
        self.assertEqual(by_policy["baseline_no_expansion"]["automatic_fp"], 0)
        self.assertEqual(by_policy["all_gates_two_experts"]["promoted_tp"], 1)
        self.assertEqual(by_policy["all_gates_two_experts"]["promoted_fp"], 0)
        self.assertEqual(by_policy["any_gate_any_support"]["promoted_tp"], 1)
        self.assertEqual(by_policy["any_gate_any_support"]["promoted_fp"], 1)
        self.assertGreater(
            by_policy["any_gate_any_support"]["automatic_recall"],
            by_policy["baseline_no_expansion"]["automatic_recall"],
        )


if __name__ == "__main__":
    unittest.main()
