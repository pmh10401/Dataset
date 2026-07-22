from __future__ import annotations

import unittest

import numpy as np

from experiments.exp295_moe_hard_confidence_tiering.pipeline import (
    build_prediction_rows,
    evaluate_prediction_rows,
)


GATES = (
    "risk_first_top2",
    "stability_first_top2",
    "balanced_rank_top2",
)


class ConfidencePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = [{"dataset": "demo", "n_test": 4, "automatic_hard_indices": [1, 2]}]
        self.gates = [
            {
                "dataset": "demo",
                "gate_variant": variant,
                "ordered_experts": ["statistical", "structural", "change_point", "official_multirocket"],
                "selected_experts": ["statistical", "structural"],
            }
            for variant in GATES
        ]
        self.support = {
            "statistical": {"demo": {1}},
            "structural": {"demo": {1}},
            "change_point": {"demo": {2}},
            "official_multirocket": {"demo": set()},
        }

    def test_prediction_preserves_hard_set_for_every_gate_without_metrics(self) -> None:
        rows = build_prediction_rows(
            baseline_rows=self.baseline,
            gate_rows=self.gates,
            all_support=self.support,
            expected_datasets=1,
        )

        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["automatic_hard_indices"], [1, 2])
            self.assertEqual(row["automatic_policy_changed"], False)
            self.assertEqual(row["contains_test_labels"], False)
            self.assertEqual(row["contains_evaluation_metrics"], False)
            self.assertNotIn("hard_tp", row)
            self.assertEqual(
                {record["candidate_index"] for record in row["hard_confidence_records"]},
                {1, 2},
            )

    def test_evaluation_reports_all_hard_and_exclusive_tiers(self) -> None:
        predictions = build_prediction_rows(
            baseline_rows=self.baseline,
            gate_rows=self.gates,
            all_support=self.support,
            expected_datasets=1,
        )

        results = evaluate_prediction_rows(
            predictions=predictions,
            labels_by_dataset={"demo": np.asarray([0, 1, 0, 1], dtype=np.int8)},
            scopes=("all_1117",),
        )

        risk = {row["confidence_tier"]: row for row in results if row["gate_variant"] == "risk_first_top2"}
        self.assertEqual(risk["all_hard"]["hard_tp"], 1)
        self.assertEqual(risk["all_hard"]["hard_fp"], 1)
        self.assertEqual(risk["high"]["hard_tp"], 1)
        self.assertEqual(risk["high"]["hard_fp"], 0)
        self.assertEqual(risk["needs_confirmation"]["hard_tp"], 0)
        self.assertEqual(risk["needs_confirmation"]["hard_fp"], 1)
        self.assertEqual(
            sum(risk[tier]["hard_count"] for tier in ("high", "medium", "needs_confirmation")),
            risk["all_hard"]["hard_count"],
        )


if __name__ == "__main__":
    unittest.main()
