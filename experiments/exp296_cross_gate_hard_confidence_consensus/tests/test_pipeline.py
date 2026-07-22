from __future__ import annotations

import unittest

import numpy as np

from experiments.exp296_cross_gate_hard_confidence_consensus.pipeline import (
    build_prediction_rows,
    evaluate_prediction_rows,
)


VARIANTS = (
    "risk_first_top2",
    "stability_first_top2",
    "balanced_rank_top2",
)


def exp295_rows() -> list[dict[str, object]]:
    counts = {
        VARIANTS[0]: {1: 2, 2: 1, 3: 1, 4: 0},
        VARIANTS[1]: {1: 2, 2: 1, 3: 0, 4: 0},
        VARIANTS[2]: {1: 2, 2: 2, 3: 0, 4: 0},
    }
    return [
        {
            "dataset": "demo",
            "n_test": 5,
            "gate_variant": variant,
            "gate_abstained": False,
            "automatic_hard_indices": [1, 2, 3, 4],
            "hard_confidence_records": [
                {
                    "candidate_index": index,
                    "selected_expert_support_count": count,
                    "verified_expert_support_count": count,
                }
                for index, count in sorted(values.items())
            ],
        }
        for variant, values in counts.items()
    ]


class ConsensusPipelineTests(unittest.TestCase):
    def test_prediction_collapses_three_gate_rows_without_changing_hard_indices(self) -> None:
        rows = build_prediction_rows(exp295_rows=exp295_rows(), expected_datasets=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["automatic_hard_indices"], [1, 2, 3, 4])
        self.assertEqual(len(rows[0]["consensus_records"]), 4)
        self.assertFalse(rows[0]["contains_test_labels"])
        self.assertFalse(rows[0]["contains_evaluation_metrics"])
        self.assertFalse(rows[0]["automatic_policy_changed"])

    def test_evaluation_reports_all_hard_and_each_consensus_tier(self) -> None:
        predictions = build_prediction_rows(exp295_rows=exp295_rows(), expected_datasets=1)
        results = evaluate_prediction_rows(
            predictions=predictions,
            labels_by_dataset={"demo": np.asarray([0, 1, 0, 1, 1], dtype=np.int8)},
            scopes=("all_1117",),
        )
        by_tier = {row["consensus_tier"]: row for row in results}

        self.assertEqual(by_tier["all_hard"]["hard_tp"], 3)
        self.assertEqual(by_tier["all_hard"]["hard_fp"], 1)
        self.assertEqual(by_tier["unanimous_high"]["hard_tp"], 1)
        self.assertEqual(by_tier["unanimous_supported"]["hard_fp"], 1)
        self.assertEqual(by_tier["partial_supported"]["hard_tp"], 1)
        self.assertEqual(by_tier["no_selected_support"]["hard_tp"], 1)
        self.assertEqual(
            sum(by_tier[tier]["hard_count"] for tier in (
                "unanimous_high", "unanimous_supported",
                "partial_supported", "no_selected_support",
            )),
            4,
        )


if __name__ == "__main__":
    unittest.main()
