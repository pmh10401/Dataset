from __future__ import annotations

import unittest

import numpy as np

from experiments.exp304_reconstruction_fifth_source_common_support.pipeline import (
    build_prediction_rows,
    evaluate_prediction_rows,
)


GATES = ("risk_first_top2", "stability_first_top2", "balanced_rank_top2")


def source_row(variant: str) -> dict[str, object]:
    supports = {4: ["statistical"], 7: []}
    return {
        "dataset": "demo",
        "n_test": 10,
        "gate_variant": variant,
        "automatic_hard_indices": [1],
        "standard_review_indices": [4, 7],
        "candidate_rank_records": [
            {
                "dataset": "demo",
                "candidate_index": index,
                "selected_expert_support_count": 0,
                "verified_expert_support_count": len(experts),
                "supporting_experts": experts,
                "expert_rank_sum": 0,
            }
            for index, experts in supports.items()
        ],
    }


class FifthSourcePipelineTests(unittest.TestCase):
    def test_prediction_requires_exact_common_support_and_contains_no_labels(self) -> None:
        rows = build_prediction_rows(
            [source_row(variant) for variant in GATES],
            reconstruction_support_by_dataset={"demo": {7}},
            expected_datasets=1,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["promoted_indices_by_policy"]["reconstruction_only_residual"], [7])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertFalse(row["automatic_policy_changed"])

    def test_prediction_rejects_dataset_without_reconstruction_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "common-support coverage mismatch"):
            build_prediction_rows(
                [source_row(variant) for variant in GATES],
                reconstruction_support_by_dataset={},
                expected_datasets=1,
            )

    def test_evaluation_reports_reconstruction_only_increment(self) -> None:
        predictions = build_prediction_rows(
            [source_row(variant) for variant in GATES],
            reconstruction_support_by_dataset={"demo": {7}},
            expected_datasets=1,
        )
        results = evaluate_prediction_rows(
            predictions,
            labels_by_dataset={"demo": np.asarray([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])},
        )
        by_policy = {row["expansion_policy"]: row for row in results}

        self.assertEqual(by_policy["reconstruction_only_residual"]["promoted_tp"], 1)
        self.assertEqual(by_policy["reconstruction_only_residual"]["promoted_fp"], 0)
        self.assertFalse(by_policy["reconstruction_only_residual"]["precision_floor_used"])


if __name__ == "__main__":
    unittest.main()
