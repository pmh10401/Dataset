from __future__ import annotations

import unittest

import numpy as np

from experiments.exp303_expanded_train_only_gates.pipeline import (
    build_dataset_prediction,
    evaluate_prediction_rows,
)


GATES = ("risk_first_top2", "stability_first_top2", "balanced_rank_top2")


def source_row(variant: str) -> dict[str, object]:
    supports = {
        4: ["statistical", "structural", "change_point", "official_multirocket"],
        7: ["change_point"],
        8: ["statistical", "structural"],
    }
    return {
        "dataset": "demo",
        "n_test": 12,
        "gate_variant": variant,
        "automatic_hard_indices": [1],
        "standard_review_indices": [4, 7, 8],
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


class ExpandedGatePipelineTests(unittest.TestCase):
    def test_prediction_builds_all_expert_and_new_gate_variants(self) -> None:
        row = build_dataset_prediction(
            [source_row(variant) for variant in GATES],
            diversity_pair=["change_point", "statistical"],
            calibration_pair=["statistical", "structural"],
        )

        promoted = row["promoted_indices_by_policy"]
        self.assertEqual(promoted["verified_4of4"], [4])
        self.assertEqual(promoted["verified_2of4"], [4, 8])
        self.assertEqual(promoted["verified_1of4"], [4, 7, 8])
        self.assertEqual(promoted["diversity_2of2"], [4])
        self.assertEqual(promoted["diversity_1of2"], [4, 7, 8])
        self.assertEqual(promoted["calibration_2of2"], [4, 8])
        self.assertEqual(promoted["calibration_1of2"], [4, 8])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_evaluation_occurs_after_prediction_and_reports_incremental_quality(self) -> None:
        prediction = build_dataset_prediction(
            [source_row(variant) for variant in GATES],
            diversity_pair=["change_point", "statistical"],
            calibration_pair=["statistical", "structural"],
        )
        results = evaluate_prediction_rows(
            [prediction],
            labels_by_dataset={"demo": np.asarray([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])},
            scopes=("all_1117",),
        )
        by_policy = {row["expansion_policy"]: row for row in results}

        self.assertEqual(by_policy["verified_4of4"]["promoted_tp"], 1)
        self.assertEqual(by_policy["verified_4of4"]["promoted_fp"], 0)
        self.assertEqual(by_policy["verified_1of4"]["promoted_tp"], 1)
        self.assertEqual(by_policy["verified_1of4"]["promoted_fp"], 2)
        self.assertFalse(by_policy["verified_1of4"]["precision_floor_used"])


if __name__ == "__main__":
    unittest.main()
