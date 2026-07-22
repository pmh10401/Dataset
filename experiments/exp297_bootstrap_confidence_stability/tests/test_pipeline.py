from __future__ import annotations

import unittest

import numpy as np

from experiments.exp297_bootstrap_confidence_stability.pipeline import (
    build_prediction_rows,
    evaluate_prediction_rows,
)


VARIANTS = (
    "risk_first_top2",
    "stability_first_top2",
    "balanced_rank_top2",
)
EXPERTS = ("statistical", "structural", "change_point", "official_multirocket")


class StabilityPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exp296 = [{
            "dataset": "demo",
            "n_test": 3,
            "automatic_hard_indices": [1],
            "consensus_records": [{
                "candidate_index": 1,
                "consensus_tier": "no_selected_support",
                "supporting_gate_count": 0,
                "high_gate_count": 0,
            }],
        }]
        self.exp295 = [{
            "dataset": "demo",
            "gate_variant": variant,
            "automatic_hard_indices": [1],
            "hard_confidence_records": [{
                "candidate_index": 1,
                "supporting_experts": [],
                "selected_expert_support_count": 0,
            }],
        } for variant in VARIANTS]
        self.train = {
            expert: {"demo": {
                "v1": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
                "v2": np.asarray([0.2, 0.1, 0.4, 0.3, 0.6]),
            }}
            for expert in EXPERTS
        }

    def test_prediction_preserves_hard_and_adds_label_free_stability(self) -> None:
        progress = []
        rows = build_prediction_rows(
            exp296_rows=self.exp296,
            exp295_rows=self.exp295,
            primary_train_scores=self.train,
            expected_datasets=1,
            replicates=4,
            seed=20260722,
            alpha=0.005,
            progress_callback=lambda dataset, done: progress.append((dataset, done)),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["automatic_hard_indices"], [1])
        record = rows[0]["stability_records"][0]
        self.assertEqual(record["consensus_tier"], "no_selected_support")
        self.assertEqual(record["stability_rate"], 1.0)
        self.assertNotIn("tp", record)
        self.assertFalse(rows[0]["contains_test_labels"])
        self.assertFalse(rows[0]["contains_evaluation_metrics"])
        self.assertEqual(progress, [("demo", 1)])

    def test_evaluation_reports_tier_metrics_and_stability_without_routing(self) -> None:
        predictions = build_prediction_rows(
            exp296_rows=self.exp296,
            exp295_rows=self.exp295,
            primary_train_scores=self.train,
            expected_datasets=1,
            replicates=4,
            seed=20260722,
            alpha=0.005,
        )
        results = evaluate_prediction_rows(
            predictions=predictions,
            labels_by_dataset={"demo": np.asarray([0, 1, 0], dtype=np.int8)},
            scopes=("all_1117",),
        )
        by_tier = {row["consensus_tier"]: row for row in results}

        self.assertEqual(by_tier["all_hard"]["hard_tp"], 1)
        self.assertEqual(by_tier["no_selected_support"]["hard_tp"], 1)
        self.assertEqual(by_tier["no_selected_support"]["eligible_alerts"], 1)
        self.assertEqual(by_tier["no_selected_support"]["median_stability_rate"], 1.0)
        self.assertFalse(by_tier["all_hard"]["automatic_policy_changed"])


if __name__ == "__main__":
    unittest.main()
