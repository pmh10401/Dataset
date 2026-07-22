from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp292_standard_review_evidence_tiering.contracts import (
    freeze_prediction_rows,
    verify_prediction_rows,
)
from experiments.exp292_standard_review_evidence_tiering.routing import (
    build_stable_family_support,
    tier_standard_review,
)


class StableFamilySupportTests(unittest.TestCase):
    def test_family_casts_one_vote_when_at_least_half_of_variants_support_index(self) -> None:
        rows = [
            {"variant": "a", "candidate_indices": [2, 4]},
            {"variant": "b", "candidate_indices": [2]},
            {"variant": "c", "candidate_indices": [4]},
            {"variant": "d", "candidate_indices": []},
        ]

        self.assertEqual(build_stable_family_support(rows, expected_variants=("a", "b", "c", "d")), {2, 4})

    def test_missing_expected_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "variant coverage"):
            build_stable_family_support(
                [{"variant": "a", "candidate_indices": [1]}],
                expected_variants=("a", "b"),
            )


class ReviewTierTests(unittest.TestCase):
    def test_standard_review_is_partitioned_without_changing_other_lanes(self) -> None:
        exp291 = {
            "dataset": "demo",
            "n_test": 8,
            "automatic_hard_indices": [7],
            "automatic_normal_exact_indices": [0],
            "standard_review_indices": [1, 2, 3, 4],
            "priority_review_indices": [5],
            "shadow_review_indices": [],
            "recovery_shadow_review_indices": [6],
            "research_only_recovery_priority_indices": [],
        }
        row = tier_standard_review(
            exp291,
            {
                "statistical": {1, 2},
                "structural": {1, 3},
                "masked_reconstruction": None,
            },
        )

        self.assertEqual(row["review_priority_1_indices"], [1])
        self.assertEqual(row["review_priority_2_indices"], [2, 3])
        self.assertEqual(row["review_priority_3_indices"], [4])
        self.assertEqual(row["available_source_families"], ["statistical", "structural"])
        self.assertEqual(row["automatic_hard_indices"], [7])
        self.assertEqual(row["standard_review_indices"], [1, 2, 3, 4])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_non_standard_source_hits_are_not_added(self) -> None:
        row = tier_standard_review(
            {"dataset": "demo", "n_test": 5, "standard_review_indices": [2]},
            {"statistical": {1, 2, 3}, "structural": {3}},
        )
        self.assertEqual(row["review_priority_2_indices"], [2])
        self.assertEqual(
            set(row["review_priority_1_indices"])
            | set(row["review_priority_2_indices"])
            | set(row["review_priority_3_indices"]),
            {2},
        )


class FreezeContractTests(unittest.TestCase):
    def test_prediction_rejects_label_or_metric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                freeze_prediction_rows(
                    [{"dataset": "demo", "contains_test_labels": False, "contains_evaluation_metrics": False, "hard_tp": 1}],
                    root / "prediction.jsonl",
                    root / "manifest.json",
                )

    def test_prediction_hash_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.jsonl"
            manifest = root / "manifest.json"
            freeze_prediction_rows(
                [{"dataset": "demo", "contains_test_labels": False, "contains_evaluation_metrics": False}],
                prediction,
                manifest,
            )
            prediction.write_text(prediction.read_text(encoding="utf-8") + json.dumps({"dataset": "extra"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_prediction_rows(prediction, manifest)


if __name__ == "__main__":
    unittest.main()
