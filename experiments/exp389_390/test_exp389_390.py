from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp389_390.dual_confidence_core import (
    build_dual_tier_row,
    freeze_predictions,
    load_frozen_predictions,
)
from experiments.exp389_390.run_experiment_390_dual_confidence_evaluation import (
    summarize_detail_rows,
)


def source_row(alpha: float, strong: list[int]) -> dict:
    return {
        "dataset": "demo",
        "n_test": 5,
        "automatic_hard_indices": [1, 2, 3],
        "strong_hard_indices": strong,
        "standard_review_indices": [0],
        "priority_review_indices": [4],
        "automatic_normal_exact_indices": [],
        "quality_conflict_hard_indices": [],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "alpha": alpha,
    }


class DualConfidenceContractTests(unittest.TestCase):
    def test_builds_three_disjoint_tiers_without_changing_hard(self) -> None:
        row = build_dual_tier_row(
            source_row(0.005, [1]),
            source_row(0.010, [1, 2]),
        )
        self.assertEqual(row["highest_confidence_indices"], [1])
        self.assertEqual(row["extended_confidence_indices"], [2])
        self.assertEqual(row["general_confidence_indices"], [3])
        self.assertEqual(row["automatic_hard_indices"], [1, 2, 3])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        forbidden = {"y_test", "tp", "fp", "fn", "precision", "recall", "f1"}
        self.assertFalse(forbidden & set(row))

    def test_rejects_non_nested_strong_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "nested"):
            build_dual_tier_row(
                source_row(0.005, [1]),
                source_row(0.010, [2]),
            )

    def test_rejects_hard_lane_mismatch(self) -> None:
        wider = source_row(0.010, [1, 2])
        wider["automatic_hard_indices"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "Hard lane"):
            build_dual_tier_row(source_row(0.005, [1]), wider)

    def test_frozen_prediction_detects_tampering(self) -> None:
        row = build_dual_tier_row(
            source_row(0.005, [1]),
            source_row(0.010, [1, 2]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.jsonl"
            manifest = root / "manifest.json"
            freeze_predictions([row], prediction, manifest)
            loaded = load_frozen_predictions(prediction, manifest)
            self.assertEqual(loaded, [row])
            with prediction.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_frozen_predictions(prediction, manifest)

    def test_summary_keeps_autonomous_and_confidence_metrics_separate(self) -> None:
        rows = [
            {
                "hard_count": 3,
                "hard_tp": 2,
                "hard_fp": 1,
                "hard_f1": 0.8,
                "highest_count": 1,
                "highest_tp": 1,
                "highest_fp": 0,
                "extended_count": 1,
                "extended_tp": 1,
                "extended_fp": 0,
                "general_count": 1,
                "general_tp": 0,
                "general_fp": 1,
            },
            {
                "hard_count": 1,
                "hard_tp": 1,
                "hard_fp": 0,
                "hard_f1": 1.0,
                "highest_count": 0,
                "highest_tp": 0,
                "highest_fp": 0,
                "extended_count": 1,
                "extended_tp": 1,
                "extended_fp": 0,
                "general_count": 0,
                "general_tp": 0,
                "general_fp": 0,
            },
        ]
        summary = summarize_detail_rows(rows)
        self.assertEqual(summary["hard_count"], 4)
        self.assertEqual(summary["hard_tp"], 3)
        self.assertEqual(summary["hard_fp"], 1)
        self.assertEqual(summary["highest_count"], 1)
        self.assertEqual(summary["extended_count"], 2)
        self.assertEqual(summary["alpha010_strong_count"], 3)
        self.assertAlmostEqual(summary["mean_hard_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()
