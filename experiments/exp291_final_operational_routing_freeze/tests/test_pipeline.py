from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.exp291_final_operational_routing_freeze.evaluation import evaluate_rows
from experiments.exp291_final_operational_routing_freeze.contracts import freeze_rows, verify_rows


class EvaluationScopeTest(unittest.TestCase):
    def test_frozen_routing_rejects_hash_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.jsonl"
            manifest = root / "manifest.json"
            freeze_rows([{"dataset": "demo", "contains_test_labels": False, "contains_evaluation_metrics": False}], prediction, manifest)
            self.assertEqual(len(verify_rows(prediction, manifest)), 1)
            prediction.write_text(prediction.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_rows(prediction, manifest)

    def test_autonomous_and_review_metrics_remain_separate(self) -> None:
        prediction = {
            "dataset": "demo", "n_test": 5,
            "source_baseline_hard_indices": [0, 1],
            "automatic_hard_indices": [1], "automatic_normal_exact_indices": [0],
            "standard_review_indices": [2], "priority_review_indices": [],
            "shadow_review_indices": [], "recovery_shadow_review_indices": [3],
            "research_only_recovery_priority_indices": [4],
            "default_review_indices": [2, 3],
        }
        labels = {"demo": np.asarray([0, 1, 0, 1, 1], dtype=np.int8)}
        evaluated, summary = evaluate_rows([prediction], labels)
        self.assertEqual(summary["automatic_hard_count"], 1)
        self.assertEqual(summary["automatic_hard_tp"], 1)
        self.assertEqual(summary["automatic_hard_fp"], 0)
        self.assertEqual(summary["automatic_normal_exact_fp"], 1)
        self.assertEqual(summary["default_review_count"], 2)
        self.assertEqual(summary["default_review_tp"], 1)
        self.assertEqual(summary["default_review_fp"], 1)
        self.assertEqual(summary["research_only_recovery_priority_tp"], 1)
        self.assertTrue(summary["review_metrics_human_assisted_only"])
        self.assertEqual(evaluated[0]["automatic_hard_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
