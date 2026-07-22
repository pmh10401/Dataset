from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp254_sktime_structural_features.evaluation import metrics, verify_prediction_manifest
from experiments.exp254_sktime_structural_features.prediction import build_prediction_row, predict_experiment


class StructuralEvaluationTest(unittest.TestCase):
    def test_prediction_row_has_disjoint_views_and_no_labels(self) -> None:
        row = build_prediction_row(
            variant="demo", dataset="demo", alpha=0.005, n_test=10,
            candidates={1, 2, 3, 7}, hard={1}, standard={2, 3}, priority={4},
            exp253_hard={1, 2}, exp253_standard={3, 5},
            abstained=False, resolution_blocked=False, attainable_p=0.005,
        )
        self.assertEqual(row["hard_confirmation_indices"], [1])
        self.assertEqual(row["standard_priority_indices"], [2, 3])
        self.assertEqual(row["outside_lane_indices"], [7])
        self.assertEqual(row["exp253_hard_confirmation_indices"], [1, 2])
        self.assertEqual(row["exp253_remaining_standard_priority_indices"], [3])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertNotIn("y_test", inspect.signature(predict_experiment).parameters)

    def test_prediction_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "frozen_predictions.jsonl"
            prediction.write_text("{}\n")
            manifest = root / "prediction_manifest.json"
            manifest.write_text(json.dumps({"prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}))
            prediction.write_text("{}\n{}\n")
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                verify_prediction_manifest(prediction, manifest)

    def test_empty_metrics_are_zero(self) -> None:
        self.assertEqual(metrics([], {1})["precision"], 0.0)
        self.assertEqual(metrics([], {1})["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
