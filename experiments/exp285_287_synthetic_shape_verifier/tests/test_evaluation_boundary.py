import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp285_287_synthetic_shape_verifier.evaluation import evaluate_predictions


class EvaluationBoundaryTest(unittest.TestCase):
    def test_tamper_stops_before_labels(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); prediction = root / "prediction.jsonl"; manifest = root / "manifest.json"
            row = {"dataset": "demo", "n_train": 100, "n_test": 1, "hard_indices": [], "candidate_records": [],
                   "contains_test_labels": False, "contains_evaluation_metrics": False}
            prediction.write_text(json.dumps(row) + "\n"); manifest.write_text(json.dumps({"prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}))
            prediction.write_text(prediction.read_text() + " "); called = False

            def loader(_datasets):
                nonlocal called; called = True; return {"demo": set()}

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_predictions(prediction, manifest, output_root=root, truth_loader=loader)
            self.assertFalse(called)


if __name__ == "__main__": unittest.main()
