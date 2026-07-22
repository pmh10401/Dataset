from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp278_281_train_only_recovery.evaluation import evaluate_verifier_predictions


class VerifierBoundaryTest(unittest.TestCase):
    def test_tamper_prevents_label_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); prediction = root / "prediction.jsonl"; manifest = root / "manifest.json"
            row = {
                "dataset": "demo", "n_train": 10, "n_test": 3, "hard_indices": [],
                "candidate_records": [], "contains_test_labels": False,
                "contains_evaluation_metrics": False, "hard_policy_changed": False,
                "review_only": True, "retrospective": True,
            }
            prediction.write_text(json.dumps(row, sort_keys=True) + "\n")
            manifest.write_text(json.dumps({"prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}))
            prediction.write_text(prediction.read_text() + " ")
            called = False

            def loader(_datasets):
                nonlocal called
                called = True
                return {"demo": set()}

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_verifier_predictions(
                    prediction, manifest, expected_datasets=("demo",), output_root=root,
                    truth_loader=loader,
                )
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
