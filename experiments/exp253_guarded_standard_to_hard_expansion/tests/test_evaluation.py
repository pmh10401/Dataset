from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.exp253_guarded_standard_to_hard_expansion.evaluation import metrics, verify_prediction_manifest


class EvaluationContractTest(unittest.TestCase):
    def test_empty_selection_metrics(self) -> None:
        result = metrics(set(), {1})
        self.assertEqual(result, {
            "count": 0, "tp": 0, "fp": 0, "fn": 1,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        })

    def test_prediction_tamper_is_rejected(self) -> None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "frozen_predictions.jsonl"
            prediction.write_text("{}\n")
            manifest = root / "prediction_manifest.json"
            manifest.write_text(json.dumps({"prediction_sha256": "bad"}))
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                verify_prediction_manifest(prediction, manifest)


if __name__ == "__main__":
    unittest.main()
