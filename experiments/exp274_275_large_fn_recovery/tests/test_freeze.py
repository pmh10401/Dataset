from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp274_275_large_fn_recovery.freeze import verify_prediction


class FreezeTests(unittest.TestCase):
    def test_tampered_prediction_is_rejected_before_truth_loader(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction = root / "prediction.jsonl"
            prediction.write_text('{"dataset":"a"}\n', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                "datasets": 1, "rows": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }), encoding="utf-8")
            prediction.write_text('{"dataset":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_prediction(prediction, manifest, expected_datasets=("a",))


if __name__ == "__main__":
    unittest.main()
