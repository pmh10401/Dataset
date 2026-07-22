from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp317_318_train_reliability_fusion.contracts import (
    freeze_jsonl,
    verify_frozen_jsonl,
)


class FrozenContractTests(unittest.TestCase):
    def test_freeze_rejects_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                freeze_jsonl(
                    [{"dataset": "D", "contains_test_labels": False,
                      "contains_evaluation_metrics": False, "tp": 1}],
                    root / "rows.jsonl", root / "manifest.json",
                    experiment_id="experiment_test",
                )

    def test_verify_detects_hash_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows_path = root / "rows.jsonl"; manifest_path = root / "manifest.json"
            freeze_jsonl(
                [{"dataset": "D", "contains_test_labels": False,
                  "contains_evaluation_metrics": False}],
                rows_path, manifest_path, experiment_id="experiment_test",
            )
            rows_path.write_text(rows_path.read_text() + json.dumps({"dataset": "tamper"}) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_frozen_jsonl(rows_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
