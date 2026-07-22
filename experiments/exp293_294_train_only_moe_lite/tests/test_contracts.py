from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, verify_rows


class PredictionContractTests(unittest.TestCase):
    def test_forbidden_evaluation_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                freeze_rows(
                    [{"dataset": "demo", "contains_test_labels": False, "contains_evaluation_metrics": False, "tp": 1}],
                    root / "rows.jsonl", root / "manifest.json", experiment_id="experiment_294",
                )

    def test_prediction_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / "rows.jsonl"
            manifest = root / "manifest.json"
            freeze_rows(
                [{"dataset": "demo", "contains_test_labels": False, "contains_evaluation_metrics": False}],
                rows, manifest, experiment_id="experiment_294",
            )
            rows.write_text(rows.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_rows(rows, manifest)


if __name__ == "__main__":
    unittest.main()
