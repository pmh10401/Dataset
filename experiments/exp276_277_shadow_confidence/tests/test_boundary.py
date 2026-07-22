from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp276_277_shadow_confidence.evaluation import evaluate_confidence_predictions
from experiments.exp276_277_shadow_confidence.freeze import verify_confidence_prediction


class BoundaryTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        row = {
            "dataset": "demo", "n_train": 10, "n_test": 4,
            "hard_indices": [0], "recovery_shadow_indices": [2],
            "candidate_records": [{
                "candidate_index": 2, "confidence_score": 0.8,
                "guard_all_deep": True, "guard_median_deep": True,
                "guard_consistent_deep": True,
            }],
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "hard_policy_changed": False, "review_only": True, "retrospective": True,
        }
        prediction = root / "prediction.jsonl"
        prediction.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "rows": 1, "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        }))
        return prediction, manifest

    def test_prediction_boundary_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prediction, manifest = self._fixture(Path(raw))
            self.assertEqual(len(verify_confidence_prediction(prediction, manifest, expected_datasets=("demo",))), 1)
            prediction.write_text(prediction.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_confidence_prediction(prediction, manifest, expected_datasets=("demo",))

    def test_labels_are_loaded_only_after_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prediction, manifest = self._fixture(Path(raw))
            prediction.write_text(prediction.read_text() + " ")
            called = False

            def loader(_datasets):
                nonlocal called
                called = True
                return {"demo": {2}}

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_confidence_predictions(
                    prediction, manifest, expected_datasets=("demo",),
                    output_root=Path(raw), truth_loader=loader,
                )
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
