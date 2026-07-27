from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp400_402.artifact_io import (
    MANIFEST_NAME,
    PREDICTION_NAME,
    load_verified_prediction,
    write_prediction,
)
from experiments.exp400_402.prediction import build_shadow_predictions
from experiments.exp400_402.test_prediction import exp292_fixture, exp397_fixture


class ArtifactIoTests(unittest.TestCase):
    def test_round_trip_and_manifest_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
            manifest = write_prediction(root, rows, full_run=False)
            loaded, verified = load_verified_prediction(
                root / PREDICTION_NAME,
                root / MANIFEST_NAME,
            )
            self.assertEqual(loaded, rows)
            self.assertEqual(verified["datasets"], 3)
            self.assertFalse(manifest["contains_test_labels"])
            self.assertFalse(manifest["prospective"])

    def test_rejects_modified_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
            write_prediction(root, rows, full_run=False)
            with (root / PREDICTION_NAME).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_prediction(
                    root / PREDICTION_NAME,
                    root / MANIFEST_NAME,
                )


if __name__ == "__main__":
    unittest.main()
