from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp239_248_diverse_score_sources.evaluation import (
    outside_lane_indices,
    verify_prediction_manifest,
)


class ReportingContractTest(unittest.TestCase):
    def test_outside_lane_excludes_every_frozen_lane(self) -> None:
        outside = outside_lane_indices({1, 2, 3, 4}, hard={1}, standard={2}, priority={3})
        self.assertEqual(outside, frozenset({4}))

    def test_evaluation_rejects_prediction_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "prediction.jsonl"
            manifest = Path(directory) / "manifest.json"
            prediction.write_text("{}\n")
            manifest.write_text('{"prediction_sha256":"deadbeef"}\n')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_prediction_manifest(prediction, manifest)


if __name__ == "__main__":
    unittest.main()
