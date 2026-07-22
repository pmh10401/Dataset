from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp292_standard_review_evidence_tiering.contracts import verify_prediction_rows
from experiments.exp292_standard_review_evidence_tiering.run_pipeline import run_prediction


class FullPredictionContractTests(unittest.TestCase):
    def test_frozen_inputs_produce_expected_label_free_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = run_prediction(root)
            rows = verify_prediction_rows(
                root / "exp292_frozen_review_tiers.jsonl",
                root / "exp292_prediction_manifest.json",
            )

        self.assertEqual(manifest["rows"], 1117)
        self.assertEqual(manifest["tier_counts"], {"review_priority_1": 17, "review_priority_2": 57, "review_priority_3": 732})
        self.assertEqual(sum(len(row["standard_review_indices"]) for row in rows), 806)
        self.assertEqual(sum(len(row["review_priority_1_indices"]) for row in rows), 17)
        self.assertEqual(sum(len(row["review_priority_2_indices"]) for row in rows), 57)
        self.assertEqual(sum(len(row["review_priority_3_indices"]) for row in rows), 732)


if __name__ == "__main__":
    unittest.main()
