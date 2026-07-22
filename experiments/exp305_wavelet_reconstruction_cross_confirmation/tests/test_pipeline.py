from __future__ import annotations

import unittest

from experiments.exp305_wavelet_reconstruction_cross_confirmation.pipeline import (
    build_prediction_rows,
)


class Exp305PipelineTests(unittest.TestCase):
    def test_build_rows_preserves_hard_and_freezes_wavelet_evidence(self):
        source = [{
            "dataset": "D",
            "n_test": 6,
            "automatic_hard_indices": [0],
            "standard_review_indices": [1, 2],
            "candidate_policy_records": [
                {"candidate_index": 1, "existing_support_count": 0, "reconstruction_support": True},
                {"candidate_index": 2, "existing_support_count": 1, "reconstruction_support": False},
            ],
        }]
        rows = build_prediction_rows(source, wavelet_support_by_dataset={"D": {1, 2}}, expected_datasets=1)
        self.assertEqual(rows[0]["automatic_hard_indices"], [0])
        self.assertEqual(
            rows[0]["promoted_indices_by_policy"]["reconstruction_wavelet_cross_confirmed"],
            [1],
        )
        self.assertEqual(rows[0]["promoted_indices_by_policy"]["six_source_2of6"], [1, 2])
        self.assertFalse(rows[0]["contains_test_labels"])
        self.assertFalse(rows[0]["automatic_policy_changed"])

    def test_missing_wavelet_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            build_prediction_rows([], wavelet_support_by_dataset={"D": set()}, expected_datasets=2)


if __name__ == "__main__":
    unittest.main()
