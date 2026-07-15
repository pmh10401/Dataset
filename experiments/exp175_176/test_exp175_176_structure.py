from __future__ import annotations

import unittest

import numpy as np

from experiments.exp175_176 import diagnostic_core as core
from experiments.exp175_176 import run_experiment_175_176_large_train_miss_diagnostic as runner


class StructureTests(unittest.TestCase):
    def test_frozen_record_rejects_evaluation_fields(self):
        with self.assertRaisesRegex(ValueError, "evaluation field"):
            core.validate_frozen_record(
                {
                    "dataset": "demo",
                    "n_train": 20,
                    "n_test": 3,
                    "sources": {},
                    "hard_tp": 1,
                }
            )

    def test_frozen_record_checks_source_lengths(self):
        record = {
            "dataset": "demo",
            "n_train": 20,
            "n_test": 3,
            "sources": {
                name: {
                    "train": np.arange(20, dtype=float),
                    "test": np.arange(3, dtype=float),
                }
                for name in core.SOURCE_NAMES
            },
        }
        core.validate_frozen_record(record)
        record["sources"][core.SOURCE_NAMES[0]]["test"] = np.arange(2, dtype=float)
        with self.assertRaisesRegex(ValueError, "TEST score length"):
            core.validate_frozen_record(record)

    def test_frozen_payload_contains_no_evaluation_fields(self):
        record = {
            "dataset": "demo",
            "n_train": 20,
            "n_test": 3,
            "sources": {
                name: {
                    "train": np.arange(20, dtype=float),
                    "test": np.arange(3, dtype=float),
                }
                for name in core.SOURCE_NAMES
            },
        }
        payload = runner.freeze_score_diagnostics(record, "cache-sha")
        forbidden = {"y_test", "labels", "tp", "fp", "fn", "f1", "anomaly_count"}
        self.assertFalse(forbidden & set(payload))
        self.assertEqual(payload["cache_sha256"], "cache-sha")
        self.assertEqual(len(payload["mean_rank"]), 3)

    def test_parse_indices_accepts_empty_and_space_separated(self):
        self.assertEqual(runner.parse_indices(""), set())
        self.assertEqual(runner.parse_indices("4 1 4"), {1, 4})


if __name__ == "__main__":
    unittest.main()
