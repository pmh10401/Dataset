from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp211_214 import run_experiment_211_214 as runner
from run_rank_experiments_sequential import EXP211_214_BATCH, EXPERIMENTS


class Exp211214StructureTests(unittest.TestCase):
    def test_prediction_builder_does_not_accept_labels(self):
        self.assertNotIn("y_test", inspect.signature(runner.build_prediction_rows).parameters)
        self.assertNotIn("labels", inspect.signature(runner.build_prediction_rows).parameters)

    def test_stage_ids_are_frozen(self):
        self.assertEqual(list(runner.STAGE_META), ["211", "212", "213", "214"])

    def test_sequential_runner_registers_every_stage(self):
        self.assertEqual(len(EXP211_214_BATCH), 4)
        registered = {row["id"] for row in EXPERIMENTS}
        self.assertTrue(set(EXP211_214_BATCH) <= registered)

    def test_fold_cache_loader_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            with path.open("wb") as handle:
                np.savez_compressed(handle, metadata_json='{"config_hash":"x"}', values=[1])
            expected = runner.sha256_file(path)
            runner._load_verified_fold_artifact(path, expected)
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                runner._load_verified_fold_artifact(path, expected)


if __name__ == "__main__":
    unittest.main()
