from __future__ import annotations

import csv
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from experiments.exp206_210 import artifacts
from experiments.exp206_210.run_experiment_206_210 import STAGE_META


class Exp206210StructureTests(unittest.TestCase):
    def test_prediction_loaders_do_not_accept_y_test(self):
        self.assertNotIn("y_test", inspect.signature(artifacts.load_population_values_only).parameters)

    def test_evaluation_label_loader_is_separate(self):
        self.assertTrue(callable(artifacts.load_test_labels_for_evaluation))

    def test_stage_ids_are_frozen(self):
        self.assertEqual(list(STAGE_META), ["206", "207", "208", "209", "210"])

    def test_runner_can_start_by_absolute_script_path_without_pythonpath(self):
        script = Path(__file__).with_name("run_experiment_206_210.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                env={"PATH": str(Path(sys.executable).parent)},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    @patch("experiments.exp206_210.run_experiment_206_210.load_population_values_only")
    @patch("experiments.exp206_210.run_experiment_206_210.load_dataset_series_only")
    @patch("experiments.exp206_210.run_experiment_206_210.load_current_oof_cache")
    @patch("experiments.exp206_210.run_experiment_206_210.load_exp206_cache")
    @patch("experiments.exp206_210.run_experiment_206_210.load_test_labels_for_evaluation")
    def test_pipeline_stages_run_with_mocked_io(
        self,
        mock_load_labels,
        mock_load_exp206,
        mock_load_current,
        mock_load_series,
        mock_load_pop,
    ):
        # Mock population values (2 datasets)
        mock_load_pop.return_value = [
            {"dataset": "ds_A", "n_train": 100, "n_test": 50, "hard_indices": {1, 2}, "standard_indices": set(), "priority_indices": set()},
            {"dataset": "ds_B", "n_train": 150, "n_test": 60, "hard_indices": set(), "standard_indices": {3}, "priority_indices": set()},
        ]

        # Mock database series loading
        def side_effect_series(dataset):
            n_train = 100 if dataset == "ds_A" else 150
            n_test = 50 if dataset == "ds_A" else 60
            return (
                np.random.randn(n_train, 50),
                np.random.randn(n_test, 50),
                50
            )
        mock_load_series.side_effect = side_effect_series

        # Mock current OOF cache (4 sources)
        mock_load_current.return_value = {
            "rocket_exp40_train": np.random.randn(100),
            "rocket_exp40_test": np.random.randn(50),
            "exp55_best_train": np.random.randn(100),
            "exp55_best_test": np.random.randn(50),
            "exp56_best_train": np.random.randn(100),
            "exp56_best_test": np.random.randn(50),
            "exp84_train": np.random.randn(100),
            "exp84_test": np.random.randn(50),
        }

        # Mock test labels
        mock_load_labels.return_value = np.zeros(50, dtype=np.int8)

        # Let's run all stages in a temporary output root
        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)

            # Stage 206
            from experiments.exp206_210.run_experiment_206_210 import run_stage_206
            run_stage_206(None, out_root, 1)
            self.assertTrue((out_root / "01_score_cache_manifest.csv").exists())
            self.assertTrue((out_root / "cache" / "ds_A.npz").exists())
            self.assertTrue((out_root / "cache" / "ds_B.npz").exists())

            # Now mock load_exp206_cache to return the saved cached files
            def side_effect_exp206(dataset):
                cache_path = out_root / "cache" / f"{dataset}.npz"
                with np.load(cache_path, allow_pickle=False) as archive:
                    return {
                        f"{src}_{split}": np.asarray(archive[f"{src}_{split}"], dtype=np.float32)
                        for src in ["spectral", "shapelet", "interval"]
                        for split in ["train", "test"]
                    }
            mock_load_exp206.side_effect = side_effect_exp206

            # Stage 207
            from experiments.exp206_210.run_experiment_206_210 import run_stage_207
            run_stage_207(None, out_root)
            self.assertTrue((out_root / "exp207_frozen_rules.json").exists())
            self.assertTrue((out_root / "exp207_summary.csv").exists())

            # Stage 208
            from experiments.exp206_210.run_experiment_206_210 import run_stage_208
            # In order to make evaluation work, we need mock_load_labels to return arrays of correct size (n_test)
            def side_effect_labels(dataset):
                n_test = 50 if dataset == "ds_A" else 60
                return np.zeros(n_test, dtype=np.int8)
            mock_load_labels.side_effect = side_effect_labels

            # Also mock_load_current to return arrays of correct size (100 and 150 train, 50 and 60 test)
            def side_effect_current(dataset):
                n_train = 100 if dataset == "ds_A" else 150
                n_test = 50 if dataset == "ds_A" else 60
                return {
                    "rocket_exp40_train": np.random.randn(n_train),
                    "rocket_exp40_test": np.random.randn(n_test),
                    "exp55_best_train": np.random.randn(n_train),
                    "exp55_best_test": np.random.randn(n_test),
                    "exp56_best_train": np.random.randn(n_train),
                    "exp56_best_test": np.random.randn(n_test),
                    "exp84_train": np.random.randn(n_train),
                    "exp84_test": np.random.randn(n_test),
                }
            mock_load_current.side_effect = side_effect_current

            from experiments.exp206_210.run_experiment_206_210 import run_stage_208
            run_stage_208(None, out_root)
            self.assertTrue((out_root / "exp208_frozen_predictions.jsonl").exists())
            self.assertTrue((out_root / "experiment_208_continuous_alternative_source_fusion_results.csv").exists())

            # Stage 209
            from experiments.exp206_210.run_experiment_206_210 import run_stage_209
            run_stage_209(None, out_root)
            self.assertTrue((out_root / "exp209_frozen_predictions.jsonl").exists())
            self.assertTrue((out_root / "experiment_209_seven_source_joint_normal_novelty_results.csv").exists())

            # Stage 210
            from experiments.exp206_210.run_experiment_206_210 import run_stage_210
            run_stage_210(None, out_root)
            self.assertTrue((out_root / "experiment_210_independent_frontier_reaggregation_results.csv").exists())
            self.assertTrue((out_root / "experiment_210_independent_frontier_reaggregation_summary.csv").exists())
            self.assertTrue((out_root / "exp210_candidate_overlap.csv").exists())


if __name__ == "__main__":
    unittest.main()
