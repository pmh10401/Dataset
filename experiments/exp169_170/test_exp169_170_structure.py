from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp169_170 import run_experiment_169_170_oob_pooled_calibration as runner
from experiments.exp169_170 import independent_reaggregate_exp169_170 as reaggregate


class TestExp169170Structure(unittest.TestCase):
    def test_prediction_signature_has_no_label_argument(self):
        parameters = inspect.signature(runner.run_prediction).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_prediction_source_does_not_load_labels(self):
        source = inspect.getsource(runner.run_prediction)
        self.assertNotIn("load_test_labels", source)
        self.assertNotIn("y_test", source)

    def test_value_loader_selects_no_label_column(self):
        source = inspect.getsource(runner.load_values_only)
        self.assertIn("SELECT i.values_blob", source)
        self.assertNotIn("i.label", source)

    def test_contract_constants_are_frozen(self):
        self.assertEqual(
            runner.DB_PATH,
            Path("/Users/minho/Documents/Dataset/univariate_ts.db"),
        )
        self.assertEqual(runner.EXPECTED_DATASETS, 316)
        self.assertEqual(runner.TARGET_MIN_TRAIN, 5)
        self.assertEqual(runner.TARGET_MAX_TRAIN_EXCLUSIVE, 20)
        self.assertEqual(runner.ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(runner.PREDICTION_ROWS_PER_DATASET, 60)

    def test_contract_records_the_effective_minimum_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            runner.write_contract(output_dir, minimum_reference=20)
            contract = json.loads((output_dir / "01_contract.json").read_text())

        self.assertEqual(contract["config"]["minimum_reference"], 20)
        self.assertEqual(
            contract["config_hash"],
            runner.stable_hash(runner.frozen_config(minimum_reference=20)),
        )
        self.assertNotEqual(
            contract["config_hash"],
            runner.stable_hash(runner.frozen_config(minimum_reference=200)),
        )

    def test_oob_score_cache_hash_is_independent_of_pool_reference_policy(self):
        score_config = runner.frozen_score_config()
        self.assertNotIn("minimum_reference", score_config)
        self.assertNotEqual(
            runner.stable_hash(runner.frozen_config(minimum_reference=20)),
            runner.stable_hash(runner.frozen_config(minimum_reference=200)),
        )
        self.assertEqual(
            runner.score_config_hash(),
            runner.stable_hash(score_config),
        )

    def test_runner_has_no_forbidden_policy_symbols(self):
        source = inspect.getsource(runner)
        forbidden = (
            "large_data_budget",
            "tail_replacement",
            "WEAK_FAMILIES",
            "family_guard_v1",
            "high_confidence_indices",
            "review_candidate_indices",
            "load_routing_maps",
        )
        for symbol in forbidden:
            self.assertNotIn(symbol, source)

    def test_comparison_input_parse_errors_are_not_silently_ignored(self):
        source = inspect.getsource(runner.run_evaluation)
        self.assertNotIn("except Exception:\n            pass", source)

    def test_prediction_and_evaluation_are_separate(self):
        prediction_source = inspect.getsource(runner.run_prediction)
        evaluation_source = inspect.getsource(runner.run_evaluation)
        self.assertNotIn("load_test_labels", prediction_source)
        self.assertIn("verify_prediction_manifest", evaluation_source)
        self.assertIn("load_test_labels", evaluation_source)

    def test_prediction_grid_rejects_missing_variant(self):
        rows = []
        for policy in runner.POLICIES:
            for alpha in runner.ALPHAS:
                for source_config in runner.SOURCE_CONFIGS:
                    rows.append(
                        {
                            "dataset": "demo",
                            "pool_policy": policy,
                            "alpha": alpha,
                            "source_config": source_config,
                            "n_test": 3,
                            "candidate_indices": [],
                            "candidate_count": 0,
                            "contains_test_labels": False,
                            "contains_evaluation_metrics": False,
                        }
                    )

        runner.validate_prediction_grid(rows, expected_datasets=1)
        with self.assertRaisesRegex(ValueError, "prediction grid"):
            runner.validate_prediction_grid(rows[:-1], expected_datasets=1)

    def test_load_cache_rejects_component_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.npz"
            cached = {
                "dataset": "demo",
                "config_hash": "config",
                "n_train": 5,
                "n_test": 3,
                "q_d": 2,
                "target_lengths": np.full(32, 16, dtype=np.int32),
                "mean_target_length": 16.0,
                "train_components": {
                    source: np.zeros((31, 5)) for source in runner.SOURCES
                },
                "test_components": {
                    source: np.zeros((32, 3)) for source in runner.SOURCES
                },
                "aggregated_train": {
                    source: np.zeros(5) for source in runner.SOURCES
                },
                "aggregated_test": {
                    source: np.zeros(3) for source in runner.SOURCES
                },
            }
            runner._write_cache(path, cached)

            with self.assertRaisesRegex(ValueError, "shape"):
                runner._load_cache(path, "demo", "config")

    def test_prediction_manifest_verifies_dependency_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            prediction_path = output_dir / "predictions.jsonl"
            cache_manifest = output_dir / "02_oob_cache_manifest.csv"
            pool_manifest = output_dir / "03_pool_manifest.csv"
            prediction_path.write_text("{}\n")
            cache_manifest.write_text("cache\n")
            pool_manifest.write_text("pool\n")
            runner.write_json(
                output_dir / "01_contract.json",
                {"config_hash": "policy"},
            )
            runner.write_json(
                output_dir / "07_prediction_manifest.json",
                {
                    "prediction_file": prediction_path.name,
                    "prediction_sha256": runner.sha256_file(prediction_path),
                    "cache_manifest_file": cache_manifest.name,
                    "cache_manifest_sha256": runner.sha256_file(cache_manifest),
                    "pool_manifest_file": pool_manifest.name,
                    "pool_manifest_sha256": runner.sha256_file(pool_manifest),
                    "config_hash": "policy",
                },
            )
            cache_manifest.write_text("tampered\n")

            with self.assertRaisesRegex(ValueError, "cache manifest hash mismatch"):
                runner.verify_prediction_manifest(output_dir)

    def test_prediction_manifest_rejects_contract_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            prediction_path = output_dir / "predictions.jsonl"
            cache_manifest = output_dir / "02_oob_cache_manifest.csv"
            pool_manifest = output_dir / "03_pool_manifest.csv"
            prediction_path.write_text("{}\n")
            cache_manifest.write_text("cache\n")
            pool_manifest.write_text("pool\n")
            runner.write_json(output_dir / "01_contract.json", {"config_hash": "expected"})
            runner.write_json(
                output_dir / "07_prediction_manifest.json",
                {
                    "prediction_file": prediction_path.name,
                    "prediction_sha256": runner.sha256_file(prediction_path),
                    "cache_manifest_file": cache_manifest.name,
                    "cache_manifest_sha256": runner.sha256_file(cache_manifest),
                    "pool_manifest_file": pool_manifest.name,
                    "pool_manifest_sha256": runner.sha256_file(pool_manifest),
                    "config_hash": "wrong",
                },
            )

            with self.assertRaisesRegex(ValueError, "contract config hash mismatch"):
                runner.verify_prediction_manifest(output_dir)

    def test_cache_manifest_rejects_tampered_cache_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            cache_path = output_dir / "cache" / "demo_oob_cache.npz"
            cached = {
                "dataset": "demo",
                "config_hash": "config",
                "n_train": 5,
                "n_test": 3,
                "q_d": 2,
                "target_lengths": np.full(32, 16, dtype=np.int32),
                "mean_target_length": 16.0,
                "train_components": {
                    source: np.zeros((32, 5)) for source in runner.SOURCES
                },
                "test_components": {
                    source: np.zeros((32, 3)) for source in runner.SOURCES
                },
                "aggregated_train": {
                    source: np.zeros(5) for source in runner.SOURCES
                },
                "aggregated_test": {
                    source: np.zeros(3) for source in runner.SOURCES
                },
            }
            runner._write_cache(cache_path, cached)
            runner.write_csv(
                output_dir / "02_oob_cache_manifest.csv",
                [
                    {
                        "dataset": "demo",
                        "status": "computed",
                        "n_train": 5,
                        "n_test": 3,
                        "q_d": 2,
                        "mean_target_length": 16.0,
                        "cache_file": str(cache_path),
                        "cache_sha256": runner.sha256_file(cache_path),
                        "config_hash": "config",
                        "score_config_hash": "config",
                    }
                ],
            )
            cache_path.write_bytes(cache_path.read_bytes() + b"tampered")

            with self.assertRaisesRegex(ValueError, "cache hash mismatch"):
                runner.load_validated_cache_packs(
                    output_dir,
                    expected_datasets=["demo"],
                    config_hash="config",
                    include_test=True,
                )

    def test_database_hash_gate_rejects_mismatch(self):
        with patch.object(runner, "sha256_file", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "database hash mismatch"):
                runner.verify_database_hash()

    def test_pool_manifest_rejects_policy_config_mismatch(self):
        packs = {}
        for index, dataset in enumerate(("A", "B", "C", "D", "E")):
            packs[dataset] = {
                "n_train": 10,
                "target_length": 100 + index,
                "oob_scores": {
                    source: np.arange(10, dtype=float) + index
                    for source in runner.SOURCES
                },
            }
        rows, _ = runner.build_pool_manifest_rows(
            packs,
            minimum_reference=10,
            donor_cap=5,
            policy_config_hash="expected",
        )
        rows[0]["policy_config_hash"] = "wrong"

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            runner.write_csv(output_dir / "03_pool_manifest.csv", rows)
            with self.assertRaisesRegex(ValueError, "pool manifest"):
                runner.validate_pool_manifest_file(
                    output_dir,
                    packs,
                    minimum_reference=10,
                    donor_cap=5,
                    policy_config_hash="expected",
                )

    def test_independent_reaggregation_rejects_tp_fp_count_mismatch(self):
        prediction = {
            "dataset": "demo",
            "pool_policy": "similarity_pool",
            "alpha": 0.01,
            "source_config": "2of3",
            "n_test": 3,
            "candidate_indices": [0, 2],
            "candidate_count": 2,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        metric = {
            "dataset": "demo",
            "pool_policy": "similarity_pool",
            "alpha": 0.01,
            "source_config": "2of3",
            "n_test": 3,
            "predicted_count": 2,
            "tp": 1,
            "fp": 0,
            "fn": 1,
        }
        with self.assertRaisesRegex(ValueError, "TP plus FP"):
            reaggregate.validate_metric_prediction_alignment([prediction], [metric])

    def test_independent_reaggregation_output_is_not_wall_clock_dependent(self):
        source = inspect.getsource(reaggregate)
        self.assertNotIn('"timestamp": time.strftime', source)


if __name__ == "__main__":
    unittest.main()
