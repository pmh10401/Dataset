from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from experiments.exp167_168 import run_experiment_167_168_score_calibration_parity as runner


class Exp167168StructureTests(unittest.TestCase):
    def test_prediction_signature_has_no_label_argument(self):
        parameters = inspect.signature(runner.predict_dataset).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_prediction_source_does_not_load_labels(self):
        source = inspect.getsource(runner.predict_dataset)
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
        self.assertEqual(runner.PREDICTION_ROWS_PER_DATASET, 20)

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

    def test_prediction_and_evaluation_are_separate(self):
        prediction_source = inspect.getsource(runner.run_prediction)
        evaluation_source = inspect.getsource(runner.run_evaluation)
        self.assertNotIn("load_test_labels", prediction_source)
        self.assertIn("verify_prediction_manifest", evaluation_source)
        self.assertIn("load_test_labels", evaluation_source)

    def test_prediction_grid_requires_every_fixed_row_once(self):
        p_values = {
            source: runner.np.array([0.01, 0.50]) for source in runner.SOURCES
        }
        rows = runner.candidate_rows(
            dataset="demo",
            n_train=9,
            n_test=2,
            p_values=p_values,
            config_hash="abc",
        )
        runner.validate_prediction_grid(rows, expected_datasets=1)

        with self.assertRaises(ValueError):
            runner.validate_prediction_grid(rows[:-1], expected_datasets=1)
        with self.assertRaises(ValueError):
            runner.validate_prediction_grid(rows + [dict(rows[0])], expected_datasets=1)

    def test_prediction_manifest_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            prediction_path = output_dir / "04_frozen_predictions.jsonl"
            prediction_path.write_text('{"dataset":"demo"}\n', encoding="utf-8")
            runner.write_json(
                output_dir / "05_prediction_manifest.json",
                {
                    "prediction_file": prediction_path.name,
                    "prediction_sha256": runner.sha256_file(prediction_path),
                },
            )
            runner.verify_prediction_manifest(output_dir)
            prediction_path.write_text('{"dataset":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                runner.verify_prediction_manifest(output_dir)

    def test_stable_hash_is_independent_of_mapping_order(self):
        self.assertEqual(
            runner.stable_hash({"a": 1, "b": 2}),
            runner.stable_hash({"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
