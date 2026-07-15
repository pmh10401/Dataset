from __future__ import annotations

import inspect
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


if __name__ == "__main__":
    unittest.main()
