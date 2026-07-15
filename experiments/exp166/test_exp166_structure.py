from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from experiments.exp166 import exp166_core as core
from experiments.exp166 import run_experiment_166_small_sample_anomaly_coverage as runner


class Exp166StructureTests(unittest.TestCase):
    def test_prediction_signature_has_no_label_argument(self):
        parameters = inspect.signature(runner.predict_dataset).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_value_loader_selects_no_label_column(self):
        source = inspect.getsource(runner.load_values_only).upper()
        self.assertNotIn("SELECT I.LABEL", source)
        self.assertNotIn("LABELS_BLOB", source)

    def test_prediction_rows_have_no_evaluation_metrics(self):
        source = inspect.getsource(core.candidate_rows)
        tree = ast.parse(source)
        forbidden = {"tp", "fp", "fn", "f1", "precision", "recall", "y_test"}
        string_values = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertFalse(forbidden & string_values)

    def test_fixed_sensitivity_grid(self):
        self.assertEqual(core.ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(core.AGREEMENTS, (2, 3))

    def test_paths_remain_fixed(self):
        self.assertEqual(runner.DB_PATH, Path("/Users/minho/Documents/Dataset/univariate_ts.db"))


if __name__ == "__main__":
    unittest.main()
