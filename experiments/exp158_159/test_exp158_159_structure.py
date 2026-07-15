from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PackageStructureTests(unittest.TestCase):

    def test_expected_modules_exist(self):
        expected = {
            "__init__.py",
            "similarity_pool_core.py",
            "run_experiment_158_159_similarity_pooling.py",
            "test_exp158_159_similarity_pooling.py",
            "test_exp158_159_structure.py",
        }
        self.assertTrue(expected <= {path.name for path in HERE.glob("*.py")})

    def test_prediction_functions_do_not_accept_y_test(self):
        # Parse run_experiment_158_159_similarity_pooling.py to ensure predict functions do not receive y_test
        tree = ast.parse((HERE / "run_experiment_158_159_similarity_pooling.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and "predict" in node.name:
                names = [arg.arg for arg in node.args.args]
                self.assertNotIn("y_test", names, msg=f"run_experiment_158_159_similarity_pooling.py:{node.name}")

    def test_experiment_ids_are_correct(self):
        expected = {
            "experiment_158_train_score_similarity_pooling",
            "experiment_159_similarity_pool_review_integration",
        }
        source = (HERE / "run_experiment_158_159_similarity_pooling.py").read_text(encoding="utf-8")
        for exp_id in expected:
            self.assertIn(exp_id, source)

    def test_no_test_labels_or_metrics_in_predictions_source(self):
        source = (HERE / "run_experiment_158_159_similarity_pooling.py").read_text(encoding="utf-8")
        self.assertIn('"contains_test_labels": False', source)
        self.assertIn('"contains_evaluation_metrics": False', source)


if __name__ == "__main__":
    unittest.main()
