from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PackageStructureTests(unittest.TestCase):

    def test_expected_modules_exist(self):
        expected = {
            "__init__.py",
            "candidate_zero_core.py",
            "run_experiment_160_candidate_zero_similarity_recovery.py",
            "test_exp160_candidate_zero_recovery.py",
            "test_exp160_structure.py",
        }
        self.assertTrue(expected <= {path.name for path in HERE.glob("*.py")})

    def test_prediction_functions_do_not_accept_y_test(self):
        # Parse run_experiment_160_candidate_zero_similarity_recovery.py to ensure predict functions do not receive y_test
        tree = ast.parse((HERE / "run_experiment_160_candidate_zero_similarity_recovery.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and "predict" in node.name:
                names = [arg.arg for arg in node.args.args]
                self.assertNotIn("y_test", names, msg=f"run_experiment_160_candidate_zero_similarity_recovery.py:{node.name}")

    def test_experiment_ids_are_correct(self):
        expected = {
            "experiment_160_candidate_zero_similarity_recovery",
        }
        source = (HERE / "run_experiment_160_candidate_zero_similarity_recovery.py").read_text(encoding="utf-8")
        for exp_id in expected:
            self.assertIn(exp_id, source)

    def test_no_test_labels_or_metrics_in_predictions_source(self):
        source = (HERE / "run_experiment_160_candidate_zero_similarity_recovery.py").read_text(encoding="utf-8")
        self.assertIn("prediction_policy_metadata(", source)
        self.assertNotIn('"y_test"', source)
        self.assertNotIn('"hard_tp"', source)
        self.assertNotIn('"hard_fp"', source)
        self.assertNotIn('"combined_f1"', source)

    def test_pooling_is_not_recomputed_inside_alpha_loop(self):
        tree = ast.parse(
            (HERE / "run_experiment_160_candidate_zero_similarity_recovery.py").read_text(
                encoding="utf-8"
            )
        )
        alpha_loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Attribute)
            and node.iter.attr == "ALPHAS"
        ]
        self.assertEqual(len(alpha_loops), 1)
        calls_in_alpha_loop = {
            node.func.attr
            for node in ast.walk(alpha_loops[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("select_donors_by_source_counts", calls_in_alpha_loop)
        self.assertNotIn("empirical_conformal_p_values", calls_in_alpha_loop)


if __name__ == "__main__":
    unittest.main()
