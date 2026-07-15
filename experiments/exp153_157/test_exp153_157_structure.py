from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PackageStructureTests(unittest.TestCase):
    def test_expected_modules_exist(self):
        expected = {
            "exp153_157_bridge.py",
            "exp153_157_reporting.py",
            "build_exp153_157_score_cache.py",
            "run_experiment_153_train_normal_calibration_audit.py",
            "run_experiment_154_pooled_conformal.py",
            "run_experiment_155_source_fusion.py",
            "run_experiment_156_confirmation_calibration.py",
            "run_experiment_157_risk_controlled_alpha_policy.py",
            "run_experiment_153_157_pipeline.py",
        }
        self.assertTrue(expected <= {path.name for path in HERE.glob("*.py")})

    def test_bridge_keeps_local_database_path(self):
        source = (HERE / "exp153_157_bridge.py").read_text(encoding="utf-8")
        self.assertIn('DATA_DIR = Path("/Users/minho/Documents/Dataset")', source)
        self.assertIn('DB_PATH = DATA_DIR / "univariate_ts.db"', source)

    def test_prediction_functions_do_not_accept_y_test(self):
        modules = [
            "run_experiment_154_pooled_conformal.py",
            "run_experiment_155_source_fusion.py",
            "run_experiment_156_confirmation_calibration.py",
            "run_experiment_157_risk_controlled_alpha_policy.py",
        ]
        for module in modules:
            tree = ast.parse((HERE / module).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("predict"):
                    names = [arg.arg for arg in node.args.args]
                    self.assertNotIn("y_test", names, msg=f"{module}:{node.name}")

    def test_experiment_ids_are_new_and_distinct(self):
        expected = {
            "experiment_153_train_normal_calibration_validity_audit",
            "experiment_154_small_sample_pooled_conformal",
            "experiment_155_source_independence_continuous_fusion",
            "experiment_156_confirmation_calibration",
            "experiment_157_train_only_risk_controlled_alpha_policy",
        }
        combined = "\n".join(path.read_text(encoding="utf-8") for path in HERE.glob("run_experiment_*.py"))
        for experiment_id in expected:
            self.assertIn(experiment_id, combined)

    def test_evaluation_is_separate_from_prediction(self):
        source = (HERE / "exp153_157_reporting.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("freeze_prediction_file", function_names)
        self.assertIn("evaluate_prediction_file", function_names)


if __name__ == "__main__":
    unittest.main()
