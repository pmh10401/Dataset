from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/exp165/run_experiment_165_small_sample_calibration.py"


class Exp165StructureTests(unittest.TestCase):
    def test_runner_exists_and_has_no_forbidden_test_access(self):
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = [
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        forbidden = ("y_test", "test label", "anomaly_position", "labels_blob", "test length")
        for value in constants:
            self.assertFalse(any(token in value for token in forbidden), value)

    def test_runner_records_train_only_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"contains_test_values": False', source)
        self.assertIn('"contains_test_labels": False', source)
        self.assertIn('"contains_evaluation_metrics": False', source)
        self.assertIn("MIN_POOL_REFERENCE = 200", source)
        self.assertIn("ALPHAS = (0.005, 0.01, 0.02, 0.05)", source)


if __name__ == "__main__":
    unittest.main()
