from __future__ import annotations

import ast
import unittest
from pathlib import Path


RUNNER = Path(__file__).with_name("run_experiment_177_185_unattended.py")


class Exp177185StructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_database_path_is_unchanged(self):
        self.assertIn('/Users/minho/Documents/Dataset', self.source)
        self.assertIn('univariate_ts.db', self.source)

    def test_prediction_payload_declares_no_labels_or_metrics(self):
        self.assertIn('"contains_test_labels": False', self.source)
        self.assertIn('"contains_evaluation_metrics": False', self.source)

    def test_label_loader_is_only_called_by_evaluation(self):
        calls = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_test_labels":
                calls.append(node.lineno)
        evaluate_line = next(
            node.lineno for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_stages"
        )
        stage185_line = next(
            node.lineno for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "stage185"
        )
        self.assertTrue(calls)
        self.assertTrue(all(evaluate_line < line < stage185_line for line in calls))

    def test_no_tail_replacement_or_test_length_budget(self):
        self.assertNotIn("tail_replacement", self.source)
        self.assertNotIn("large_data_budget", self.source)
        self.assertNotIn("is_large_data_case", self.source)

    def test_fixed_grid_is_imported_from_core(self):
        self.assertIn("for alpha in core.ALPHAS", self.source)
        self.assertIn("for target in core.RISK_TARGETS", self.source)


if __name__ == "__main__":
    unittest.main()
