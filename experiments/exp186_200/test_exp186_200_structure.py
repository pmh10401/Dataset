from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from experiments.exp186_200.core import validate_prediction_payload
from experiments.exp186_200.artifacts import load_verified_predictions

# We will import run_experiment_186_200 inside tests once we create it.
RUNNER_PATH = Path(__file__).parent / "run_experiment_186_200.py"


class Exp186200StructureTest(unittest.TestCase):
    def test_prediction_payload_rejects_evaluation_fields(self):
        with self.assertRaises(ValueError):
            validate_prediction_payload({
                "dataset": "x",
                "n_test": 10,
                "candidate_indices": [],
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "hard_tp": 0
            })

    def test_predict_signature_has_no_y_test(self):
        # Once run_experiment_186_200 is implemented:
        from experiments.exp186_200.run_experiment_186_200 import predict_dataset
        sig = inspect.signature(predict_dataset)
        self.assertNotIn("y_test", sig.parameters)

    def test_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir)
            jsonl = dirpath / "exp_predictions.jsonl"
            jsonl.write_text('{"dataset": "x", "variant": "v", "n_test": 10, "candidate_indices": [], "contains_test_labels": false, "contains_evaluation_metrics": false}\n', encoding="utf-8")
            
            manifest = dirpath / "exp_manifest.json"
            manifest.write_text('{"prediction_sha256": "wrong_hash", "prediction_file": "exp_predictions.jsonl"}', encoding="utf-8")
            
            with self.assertRaises(ValueError):
                load_verified_predictions(jsonl, manifest)

    def test_stage_completion_happens_after_evaluation(self):
        from experiments.exp186_200 import run_experiment_186_200 as runner

        calls = []
        with mock.patch.object(runner, "run_prediction_stage", side_effect=lambda *a, **k: calls.append("predict")), \
             mock.patch.object(runner, "evaluate_stage", side_effect=lambda *a, **k: calls.append("evaluate")), \
             mock.patch.object(runner, "write_completion", side_effect=lambda *a, **k: calls.append("complete")), \
             mock.patch.object(runner, "run_reaggregation", side_effect=lambda *a, **k: calls.append("reaggregate")):
            runner.run_stage("186")
        self.assertEqual(calls, ["predict", "evaluate", "complete"])

    def test_stage_200_runs_reaggregation_before_completion(self):
        from experiments.exp186_200 import run_experiment_186_200 as runner

        calls = []
        with mock.patch.object(runner, "run_prediction_stage", side_effect=lambda *a, **k: calls.append("predict")), \
             mock.patch.object(runner, "evaluate_stage", side_effect=lambda *a, **k: calls.append("evaluate")), \
             mock.patch.object(runner, "write_completion", side_effect=lambda *a, **k: calls.append("complete")), \
             mock.patch.object(runner, "run_reaggregation", side_effect=lambda *a, **k: calls.append("reaggregate")):
            runner.run_stage("200")
        self.assertEqual(calls, ["predict", "evaluate", "reaggregate", "complete"])

    def test_database_path_is_correct(self):
        source = RUNNER_PATH.read_text(encoding="utf-8") if RUNNER_PATH.exists() else ""
        if source:
            self.assertIn('/Users/minho/Documents/Dataset', source)
            self.assertIn('univariate_ts.db', source)

    def test_prediction_payload_declares_no_labels_or_metrics(self):
        source = RUNNER_PATH.read_text(encoding="utf-8") if RUNNER_PATH.exists() else ""
        if source:
            self.assertIn('"contains_test_labels": False', source)
            self.assertIn('"contains_evaluation_metrics": False', source)

    def test_label_loader_is_only_called_by_evaluation(self):
        if not RUNNER_PATH.exists():
            return
        source = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_test_labels_for_evaluation":
                calls.append(node.lineno)
        
        # Check that label loader is only called within functions that evaluate stages
        eval_defs = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and ("eval" in node.name or "reaggregate" in node.name)
        ]
        
        for call_line in calls:
            # The call should be within the range of one of the evaluation function definitions
            inside_eval = False
            for def_node in eval_defs:
                # Find end line of definition
                end_line = def_node.body[-1].lineno
                if hasattr(def_node, "end_lineno"):
                    end_line = def_node.end_lineno
                if def_node.lineno <= call_line <= end_line:
                    inside_eval = True
                    break
            self.assertTrue(inside_eval, f"Label loader called at line {call_line} outside evaluation/reaggregation functions")


if __name__ == "__main__":
    unittest.main()
