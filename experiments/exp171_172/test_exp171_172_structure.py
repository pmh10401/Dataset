from __future__ import annotations

import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp171_172 import policy_core
from experiments.exp171_172 import run_experiment_171_172_train_risk_policy as runner


class StructureTests(unittest.TestCase):
    def test_frozen_policy_is_byte_deterministic(self):
        class ProgressStub:
            def __init__(self, **_kwargs):
                pass

            def start(self, **_kwargs):
                pass

            def finish(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with patch.object(runner, "ExperimentProgress", ProgressStub):
                runner.run_audit(Path(first))
                runner.run_audit(Path(second))
            self.assertEqual(
                (Path(first) / "03_exp171_frozen_policy.json").read_bytes(),
                (Path(second) / "03_exp171_frozen_policy.json").read_bytes(),
            )

    def test_policy_core_does_not_reference_exp170_metrics(self):
        source = inspect.getsource(policy_core)
        self.assertNotIn("08_exp170_dataset_metrics", source)
        self.assertNotIn("09_exp170_global_summary", source)
        self.assertNotIn("y_test", source)

    def test_prediction_function_has_no_label_input_or_label_loader(self):
        signature = inspect.signature(runner.run_prediction)
        self.assertNotIn("y_test", signature.parameters)
        tree = ast.parse(inspect.getsource(runner.run_prediction))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn("load_test_labels", names | attrs)

    def test_prediction_row_contract_rejects_evaluation_fields(self):
        clean = {
            "dataset": "d",
            "n_test": 2,
            "candidate_indices": [1],
            "candidate_count": 1,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        runner.assert_label_free_selected_prediction(clean)
        for forbidden in ("y_test", "tp", "fp", "precision", "recall", "f1"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    runner.assert_label_free_selected_prediction({**clean, forbidden: 0})

    def test_manifest_detects_prediction_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            policy_path = output / "03_exp171_frozen_policy.json"
            prediction_path = output / "04_exp172_frozen_predictions.jsonl"
            policy_path.write_text('{"selected": {}}\n', encoding="utf-8")
            prediction_path.write_text('{"dataset": "d"}\n', encoding="utf-8")
            manifest = {
                "policy_file": policy_path.name,
                "policy_sha256": runner.sha256_file(policy_path),
                "prediction_file": prediction_path.name,
                "prediction_sha256": runner.sha256_file(prediction_path),
                "dataset_count": 1,
            }
            runner.write_json(output / "05_prediction_manifest.json", manifest)
            runner.verify_prediction_manifest(output)
            prediction_path.write_text('{"dataset": "changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction hash"):
                runner.verify_prediction_manifest(output)


if __name__ == "__main__":
    unittest.main()
