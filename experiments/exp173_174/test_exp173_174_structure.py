from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp173_174 import run_experiment_173_174_balanced_train_selector as runner


class RunnerStructureTests(unittest.TestCase):
    def test_contract_freezes_balance_rule(self) -> None:
        config = runner.frozen_contract()["config"]
        self.assertEqual(config["risk_target"], 0.01)
        self.assertEqual(config["source_agreement"], "2of3")
        self.assertEqual(config["fold_count"], 5)
        self.assertFalse(config["test_metric_selection"])
        self.assertFalse(config["family_routing"])
        self.assertFalse(config["test_length_budget"])
        self.assertFalse(config["autonomous_hard_alert"])

    def test_prediction_path_does_not_load_test_labels(self) -> None:
        source = inspect.getsource(runner.run_prediction)
        self.assertNotIn("_load_test_labels", source)
        self.assertNotIn("SOURCE_TEST_METRICS", source)
        self.assertNotIn("y_test", inspect.signature(runner.run_prediction).parameters)

    def test_label_free_prediction_rejects_evaluation_fields(self) -> None:
        row = {
            "dataset": "demo",
            "n_test": 3,
            "candidate_indices": [1],
            "candidate_count": 1,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "f1": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "forbidden"):
            runner.assert_label_free_selected_prediction(row)

    def test_prediction_manifest_rejects_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            policy_path = output_dir / "03_exp173_frozen_policy.json"
            prediction_path = output_dir / "04_exp174_frozen_predictions.jsonl"
            policy_path.write_text("{}\n", encoding="utf-8")
            prediction_path.write_text("{}\n", encoding="utf-8")
            manifest = {
                "policy_file": policy_path.name,
                "policy_sha256": runner.sha256_file(policy_path),
                "prediction_file": prediction_path.name,
                "prediction_sha256": runner.sha256_file(prediction_path),
            }
            (output_dir / "05_prediction_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            prediction_path.write_text('{"tampered": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                runner.verify_prediction_manifest(output_dir)


if __name__ == "__main__":
    unittest.main()
