from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp171_172.independent_reaggregate_exp171_172 import reaggregate
from experiments.exp171_172.run_experiment_171_172_train_risk_policy import (
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


class ReaggregationTests(unittest.TestCase):
    def make_output(self, root: Path) -> None:
        policy = {
            "selected": {
                "pool_policy": "global_balanced_pool",
                "alpha": 0.02,
                "source_config": "spectral+interval",
                "policy_key": ["global_balanced_pool", 0.02, "spectral+interval"],
            }
        }
        policy_path = root / "03_exp171_frozen_policy.json"
        write_json(policy_path, policy)
        predictions = [
            {
                "dataset": "d",
                "n_test": 1,
                "n_train": 10,
                "pool_policy": "global_balanced_pool",
                "alpha": 0.02,
                "source_config": "spectral+interval",
                "candidate_indices": [0],
                "candidate_count": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        ]
        prediction_path = root / "04_exp172_frozen_predictions.jsonl"
        write_jsonl(prediction_path, predictions)
        write_json(
            root / "05_prediction_manifest.json",
            {
                "policy_file": policy_path.name,
                "policy_sha256": sha256_file(policy_path),
                "prediction_file": prediction_path.name,
                "prediction_sha256": sha256_file(prediction_path),
                "dataset_count": 1,
            },
        )
        write_csv(
            root / "06_exp172_dataset_metrics.csv",
            [
                {
                    "dataset": "d",
                    "review_candidates": 1,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
            ],
        )
        write_csv(
            root / "07_exp172_global_summary.csv",
            [
                {
                    "datasets": 1,
                    "review_candidates": 1,
                    "tp": 1,
                    "fp": 0,
                    "micro_precision": 1.0,
                    "mean_recall": 1.0,
                    "mean_f1": 1.0,
                    "zero_f1_datasets": 0,
                    "candidate_zero_datasets": 0,
                }
            ],
        )

    def test_independent_reaggregation_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            result = reaggregate(root, write_result=False)
            self.assertEqual(result["mismatch_count"], 0)
            self.assertEqual(result["datasets"], 1)
            self.assertEqual(result["tp"], 1)

    def test_summary_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            summary = root / "07_exp172_global_summary.csv"
            summary.write_text(
                "datasets,review_candidates,tp,fp,micro_precision,mean_recall,mean_f1,zero_f1_datasets,candidate_zero_datasets\n"
                "1,1,0,0,0,1,1,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "independent reaggregation mismatch"):
                reaggregate(root, write_result=False)


if __name__ == "__main__":
    unittest.main()
