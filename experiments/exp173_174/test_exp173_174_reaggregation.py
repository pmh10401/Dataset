from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp173_174.independent_reaggregate_exp173_174 import reaggregate
from experiments.exp173_174.run_experiment_173_174_balanced_train_selector import (
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


class ReaggregationTests(unittest.TestCase):
    def make_output(self, root: Path) -> None:
        policy = {
            "selected": {
                "pool_policy": "similarity_pool",
                "alpha": 0.01,
                "source_config": "2of3",
                "policy_key": ["similarity_pool", 0.01, "2of3"],
            }
        }
        policy_path = root / "03_exp173_frozen_policy.json"
        write_json(policy_path, policy)
        predictions = [
            {
                "dataset": "d",
                "n_test": 1,
                "n_train": 10,
                "pool_policy": "similarity_pool",
                "alpha": 0.01,
                "source_config": "2of3",
                "candidate_indices": [0],
                "candidate_count": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        ]
        prediction_path = root / "04_exp174_frozen_predictions.jsonl"
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
            root / "06_exp174_dataset_metrics.csv",
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
            root / "07_exp174_global_summary.csv",
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
                    "mean_candidates_per_dataset": 1.0,
                    "median_candidates_per_dataset": 1.0,
                    "p90_candidates_per_dataset": 1.0,
                    "p95_candidates_per_dataset": 1.0,
                    "max_candidates_per_dataset": 1,
                }
            ],
        )

    def test_independent_reaggregation_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            result = reaggregate(root, write_result=False)
            self.assertEqual(result["mismatch_count"], 0)
            self.assertEqual(result["datasets"], 1)
            self.assertEqual(result["tp"], 1)
            self.assertEqual(result["p95_candidates_per_dataset"], 1.0)

    def test_summary_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_output(root)
            summary = root / "07_exp174_global_summary.csv"
            text = summary.read_text(encoding="utf-8").replace(
                ",1.0,1.0,1\n",
                ",1.0,9.0,1\n",
            )
            summary.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent reaggregation mismatch"):
                reaggregate(root, write_result=False)


if __name__ == "__main__":
    unittest.main()
