from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.exp290_train_raw_exact_normal_guard.contracts import (
    freeze_prediction_rows,
    verify_prediction_rows,
)
from experiments.exp290_train_raw_exact_normal_guard.evaluation import evaluate_rows


def _row(dataset: str, baseline: list[int], guarded: list[int], removed: list[int]) -> dict[str, object]:
    return {
        "experiment_id": "experiment_290_train_raw_exact_normal_guard",
        "dataset": dataset,
        "n_test": 3,
        "baseline_hard_indices": baseline,
        "guarded_hard_indices": guarded,
        "train_raw_exact_normal_indices": removed,
        "exact_train_indices_by_candidate": {str(index): [0] for index in removed},
        "match_definition": "sqlite_values_blob_byte_exact",
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective_counterfactual": True,
        "upstream_end_to_end_strict_train_only": False,
    }


class FrozenPredictionTest(unittest.TestCase):
    def test_hash_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions.jsonl"
            manifest = root / "manifest.json"
            freeze_prediction_rows([_row("a", [0], [0], [])], prediction, manifest)
            self.assertEqual(len(verify_prediction_rows(prediction, manifest)), 1)
            prediction.write_text(prediction.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_prediction_rows(prediction, manifest)

    def test_evaluation_separates_baseline_guarded_and_removed(self) -> None:
        rows = [
            _row("a", [0, 1], [1], [0]),
            _row("b", [0], [0], []),
        ]
        labels = {
            "a": np.asarray([0, 1, 0], dtype=np.int8),
            "b": np.asarray([1, 0, 0], dtype=np.int8),
        }
        evaluated, summary = evaluate_rows(rows, labels)
        self.assertEqual(summary["baseline_hard_count"], 3)
        self.assertEqual(summary["baseline_hard_tp"], 2)
        self.assertEqual(summary["baseline_hard_fp"], 1)
        self.assertEqual(summary["guarded_hard_count"], 2)
        self.assertEqual(summary["guarded_hard_tp"], 2)
        self.assertEqual(summary["guarded_hard_fp"], 0)
        self.assertEqual(summary["train_raw_exact_normal_count"], 1)
        self.assertEqual(summary["train_raw_exact_normal_tp"], 0)
        self.assertEqual(summary["train_raw_exact_normal_fp"], 1)
        self.assertEqual(evaluated[0]["guarded_hard_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
