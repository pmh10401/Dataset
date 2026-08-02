import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp422_test_eval.config import Exp422TestEvalConfig
from experiments.exp422_test_eval.inputs import (
    freeze_prediction_inputs,
    load_frozen_test_labels,
)


def _config(root: Path) -> Exp422TestEvalConfig:
    return Exp422TestEvalConfig(
        expected_datasets=2,
        expected_label_rows=4,
        database_sha256="d" * 64,
    )


def _write_prediction(root: Path, *, labelled: bool = False) -> tuple[Path, Path, Exp422TestEvalConfig]:
    prediction = root / "prediction.jsonl"
    row = {"dataset": "A", "instance_index": 0}
    if labelled:
        row["contains_test_labels"] = True
    prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "experiment_id": "Exp422-TEST-EVAL",
                "source_experiment_id": "Exp422",
                "dataset_scope": ["A", "B"],
                "prediction_coordinate_count": 1,
                "prediction_dataset_count": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "test_informed_reselection": False,
                "operational_policy_changed": False,
                "candidate_semantics": "predicted_positive",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prediction, manifest, _config(root)


class InputBoundaryTests(unittest.TestCase):
    def test_prediction_is_frozen_before_labels_are_opened(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            prediction, manifest, config = _write_prediction(root)
            frozen = freeze_prediction_inputs(prediction, manifest, config=config)
            labels = root / "labels.csv"
            labels.write_text(
                "dataset,candidate_index,is_true_anomaly\n"
                "A,0,1\nA,1,0\nB,0,0\nB,1,0\n",
                encoding="utf-8",
            )
            config = Exp422TestEvalConfig(
                expected_datasets=2,
                expected_label_rows=4,
                database_sha256="d" * 64,
                labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
            )
            loaded = load_frozen_test_labels(labels, frozen, config=config)
            self.assertEqual(len(loaded.rows), 4)

    def test_label_loader_rejects_invalid_prediction_without_opening_labels(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.csv"
            labels.write_text("not opened", encoding="utf-8")
            with patch("experiments.exp422_test_eval.inputs.sha256_file", side_effect=AssertionError("label opened")):
                with self.assertRaises(ValueError):
                    load_frozen_test_labels(labels, object())  # type: ignore[arg-type]

    def test_prediction_rows_cannot_contain_test_labels(self) -> None:
        with TemporaryDirectory() as temp:
            prediction, manifest, config = _write_prediction(Path(temp), labelled=True)
            with self.assertRaisesRegex(ValueError, "label"):
                freeze_prediction_inputs(prediction, manifest, config=config)


if __name__ == "__main__":
    unittest.main()
