from __future__ import annotations

import csv
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp263_264_full_missing_aware_integration.inputs import (
    EXPECTED_POPULATION,
    load_source_maps,
)
from experiments.exp263_264_full_missing_aware_integration.prediction import (
    _json_ready,
    predict_all,
)


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PredictionEvaluationBoundaryTest(unittest.TestCase):
    def test_prediction_signature_excludes_test_labels(self) -> None:
        parameters = inspect.signature(predict_all).parameters

        self.assertNotIn("y_test", parameters)
        self.assertFalse({"labels", "test_labels"} & set(parameters))
        source = inspect.getsource(inspect.getmodule(predict_all))
        self.assertNotIn("load_test_labels_for_evaluation", source)

    def test_json_ready_converts_numpy_values(self) -> None:
        value = {
            "scalar": np.int64(3),
            "array": np.asarray([np.int64(1), np.float64(2.5)]),
        }

        self.assertEqual(_json_ready(value), {"scalar": 3, "array": [1, 2.5]})

    def test_predict_all_freezes_complete_deterministic_label_free_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            first = predict_all(output_root)
            first_bytes = first.read_bytes()
            second = predict_all(output_root)
            second_bytes = second.read_bytes()
            rows = _read_rows(second)
            manifest = json.loads(
                (output_root / "prediction_manifest.json").read_text(encoding="utf-8")
            )
            with (output_root / "source_agreement_diagnostics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                diagnostics = list(csv.DictReader(handle))

        names = [str(row["dataset"]) for row in rows]
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(len(rows), EXPECTED_POPULATION)
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(set(names)), EXPECTED_POPULATION)
        self.assertEqual(manifest["rows"], EXPECTED_POPULATION)
        self.assertEqual(manifest["datasets"], EXPECTED_POPULATION)
        self.assertEqual(manifest["missing_override_datasets"], 20)
        self.assertEqual(manifest["prediction_sha256"], hashlib.sha256(first_bytes).hexdigest())
        self.assertFalse(manifest["contains_test_labels"])
        self.assertFalse(manifest["contains_evaluation_metrics"])
        self.assertEqual(len(diagnostics), EXPECTED_POPULATION)
        self.assertEqual([row["dataset"] for row in diagnostics], names)

        for row in rows:
            lanes = [
                set(row[key])
                for key in (
                    "final_hard_indices",
                    "final_standard_indices",
                    "final_priority_indices",
                )
            ]
            self.assertTrue(
                all(
                    not lanes[left] & lanes[right]
                    for left in range(len(lanes))
                    for right in range(left + 1, len(lanes))
                )
            )
            self.assertTrue(
                set(row["baseline_hard_indices"]) <= set(row["final_hard_indices"])
            )
            self.assertTrue(
                all(0 <= index < row["n_test"] for lane in lanes for index in lane)
            )

    def test_missing_rows_reproduce_exp260_routing_and_sources_exactly(self) -> None:
        sources = load_source_maps()
        with tempfile.TemporaryDirectory() as directory:
            rows = {
                str(row["dataset"]): row
                for row in _read_rows(predict_all(Path(directory)))
            }

        self.assertEqual(len(sources.missing_override_rows), 20)
        for dataset, override in sources.missing_override_rows.items():
            with self.subTest(dataset=dataset):
                row = rows[dataset]
                self.assertTrue(row["missing_override"])
                self.assertEqual(row["source_indices"], override["source_indices"])
                self.assertEqual(row["quality_flag_indices"], override["quality_flag_indices"])
                for field in (
                    "n_train",
                    "n_test",
                    "baseline_hard_indices",
                    "promoted_indices",
                    "quality_blocked_indices",
                    "final_hard_indices",
                    "final_standard_indices",
                    "final_priority_indices",
                ):
                    self.assertEqual(row[field], override[field])


if __name__ == "__main__":
    unittest.main()
