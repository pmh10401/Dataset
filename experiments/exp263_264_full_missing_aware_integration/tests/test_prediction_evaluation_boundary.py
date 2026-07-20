from __future__ import annotations

import csv
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_evaluation_rejects_prediction_hash_mismatch_before_loading_labels(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.evaluation import (
            evaluate_all,
        )

        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "frozen_predictions.jsonl"
            prediction.write_text('{"dataset":"x"}\n', encoding="utf-8")
            with patch(
                "experiments.exp226_235_hard_alert_quality.database.load_test_labels_for_evaluation"
            ) as labels:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    evaluate_all(prediction, expected_sha256="wrong")

            labels.assert_not_called()

    def test_evaluation_reports_lanes_metrics_baseline_and_diagnostic_groups(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.evaluation import (
            evaluate_all,
        )

        rows = [
            {
                "dataset": "small_nonmissing",
                "n_train": 99,
                "n_test": 3,
                "missing_override": False,
                "baseline_hard_indices": [0],
                "baseline_standard_indices": [1, 2],
                "baseline_priority_indices": [],
                "final_hard_indices": [0],
                "final_standard_indices": [1],
                "final_priority_indices": [2],
                "promoted_indices": [],
            },
            {
                "dataset": "large_missing",
                "n_train": 100,
                "n_test": 3,
                "missing_override": True,
                "baseline_hard_indices": [],
                "baseline_standard_indices": [1],
                "baseline_priority_indices": [],
                "final_hard_indices": [1],
                "final_standard_indices": [],
                "final_priority_indices": [],
                "promoted_indices": [1],
            },
        ]
        labels = {
            "small_nonmissing": np.asarray([1, 0, 1], dtype=np.int8),
            "large_missing": np.asarray([0, 1, 0], dtype=np.int8),
        }
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()

        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "frozen_predictions.jsonl"
            prediction.write_bytes(payload)
            with patch(
                "experiments.exp226_235_hard_alert_quality.database.load_test_labels_for_evaluation",
                side_effect=lambda dataset: labels[dataset],
            ):
                evaluated, summary = evaluate_all(
                    prediction,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_population=("small_nonmissing", "large_missing"),
                    expected_total_anomalies=3,
                )

            self.assertEqual(len(evaluated), 2)
            self.assertEqual(summary["prediction_sha256"], hashlib.sha256(payload).hexdigest())
            autonomous_hard = summary["autonomous_hard"]
            self.assertEqual({
                "datasets": 2,
                "alerts": 2,
                "tp": 2,
                "fp": 0,
                "fn": 1,
                "total_anomalies": 3,
                "micro_precision": 1.0,
                "micro_recall": 2 / 3,
                "zero_f1_datasets": 0,
            }, {key: value for key, value in autonomous_hard.items() if key != "macro_mean_f1"})
            self.assertAlmostEqual(autonomous_hard["macro_mean_f1"], 5 / 6)
            self.assertEqual(summary["standard_review"]["alerts"], 1)
            self.assertEqual(summary["standard_review"]["tp"], 0)
            self.assertEqual(summary["priority_review"]["tp"], 1)
            self.assertEqual(summary["review"]["alerts"], 2)
            self.assertEqual(summary["paired_baseline_comparison"]["datasets"], 2)
            self.assertEqual(
                summary["paired_baseline_comparison"]["macro_mean_f1_delta"], 0.5
            )
            self.assertEqual(summary["diagnostic_subgroups"]["missing_override"]["datasets"], 1)
            self.assertEqual(summary["diagnostic_subgroups"]["nonmissing"]["datasets"], 1)
            self.assertEqual(summary["diagnostic_subgroups"]["n_train_lt_100"]["datasets"], 1)
            self.assertEqual(summary["diagnostic_subgroups"]["n_train_ge_100"]["datasets"], 1)
            self.assertTrue(
                all(
                    subgroup["used_for_policy_selection"] is False
                    for subgroup in summary["diagnostic_subgroups"].values()
                )
            )
            self.assertTrue((prediction.parent / "results.csv").is_file())
            self.assertTrue((prediction.parent / "summary.json").is_file())
            self.assertTrue((prediction.parent / "lane_transitions.csv").is_file())
            with (prediction.parent / "lane_transitions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                transitions = {row["dataset"]: row for row in csv.DictReader(handle)}
            self.assertEqual(transitions["small_nonmissing"]["baseline_standard_count"], "2")
            self.assertEqual(transitions["small_nonmissing"]["standard_to_priority_additions"], "1")
            self.assertEqual(transitions["large_missing"]["standard_to_hard_additions"], "1")
            self.assertEqual(summary["lane_transitions"]["standard_to_hard_additions"], 1)
            self.assertEqual(summary["lane_transitions"]["standard_to_priority_additions"], 1)

    def test_evaluation_reads_prediction_once_and_parses_verified_payload(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.evaluation import (
            evaluate_all,
        )

        row = {
            "dataset": "only",
            "n_train": 10,
            "n_test": 1,
            "missing_override": False,
            "baseline_hard_indices": [],
            "baseline_standard_indices": [],
            "baseline_priority_indices": [],
            "final_hard_indices": [],
            "final_standard_indices": [],
            "final_priority_indices": [],
            "promoted_indices": [],
        }
        payload = (json.dumps(row) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "frozen_predictions.jsonl"
            prediction.write_bytes(payload)
            with (
                patch.object(Path, "read_bytes", wraps=prediction.read_bytes) as read_bytes,
                patch.object(Path, "read_text", side_effect=AssertionError("must parse bytes")),
                patch(
                    "experiments.exp226_235_hard_alert_quality.database.load_test_labels_for_evaluation",
                    return_value=np.asarray([0], dtype=np.int8),
                ),
            ):
                evaluate_all(
                    prediction,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_population=("only",),
                    expected_total_anomalies=0,
                )
        self.assertEqual(read_bytes.call_count, 1)

    def test_evaluation_rejects_partial_population_before_outputs(self) -> None:
        row = {
            "dataset": "one", "n_train": 10, "n_test": 1, "missing_override": False,
            "baseline_hard_indices": [], "baseline_standard_indices": [],
            "baseline_priority_indices": [], "final_hard_indices": [],
            "final_standard_indices": [], "final_priority_indices": [], "promoted_indices": [],
        }
        from experiments.exp263_264_full_missing_aware_integration.evaluation import evaluate_all
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "frozen_predictions.jsonl"
            payload = (json.dumps(row) + "\n").encode()
            prediction.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "population"):
                evaluate_all(
                    prediction, expected_sha256=hashlib.sha256(payload).hexdigest(),
                    expected_population=("one", "two"), expected_total_anomalies=0,
                )
            self.assertFalse((prediction.parent / "results.csv").exists())

    def test_evaluation_rejects_duplicate_dataset_names(self) -> None:
        rows = [
            {
                "dataset": "one", "n_train": 10, "n_test": 1, "missing_override": False,
                "baseline_hard_indices": [], "baseline_standard_indices": [],
                "baseline_priority_indices": [], "final_hard_indices": [],
                "final_standard_indices": [], "final_priority_indices": [], "promoted_indices": [],
            },
        ] * 2
        self._assert_fixture_rejected(rows, ("one", "two"), {}, 0, "duplicate")

    def test_evaluation_rejects_unexpected_dataset_name(self) -> None:
        row = {
            "dataset": "unexpected", "n_train": 10, "n_test": 1, "missing_override": False,
            "baseline_hard_indices": [], "baseline_standard_indices": [],
            "baseline_priority_indices": [], "final_hard_indices": [],
            "final_standard_indices": [], "final_priority_indices": [], "promoted_indices": [],
        }
        self._assert_fixture_rejected([row], ("expected",), {}, 0, "names")

    def test_evaluation_rejects_test_label_length_mismatch(self) -> None:
        row = {
            "dataset": "one", "n_train": 10, "n_test": 2, "missing_override": False,
            "baseline_hard_indices": [], "baseline_standard_indices": [],
            "baseline_priority_indices": [], "final_hard_indices": [],
            "final_standard_indices": [], "final_priority_indices": [], "promoted_indices": [],
        }
        self._assert_fixture_rejected(
            [row], ("one",), {"one": np.asarray([0], dtype=np.int8)}, 0, "length"
        )

    def test_evaluation_rejects_total_anomaly_mismatch(self) -> None:
        row = {
            "dataset": "one", "n_train": 10, "n_test": 1, "missing_override": False,
            "baseline_hard_indices": [], "baseline_standard_indices": [],
            "baseline_priority_indices": [], "final_hard_indices": [],
            "final_standard_indices": [], "final_priority_indices": [], "promoted_indices": [],
        }
        self._assert_fixture_rejected(
            [row], ("one",), {"one": np.asarray([1], dtype=np.int8)}, 2, "anomaly"
        )

    def test_evaluation_rejects_overlapping_final_lanes_before_metrics(self) -> None:
        row = {
            "dataset": "one", "n_train": 10, "n_test": 1, "missing_override": False,
            "baseline_hard_indices": [], "baseline_standard_indices": [0],
            "baseline_priority_indices": [], "final_hard_indices": [0],
            "final_standard_indices": [0], "final_priority_indices": [], "promoted_indices": [0],
        }
        self._assert_fixture_rejected(
            [row], ("one",), {"one": np.asarray([1], dtype=np.int8)}, 1, "overlap"
        )

    def _assert_fixture_rejected(
        self,
        rows: list[dict[str, object]],
        expected_population: tuple[str, ...],
        labels: dict[str, np.ndarray],
        expected_total_anomalies: int,
        message: str,
    ) -> None:
        from experiments.exp263_264_full_missing_aware_integration.evaluation import evaluate_all

        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        with tempfile.TemporaryDirectory() as directory:
            prediction = Path(directory) / "frozen_predictions.jsonl"
            prediction.write_bytes(payload)
            with patch(
                "experiments.exp226_235_hard_alert_quality.database.load_test_labels_for_evaluation",
                side_effect=lambda dataset: labels[dataset],
            ):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_all(
                        prediction,
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        expected_population=expected_population,
                        expected_total_anomalies=expected_total_anomalies,
                    )
            self.assertFalse((prediction.parent / "results.csv").exists())

    def test_evaluation_metrics_define_zero_denominators_as_zero(self) -> None:
        from experiments.exp263_264_full_missing_aware_integration.evaluation import metrics

        self.assertEqual(
            metrics(set(), set()),
            {
                "alerts": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision_denominator": 0,
                "recall_denominator": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )

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
