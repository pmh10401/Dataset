from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp266_fn_recovery_shadow.evaluation import (
    aggregate,
    evaluate_predictions,
    lane_metrics,
)
from experiments.exp266_fn_recovery_shadow.independent_reaggregate import (
    verify_summary,
    write_independent_reaggregation,
)
from experiments.exp266_fn_recovery_shadow.inputs import sha256_file
from experiments.exp266_fn_recovery_shadow.reporting import write_reports


def write_frozen_fixture(root: Path) -> tuple[Path, Path]:
    rows = [
        {
            "dataset": "demo",
            "n_train": 8,
            "hard_indices": [0],
            "shadow_indices": [4],
            "priority_indices": [],
            "standard_indices": [],
            "recovery_shadow_indices": [1],
            "recovery_priority_indices": [2],
            "data_quality_review_indices": [3],
            "support_by_index": {"1": 3, "2": 2, "3": 4},
            "source_availability": {
                "kernel": True,
                "statistical": True,
                "structural": True,
                "change_point": True,
                "reconstruction": False,
            },
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "retrospective": True,
        }
    ]
    prediction = root / "03_frozen_predictions.jsonl"
    prediction.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = root / "04_prediction_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prediction_sha256": sha256_file(prediction),
                "retrospective": True,
                "deployment_eligible": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prediction, manifest


def write_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
        CREATE TABLE instances (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            split TEXT NOT NULL,
            instance_index INTEGER NOT NULL,
            label TEXT NOT NULL
        );
        INSERT INTO datasets(id, name) VALUES (1, 'demo');
        INSERT INTO instances(dataset_id, split, instance_index, label) VALUES
            (1, 'TRAIN', 0, '0'),
            (1, 'TEST', 0, '1'),
            (1, 'TEST', 1, '1'),
            (1, 'TEST', 2, '0'),
            (1, 'TEST', 3, '1'),
            (1, 'TEST', 4, '0');
        """
    )
    connection.commit()
    connection.close()


class EvaluationTests(unittest.TestCase):
    def test_evaluation_refuses_tampered_prediction_before_sqlite_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest = write_frozen_fixture(root)
            prediction.write_text(prediction.read_text(encoding="utf-8") + "{}\n")
            with patch(
                "experiments.exp266_fn_recovery_shadow.evaluation.sqlite3.connect"
            ) as connect, self.assertRaisesRegex(ValueError, "SHA256"):
                evaluate_predictions(prediction, manifest, database_path=root / "missing.db")
            connect.assert_not_called()

    def test_point_level_metrics_and_zero_denominators(self) -> None:
        self.assertEqual(
            lane_metrics([], {0: 0, 1: 1}),
            {
                "count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        self.assertEqual(lane_metrics([0], {0: 0, 1: 1})["tp"], 0)
        self.assertEqual(lane_metrics([0], {0: 0, 1: 1})["fp"], 1)

    def test_evaluation_keeps_hard_and_review_lanes_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest = write_frozen_fixture(root)
            database = root / "labels.db"
            write_database(database)

            rows, summary = evaluate_predictions(
                prediction, manifest, database_path=database
            )

            self.assertEqual(rows[0]["hard_alerts"], 1)
            self.assertEqual(rows[0]["hard_tp"], 1)
            self.assertEqual(rows[0]["recovery_shadow_tp"], 1)
            self.assertEqual(rows[0]["recovery_priority_fp"], 1)
            self.assertEqual(rows[0]["data_quality_review_tp"], 1)
            self.assertEqual(summary["hard_alert_delta"], 0)
            self.assertEqual(summary["hard_alerts"], summary["baseline_hard_alerts"])
            self.assertEqual(summary["baseline_shadow_count"], 1)
            self.assertEqual(summary["baseline_priority_count"], 0)
            self.assertEqual(summary["baseline_standard_count"], 0)
            self.assertEqual(summary["baseline_union_count"], 2)
            self.assertEqual(summary["recovery_shadow_count"], 1)
            self.assertEqual(summary["recovered_unique_fn_count"], 1)
            self.assertEqual(summary["human_assisted_review_count"], 3)
            self.assertEqual(summary["review_request_rate"], 3 / 5)
            self.assertTrue(summary["retrospective"])
            self.assertFalse(summary["deployment_eligible"])

            with (root / "07_source_agreement_diagnostics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                diagnostic = next(csv.DictReader(handle))
            self.assertEqual(diagnostic["agreement_0_count"], "0")
            self.assertEqual(diagnostic["agreement_2_count"], "1")
            self.assertEqual(diagnostic["agreement_3_count"], "1")
            self.assertEqual(diagnostic["agreement_4_count"], "1")

    def test_recovery_review_never_changes_hard(self) -> None:
        summary = aggregate(
            [
                {
                    "hard_alerts": 1,
                    "baseline_hard_alerts": 1,
                    "hard_tp": 1,
                    "hard_fp": 0,
                    "recovery_shadow_count": 1,
                    "recovery_shadow_tp": 1,
                    "recovery_shadow_fp": 0,
                }
            ]
        )
        self.assertEqual(summary["hard_alert_delta"], 0)
        self.assertEqual(summary["hard_alerts"], summary["baseline_hard_alerts"])
        self.assertEqual(summary["hard_fn"], 0)
        self.assertEqual(summary["recovery_shadow_count"], 1)

    def test_independent_reaggregation_rejects_post_evaluation_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest_path = write_frozen_fixture(root)
            database = root / "labels.db"
            write_database(database)
            evaluate_predictions(prediction, manifest_path, database_path=database)

            prediction.write_text(prediction.read_text(encoding="utf-8") + "\n")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prediction_sha256"] = sha256_file(prediction)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = write_independent_reaggregation(root)
            self.assertFalse(result["passed"])
            self.assertFalse(result["evaluation_prediction_sha256_matches"])

    def test_independent_reaggregation_and_reports_detect_summary_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest = write_frozen_fixture(root)
            database = root / "labels.db"
            write_database(database)
            evaluate_predictions(prediction, manifest, database_path=database)

            result = verify_summary(
                root / "05_dataset_results.csv", root / "06_global_summary.csv"
            )
            self.assertTrue(result["passed"])
            outputs = write_reports(root)
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "08_invariance_report.md",
                    "09_result_report.md",
                    "10_independent_reaggregation.json",
                    "MANIFEST.csv",
                },
            )
            reaggregation = json.loads(
                (root / "10_independent_reaggregation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(reaggregation["passed"])
            self.assertEqual(
                reaggregation["baseline_lane_totals"],
                {"hard": 1, "priority": 0, "shadow": 1, "standard": 0, "union": 2},
            )
            self.assertEqual(reaggregation["prediction_label_fields"], 0)
            self.assertEqual(reaggregation["prediction_evaluation_fields"], 0)
            self.assertEqual(reaggregation["missing_datasets"], 0)
            self.assertEqual(reaggregation["index_errors"], 0)
            self.assertEqual(reaggregation["lane_overlaps"], 0)

            summary_path = root / "06_global_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            row["hard_alerts"] = "2"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            result = verify_summary(root / "05_dataset_results.csv", summary_path)
            self.assertFalse(result["passed"])
            self.assertIn("hard_alerts", result["mismatches"])


if __name__ == "__main__":
    unittest.main()
