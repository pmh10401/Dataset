from __future__ import annotations

import unittest
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.exp290_train_raw_exact_normal_guard.prediction import (
    build_prediction_rows,
    load_blob_bundle_for_prediction,
    partition_hard_indices,
    validate_prediction_row,
)


class ExactNormalGuardTest(unittest.TestCase):
    def test_exact_matches_are_removed_from_hard_as_normal(self) -> None:
        result = partition_hard_indices({1, 2, 3}, {2, 4})
        self.assertEqual(result.guarded_hard_indices, (1, 3))
        self.assertEqual(result.train_raw_exact_normal_indices, (2,))

    def test_partition_preserves_and_separates_baseline_hard(self) -> None:
        result = partition_hard_indices({5, 1}, {1})
        self.assertFalse(set(result.guarded_hard_indices) & set(result.train_raw_exact_normal_indices))
        self.assertEqual(
            set(result.guarded_hard_indices) | set(result.train_raw_exact_normal_indices),
            {1, 5},
        )

    def test_prediction_row_rejects_label_and_metric_fields(self) -> None:
        valid = {
            "dataset": "demo",
            "n_test": 3,
            "baseline_hard_indices": [1],
            "guarded_hard_indices": [],
            "train_raw_exact_normal_indices": [1],
            "exact_train_indices_by_candidate": {"1": [0]},
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        validate_prediction_row(valid)
        for forbidden in ("y_test", "hard_tp", "hard_fp", "hard_f1", "true_label"):
            with self.subTest(forbidden=forbidden):
                invalid = {**valid, forbidden: 1}
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_prediction_row(invalid)

    def test_non_exact_candidates_stay_in_hard(self) -> None:
        result = partition_hard_indices({0, 2}, set())
        self.assertEqual(result.guarded_hard_indices, (0, 2))
        self.assertEqual(result.train_raw_exact_normal_indices, ())

    def test_blob_loader_and_prediction_do_not_need_test_labels(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.db"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
                CREATE TABLE instances (
                    dataset_id INTEGER, split TEXT, instance_index INTEGER,
                    label TEXT, values_blob BLOB, labels_blob BLOB
                );
                INSERT INTO datasets VALUES (1, 'demo');
                """
            )
            train_blob = b"normal-series"
            connection.executemany(
                "INSERT INTO instances VALUES (1, ?, ?, ?, ?, NULL)",
                [
                    ("TRAIN", 0, "0", train_blob),
                    ("TEST", 0, "1", train_blob),
                    ("TEST", 1, "0", b"different-series"),
                ],
            )
            connection.commit()
            connection.close()

            train, test = load_blob_bundle_for_prediction("demo", db_path)
            self.assertEqual(train, {0: train_blob})
            self.assertEqual(test, {0: train_blob, 1: b"different-series"})
            rows = build_prediction_rows(
                [{"dataset": "demo", "n_test": 2, "final_hard_indices": [0, 1]}],
                db_path=db_path,
                expected_datasets=1,
                expected_hard=2,
            )
            self.assertEqual(rows[0]["guarded_hard_indices"], [1])
            self.assertEqual(rows[0]["train_raw_exact_normal_indices"], [0])
            self.assertNotIn("label", rows[0])


if __name__ == "__main__":
    unittest.main()
