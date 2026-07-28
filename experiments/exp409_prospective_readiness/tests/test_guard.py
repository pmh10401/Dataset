from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.exp409_prospective_readiness.guard import (
    LabelBlindSqlTrace,
    load_values_bundle_label_blind,
    reject_forbidden_inputs,
    validate_label_blind_payload,
)


class GuardTests(unittest.TestCase):
    def test_test_label_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_label_blind_payload(
                {"dataset": "a", "test_labels": [0, 1]}
            )

    def test_exp408_evaluation_csv_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation input"):
            reject_forbidden_inputs(
                [Path("/data/exp408_variant_candidate_evaluation.csv")]
            )

    def test_exp408_checkpoint_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            reject_forbidden_inputs(
                [
                    Path(
                        "/data/exp403_408_feature_ml_validation/checkpoints"
                    )
                ]
            )

    def test_normal_train_and_unlabeled_test_values_are_allowed(self) -> None:
        trace = LabelBlindSqlTrace()
        trace("SELECT value FROM train_normal WHERE dataset = 'a'")
        trace("SELECT value FROM test_values WHERE dataset = 'a'")
        trace.assert_clean()

    def test_test_label_query_is_rejected(self) -> None:
        trace = LabelBlindSqlTrace()
        trace("SELECT is_anomaly FROM test_labels WHERE dataset = 'a'")
        with self.assertRaisesRegex(ValueError, "TEST label"):
            trace.assert_clean()

    def test_test_derived_ordering_is_rejected(self) -> None:
        trace = LabelBlindSqlTrace()
        trace("SELECT dataset FROM test_metadata ORDER BY anomaly_count")
        with self.assertRaisesRegex(ValueError, "TEST-derived"):
            trace.assert_clean()

    def test_values_loader_never_reads_label_column(self) -> None:
        with TemporaryDirectory() as directory:
            database = Path(directory) / "fixture.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE datasets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE instances (
                    dataset_id INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    instance_index INTEGER NOT NULL,
                    values_blob BLOB NOT NULL,
                    label INTEGER
                );
                INSERT INTO datasets(id, name) VALUES (1, 'sample');
                """
            )
            rows = [
                (1, "TRAIN", 0, np.asarray([1, 2], dtype=np.float32).tobytes(), None),
                (1, "TRAIN", 1, np.asarray([3, 4], dtype=np.float32).tobytes(), None),
                (1, "TEST", 0, np.asarray([5, 6], dtype=np.float32).tobytes(), 1),
            ]
            connection.executemany(
                "INSERT INTO instances VALUES (?, ?, ?, ?, ?)", rows
            )
            connection.commit()
            connection.close()

            trace = LabelBlindSqlTrace()
            bundle = load_values_bundle_label_blind(
                "sample", database, trace=trace
            )

            self.assertEqual(len(bundle.train_normal), 2)
            self.assertEqual(len(bundle.test_values), 1)
            trace.assert_clean()
            statements = "\n".join(trace.statements).lower()
            self.assertIn("values_blob", statements)
            self.assertNotIn("i.label", statements)
            self.assertNotIn("is_anomaly", statements)


if __name__ == "__main__":
    unittest.main()
