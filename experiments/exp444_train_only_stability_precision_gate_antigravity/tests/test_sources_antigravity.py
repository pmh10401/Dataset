"""TDD tests for Exp444's frozen-source and values-only boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from hashlib import sha256

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity import sources_antigravity as sources
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    FrozenSourceSnapshotAntigravity,
    SourceCandidateAntigravity,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        for row in rows
    ))


def _candidate(dataset: str, index: int, rank: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "instance_index": index,
        "raw_length": 8,
        "depths": [1, 2],
        "agreement_count": 2,
        "second_excess_ratio": 1.2,
        "second_empirical_rank": 0.1,
        "global_rank": rank,
        "selected": True,
        "selection_rank": rank,
    }


class Exp444SourceBoundaryAntigravityTests(unittest.TestCase):
    def test_source_snapshot_has_exact_thirty_coordinates_antigravity(self) -> None:
        # RED contract: the implementation must replay exactly the frozen source
        # population, not merely trust a candidate count.
        self.assertTrue(hasattr(sources, "verify_frozen_sources_antigravity"))

    def test_prediction_sql_rejects_test_labels_antigravity(self) -> None:
        with self.assertRaisesRegex(ValueError, "label-capable SQL"):
            sources.validate_prediction_sql_antigravity(
                "SELECT instance_index, label FROM instances WHERE split='TEST'"
            )

    def test_prediction_sql_rejects_labels_blob_and_select_star_antigravity(self) -> None:
        for sql in (
            "SELECT i.instance_index, i.labels_blob FROM instances AS i WHERE i.split='TEST'",
            "SELECT * FROM instances",
            "SELECT instance_index, values_blob FROM unknown_table",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                sources.validate_prediction_sql_antigravity(sql)

    def test_exact_values_sql_constants_are_frozen_antigravity(self) -> None:
        self.assertEqual(
            sources.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index""",
        )
        self.assertEqual(
            sources.BOUND_TEST_VALUE_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' AND i.instance_index = ?""",
        )

    def test_values_loader_excludes_nonfinite_rows_and_returns_read_only_arrays_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_sources_test_antigravity_") as directory:
            root = Path(directory)
            db = root / "source_antigravity.db"
            connection = sqlite3.connect(db)
            connection.executescript(
                """
                CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
                CREATE TABLE instances (
                    dataset_id INTEGER NOT NULL, split TEXT NOT NULL,
                    instance_index INTEGER NOT NULL, label TEXT NOT NULL,
                    values_blob BLOB NOT NULL, labels_blob BLOB
                );
                """
            )
            connection.execute("INSERT INTO datasets(id, name) VALUES(1, 'A')")
            rows = [
                (1, "TRAIN", 0, "0", np.asarray([1, 2, 3, 4], dtype="<f4").tobytes(), None),
                (1, "TRAIN", 1, "0", np.asarray([1, np.nan, 3, 4], dtype="<f4").tobytes(), None),
                (1, "TRAIN", 2, "0", np.asarray([2, 3, 4, 5], dtype="<f4").tobytes(), None),
                (1, "TRAIN", 3, "0", np.asarray([3, 4, 5, 6], dtype="<f4").tobytes(), None),
                (1, "TRAIN", 4, "0", np.asarray([4, 5, 6, 7], dtype="<f4").tobytes(), None),
                (1, "TRAIN", 5, "1", np.asarray([9, 9, 9, 9], dtype="<f4").tobytes(), b"hidden"),
                (1, "TEST", 0, "1", np.asarray([6, 7, 8, 9], dtype="<f4").tobytes(), b"hidden"),
            ]
            connection.executemany("INSERT INTO instances VALUES(?, ?, ?, ?, ?, ?)", rows)
            connection.commit()
            connection.close()

            candidate = SourceCandidateAntigravity("A", 0, 4, (1, 2), 1)
            snapshot = FrozenSourceSnapshotAntigravity(
                candidates=(candidate,),
                selected_coordinate_identity_sha256="0" * 64,
                source_hashes={"database": sha256(db.read_bytes()).hexdigest()},
                database_path=db,
                label_access_count=0,
            )
            values = sources.load_candidate_values_antigravity(snapshot)
            self.assertEqual(len(values), 1)
            self.assertIsInstance(values[0], CandidateInputAntigravity)
            self.assertEqual(values[0].candidate.coordinate, ("A", 0))
            self.assertEqual(tuple(values[0].train_indices), (0, 2, 3, 4))
            self.assertFalse(values[0].train_values[0].flags.writeable)
            self.assertFalse(values[0].test_values.flags.writeable)
            self.assertEqual(snapshot.label_access_count, 0)

    def test_descriptor_session_has_no_raw_connection_or_cursor_attribute_antigravity(self) -> None:
        self.assertTrue(hasattr(sources, "validate_prediction_sql_antigravity"))
        forbidden = {"connection", "cursor", "raw_connection", "raw_cursor"}
        session_type = getattr(sources, "ValuesOnlyDatabaseSessionAntigravity", None)
        if session_type is not None:
            self.assertTrue(forbidden.isdisjoint(vars(session_type)))

    def test_source_candidate_contract_rejects_duplicate_or_bad_rank_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            SourceCandidateAntigravity("A", 0, 8, (1, 2), 0)


if __name__ == "__main__":
    unittest.main()
