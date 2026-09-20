from __future__ import annotations

import inspect
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


class RecordingConnectionAntigravity:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.sql: list[str] = []
        self.parameters: list[tuple[object, ...]] = []
        self.closed = False

    def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
        self.sql.append(sql)
        self.parameters.append(parameters)
        return self.connection.execute(sql, parameters)

    def close(self) -> None:
        self.closed = True
        self.connection.close()


class StaticConnectionAntigravity:
    def __init__(self, rows: tuple[tuple[object, object], ...]) -> None:
        self.rows = rows
        self.closed = False

    def execute(self, _sql: str, _parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
        return self.rows

    def close(self) -> None:
        self.closed = True


class SourcesAntigravityTest(unittest.TestCase):
    def test_loader_exports_only_the_exact_normal2_train_entrypoint_antigravity(self) -> None:
        """Break caught: Task 3 grows a dataset, TEST, or label-capable source API."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            sources_antigravity as sources,
        )

        self.assertEqual(tuple(sources.__all__), ("load_normal2_train_antigravity",))
        self.assertEqual(
            tuple(inspect.signature(sources.load_normal2_train_antigravity).parameters),
            (),
        )
        self.assertFalse(hasattr(sources, "load_ecg_train_antigravity"))
        self.assertEqual(sources._EXPECTED_TRAIN_ROWS_ANTIGRAVITY, 1413)
        self.assertEqual(sources._EXPECTED_WIDTH_ANTIGRAVITY, 140)

    def test_loader_issues_one_normal2_train_value_query_with_no_labels_or_test_antigravity(
        self,
    ) -> None:
        """Break caught: source loading reads TEST rows, labels, or more than one DB query."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            sources_antigravity as sources,
        )

        with tempfile.TemporaryDirectory(prefix="exp459_source_antigravity_") as temporary:
            path = Path(temporary) / "fixture_antigravity.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                "CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
                "CREATE TABLE instances (dataset_id INTEGER NOT NULL, split TEXT NOT NULL, "
                "instance_index INTEGER NOT NULL, label TEXT NOT NULL, values_blob BLOB NOT NULL);"
            )
            connection.executemany(
                "INSERT INTO datasets VALUES (?, ?)",
                ((1, "ECG5000_normal_1"), (2, "ECG5000_normal_2")),
            )
            blob = lambda values: np.asarray(values, dtype="<f4").tobytes()
            connection.executemany(
                "INSERT INTO instances VALUES (?, ?, ?, ?, ?)",
                (
                    (2, "TRAIN", 2, "hidden-c", blob((5.0, 6.0, 7.0))),
                    (2, "TEST", 0, "hidden-test", blob((90.0, 91.0, 92.0))),
                    (1, "TRAIN", 0, "wrong-dataset", blob((70.0, 71.0, 72.0))),
                    (2, "TRAIN", 0, "hidden-a", blob((1.0, 2.0, 3.0))),
                    (2, "TRAIN", 1, "hidden-b", blob((3.0, 4.0, 5.0))),
                ),
            )
            connection.commit()
            connection.close()

            recording = RecordingConnectionAntigravity(sqlite3.connect(path))
            with (
                patch.object(sources, "_connection_antigravity", return_value=recording),
                patch.object(sources, "_EXPECTED_TRAIN_ROWS_ANTIGRAVITY", 3),
                patch.object(sources, "_EXPECTED_WIDTH_ANTIGRAVITY", 3),
            ):
                actual = sources.load_normal2_train_antigravity()

        np.testing.assert_array_equal(
            actual,
            np.asarray(
                ((1.0, 2.0, 3.0), (3.0, 4.0, 5.0), (5.0, 6.0, 7.0)),
                dtype=np.float32,
            ),
        )
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        self.assertEqual(actual.shape, (3, 3))
        self.assertTrue(actual.flags.owndata)
        self.assertFalse(actual.flags.writeable)
        self.assertTrue(recording.closed)
        self.assertEqual(len(recording.sql), 1)
        normalized_sql = " ".join(recording.sql[0].casefold().split())
        self.assertIn("select i.instance_index, i.values_blob", normalized_sql)
        self.assertIn("i.split = 'train'", normalized_sql)
        self.assertNotIn("label", normalized_sql)
        self.assertNotIn("test", normalized_sql)
        self.assertEqual(recording.parameters, [("ECG5000_normal_2",)])

    def test_loader_rejects_malformed_train_rows_and_never_aliases_results_antigravity(
        self,
    ) -> None:
        """Break caught: malformed row count, indices, width, nonfinite data, or BLOB aliasing is accepted."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            sources_antigravity as sources,
        )

        valid = np.asarray((1.0, 2.0, 3.0), dtype="<f4").tobytes()
        short = np.asarray((1.0, 2.0), dtype="<f4").tobytes()
        nonfinite = np.asarray((1.0, np.nan, 3.0), dtype="<f4").tobytes()
        cases = (
            ("wrong-count", ((0, valid), (1, valid)), "row count"),
            ("wrong-start", ((1, valid), (2, valid), (3, valid)), "indices"),
            ("missing", ((0, valid), (2, valid), (3, valid)), "indices"),
            ("duplicate", ((0, valid), (1, valid), (1, valid)), "indices"),
            ("width", ((0, valid), (1, short), (2, valid)), "shape"),
            ("finite", ((0, valid), (1, nonfinite), (2, valid)), "finite"),
            ("not-bytes", ((0, valid), (1, memoryview(valid)), (2, valid)), "bytes"),
        )
        for name, rows, message in cases:
            with self.subTest(name=name):
                connection = StaticConnectionAntigravity(rows)
                with (
                    patch.object(sources, "_connection_antigravity", return_value=connection),
                    patch.object(sources, "_EXPECTED_TRAIN_ROWS_ANTIGRAVITY", 3),
                    patch.object(sources, "_EXPECTED_WIDTH_ANTIGRAVITY", 3),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    sources.load_normal2_train_antigravity()
                self.assertTrue(connection.closed)

        first_connection = StaticConnectionAntigravity(((0, valid), (1, valid), (2, valid)))
        second_connection = StaticConnectionAntigravity(((0, valid), (1, valid), (2, valid)))
        with (
            patch.object(
                sources,
                "_connection_antigravity",
                side_effect=(first_connection, second_connection),
            ),
            patch.object(sources, "_EXPECTED_TRAIN_ROWS_ANTIGRAVITY", 3),
            patch.object(sources, "_EXPECTED_WIDTH_ANTIGRAVITY", 3),
        ):
            first = sources.load_normal2_train_antigravity()
            second = sources.load_normal2_train_antigravity()
        self.assertFalse(np.shares_memory(first, second))
        self.assertIsNone(first.base)
        self.assertIsNone(second.base)


if __name__ == "__main__":
    unittest.main()
