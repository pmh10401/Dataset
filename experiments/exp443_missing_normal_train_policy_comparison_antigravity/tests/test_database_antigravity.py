"""Strict RED tests for the Exp443 descriptor-bound database boundary."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity import database_antigravity as database
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import RawSeriesAntigravity


class Exp443DatabaseAntigravityTests(unittest.TestCase):
    def _db(
        self,
        root: Path,
        name: str = "Tiny_normal_0",
        *,
        ragged: bool = False,
        train_indices: tuple[int, int] = (0, 1),
        test_indices: tuple[int, int] = (0, 1),
    ) -> Path:
        path = root / "source_antigravity.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                univariate BOOLEAN NOT NULL DEFAULT 1,
                equal_length BOOLEAN NOT NULL DEFAULT 1,
                series_length INTEGER, has_missing BOOLEAN NOT NULL DEFAULT 0,
                train_normal_count INTEGER NOT NULL DEFAULT 2,
                train_anomaly_count INTEGER NOT NULL DEFAULT 0,
                train_total_count INTEGER NOT NULL DEFAULT 2,
                test_normal_count INTEGER NOT NULL DEFAULT 2,
                test_anomaly_count INTEGER NOT NULL DEFAULT 0,
                test_total_count INTEGER NOT NULL DEFAULT 2,
                total_normal_count INTEGER NOT NULL DEFAULT 4,
                total_anomaly_count INTEGER NOT NULL DEFAULT 0,
                total_count INTEGER NOT NULL DEFAULT 4
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL, instance_index INTEGER NOT NULL,
                label TEXT NOT NULL, values_blob BLOB NOT NULL, labels_blob BLOB
            );
            CREATE TABLE evaluation (dataset TEXT NOT NULL, metric REAL);
            INSERT INTO datasets(id, name, series_length) VALUES (1, 'Tiny_normal_0', 3);
            """
        )
        if ragged:
            train = [
                np.asarray([1.0, np.nan, 3.0], dtype="<f4"),
                np.asarray([2.0, 4.0, 6.0, 8.0], dtype="<f4"),
            ]
            test = [
                np.asarray([7.0, 8.0], dtype="<f4"),
                np.asarray([10.0, 11.0, 12.0, 13.0, 14.0], dtype="<f4"),
            ]
        else:
            train = [np.asarray([1.0, np.nan, 3.0], dtype="<f4"), np.asarray([2.0, 4.0, 6.0], dtype="<f4")]
            test = [np.asarray([7.0, 8.0, 9.0], dtype="<f4"), np.asarray([10.0, 11.0, 12.0], dtype="<f4")]
        rows = [(1, "TRAIN", train_indices[0], "0", train[0].tobytes(), b"must-not-read"), (1, "TRAIN", train_indices[1], "0", train[1].tobytes(), b"must-not-read"), (1, "TEST", test_indices[0], "0", test[0].tobytes(), b"must-not-read"), (1, "TEST", test_indices[1], "1", test[1].tobytes(), b"must-not-read")]
        connection.executemany("INSERT INTO instances(dataset_id, split, instance_index, label, values_blob, labels_blob) VALUES (?, ?, ?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()
        return path

    def test_sql_allowlist_is_exact_and_prediction_firewall_has_no_label_capable_test_query(self) -> None:
        self.assertEqual(
            database.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index""",
        )
        self.assertEqual(
            database.TEST_VALUES_SQL_ANTIGRAVITY,
            """SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index""",
        )
        self.assertNotIn("labels_blob", database.TEST_VALUES_SQL_ANTIGRAVITY.lower())
        self.assertNotIn("label =", database.TEST_VALUES_SQL_ANTIGRAVITY.lower())
        with self.assertRaises(ValueError):
            database.validate_prediction_sql_antigravity("SELECT label, values_blob FROM instances WHERE split='TEST'")
        with self.assertRaises(ValueError):
            database.validate_prediction_sql_antigravity("SELECT i.values_blob, i.labels_blob FROM instances AS i WHERE i.split='TEST'")
        with self.assertRaises(ValueError):
            database.validate_prediction_sql_antigravity("SELECT * FROM evaluation")

    def test_symlink_and_fifo_are_rejected_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            link = root / "symlink_antigravity.db"
            link.symlink_to(source)
            with self.assertRaises((OSError, ValueError)):
                database.pin_database_snapshot_antigravity(link, root / "attempt_antigravity")
            fifo = root / "fifo_antigravity.db"
            os.mkfifo(fifo)
            with self.assertRaises((OSError, ValueError)):
                database.pin_database_snapshot_antigravity(fifo, root / "attempt_fifo_antigravity")

    def test_wrong_hash_is_rejected_and_pin_is_same_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            with self.assertRaises(ValueError):
                database.pin_database_snapshot_antigravity(source, root / "wrong_antigravity", expected_sha256="0" * 64)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                source_info = source.stat()
                pin_info = pinned.pin_path.lstat()
                self.assertEqual((source_info.st_dev, source_info.st_ino), (pin_info.st_dev, pin_info.st_ino))
                self.assertEqual(pinned.database_sha256, expected)
                self.assertEqual(pinned.parent_descriptor_inode, (source_info.st_dev, source_info.st_ino))
                self.assertTrue(pinned.pin_path.name == "database_pin_antigravity.db")
            finally:
                pinned.close(terminal_evidence_durable=True)
            self.assertFalse((root / "attempt_antigravity" / "database_pin_antigravity.db").exists())

    def test_resumable_pin_can_be_reopened_without_creating_a_second_hard_link(self) -> None:
        """An interrupted attempt keeps its pin and resume must reuse it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            attempt = root / "attempt_resume_antigravity"
            pinned = database.pin_database_snapshot_antigravity(source, attempt, expected_sha256=expected)
            original_inode = pinned.pin_inode
            # A resumable interruption closes the descriptor but intentionally
            # leaves the attempt-local hard link in place.
            pinned.close(terminal_evidence_durable=False)
            self.assertTrue((attempt / "database_pin_antigravity.db").exists())
            resumed = database.reopen_pinned_database_snapshot_antigravity(
                source, attempt, expected_sha256=expected
            )
            try:
                self.assertEqual(resumed.pin_inode, original_inode)
                with resumed.worker_connection_antigravity() as connection:
                    self.assertEqual(
                        len(connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",)).fetchall()),
                        2,
                    )
            finally:
                resumed.close(terminal_evidence_durable=True)
            self.assertFalse((attempt / "database_pin_antigravity.db").exists())

    def test_worker_connection_is_bound_to_parent_pin_and_trace_excludes_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                with pinned.worker_connection_antigravity() as connection:
                    result = connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",)).fetchall()
                    self.assertEqual([row[0] for row in result], [0, 1])
                    self.assertEqual(connection.execute("PRAGMA query_only").fetchone()[0], 1)
                self.assertEqual(pinned.worker_pin_inode_antigravity(), pinned.pin_inode)
                self.assertTrue(all("labels_blob" not in sql.lower() for sql in pinned.sql_trace))
                self.assertFalse(any("split = 'test'" in sql.lower() and "label" in sql.lower() for sql in pinned.sql_trace))
            finally:
                pinned.close(terminal_evidence_durable=True)

    def test_worker_execute_gate_denies_before_raw_execution_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                with pinned.worker_connection_antigravity() as connection:
                    before = pinned.raw_sql_execute_count
                    forbidden = (
                        "SELECT i.instance_index, i.values_blob FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id WHERE d.name = ? AND i.split = 'TEST' AND i.label = '0' ORDER BY i.instance_index",
                        "SELECT i.instance_index, i.labels_blob FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0' ORDER BY i.instance_index",
                        "SELECT * FROM evaluation",
                    )
                    for sql in forbidden:
                        with self.assertRaises(ValueError):
                            connection.execute(sql, ("Tiny_normal_0",))
                    self.assertEqual(pinned.raw_sql_execute_count, before)
                    self.assertEqual(pinned.denied_sql_attempts, 3)
                    self.assertEqual(pinned.label_access_count, 0)
                    self.assertEqual(len(pinned.denied_sql_trace), 3)
                    cursor = connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",))
                    with self.assertRaises(ValueError):
                        cursor.execute("SELECT * FROM evaluation")
                    self.assertEqual(pinned.raw_sql_execute_count, before + 1)
                    approved = connection.execute(database.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",)).fetchall()
                    self.assertEqual([row[0] for row in approved], [0, 1])
                self.assertEqual(pinned.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)

    def test_facades_have_no_raw_escape_and_authorizer_binds_label_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            forbidden_test_label = "SELECT i.instance_index, i.values_blob FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id WHERE d.name = ? AND i.split = 'TEST' AND i.label = '0' ORDER BY i.instance_index"
            forbidden_blob = "SELECT i.instance_index, i.labels_blob FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0' ORDER BY i.instance_index"
            try:
                with pinned.worker_connection_antigravity() as connection:
                    self.assertFalse(hasattr(connection, "_connection"))
                    cursor = connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",))
                    self.assertFalse(hasattr(cursor, "_cursor"))
                    if hasattr(cursor, "connection"):
                        self.assertFalse(hasattr(cursor.connection, "_connection"))
                        with self.assertRaises(ValueError):
                            cursor.connection.execute(forbidden_test_label, ("Tiny_normal_0",))
                    for surface_name in ("_connection", "connection", "raw_connection"):
                        if hasattr(connection, surface_name):
                            surface = getattr(connection, surface_name)
                            with self.assertRaises((AttributeError, ValueError)):
                                surface.execute(forbidden_test_label, ("Tiny_normal_0",))
                    for surface_name in ("_cursor", "raw_cursor"):
                        if hasattr(cursor, surface_name):
                            surface = getattr(cursor, surface_name)
                            with self.assertRaises((AttributeError, ValueError)):
                                surface.execute(forbidden_blob, ("Tiny_normal_0",))
                    denied_before = pinned.denied_sql_attempts
                    with self.assertRaises(ValueError):
                        connection.execute(forbidden_test_label, ("Tiny_normal_0",))
                    with self.assertRaises(ValueError):
                        cursor.execute(forbidden_blob, ("Tiny_normal_0",))
                    self.assertEqual(pinned.denied_sql_attempts, denied_before + 2)
                    self.assertEqual(pinned.label_access_count, 0)
                self.assertEqual(pinned.label_access_count, 0)

                pinned._active_sql = database.NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY
                self.assertEqual(pinned._authorizer(sqlite3.SQLITE_READ, "instances", "label", None, None), sqlite3.SQLITE_OK)
                pinned._active_sql = database.TEST_VALUES_SQL_ANTIGRAVITY
                self.assertEqual(pinned._authorizer(sqlite3.SQLITE_READ, "instances", "label", None, None), sqlite3.SQLITE_DENY)
                self.assertEqual(pinned._authorizer(sqlite3.SQLITE_READ, "instances", "labels_blob", None, None), sqlite3.SQLITE_DENY)
                self.assertEqual(pinned._authorizer(sqlite3.SQLITE_READ, "evaluation", "metric", None, None), sqlite3.SQLITE_DENY)
                self.assertEqual(pinned.label_access_count, 0)
            finally:
                pinned._active_sql = None
                pinned.close(terminal_evidence_durable=True)

    def test_facade_attributes_have_no_raw_objects_or_bound_sqlite_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            forbidden = "SELECT 1"
            try:
                with pinned.worker_connection_antigravity() as connection:
                    cursor = connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",))
                    for facade in (connection, cursor, cursor.connection):
                        self.assertFalse(hasattr(facade, "__dict__"))
                        raw_bound_escape = 0
                        for name in dir(facade):
                            if name.startswith("__"):
                                continue
                            value = getattr(facade, name)
                            self.assertFalse(isinstance(value, (sqlite3.Connection, sqlite3.Cursor)), name)
                            if callable(value):
                                owner = getattr(value, "__self__", None)
                                if isinstance(owner, (sqlite3.Connection, sqlite3.Cursor)):
                                    raw_bound_escape += 1
                                    with self.assertRaises(ValueError):
                                        owner.execute(forbidden)
                        self.assertEqual(raw_bound_escape, 0, facade.__class__.__name__)
                    self.assertEqual(pinned.denied_sql_attempts, 0)
                    self.assertEqual(pinned.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)

    def test_database_pre_open_swap_to_fifo_and_symlink_is_fail_closed(self) -> None:
        for replacement_kind in ("fifo", "symlink"):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._db(root)
                expected = sha256(source.read_bytes()).hexdigest()
                backup = root / "source_backup_antigravity.db"

                def swap(_path: Path, *, kind: str = replacement_kind) -> None:
                    source.rename(backup)
                    if kind == "fifo":
                        os.mkfifo(source)
                    else:
                        source.symlink_to(backup)

                previous = database._DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY
                database._DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY = swap
                try:
                    with self.assertRaises((OSError, ValueError)):
                        database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
                finally:
                    database._DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY = previous
                    if source.is_symlink() or source.exists():
                        source.unlink()
                    backup.rename(source)

    def test_after_query_source_or_pin_mutation_fails_revalidation(self) -> None:
        for target in ("source", "pin"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._db(root)
                expected = sha256(source.read_bytes()).hexdigest()
                pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
                target_path = source if target == "source" else pinned.pin_path
                backup = target_path.with_name(target_path.name + ".backup_antigravity")

                def swap(_owner: object) -> None:
                    target_path.rename(backup)
                    target_path.symlink_to(backup)

                previous = database._AFTER_QUERY_HOOK_ANTIGRAVITY
                database._AFTER_QUERY_HOOK_ANTIGRAVITY = swap
                try:
                    with self.assertRaises(ValueError):
                        with pinned.worker_connection_antigravity() as connection:
                            connection.execute(database.TEST_VALUES_SQL_ANTIGRAVITY, ("Tiny_normal_0",)).fetchall()
                finally:
                    database._AFTER_QUERY_HOOK_ANTIGRAVITY = previous
                    if target_path.is_symlink() or target_path.exists():
                        target_path.unlink()
                    backup.rename(target_path)
                    try:
                        pinned.close(terminal_evidence_durable=True)
                    except ValueError:
                        # Boundary failure is the assertion; descriptor cleanup
                        # has already run and the mutation evidence is retained.
                        pass

    def test_raw_loader_decodes_nonfinite_mask_and_rejects_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                raw = database.load_raw_dataset_antigravity(pinned, "Tiny_normal_0")
                self.assertEqual(raw.dataset, "Tiny_normal_0")
                self.assertEqual(tuple(item.instance_index for item in raw.train), (0, 1))
                self.assertEqual(raw.train[0].missing_positions, (1,))
                self.assertEqual(raw.test[0].missing_positions, ())
                replacement = root / "replacement_antigravity.db"
                replacement.write_bytes(source.read_bytes().replace(b"Tiny_normal_0", b"Tiny_normal_X"))
                source.unlink()
                source.symlink_to(replacement)
                with self.assertRaises((OSError, ValueError)):
                    database.load_raw_dataset_antigravity(pinned, "Tiny_normal_0")
            finally:
                try:
                    pinned.close(terminal_evidence_durable=True)
                except ValueError:
                    # The deliberately replaced source pathname is retained as
                    # mutation evidence; cleanup must not hide that failure.
                    pass

    def test_raw_loader_preserves_ragged_float32_lengths_and_values(self) -> None:
        """The frozen Task 4 contract permits variable original row lengths."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root, ragged=True)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(
                source, root / "attempt_ragged_antigravity", expected_sha256=expected
            )
            try:
                raw = database.load_raw_dataset_antigravity(pinned, "Tiny_normal_0")
                self.assertEqual(tuple(len(row.values) for row in raw.train), (3, 4))
                self.assertEqual(tuple(len(row.values) for row in raw.test), (2, 5))
                self.assertEqual(raw.train[0].values.dtype, np.dtype("<f4"))
                self.assertTrue(np.array_equal(raw.train[0].values, np.asarray([1.0, np.nan, 3.0], dtype="<f4"), equal_nan=True))
                self.assertTrue(np.array_equal(raw.train[1].values, np.asarray([2.0, 4.0, 6.0, 8.0], dtype="<f4")))
                self.assertEqual(tuple(row.instance_index for row in raw.train), (0, 1))
                self.assertEqual(tuple(row.instance_index for row in raw.test), (0, 1))
                self.assertEqual(raw.target_length, 8)
                self.assertFalse(raw.train[0].values.flags.writeable)
                self.assertEqual(pinned.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)

    def test_raw_loader_rejects_train_and_test_identity_gaps(self) -> None:
        for kwargs in (
            {"train_indices": (0, 2)},
            {"test_indices": (0, 2)},
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self._db(root, **kwargs)
                expected = sha256(source.read_bytes()).hexdigest()
                pinned = database.pin_database_snapshot_antigravity(
                    source, root / "attempt_gap_antigravity", expected_sha256=expected
                )
                try:
                    with self.assertRaisesRegex(ValueError, "identity"):
                        database.load_raw_dataset_antigravity(pinned, "Tiny_normal_0")
                finally:
                    pinned.close(terminal_evidence_durable=True)

    def test_raw_dataset_target_is_always_clipped_train_length_median(self) -> None:
        def raw_series(length: int, split: str, index: int) -> RawSeriesAntigravity:
            values = np.arange(length, dtype="<f4")
            return RawSeriesAntigravity(
                "Demo", split, index, sha256(values.tobytes(order="C")).hexdigest(), values, ()
            )

        short = raw_series(3, "TRAIN", 0)
        test = raw_series(100, "TEST", 0)
        with self.assertRaisesRegex(ValueError, "target_length"):
            database.RawDatasetAntigravity("Demo", (short,), (test,), 3)
        self.assertEqual(database.RawDatasetAntigravity("Demo", (short,), (test,), 8).target_length, 8)

        long = raw_series(3000, "TRAIN", 0)
        long_test = raw_series(3000, "TEST", 0)
        with self.assertRaisesRegex(ValueError, "target_length"):
            database.RawDatasetAntigravity("Demo", (long,), (long_test,), 3000)
        self.assertEqual(database.RawDatasetAntigravity("Demo", (long,), (long_test,), 2048).target_length, 2048)

    def test_missing_inventory_requires_sorted_population_and_reports_real_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                with self.assertRaises(ValueError):
                    database.scan_missing_inventory_antigravity(pinned, ("Tiny_normal_0", "Tiny_normal_0"))
                inventory = database.scan_missing_inventory_antigravity(pinned, ("Tiny_normal_0",), expected_affected_family_count=1, database_exclusions=frozenset())
                self.assertEqual(inventory.population, ("Tiny_normal_0",))
                self.assertEqual(inventory.affected_rows, 1)
                self.assertEqual(inventory.nonfinite_scalars, 1)
                self.assertEqual(inventory.all_missing_rows, 0)
                self.assertEqual(inventory.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)


if __name__ == "__main__":
    unittest.main()
