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


class Exp443DatabaseAntigravityTests(unittest.TestCase):
    def _db(self, root: Path, name: str = "Tiny_normal_0") -> Path:
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
            INSERT INTO datasets(id, name, series_length) VALUES (1, 'Tiny_normal_0', 3);
            """
        )
        train = [np.asarray([1.0, np.nan, 3.0], dtype="<f4"), np.asarray([2.0, 4.0, 6.0], dtype="<f4")]
        test = [np.asarray([7.0, 8.0, 9.0], dtype="<f4"), np.asarray([10.0, 11.0, 12.0], dtype="<f4")]
        rows = [(1, "TRAIN", 0, "0", train[0].tobytes(), b"must-not-read"), (1, "TRAIN", 1, "0", train[1].tobytes(), b"must-not-read"), (1, "TEST", 0, "0", test[0].tobytes(), b"must-not-read"), (1, "TEST", 1, "1", test[1].tobytes(), b"must-not-read")]
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
                try:
                    pinned.close(terminal_evidence_durable=True)
                except ValueError:
                    # The deliberately replaced source pathname is retained as
                    # mutation evidence; cleanup must not hide that failure.
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

    def test_missing_inventory_requires_sorted_population_and_reports_real_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._db(root)
            expected = sha256(source.read_bytes()).hexdigest()
            pinned = database.pin_database_snapshot_antigravity(source, root / "attempt_antigravity", expected_sha256=expected)
            try:
                with self.assertRaises(ValueError):
                    database.scan_missing_inventory_antigravity(pinned, ("Tiny_normal_0", "Tiny_normal_0"))
                inventory = database.scan_missing_inventory_antigravity(pinned, ("Tiny_normal_0",))
                self.assertEqual(inventory.population, ("Tiny_normal_0",))
                self.assertEqual(inventory.affected_rows, 1)
                self.assertEqual(inventory.nonfinite_scalars, 1)
                self.assertEqual(inventory.all_missing_rows, 0)
                self.assertEqual(inventory.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)


if __name__ == "__main__":
    unittest.main()
