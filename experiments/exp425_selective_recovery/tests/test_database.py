"""One-hash Exp425 database snapshot tests."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import shutil
import sqlite3
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp425_selective_recovery import database as snapshot_database
from experiments.exp425_selective_recovery.database import open_database_snapshot


class DatabaseSnapshotTests(unittest.TestCase):
    def test_snapshot_module_exposes_context_manager(self) -> None:
        self.assertTrue(callable(open_database_snapshot))

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "input.sqlite"
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE instances (dataset_id INTEGER, instance_index INTEGER, split TEXT, label TEXT, values_blob BLOB);")
            values = struct.pack("<2f", 1.0, 2.0)
            connection.executemany("INSERT INTO datasets VALUES (?, ?)", ((1, "A"), (2, "B")))
            connection.executemany("INSERT INTO instances VALUES (?, ?, ?, ?, ?)", ((1, 0, "TRAIN", "0", values), (1, 0, "TEST", "hidden", values), (2, 0, "TRAIN", "0", values), (2, 0, "TEST", "hidden", values)))
            connection.commit()
        finally:
            connection.close()
        self.digest = sha256(self.database.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_retained_descriptor_hashes_once_across_boundaries_and_close(self) -> None:
        with patch.object(snapshot_database, "_sha256_descriptor", wraps=snapshot_database._sha256_descriptor) as hashed:
            manager = open_database_snapshot(self.database, expected_sha256=self.digest)
            snapshot = manager.__enter__()
            snapshot.verify_boundary("first")
            snapshot.verify_boundary("second")
            snapshot.verify_boundary("third")
            manager.__exit__(None, None, None)
        self.assertEqual(hashed.call_count, 1)
        self.assertEqual(snapshot.audit.full_hash_count, 1)
        self.assertEqual(snapshot.audit.boundaries, ("first", "second", "third", "close:before"))

    def test_boundary_rejects_path_replacement_without_rehashing(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        replacement = self.root / "replacement.sqlite"
        shutil.copyfile(self.database, replacement)
        os.replace(replacement, self.database)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            snapshot.verify_boundary("replacement")
        with self.assertRaises(ValueError):
            manager.__exit__(None, None, None)

    def test_sidecars_and_symlink_inputs_fail_closed(self) -> None:
        (self.root / "input.sqlite-wal").write_bytes(b"writer")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            with open_database_snapshot(self.database, expected_sha256=self.digest):
                pass
        (self.root / "input.sqlite-wal").unlink()
        alias = self.root / "alias.sqlite"
        alias.symlink_to(self.database)
        with self.assertRaisesRegex(ValueError, "symlink"):
            with open_database_snapshot(alias, expected_sha256=self.digest):
                pass

    def test_close_and_invalid_digest_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA256"):
            with open_database_snapshot(self.database, expected_sha256="0" * 64):
                pass
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        manager.__exit__(None, None, None)
        with self.assertRaisesRegex(ValueError, "closed"):
            snapshot.verify_boundary("after-close")
