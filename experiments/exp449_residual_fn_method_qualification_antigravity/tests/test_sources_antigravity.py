"""Task 3 label-firewall regression tests; label rows are synthetic only."""
from __future__ import annotations

import unittest
import os
import sys
import types
import hashlib
from pathlib import Path
import tempfile
import sqlite3
from unittest.mock import patch

import numpy as np


class RecordingConnectionAntigravity:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.closed = False

    def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
        self.sql.append(sql)
        if "values_blob" in sql:
            return (("TRAIN", 0, np.asarray([1.0, 2.0], dtype="<f4").tobytes()), ("TRAIN", 1, np.asarray([2.0, 3.0], dtype="<f4").tobytes()), ("TEST", 0, np.asarray([3.0, 4.0], dtype="<f4").tobytes()))
        return ()

    def close(self) -> None:
        self.closed = True


class SourcesAntigravityTests(unittest.TestCase):
    def _database_constants_antigravity(self, sources, path: Path):  # type: ignore[no-untyped-def]
        info = path.stat()
        return patch.object(sources, "_DATABASE_ANTIGRAVITY", path), patch.object(sources, "_DATABASE_SHA256_ANTIGRAVITY", hashlib.sha256(path.read_bytes()).hexdigest()), patch.object(sources, "_DATABASE_INODE_ANTIGRAVITY", info.st_ino)
    def _install_prediction_module_antigravity(self, order: tuple[str, ...] | None = None) -> tuple[types.ModuleType, object]:
        name = "experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity"
        module = types.ModuleType(name)
        order = ("Crop_normal_1", *(f"synthetic_target_{index}_antigravity" for index in range(47))) if order is None else order
        class FrozenPredictionAuthorityAntigravity:
            family_method_pairs = 16
            dataset_units = 96
            label_access_count = 0
            target_dataset_order = order
        FrozenPredictionAuthorityAntigravity.__module__ = name
        authority = FrozenPredictionAuthorityAntigravity()
        module.FrozenPredictionAuthorityAntigravity = FrozenPredictionAuthorityAntigravity
        module.validate_frozen_prediction_authority_antigravity = lambda value: order
        self.addCleanup(lambda: sys.modules.pop(name, None))
        sys.modules[name] = module
        return module, authority
    def test_values_bundle_never_selects_label_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import load_target_bundle_antigravity

        connection = RecordingConnectionAntigravity()
        bundle = load_target_bundle_antigravity("Crop_normal_1", connection_factory=lambda _: connection)
        self.assertEqual(bundle.dataset, "Crop_normal_1")
        self.assertEqual(bundle.train_normal.shape, (2, 2))
        self.assertEqual(bundle.test_values.shape, (1, 2))
        self.assertFalse(any("label" in sql.casefold() for sql in connection.sql))
        self.assertTrue(connection.closed)

    def test_database_identity_rejects_sha_inode_link_and_symlink_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp449_db_identity_antigravity_") as temporary:
            path = Path(temporary) / "source_antigravity.db"
            path.write_bytes(b"synthetic database bytes\n")
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            with path_patch, sha_patch, inode_patch:
                descriptor, identity = sources._verify_database_antigravity()
                os.close(descriptor)
            with patch.object(sources, "_DATABASE_ANTIGRAVITY", path), patch.object(sources, "_DATABASE_SHA256_ANTIGRAVITY", "0" * 64), patch.object(sources, "_DATABASE_INODE_ANTIGRAVITY", path.stat().st_ino), self.assertRaises(ValueError):
                sources._verify_database_antigravity()
            with patch.object(sources, "_DATABASE_ANTIGRAVITY", path), patch.object(sources, "_DATABASE_SHA256_ANTIGRAVITY", hashlib.sha256(path.read_bytes()).hexdigest()), patch.object(sources, "_DATABASE_INODE_ANTIGRAVITY", path.stat().st_ino + 1), self.assertRaises(ValueError):
                sources._verify_database_antigravity()
            linked = Path(temporary) / "linked_antigravity.db"
            os.link(path, linked)
            with patch.object(sources, "_DATABASE_ANTIGRAVITY", path), patch.object(sources, "_DATABASE_SHA256_ANTIGRAVITY", hashlib.sha256(path.read_bytes()).hexdigest()), patch.object(sources, "_DATABASE_INODE_ANTIGRAVITY", path.stat().st_ino), self.assertRaises(ValueError):
                sources._verify_database_antigravity()
            linked.unlink()
            symlink = Path(temporary) / "symlink_antigravity.db"
            symlink.symlink_to(path)
            with patch.object(sources, "_DATABASE_ANTIGRAVITY", symlink), patch.object(sources, "_DATABASE_SHA256_ANTIGRAVITY", hashlib.sha256(path.read_bytes()).hexdigest()), patch.object(sources, "_DATABASE_INODE_ANTIGRAVITY", path.stat().st_ino), self.assertRaises(ValueError):
                sources._verify_database_antigravity()

    def test_database_postverify_metadata_and_content_drift_are_rejected_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp449_db_drift_antigravity_") as temporary:
            path = Path(temporary) / "source_antigravity.db"
            path.write_bytes(b"synthetic database bytes\n")
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            with path_patch, sha_patch, inode_patch:
                descriptor, identity = sources._verify_database_antigravity()
                try:
                    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
                    with self.assertRaisesRegex(ValueError, "changed during read"):
                        sources._assert_database_unchanged_antigravity(descriptor, identity)
                finally:
                    os.close(descriptor)

    def test_database_inode_replacement_and_public_value_drift_are_rejected_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp449_public_db_antigravity_") as temporary:
            path = Path(temporary) / "source_antigravity.db"
            path.write_bytes(b"synthetic database bytes\n")
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            with path_patch, sha_patch, inode_patch:
                descriptor, identity = sources._verify_database_antigravity()
                replacement = Path(temporary) / "replacement_antigravity.db"
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
                try:
                    with self.assertRaisesRegex(ValueError, "changed during read"):
                        sources._assert_database_unchanged_antigravity(descriptor, identity)
                finally:
                    os.close(descriptor)
            path.unlink()
            connection = sqlite3.connect(path)
            connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE instances (dataset_id INTEGER, split TEXT, instance_index INTEGER, values_blob BLOB);")
            blob = np.asarray([1.0, 2.0], dtype="<f4").tobytes(); blob2 = np.asarray([2.0, 4.0], dtype="<f4").tobytes(); blob3 = np.asarray([3.0, 6.0], dtype="<f4").tobytes()
            connection.execute("INSERT INTO datasets VALUES (1, 'Crop_normal_1')")
            connection.executemany("INSERT INTO instances VALUES (?,?,?,?)", [(1, "TRAIN", 0, blob), (1, "TRAIN", 1, blob2), (1, "TEST", 0, blob3)])
            connection.commit(); connection.close()
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            original_normalise = sources._normalise_antigravity
            def drift(rows):  # type: ignore[no-untyped-def]
                connection = sqlite3.connect(path); connection.execute("PRAGMA user_version=1"); connection.commit(); connection.close()
                return original_normalise(rows)
            with path_patch, sha_patch, inode_patch, patch.object(sources, "_normalise_antigravity", side_effect=drift), self.assertRaisesRegex(ValueError, "changed during read"):
                sources.load_target_bundle_antigravity("Crop_normal_1")
            path.write_bytes(b"synthetic database bytEs\n")
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            with path_patch, sha_patch, inode_patch:
                descriptor, identity = sources._verify_database_antigravity()
                try:
                    path.write_bytes(b"synthetic database bytXs\n")
                    with self.assertRaisesRegex(ValueError, "changed during read"):
                        sources._assert_database_unchanged_antigravity(descriptor, identity)
                finally:
                    os.close(descriptor)

    def test_forged_prediction_authority_is_rejected_before_database_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import open_test_labels_once_antigravity

        with self.assertRaisesRegex(ValueError, "prediction authority is invalid"):
            open_test_labels_once_antigravity(object())

    def test_module_qualified_authority_requires_its_fixed_validator_before_database_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        module, authority = self._install_prediction_module_antigravity()
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (_ for _ in ()).throw(AssertionError("database reached"))  # type: ignore[assignment]
            module.validate_frozen_prediction_authority_antigravity = lambda value: (_ for _ in ()).throw(ValueError("stale manifest"))
            with self.assertRaisesRegex(ValueError, "prediction authority is invalid"):
                sources.open_test_labels_once_antigravity(authority)
            self.assertIs(type(authority), module.FrozenPredictionAuthorityAntigravity)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_exact_text_label_converter_and_contiguous_rows_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import convert_test_label_row_antigravity

        self.assertEqual(convert_test_label_row_antigravity(0, "0"), (0, 0))
        self.assertEqual(convert_test_label_row_antigravity(1, "1"), (1, 1))
        for bad in (0, 1, b"0", "2", None):
            with self.assertRaises(ValueError):
                convert_test_label_row_antigravity(0, bad)

    def test_label_session_close_is_truthful_after_incomplete_order_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (-1, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "did not complete"):
                sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: RecordingConnectionAntigravity()).close_antigravity()
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_query_failure_is_terminal_and_closes_resources_once_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class FailingConnectionAntigravity:
            def __init__(self) -> None:
                self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
                raise RuntimeError("query boom")
            def close(self) -> None:
                self.close_calls += 1
        connection = FailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(connection.close_calls, 1)
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.labels_for_dataset_antigravity("Crop_normal_1")
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_connection_close_failure_is_terminal_close_failed_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        order = tuple(f"close_target_{index}_antigravity" for index in range(48))
        _, authority = self._install_prediction_module_antigravity(order)
        class CloseFailingConnectionAntigravity:
            def __init__(self) -> None:
                self.close_calls = 0
                self.query_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
                self.query_calls += 1
                size = 1518 if self.query_calls == 48 else 1506
                return tuple((index, "0") for index in range(size))
            def close(self) -> None:
                self.close_calls += 1
                raise OSError("close boom")
        connection = CloseFailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            for dataset in order:
                session.labels_for_dataset_antigravity(dataset)
            with self.assertRaisesRegex(OSError, "close boom"):
                session.close_antigravity()
            self.assertEqual(session.close_state_antigravity(), "close_failed")
            self.assertEqual(connection.close_calls, 1)
            with self.assertRaisesRegex(OSError, "close boom"):
                session.close_antigravity()
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_query_oserror_with_clean_cleanup_is_terminal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class QueryOSErrorConnectionAntigravity:
            def __init__(self) -> None: self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()): raise OSError("query io")  # type: ignore[no-untyped-def]
            def close(self) -> None: self.close_calls += 1
        connection = QueryOSErrorConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(OSError, "query io"): session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(session.close_state_antigravity(), "terminal")
            self.assertIsNone(session.close_error_antigravity())
            self.assertIsInstance(session.terminal_error_antigravity(), OSError)
            with self.assertRaisesRegex(OSError, "query io"): session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_row_failure_and_cleanup_failure_are_recorded_separately_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class BadRowCloseFailingConnectionAntigravity:
            def __init__(self) -> None: self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()): return ((0, "bad"),)  # type: ignore[no-untyped-def]
            def close(self) -> None:
                self.close_calls += 1
                raise OSError("cleanup io")
        connection = BadRowCloseFailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(ValueError, "TEST label row is invalid"): session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(session.close_state_antigravity(), "close_failed")
            self.assertIsInstance(session.terminal_error_antigravity(), ValueError)
            self.assertIsInstance(session.close_error_antigravity(), OSError)
            with self.assertRaisesRegex(ValueError, "TEST label row is invalid"): session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_session_rejects_empty_bad_index_bad_label_and_iterator_rows_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        cases = {
            "empty": (), "bool": ((True, "0"),), "gap": ((1, "0"),), "label": ((0, b"0"),),
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                _, authority = self._install_prediction_module_antigravity()
                class ConnectionAntigravity:
                    def __init__(self): self.close_calls = 0
                    def execute(self, sql, parameters=()): return rows
                    def close(self): self.close_calls += 1
                connection = ConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
                with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))):
                    session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
                    with self.assertRaises(ValueError): session.labels_for_dataset_antigravity("Crop_normal_1")
                    self.assertEqual(session.close_state_antigravity(), "terminal")
                    self.assertEqual(connection.close_calls, 1)
                    with self.assertRaises(ValueError): session.labels_for_dataset_antigravity("Crop_normal_1")
        _, authority = self._install_prediction_module_antigravity()
        class IteratorFailureConnectionAntigravity:
            def __init__(self): self.close_calls = 0
            def execute(self, sql, parameters=()):
                def rows():
                    yield (0, "0")
                    raise RuntimeError("iterator boom")
                return rows()
            def close(self): self.close_calls += 1
        connection = IteratorFailureConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
        with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))):
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(RuntimeError, "iterator boom"): session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(connection.close_calls, 1)

    def test_session_wrong_order_and_total_under_over_are_terminal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class SimpleConnectionAntigravity:
            def __init__(self): self.close_calls = 0
            def execute(self, sql, parameters=()): return ((0, "0"),)
            def close(self): self.close_calls += 1
        connection = SimpleConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
        with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))):
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaises(ValueError): session.labels_for_dataset_antigravity("wrong_antigravity")
            self.assertEqual(connection.close_calls, 1)
        for count in (72300 - 1, 72300 + 1):
            with self.subTest(count=count):
                order = tuple(f"count_target_{index}_antigravity" for index in range(48))
                _, authority = self._install_prediction_module_antigravity(order)
                class CountConnectionAntigravity:
                    def __init__(self): self.calls = 0; self.close_calls = 0
                    def execute(self, sql, parameters=()):
                        self.calls += 1
                        size = 1 if self.calls < 48 else count - 47
                        return tuple((index, "0") for index in range(size))
                    def close(self): self.close_calls += 1
                connection = CountConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
                with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))):
                    session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
                    for dataset in order[:-1]: session.labels_for_dataset_antigravity(dataset)
                    with self.assertRaisesRegex(ValueError, "row count"): session.labels_for_dataset_antigravity(order[-1])
                    self.assertEqual(session.close_state_antigravity(), "terminal")
                    self.assertEqual(connection.close_calls, 1)

    def test_descriptor_release_failure_is_cleanup_failure_not_terminal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        order = tuple(f"release_target_{index}_antigravity" for index in range(48))
        _, authority = self._install_prediction_module_antigravity(order)
        class CompleteConnectionAntigravity:
            def __init__(self): self.calls = 0; self.close_calls = 0
            def execute(self, sql, parameters=()):
                self.calls += 1
                size = 1518 if self.calls == 48 else 1506
                return tuple((index, "0") for index in range(size))
            def close(self): self.close_calls += 1
        connection = CompleteConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
        original_close = sources.os.close
        def failing_close(fd: int) -> None:
            if fd == descriptor: raise OSError("descriptor release boom")
            original_close(fd)
        with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))), patch.object(sources, "_assert_database_unchanged_antigravity"), patch.object(sources.os, "close", side_effect=failing_close):
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            for dataset in order: session.labels_for_dataset_antigravity(dataset)
            with self.assertRaisesRegex(OSError, "descriptor release boom"): session.close_antigravity()
            self.assertEqual(session.close_state_antigravity(), "close_failed")
            self.assertIsNone(session.terminal_error_antigravity())
            self.assertIsInstance(session.close_error_antigravity(), OSError)
            with self.assertRaisesRegex(OSError, "descriptor release boom"): session.labels_for_dataset_antigravity(order[0])
            self.assertEqual(connection.close_calls, 1)

    def test_public_label_session_postverify_drift_is_terminal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        order = tuple(f"session_target_{index}_antigravity" for index in range(48))
        _, authority = self._install_prediction_module_antigravity(order)
        with tempfile.TemporaryDirectory(prefix="exp449_label_drift_antigravity_") as temporary:
            path = Path(temporary) / "labels_antigravity.db"
            connection = sqlite3.connect(path)
            connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE instances (dataset_id INTEGER, split TEXT, instance_index INTEGER, label TEXT, values_blob BLOB);")
            connection.executemany("INSERT INTO datasets VALUES (?,?)", tuple((index + 1, name) for index, name in enumerate(order)))
            rows = []
            for dataset_index in range(48):
                count = 1518 if dataset_index == 47 else 1506
                rows.extend((dataset_index + 1, "TEST", index, "0", b"x") for index in range(count))
            connection.executemany("INSERT INTO instances VALUES (?,?,?,?,?)", rows)
            connection.commit(); connection.close()
            path_patch, sha_patch, inode_patch = self._database_constants_antigravity(sources, path)
            with path_patch, sha_patch, inode_patch:
                session = sources.open_test_labels_once_antigravity(authority)
                for dataset in order: session.labels_for_dataset_antigravity(dataset)
                connection = sqlite3.connect(path); connection.execute("PRAGMA user_version=2"); connection.commit(); connection.close()
                with self.assertRaisesRegex(ValueError, "changed during read"): session.close_antigravity()
                self.assertEqual(session.close_state_antigravity(), "terminal")
                self.assertIsInstance(session.terminal_error_antigravity(), ValueError)

    def test_clean_48_query_session_closes_and_blocks_reuse_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        order = tuple(f"clean_target_{index}_antigravity" for index in range(48))
        _, authority = self._install_prediction_module_antigravity(order)
        class CompleteConnectionAntigravity:
            def __init__(self): self.calls = 0; self.close_calls = 0
            def execute(self, sql, parameters=()):
                self.calls += 1
                size = 1518 if self.calls == 48 else 1506
                return tuple((index, "0") for index in range(size))
            def close(self): self.close_calls += 1
        connection = CompleteConnectionAntigravity(); descriptor = os.open("/dev/null", os.O_RDONLY)
        with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0, 0, 0, 1, 0, 0, 0))), patch.object(sources, "_assert_database_unchanged_antigravity"):
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            for dataset in order: session.labels_for_dataset_antigravity(dataset)
            session.close_antigravity()
            self.assertEqual(session.close_state_antigravity(), "closed")
            self.assertEqual(connection.close_calls, 1)
            with self.assertRaisesRegex(ValueError, "closed"): session.labels_for_dataset_antigravity(order[0])
            session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
