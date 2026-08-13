"""Synthetic-only tests for the Task 4 value/label source firewall."""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


class SourcesAntigravityTest(unittest.TestCase):
    def _authority_antigravity(self):  # type: ignore[no-untyped-def]
        from experiments.exp451_rdst_local_gap_qualification_antigravity.contracts_antigravity import FrozenPredictionAuthorityAntigravity
        names = tuple(f"Crop_{index:02d}" for index in range(48))
        return FrozenPredictionAuthorityAntigravity("a" * 40, tuple((i, "Crop", name) for i, name in enumerate(names)), tuple([1500] * 47 + [1800]), tuple(() for _ in names))

    def test_value_fixture_is_normalized_immutable_and_has_no_label_surface_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        connection = sqlite3.connect(":memory:")
        connection.executescript("CREATE TABLE datasets(id INTEGER, name TEXT); CREATE TABLE instances(dataset_id INTEGER, split TEXT, instance_index INTEGER, values_blob BLOB, label TEXT); INSERT INTO datasets VALUES(1,'Crop_00');")
        rows = [(1, "TRAIN", 0, np.array([1, 2, 4], dtype="<f4").tobytes(), "1"), (1, "TEST", 0, np.array([2, 4, 8], dtype="<f4").tobytes(), "0")]
        connection.executemany("INSERT INTO instances VALUES(?,?,?,?,?)", rows)
        with patch.object(sources, "verify_exp448_predecessor_antigravity") as verify:
            verify.return_value.target_dataset_names = ("Crop_00",)
            verify.return_value.target_dataset_order = (0,)
            verify.return_value.target_baseline_rows = (Mock(family="Crop"),)
            bundle = sources._load_target_bundle_with_connection_antigravity("Crop_00", lambda: connection)
        self.assertEqual(tuple(bundle.__dataclass_fields__), ("dataset_order", "family", "dataset", "train_normal", "test_values", "target_length"))
        self.assertNotIn("label", repr(bundle).lower())
        self.assertFalse(bundle.train_normal.flags.writeable)
        self.assertAlmostEqual(float(bundle.train_normal.mean()), 0.0)

    def test_value_rows_reject_gap_order_bool_blob_nonfinite_variance_and_length_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        good = np.array([1, 2], dtype="<f4").tobytes()
        attacks = (
            (("TRAIN", True, good), ("TEST", 0, good)),
            (("TRAIN", 1, good), ("TEST", 0, good)),
            (("TEST", 0, good), ("TRAIN", 0, good)),
            (("TRAIN", 0, b"x"), ("TEST", 0, good)),
            (("TRAIN", 0, np.array([1, np.nan], dtype="<f4").tobytes()), ("TEST", 0, good)),
            (("TRAIN", 0, np.array([1, 1], dtype="<f4").tobytes()), ("TEST", 0, good)),
            (("TRAIN", 0, good), ("TEST", 0, np.array([1, 2, 3], dtype="<f4").tobytes())),
        )
        for rows in attacks:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                sources._bundle_from_rows_antigravity(0, "Crop", "Crop_00", rows)

    def test_private_label_seam_validates_before_opener_and_never_calls_production_factory_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        authority = self._authority_antigravity()
        production = Mock(side_effect=AssertionError("production opener called"))
        fixture = sqlite3.connect(":memory:")
        fixture.executescript("CREATE TABLE datasets(id INTEGER, name TEXT); CREATE TABLE instances(dataset_id INTEGER, split TEXT, instance_index INTEGER, values_blob BLOB, label TEXT);")
        lengths = authority.target_lengths
        for dataset_id, ((_, _, name), length) in enumerate(zip(authority.dataset_identities, lengths, strict=True), 1):
            fixture.execute("INSERT INTO datasets VALUES(?,?)", (dataset_id, name))
            fixture.executemany("INSERT INTO instances VALUES(?,'TEST',?,X'00000000',?)", ((dataset_id, i, str(i % 2)) for i in range(length)))
        guard = sources.LabelSessionGuardAntigravity()
        dependencies = sources.LabelSourceDependenciesAntigravity(lambda: fixture, sources.ValidatedPredictionAuthorityAntigravity.from_authority_antigravity, guard)
        with patch.object(sources, "PRODUCTION_LABEL_CONNECTION_FACTORY_PATH", production):
            snapshot = sources._open_ordered_label_session_with_dependencies_antigravity(authority, dependencies)
        production.assert_not_called()
        self.assertEqual(snapshot.row_count, 72_300)
        self.assertEqual(snapshot.session_count, 1)
        self.assertEqual(snapshot.dataset_order, tuple(identity[2] for identity in authority.dataset_identities))
        self.assertTrue(all(not labels.flags.writeable for labels in snapshot.labels))
        with self.assertRaises(ValueError):
            sources._open_ordered_label_session_with_dependencies_antigravity(authority, dependencies)

    def test_invalid_prediction_fails_before_fixture_opener_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        opener = Mock()
        dependencies = sources.LabelSourceDependenciesAntigravity(opener, lambda _item: object(), sources.LabelSessionGuardAntigravity())
        with self.assertRaises(ValueError):
            sources._open_ordered_label_session_with_dependencies_antigravity(self._authority_antigravity(), dependencies)
        opener.assert_not_called()

    def test_production_validator_delegates_to_task5_and_missing_module_fails_before_opener_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        opener = Mock()
        with patch.object(sources, "PRODUCTION_LABEL_CONNECTION_FACTORY_PATH", opener), patch.object(sources.importlib, "import_module", side_effect=ImportError("absent")), self.assertRaises(ValueError):
            sources.open_ordered_label_session_antigravity(self._authority_antigravity())
        opener.assert_not_called()

    def test_private_validator_result_binds_exact_canonical_authority_before_open_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        authority = self._authority_antigravity()
        validated = sources.ValidatedPredictionAuthorityAntigravity.from_authority_antigravity(authority)
        opener = Mock()
        attacks = (
            {"reviewed_commit": "b" * 40},
            {"canonical_sha256": "0" * 64},
            {"target_lengths": tuple([1499, 1501] + list(authority.target_lengths[2:]))},
            {"dataset_identities": ((0, "Other", authority.dataset_identities[0][2]),) + authority.dataset_identities[1:]},
            {"candidate_indices": ((0,),) + authority.candidate_indices[1:]},
        )
        for change in attacks:
            with self.subTest(change=change):
                bad = sources.dataclasses.replace(validated, **change)
                deps = sources.LabelSourceDependenciesAntigravity(opener, lambda _item, value=bad: value, sources.LabelSessionGuardAntigravity())
                with self.assertRaises(ValueError):
                    sources._open_ordered_label_session_with_dependencies_antigravity(authority, deps)
        opener.assert_not_called()

    def test_values_production_postverify_runs_once_on_read_error_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        connection = Mock()
        connection.execute.side_effect = RuntimeError("query")
        descriptor = os.open(__file__, os.O_RDONLY)
        verify_after = Mock()
        with patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, (0,) * 7)), patch.object(sources, "_connection_from_descriptor_antigravity", return_value=connection), patch.object(sources, "_assert_database_unchanged_antigravity", verify_after), patch.object(sources, "verify_exp448_predecessor_antigravity") as predecessor, self.assertRaisesRegex(RuntimeError, "query"):
            predecessor.return_value.target_dataset_names = ("Crop_00",)
            predecessor.return_value.target_dataset_order = (0,)
            predecessor.return_value.target_baseline_rows = (Mock(family="Crop"),)
            sources.load_target_bundle_antigravity("Crop_00")
        verify_after.assert_called_once()
        connection.close.assert_called_once()

    def test_values_authorizer_denies_labels_schema_functions_and_mutation_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        allowed = frozenset({("instances", "values_blob")})
        authorizer = sources._authorizer_antigravity(allowed)
        self.assertEqual(authorizer(sqlite3.SQLITE_READ, "instances", "values_blob", "main", None), sqlite3.SQLITE_OK)
        for action, table, column in (
            (sqlite3.SQLITE_READ, "instances", "label"),
            (sqlite3.SQLITE_READ, "sqlite_master", "name"),
            (sqlite3.SQLITE_FUNCTION, None, "random"),
            (sqlite3.SQLITE_UPDATE, "instances", "values_blob"),
            (sqlite3.SQLITE_PRAGMA, None, "table_info"),
            (sqlite3.SQLITE_ATTACH, None, None),
        ):
            self.assertEqual(authorizer(action, table, column, "main", None), sqlite3.SQLITE_DENY)

    def test_database_verifier_rejects_symlink_hardlink_inode_and_sha_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        with tempfile.TemporaryDirectory(prefix="exp451_source_antigravity_") as temporary:
            path = Path(temporary) / "database_antigravity.db"
            path.write_bytes(b"database")
            info = path.stat()
            constants = {"_DATABASE_ANTIGRAVITY": path, "_DATABASE_INODE_ANTIGRAVITY": info.st_ino, "_DATABASE_SIZE_ANTIGRAVITY": info.st_size, "_DATABASE_SHA256_ANTIGRAVITY": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
            with patch.multiple(sources, **constants):
                descriptor, _identity = sources._verify_database_antigravity()
                os.close(descriptor)
            for key, value in (("_DATABASE_INODE_ANTIGRAVITY", info.st_ino + 1), ("_DATABASE_SHA256_ANTIGRAVITY", "0" * 64)):
                with self.subTest(key=key), patch.multiple(sources, **{**constants, key: value}), self.assertRaises(ValueError):
                    sources._verify_database_antigravity()
            hardlink = Path(temporary) / "hardlink_antigravity.db"
            os.link(path, hardlink)
            with patch.multiple(sources, **constants), self.assertRaises(ValueError):
                sources._verify_database_antigravity()
            hardlink.unlink()
            symlink = Path(temporary) / "symlink_antigravity.db"
            symlink.symlink_to(path)
            with patch.multiple(sources, **{**constants, "_DATABASE_ANTIGRAVITY": symlink}), self.assertRaises((ValueError, OSError)):
                sources._verify_database_antigravity()

    def test_label_query_and_close_combined_failure_is_terminal_and_closes_once_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        authority = self._authority_antigravity()
        connection = Mock()
        connection.execute.side_effect = RuntimeError("iterator/query")
        connection.close.side_effect = RuntimeError("close")
        dependencies = sources.LabelSourceDependenciesAntigravity(lambda: connection, sources.ValidatedPredictionAuthorityAntigravity.from_authority_antigravity, sources.LabelSessionGuardAntigravity())
        with self.assertRaises(ExceptionGroup) as captured:
            sources._open_ordered_label_session_with_dependencies_antigravity(authority, dependencies)
        self.assertEqual([str(error) for error in captured.exception.exceptions], ["iterator/query", "close"])
        connection.close.assert_called_once()
        with self.assertRaises(ValueError):
            sources._open_ordered_label_session_with_dependencies_antigravity(authority, dependencies)

    def test_label_rows_reject_under_over_bool_gap_and_raw_nontext_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity import sources_antigravity as sources
        authority = self._authority_antigravity()
        attacks = (
            [(index, "0") for index in range(1499)],
            [(index, "0") for index in range(1501)],
            [(True, "0")] + [(index, "0") for index in range(1, 1500)],
            [(0, "0"), (2, "0")] + [(index, "0") for index in range(2, 1500)],
            [(0, 0)] + [(index, "0") for index in range(1, 1500)],
        )
        for rows in attacks:
            connection = Mock()
            connection.execute.return_value = iter(rows)
            deps = sources.LabelSourceDependenciesAntigravity(lambda: connection, sources.ValidatedPredictionAuthorityAntigravity.from_authority_antigravity, sources.LabelSessionGuardAntigravity())
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                sources._open_ordered_label_session_with_dependencies_antigravity(authority, deps)
            connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
