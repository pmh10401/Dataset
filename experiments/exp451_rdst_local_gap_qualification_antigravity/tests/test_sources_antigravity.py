"""Synthetic-only tests for the Task 4 value/label source firewall."""

from __future__ import annotations

import sqlite3
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
        dependencies = sources.LabelSourceDependenciesAntigravity(lambda: fixture, lambda item: tuple(identity[2] for identity in item.dataset_identities), guard)
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
        dependencies = sources.LabelSourceDependenciesAntigravity(opener, lambda _item: tuple(["x"] * 48), sources.LabelSessionGuardAntigravity())
        with self.assertRaises(ValueError):
            sources._open_ordered_label_session_with_dependencies_antigravity(self._authority_antigravity(), dependencies)
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
