from collections import Counter
from pathlib import Path
import sqlite3
import unittest
import numpy as np

from services.time_series_anomaly_diagnosis.model_resources import (
    ModelResources,
    load_model_resources,
)
from services.time_series_anomaly_diagnosis.dataset_repository import (
    PreparedDatasetValues,
    get_dataset_catalog,
    load_dataset_values,
    load_series_for_chart,
    load_test_labels,
    open_source_database,
    prepare_dataset_values,
    resample_series,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import (
    load_target_bundle_antigravity as load_verified_standard_values,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    prepare_dataset_values_antigravity as load_verified_recovery_values,
)

SOURCE_DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
RESOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "resources"


class DataAccessRealDatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = load_model_resources(RESOURCE_DIRECTORY)

    def test_model_resources_contract(self) -> None:
        self.assertIsInstance(self.resources, ModelResources)
        self.assertEqual(len(self.resources.route_by_dataset), 1117)
        self.assertEqual(len(self.resources.recovery_datasets), 155)
        self.assertTrue(self.resources.recovery_datasets <= self.resources.route_by_dataset.keys())
        allowed_methods = {"minirocket_local_gap", "hydra_local_gap", "multirocket_local_gap"}
        self.assertTrue(set(self.resources.route_by_dataset.values()) <= allowed_methods)

    def test_source_database_connection_rejects_writes(self) -> None:
        connection = open_source_database(SOURCE_DATABASE)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE read_only_probe (id INTEGER)")
        finally:
            connection.close()

    def test_real_catalog_partitions_every_database_dataset(self) -> None:
        connection = open_source_database(SOURCE_DATABASE)
        try:
            catalog = get_dataset_catalog(connection, self.resources)
        finally:
            connection.close()

        self.assertEqual(len(catalog), 1119)
        counts = Counter(entry["availability"] for entry in catalog)
        self.assertEqual(
            counts,
            {
                "standard": 801,
                "recovery_numba": 155,
                "insufficient_training_data": 161,
                "missing_verified_route": 2,
            },
        )
        unavailable = {entry["dataset_name"] for entry in catalog if not entry["analysis_available"]}
        self.assertIn("CornellWhaleChallenge", unavailable)
        self.assertIn("Phoneme_normal_39", unavailable)

        catalog_by_name = {entry["dataset_name"]: entry for entry in catalog}
        self.assertEqual(catalog_by_name["CornellWhaleChallenge"]["availability"], "missing_verified_route")
        self.assertFalse(catalog_by_name["CornellWhaleChallenge"]["analysis_available"])
        self.assertEqual(catalog_by_name["Phoneme_normal_39"]["availability"], "missing_verified_route")
        self.assertFalse(catalog_by_name["Phoneme_normal_39"]["analysis_available"])
        self.assertEqual(catalog_by_name["Phoneme_normal_10"]["availability"], "insufficient_training_data")
        self.assertFalse(catalog_by_name["Phoneme_normal_10"]["analysis_available"])

        required_keys = {
            "dataset_name",
            "method_name",
            "train_count",
            "test_count",
            "series_length",
            "availability",
            "analysis_available",
        }
        for entry in catalog:
            self.assertEqual(set(entry.keys()), required_keys)
            self.assertIsInstance(entry["dataset_name"], str)
            self.assertIsInstance(entry["train_count"], int)
            self.assertIsInstance(entry["test_count"], int)
            self.assertIsInstance(entry["series_length"], int)
            self.assertIsInstance(entry["availability"], str)
            self.assertIsInstance(entry["analysis_available"], bool)

    def test_catalog_query_does_not_read_labels_or_blobs(self) -> None:
        forbidden_accessed: list[tuple[str, str]] = []

        def authorizer(action: int, arg1: str | None, arg2: str | None, _db: str | None, _trig: str | None) -> int:
            # SQLITE_READ = 20
            if action == 20 and arg1 == "instances" and arg2 in ("label", "values_blob"):
                forbidden_accessed.append((arg1, arg2))
            return sqlite3.SQLITE_OK

        connection = open_source_database(SOURCE_DATABASE)
        try:
            connection.set_authorizer(authorizer)
            catalog = get_dataset_catalog(connection, self.resources)
            self.assertEqual(len(catalog), 1119)
        finally:
            connection.close()
        self.assertEqual(forbidden_accessed, [])

    def test_standard_values_match_the_verified_loader(self) -> None:
        connection = open_source_database(SOURCE_DATABASE)
        try:
            prepared = load_dataset_values(connection, "ACSF1_normal_0", "standard")
        finally:
            connection.close()

        verified = load_verified_standard_values("ACSF1_normal_0")
        self.assertEqual(prepared.dataset_name, "ACSF1_normal_0")
        self.assertEqual(prepared.preparation_mode, "standard")
        self.assertEqual(len(prepared.test_indices), len(verified.test_values))
        np.testing.assert_array_equal(prepared.train_values, verified.train_normal)
        np.testing.assert_array_equal(prepared.test_values, verified.test_values)

    def test_recovery_values_match_the_verified_loader(self) -> None:
        connection = open_source_database(SOURCE_DATABASE)
        try:
            prepared = load_dataset_values(connection, "AllGestureWiimoteX_normal_1", "recovery_numba")
            universal = prepare_dataset_values(connection, "AllGestureWiimoteX_normal_1")
            verified = load_verified_recovery_values(connection, "AllGestureWiimoteX_normal_1")
        finally:
            connection.close()

        self.assertEqual(prepared.dataset_name, "AllGestureWiimoteX_normal_1")
        self.assertEqual(prepared.preparation_mode, "recovery_numba")
        self.assertEqual(prepared.test_indices, verified.test_indices)
        np.testing.assert_array_equal(prepared.train_values, verified.train_rows)
        np.testing.assert_array_equal(prepared.test_values, verified.test_rows)
        self.assertEqual(universal.target_length, verified.target_length)
        self.assertEqual(universal.test_indices, verified.test_indices)
        np.testing.assert_array_equal(universal.train_values, verified.train_rows)
        np.testing.assert_array_equal(universal.test_values, verified.test_rows)


class DataAccessSyntheticDatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                train_total_count INTEGER,
                test_total_count INTEGER,
                series_length INTEGER
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER,
                split TEXT,
                instance_index INTEGER,
                label TEXT,
                values_blob BLOB
            )
            """
        )
        self.conn.execute(
            "INSERT INTO datasets VALUES (1, 'Synth_Dataset', 2, 2, 4)"
        )
        row0 = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
        row1 = np.array([2.0, 3.0, 4.0, 5.0], dtype="<f4").tobytes()
        trow0 = np.array([5.0, 6.0, 7.0, 8.0], dtype="<f4").tobytes()
        trow1 = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f4").tobytes()

        # Synthetic labels: '0' for normal, '1' for anomaly
        self.conn.execute("INSERT INTO instances VALUES (1, 1, 'TRAIN', 0, '0', ?)", (row0,))
        self.conn.execute("INSERT INTO instances VALUES (2, 1, 'TRAIN', 1, '0', ?)", (row1,))
        self.conn.execute("INSERT INTO instances VALUES (3, 1, 'TEST', 0, '0', ?)", (trow0,))
        self.conn.execute("INSERT INTO instances VALUES (4, 1, 'TEST', 1, '1', ?)", (trow1,))
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()

    def add_dataset(
        self,
        dataset_id: int,
        name: str,
        train: list[tuple[int, str, object]],
        test: list[tuple[int, str, object]],
    ) -> None:
        self.conn.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?)",
            (dataset_id, name, len(train), len(test), 0),
        )
        next_id = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM instances").fetchone()[0]
        for split, rows in (("TRAIN", train), ("TEST", test)):
            for instance_index, label, values in rows:
                blob = values if not isinstance(values, list) else np.array(values, dtype="<f4").tobytes()
                self.conn.execute(
                    "INSERT INTO instances VALUES (?, ?, ?, ?, ?, ?)",
                    (next_id, dataset_id, split, instance_index, label, blob),
                )
                next_id += 1
        self.conn.commit()

    def test_prepare_values_is_name_independent_and_uses_ragged_lower_median(self) -> None:
        train = [
            (0, "0", [0, 1, 2, 3]),
            (1, "0", [0, 1, 2, 3, 4, 5, 6, 7]),
            (2, "0", [1, 2, 3, 4, 5, 6, 7, 8]),
            (3, "0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (4, "0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ]
        test = [(0, "0", [2, 4, 6, 8]), (2, "1", [1, 2, 4, 8, 16])]
        self.add_dataset(2, "Dataset_A", train, test)
        self.add_dataset(3, "Renamed_Copy", train, test)

        first = prepare_dataset_values(self.conn, "Dataset_A")
        renamed = prepare_dataset_values(self.conn, "Renamed_Copy")

        self.assertEqual(first.preparation_mode, "value_only_numba")
        self.assertEqual(first.target_length, 8)
        self.assertEqual(first.test_indices, (0, 2))
        np.testing.assert_array_equal(first.train_values, renamed.train_values)
        np.testing.assert_array_equal(first.test_values, renamed.test_values)
        self.assertEqual(tuple(map(len, first.raw_train_rows)), (4, 8, 8, 10, 12))
        for actual, (_, _, expected) in zip(first.raw_train_rows, train, strict=True):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float64)
            self.assertFalse(actual.flags.writeable)

    def test_prepare_values_does_not_depend_on_test_labels(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        test = [(0, "0", [1, 4, 9]), (1, "1", [2, 5, 10])]
        self.add_dataset(2, "Label_Free", train, test)
        before = prepare_dataset_values(self.conn, "Label_Free")
        self.conn.execute("UPDATE instances SET label='changed' WHERE dataset_id=2 AND split='TEST'")
        after = prepare_dataset_values(self.conn, "Label_Free")
        np.testing.assert_array_equal(before.train_values, after.train_values)
        np.testing.assert_array_equal(before.test_values, after.test_values)
        self.assertEqual(before.test_indices, after.test_indices)

    def test_prepare_values_excludes_nonzero_train_labels(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        train.append((5, "1", b"bad"))
        self.add_dataset(2, "Normal_Only", train, [(0, "1", [1, 2, 4])])
        prepared = prepare_dataset_values(self.conn, "Normal_Only")
        self.assertEqual(prepared.train_values.shape, (5, 3))

    def test_prepare_values_rejects_invalid_blob(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        for dataset_id, blob in enumerate((b"", b"bad", "not bytes"), 2):
            name = f"Invalid_Blob_{dataset_id}"
            self.add_dataset(dataset_id, name, train, [(0, "1", blob)])
            with self.assertRaisesRegex(ValueError, "little-endian float32"):
                prepare_dataset_values(self.conn, name)

    def test_prepare_values_excludes_nonfinite_rows_and_preserves_test_indices(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        train.extend([(5, "0", [1, np.nan, 3]), (6, "0", [1, np.inf, 3])])
        test = [
            (0, "0", [1, np.nan, 3]),
            (2, "1", [2, 4, 8]),
            (4, "1", [1, np.inf, 3]),
        ]
        self.add_dataset(2, "Finite_Only", train, test)
        prepared = prepare_dataset_values(self.conn, "Finite_Only")
        self.assertEqual(prepared.train_values.shape, (5, 3))
        self.assertEqual(prepared.test_indices, (2,))

    def test_prepare_values_requires_five_retained_normal_train_rows(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(4)]
        self.add_dataset(2, "Too_Small", train, [(0, "1", [1, 2, 4])])
        with self.assertRaisesRegex(ValueError, "at least five"):
            prepare_dataset_values(self.conn, "Too_Small")

    def test_prepare_values_requires_a_retained_test_row(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        self.add_dataset(2, "No_Finite_Test", train, [(0, "1", [1, np.nan, 4])])
        with self.assertRaisesRegex(ValueError, "at least one"):
            prepare_dataset_values(self.conn, "No_Finite_Test")

    def test_prepare_values_rejects_a_constant_retained_row(self) -> None:
        train = [(index, "0", [index, index + 1, index + 3]) for index in range(5)]
        self.add_dataset(2, "Constant_Row", train, [(0, "1", [3, 3, 3])])
        with self.assertRaisesRegex(ValueError, "non-constant"):
            prepare_dataset_values(self.conn, "Constant_Row")

    def test_resample_series_copies_equal_length_and_handles_target_one(self) -> None:
        values = np.array([2.0, 4.0, 8.0], dtype=np.float64)
        copied = resample_series(values, 3)
        np.testing.assert_array_equal(copied, values)
        self.assertIsNot(copied, values)
        np.testing.assert_array_equal(resample_series(values, 1), np.array([2.0]))
        for invalid, target in ((np.array([]), 1), (np.array([[1.0]]), 1), (values, 0)):
            with self.assertRaises(ValueError):
                resample_series(invalid, target)

    def test_synthetic_load_dataset_values_standard(self) -> None:
        prepared = load_dataset_values(self.conn, "Synth_Dataset", "standard")
        self.assertEqual(prepared.dataset_name, "Synth_Dataset")
        self.assertEqual(prepared.preparation_mode, "standard")
        self.assertEqual(prepared.test_indices, (0, 1))
        np.testing.assert_array_equal(prepared.raw_train_rows[0], [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(prepared.raw_train_rows[0].flags.writeable)
        for row in prepared.train_values:
            self.assertAlmostEqual(float(row.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(row.std()), 1.0, places=6)
        for row in prepared.test_values:
            self.assertAlmostEqual(float(row.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(row.std()), 1.0, places=6)

    def test_recovery_forwards_authoritative_original_train_rows(self) -> None:
        train = [(i, "0", [i, i + 1, i + 3 + j]) for i, j in enumerate(range(5))]
        self.add_dataset(2, "Raw_Recovery", train, [(0, "1", [1, 2, 4])])
        universal = prepare_dataset_values(self.conn, "Raw_Recovery")
        recovery = load_dataset_values(self.conn, "Raw_Recovery", "recovery_numba")
        self.assertEqual(len(recovery.raw_train_rows), 5)
        for left, right in zip(recovery.raw_train_rows, universal.raw_train_rows, strict=True):
            np.testing.assert_array_equal(left, right)
            self.assertIsInstance(left.base, bytes)
            self.assertFalse(left.flags.writeable)

    def test_five_position_prepared_construction_remains_compatible(self) -> None:
        prepared = PreparedDatasetValues("legacy", np.zeros((5, 2)), np.zeros((1, 2)), (0,), "standard")
        self.assertIsNone(prepared.raw_train_rows)

    def test_synthetic_load_dataset_values_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            load_dataset_values(self.conn, "Synth_Dataset", "unsupported_mode")

    def test_synthetic_load_dataset_values_missing_dataset(self) -> None:
        with self.assertRaises(ValueError):
            load_dataset_values(self.conn, "NonExistent", "standard")

    def test_synthetic_load_test_labels_success(self) -> None:
        labels = load_test_labels(self.conn, "Synth_Dataset", (0, 1))
        self.assertEqual(labels, {0: 0, 1: 1})
        labels_rev = load_test_labels(self.conn, "Synth_Dataset", (1, 0))
        self.assertEqual(list(labels_rev.keys()), [1, 0])
        self.assertEqual(labels_rev[1], 1)
        self.assertEqual(labels_rev[0], 0)

    def test_synthetic_load_test_labels_missing_index(self) -> None:
        with self.assertRaises(ValueError):
            load_test_labels(self.conn, "Synth_Dataset", (0, 99))

    def test_synthetic_load_test_labels_invalid_label_value(self) -> None:
        self.conn.execute("UPDATE instances SET label='invalid' WHERE split='TEST' AND instance_index=0")
        self.conn.commit()
        with self.assertRaises(ValueError):
            load_test_labels(self.conn, "Synth_Dataset", (0, 1))

    def test_synthetic_load_series_for_chart_standard(self) -> None:
        series = load_series_for_chart(self.conn, "Synth_Dataset", 0, "standard", 4)
        self.assertIsInstance(series, list)
        self.assertEqual(len(series), 4)
        arr = np.array(series)
        self.assertAlmostEqual(float(arr.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(arr.std()), 1.0, places=6)

    def test_synthetic_load_series_for_chart_recovery(self) -> None:
        series = load_series_for_chart(self.conn, "Synth_Dataset", 0, "recovery_numba", 6)
        self.assertIsInstance(series, list)
        self.assertEqual(len(series), 6)
        arr = np.array(series)
        self.assertAlmostEqual(float(arr.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(arr.std()), 1.0, places=6)

    def test_synthetic_load_series_for_chart_unknown_index(self) -> None:
        with self.assertRaises(ValueError):
            load_series_for_chart(self.conn, "Synth_Dataset", 99, "standard", 4)

    def test_synthetic_load_series_for_chart_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            load_series_for_chart(self.conn, "Synth_Dataset", 0, "standard", 10)

    def test_synthetic_load_series_for_chart_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            load_series_for_chart(self.conn, "Synth_Dataset", 0, "unsupported_mode", 4)

    def test_synthetic_load_series_for_chart_constant_series(self) -> None:
        const_blob = np.array([3.0, 3.0, 3.0, 3.0], dtype="<f4").tobytes()
        self.conn.execute("UPDATE instances SET values_blob=? WHERE split='TEST' AND instance_index=0", (const_blob,))
        self.conn.commit()
        with self.assertRaises(ValueError):
            load_series_for_chart(self.conn, "Synth_Dataset", 0, "standard", 4)


if __name__ == "__main__":
    unittest.main()
