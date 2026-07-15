import unittest
import os
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from experiments.exp162.full_coverage_score_core import (
    spectral_features,
    shapelet_features,
    interval_features,
    robust_score_pair,
    source_config_hash,
    choose_shapelets,
    shapelet_distance_features,
)
from experiments.exp162.exp162_bridge import load_dataset_series_only, build_dataset_cache
from experiments.exp162.cache_io import write_cache_atomic, validate_cache


class TestExp162ScoreCacheCore(unittest.TestCase):
    def test_source_config_hash_stability(self):
        """Verify configuration hash is fixed and stable."""
        h1 = source_config_hash()
        h2 = source_config_hash()
        self.assertEqual(h1, h2)
        self.assertIsInstance(h1, str)
        self.assertEqual(len(h1), 64)

    def test_deterministic_feature_output(self):
        """Verify feature extraction is deterministic given the same inputs."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(5, 100))
        X_test = rng.normal(size=(3, 100))

        # Spectral features
        s1 = spectral_features(X_train)
        s2 = spectral_features(X_train)
        np.testing.assert_array_equal(s1, s2)

        # Shapelet features
        sh1_train, sh1_test = shapelet_features("test_ds", X_train, X_test, 100)
        sh2_train, sh2_test = shapelet_features("test_ds", X_train, X_test, 100)
        np.testing.assert_array_equal(sh1_train, sh2_train)
        np.testing.assert_array_equal(sh1_test, sh2_test)

        # Interval features
        i1 = interval_features(X_train)
        i2 = interval_features(X_train)
        np.testing.assert_array_equal(i1, i2)

    def test_six_finite_score_arrays(self):
        """Verify all outputs contain only finite values (no NaN or Inf)."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(10, 150))
        X_test = rng.normal(size=(5, 150))

        # Core extractions
        spec_train_f = spectral_features(X_train)
        spec_test_f = spectral_features(X_test)
        spec_tr, spec_te = robust_score_pair(spec_train_f, spec_test_f)

        shap_tr, shap_te = shapelet_features("test_ds", X_train, X_test, 150)
        shap_tr_score, shap_te_score = robust_score_pair(shap_tr, shap_te)

        int_train_f = interval_features(X_train)
        int_test_f = interval_features(X_test)
        int_tr, int_te = robust_score_pair(int_train_f, int_test_f)

        for arr in [spec_tr, spec_te, shap_tr_score, shap_te_score, int_tr, int_te]:
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertEqual(arr.ndim, 1)

    def test_train_only_fit_and_test_only_invariance(self):
        """
        Verify that changing test-only features does not alter train scores,
        and changing train features changes test scores.
        """
        rng = np.random.default_rng(42)
        X_train = rng.normal(size=(10, 100))
        X_test_1 = rng.normal(size=(5, 100))
        X_test_2 = rng.normal(size=(5, 100)) + 5.0

        train_feat = spectral_features(X_train)
        test_feat_1 = spectral_features(X_test_1)
        test_feat_2 = spectral_features(X_test_2)

        train_score_1, test_score_1 = robust_score_pair(train_feat, test_feat_1)
        train_score_2, test_score_2 = robust_score_pair(train_feat, test_feat_2)

        np.testing.assert_array_almost_equal(train_score_1, train_score_2)
        self.assertFalse(np.array_equal(test_score_1, test_score_2))

    def test_shapelet_resampling_is_capped_at_512(self):
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(3, 700)).astype(np.float32)
        x_test = rng.normal(size=(2, 700)).astype(np.float32)
        observed = {}

        def fake_choose(x_fit, num_shapelets, lengths, seed):
            observed["fit_length"] = x_fit.shape[1]
            return [np.zeros(8, dtype=np.float32) for _ in range(num_shapelets)]

        with patch(
            "experiments.exp162.full_coverage_score_core.choose_shapelets",
            side_effect=fake_choose,
        ):
            shapelet_features("long_ds", x_train, x_test, 700)

        self.assertEqual(observed["fit_length"], 512)

    def test_shapelet_distance_batching_preserves_scores(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(7, 40)).astype(np.float32)
        prototypes = choose_shapelets(x, 6, [8, 16], seed=99)
        small_batches = shapelet_distance_features(x, prototypes, batch_size=2)
        large_batches = shapelet_distance_features(x, prototypes, batch_size=20)
        np.testing.assert_allclose(small_batches, large_batches, rtol=0, atol=0)


class TestExp162BridgeAndCache(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.tmp_dir / "univariate_ts.db"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @patch("experiments.exp162.exp162_bridge.sqlite3")
    def test_values_only_sql_guard(self, mock_sqlite3):
        """
        Verify that all SQL queries run by the bridge never query the 'label' or 'anomaly' columns,
        acting as a forbidden label access guard.
        """
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock metadata query and values_blob results
        mock_cursor.fetchone.return_value = (100,)

        # We need mock raw arrays (10 elements each, floats)
        fake_blob = np.random.normal(size=100).astype(np.float32).tobytes()
        mock_cursor.fetchall.return_value = [(fake_blob,)]

        # Run load_dataset_series_only
        with patch("experiments.exp162.exp162_bridge.DB_PATH", str(self.test_db_path)):
            with patch("experiments.exp162.exp162_bridge.os.path.exists", return_value=True):
                load_dataset_series_only("test_dataset")

        # Capture and inspect all queries executed
        executed_queries = [call[0][0].upper() for call in mock_cursor.execute.call_args_list]
        for query in executed_queries:
            # Confirm no selection of label or anomaly
            self.assertNotIn("LABEL", query, f"Forbidden 'label' selection in query: {query}")
            self.assertNotIn("ANOMALY", query, f"Forbidden 'anomaly' selection in query: {query}")

    def test_required_and_forbidden_metadata_keys(self):
        """
        Verify that building cache constructs valid metadata with all required keys
        and absolutely no forbidden keys.
        """
        # Create a mock database file with one small dataset
        import sqlite3
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT, series_length INTEGER)")
        cursor.execute("CREATE TABLE instances (id INTEGER PRIMARY KEY, dataset_id INTEGER, split TEXT, instance_index INTEGER, values_blob BLOB, label INTEGER)")

        cursor.execute("INSERT INTO datasets VALUES (1, 'mock_ds', 20)")

        train_blob = np.random.normal(size=20).astype(np.float32).tobytes()
        test_blob = np.random.normal(size=20).astype(np.float32).tobytes()

        for i in range(5):
            cursor.execute("INSERT INTO instances VALUES (?, 1, 'TRAIN', ?, ?, 0)", (i+1, i, train_blob))
        for i in range(5):
            cursor.execute("INSERT INTO instances VALUES (?, 1, 'TEST', ?, ?, 0)", (i+6, i, test_blob))

        conn.commit()
        conn.close()

        with patch("experiments.exp162.exp162_bridge.DB_PATH", str(self.test_db_path)):
            payload, metadata = build_dataset_cache("mock_ds")

        # Verify required keys
        required_keys = {"dataset", "target_length", "source_config_hash", "seed", "train_rows", "test_rows", "contains_test_labels"}
        self.assertTrue(required_keys.issubset(metadata.keys()))
        self.assertFalse(metadata["contains_test_labels"])

        # Verify forbidden keys
        forbidden_keys = {"TP", "FP", "FN", "F1", "anomaly_count", "TEST label", "candidate_index", "threshold", "lane"}
        for key in forbidden_keys:
            self.assertNotIn(key, metadata)

        # Verify writing cache with forbidden keys raises ValueError
        bad_metadata = dict(metadata)
        bad_metadata["F1"] = 0.95
        cache_file = self.tmp_dir / "cache.npz"
        with self.assertRaises(ValueError):
            write_cache_atomic(cache_file, payload, bad_metadata)

    def test_variable_length_rows_are_aligned_from_train_median(self):
        import sqlite3

        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT, series_length INTEGER)")
        cursor.execute("CREATE TABLE instances (id INTEGER PRIMARY KEY, dataset_id INTEGER, split TEXT, instance_index INTEGER, values_blob BLOB, label INTEGER)")
        cursor.execute("INSERT INTO datasets VALUES (1, 'variable_ds', NULL)")
        rows = [
            (1, "TRAIN", 0, np.arange(8, dtype=np.float32)),
            (2, "TRAIN", 1, np.arange(12, dtype=np.float32)),
            (3, "TEST", 0, np.arange(9, dtype=np.float32)),
            (4, "TEST", 1, np.arange(15, dtype=np.float32)),
        ]
        for row_id, split, index, values in rows:
            cursor.execute(
                "INSERT INTO instances VALUES (?, 1, ?, ?, ?, 0)",
                (row_id, split, index, values.tobytes()),
            )
        conn.commit()
        conn.close()

        with patch("experiments.exp162.exp162_bridge.DB_PATH", str(self.test_db_path)):
            x_train, x_test, target_len = load_dataset_series_only("variable_ds")

        self.assertEqual(target_len, 10)
        self.assertEqual(x_train.shape, (2, 10))
        self.assertEqual(x_test.shape, (2, 10))

    def test_atomic_cache_validation_and_tamper_detection(self):
        """Verify atomic writing, valid loading, and validation failure upon tempering."""
        payload = {
            "spectral_train": np.random.normal(size=10),
            "spectral_test": np.random.normal(size=5),
            "shapelet_train": np.random.normal(size=10),
            "shapelet_test": np.random.normal(size=5),
            "interval_train": np.random.normal(size=10),
            "interval_test": np.random.normal(size=5),
        }
        metadata = {
            "dataset": "test_ds",
            "target_length": 100,
            "source_config_hash": "dummy_config_hash",
            "seed": 12345,
            "train_rows": 10,
            "test_rows": 5,
            "contains_test_labels": False
        }

        cache_file = self.tmp_dir / "test_cache.npz"
        write_cache_atomic(cache_file, payload, metadata)

        # 1. Validation passes under correct conditions
        self.assertTrue(validate_cache(cache_file, "dummy_config_hash"))

        # 2. Validation fails under config hash mismatch
        self.assertFalse(validate_cache(cache_file, "different_config_hash"))

        # 3. Validation fails if file is tampered (e.g. truncated or bad format)
        tampered_file = self.tmp_dir / "tampered_cache.npz"
        with open(tampered_file, "w") as f:
            f.write("corrupted data")
        self.assertFalse(validate_cache(tampered_file, "dummy_config_hash"))

        # 4. Metadata row counts must match all six score arrays.
        bad_lengths = dict(payload)
        bad_lengths["spectral_test"] = np.random.normal(size=4)
        mismatch_file = self.tmp_dir / "mismatch_cache.npz"
        write_cache_atomic(mismatch_file, bad_lengths, metadata)
        self.assertFalse(validate_cache(mismatch_file, "dummy_config_hash"))


if __name__ == "__main__":
    unittest.main()
