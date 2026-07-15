from __future__ import annotations
import unittest
import numpy as np
from experiments.exp164.train_normal_risk_core import (
    fold_plan,
    higher_threshold,
    source_candidate_flags,
    agreement_flags,
    wilson_upper_bound,
    n_train_bin,
)

class TestTrainNormalRiskCore(unittest.TestCase):
    def test_fold_plan_abstention(self):
        # n_train < 5 should return empty plan
        self.assertEqual(fold_plan(4, 20260715), [])
        self.assertEqual(fold_plan(0, 20260715), [])

    def test_n_train_bins_are_fixed_before_results(self):
        self.assertEqual(n_train_bin(4), "lt5")
        self.assertEqual(n_train_bin(5), "5_19")
        self.assertEqual(n_train_bin(20), "20_49")
        self.assertEqual(n_train_bin(50), "50_99")
        self.assertEqual(n_train_bin(100), "ge100")

    def test_fold_plan_loo(self):
        # 5 <= n_train < 20: leave-one-out
        n_train = 10
        plan = fold_plan(n_train, 20260715)
        self.assertEqual(len(plan), n_train)

        all_queries = []
        for idx, (fit_idx, query_idx) in enumerate(plan):
            self.assertEqual(len(query_idx), 1)
            self.assertEqual(query_idx[0], idx)
            self.assertEqual(len(fit_idx), n_train - 1)
            self.assertTrue(idx not in fit_idx)
            # check no overlap
            self.assertEqual(len(set(fit_idx).intersection(set(query_idx))), 0)
            all_queries.extend(query_idx)

        # Every train index should be queried exactly once
        self.assertEqual(sorted(all_queries), list(range(n_train)))

    def test_fold_plan_5fold(self):
        # n_train >= 20: shuffled 5-fold
        n_train = 25
        plan1 = fold_plan(n_train, 20260715)
        plan2 = fold_plan(n_train, 20260715)

        # Check determinism
        self.assertEqual(len(plan1), 5)
        for f1, f2 in zip(plan1, plan2):
            np.testing.assert_array_equal(f1[0], f2[0])
            np.testing.assert_array_equal(f1[1], f2[1])

        all_queries = []
        for fit_idx, query_idx in plan1:
            self.assertEqual(len(set(fit_idx).intersection(set(query_idx))), 0)
            all_queries.extend(query_idx.tolist())

        self.assertEqual(sorted(all_queries), list(range(n_train)))

    def test_higher_threshold(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # alpha = 0.3: 1-alpha = 0.7.
        # index 2 is 0.5 (3.0), index 3 is 0.75 (4.0).
        # Since 0.7 is between 0.5 and 0.75, method="higher" must yield index 3 (4.0).
        t = higher_threshold(scores, 0.3)
        self.assertEqual(t, 4.0)

    def test_higher_threshold_ignores_non_finite_reference_scores(self):
        scores = np.array([1.0, np.nan, 2.0, np.inf, -np.inf])
        self.assertEqual(higher_threshold(scores, 0.25), 2.0)
        with self.assertRaises(ValueError):
            higher_threshold(np.array([np.nan, np.inf]), 0.01)

    def test_higher_threshold_ignores_non_finite_reference_scores(self):
        scores = np.array([1.0, np.nan, 2.0, np.inf, -np.inf])
        self.assertEqual(higher_threshold(scores, 0.25), 2.0)
        with self.assertRaises(ValueError):
            higher_threshold(np.array([np.nan, np.inf]), 0.01)


    def test_source_candidate_flags(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        # threshold is 2.0. Strict > should yield [False, False, True, True]
        flags = source_candidate_flags(scores, 2.0)
        np.testing.assert_array_equal(flags, [False, False, True, True])

    def test_source_candidate_flags_rejects_non_finite_scores(self):
        scores = np.array([np.nan, np.inf, -np.inf, 3.0])
        flags = source_candidate_flags(scores, 2.0)
        np.testing.assert_array_equal(flags, [False, False, False, True])

    def test_source_candidate_flags_rejects_non_finite_scores(self):
        scores = np.array([np.nan, np.inf, -np.inf, 3.0])
        flags = source_candidate_flags(scores, 2.0)
        np.testing.assert_array_equal(flags, [False, False, False, True])

    def test_agreement_flags(self):
        # 3 sources
        s1 = np.array([True, False, True, False])
        s2 = np.array([True, True, False, False])
        s3 = np.array([False, True, True, False])

        # 2-of-3 agreement
        a2 = agreement_flags([s1, s2, s3], 2)
        np.testing.assert_array_equal(a2, [True, True, True, False])

        # 3-of-3 agreement
        a3 = agreement_flags([s1, s2, s3], 3)
        np.testing.assert_array_equal(a3, [False, False, False, False])

    def test_wilson_upper_bound(self):
        # Wilson score interval test cases
        # For 0 false reviews in 100 runs, lower is 0, but upper bound should be positive
        bound = wilson_upper_bound(0, 100, 0.95)
        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 0.05) # typical upper bound for 0/100 is around 0.03

        # Compare with known pre-calculated Wilson upper bound for 2/100
        # z = 1.644853 (one-sided 95%)
        # p = 0.02
        bound2 = wilson_upper_bound(2, 100, 0.95)
        self.assertGreater(bound2, 0.02)
        self.assertLess(bound2, 0.10)

        self.assertGreater(
            wilson_upper_bound(2, 100, 0.99),
            wilson_upper_bound(2, 100, 0.90),
        )
        self.assertEqual(wilson_upper_bound(0, 0, 0.95), 1.0)

        self.assertGreater(
            wilson_upper_bound(2, 100, 0.99),
            wilson_upper_bound(2, 100, 0.90),
        )
        self.assertEqual(wilson_upper_bound(0, 0, 0.95), 1.0)

    def test_load_train_normal_raw_mock(self):
        from unittest.mock import patch, MagicMock
        from experiments.exp164.exp164_train_bridge import load_train_normal_raw

        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            mock_cursor.fetchall.return_value = [
                (np.zeros(10, dtype=np.float32).tobytes(),),
                (np.ones(10, dtype=np.float32).tobytes(),),
            ]

            res = load_train_normal_raw("dummy_dataset")
            self.assertEqual(len(res), 2)

            # Check the SQL query structure
            calls = mock_cursor.execute.call_args_list
            self.assertTrue(len(calls) > 0)
            executed_sql = calls[0][0][0]
            self.assertIn("split = 'TRAIN'", executed_sql)
            self.assertNotIn("TEST", executed_sql.upper())
            self.assertNotIn("LABEL", executed_sql.upper())
            self.assertNotIn("ANOMALY", executed_sql.upper())

    def test_crossfit_target_length_isolation(self):
        from unittest.mock import patch
        from experiments.exp164.crossfit_sources import score_fold

        # We will mock the feature extraction functions to avoid full heavy math
        # but capture target_len in resample_rows.
        fit_raw = [np.ones(10), np.ones(12), np.ones(14)]  # median length is 12
        query_raw = [np.ones(100)]  # query length is 100

        with patch("experiments.exp164.crossfit_sources.resample_rows") as mock_resample:
            # mock resample_rows to return something matching shapes
            def side_effect(X, target_len):
                return np.zeros((X.shape[0], target_len), dtype=np.float32)
            mock_resample.side_effect = side_effect

            with patch("experiments.exp164.crossfit_sources.spectral_features") as mock_spec, \
                 patch("experiments.exp164.crossfit_sources.shapelet_features") as mock_shap, \
                 patch("experiments.exp164.crossfit_sources.interval_features") as mock_int, \
                 patch("experiments.exp164.crossfit_sources.robust_score_pair") as mock_robust:

                mock_robust.return_value = (np.zeros(3), np.zeros(1))
                mock_shap.return_value = (np.zeros((3, 32)), np.zeros((1, 32)))

                score_fold("dummy", fit_raw, query_raw)

                # Check target_len passed to resample_rows was indeed 12
                # (derived from fit_raw only)
                for call in mock_resample.call_args_list:
                    # call[0][1] or call[1]['target_len']
                    t_len = call[0][1] if len(call[0]) > 1 else call[1]['target_len']
                    self.assertEqual(t_len, 12)

    def test_audit_dataset_abstention(self):
        from experiments.exp164.crossfit_sources import audit_dataset
        # n_train < 5: abstention
        res = audit_dataset("dummy_dataset", [np.ones(10)] * 4)
        self.assertIsNone(res)

    def test_audit_dataset_run(self):
        from experiments.exp164.crossfit_sources import audit_dataset
        # A small run with 6 instances (leave-one-out)
        raw_series = [np.random.randn(20) for _ in range(6)]
        res = audit_dataset("dummy_dataset", raw_series)
        self.assertIsNotNone(res)
        self.assertEqual(res["n_train"], 6)
        self.assertEqual(res["fold_method"], "leave-one-out")

        # Check arrays sizes
        self.assertEqual(len(res["query_scores"]["spectral"]), 6)
        self.assertEqual(len(res["fold_indices"]), 6)
        np.testing.assert_array_equal(res["pseudo_run_indices"], np.arange(6))
        np.testing.assert_array_equal(res["fit_counts"], np.full(6, 5))
        self.assertTrue(np.all(res["target_lengths"] >= 8))
        np.testing.assert_array_equal(res["pseudo_run_indices"], np.arange(6))
        np.testing.assert_array_equal(res["fit_counts"], np.full(6, 5))
        self.assertTrue(np.all(res["target_lengths"] >= 8))
        # Check agreements exist for alpha=0.01
        self.assertEqual(len(res["agreement_2of3"][0.01]), 6)
        self.assertEqual(len(res["agreement_3of3"][0.01]), 6)

    def test_cache_io_write_and_validate(self):
        import tempfile
        from pathlib import Path
        from experiments.exp164.exp164_cache_io import write_cache_atomic, validate_cache

        n_train = 10
        payload = {
            "query_score_spectral": np.zeros(n_train, dtype=np.float64),
            "query_score_shapelet": np.zeros(n_train, dtype=np.float64),
            "query_score_interval": np.zeros(n_train, dtype=np.float64),
            "fold_indices": np.zeros(n_train, dtype=np.int32),
            "pseudo_run_indices": np.arange(n_train, dtype=np.int32),
            "fit_counts": np.full(n_train, n_train - 1, dtype=np.int32),
            "target_lengths": np.full(n_train, 32, dtype=np.int32),
            "pseudo_run_indices": np.arange(n_train, dtype=np.int32),
            "fit_counts": np.full(n_train, n_train - 1, dtype=np.int32),
            "target_lengths": np.full(n_train, 32, dtype=np.int32),
            "thresholds_spectral": np.ones((n_train, 4), dtype=np.float64),
            "thresholds_shapelet": np.ones((n_train, 4), dtype=np.float64),
            "thresholds_interval": np.ones((n_train, 4), dtype=np.float64),
            "flags_spectral": np.zeros((n_train, 4), dtype=bool),
            "flags_shapelet": np.zeros((n_train, 4), dtype=bool),
            "flags_interval": np.zeros((n_train, 4), dtype=bool),
            "agreement_2of3": np.zeros((n_train, 4), dtype=bool),
            "agreement_3of3": np.zeros((n_train, 4), dtype=bool),
        }

        metadata = {
            "dataset": "dummy_dataset",
            "n_train": n_train,
            "fold_method": "leave-one-out",
            "source_config_hash": "dummy_hash",
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "alphas": [0.005, 0.01, 0.02, 0.05]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "dummy.npz"

            # Successful write
            sha_hash = write_cache_atomic(cache_path, payload, metadata)
            self.assertTrue(cache_path.exists())
            self.assertTrue(len(sha_hash) > 0)

            # Successful validation
            valid, v_hash = validate_cache(cache_path, "dummy_hash", "dummy_dataset")
            self.assertTrue(valid)
            self.assertEqual(sha_hash, v_hash)

            # Failed validation: mismatch config hash
            valid, _ = validate_cache(cache_path, "mismatch_hash", "dummy_dataset")
            self.assertFalse(valid)

            # Failed validation: mismatch dataset
            valid, _ = validate_cache(cache_path, "dummy_hash", "other_dataset")
            self.assertFalse(valid)

            # Failed validation: forbidden keys in metadata
            bad_meta = metadata.copy()
            bad_meta["TP"] = 10
            bad_path = Path(tmpdir) / "bad.npz"
            with self.assertRaises(ValueError):
                write_cache_atomic(bad_path, payload, bad_meta)

    def test_cache_validation_rejects_inconsistent_flags(self):
        import tempfile
        from pathlib import Path
        from experiments.exp164.exp164_cache_io import write_cache_atomic, validate_cache

        n_train = 5
        scores = np.arange(n_train, dtype=np.float64)
        thresholds = np.full((n_train, 4), 2.0, dtype=np.float64)
        flags = scores[:, None] > thresholds
        payload = {
            "query_score_spectral": scores,
            "query_score_shapelet": scores,
            "query_score_interval": scores,
            "fold_indices": np.arange(n_train, dtype=np.int32),
            "pseudo_run_indices": np.arange(n_train, dtype=np.int32),
            "fit_counts": np.full(n_train, n_train - 1, dtype=np.int32),
            "target_lengths": np.full(n_train, 32, dtype=np.int32),
            "thresholds_spectral": thresholds,
            "thresholds_shapelet": thresholds,
            "thresholds_interval": thresholds,
            "flags_spectral": flags.copy(),
            "flags_shapelet": flags.copy(),
            "flags_interval": flags.copy(),
            "agreement_2of3": flags.copy(),
            "agreement_3of3": flags.copy(),
        }
        metadata = {
            "dataset": "dummy_dataset",
            "n_train": n_train,
            "fold_method": "leave-one-out",
            "source_config_hash": "dummy_hash",
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "alphas": [0.005, 0.01, 0.02, 0.05],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.npz"
            write_cache_atomic(path, payload, metadata)
            self.assertTrue(validate_cache(path, "dummy_hash", "dummy_dataset")[0])

            payload["flags_spectral"][0, 0] = True
            write_cache_atomic(path, payload, metadata)
            self.assertFalse(validate_cache(path, "dummy_hash", "dummy_dataset")[0])

    def test_checkpoint_contract_includes_population_hash_for_abstention(self):
        from experiments.exp164.run_experiment_164_train_normal_agreement_risk_audit import (
            checkpoint_contract_matches,
        )

        checkpoint = {
            "config_hash": "config",
            "database_sha256": "db",
            "population_sha256": "population",
        }
        self.assertTrue(
            checkpoint_contract_matches(checkpoint, "config", "db", "population")
        )
        self.assertFalse(
            checkpoint_contract_matches(checkpoint, "config", "db", "changed")
        )

if __name__ == "__main__":
    unittest.main()
