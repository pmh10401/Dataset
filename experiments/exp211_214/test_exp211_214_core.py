from __future__ import annotations

import unittest

import numpy as np

from experiments.exp211_214.core import (
    STABILITY_CONTRACT,
    aggregate_fold_pvalues,
    pooled_cross_conformal_pvalues,
    source_stability,
    transport_metrics,
)
from experiments.exp211_214.fold_transport import build_fold_transport_artifact


class Exp211214CoreTests(unittest.TestCase):
    def test_fold_pvalue_aggregation_is_deterministic(self):
        pvalues = np.asarray(
            [[0.01, 0.40], [0.02, 0.30], [0.03, 0.20], [0.04, 0.10], [0.05, 0.50]]
        )
        np.testing.assert_allclose(
            aggregate_fold_pvalues(pvalues, "median"), [0.03, 0.30]
        )
        np.testing.assert_allclose(
            aggregate_fold_pvalues(pvalues, "conservative_max"), [0.05, 0.50]
        )

    def test_transport_metrics_identical_vectors(self):
        values = np.linspace(0.0, 1.0, 101)
        metrics = transport_metrics(values, values.copy(), values.copy())
        self.assertAlmostEqual(metrics["oof_full_rank_corr"], 1.0)
        self.assertAlmostEqual(metrics["oof_full_median_shift_robust"], 0.0)
        self.assertAlmostEqual(metrics["oof_full_mad_ratio"], 1.0)
        self.assertAlmostEqual(metrics["oof_full_ks"], 0.0)
        self.assertAlmostEqual(metrics["oof_full_wasserstein"], 0.0)

    def test_pooled_cross_conformal_restores_full_train_resolution(self):
        counts = np.asarray([[0, 2], [0, 1], [0, 0], [0, 3], [0, 4]])
        sizes = np.asarray([20, 20, 20, 20, 20])
        pvalues = pooled_cross_conformal_pvalues(counts, sizes)
        np.testing.assert_allclose(pvalues, [1.0 / 101.0, 11.0 / 101.0])

    def test_source_stability_uses_frozen_train_only_contract(self):
        stable = {
            "oof_full_rank_corr": STABILITY_CONTRACT["min_rank_corr"],
            "oof_full_median_shift_robust": STABILITY_CONTRACT["max_abs_median_shift"],
            "oof_full_mad_ratio": STABILITY_CONTRACT["min_mad_ratio"],
        }
        self.assertTrue(source_stability(stable))
        unstable = dict(stable, oof_full_rank_corr=0.2)
        self.assertFalse(source_stability(unstable))

    def test_transport_metrics_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            transport_metrics(np.arange(5), np.arange(4), np.arange(3))

    def test_fold_transport_is_deterministic_and_query_safe(self):
        rng = np.random.default_rng(211)
        train = rng.normal(size=(15, 32)).astype(np.float32)
        test = rng.normal(size=(4, 32)).astype(np.float32)
        first = build_fold_transport_artifact(train, test, seed=17)
        second = build_fold_transport_artifact(train, test, seed=17)
        changed_test = test.copy()
        changed_test[0] += 8.0
        changed = build_fold_transport_artifact(train, changed_test, seed=17)
        for source in ("spectral", "shapelet", "interval"):
            np.testing.assert_array_equal(
                first[f"{source}_train_oof"], second[f"{source}_train_oof"]
            )
            np.testing.assert_array_equal(
                first[f"{source}_train_oof"], changed[f"{source}_train_oof"]
            )
            self.assertEqual(first[f"{source}_fold_test_pvalues"].shape, (5, 4))
            self.assertEqual(first[f"{source}_fold_test_exceedance_counts"].shape, (5, 4))
            self.assertEqual(int(np.sum(first[f"{source}_fold_calibration_sizes"])), 15)
            self.assertTrue(np.all(np.isfinite(first[f"{source}_fold_test_pvalues"])))


if __name__ == "__main__":
    unittest.main()
