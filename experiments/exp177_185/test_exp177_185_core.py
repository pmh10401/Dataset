from __future__ import annotations

import unittest

import numpy as np

from experiments.exp177_185 import core


class Exp177185CoreTest(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual(core.ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(core.RISK_TARGETS, (0.005, 0.01, 0.02))
        self.assertEqual(core.SOURCE_NAMES, (
            "temporal_random_kernel",
            "spectral_image",
            "texture_image",
            "interval_statistical",
        ))

    def test_empirical_upper_tail_pvalues(self):
        reference = np.asarray([1.0, 2.0, 3.0])
        actual = core.empirical_upper_tail_pvalues(reference, np.asarray([4.0, 2.0]))
        np.testing.assert_allclose(actual, [0.25, 0.75])

    def test_agreement_indices(self):
        pvalues = {
            "a": np.asarray([0.01, 0.20, 0.01]),
            "b": np.asarray([0.02, 0.01, 0.20]),
            "c": np.asarray([0.20, 0.02, 0.03]),
        }
        self.assertEqual(core.agreement_indices(pvalues, 0.05, 2), {0, 1, 2})

    def test_lanes_are_exclusive(self):
        hard, standard, priority = core.exclusive_lanes({1}, {1, 2}, {2, 3})
        self.assertEqual(hard, {1})
        self.assertEqual(standard, {2})
        self.assertEqual(priority, {3})

    def test_feature_extractors_are_finite_and_fixed_width(self):
        x = np.asarray([
            np.sin(np.linspace(0, 6.28, 64)),
            np.cos(np.linspace(0, 6.28, 64)),
        ], dtype=np.float32)
        canonical = core.canonical_features(x)
        wavelet = core.wavelet_features(x)
        self.assertEqual(canonical.shape[0], 2)
        self.assertEqual(wavelet.shape[0], 2)
        self.assertGreaterEqual(canonical.shape[1], 22)
        self.assertGreaterEqual(wavelet.shape[1], 12)
        self.assertTrue(np.all(np.isfinite(canonical)))
        self.assertTrue(np.all(np.isfinite(wavelet)))

    def test_crossfit_scores_are_deterministic(self):
        rng = np.random.default_rng(7)
        train = rng.normal(size=(12, 8))
        test = rng.normal(size=(5, 8))
        first = core.crossfit_scores(train, test, backend="knn_local_gap", seed=11)
        second = core.crossfit_scores(train, test, backend="knn_local_gap", seed=11)
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertEqual(first[0].shape, (12,))
        self.assertEqual(first[1].shape, (5,))

    def test_frozen_prediction_rejects_metrics(self):
        core.validate_prediction_payload({
            "dataset": "x", "n_test": 2, "candidate_indices": [0],
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
        with self.assertRaises(ValueError):
            core.validate_prediction_payload({
                "dataset": "x", "n_test": 2, "candidate_indices": [0],
                "hard_tp": 1, "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            })

    def test_device_decision_never_claims_unavailable_mps(self):
        decision = core.choose_compute_device(force="cpu")
        self.assertEqual(decision["device"], "cpu")
        self.assertIn("reason", decision)


if __name__ == "__main__":
    unittest.main()
