from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.predictive_residual import _windows, score_predictive_residual


class PredictiveResidualTests(unittest.TestCase):
    def test_predictive_residual_is_finite_and_detects_broken_continuation(self):
        base = np.sin(np.linspace(0, 8 * np.pi, 96, endpoint=False)).astype(np.float32)
        train = np.stack([np.roll(base, shift % 4) for shift in range(24)])
        test = np.stack([base.copy(), base.copy()]); test[1, 64:] = test[1, 64:][::-1]
        artifact = score_predictive_residual(train, test, device="cpu", seed=20260722, epochs=4)
        self.assertTrue(np.isfinite(artifact.test_scores).all())
        self.assertGreater(artifact.test_scores[1], artifact.test_scores[0])
        self.assertEqual(artifact.diagnostics["calibration_method"], "three_way_train_split")

    def test_window_cap_preserves_at_least_one_window_per_run(self):
        rows = np.tile(np.arange(16, dtype=np.float32), (4250, 1))
        _x, _y, owners = _windows(rows, context=8, horizon=4, cap=4096)
        self.assertEqual(len(np.unique(owners)), len(rows))
        self.assertEqual(int(owners.min()), 0)
        self.assertEqual(int(owners.max()), len(rows) - 1)


if __name__ == "__main__": unittest.main()
