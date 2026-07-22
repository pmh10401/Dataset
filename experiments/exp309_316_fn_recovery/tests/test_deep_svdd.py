from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.deep_svdd import score_deep_svdd


class DeepSvddTests(unittest.TestCase):
    def test_shifted_trace_has_finite_center_distance(self):
        base = np.sin(np.linspace(0, 6 * np.pi, 96, endpoint=False)).astype(np.float32)
        train = np.stack([np.roll(base, shift % 6) for shift in range(24)])
        test = np.stack([base, base + np.linspace(0, 4, 96, dtype=np.float32)])
        artifact = score_deep_svdd(train, test, device="cpu", seed=20260722, epochs=3)
        self.assertTrue(np.isfinite(artifact.test_scores).all())
        self.assertEqual(artifact.diagnostics["embedding_dim"], 32)


if __name__ == "__main__": unittest.main()
