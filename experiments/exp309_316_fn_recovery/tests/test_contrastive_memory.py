from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.contrastive_memory import score_contrastive_memory


class ContrastiveMemoryTests(unittest.TestCase):
    def test_shape_reversal_has_finite_memory_distance(self):
        base = np.sin(np.linspace(0, 6 * np.pi, 96, endpoint=False)).astype(np.float32)
        train = np.stack([np.roll(base, shift % 6) for shift in range(24)])
        test = np.stack([base, base[::-1].copy()])
        artifact = score_contrastive_memory(train, test, device="cpu", seed=20260722, epochs=3)
        self.assertTrue(np.isfinite(artifact.test_scores).all())
        self.assertEqual(len(artifact.test_scores), 2)
        self.assertEqual(artifact.diagnostics["embedding_dim"], 32)


if __name__ == "__main__": unittest.main()
