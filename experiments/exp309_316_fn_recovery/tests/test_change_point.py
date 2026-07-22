from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.change_point import score_change_point


class ChangePointTests(unittest.TestCase):
    def test_level_shift_and_drift_exceed_flat_normal(self):
        rng = np.random.default_rng(7)
        train = rng.normal(0, 0.03, (10, 96)).astype(np.float32)
        test = rng.normal(0, 0.03, (3, 96)).astype(np.float32)
        test[1, 48:] += 3; test[2] += np.linspace(0, 3, 96)
        scores = score_change_point(train, test, seed=20260722).test_scores
        self.assertGreater(scores[1], scores[0]); self.assertGreater(scores[2], scores[0])


if __name__ == "__main__": unittest.main()
