from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.wavelet_phase import score_wavelet_phase


class WaveletPhaseTests(unittest.TestCase):
    def test_phase_disruption_scores_above_stable_periodic_trace(self):
        x = np.sin(np.linspace(0, 12 * np.pi, 128, endpoint=False)).astype(np.float32)
        train = np.stack([np.roll(x, shift) for shift in range(10)])
        test = np.stack([x, np.concatenate([x[:64], -x[64:]])])
        scores = score_wavelet_phase(train, test, seed=20260722).test_scores
        self.assertGreater(scores[1], scores[0])


if __name__ == "__main__": unittest.main()
