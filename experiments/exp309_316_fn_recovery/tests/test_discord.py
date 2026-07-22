from __future__ import annotations

import unittest

import numpy as np

from experiments.exp309_316_fn_recovery.discord import score_discord


class DiscordTests(unittest.TestCase):
    def test_local_patch_change_scores_above_repeated_normal(self):
        base = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False)).astype(np.float32)
        train = np.stack([np.roll(base, shift) for shift in range(8)])
        test = np.stack([base.copy(), base.copy()]); test[1, 24:32] += 5
        artifact = score_discord(train, test, seed=20260722)
        self.assertGreater(artifact.test_scores[1], artifact.test_scores[0])
        self.assertEqual(artifact.diagnostics["candidate_grain"], "test_instance_virtual_run")


if __name__ == "__main__": unittest.main()
