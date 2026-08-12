from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.statistics import (
    bootstrap_draws,
    paired_bootstrap,
)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_draws_repeat_and_bind_sha(self) -> None:
        first, first_sha = bootstrap_draws(4, seed=11, samples=20)
        second, second_sha = bootstrap_draws(4, seed=11, samples=20)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first.shape, (20, 4))
        self.assertEqual(len(first_sha), 64)

    def test_paired_bootstrap_uses_supplied_shared_draws(self) -> None:
        draws, draw_sha = bootstrap_draws(4, seed=11, samples=200)
        baseline = np.asarray([0.2, 0.4, 0.6, 0.8])
        proposal = np.asarray([0.1, 0.3, 0.5, 0.7])

        result = paired_bootstrap(
            baseline,
            proposal,
            draws=draws,
            draw_sha256=draw_sha,
            seed=11,
        )

        self.assertAlmostEqual(result["point_estimate"], 0.1)
        self.assertAlmostEqual(result["lower"], 0.1)
        self.assertAlmostEqual(result["upper"], 0.1)
        self.assertEqual(result["draw_count"], 200)
        self.assertEqual(result["draw_sha256"], draw_sha)

    def test_paired_bootstrap_rejects_unaligned_values(self) -> None:
        draws, draw_sha = bootstrap_draws(2, seed=11, samples=20)
        with self.assertRaisesRegex(ValueError, "aligned"):
            paired_bootstrap(
                np.asarray([0.1]),
                np.asarray([0.1, 0.2]),
                draws=draws,
                draw_sha256=draw_sha,
                seed=11,
            )


if __name__ == "__main__":
    unittest.main()
