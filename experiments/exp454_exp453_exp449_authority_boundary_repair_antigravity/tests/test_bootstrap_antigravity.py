from __future__ import annotations

import unittest

import numpy as np

from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.bootstrap_antigravity import (
    BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY,
    BOOTSTRAP_POPULATION_ANTIGRAVITY,
    BOOTSTRAP_SAMPLES_ANTIGRAVITY,
    BOOTSTRAP_SEED_ANTIGRAVITY,
    bootstrap_draws_antigravity,
    paired_delta_antigravity,
)


class BootstrapAntigravityTest(unittest.TestCase):
    def test_bootstrap_authority_is_byte_exact_antigravity(self) -> None:
        """Break caught: changing PCG64 seed, sample count, population, dtype, or draw bytes."""
        draws = bootstrap_draws_antigravity()
        self.assertEqual(
            (
                BOOTSTRAP_SEED_ANTIGRAVITY,
                BOOTSTRAP_SAMPLES_ANTIGRAVITY,
                BOOTSTRAP_POPULATION_ANTIGRAVITY,
                BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY,
            ),
            (
                44920260812,
                20_000,
                48,
                "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a",
            ),
        )
        self.assertEqual((draws.dtype.name, draws.shape, draws.flags.writeable), ("int64", (20_000, 48), False))

    def test_paired_delta_matches_hand_checked_constant_population_antigravity(self) -> None:
        """Break caught: changing FN/FP/F1 delta signs, percentiles, or metadata."""
        baseline = np.tile(np.asarray([1, 0, 1, 2], dtype=np.int64), (48, 1))
        candidate = np.tile(np.asarray([1, 0, 0, 3], dtype=np.int64), (48, 1))
        result = paired_delta_antigravity(baseline, candidate)
        self.assertEqual(result["fn_delta"], -48)
        self.assertEqual(result["fp_delta"], 0)
        self.assertEqual(result["fn_interval"], [-48.0, -48.0])
        self.assertEqual(result["fp_interval"], [0.0, 0.0])
        self.assertAlmostEqual(result["f1_delta"], 1.0 / 3.0)
        self.assertEqual(
            {key: result[key] for key in ("draw_sha256", "seed", "samples", "population", "dtype", "endpoint", "percentile_method")},
            {
                "draw_sha256": "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a",
                "seed": 44920260812,
                "samples": 20_000,
                "population": 48,
                "dtype": "int64",
                "endpoint": False,
                "percentile_method": "linear",
            },
        )

    def test_paired_delta_rejects_wrong_population_and_negative_counts_antigravity(self) -> None:
        """Break caught: silently padding, truncating, or accepting invalid confusion rows."""
        with self.assertRaisesRegex(ValueError, "fixed 48-dataset population"):
            paired_delta_antigravity(np.zeros((47, 4), dtype=np.int64), np.zeros((47, 4), dtype=np.int64))
        negative = np.zeros((48, 4), dtype=np.int64)
        negative[0, 2] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            paired_delta_antigravity(negative, np.zeros((48, 4), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
