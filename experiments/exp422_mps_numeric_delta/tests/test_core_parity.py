"""Focused deterministic unit tests for Exp422 core parity evidence."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.exp422_mps_numeric_delta.capability import DeviceCapability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
from experiments.exp422_mps_numeric_delta.core_parity import (
    candidate_coordinate_digest,
    compare_tensor,
    run_core_parity,
    tie_aware_spearman,
)


class CoreParityTests(unittest.TestCase):
    def test_ulp_metric_is_zero_for_identical_float32(self) -> None:
        """Changing the ordered-float ULP calculation breaks exact equality."""
        delta = compare_tensor(
            np.array([1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            name="x",
        )
        self.assertEqual(delta.max_ulp, 0)
        self.assertEqual(delta.max_normalized_error, 0.0)
        self.assertTrue(delta.passes)

    def test_nonfinite_mps_output_fails_closed(self) -> None:
        """Dropping the finite gate would incorrectly mark NaN output as usable."""
        delta = compare_tensor(
            np.array([1.0], dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
            name="x",
        )
        self.assertFalse(delta.finite)
        self.assertFalse(delta.passes)
        self.assertEqual(delta.failing_index, 0)

    def test_candidate_digest_is_stable_for_tied_scores(self) -> None:
        """Replacing stable ordering changes the selected coordinate digest."""
        scores = np.array([0.4, 0.1, 0.1, 0.3], dtype=np.float32)
        digest = candidate_coordinate_digest(scores, 0.5)
        self.assertEqual(
            digest,
            "8834c4d4bb5166ac2654dd3566be87d6519dda3aa17a93a2824a243c53f9d533",
        )

    def test_tie_aware_spearman_returns_zero_for_constant_scores(self) -> None:
        """A constant rank vector has no correlation and must fail closed."""
        self.assertEqual(
            tie_aware_spearman(np.ones(4, dtype=np.float32), np.arange(4, dtype=np.float32)),
            0.0,
        )

    def test_unavailable_mps_yields_every_frozen_row(self) -> None:
        """Skipping unavailable MPS must retain a structured failure for every key."""
        capability = DeviceCapability(
            torch_version="unavailable",
            mps_built=False,
            mps_available=False,
            device_name="unknown CPU",
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_availability="unavailable",
            fallback_reason="mps_unavailable",
        )
        result = run_core_parity(FROZEN_CONFIG, capability)
        self.assertEqual(result.stage.state, "failed")
        self.assertEqual(result.stage.reason, "mps_unavailable")
        self.assertEqual(
            len(result.reports),
            len(FROZEN_CONFIG.candidate_batch_sizes)
            * len(FROZEN_CONFIG.model_seeds)
            * 4,
        )
        self.assertTrue(all(not report.tensor_deltas[0].finite for report in result.reports))


if __name__ == "__main__":
    unittest.main()
