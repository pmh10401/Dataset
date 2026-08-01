"""Focused deterministic unit tests for Exp422 core parity evidence."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp422_mps_numeric_delta.capability import DeviceCapability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG
import experiments.exp422_mps_numeric_delta.core_parity as core_parity
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

    def test_ulp_metric_collapses_signed_zero(self) -> None:
        """A sign-bit implementation must not report a gap between -0 and +0."""
        delta = compare_tensor(
            np.array([-0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            name="signed_zero",
        )
        self.assertEqual(delta.max_ulp, 0)
        self.assertEqual(delta.max_absolute_error, 0.0)

    def test_ulp_metric_counts_one_negative_float32_neighbor(self) -> None:
        """Adjacent negative float32 values must have an ordered distance of one."""
        left = np.array([-1.0], dtype=np.float32)
        right = np.nextafter(left, np.array([-np.inf], dtype=np.float32))
        delta = compare_tensor(left, right, name="negative_neighbor")
        self.assertEqual(delta.max_ulp, 1)

    def test_unexpected_mps_key_error_is_not_masked(self) -> None:
        """A coding defect in the MPS branch must surface instead of becoming fallback."""
        class FakeModel:
            def __init__(self, horizons):
                self.horizons = tuple(horizons)

            def eval(self):
                return self

            def to(self, _device):
                return self

            def state_dict(self):
                return {}

            def load_state_dict(self, _state):
                return None

        def fake_forward(model, values, device):
            if device.type == "mps":
                raise KeyError("unexpected local defect")
            outputs = {
                horizon: np.zeros((values.shape[0], values.shape[2] - horizon), dtype=np.float32)
                for horizon in model.horizons
            }
            from experiments.exp422_mps_numeric_delta.contracts import TimingEvidence

            return outputs, TimingEvidence("cpu", 0.0, 0.0, 0.0, 0.0)

        capability = DeviceCapability(
            torch_version="2.13.0",
            mps_built=True,
            mps_available=True,
            device_name="Darwin arm64 MPS",
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_availability="unavailable",
            fallback_reason=None,
        )
        with patch.object(core_parity, "CausalMultiHorizonCNN", FakeModel), patch.object(
            core_parity, "_timed_forward", fake_forward
        ):
            with self.assertRaises(KeyError):
                run_core_parity(FROZEN_CONFIG, capability)

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
