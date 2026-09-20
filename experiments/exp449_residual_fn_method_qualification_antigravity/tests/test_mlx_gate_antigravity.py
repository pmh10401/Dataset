"""Behavior tests for the stage-specific CPU/MLX/MPS governance gate."""
from __future__ import annotations

import unittest

from experiments.exp449_residual_fn_method_qualification_antigravity.mlx_gate_antigravity import (
    MlxDecisionAntigravity,
    assess_mlx_antigravity,
)


class MlxGateAntigravityTests(unittest.TestCase):
    """Availability is deliberately not an approval signal."""

    def test_missing_mlx_is_a_recorded_cpu_fallback_antigravity(self) -> None:
        """Catches treating an absent MLX runtime as an unrecorded/unknown backend."""
        decision = assess_mlx_antigravity(package_probe=lambda _: None)
        self.assertEqual(decision.backend, "cpu")
        self.assertEqual(decision.reason, "mlx_and_mlx_metal_absent_in_fixed_environment")
        self.assertFalse(decision.parity_run)
        self.assertFalse(decision.tensor_eligible)
        self.assertFalse(decision.benchmark_includes_transfer_sync_warmup_compile)

    def test_torch_mps_availability_is_recorded_without_approval_antigravity(self) -> None:
        """Catches selecting MPS merely because it is built and available."""
        decision = assess_mlx_antigravity(
            package_probe=lambda name: {"torch": "2.13.0"}.get(name),
            torch_probe=lambda: (True, True),
        )
        self.assertEqual(decision.backend, "cpu")
        self.assertEqual(decision.torch_version, "2.13.0")
        self.assertIs(decision.mps_built, True)
        self.assertIs(decision.mps_available, True)
        self.assertEqual(decision.reason, "mlx_and_mlx_metal_absent_in_fixed_environment")

    def test_incomplete_or_failed_parity_speed_gate_keeps_cpu_canonical_antigravity(self) -> None:
        """Catches approval with a timing omission, non-finite scores, mismatch, tolerance failure, or low speedup."""
        cases = (
            ({"tensor_eligible": True, "finite": True, "candidate_indices_identical": True, "numerical_tolerance_met": True, "benchmark_includes_transfer_sync_warmup_compile": False, "speedup": 2.0}, "benchmark_not_inclusive"),
            ({"tensor_eligible": True, "finite": False, "candidate_indices_identical": True, "numerical_tolerance_met": True, "benchmark_includes_transfer_sync_warmup_compile": True, "speedup": 2.0}, "nonfinite_outputs"),
            ({"tensor_eligible": True, "finite": True, "candidate_indices_identical": False, "numerical_tolerance_met": True, "benchmark_includes_transfer_sync_warmup_compile": True, "speedup": 2.0}, "candidate_indices_not_identical"),
            ({"tensor_eligible": True, "finite": True, "candidate_indices_identical": True, "numerical_tolerance_met": False, "benchmark_includes_transfer_sync_warmup_compile": True, "speedup": 2.0}, "numerical_tolerance_not_met"),
            ({"tensor_eligible": True, "finite": True, "candidate_indices_identical": True, "numerical_tolerance_met": True, "benchmark_includes_transfer_sync_warmup_compile": True, "speedup": 1.09}, "speedup_below_1_1"),
        )
        for probe, reason in cases:
            with self.subTest(reason=reason):
                decision = assess_mlx_antigravity(
                    package_probe=lambda name: {"mlx": "0.30.0", "mlx-metal": "0.30.0"}.get(name),
                    stage_probe=lambda: probe,
                )
                self.assertEqual(decision.backend, "cpu")
                self.assertEqual(decision.reason, reason)
                self.assertTrue(decision.parity_run)

    def test_backend_specific_availability_and_complete_passes_antigravity(self) -> None:
        """Catches hardcoding MLX approval or treating MPS availability as MLX absence."""
        complete = {
            "tensor_eligible": True,
            "finite": True,
            "candidate_indices_identical": True,
            "numerical_tolerance_met": True,
            "benchmark_includes_transfer_sync_warmup_compile": True,
            "speedup": 1.1,
        }
        mps = assess_mlx_antigravity(
            package_probe=lambda name: {"torch": "2.13.0"}.get(name),
            torch_probe=lambda: (True, True),
            stage_probe=lambda: {**complete, "backend": "mps"},
        )
        self.assertEqual((mps.backend, mps.reason, mps.device), ("mps", "complete_mps_gate_passed", "mps"))
        mlx = assess_mlx_antigravity(
            package_probe=lambda name: {"mlx": "0.30.0", "mlx-metal": "0.30.0"}.get(name),
            stage_probe=lambda: {**complete, "backend": "mlx"},
        )
        self.assertEqual((mlx.backend, mlx.reason, mlx.device), ("mlx", "complete_mlx_gate_passed", "mlx"))

    def test_partial_or_unavailable_backend_is_recorded_cpu_fallback_antigravity(self) -> None:
        """Catches attempting a backend whose required package/device evidence is incomplete."""
        stage = lambda backend: {
            "backend": backend,
            "tensor_eligible": True,
            "finite": True,
            "candidate_indices_identical": True,
            "numerical_tolerance_met": True,
            "benchmark_includes_transfer_sync_warmup_compile": True,
            "speedup": 1.2,
        }
        partial_mlx = assess_mlx_antigravity(
            package_probe=lambda name: {"mlx": "0.30.0"}.get(name), stage_probe=lambda: stage("mlx")
        )
        self.assertEqual((partial_mlx.backend, partial_mlx.reason), ("cpu", "mlx_runtime_incomplete"))
        unavailable_mps = assess_mlx_antigravity(
            package_probe=lambda name: {"torch": "2.13.0"}.get(name),
            torch_probe=lambda: (True, False),
            stage_probe=lambda: stage("mps"),
        )
        self.assertEqual((unavailable_mps.backend, unavailable_mps.reason), ("cpu", "mps_unavailable"))

    def test_invalid_timing_speed_measurements_are_recorded_fallbacks_antigravity(self) -> None:
        """Catches propagating non-finite or negative timing/speed values instead of failing closed."""
        base = {
            "backend": "mlx",
            "tensor_eligible": True,
            "finite": True,
            "candidate_indices_identical": True,
            "numerical_tolerance_met": True,
            "benchmark_includes_transfer_sync_warmup_compile": True,
        }
        for speedup in (float("nan"), float("inf"), -0.01):
            with self.subTest(speedup=speedup):
                decision = assess_mlx_antigravity(
                    package_probe=lambda name: {"mlx": "0.30.0", "mlx-metal": "0.30.0"}.get(name),
                    stage_probe=lambda: {**base, "speedup": speedup},
                )
                self.assertEqual((decision.backend, decision.reason, decision.speedup), ("cpu", "invalid_speedup", None))

    def test_decision_rejects_boolean_numeric_speedup_antigravity(self) -> None:
        """Catches bool being admitted as a numeric performance measurement."""
        with self.assertRaisesRegex(ValueError, "speedup"):
            MlxDecisionAntigravity("cpu", "reason", False, None, None, None, None, None, None, False, None, None, False, True)


if __name__ == "__main__":
    unittest.main()
