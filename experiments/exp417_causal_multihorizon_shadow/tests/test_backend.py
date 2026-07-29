"""Synthetic-only parity and adoption-gate tests for Exp417 backends."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.backend import (
    TorchBenchmarkMeasurement,
    benchmark_scoring_backend,
    benchmark_torch_devices,
    persistence_numba,
    persistence_numpy,
    probe_mps_host,
    select_canonical_backend,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp417_causal_multihorizon_shadow.model import CausalMultiHorizonCNN
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons


class PersistenceParityTests(unittest.TestCase):
    def test_numpy_and_numba_keep_exact_persistence_and_candidate_identity(self) -> None:
        fixtures = (
            np.array([], dtype=bool),
            np.array([False, False, False], dtype=bool),
            np.array([True, True, True], dtype=bool),
            np.array([True, False, True, False, True, False], dtype=bool),
            np.random.default_rng(417).random(127) > 0.73,
            np.array([False, True, True, False, True, False, False, True], dtype=bool)[::2],
        )
        for mask in fixtures:
            with self.subTest(shape=mask.shape, contiguous=mask.flags.c_contiguous):
                self.assertEqual(persistence_numpy(mask), persistence_numba(mask))

        audit = benchmark_scoring_backend(fixtures, candidate_masks=fixtures, repeats=1)
        self.assertEqual(audit.backend, "numba")
        self.assertTrue(audit.candidate_identity)
        self.assertEqual(audit.max_normalized_error, 0.0)
        self.assertEqual(audit.spearman, 1.0)
        self.assertFalse(audit.fallback)


class MPSGateTests(unittest.TestCase):
    @staticmethod
    def measurement(**changes: object) -> TorchBenchmarkMeasurement:
        base: dict[str, object] = {
            "candidate_identity": True,
            "max_normalized_error": 1e-4,
            "spearman": 0.999,
            "speedup": 1.1,
            "finite_outputs": True,
            "fallback": False,
            "environment": "host",
        }
        base.update(changes)
        return TorchBenchmarkMeasurement(**base)  # type: ignore[arg-type]

    def test_mps_adopts_only_when_every_frozen_boundary_passes(self) -> None:
        audit = benchmark_torch_devices(measurement=self.measurement())
        self.assertEqual(audit.backend, "mps")
        self.assertFalse(audit.fallback)
        self.assertEqual(select_canonical_backend((audit,)), "mps")

    def test_each_failed_mps_gate_falls_back_to_cpu_with_its_exact_reason(self) -> None:
        failures = (
            ("candidate_identity", False, "candidate_identity"),
            ("max_normalized_error", 0.000100001, "max_normalized_error"),
            ("spearman", 0.998999, "spearman"),
            ("speedup", 1.099999, "speedup"),
            ("finite_outputs", False, "nonfinite_outputs"),
            ("fallback", True, "device_fallback"),
        )
        for field, value, reason in failures:
            with self.subTest(field=field):
                audit = benchmark_torch_devices(measurement=self.measurement(**{field: value}))
                self.assertEqual(audit.backend, "cpu")
                self.assertTrue(audit.fallback)
                self.assertEqual(audit.fallback_reason, reason)
                self.assertEqual(select_canonical_backend((audit,)), "cpu")

    def test_isolated_unavailability_is_fail_closed_but_not_host_capability_evidence(self) -> None:
        audit = benchmark_torch_devices(
            measurement=self.measurement(environment="isolated", available=False)
        )
        self.assertEqual(audit.backend, "cpu")
        self.assertEqual(audit.fallback_reason, "isolated_mps_unavailable")
        self.assertEqual(select_canonical_backend((audit,)), "cpu")


class HostMPSProbeTests(unittest.TestCase):
    def test_host_probe_reports_only_its_own_environment(self) -> None:
        probe = probe_mps_host()
        self.assertEqual(probe.environment, "host")
        if probe.available:
            self.assertTrue(probe.built)
            self.assertTrue(probe.forward_backward_synchronized)
            self.assertTrue(probe.finite)
            self.assertFalse(probe.fallback)
        else:
            self.assertEqual(probe.reason, "mps_unavailable")

    def test_real_synthetic_model_benchmark_is_finite_or_fails_closed(self) -> None:
        torch.manual_seed(41701)
        model = CausalMultiHorizonCNN(build_horizons(64))
        inputs = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(1, 2, 64)
        audit = benchmark_torch_devices(
            model=model,
            inputs=inputs,
            candidate_selector=lambda output: tuple(
                np.flatnonzero(output[min(output)].reshape(-1) >= 0.0).tolist()
            ),
            repeats=1,
        )
        self.assertEqual(select_canonical_backend((audit,)), audit.backend)
        self.assertTrue(np.isfinite((audit.max_normalized_error, audit.spearman, audit.speedup)).all())


class CanonicalSelectionTests(unittest.TestCase):
    def test_selection_ignores_numba_and_requires_a_passing_mps_audit(self) -> None:
        numba = BackendAudit("numba", True, 0.0, 1.0, 10.0, False)
        cpu = BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "mps_unavailable")
        self.assertEqual(select_canonical_backend((numba, cpu)), "cpu")
        with self.assertRaisesRegex(ValueError, "MPS"):
            select_canonical_backend(())


if __name__ == "__main__":
    unittest.main()
