import math
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp421_mps_batched_acceleration.contracts import (
    BackendMeasurement,
    BackendPolicy,
    BatchKey,
    Exp421Manifest,
    MlxAssessment,
    ParityReport,
)


class Exp421ContractTests(unittest.TestCase):
    def _batch_key(self, **changes):
        values = {"batch_size": 1, "model_seed": 41701, "perturbation_seed": 41711}
        values.update(changes)
        return BatchKey(**values)

    def _measurement(self, backend, **changes):
        values = {
            "batch_size": 1,
            "backend": backend,
            "transfer_seconds": 0.1,
            "compute_seconds": 0.2,
            "synchronize_seconds": 0.1,
            "total_seconds": 0.4,
            "finite": True,
            "candidate_coordinates_sha256": "a" * 64,
        }
        values.update(changes)
        return BackendMeasurement(**values)

    def _parity(self, **changes):
        values = {
            "batch_key": self._batch_key(),
            "candidate_indices_by_alpha": {0.01: np.array([1, 4], dtype=np.int64)},
            "max_normalized_error": 1e-5,
            "spearman": 1.0,
        }
        values.update(changes)
        return ParityReport(**values)

    def _assessment(self, **changes):
        values = {
            "batch_key": self._batch_key(),
            "cpu_measurement": self._measurement("cpu"),
            "mlx_measurement": self._measurement(
                "mlx", transfer_seconds=0.05, compute_seconds=0.1,
                synchronize_seconds=0.05, total_seconds=0.2,
            ),
            "parity_report": self._parity(),
            "speedup": 2.0,
            "uses_mlx": True,
            "fallback_reason": None,
        }
        values.update(changes)
        return MlxAssessment(**values)

    def _manifest(self, **changes):
        cached = np.array([1, 4], dtype=np.int64)
        cached.setflags(write=False)
        values = {
            "state": "draft",
            "policy": BackendPolicy("mlx", self._assessment()),
            "cached_candidate_indices": cached,
            "provenance": {"source": "exp417"},
        }
        values.update(changes)
        return Exp421Manifest(**values)

    def test_valid_records_freeze_mapping_and_cached_arrays(self):
        parity = self._parity()
        manifest = self._manifest()
        self.assertIsInstance(parity.candidate_indices_by_alpha, MappingProxyType)
        self.assertFalse(parity.candidate_indices_by_alpha[0.01].flags.writeable)
        self.assertIsInstance(manifest.provenance, MappingProxyType)
        self.assertFalse(manifest.cached_candidate_indices.flags.writeable)

    def test_batch_key_rejects_candidate_batch_outside_frozen_schedule(self):
        with self.assertRaises(ValueError):
            self._batch_key(batch_size=2)

    def test_measurement_rejects_nonfinite_or_inconsistent_timing(self):
        for changes in ({"compute_seconds": math.inf}, {"total_seconds": 0.3}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._measurement("cpu", **changes)

    def test_assessment_requires_fallback_when_speedup_misses_gate(self):
        slow_cpu = self._measurement(
            "cpu", transfer_seconds=0.03, compute_seconds=0.13,
            synchronize_seconds=0.058, total_seconds=0.218,
        )
        with self.assertRaises(ValueError):
            self._assessment(
                cpu_measurement=slow_cpu, speedup=1.09,
                uses_mlx=False, fallback_reason=None,
            )
        assessment = self._assessment(
            cpu_measurement=slow_cpu, speedup=1.09,
            uses_mlx=False, fallback_reason="speedup",
        )
        self.assertFalse(assessment.uses_mlx)

    def test_manifest_rejects_test_token_invalid_state_and_writable_cached_array(self):
        writable = np.array([1, 4], dtype=np.int64)
        for changes in (
            {"provenance": {"source": "TEST-only"}},
            {"state": "published"},
            {"cached_candidate_indices": writable},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._manifest(**changes)


if __name__ == "__main__":
    unittest.main()
