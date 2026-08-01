import math
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp421_mps_batched_acceleration.contracts import (
    BackendMeasurement,
    BackendPolicy,
    BatchKey,
    Exp421Manifest,
    MpsAssessment,
    ParityReport,
)
from experiments.exp421_mps_batched_acceleration.config import (
    Exp421Config,
    frozen_config_sha256,
)


class Exp421ContractTests(unittest.TestCase):
    def _batch_key(self, **changes):
        values = {"batch_size": 4, "model_seed": 41701, "perturbation_seed": 41711}
        values.update(changes)
        return BatchKey(**values)

    def _measurement(self, backend, **changes):
        values = {
            "batch_size": 4,
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
            "cpu_candidate_indices_by_alpha": {0.01: np.array([1, 4], dtype=np.int64)},
            "mps_candidate_indices_by_alpha": {0.01: np.array([1, 4], dtype=np.int64)},
            "max_normalized_error": 1e-5,
            "spearman": 1.0,
        }
        values.update(changes)
        return ParityReport(**values)

    def _assessment(self, **changes):
        values = {
            "batch_key": self._batch_key(),
            "cpu_measurement": self._measurement("cpu"),
            "mps_measurement": self._measurement(
                "mps", transfer_seconds=0.05, compute_seconds=0.1,
                synchronize_seconds=0.05, total_seconds=0.2,
            ),
            "parity_report": self._parity(),
            "speedup": 2.0,
            "uses_mps": True,
            "fallback_reason": None,
            "mps_available": True,
            "device": "Apple M4",
        }
        values.update(changes)
        return MpsAssessment(**values)

    def _manifest(self, **changes):
        cached = np.array([1, 4], dtype=np.int64)
        cached.setflags(write=False)
        values = {
            "state": "draft",
            "policy": BackendPolicy("mps", self._assessment()),
            "cached_candidate_indices": cached,
            "provenance": {
                "git_commit": "a" * 40,
                "database_sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
                "config_sha256": frozen_config_sha256(Exp421Config()),
                "device": "Apple M4",
                "runtime": "Python 3.14",
                "source": "exp417",
                "fresh_no_clobber_attempt": True,
                "train_only": True,
                "mlx_availability": "available",
                "mlx_version": "0.20.0",
                "mlx_metal_version": "0.20.0",
                "mlx_eligibility_reason": "eligible",
            },
        }
        values.update(changes)
        return Exp421Manifest(**values)

    def test_valid_records_freeze_mapping_and_cached_arrays(self):
        parity = self._parity()
        manifest = self._manifest()
        self.assertIsInstance(parity.cpu_candidate_indices_by_alpha, MappingProxyType)
        self.assertFalse(parity.cpu_candidate_indices_by_alpha[0.01].flags.writeable)
        self.assertIsInstance(manifest.provenance, MappingProxyType)
        self.assertFalse(manifest.cached_candidate_indices.flags.writeable)

    def test_batch_key_rejects_candidate_batch_outside_frozen_schedule(self):
        for batch_size in (2, 1.0, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self._batch_key(batch_size=batch_size)

    def test_measurement_requires_an_exact_integer_batch_size(self):
        for batch_size in (1.0, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self._measurement("cpu", batch_size=batch_size)

    def test_measurement_rejects_nonfinite_or_inconsistent_timing(self):
        for changes in ({"compute_seconds": math.inf}, {"total_seconds": 0.3}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._measurement("cpu", **changes)

    def test_measurement_accepts_only_cpu_and_mps_backend_names(self):
        self._measurement("mps")
        with self.assertRaises(ValueError):
            self._measurement("mlx")

    def test_mps_assessment_requires_fallback_when_speedup_misses_gate(self):
        slow_cpu = self._measurement(
            "cpu", transfer_seconds=0.03, compute_seconds=0.13,
            synchronize_seconds=0.058, total_seconds=0.218,
        )
        with self.assertRaises(ValueError):
            self._assessment(
                cpu_measurement=slow_cpu, speedup=1.09,
                uses_mps=False, fallback_reason=None,
            )
        assessment = self._assessment(
            cpu_measurement=slow_cpu, speedup=1.09,
            uses_mps=False, fallback_reason="speedup",
        )
        self.assertFalse(assessment.uses_mps)

    def test_batch_one_is_cpu_mandatory_even_when_mps_gates_pass(self):
        batch_key = self._batch_key(batch_size=1)
        cpu = self._measurement("cpu", batch_size=1)
        mps = self._measurement(
            "mps", batch_size=1, transfer_seconds=0.05,
            compute_seconds=0.1, synchronize_seconds=0.05, total_seconds=0.2,
        )
        parity = self._parity(batch_key=batch_key)
        assessment = self._assessment(
            batch_key=batch_key, cpu_measurement=cpu, mps_measurement=mps,
            parity_report=parity, speedup=2.0, uses_mps=False,
            fallback_reason="batch_size_one_cpu_mandatory",
        )
        self.assertFalse(assessment.uses_mps)

    def test_mps_unavailable_is_a_complete_cpu_fallback_record(self):
        assessment = self._assessment(
            mps_measurement=None, parity_report=None, speedup=None,
            uses_mps=False, fallback_reason="mps_unavailable",
            mps_available=False, device="unavailable",
        )
        self.assertFalse(assessment.uses_mps)
        self.assertFalse(assessment.mps_available)

    def test_mps_candidate_indices_must_match_cpu_at_every_alpha(self):
        parity = self._parity(
            mps_candidate_indices_by_alpha={0.01: np.array([1, 5], dtype=np.int64)},
        )
        with self.assertRaises(ValueError):
            self._assessment(parity_report=parity)

    def test_manifest_rejects_incoherent_mlx_capability_provenance(self):
        provenance = dict(self._manifest().provenance)
        provenance.update(
            mlx_availability="unavailable",
            mlx_version="0.20.0",
            mlx_metal_version="0.20.0",
            mlx_eligibility_reason="eligible",
        )
        with self.assertRaises(ValueError):
            self._manifest(provenance=provenance)
        provenance.update(
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_eligibility_reason="package_absent",
        )
        self._manifest(provenance=provenance)

    def test_manifest_rejects_test_token_invalid_state_and_writable_cached_array(self):
        writable = np.array([1, 4], dtype=np.int64)
        for changes in (
            {"provenance": {**self._manifest().provenance, "source": "TEST-only"}},
            {"state": "published"},
            {"cached_candidate_indices": writable},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._manifest(**changes)

    def test_manifest_requires_complete_train_only_provenance(self):
        provenance = dict(self._manifest().provenance)
        del provenance["git_commit"]
        with self.assertRaises(ValueError):
            self._manifest(provenance=provenance)
        self.assertEqual(self._manifest(provenance={**self._manifest().provenance, "source": "latest"}).provenance["source"], "latest")
        with self.assertRaises(ValueError):
            self._manifest(provenance={**self._manifest().provenance, "train_only": False})
        with self.assertRaises(ValueError):
            self._manifest(provenance={**self._manifest().provenance, "config_sha256": "b" * 64})


if __name__ == "__main__":
    unittest.main()
