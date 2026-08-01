from types import SimpleNamespace
import unittest

import numpy as np

from experiments.exp421_mps_batched_acceleration.config import Exp421Config
from experiments.exp421_mps_batched_acceleration.contracts import (
    BackendMeasurement,
    BatchKey,
    MpsAssessment,
    ParityReport,
)
from experiments.exp421_mps_batched_acceleration.policy import (
    BackendPolicyBundle,
    assess_mps_capability,
    capability_provenance,
    choose_backend,
    validate_mps_capability,
)


class BackendPolicyTests(unittest.TestCase):
    def _key(self, batch_size):
        return BatchKey(batch_size, 41701, 41711)

    def _measurement(self, backend, batch_size, total):
        transfer = total * 0.25
        compute = total * 0.5
        synchronize = total - transfer - compute
        return BackendMeasurement(
            batch_size=batch_size,
            backend=backend,
            transfer_seconds=transfer,
            compute_seconds=compute,
            synchronize_seconds=synchronize,
            total_seconds=total,
            finite=True,
            candidate_coordinates_sha256="a" * 64,
        )

    def _parity(self, batch_size, *, error=1e-5, spearman=1.0, mismatch=False):
        cpu = np.array([1, 4], dtype=np.int64)
        mps = np.array([1, 5], dtype=np.int64) if mismatch else np.array([1, 4], dtype=np.int64)
        return ParityReport(
            batch_key=self._key(batch_size),
            cpu_candidate_indices_by_alpha={0.01: cpu},
            mps_candidate_indices_by_alpha={0.01: mps},
            max_normalized_error=error,
            spearman=spearman,
        )

    def _assessment(self, batch_size, *, available=True, use_mps=None, fallback=None,
                    error=1e-5, spearman=1.0, mps_total=0.2):
        cpu = self._measurement("cpu", batch_size, 0.4)
        if not available:
            return MpsAssessment(
                batch_key=self._key(batch_size),
                cpu_measurement=cpu,
                mps_measurement=None,
                parity_report=None,
                speedup=None,
                uses_mps=False,
                fallback_reason=fallback or "mps_unavailable",
                mps_available=False,
                device="unavailable",
            )
        mps = self._measurement("mps", batch_size, mps_total)
        report = self._parity(batch_size, error=error, spearman=spearman, mismatch=False)
        eligible = batch_size != 1 and report.passes and (0.4 / mps_total) >= 1.1
        return MpsAssessment(
            batch_key=self._key(batch_size),
            cpu_measurement=cpu,
            mps_measurement=mps,
            parity_report=report,
            speedup=0.4 / mps_total,
            uses_mps=eligible if use_mps is None else use_mps,
            fallback_reason=None if eligible and use_mps is None else (fallback or "gate_failed"),
            mps_available=True,
            device="Apple M4",
        )

    def _all_assessments(self, **kwargs):
        return {
            size: self._assessment(size, **kwargs)
            for size in Exp421Config().candidate_batch_sizes
        }

    def test_choose_backend_requires_complete_frozen_batch_coverage(self):
        measurements = self._all_assessments()
        del measurements[32]
        with self.assertRaises(ValueError):
            choose_backend(measurements, Exp421Config())

    def test_complete_measurements_are_order_independent(self):
        measurements = self._all_assessments()
        reversed_measurements = dict(reversed(tuple(measurements.items())))
        result = choose_backend(reversed_measurements, Exp421Config())
        self.assertEqual(tuple(result.policies), (1, 4, 8, 16, 32))

    def test_batch_one_is_cpu_even_when_every_other_gate_passes(self):
        result = choose_backend(self._all_assessments(), Exp421Config())
        self.assertIsInstance(result, BackendPolicyBundle)
        self.assertEqual(result.policies[1].backend, "cpu")
        self.assertEqual(result.reasons[1], "batch_size_one_cpu_mandatory")
        self.assertTrue(result.fail_closed)

    def test_mps_is_selected_only_for_complete_passing_nontrivial_batches(self):
        result = choose_backend(self._all_assessments(), Exp421Config())
        self.assertEqual(
            {size: result.policies[size].backend for size in (1, 4, 8, 16, 32)},
            {1: "cpu", 4: "mps", 8: "mps", 16: "mps", 32: "mps"},
        )
        self.assertEqual(result.reasons[4], "eligible")

    def test_unavailable_mps_falls_back_to_cpu_with_explicit_reason(self):
        measurements = self._all_assessments()
        measurements[8] = self._assessment(8, available=False)
        result = choose_backend(measurements, Exp421Config())
        self.assertEqual(result.policies[8].backend, "cpu")
        self.assertEqual(result.reasons[8], "mps_unavailable")

    def test_failed_parity_error_rank_and_speed_gates_are_cpu_reasons(self):
        cases = {
            4: self._assessment(4, error=1.1e-4, fallback="normalized_error_above_1e-4"),
            8: self._assessment(8, spearman=0.998, fallback="spearman_below_0.999"),
            16: self._assessment(16, mps_total=0.37, fallback="speedup_below_1.1"),
        }
        measurements = self._all_assessments()
        measurements.update(cases)
        result = choose_backend(measurements, Exp421Config())
        for size, reason in (
            (4, "normalized_error_above_1e-4"),
            (8, "spearman_below_0.999"),
            (16, "speedup_below_1.1"),
        ):
            self.assertEqual(result.policies[size].backend, "cpu")
            self.assertEqual(result.reasons[size], reason)

    def test_policy_rejects_coordinate_mismatch_at_any_preregistered_alpha(self):
        measurements = self._all_assessments()
        malformed = object.__new__(MpsAssessment)
        object.__setattr__(malformed, "batch_key", self._key(4))
        object.__setattr__(malformed, "cpu_measurement", self._measurement("cpu", 4, 0.4))
        object.__setattr__(malformed, "mps_measurement", self._measurement("mps", 4, 0.2))
        object.__setattr__(malformed, "parity_report", self._parity(4, mismatch=True))
        object.__setattr__(malformed, "speedup", 2.0)
        object.__setattr__(malformed, "uses_mps", True)
        object.__setattr__(malformed, "fallback_reason", None)
        object.__setattr__(malformed, "mps_available", True)
        object.__setattr__(malformed, "device", "Apple M4")
        measurements[4] = malformed
        with self.assertRaises(ValueError):
            choose_backend(measurements, Exp421Config())

    def test_policy_rejects_nonfinite_and_missing_stage_timings(self):
        measurements = self._all_assessments()
        malformed = object.__new__(MpsAssessment)
        object.__setattr__(malformed, "batch_key", self._key(4))
        object.__setattr__(malformed, "cpu_measurement", SimpleNamespace(
            batch_size=4, backend="cpu", transfer_seconds=None,
            compute_seconds=0.2, synchronize_seconds=0.1,
            total_seconds=0.4, finite=True,
            candidate_coordinates_sha256="a" * 64,
        ))
        object.__setattr__(malformed, "mps_measurement", measurements[4].mps_measurement)
        object.__setattr__(malformed, "parity_report", measurements[4].parity_report)
        object.__setattr__(malformed, "speedup", 2.0)
        object.__setattr__(malformed, "uses_mps", True)
        object.__setattr__(malformed, "fallback_reason", None)
        object.__setattr__(malformed, "mps_available", True)
        object.__setattr__(malformed, "device", "Apple M4")
        measurements[4] = malformed
        with self.assertRaises(ValueError):
            choose_backend(measurements, Exp421Config())

    def test_nonfinite_mps_output_falls_back_to_cpu(self):
        measurements = self._all_assessments()
        source = measurements[4]
        malformed = object.__new__(MpsAssessment)
        object.__setattr__(malformed, "batch_key", source.batch_key)
        object.__setattr__(malformed, "cpu_measurement", source.cpu_measurement)
        nonfinite = object.__new__(BackendMeasurement)
        for field in (
            "batch_size", "backend", "transfer_seconds", "compute_seconds",
            "synchronize_seconds", "total_seconds", "candidate_coordinates_sha256",
        ):
            object.__setattr__(nonfinite, field, getattr(source.mps_measurement, field))
        object.__setattr__(nonfinite, "finite", False)
        object.__setattr__(malformed, "mps_measurement", nonfinite)
        object.__setattr__(malformed, "parity_report", source.parity_report)
        object.__setattr__(malformed, "speedup", source.speedup)
        object.__setattr__(malformed, "uses_mps", False)
        object.__setattr__(malformed, "fallback_reason", "nonfinite_output")
        object.__setattr__(malformed, "mps_available", True)
        object.__setattr__(malformed, "device", source.device)
        measurements[4] = malformed
        result = choose_backend(measurements, Exp421Config())
        self.assertEqual(result.policies[4].backend, "cpu")
        self.assertEqual(result.reasons[4], "nonfinite_output")

    def test_config_and_assessment_fields_cannot_be_retrofitted_by_input(self):
        config = Exp421Config()
        result = choose_backend(self._all_assessments(), config)
        self.assertEqual(config.candidate_batch_sizes, (1, 4, 8, 16, 32))
        self.assertEqual(config.max_normalized_error, 1e-4)
        self.assertEqual(config.min_spearman, 0.999)
        self.assertEqual(config.min_speedup, 1.1)
        self.assertEqual(config.frozen_model_seeds, (41701, 41702, 41703))
        self.assertEqual(config.frozen_perturbation_seeds, (41711, 41712, 41713))
        with self.assertRaises(TypeError):
            result.policies[1] = result.policies[1]

    def test_capability_assessment_records_mps_and_mlx_provenance_without_importing_mlx(self):
        capability = assess_mps_capability()
        self.assertIsInstance(capability.torch_version, str)
        self.assertIsInstance(capability.mps_built, bool)
        self.assertIsInstance(capability.mps_available, bool)
        self.assertTrue(capability.device_name)
        self.assertIsInstance(capability.mlx_version, str)
        self.assertIsInstance(capability.mlx_metal_version, str)
        self.assertIn(capability.mlx_eligibility_reason, {"package_absent", "algorithm_rewrite_required"})
        self.assertIn(capability.mlx_availability, {"available", "unavailable"})

    def test_capability_rejects_inconsistent_mlx_provenance(self):
        capability = assess_mps_capability()
        values = capability.__dict__.copy()
        values.update(mlx_availability="unavailable", mlx_version="0.0.0")
        with self.assertRaises(ValueError):
            type(capability)(**values)

    def test_capability_provenance_is_immutable_and_revalidated(self):
        capability = assess_mps_capability()
        checked = validate_mps_capability(capability)
        provenance = capability_provenance(checked)
        self.assertEqual(provenance["torch_version"], checked.torch_version)
        with self.assertRaises(TypeError):
            provenance["torch_version"] = "changed"


if __name__ == "__main__":
    unittest.main()
