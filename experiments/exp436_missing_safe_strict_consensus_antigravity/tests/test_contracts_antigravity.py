"""Adversarial contract tests for the Exp436 missing-safe boundary."""

from dataclasses import replace
import json
import math
import unittest

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    BackendBatchEvidence,
    BackendBenchmarkEvidence,
    BackendDecision,
    ChannelCalibration,
    CalibratedTrainGroup,
    ExcludedTrainRowAudit,
    Exp436Candidate,
    Exp436DatasetResult,
    Exp436RunSummary,
    InstanceScoreEvidence,
    MissingSafeRawDataset,
    PreparedTrainGroup,
    RawValueSeries,
    TrainPreparation,
    UnavailableTrainGroup,
)


class Exp436ContractsTests(unittest.TestCase):
    @staticmethod
    def _evidence(index: int, **changes: object) -> InstanceScoreEvidence:
        values: dict[str, object] = {
            "dataset": "Demo",
            "instance_index": index,
            "length": 16,
            "prototype_abnormal": True,
            "gradient_abnormal": True,
            "acf_abnormal": True,
        }
        values.update(changes)
        return InstanceScoreEvidence.fixture(**values)  # type: ignore[arg-type]

    def test_raw_preserves_nonfinite_values_but_scalars_reject_them(self) -> None:
        raw = RawValueSeries(2, (1, float("nan"), float("inf")), "a" * 64)
        dataset = MissingSafeRawDataset("Demo", "TEST", (raw,))
        self.assertTrue(math.isnan(dataset.series[0].values[1]))
        self.assertTrue(math.isinf(dataset.series[0].values[2]))
        with self.assertRaisesRegex(ValueError, "prototype_score"):
            replace(self._evidence(2), prototype_score=float("nan"))
        with self.assertRaisesRegex(ValueError, "numeric"):
            RawValueSeries(3, ("1.0",), "a" * 64)

    def test_strict_and_rejects_or_consensus(self) -> None:
        evidence = self._evidence(2, acf_abnormal=False)
        self.assertFalse(evidence.consensus_candidate)
        with self.assertRaisesRegex(ValueError, "consensus"):
            replace(evidence, consensus_candidate=True)

    def test_nonfinite_test_abstention_can_never_be_a_candidate(self) -> None:
        # A deliberately contradictory raw-Test marker must fail even though
        # the three finite channel fixtures otherwise agree.
        with self.assertRaisesRegex(ValueError, "nonfinite_test"):
            self._evidence(2, exclusion_reason="nonfinite_test")

    def test_instance_evidence_rejects_per_row_failed_and_unknown_reasons(self) -> None:
        with self.assertRaisesRegex(ValueError, "status"):
            InstanceScoreEvidence.fixture(status="failed")
        with self.assertRaisesRegex(ValueError, "exclusion"):
            InstanceScoreEvidence.fixture(exclusion_reason="invented")
        with self.assertRaisesRegex(ValueError, "unavailability"):
            InstanceScoreEvidence(
                dataset="Demo", instance_index=3, length=16,
                prototype_unavailability_reason="invented",
                status="abstained", exclusion_reason="three_channel_consensus_unavailable",
            )

    def test_channel_unavailability_preserves_only_calculable_siblings(self) -> None:
        evidence = InstanceScoreEvidence(
            dataset="Demo", instance_index=3, length=16,
            prototype_unavailability_reason="zero_scale",
            gradient_score=2.0, gradient_threshold=1.0, gradient_abnormal=True,
            acf_score=0.0, acf_threshold=1.0, acf_abnormal=False,
            status="abstained", exclusion_reason="three_channel_consensus_unavailable",
        )
        self.assertEqual(evidence.prototype_unavailability_reason, "zero_scale")
        self.assertEqual(evidence.gradient_score, 2.0)
        self.assertIn("prototype_unavailability_reason", evidence.to_payload())
        with self.assertRaisesRegex(ValueError, "scalar-free"):
            InstanceScoreEvidence.fixture(status="abstained", exclusion_reason="missing_length")

    def test_acf_unavailability_reason_matches_test_length_context(self) -> None:
        def evidence(length: int, reason: str) -> InstanceScoreEvidence:
            return InstanceScoreEvidence(
                dataset="Demo", instance_index=3, length=length,
                prototype_score=0.0, prototype_threshold=1.0,
                gradient_score=0.0, gradient_threshold=1.0,
                acf_unavailability_reason=reason,
                status="abstained", exclusion_reason="three_channel_consensus_unavailable",
            )
        self.assertEqual(evidence(15, "short_series").acf_unavailability_reason, "short_series")
        self.assertEqual(evidence(16, "zero_acf_energy").acf_unavailability_reason, "zero_acf_energy")
        with self.assertRaisesRegex(ValueError, "acf unavailability"):
            evidence(16, "short_series")
        with self.assertRaisesRegex(ValueError, "acf unavailability"):
            evidence(15, "zero_acf_energy")

    def test_routing_reason_matches_raw_length_context(self) -> None:
        def evidence(length: int, reason: str) -> InstanceScoreEvidence:
            return InstanceScoreEvidence(
                "Demo", 3, length, status="abstained", exclusion_reason=reason
            )
        for length, reason in ((1, "nonfinite_test"), (1, "insufficient_series_length"),
                               (2, "missing_length"), (2, "insufficient_finite_train_rows")):
            with self.subTest(length=length, reason=reason):
                self.assertEqual(evidence(length, reason).exclusion_reason, reason)
        for length, reason in ((2, "insufficient_series_length"), (1, "missing_length"),
                               (1, "insufficient_finite_train_rows")):
            with self.subTest(length=length, reason=reason):
                with self.assertRaisesRegex(ValueError, "routing"):
                    evidence(length, reason)

    def test_dataset_requires_exact_non_contiguous_test_identity_coverage(self) -> None:
        first, second = self._evidence(2), self._evidence(9, prototype_abnormal=False)
        candidate = Exp436Candidate("Demo", 2, 16, first.evidence_digest)
        result = Exp436DatasetResult(
            "Demo", "scored", (2, 9), (candidate,), (first, second)
        )
        self.assertEqual(result.test_size, 2)
        with self.assertRaisesRegex(ValueError, "exactly"):
            Exp436DatasetResult("Demo", "scored", (2, 9), (), (first,))
        extra = self._evidence(10)
        with self.assertRaisesRegex(ValueError, "exactly"):
            Exp436DatasetResult("Demo", "scored", (2, 9), (), (first, second, extra))

    def test_dataset_result_requires_every_eligible_candidate(self) -> None:
        first, second = self._evidence(2), self._evidence(9)
        candidate = Exp436Candidate("Demo", 2, 16, first.evidence_digest)
        with self.assertRaisesRegex(ValueError, "candidates must exactly"):
            Exp436DatasetResult("Demo", "scored", (2,), (), (first,))
        with self.assertRaisesRegex(ValueError, "candidates must exactly"):
            Exp436DatasetResult("Demo", "scored", (2, 9), (candidate,), (first, second))

    def test_dataset_result_rejects_status_and_abstention_evasion(self) -> None:
        abstained = InstanceScoreEvidence("Demo", 0, 1, status="abstained", exclusion_reason="insufficient_series_length")
        scored = self._evidence(1)
        with self.assertRaisesRegex(ValueError, "scored result requires"):
            Exp436DatasetResult("Demo", "scored", (0,), (), (abstained,))
        with self.assertRaisesRegex(ValueError, "abstained result requires"):
            Exp436DatasetResult("Demo", "abstained", (0, 1), (), (abstained, scored), abstention_reason="all_instances_abstained")
        with self.assertRaisesRegex(ValueError, "all_instances_abstained"):
            Exp436DatasetResult("Demo", "abstained", (0,), (), (abstained,), abstention_reason="wrong_reason")

    def test_dataset_result_rejects_101_eligible_rows_with_only_100_candidates(self) -> None:
        evidence = tuple(self._evidence(index) for index in range(101))
        candidates = tuple(
            Exp436Candidate(item.dataset, item.instance_index, item.length, item.evidence_digest)
            for item in evidence[:100]
        )
        with self.assertRaisesRegex(ValueError, "candidates must exactly"):
            Exp436DatasetResult("Demo", "scored", tuple(range(101)), candidates, evidence)

    def test_train_arrays_use_an_immutable_bytes_backing_store(self) -> None:
        group = PreparedTrainGroup("Demo", 2, (1, 3, 5, 7), np.ones((4, 2)))
        self.assertEqual(group.values.dtype, np.dtype(np.float32))
        self.assertTrue(group.values.flags.c_contiguous)
        with self.assertRaises(ValueError):
            group.values.setflags(write=True)

    def test_calibrated_group_retains_sibling_channel_evidence_when_prototype_is_zero_scale(self) -> None:
        values = np.ones((4, 16), dtype=np.float32)
        calibrated = CalibratedTrainGroup(
            "Demo", 16, (1, 2, 3, 4), None, values, np.ones((4, 4), dtype=np.float32),
            ChannelCalibration.unavailable("prototype", "zero_scale"),
            ChannelCalibration.fixture("gradient"), ChannelCalibration.fixture("acf"),
        )
        self.assertIsNone(calibrated.values)
        self.assertIsNone(calibrated.to_payload()["values"])
        self.assertFalse(calibrated.values is not None)

    def test_calibrated_group_rejects_fewer_than_four_direct_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient references"):
            CalibratedTrainGroup(
                "Demo", 16, (1, 2, 3), None, None, None,
                ChannelCalibration.unavailable("prototype", "zero_scale"),
                ChannelCalibration.unavailable("gradient", "zero_scale"),
                ChannelCalibration.unavailable("acf", "short_series"),
            )

    def test_calibrated_group_rejects_contextually_impossible_unavailable_reasons(self) -> None:
        prototype = ChannelCalibration.unavailable("prototype", "zero_scale")
        valid_gradient = ChannelCalibration.unavailable("gradient", "zero_scale")
        valid_acf = ChannelCalibration.unavailable("acf", "zero_acf_energy")
        for calibration, length, message in (
            (ChannelCalibration.unavailable("prototype", "invented"), 16, "prototype"),
            (ChannelCalibration.unavailable("gradient", "nonfinite_train"), 16, "gradient"),
            (ChannelCalibration.unavailable("gradient", "invented"), 16, "gradient"),
            (ChannelCalibration.unavailable("acf", "short_series"), 16, "acf"),
            (ChannelCalibration.unavailable("acf", "zero_acf_energy"), 15, "acf"),
            (ChannelCalibration.unavailable("acf", "invented"), 16, "acf"),
        ):
            with self.subTest(calibration=calibration.channel, reason=calibration.reason, length=length):
                with self.assertRaisesRegex(ValueError, message):
                    CalibratedTrainGroup(
                        "Demo", length, (1, 2, 3, 4), None, None, None,
                        calibration if calibration.channel == "prototype" else prototype,
                        calibration if calibration.channel == "gradient" else valid_gradient,
                        calibration if calibration.channel == "acf" else valid_acf,
                    )

    @staticmethod
    def _batch(batch_size: int, cpu: float = 2.0, device: float = 1.0) -> BackendBatchEvidence:
        pairs = tuple((cpu, device) for _ in range(5))
        return BackendBatchEvidence(
            backend="mlx", batch_size=batch_size, state="completed", warmup_pairs_completed=2, timing_pairs=pairs,
            finite_outputs=True, candidate_identity_match=True,
            absolute_error_max=1e-6, normalized_error_max=1e-6,
            fixture_input_sha256="a" * 64, fixture_output_sha256="b" * 64,
            cpu_median_seconds=cpu, device_median_seconds=device, speedup=cpu / device,
            parity_passed=True, speed_passed=True, passed=True, reason="passed",
        )

    @classmethod
    def _benchmark(cls, backend: str, batches: tuple[BackendBatchEvidence, ...] | None = None) -> BackendBenchmarkEvidence:
        source = batches or tuple(replace(cls._batch(size), backend=backend) for size in (16, 32, 64, 128, 256))
        passing = tuple(item for item in source if item.passed)
        selected = min(passing, key=lambda item: (item.device_median_seconds, item.batch_size)) if passing else None
        return BackendBenchmarkEvidence(
            backend, True, source, "python", "numpy", "gpu:0" if backend == "mlx" else "mps", "info", True, "paired-synchronized-v1",
            None if selected is None else selected.batch_size,
            None if selected is None else selected.device_median_seconds,
            "no_passing_batch" if selected is None else None,
            mlx_version="mlx" if backend == "mlx" else None,
            mlx_metal_version="metal" if backend == "mlx" else None,
            pytorch_version="torch" if backend == "mps" else None,
        )

    def test_batch_rejects_bad_pair_counts_and_derived_values(self) -> None:
        batch = self._batch(16)
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            replace(batch, timing_pairs=batch.timing_pairs[:-1])
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(batch, timing_pairs=((0.0, 1.0),) * 5)
        with self.assertRaisesRegex(ValueError, "finite"):
            replace(batch, timing_pairs=((float("nan"), 1.0),) * 5)
        with self.assertRaisesRegex(ValueError, "median"):
            replace(batch, cpu_median_seconds=3.0)
        with self.assertRaisesRegex(ValueError, "speedup"):
            replace(batch, speedup=1.9)

    def test_backend_benchmark_requires_complete_batches_and_tie_breaks_batch_size(self) -> None:
        batches = tuple(
            self._batch(size, cpu=(1.0 if size in (16, 32) else 2.0) * 2.0,
                        device=1.0 if size in (16, 32) else 2.0)
            for size in (16, 32, 64, 128, 256)
        )
        benchmark = self._benchmark("mlx", batches=batches)
        self.assertEqual(benchmark.selected_batch_size, 16)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self._benchmark("mlx", batches=batches[:-1])

    def test_backend_decision_uses_lexical_tie_and_cpu_fallback(self) -> None:
        mlx = self._benchmark("mlx")
        mps = self._benchmark("mps")
        decision = BackendDecision(mlx, mps, "mlx", True, "selected_fastest_passing_backend")
        self.assertEqual(decision.outer_worker_count, 1)
        failed_mlx = BackendBenchmarkEvidence.unavailable("mlx", "missing")
        failed_mps = BackendBenchmarkEvidence.unavailable("mps", "missing")
        cpu = BackendDecision(failed_mlx, failed_mps, "cpu", False, "no_accelerator_passed")
        self.assertGreaterEqual(cpu.outer_worker_count, 1)

    def test_channel_calibration_requires_finite_public_scalars(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            ChannelCalibration("prototype", threshold=float("nan"))

    def test_unavailable_group_allows_zero_surviving_finite_references(self) -> None:
        group = UnavailableTrainGroup("Demo", 16, ())
        self.assertEqual(group.instance_indices, ())
        self.assertEqual(group.reference_count, 0)

    def test_nonfinite_test_marker_must_be_a_scalar_free_abstention(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite_test"):
            InstanceScoreEvidence("Demo", 0, 16, status="scored", exclusion_reason="nonfinite_test")
        abstained = InstanceScoreEvidence("Demo", 0, 16, status="abstained", exclusion_reason="nonfinite_test")
        self.assertEqual(abstained.status, "abstained")
        self.assertIsNone(abstained.prototype_score)

    def test_metal_unavailable_benchmark_cannot_pass(self) -> None:
        with self.assertRaisesRegex(ValueError, "metal_available"):
            replace(self._benchmark("mlx"), metal_available=False)

    def test_complete_review_only_summary_is_the_final_publication_state(self) -> None:
        summary = Exp436RunSummary(
            "exp436-run", "complete_review_only", 1117, 1117, 1117, 0, 0, 0,
            BackendDecision.cpu_fallback(), prediction_sha256="a" * 64,
            manifest_sha256="b" * 64, artifact_sha256="c" * 64,
        )
        self.assertEqual(summary.status, "complete_review_only")

    def test_preparation_partitions_every_source_train_identity_and_length(self) -> None:
        prepared = PreparedTrainGroup("Demo", 16, (0, 1, 2, 3), np.ones((4, 16)))
        unavailable = UnavailableTrainGroup("Demo", 17, ())
        audit = ExcludedTrainRowAudit("Demo", 4, 17, 1, "d" * 64)
        preparation = TrainPreparation(
            "Demo", ((0, 16), (1, 16), (2, 16), (3, 16), (4, 17)),
            (prepared, unavailable), (audit,),
        )
        self.assertEqual(preparation.train_identity_lengths[-1], (4, 17))
        with self.assertRaisesRegex(ValueError, "partition"):
            TrainPreparation(
                "Demo", ((0, 16), (1, 16), (2, 16), (3, 16), (4, 17), (5, 17)),
                (prepared, unavailable), (audit,),
            )
        with self.assertRaisesRegex(ValueError, "min_finite_train_rows"):
            PreparedTrainGroup("Demo", 16, (0, 1, 2), np.ones((3, 16)))

    def test_nonfinite_fixture_and_checkpoint_complete_lifecycle(self) -> None:
        evidence = InstanceScoreEvidence.fixture(status="abstained", exclusion_reason="nonfinite_test")
        self.assertIsNone(evidence.acf_score)
        checkpoint = Exp436RunSummary(
            "exp436-run", "checkpoint_complete", 1117, 1117, 1117, 0, 0, 0,
            BackendDecision.cpu_fallback(),
        )
        self.assertEqual(checkpoint.status, "checkpoint_complete")

    def test_public_contract_payloads_are_json_safe_and_matrix_free(self) -> None:
        raw = RawValueSeries(0, (float("nan"), 1.0), "a" * 64)
        source = MissingSafeRawDataset("Demo", "TRAIN", (raw,))
        unavailable = UnavailableTrainGroup("Demo", 2, ())
        audit = ExcludedTrainRowAudit("Demo", 0, 2, 1, "d" * 64)
        preparation = TrainPreparation("Demo", ((0, 2),), (unavailable,), (audit,))
        values = np.ones((4, 16))
        calibrated = CalibratedTrainGroup(
            "Demo", 16, (1, 2, 3, 4), values, values, np.ones((4, 4)),
            ChannelCalibration.fixture("prototype"), ChannelCalibration.fixture("gradient"),
            ChannelCalibration.fixture("acf"),
        )
        evidence = self._evidence(3)
        result = Exp436DatasetResult("Demo", "scored", (3,),
                                      (Exp436Candidate("Demo", 3, 16, evidence.evidence_digest),), (evidence,))
        batch = self._batch(16)
        benchmark = self._benchmark("mlx")
        decision = BackendDecision(benchmark, self._benchmark("mps"), "mlx", True, "selected_fastest_passing_backend")
        summary = Exp436RunSummary("run", "complete_review_only", 1117, 1117, 1117, 0, 0, 0,
                                   BackendDecision.cpu_fallback(), prediction_sha256="a" * 64,
                                   manifest_sha256="b" * 64, artifact_sha256="c" * 64)
        for record in (raw, source, audit, unavailable, preparation, ChannelCalibration.fixture("prototype"),
                       calibrated, evidence, result, batch, benchmark, decision, summary):
            with self.subTest(record=type(record).__name__):
                json.dumps(record.to_payload(), sort_keys=True)
        self.assertEqual(calibrated.to_payload()["values"]["sha256"], __import__("hashlib").sha256(calibrated.values.tobytes()).hexdigest())
        self.assertEqual(decision.to_payload()["selected_batch_size"], 16)

    def test_backend_protocol_is_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol"):
            replace(self._benchmark("mps"), protocol="invented")

    def test_backend_evidence_recomputes_native_scalars_for_strict_json(self) -> None:
        batch = replace(
            self._batch(16), cpu_median_seconds=np.float32(2.0),
            device_median_seconds=np.float32(1.0), speedup=np.float32(2.0),
        )
        self.assertIs(type(batch.cpu_median_seconds), float)
        self.assertIs(type(batch.device_median_seconds), float)
        self.assertIs(type(batch.speedup), float)
        self.assertEqual(batch.cpu_median_seconds, 2.0)
        self.assertEqual(batch.speedup, 2.0)
        benchmark = replace(
            self._benchmark("mlx"), selected_batch_size=np.int64(16),
            selected_device_median_seconds=np.float32(1.0),
        )
        self.assertIs(type(benchmark.selected_batch_size), int)
        self.assertIs(type(benchmark.selected_device_median_seconds), float)
        decision = BackendDecision(benchmark, self._benchmark("mps"), "mlx", True, "selected_fastest_passing_backend")
        for payload in (batch.to_payload(), benchmark.to_payload(), decision.to_payload()):
            json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_near_tolerance_backend_declarations_are_not_retained(self) -> None:
        base = self._batch(16)
        canonical = replace(base, cpu_median_seconds=2.0 + 1e-13, speedup=2.0 + 1e-13)
        self.assertEqual(canonical.cpu_median_seconds, 2.0)
        self.assertEqual(canonical.speedup, 2.0)
        with self.assertRaisesRegex(ValueError, "median"):
            replace(base, cpu_median_seconds=2.1)

    def test_runtime_failed_batch_retains_only_completed_real_pairs(self) -> None:
        failed = BackendBatchEvidence(
            "mlx", 16, "runtime_failed", 1, ((2.0, 1.0),), "a" * 64, "b" * 64,
            None, None, None, None, None, None, None, None, None, False, "mlx_runtime_failed:RuntimeError",
        )
        self.assertEqual(failed.timing_pairs, ((2.0, 1.0),))
        self.assertIsNone(failed.speedup)
        with self.assertRaisesRegex(ValueError, "no derived"):
            replace(failed, speedup=2.0)

    def test_unavailable_backend_has_empty_batches_and_no_invented_selection(self) -> None:
        unavailable = BackendBenchmarkEvidence.unavailable("mps", "framework_unavailable")
        self.assertFalse(unavailable.available)
        self.assertEqual(unavailable.batches, ())
        self.assertIsNone(unavailable.selected_device_median_seconds)

    def test_backend_reasons_and_unavailable_provenance_are_canonical(self) -> None:
        with self.assertRaisesRegex(ValueError, "reason"):
            replace(self._batch(16), reason="invented")
        with self.assertRaisesRegex(ValueError, "canonical"):
            BackendBatchEvidence(
                "mps", 16, "runtime_failed", 0, (), "a" * 64, "b" * 64,
                None, None, None, None, None, None, None, None, None, False, "runtime_failed",
            )
        with self.assertRaisesRegex(ValueError, "cannot claim MLX"):
            BackendBenchmarkEvidence.unavailable("mps", "framework_unavailable", mlx_version="wrong")
        with self.assertRaisesRegex(ValueError, "decision reason"):
            BackendDecision(self._benchmark("mlx"), self._benchmark("mps"), "mlx", True, "selected")

    def test_nonfinite_completed_reason_precedes_every_other_gate(self) -> None:
        fast_nonfinite = replace(
            self._batch(16), finite_outputs=False, parity_passed=False, passed=False,
            reason="nonfinite_output",
        )
        self.assertTrue(fast_nonfinite.speed_passed)
        slow_nonfinite = replace(
            fast_nonfinite, timing_pairs=((1.0, 1.0),) * 5,
            cpu_median_seconds=1.0, device_median_seconds=1.0, speedup=1.0,
            speed_passed=False,
        )
        self.assertFalse(slow_nonfinite.speed_passed)
        with self.assertRaisesRegex(ValueError, "reason"):
            replace(fast_nonfinite, reason="speedup_below_threshold")

    def test_available_benchmark_requires_backend_specific_device_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "device"):
            replace(self._benchmark("mlx"), device="cpu:0")
        with self.assertRaisesRegex(ValueError, "device"):
            replace(self._benchmark("mps"), device="cpu")

    def test_available_benchmark_requires_one_shared_fixture_digest_pair(self) -> None:
        batches = self._benchmark("mlx").batches
        with self.assertRaisesRegex(ValueError, "fixture"):
            self._benchmark("mlx", batches=(batches[0], replace(batches[1], fixture_input_sha256="c" * 64), *batches[2:]))
        with self.assertRaisesRegex(ValueError, "fixture"):
            self._benchmark("mlx", batches=(batches[0], replace(batches[1], fixture_output_sha256="d" * 64), *batches[2:]))

    def test_available_backends_require_matching_fixture_digest_pairs_even_when_one_fails(self) -> None:
        mlx = self._benchmark("mlx")
        mismatched_mps = self._benchmark(
            "mps", batches=tuple(replace(batch, fixture_input_sha256="c" * 64) for batch in self._benchmark("mps").batches),
        )
        with self.assertRaisesRegex(ValueError, "fixture"):
            BackendDecision(mlx, mismatched_mps, "mlx", True, "selected_fastest_passing_backend")

        failed_mps = self._benchmark(
            "mps", batches=tuple(
                replace(
                    batch, fixture_output_sha256="d" * 64, timing_pairs=((1.0, 1.0),) * 5,
                    cpu_median_seconds=1.0, device_median_seconds=1.0, speedup=1.0,
                    speed_passed=False, passed=False, reason="speedup_below_threshold",
                )
                for batch in self._benchmark("mps").batches
            ),
        )
        with self.assertRaisesRegex(ValueError, "fixture"):
            BackendDecision(mlx, failed_mps, "mlx", True, "selected_fastest_passing_backend")
