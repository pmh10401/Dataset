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

    def test_train_arrays_use_an_immutable_bytes_backing_store(self) -> None:
        group = PreparedTrainGroup("Demo", 2, (1, 3, 5, 7), np.ones((4, 2)))
        self.assertEqual(group.values.dtype, np.dtype(np.float32))
        self.assertTrue(group.values.flags.c_contiguous)
        with self.assertRaises(ValueError):
            group.values.setflags(write=True)

    @staticmethod
    def _batch(batch_size: int, cpu: float = 2.0, device: float = 1.0) -> BackendBatchEvidence:
        pairs = tuple((cpu, device) for _ in range(5))
        return BackendBatchEvidence(
            backend="mlx", batch_size=batch_size, timing_pairs=pairs,
            finite_outputs=True, candidate_identity_match=True,
            absolute_error_max=1e-6, normalized_error_max=1e-6,
            fixture_input_sha256="a" * 64, fixture_output_sha256="b" * 64,
            cpu_median_seconds=cpu, device_median_seconds=device, speedup=cpu / device,
            parity_passed=True, speed_passed=True, passed=True, reason="passed",
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
        benchmark = BackendBenchmarkEvidence.fixture("mlx", batches=batches)
        self.assertEqual(benchmark.selected_batch_size, 16)
        with self.assertRaisesRegex(ValueError, "exactly"):
            BackendBenchmarkEvidence.fixture("mlx", batches=batches[:-1])

    def test_backend_decision_uses_lexical_tie_and_cpu_fallback(self) -> None:
        mlx = BackendBenchmarkEvidence.fixture("mlx")
        mps = BackendBenchmarkEvidence.fixture("mps")
        decision = BackendDecision(mlx, mps, "mlx", True, "selected")
        self.assertEqual(decision.outer_worker_count, 1)
        failed_mlx = BackendBenchmarkEvidence.fixture("mlx", batches=())
        failed_mps = BackendBenchmarkEvidence.fixture("mps", batches=())
        cpu = BackendDecision(failed_mlx, failed_mps, "cpu", False, "both_failed")
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
            replace(BackendBenchmarkEvidence.fixture("mlx"), metal_available=False)

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
        benchmark = BackendBenchmarkEvidence.fixture("mlx")
        decision = BackendDecision(benchmark, BackendBenchmarkEvidence.fixture("mps"), "mlx", True, "selected")
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
            replace(BackendBenchmarkEvidence.fixture("mps"), protocol="invented")

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
            BackendBenchmarkEvidence.fixture("mlx"), selected_batch_size=np.int64(16),
            selected_device_median_seconds=np.float32(1.0),
        )
        self.assertIs(type(benchmark.selected_batch_size), int)
        self.assertIs(type(benchmark.selected_device_median_seconds), float)
        decision = BackendDecision(benchmark, BackendBenchmarkEvidence.fixture("mps"), "mlx", True, "selected")
        for payload in (batch.to_payload(), benchmark.to_payload(), decision.to_payload()):
            json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_near_tolerance_backend_declarations_are_not_retained(self) -> None:
        base = self._batch(16)
        canonical = replace(base, cpu_median_seconds=2.0 + 1e-13, speedup=2.0 + 1e-13)
        self.assertEqual(canonical.cpu_median_seconds, 2.0)
        self.assertEqual(canonical.speedup, 2.0)
        with self.assertRaisesRegex(ValueError, "median"):
            replace(base, cpu_median_seconds=2.1)
