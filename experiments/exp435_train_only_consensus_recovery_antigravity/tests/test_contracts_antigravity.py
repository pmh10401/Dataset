"""TDD tests for Exp435's immutable shared prediction contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    BackendDecision,
    CalibratedTrainGroup,
    ChannelCalibration,
    Exp435Candidate,
    Exp435DatasetResult,
    Exp435RunSummary,
    InstanceScoreEvidence,
    PreparedTrainGroup,
    UnavailableTrainGroup,
)


class Exp435ContractTests(unittest.TestCase):
    @staticmethod
    def _evidence(
        *,
        dataset: str = "Demo",
        instance_index: int = 3,
        length: int = 64,
        exclusion_reason: str | None = None,
        prototype_abnormal: bool = True,
        gradient_abnormal: bool = False,
        acf_abnormal: bool = True,
    ) -> InstanceScoreEvidence:
        return InstanceScoreEvidence.fixture(
            dataset=dataset,
            instance_index=instance_index,
            length=length,
            prototype_abnormal=prototype_abnormal,
            gradient_abnormal=gradient_abnormal,
            acf_abnormal=acf_abnormal,
            exclusion_reason=exclusion_reason,
        )

    @classmethod
    def _candidate(cls, evidence: InstanceScoreEvidence) -> Exp435Candidate:
        return Exp435Candidate(
            evidence.dataset,
            evidence.instance_index,
            evidence.length,
            evidence_digest=evidence.evidence_digest,
        )

    @staticmethod
    def _adopted_backend(**changes: object) -> BackendDecision:
        values: dict[str, object] = {
            "backend": "mlx",
            "adopted": True,
            "smoke_seed": 43501,
            "finite_outputs": True,
            "candidate_identity_match": True,
            "parity_passed": True,
            "normalized_error_max": 1e-6,
            "absolute_error_max": 1e-6,
            "speedup": 1.2,
            "batch_size": 16,
            "cpu_median_seconds": 0.12,
            "mlx_median_seconds": 0.1,
            "mlx_version": "0.29.0",
            "mlx_metal_version": "0.29.0",
            "device": "gpu",
            "timings": ((16, 0.12, 0.1),),
            "reason": "adopted_after_parity_and_speed_gate",
        }
        values.update(changes)
        return BackendDecision(**values)  # type: ignore[arg-type]

    @staticmethod
    def _complete_summary(**changes: object) -> Exp435RunSummary:
        values: dict[str, object] = {
            "run_identity": "exp435-complete",
            "status": "complete",
            "total_datasets": 1117,
            "completed_datasets": 1117,
            "successful_datasets": 1117,
            "abstained_datasets": 0,
            "failed_datasets": 0,
            "candidate_count": 0,
            "backend": BackendDecision.cpu_fallback("mlx_unavailable"),
            "prediction_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "artifact_sha256": "c" * 64,
        }
        values.update(changes)
        return Exp435RunSummary(**values)  # type: ignore[arg-type]

    def test_candidate_requires_prototype_and_one_secondary_channel(self) -> None:
        evidence = InstanceScoreEvidence.fixture(
            prototype_abnormal=True, gradient_abnormal=False, acf_abnormal=True
        )
        self.assertTrue(evidence.consensus_candidate)
        with self.assertRaisesRegex(ValueError, "consensus"):
            replace(evidence, consensus_candidate=False)

    def test_consensus_rejects_prototype_or_secondary_absence(self) -> None:
        for changes in (
            {"prototype_abnormal": False, "gradient_abnormal": True, "acf_abnormal": True},
            {"prototype_abnormal": True, "gradient_abnormal": False, "acf_abnormal": False},
        ):
            evidence = InstanceScoreEvidence.fixture(**changes)
            self.assertFalse(evidence.consensus_candidate)
            with self.assertRaisesRegex(ValueError, "consensus"):
                replace(evidence, consensus_candidate=True)

    def test_dataset_result_rejects_duplicate_coordinates(self) -> None:
        evidence = self._evidence()
        candidate = self._candidate(evidence)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Exp435DatasetResult.fixture(candidates=(candidate, candidate), evidence=(evidence,))

    def test_dataset_result_validates_order_dataset_and_test_bounds(self) -> None:
        first_evidence = self._evidence(instance_index=3)
        second_evidence = self._evidence(instance_index=4)
        first = self._candidate(first_evidence)
        second = self._candidate(second_evidence)
        result = Exp435DatasetResult.fixture(
            candidates=(first, second), evidence=(first_evidence, second_evidence)
        )
        self.assertEqual(result.candidates, (first, second))
        for candidates, message in (
            ((second, first), "sorted"),
            ((Exp435Candidate("Other", 3, 64, evidence_digest=first.evidence_digest),), "dataset"),
            ((Exp435Candidate("Demo", 64, 64, evidence_digest=first.evidence_digest),), "TEST"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Exp435DatasetResult.fixture(candidates=candidates, evidence=(first_evidence, second_evidence))

    def test_dataset_result_requires_exact_scored_consensus_evidence(self) -> None:
        valid_evidence = self._evidence()
        valid_candidate = self._candidate(valid_evidence)
        with self.assertRaisesRegex(ValueError, "evidence"):
            Exp435DatasetResult.fixture(candidates=(valid_candidate,))

        cases = (
            (self._evidence(prototype_abnormal=False), "consensus"),
            (self._evidence(instance_index=4), "coordinate"),
            (self._evidence(length=32), "length"),
            (self._evidence(exclusion_reason="frozen_lane"), "exclusion"),
            (self._evidence(), "digest"),
        )
        for evidence, message in cases:
            if message == "digest":
                candidate = replace(valid_candidate, evidence_digest="f" * 64)
                evidence = valid_evidence
            else:
                candidate = self._candidate(valid_evidence)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Exp435DatasetResult.fixture(candidates=(candidate,), evidence=(evidence,))

    def test_abstained_and_failed_results_cannot_publish_candidates(self) -> None:
        evidence = self._evidence()
        candidate = self._candidate(evidence)
        for status in ("abstained", "failed"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "candidate"):
                Exp435DatasetResult.fixture(status=status, candidates=(candidate,), evidence=(evidence,))

    def test_published_records_contain_only_tuples_and_scalars(self) -> None:
        evidence = self._evidence()
        candidate = self._candidate(evidence)
        result = Exp435DatasetResult.fixture(candidates=(candidate,), evidence=(evidence,))
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.evidence, tuple)
        self.assertNotIsInstance(result.to_payload()["candidates"], np.ndarray)
        with self.assertRaises(FrozenInstanceError):
            result.dataset = "changed"  # type: ignore[misc]

    def test_group_arrays_are_float32_and_read_only(self) -> None:
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        group = PreparedTrainGroup("Demo", 4, (0, 1, 2, 3), values)
        self.assertEqual(group.values.dtype, np.dtype(np.float32))
        self.assertFalse(group.values.flags.writeable)
        with self.assertRaises(ValueError):
            group.values[0, 0] = 99.0
        calibration = ChannelCalibration.fixture("prototype", reference_count=4)
        calibrated = CalibratedTrainGroup(
            dataset="Demo",
            length=4,
            instance_indices=(0, 1, 2, 3),
            values=group.values,
            gradients=None,
            acf_values=None,
            prototype=calibration,
            gradient=ChannelCalibration.unavailable("gradient", "not_fitted"),
            acf=ChannelCalibration.unavailable("acf", "not_fitted"),
        )
        self.assertFalse(calibrated.values.flags.writeable)

    def test_unavailable_train_group_is_immutable_and_digest_bound(self) -> None:
        outcome = UnavailableTrainGroup(
            dataset="Demo",
            length=1,
            instance_indices=(4, 5),
            reason="insufficient_series_length",
        )
        self.assertFalse(outcome.available)
        self.assertEqual(outcome.reference_count, 2)
        self.assertEqual(outcome, UnavailableTrainGroup("Demo", 1, (4, 5), "insufficient_series_length"))
        self.assertEqual(len(outcome.identity_digest), 64)
        self.assertEqual(outcome.evidence_digest, outcome.identity_digest)
        with self.assertRaisesRegex(ValueError, "identity_digest"):
            replace(outcome, reason="insufficient_train_rows")
        with self.assertRaises(FrozenInstanceError):
            outcome.reason = "changed"  # type: ignore[misc]

    def test_unavailable_train_group_rejects_unknown_reason_and_bad_order(self) -> None:
        for changes, message in (
            ({"reason": "not_preregistered"}, "reason"),
            ({"instance_indices": (2, 1)}, "sorted"),
            ({"instance_indices": ()}, "non-empty"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                fields: dict[str, object] = {
                    "dataset": "Demo",
                    "length": 1,
                    "instance_indices": (1, 2),
                    "reason": "insufficient_series_length",
                }
                fields.update(changes)
                UnavailableTrainGroup(**fields)  # type: ignore[arg-type]

    def test_backend_and_run_summary_are_validated_and_immutable(self) -> None:
        backend = BackendDecision.cpu_fallback("mlx_unavailable")
        self.assertEqual(backend.backend, "cpu")
        self.assertFalse(backend.adopted)
        self.assertEqual(backend.smoke_seed, 43501)
        self.assertIn("normalized_error_max", backend.to_dict())
        summary = Exp435RunSummary(
            run_identity="exp435-test",
            status="running",
            total_datasets=1,
            completed_datasets=1,
            successful_datasets=1,
            abstained_datasets=0,
            failed_datasets=0,
            candidate_count=0,
            backend=backend,
        )
        self.assertEqual(summary.remaining_datasets, 0)
        with self.assertRaises(FrozenInstanceError):
            summary.status = "failed"  # type: ignore[misc]

    def test_backend_adoption_enforces_every_frozen_gate(self) -> None:
        valid = self._adopted_backend()
        self.assertTrue(valid.adopted)
        invalid_cases = (
            ({"smoke_seed": 43502}, "seed"),
            ({"finite_outputs": False}, "finite"),
            ({"candidate_identity_match": False}, "candidate"),
            ({"parity_passed": False}, "parity"),
            ({"normalized_error_max": None}, "normalized"),
            ({"normalized_error_max": 1.0001e-5}, "normalized"),
            ({"absolute_error_max": None}, "absolute"),
            ({"absolute_error_max": 1.0001e-5}, "absolute"),
            ({"speedup": None}, "speed"),
            ({"speedup": 1.099}, "speed"),
            ({"mlx_version": None}, "version"),
            ({"mlx_metal_version": None}, "metal"),
            ({"device": None}, "device"),
            ({"timings": ()}, "timing"),
            ({"timings": ((16, 0.0, 0.1),)}, "timing"),
        )
        for changes, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replace(valid, **changes)

        with self.assertRaisesRegex(ValueError, "reason"):
            BackendDecision(backend="cpu", adopted=False, reason="")

    def test_backend_timing_evidence_is_canonical_and_self_consistent(self) -> None:
        valid = self._adopted_backend()
        self.assertEqual(valid.selected_batch_size, 16)
        self.assertEqual(valid.batch_size, valid.selected_batch_size)
        with self.assertRaisesRegex(ValueError, "speedup"):
            self._adopted_backend(
                speedup=99.0,
                cpu_median_seconds=0.10,
                mlx_median_seconds=0.20,
                timings=((16, 0.10, 0.20),),
            )
        with self.assertRaisesRegex(ValueError, "median"):
            self._adopted_backend(timings=((16, 0.11, 0.10),))
        with self.assertRaisesRegex(ValueError, "selected"):
            self._adopted_backend(timings=((8, 0.12, 0.10),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._adopted_backend(timings=((16, 0.12, 0.10), (16, 0.12, 0.10)))
        with self.assertRaisesRegex(ValueError, "speedup"):
            BackendDecision(
                backend="cpu",
                adopted=False,
                speedup=99.0,
                cpu_median_seconds=0.10,
                mlx_median_seconds=0.20,
                selected_batch_size=16,
                timings=((16, 0.10, 0.20),),
                reason="recorded_benchmark_not_adopted",
            )

    def test_available_calibration_matches_train_count_and_higher_quantile(self) -> None:
        loo_scores = (0.0, 0.1, 0.2, 0.3)
        threshold = float(np.quantile(np.asarray(loo_scores, dtype=np.float32), 0.99, method="higher"))
        calibration = ChannelCalibration(
            channel="prototype",
            threshold=threshold,
            loo_scores=loo_scores,
            center=0.0,
            scale=1.0,
            reference_count=4,
        )
        self.assertEqual(calibration.threshold, threshold)
        for changes, message in (
            ({"reference_count": 3}, "reference_count"),
            ({"loo_scores": loo_scores + (0.4,)}, "reference_count"),
            ({"threshold": threshold - 0.1}, "quantile"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replace(calibration, **changes)

    def test_unavailable_calibration_has_no_usable_scores_or_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ChannelCalibration(
                channel="gradient",
                available=False,
                loo_scores=(0.0, 0.1, 0.2, 0.3),
                reason="zero_scale",
            )
        with self.assertRaisesRegex(ValueError, "reference_count"):
            ChannelCalibration(
                channel="gradient",
                available=False,
                reference_count=1,
                reason="zero_scale",
            )

    def test_calibrated_group_couples_arrays_availability_shapes_and_counts(self) -> None:
        values = np.ones((4, 16), dtype=np.float32)
        gradients = np.ones_like(values)
        acf_values = np.ones((4, 4), dtype=np.float32)
        prototype = ChannelCalibration.fixture("prototype", reference_count=4)
        gradient = ChannelCalibration.fixture("gradient", reference_count=4)
        acf = ChannelCalibration.fixture("acf", reference_count=4)

        def group(**changes: object) -> CalibratedTrainGroup:
            fields: dict[str, object] = {
                "dataset": "Demo",
                "length": 16,
                "instance_indices": (0, 1, 2, 3),
                "values": values,
                "gradients": gradients,
                "acf_values": acf_values,
                "prototype": prototype,
                "gradient": gradient,
                "acf": acf,
            }
            fields.update(changes)
            return CalibratedTrainGroup(**fields)  # type: ignore[arg-type]

        group()
        invalid_cases = (
            ({"prototype": ChannelCalibration.unavailable("prototype", "zero_scale")}, "prototype"),
            ({"gradients": None}, "gradient"),
            ({"gradient": ChannelCalibration.unavailable("gradient", "zero_scale")}, "gradient"),
            ({"acf_values": None}, "acf"),
            ({"acf": ChannelCalibration.unavailable("acf", "short_series")}, "acf"),
            ({"length": 8, "values": np.ones((4, 8), dtype=np.float32), "gradients": np.ones((4, 8), dtype=np.float32), "acf_values": np.ones((4, 2), dtype=np.float32)}, "ACF"),
            (
                {
                    "instance_indices": (0, 1, 2, 3, 4),
                    "values": np.ones((5, 16), dtype=np.float32),
                    "gradients": np.ones((5, 16), dtype=np.float32),
                    "acf_values": np.ones((5, 4), dtype=np.float32),
                },
                "reference",
            ),
            ({"gradients": np.ones((3, 16), dtype=np.float32)}, "shape"),
        )
        for changes, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                group(**changes)

    def test_evidence_digest_must_match_canonical_payload(self) -> None:
        evidence = self._evidence()
        with self.assertRaisesRegex(ValueError, "digest"):
            replace(evidence, evidence_digest="f" * 64)

    def test_complete_summary_requires_full_population_and_frozen_artifacts(self) -> None:
        complete = self._complete_summary()
        self.assertEqual(complete.total_datasets, 1117)
        self.assertEqual(complete.artifact_sha256, "c" * 64)
        invalid_cases = (
            ({"total_datasets": 1116}, "complete"),
            ({"completed_datasets": 1116}, "complete"),
            ({"failed_datasets": 1, "successful_datasets": 1116}, "complete"),
            ({"candidate_count": 101}, "budget"),
            ({"prediction_sha256": None}, "hash"),
            ({"manifest_sha256": None}, "hash"),
            ({"artifact_sha256": None}, "hash"),
            ({"artifact_sha256": "Z" * 64}, "SHA"),
        )
        for changes, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replace(complete, **changes)


if __name__ == "__main__":
    unittest.main()
