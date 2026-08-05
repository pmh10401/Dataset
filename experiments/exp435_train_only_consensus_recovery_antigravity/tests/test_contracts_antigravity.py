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
)


class Exp435ContractTests(unittest.TestCase):
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
        candidate = Exp435Candidate("Demo", 3, 64, evidence_digest="a" * 64)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Exp435DatasetResult.fixture(candidates=(candidate, candidate))

    def test_dataset_result_validates_order_dataset_and_test_bounds(self) -> None:
        first = Exp435Candidate("Demo", 3, 64, evidence_digest="a" * 64)
        second = Exp435Candidate("Demo", 4, 64, evidence_digest="b" * 64)
        result = Exp435DatasetResult.fixture(candidates=(first, second))
        self.assertEqual(result.candidates, (first, second))
        for candidates, message in (
            ((second, first), "sorted"),
            ((Exp435Candidate("Other", 3, 64, evidence_digest="a" * 64),), "dataset"),
            ((Exp435Candidate("Demo", 64, 64, evidence_digest="a" * 64),), "TEST"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                Exp435DatasetResult.fixture(candidates=candidates)

    def test_abstained_and_failed_results_cannot_publish_candidates(self) -> None:
        candidate = Exp435Candidate("Demo", 3, 64, evidence_digest="a" * 64)
        for status in ("abstained", "failed"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "candidate"):
                Exp435DatasetResult.fixture(status=status, candidates=(candidate,))

    def test_published_records_contain_only_tuples_and_scalars(self) -> None:
        evidence = InstanceScoreEvidence.fixture()
        candidate = Exp435Candidate("Demo", 3, 64, evidence_digest=evidence.evidence_digest)
        result = Exp435DatasetResult.fixture(candidates=(candidate,), evidence=(evidence,))
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.evidence, tuple)
        self.assertNotIsInstance(result.to_payload()["candidates"], np.ndarray)
        with self.assertRaises(FrozenInstanceError):
            result.dataset = "changed"  # type: ignore[misc]

    def test_group_arrays_are_float32_and_read_only(self) -> None:
        values = np.arange(8, dtype=np.float64).reshape(2, 4)
        group = PreparedTrainGroup("Demo", 4, (0, 1), values)
        self.assertEqual(group.values.dtype, np.dtype(np.float32))
        self.assertFalse(group.values.flags.writeable)
        with self.assertRaises(ValueError):
            group.values[0, 0] = 99.0
        calibration = ChannelCalibration.fixture("prototype", reference_count=2)
        calibrated = CalibratedTrainGroup(
            dataset="Demo",
            length=4,
            instance_indices=(0, 1),
            values=group.values,
            gradients=None,
            acf_values=None,
            prototype=calibration,
            gradient=ChannelCalibration.unavailable("gradient", "not_fitted"),
            acf=ChannelCalibration.unavailable("acf", "not_fitted"),
        )
        self.assertFalse(calibrated.values.flags.writeable)

    def test_backend_and_run_summary_are_validated_and_immutable(self) -> None:
        backend = BackendDecision.cpu_fallback("mlx_unavailable")
        self.assertEqual(backend.backend, "cpu")
        self.assertFalse(backend.adopted)
        self.assertEqual(backend.smoke_seed, 43501)
        self.assertIn("normalized_error_max", backend.to_dict())
        summary = Exp435RunSummary(
            run_identity="exp435-test",
            status="complete",
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


if __name__ == "__main__":
    unittest.main()
