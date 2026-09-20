"""TDD tests for Exp435's TRAIN-only CPU scoring boundary."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTestDataset
from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
    UnavailableTrainGroup,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.scoring_antigravity import (
    CandidateBudgetExceeded,
    score_dataset_cpu,
    validate_global_candidates,
)


class _Lanes:
    """Small lane protocol fixture; FrozenLaneSnapshot has frozen production totals."""

    def __init__(self, *coordinates: tuple[str, int]) -> None:
        self.coordinates = set(coordinates)

    def contains(self, dataset: str, instance_index: int) -> bool:
        return (dataset, instance_index) in self.coordinates


def _calibration(channel: str, threshold: float = 1.0) -> ChannelCalibration:
    scores = (0.0, 0.01, 0.02, threshold)
    return ChannelCalibration(
        channel=channel,
        available=True,
        threshold=float(
            np.quantile(np.asarray(scores, dtype=np.float32), 0.99, method="higher")
        ),
        loo_scores=scores,
        center=0.0,
        scale=1.0,
        reference_count=4,
        reference_digest="0" * 64,
    )


def calibrated_group(
    *,
    dataset: str = "Demo",
    length: int = 16,
    prototype: bool = True,
    gradient: bool = True,
    acf: bool = True,
) -> CalibratedTrainGroup | UnavailableTrainGroup:
    if not prototype:
        return UnavailableTrainGroup(dataset, length, (0, 1, 2, 3), "zero_scale")
    values = np.zeros((4, length), dtype=np.float32)
    gradient_values = np.zeros_like(values) if gradient else None
    acf_columns = min(32, length // 4)
    acf_values = np.zeros((4, acf_columns), dtype=np.float32) if acf else None
    gradient_calibration = (
        _calibration("gradient") if gradient else ChannelCalibration.unavailable("gradient", "zero_scale")
    )
    acf_calibration = (
        _calibration("acf", threshold=0.1) if acf else ChannelCalibration.unavailable("acf", "short_series")
    )
    return CalibratedTrainGroup(
        dataset=dataset,
        length=length,
        instance_indices=(0, 1, 2, 3),
        values=values,
        gradients=gradient_values,
        acf_values=acf_values,
        prototype=_calibration("prototype"),
        gradient=gradient_calibration,
        acf=acf_calibration,
    )


class ScoringTests(unittest.TestCase):
    def _test_row(self, index: int = 0, length: int = 16) -> RawSeries:
        return RawSeries(index, tuple(2.0 if offset % 2 == 0 else -2.0 for offset in range(length)))

    def _score(self, *, candidate_index: int = 0, hard: dict[str, tuple[int, ...]] | None = None, **channels: bool) -> Exp435DatasetResult:
        group = calibrated_group(**channels)
        test = RawTestDataset("Demo", (self._test_row(candidate_index, group.length),))
        lanes = _Lanes(*((dataset, index) for dataset, indices in (hard or {}).items() for index in indices))
        return score_dataset_cpu((group,), test, lanes)

    def test_prototype_is_mandatory(self) -> None:
        result = self._score(prototype=False, gradient=True, acf=True)
        self.assertFalse(result.evidence[0].consensus_candidate)
        self.assertEqual(result.evidence[0].status, "abstained")

    def test_one_secondary_channel_is_enough(self) -> None:
        # The score fixture is deliberately far from the all-zero TRAIN rows.
        for channels in ({"prototype": True, "gradient": True, "acf": False}, {"prototype": True, "gradient": False, "acf": True}):
            with self.subTest(channels=channels):
                result = self._score(**channels)
                self.assertTrue(result.evidence[0].consensus_candidate)

    def test_frozen_lane_coordinate_is_never_published(self) -> None:
        group = calibrated_group()
        test = RawTestDataset("Demo", tuple(self._test_row(index, group.length) for index in range(8)))
        result = score_dataset_cpu((group,), test, _Lanes(("Demo", 7)))
        self.assertNotIn(7, tuple(item.instance_index for item in result.candidates))
        self.assertEqual(result.evidence[7].exclusion_reason, "frozen_lane")

    def test_exact_length_groups_do_not_concatenate(self) -> None:
        short = calibrated_group(length=16)
        long = calibrated_group(length=32)
        test = RawTestDataset("Demo", (self._test_row(0, 16), self._test_row(1, 32)))
        result = score_dataset_cpu((short, long), test, _Lanes())
        self.assertEqual(tuple(item.length for item in result.evidence), (16, 32))
        self.assertEqual(tuple(item.instance_index for item in result.evidence), (0, 1))

    def test_missing_exact_length_is_explicit_abstention(self) -> None:
        group = calibrated_group(length=16)
        result = score_dataset_cpu((group,), RawTestDataset("Demo", (self._test_row(0, 32),)), _Lanes())
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.evidence[0].status, "abstained")
        self.assertEqual(result.evidence[0].exclusion_reason, "missing_length")

    def test_short_unavailable_group_is_explicit_per_instance_abstention(self) -> None:
        group = UnavailableTrainGroup(
            "Demo", 1, (0, 1, 2, 3), "insufficient_series_length"
        )
        result = score_dataset_cpu(
            (group,), RawTestDataset("Demo", (RawSeries(0, (1.0,)),)), _Lanes()
        )
        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.evidence[0].length, 1)
        self.assertEqual(result.evidence[0].exclusion_reason, "insufficient_series_length")

    def test_strict_greater_than_threshold_is_required(self) -> None:
        group = calibrated_group()
        assert isinstance(group, CalibratedTrainGroup)
        exact = replace(group, prototype=_calibration("prototype", threshold=2.0))
        result = score_dataset_cpu(
            (exact,), RawTestDataset("Demo", (self._test_row(),)), _Lanes()
        )
        evidence = result.evidence[0]
        self.assertEqual(evidence.prototype_score, evidence.prototype_threshold)
        self.assertFalse(evidence.prototype_abnormal)
        self.assertFalse(evidence.consensus_candidate)

    def test_unavailable_secondary_channels_are_explicit_abstention(self) -> None:
        group = calibrated_group(gradient=False, acf=False)
        result = score_dataset_cpu((group,), RawTestDataset("Demo", (self._test_row(),)), _Lanes())
        self.assertEqual(result.evidence[0].status, "abstained")
        self.assertEqual(result.evidence[0].exclusion_reason, "secondary_channels_unavailable")

    def test_failed_test_gradient_is_unavailable_not_raw_waveform_scored(self) -> None:
        group = calibrated_group(gradient=True, acf=False)
        test = RawTestDataset("Demo", (self._test_row(),))
        with patch(
            "experiments.exp435_train_only_consensus_recovery_antigravity.scoring_antigravity.np.diff",
            side_effect=ValueError("forced gradient failure"),
        ):
            result = score_dataset_cpu((group,), test, _Lanes())
        evidence = result.evidence[0]
        self.assertEqual(evidence.status, "abstained")
        self.assertIsNone(evidence.gradient_score)
        self.assertIsNone(evidence.gradient_threshold)
        self.assertEqual(evidence.exclusion_reason, "secondary_channels_unavailable")

    def test_concatenated_rows_are_not_accepted_as_two_instances(self) -> None:
        group = calibrated_group(length=16)
        first = self._test_row(0, 16)
        second = self._test_row(1, 16)
        separate = score_dataset_cpu(
            (group,), RawTestDataset("Demo", (first, second)), _Lanes()
        )
        self.assertEqual(tuple(item.length for item in separate.evidence), (16, 16))
        joined = RawSeries(0, first.values + second.values)
        concatenated = score_dataset_cpu(
            (group,), RawTestDataset("Demo", (joined,)), _Lanes()
        )
        self.assertEqual(concatenated.status, "abstained")
        self.assertEqual(concatenated.evidence[0].length, 32)
        self.assertEqual(concatenated.evidence[0].exclusion_reason, "missing_length")

    def test_candidate_budget_accepts_100_and_rejects_101_without_truncating(self) -> None:
        evidence = tuple(
            InstanceScoreEvidence.fixture(dataset=f"D{i:03d}", instance_index=0, length=16, prototype_abnormal=True, gradient_abnormal=True)
            for i in range(101)
        )
        # Build candidate-bearing records independently so each result remains valid.
        candidate_results = tuple(
            Exp435DatasetResult.fixture(
                dataset=item.dataset,
                test_size=1,
                candidates=(
                    Exp435Candidate(item.dataset, item.instance_index, item.length, item.evidence_digest),
                ),
                evidence=(item,),
            )
            for item in evidence
        )
        self.assertEqual(len(validate_global_candidates(candidate_results[:100])), 100)
        with self.assertRaises(CandidateBudgetExceeded) as context:
            validate_global_candidates(candidate_results)
        self.assertEqual(context.exception.observed, 101)
        self.assertEqual(context.exception.budget, 100)

    def test_global_candidates_are_canonical_sorted_and_duplicate_checked(self) -> None:
        evidence_a = InstanceScoreEvidence.fixture(
            dataset="A", instance_index=1, length=16, prototype_abnormal=True, gradient_abnormal=True
        )
        evidence_b = InstanceScoreEvidence.fixture(
            dataset="B", instance_index=0, length=16, prototype_abnormal=True, gradient_abnormal=True
        )
        result_a = Exp435DatasetResult.fixture(
            dataset="A",
            test_size=2,
            candidates=(Exp435Candidate("A", 1, 16, evidence_a.evidence_digest),),
            evidence=(evidence_a,),
        )
        result_b = Exp435DatasetResult.fixture(
            dataset="B",
            test_size=1,
            candidates=(Exp435Candidate("B", 0, 16, evidence_b.evidence_digest),),
            evidence=(evidence_b,),
        )
        ordered = validate_global_candidates((result_b, result_a))
        self.assertEqual(tuple(item.coordinate for item in ordered), (("A", 1), ("B", 0)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_global_candidates((result_a, result_a))


if __name__ == "__main__":
    unittest.main()
