"""Behavioral tests for Exp436's missing-safe strict TEST scoring boundary."""

from __future__ import annotations

from dataclasses import replace
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    Exp436Candidate,
    Exp436DatasetResult,
    InstanceScoreEvidence,
    MissingSafeRawDataset,
    RawValueSeries,
    UnavailableTrainGroup,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.scoring_antigravity import (
    CandidateBudgetExceeded,
    score_dataset_backend,
    score_dataset_cpu,
    score_test_instance_cpu,
    validate_global_candidates,
)


class _Lanes:
    def __init__(self, *coordinates: tuple[str, int], raises: bool = False) -> None:
        self.coordinates = set(coordinates)
        self.raises = raises

    def contains(self, dataset: str, instance_index: int) -> bool:
        if self.raises:
            raise RuntimeError("lane service unavailable")
        return (dataset, instance_index) in self.coordinates


def _calibration(channel: str, threshold: float = 1.0) -> ChannelCalibration:
    scores = (0.0, 0.1, 0.2, threshold)
    return ChannelCalibration(
        channel=channel,
        threshold=float(np.quantile(np.asarray(scores, dtype=np.float32), 0.99, method="higher")),
        loo_scores=scores,
        center=0.0,
        scale=1.0,
        reference_count=4,
        reference_digest="0" * 64,
    )


def _group(
    *,
    dataset: str = "Demo",
    length: int = 16,
    prototype: bool = True,
    gradient: bool = True,
    acf: bool = True,
) -> CalibratedTrainGroup:
    values = np.zeros((4, length), dtype=np.float32) if prototype else None
    gradients = np.zeros((4, length), dtype=np.float32) if gradient else None
    acf_values = np.zeros((4, min(FROZEN_CONFIG.max_acf_lag, length // 4)), dtype=np.float32) if acf else None
    return CalibratedTrainGroup(
        dataset=dataset,
        length=length,
        instance_indices=(0, 1, 2, 3),
        values=values,
        gradients=gradients,
        acf_values=acf_values,
        prototype=_calibration("prototype") if prototype else ChannelCalibration.unavailable("prototype", "zero_scale"),
        gradient=_calibration("gradient") if gradient else ChannelCalibration.unavailable("gradient", "zero_scale"),
        acf=_calibration("acf", threshold=0.1) if acf else ChannelCalibration.unavailable("acf", "zero_acf_energy"),
    )


def _row(index: int, *, length: int = 16, value: float = 2.0) -> RawValueSeries:
    values = tuple(value if offset % 2 == 0 else -value for offset in range(length))
    return RawValueSeries(index, values, f"{index:064x}")


class ScoringTests(unittest.TestCase):
    def test_nonfinite_row_is_scalar_free_and_finite_sibling_continues(self) -> None:
        test = MissingSafeRawDataset("Demo", "TEST", (
            RawValueSeries(2, (1.0, float("nan")), "2" * 64), _row(9),
        ))
        result = score_dataset_cpu((_group(),), test, _Lanes())
        abstained, scored = result.evidence
        self.assertEqual(abstained.status, "abstained")
        self.assertEqual(abstained.exclusion_reason, "nonfinite_test")
        self.assertIsNone(abstained.prototype_score)
        self.assertIsNone(abstained.gradient_score)
        self.assertIsNone(abstained.acf_score)
        self.assertEqual(scored.status, "scored")

    def test_raw_length_one_precedes_group_lookup(self) -> None:
        evidence = score_test_instance_cpu(
            RawValueSeries(7, (1.0,), "7" * 64), (_group(),), _Lanes()
        )
        self.assertEqual(evidence.exclusion_reason, "insufficient_series_length")
        self.assertEqual(evidence.status, "abstained")
        self.assertIsNone(evidence.prototype_threshold)

    def test_missing_and_unavailable_train_groups_are_scalar_free(self) -> None:
        missing = score_test_instance_cpu(_row(1, length=32), (_group(),), _Lanes())
        unavailable = score_test_instance_cpu(
            _row(2), (UnavailableTrainGroup("Demo", 16, (0, 1, 2)),), _Lanes()
        )
        for evidence, reason in ((missing, "missing_length"), (unavailable, "insufficient_finite_train_rows")):
            with self.subTest(reason=reason):
                self.assertEqual(evidence.exclusion_reason, reason)
                self.assertIsNone(evidence.prototype_score)
                self.assertFalse(evidence.consensus_candidate)

    def test_each_unavailable_channel_preserves_calculable_siblings(self) -> None:
        for channel, group in (("prototype", _group(prototype=False)), ("gradient", _group(gradient=False)), ("acf", _group(acf=False))):
            with self.subTest(channel=channel):
                evidence = score_test_instance_cpu(_row(5), (group,), _Lanes())
                self.assertEqual(evidence.status, "abstained")
                self.assertEqual(evidence.exclusion_reason, "three_channel_consensus_unavailable")
                self.assertEqual(getattr(evidence, f"{channel}_unavailability_reason"), "zero_scale" if channel != "acf" else "zero_acf_energy")
                self.assertIsNone(getattr(evidence, f"{channel}_score"))
                for sibling in {"prototype", "gradient", "acf"} - {channel}:
                    self.assertIsNotNone(getattr(evidence, f"{sibling}_score"))

    def test_unavailable_train_gradient_does_not_execute_gradient_transform(self) -> None:
        with patch(
            "experiments.exp436_missing_safe_strict_consensus_antigravity.scoring_antigravity.np.diff",
            side_effect=AssertionError("gradient transform must not run"),
        ):
            evidence = score_test_instance_cpu(_row(5), (_group(gradient=False),), _Lanes())
        self.assertEqual(evidence.gradient_unavailability_reason, "zero_scale")

    def test_all_unavailable_channels_retain_each_exact_reason(self) -> None:
        evidence = score_test_instance_cpu(_row(5), (_group(prototype=False, gradient=False, acf=False),), _Lanes())
        self.assertEqual(evidence.exclusion_reason, "three_channel_consensus_unavailable")
        self.assertEqual(
            (evidence.prototype_unavailability_reason, evidence.gradient_unavailability_reason, evidence.acf_unavailability_reason),
            ("zero_scale", "zero_scale", "zero_acf_energy"),
        )

    def test_constant_finite_row_marks_acf_zero_energy_and_never_scores_raw_waveform(self) -> None:
        calls: list[tuple[int, int]] = []
        def dense(query: object, references: object, k: int) -> float:
            query_array = np.asarray(query, dtype=np.float32)
            reference_array = np.asarray(references, dtype=np.float32)
            calls.append((query_array.size, reference_array.shape[1]))
            return 2.0
        evidence = score_test_instance_cpu(
            RawValueSeries(3, (4.0,) * 16, "3" * 64), (_group(),), _Lanes(), channel_scorer=dense
        )
        self.assertEqual(evidence.acf_unavailability_reason, "zero_acf_energy")
        self.assertEqual(evidence.exclusion_reason, "three_channel_consensus_unavailable")
        self.assertEqual(calls, [(16, 16), (16, 16)])

    def test_strict_three_channel_and_equality_lane_and_lane_error(self) -> None:
        all_three = score_test_instance_cpu(_row(2), (_group(),), _Lanes())
        self.assertTrue(all_three.consensus_candidate)
        self.assertEqual(score_test_instance_cpu(_row(2), (_group(),), _Lanes(("Demo", 2))).exclusion_reason, "frozen_lane")
        self.assertEqual(score_test_instance_cpu(_row(2), (_group(),), _Lanes(raises=True)).exclusion_reason, "lane_check_error")
        exact = replace(_group(), prototype=_calibration("prototype", threshold=2.0))
        equal = score_test_instance_cpu(_row(2), (exact,), _Lanes())
        self.assertFalse(equal.prototype_abnormal)
        self.assertFalse(equal.consensus_candidate)

    def test_postfinite_and_dense_numerical_errors_propagate(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense channel scorer failed"):
            score_test_instance_cpu(_row(1, value=1e20), (_group(),), _Lanes())
        with self.assertRaisesRegex(ValueError, "dense channel scorer failed"):
            score_test_instance_cpu(_row(1), (_group(),), _Lanes(), channel_scorer=lambda q, r, k: (_ for _ in ()).throw(ValueError("forced dense")))

    def test_runtime_backend_error_is_normalized_to_value_error_with_cause(self) -> None:
        def exploding_backend(query: object, references: object, k: int) -> float:
            raise RuntimeError("backend unavailable")
        with self.assertRaisesRegex(ValueError, "dense channel scorer failed") as context:
            score_dataset_backend((_group(),), MissingSafeRawDataset("Demo", "TEST", (_row(1),)), _Lanes(), exploding_backend)
        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_dataset_identity_backend_and_global_budget_contracts(self) -> None:
        test = MissingSafeRawDataset("Demo", "TEST", (_row(2), _row(9)))
        cpu = score_dataset_cpu((_group(),), test, _Lanes())
        backend = score_dataset_backend((_group(),), test, _Lanes(), lambda q, r, k: 2.0)
        self.assertEqual(tuple(item.instance_index for item in cpu.evidence), (2, 9))
        self.assertEqual(tuple(item.coordinate for item in backend.candidates), (("Demo", 2), ("Demo", 9)))
        def candidate_result(index: int) -> Exp436DatasetResult:
            evidence = InstanceScoreEvidence.fixture(
                dataset=f"D{index:03d}", instance_index=0, length=16,
                prototype_abnormal=True, gradient_abnormal=True, acf_abnormal=True,
            )
            candidate = Exp436Candidate(evidence.dataset, evidence.instance_index, evidence.length, evidence.evidence_digest)
            return Exp436DatasetResult(evidence.dataset, "scored", (0,), (candidate,), (evidence,))
        results = tuple(candidate_result(index) for index in range(101))
        self.assertEqual(len(validate_global_candidates(results[:100])), 100)
        with self.assertRaisesRegex(CandidateBudgetExceeded, "observed 101, budget 100"):
            validate_global_candidates(results)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_global_candidates((results[0], results[0]))

    def test_scoring_import_firewall_has_no_label_or_evaluation_route(self) -> None:
        source = Path(__file__).parents[1] / "scoring_antigravity.py"
        imports = [node.module or "" for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.ImportFrom)]
        forbidden = ("exp435", "evaluation", "metrics", "database", "label")
        self.assertFalse(any(any(term in module.lower() for term in forbidden) for module in imports))


if __name__ == "__main__":
    unittest.main()
