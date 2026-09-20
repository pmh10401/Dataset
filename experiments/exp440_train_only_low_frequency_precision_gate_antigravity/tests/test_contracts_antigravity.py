from __future__ import annotations

from dataclasses import replace
import math
import unittest

from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
    DepthCalibrationAntigravity,
    DepthScoreAntigravity,
    Exp440DatasetResultAntigravity,
    GroupCalibrationAntigravity,
    ProvisionalCandidateAntigravity,
    RetainedCandidateAntigravity,
    validate_retained_candidates_antigravity,
)


def _depth(depth: int = 1, *, threshold: float = 1.0) -> DepthCalibrationAntigravity:
    return DepthCalibrationAntigravity(
        depth=depth,
        oof_scores=(0.2, 0.4, 0.6, threshold),
        threshold=threshold,
        delete_one_thresholds=(1.0, 1.5, 2.0, 1.25),
        robust_scale=1.0,
        stability_ratio=2.0,
        stable=True,
    )


class Exp440ContractsAntigravityTests(unittest.TestCase):
    def test_depth_calibration_accepts_zero_threshold_and_rejects_invalid_evidence(self) -> None:
        zero = _depth(threshold=0.0)
        self.assertEqual(zero.threshold_denominator, 1e-12)
        for invalid in (
            lambda: replace(_depth(), oof_scores=(0.2, 0.4, math.nan, 1.0)),
            lambda: replace(_depth(), threshold=-1.0),
            lambda: replace(_depth(), robust_scale=0.0),
            lambda: replace(_depth(), delete_one_thresholds=(1.0, 2.01, 1.5, 1.25), stability_ratio=2.01),
        ):
            with self.assertRaises(ValueError):
                invalid()

    def test_group_requires_sorted_unique_depths_and_two_stable_views(self) -> None:
        group = GroupCalibrationAntigravity(32, 2, (_depth(1), _depth(2)), True)
        self.assertEqual(GroupCalibrationAntigravity.from_payload(group.to_payload()), group)
        with self.assertRaises(ValueError):
            GroupCalibrationAntigravity(32, 2, (_depth(2), _depth(1)), True)
        with self.assertRaises(ValueError):
            GroupCalibrationAntigravity(32, 1, (_depth(1),), True)

    def test_depth_score_recomputes_exceedance_ratio_and_rank_contract(self) -> None:
        score = DepthScoreAntigravity(1, 1.5, 1.0, True, 1.5, 0.2)
        self.assertEqual(DepthScoreAntigravity.from_payload(score.to_payload()), score)
        with self.assertRaises(ValueError):
            replace(score, exceeds=False)
        with self.assertRaises(ValueError):
            replace(score, excess_ratio=1.4)

    def test_candidate_ranking_key_and_roundtrip_are_canonical(self) -> None:
        candidate = ProvisionalCandidateAntigravity(
            dataset="A", instance_index=7, raw_length=32,
            depths=(1, 2), agreement_count=2,
            second_excess_ratio=1.5, second_empirical_rank=0.1,
        )
        self.assertEqual(candidate.ranking_key, (-2, -1.5, 0.1, "A", 7))
        self.assertEqual(ProvisionalCandidateAntigravity.from_payload(candidate.to_payload()), candidate)
        with self.assertRaises(ValueError):
            replace(candidate, depths=(2, 1))
        with self.assertRaises(ValueError):
            replace(candidate, agreement_count=1)

    def test_retained_candidates_reject_hard_duplicates_and_budget_overflow(self) -> None:
        items = tuple(
            RetainedCandidateAntigravity.from_provisional(
                ProvisionalCandidateAntigravity(f"D{i:03d}", i, 16, (1, 2), 2, 1.5, 0.1),
                i + 1,
            )
            for i in range(101)
        )
        with self.assertRaises(ValueError):
            validate_retained_candidates_antigravity(items, frozenset(), budget=100)
        with self.assertRaises(ValueError):
            validate_retained_candidates_antigravity(items[:1], frozenset({("D000", 0)}), budget=100)

    def test_dataset_result_rejects_unknown_payload_keys(self) -> None:
        result = Exp440DatasetResultAntigravity("A", "scored", (), (), 0, 1)
        raw = result.to_payload(); raw["label"] = 1
        with self.assertRaises(ValueError):
            Exp440DatasetResultAntigravity.from_payload(raw)


if __name__ == "__main__":
    unittest.main()
