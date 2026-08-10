"""TDD coverage for Exp444 original control and stability classification."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
    ProvisionalCandidateAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity import stability_antigravity as stability_module
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilitySelectionAntigravity,
    StabilityViewAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity import (
    classify_view_antigravity,
    replay_original_candidate_antigravity,
    score_stability_candidate_antigravity,
)


def _depth_antigravity(depth: int, exceeds: bool) -> DepthViewScoreAntigravity:
    threshold = 1.0
    score = 2.0 if exceeds else 1.0
    return DepthViewScoreAntigravity(depth, score, threshold, exceeds, score / threshold, 0.5)


def _fixture_input_antigravity() -> CandidateInputAntigravity:
    x = np.arange(32, dtype=np.float32)
    rng = np.random.default_rng(444)
    train = np.stack(
        tuple(
            (np.sin(x / (2.0 + i * 0.1)) + i * 0.05 + rng.normal(0.0, 0.1, x.size)).astype("<f4")
            for i in range(8)
        ),
        axis=0,
    )
    test = (train.mean(axis=0) + np.float32(8.0)).astype("<f4")
    candidate = SourceCandidateAntigravity("Fixture_normal_1", 8, 32, (1, 2), 1)
    return CandidateInputAntigravity(candidate, tuple(range(8)), train, test)


def _sixteen_views_antigravity(survival_count: int) -> tuple[StabilityViewAntigravity, ...]:
    views: list[StabilityViewAntigravity] = []
    for index in range(16):
        survived = index < survival_count
        scores = (_depth_antigravity(1, True), _depth_antigravity(2, survived))
        views.append(
            StabilityViewAntigravity(
                OmissionMaskAntigravity(index, (index,), f"{index + 1:064x}"),
                (100, 101, 102, 103),
                scores,
                (1, 2) if survived else (1,),
                survived,
            )
        )
    return tuple(views)


def _rank_replay_antigravity(rank: int, *, ratio: float) -> object:
    source = SourceCandidateAntigravity(f"Dataset{rank:02d}", rank, 32, (1, 2), rank)
    scores = (_depth_antigravity(1, True), _depth_antigravity(2, True))
    candidate = ProvisionalCandidateAntigravity(
        source.dataset,
        source.instance_index,
        source.raw_length,
        source.original_depths,
        2,
        ratio,
        0.5,
    )
    return stability_module.OriginalReplayAntigravity(source, scores, source.original_depths, candidate, True)


class Exp444StabilityAntigravityTests(unittest.TestCase):
    def test_fourteen_of_sixteen_views_is_stable_review_antigravity(self) -> None:
        source = SourceCandidateAntigravity("Stable", 1, 32, (1, 2), 1)
        result = stability_module.aggregate_stability_views_antigravity(
            source, _sixteen_views_antigravity(14)
        )
        self.assertIsInstance(result, CandidateStabilityResultAntigravity)
        self.assertEqual((result.survival_count, result.required_survival_count, result.lane), (14, 14, "stable_review"))

    def test_thirteen_of_sixteen_views_is_unstable_review_antigravity(self) -> None:
        source = SourceCandidateAntigravity("Unstable", 1, 32, (1, 2), 1)
        result = stability_module.aggregate_stability_views_antigravity(
            source, _sixteen_views_antigravity(13)
        )
        self.assertIsInstance(result, CandidateStabilityResultAntigravity)
        self.assertEqual((result.survival_count, result.required_survival_count, result.lane), (13, 14, "unstable_review"))

    def test_aggregate_replay_recomputes_all_thirty_ranks_antigravity(self) -> None:
        # Input order is reversed; only the unchanged Exp441/Exp442 global key
        # can recover the frozen 1..30 source ranks from replayed score evidence.
        replays = tuple(
            _rank_replay_antigravity(rank, ratio=float(31 - rank))
            for rank in reversed(range(1, 31))
        )
        ranked = stability_module.verify_original_replay_ranks_antigravity(replays)
        self.assertEqual(tuple(item.coordinate for item in ranked), tuple((f"Dataset{rank:02d}", rank) for rank in range(1, 31)))
        self.assertEqual(tuple(item.replayed_rank for item in ranked), tuple(range(1, 31)))
        self.assertTrue(all(item.replayed_rank == item.original_rank for item in ranked))

    def test_aggregate_replay_rejects_frozen_rank_mismatch_antigravity(self) -> None:
        replays = [
            _rank_replay_antigravity(rank, ratio=float(31 - rank))
            for rank in range(1, 31)
        ]
        first, second = replays[:2]
        replays[0] = replace(first, source=replace(first.source, original_rank=2))
        replays[1] = replace(second, source=replace(second.source, original_rank=1))
        with self.assertRaisesRegex(ValueError, "aggregate replay rank"):
            stability_module.verify_original_replay_ranks_antigravity(tuple(replays))

    def test_candidate_absent_discards_exceeding_agreement_evidence_antigravity(self) -> None:
        source = SourceCandidateAntigravity("Absent", 1, 32, (1, 2), 1)
        view = stability_module._view_from_scores_antigravity(
            OmissionMaskAntigravity(0, (0,), "1" * 64),
            (1, 2, 3, 4),
            (_depth_antigravity(1, True), _depth_antigravity(2, True)),
            source.original_depths,
            False,
        )
        self.assertEqual(view.agreeing_depths, ())
        self.assertFalse(view.survived)

    def test_empty_score_records_without_candidate_are_non_surviving_and_round_trip_antigravity(self) -> None:
        source = SourceCandidateAntigravity("Empty", 1, 32, (1, 2), 1)
        self.assertEqual(stability_module._copy_scores_antigravity(()), ())
        view = stability_module._view_from_scores_antigravity(
            OmissionMaskAntigravity(13, (0,), "2" * 64),
            (1, 2, 3, 4),
            (),
            source.original_depths,
            False,
        )

        self.assertEqual(view.depth_scores, ())
        self.assertEqual(view.agreeing_depths, ())
        self.assertFalse(view.survived)
        restored = StabilityViewAntigravity.from_payload(view.to_payload())
        self.assertEqual(restored, view)

    def test_empty_score_records_reject_survival_or_agreement_antigravity(self) -> None:
        mask = OmissionMaskAntigravity(13, (0,), "3" * 64)
        with self.assertRaisesRegex(ValueError, "empty view"):
            StabilityViewAntigravity(mask, (1, 2, 3, 4), (), (), True)
        with self.assertRaisesRegex(ValueError, "empty view"):
            StabilityViewAntigravity(mask, (1, 2, 3, 4), (), (1,), False)

    def test_original_replay_requires_exact_exp442_candidate_antigravity(self) -> None:
        fixture = _fixture_input_antigravity()
        replay = replay_original_candidate_antigravity(fixture)
        self.assertEqual(replay.coordinate, fixture.candidate.coordinate)
        self.assertEqual(replay.original_depths, fixture.candidate.original_depths)
        self.assertTrue(replay.matches_source)

    def test_view_requires_two_original_depths_antigravity(self) -> None:
        self.assertFalse(classify_view_antigravity((2, 3), (_depth_antigravity(2, True), _depth_antigravity(4, True))))
        self.assertTrue(classify_view_antigravity((2, 3), (_depth_antigravity(2, True), _depth_antigravity(3, True))))
        self.assertFalse(classify_view_antigravity((2,), (_depth_antigravity(2, True),)))

    def test_stability_result_conserves_views_and_lane_antigravity(self) -> None:
        fixture = _fixture_input_antigravity()
        result = score_stability_candidate_antigravity(fixture)
        self.assertEqual(result.coordinate, fixture.candidate.coordinate)
        self.assertEqual(result.survival_count, sum(item.survived for item in result.views))
        self.assertGreaterEqual(result.required_survival_count, 1)
        self.assertEqual(
            result.lane,
            "stable_review" if result.survival_count >= result.required_survival_count else "unstable_review",
        )

    def test_replay_rejects_altered_return_coordinate_antigravity(self) -> None:
        fixture = _fixture_input_antigravity()
        # The public Exp440 scorer must not be allowed to silently move a
        # candidate to another coordinate.  The implementation test seam
        # replaces the scorer for this one call and expects a fail-closed error.
        import experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity as module

        original = module.score_test_row_antigravity
        try:
            def altered(*args: object, **kwargs: object):
                scores, candidate = original(*args, **kwargs)
                if candidate is None:
                    return scores, None
                return scores, replace(candidate, dataset="Moved_normal_1")

            module.score_test_row_antigravity = altered
            with self.assertRaises(ValueError):
                replay_original_candidate_antigravity(fixture)
        finally:
            module.score_test_row_antigravity = original

    def test_unavailable_original_refit_fails_closed_antigravity(self) -> None:
        fixture = _fixture_input_antigravity()
        import experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity as module

        original = module.fit_group_calibration_antigravity
        try:
            def unavailable(*args: object, **kwargs: object):
                raise ValueError("fixture refit unavailable")

            module.fit_group_calibration_antigravity = unavailable
            with self.assertRaisesRegex(ValueError, "calibration is unavailable"):
                replay_original_candidate_antigravity(fixture)
        finally:
            module.fit_group_calibration_antigravity = original

    def test_nonfinite_original_score_fails_closed_antigravity(self) -> None:
        fixture = _fixture_input_antigravity()
        import experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity as module

        original_fit = module.fit_group_calibration_antigravity
        original_score = module.score_test_row_antigravity
        try:
            module.fit_group_calibration_antigravity = lambda *args, **kwargs: object()

            class NonfiniteScore:
                depth = 1
                score = float("nan")
                threshold = 1.0
                exceeds = False
                excess_ratio = 0.0
                empirical_rank = 0.5

            module.score_test_row_antigravity = lambda *args, **kwargs: ((NonfiniteScore(),), None)
            with self.assertRaisesRegex(ValueError, "invalid or nonfinite"):
                replay_original_candidate_antigravity(fixture)
        finally:
            module.fit_group_calibration_antigravity = original_fit
            module.score_test_row_antigravity = original_score

    def test_stable_and_unstable_lanes_conserve_exact_coordinates_antigravity(self) -> None:
        sources = tuple(
            SourceCandidateAntigravity(f"Dataset{index:02d}", index, 32, (1, 2), index + 1)
            for index in range(30)
        )
        stable_score = (_depth_antigravity(1, True), _depth_antigravity(2, True))
        unstable_score = (_depth_antigravity(1, True), _depth_antigravity(2, False))
        stable: list[CandidateStabilityResultAntigravity] = []
        unstable: list[CandidateStabilityResultAntigravity] = []
        for index, source in enumerate(sources):
            score = stable_score if index < 15 else unstable_score
            agreeing = (1, 2) if index < 15 else (1,)
            survived = index < 15
            view = StabilityViewAntigravity(
                OmissionMaskAntigravity(0, (0,), "0" * 64),
                (1, 2, 3, 4),
                score,
                agreeing,
                survived,
            )
            result = CandidateStabilityResultAntigravity(source, (view,), int(survived), 1, "stable_review" if survived else "unstable_review")
            (stable if survived else unstable).append(result)
        selection = StabilitySelectionAntigravity(sources, tuple(stable), tuple(unstable))
        self.assertEqual(selection.source_count, 30)
        self.assertEqual(selection.stable_count + selection.unstable_count, selection.source_count)
        self.assertEqual(
            {item.coordinate for item in selection.stable} | {item.coordinate for item in selection.unstable},
            {item.coordinate for item in selection.source},
        )


if __name__ == "__main__":
    unittest.main()
