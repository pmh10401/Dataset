"""TDD coverage for Exp444 original control and stability classification."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    DepthViewScoreAntigravity,
    SourceCandidateAntigravity,
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


class Exp444StabilityAntigravityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
