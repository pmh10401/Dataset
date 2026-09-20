from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp438_frequency_complexity_evidence_antigravity.contracts_antigravity import (
    Exp438DatasetResult,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
    PreparedTrainGroup,
    RawValueSeries,
)
from experiments.exp438_frequency_complexity_evidence_antigravity.scoring_antigravity import (
    CandidateBudgetExceededAntigravity,
    fit_calibrated_group_antigravity,
    higher_quantile_antigravity,
    robust_transform_antigravity,
    score_channel_cpu_antigravity,
    score_test_row_cpu_antigravity,
    validate_global_candidates_antigravity,
)


def _group(matrix: np.ndarray) -> PreparedTrainGroup:
    rows = tuple(
        RawValueSeries("D0001", index, "TRAIN", row, f"{index + 1:064x}")
        for index, row in enumerate(matrix)
    )
    return PreparedTrainGroup("D0001", matrix.shape[1], rows)


class Exp438ScoringAntigravityTests(unittest.TestCase):
    def test_waveform_transform_uses_one_scalar_train_median_and_mad(self) -> None:
        matrix = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 100.0]], dtype=np.float32)
        standardized, center, scale = robust_transform_antigravity(
            matrix, channel="high_frequency"
        )
        expected_center = np.median(matrix)
        expected_scale = 1.4826 * np.median(np.abs(matrix - expected_center))
        self.assertEqual(center.shape, (1,))
        self.assertEqual(scale.shape, (1,))
        self.assertAlmostEqual(float(center[0]), float(expected_center), places=6)
        self.assertAlmostEqual(float(scale[0]), float(expected_scale), places=5)
        np.testing.assert_allclose(
            standardized,
            (matrix - expected_center) / expected_scale,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_complexity_transform_is_per_column_with_scale_floor(self) -> None:
        matrix = np.tile(np.arange(9, dtype=np.float32), (4, 1))
        standardized, center, scale = robust_transform_antigravity(
            matrix, channel="amplitude_complexity"
        )
        self.assertEqual(center.shape, (9,))
        np.testing.assert_array_equal(standardized, np.zeros_like(matrix))
        np.testing.assert_array_equal(scale, np.full(9, 1e-6, dtype=np.float32))

    def test_channel_score_is_median_of_three_smallest_rms_distances(self) -> None:
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray(
            [[1.0, 1.0], [3.0, 3.0], [2.0, 2.0], [10.0, 10.0]],
            dtype=np.float32,
        )
        self.assertEqual(score_channel_cpu_antigravity(query, references, 3), 2.0)

    def test_higher_quantile_and_strict_threshold_are_exact(self) -> None:
        scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.assertAlmostEqual(higher_quantile_antigravity(scores, 0.75), 0.4, places=6)

    def test_fit_uses_leave_one_out_models_and_is_train_order_invariant(self) -> None:
        rng = np.random.default_rng(43801)
        matrix = rng.normal(size=(5, 64)).astype(np.float32)
        calls: list[int] = []

        from experiments.exp438_frequency_complexity_evidence_antigravity import scoring_antigravity

        real_select = scoring_antigravity.select_train_depth_antigravity

        def recording_select(value, **kwargs):
            calls.append(np.asarray(value).shape[0])
            return real_select(value, **kwargs)

        with patch.object(scoring_antigravity, "select_train_depth_antigravity", side_effect=recording_select):
            first = fit_calibrated_group_antigravity(_group(matrix))
        self.assertIn(5, calls)
        self.assertGreaterEqual(calls.count(4), 5)
        reordered = fit_calibrated_group_antigravity(_group(matrix[[4, 2, 0, 3, 1]]))
        self.assertEqual(first.depth, reordered.depth)
        for channel in ("high_frequency", "low_frequency", "amplitude_complexity"):
            left = first.calibrations[channel]
            right = reordered.calibrations[channel]
            self.assertEqual(left.available, right.available)
            if left.available:
                self.assertAlmostEqual(left.threshold, right.threshold, places=5)
                np.testing.assert_allclose(np.sort(left.oof_scores), np.sort(right.oof_scores), atol=1e-5)

    def test_test_scoring_preserves_three_channels_without_fusion(self) -> None:
        rng = np.random.default_rng(9)
        matrix = rng.normal(size=(5, 64)).astype(np.float32)
        calibrated = fit_calibrated_group_antigravity(_group(matrix))
        row = RawValueSeries("D0001", 99, "TEST", rng.normal(size=64), "f" * 64)
        evidence = score_test_row_cpu_antigravity(
            row, calibrated, outside_all_frozen_lanes=True
        )
        self.assertEqual(
            tuple(channel.channel for channel in evidence.channels),
            ("high_frequency", "low_frequency", "amplitude_complexity"),
        )
        self.assertFalse(hasattr(evidence, "combined_abnormal"))
        self.assertFalse(hasattr(evidence, "vote_count"))
        for channel in evidence.channels:
            if channel.available:
                self.assertEqual(channel.abnormal, channel.score > channel.threshold)

    def test_wrong_length_query_is_rejected(self) -> None:
        rng = np.random.default_rng(10)
        calibrated = fit_calibrated_group_antigravity(_group(rng.normal(size=(5, 64))))
        row = RawValueSeries("D0001", 99, "TEST", np.arange(65), "f" * 64)
        with self.assertRaisesRegex(ValueError, "exact TRAIN length"):
            score_test_row_cpu_antigravity(row, calibrated, outside_all_frozen_lanes=True)

    def test_candidate_budgets_fail_without_top_n_cutting(self) -> None:
        channels = tuple(
            InstanceChannelEvidence(name, 2.0, 1.0, True, True, None)
            for name in ("high_frequency", "low_frequency", "amplitude_complexity")
        )
        rows = tuple(
            InstanceScoreEvidence("D0001", index, 16, channels, True)
            for index in range(2)
        )
        result = Exp438DatasetResult("D0001", rows, (), ())
        small = SimpleNamespace(channel_candidate_budget=1, total_candidate_budget=3)
        with patch(
            "experiments.exp438_frequency_complexity_evidence_antigravity.scoring_antigravity.FROZEN_CONFIG",
            small,
        ):
            with self.assertRaisesRegex(CandidateBudgetExceededAntigravity, "high_frequency"):
                validate_global_candidates_antigravity((result,))
        large = SimpleNamespace(channel_candidate_budget=2, total_candidate_budget=5)
        with patch(
            "experiments.exp438_frequency_complexity_evidence_antigravity.scoring_antigravity.FROZEN_CONFIG",
            large,
        ):
            with self.assertRaisesRegex(CandidateBudgetExceededAntigravity, "total"):
                validate_global_candidates_antigravity((result,))


if __name__ == "__main__":
    unittest.main()
