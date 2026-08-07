"""TDD tests for Exp435's deterministic TRAIN-only feature pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    RawSeries,
    RawTrainDataset,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    PreparedTrainGroup,
    UnavailableTrainGroup,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.features_antigravity import (
    acf_vector,
    fit_calibrated_group,
    prepare_train_groups,
    score_channel_cpu,
)


def prepared_group(rows: list[list[float]], *, dataset: str = "Demo") -> PreparedTrainGroup:
    matrix = np.asarray(rows, dtype=np.float32)
    return PreparedTrainGroup(
        dataset=dataset,
        length=matrix.shape[1],
        instance_indices=tuple(range(matrix.shape[0])),
        values=matrix,
    )


def channel_calibration(scores: tuple[float, ...], *, channel: str = "prototype") -> ChannelCalibration:
    threshold = float(
        np.quantile(
            np.asarray(scores, dtype=np.float32),
            0.99,
            method="higher",
        )
    )
    return ChannelCalibration(
        channel=channel,
        threshold=threshold,
        loo_scores=scores,
        center=0.0,
        scale=1.0,
        reference_count=len(scores),
        reference_digest="0" * 64,
    )


class FeatureTests(unittest.TestCase):
    def test_repeated_normal_rows_remain_as_neighbors(self) -> None:
        group = prepared_group(
            [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [2, 3, 2, 3]]
        )
        fitted = fit_calibrated_group(group)
        self.assertEqual(fitted.prototype.reference_count, 4)
        self.assertEqual(fitted.prototype.loo_scores[0], 0.0)

    def test_higher_quantile_and_strict_comparison(self) -> None:
        calibration = channel_calibration((0.0, 0.1, 0.2, 0.3))
        expected_threshold = float(
            np.quantile(
                np.asarray((0.0, 0.1, 0.2, 0.3), dtype=np.float32),
                0.99,
                method="higher",
            )
        )
        self.assertEqual(calibration.threshold, expected_threshold)
        self.assertFalse(calibration.is_abnormal(expected_threshold))
        self.assertTrue(
            calibration.is_abnormal(np.nextafter(np.float32(expected_threshold), np.float32(1.0)))
        )

    def test_prepare_groups_preserves_exact_lengths_and_sorted_indices(self) -> None:
        train = RawTrainDataset(
            dataset="Demo",
            series=(
                RawSeries(0, (0.0, 1.0, 0.0, 1.0)),
                RawSeries(1, (1.0, 0.0, 1.0, 0.0)),
                RawSeries(2, (0.5, 1.5, 0.5, 1.5)),
                RawSeries(3, (1.5, 0.5, 1.5, 0.5)),
                RawSeries(4, (0.0, 1.0)),
                RawSeries(5, (1.0, 0.0)),
                RawSeries(6, (0.5, 1.5)),
                RawSeries(7, (1.5, 0.5)),
            ),
        )
        groups = prepare_train_groups(train)
        self.assertEqual(tuple(group.length for group in groups), (2, 4))
        self.assertEqual(groups[0].instance_indices, (4, 5, 6, 7))
        self.assertEqual(groups[1].instance_indices, (0, 1, 2, 3))
        self.assertEqual(groups[1].values.dtype, np.dtype(np.float32))
        self.assertFalse(groups[1].values.flags.writeable)

    def test_prepare_records_nonfinite_train_values(self) -> None:
        train = RawTrainDataset(
            dataset="Demo",
            series=(
                RawSeries(0, (0.0, 1.0)),
                RawSeries(1, (np.nan, 1.0)),
                RawSeries(2, (0.0, 1.0)),
                RawSeries(3, (0.0, 1.0)),
            ),
        )
        groups = prepare_train_groups(train)
        self.assertEqual(len(groups), 1)
        self.assertIsInstance(groups[0], UnavailableTrainGroup)
        self.assertEqual(groups[0].reason, "nonfinite_train")  # type: ignore[union-attr]

    def test_prepare_returns_unavailable_outcomes_without_suppressing_valid_siblings(self) -> None:
        series = [
            RawSeries(index, (float(index), float(index + 1)))
            for index in range(4)
        ]
        series.extend(
            RawSeries(index, (float(index),))
            for index in range(4, 8)
        )
        series.extend(
            RawSeries(index, (np.nan, float(index), float(index + 1)))
            for index in range(8, 12)
        )
        series.extend(
            RawSeries(index, (float(index),) * 4)
            for index in range(12, 15)
        )
        groups = prepare_train_groups(RawTrainDataset("Demo", tuple(series)))
        self.assertEqual(tuple(group.length for group in groups), (1, 2, 3, 4))
        self.assertIsInstance(groups[0], UnavailableTrainGroup)
        self.assertIsInstance(groups[1], PreparedTrainGroup)
        self.assertIsInstance(groups[2], UnavailableTrainGroup)
        self.assertIsInstance(groups[3], UnavailableTrainGroup)
        self.assertEqual(groups[0].reason, "insufficient_series_length")  # type: ignore[union-attr]
        self.assertEqual(groups[2].reason, "nonfinite_train")  # type: ignore[union-attr]
        self.assertEqual(groups[3].reason, "insufficient_train_rows")  # type: ignore[union-attr]
        fitted = fit_calibrated_group(groups[1])
        self.assertIsInstance(fitted, CalibratedTrainGroup)

    def test_fit_returns_structured_unavailable_outcomes_for_invalid_prototype(self) -> None:
        sparse = fit_calibrated_group(prepared_group([[0, 1], [1, 0], [0, 1]]))
        self.assertIsInstance(sparse, UnavailableTrainGroup)
        self.assertEqual(sparse.reason, "insufficient_train_rows")  # type: ignore[union-attr]
        zero_scale = fit_calibrated_group(prepared_group([[1, 1], [1, 1], [1, 1], [1, 1]]))
        self.assertIsInstance(zero_scale, UnavailableTrainGroup)
        self.assertEqual(zero_scale.reason, "zero_scale")  # type: ignore[union-attr]

    def test_score_uses_float32_rms_and_nearest_three(self) -> None:
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray(
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 100.0]], dtype=np.float32
        )
        expected = float(
            np.median(
                np.asarray(
                    [
                        np.sqrt(np.float32(0.5)),
                        np.sqrt(np.float32(2.0)),
                        np.sqrt(np.float32(4.5)),
                    ],
                    dtype=np.float32,
                )
            )
        )
        self.assertEqual(score_channel_cpu(query, references, 3), expected)

    def test_acf_requires_length_and_positive_zero_lag_energy(self) -> None:
        self.assertIsNone(acf_vector(np.ones(15, dtype=np.float32), 32, 1e-12))
        self.assertIsNone(acf_vector(np.ones(16, dtype=np.float32), 32, 1e-12))
        values = np.arange(16, dtype=np.float32)
        vector = acf_vector(values, 32, 1e-12)
        self.assertIsNotNone(vector)
        assert vector is not None
        self.assertEqual(vector.dtype, np.dtype(np.float32))
        self.assertEqual(vector.shape, (4,))
        self.assertTrue(np.isfinite(vector).all())

    def test_acf_values_match_float32_golden_vector(self) -> None:
        values = np.arange(16, dtype=np.float32)
        expected = np.asarray(
            [0.8125, 0.6279411911964417, 0.4492647051811218, 0.27941176295280457],
            dtype=np.float32,
        )
        vector = acf_vector(values, 32, 1e-12)
        self.assertIsNotNone(vector)
        assert vector is not None
        np.testing.assert_array_equal(vector, expected)

    def test_acf_respects_requested_lag_cap_and_floor(self) -> None:
        values = np.sin(np.arange(128, dtype=np.float32))
        vector = acf_vector(values, 2, 1e-12)
        self.assertIsNotNone(vector)
        assert vector is not None
        self.assertEqual(vector.shape, (2,))
        self.assertIsNone(acf_vector(values, 32, np.inf))

    def test_short_or_insufficient_groups_are_structured_with_preregistered_reason(self) -> None:
        sparse = fit_calibrated_group(prepared_group([[0, 1], [1, 0], [0, 1]]))
        self.assertIsInstance(sparse, UnavailableTrainGroup)
        self.assertEqual(sparse.reason, "insufficient_train_rows")  # type: ignore[union-attr]

        short = prepare_train_groups(
            RawTrainDataset(
                "Demo",
                tuple(RawSeries(index, (float(index),)) for index in range(4)),
            )
        )
        self.assertIsInstance(short[0], UnavailableTrainGroup)
        self.assertEqual(short[0].reason, "insufficient_series_length")  # type: ignore[union-attr]

    def test_zero_mad_is_rejected_without_fabricating_scale(self) -> None:
        group = prepared_group([[1, 1], [1, 1], [1, 1], [1, 1]])
        fitted = fit_calibrated_group(group)
        self.assertIsInstance(fitted, UnavailableTrainGroup)
        self.assertEqual(fitted.reason, "zero_scale")  # type: ignore[union-attr]

    def test_gradient_unavailable_does_not_hide_available_acf(self) -> None:
        base = np.arange(16, dtype=np.float32)
        rows = [base, base + 10.0, base + 20.0, base + 30.0]
        group = prepared_group([row.tolist() for row in rows])
        fitted = fit_calibrated_group(group)
        self.assertFalse(fitted.gradient.available)
        self.assertIsNone(fitted.gradients)
        self.assertTrue(fitted.acf.available)
        self.assertIsNotNone(fitted.acf_values)

    def test_acf_unavailable_does_not_hide_available_gradient(self) -> None:
        rows = [[float(i), float(i + 1)] for i in range(4)]
        fitted = fit_calibrated_group(prepared_group(rows))
        self.assertTrue(fitted.gradient.available)
        self.assertIsNotNone(fitted.gradients)
        self.assertFalse(fitted.acf.available)
        self.assertIsNone(fitted.acf_values)

    def test_acf_overflow_is_recorded_as_nonfinite_not_zero_energy(self) -> None:
        row = np.asarray(
            [1e38 if index % 2 else -1e38 for index in range(16)],
            dtype=np.float32,
        )
        rows = [row, -row, row + np.float32(1e30), -row + np.float32(1e30)]
        fitted = fit_calibrated_group(
            prepared_group([value.tolist() for value in rows])
        )
        self.assertFalse(fitted.acf.available)
        self.assertEqual(fitted.acf.reason, "nonfinite_train")

    def test_reference_digest_is_canonical_across_row_order(self) -> None:
        rows = [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [0.5, 1.5, 2.5, 3.5], [2.0, 3.0, 4.0, 5.0]]
        first = fit_calibrated_group(prepared_group(rows))
        permutation = (2, 0, 3, 1)
        second = fit_calibrated_group(
            PreparedTrainGroup(
                "Demo",
                4,
                (0, 1, 2, 3),
                np.asarray([rows[index] for index in permutation], dtype=np.float32),
            )
        )
        self.assertEqual(first.prototype.reference_digest, second.prototype.reference_digest)

    def test_reference_digest_is_sha256_of_canonical_float32_references(self) -> None:
        rows = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.5, 1.5, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
        fitted = fit_calibrated_group(prepared_group(rows))
        self.assertEqual(
            fitted.prototype.reference_digest,
            "8788dce3d35dae25ef7f49b4ad74e99ede5905947b229b0ad336f277e3c3f50b",
        )

    def test_fit_returns_readonly_float32_terminal_arrays(self) -> None:
        fitted = fit_calibrated_group(
            prepared_group([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])
        )
        self.assertIsInstance(fitted, CalibratedTrainGroup)
        self.assertEqual(fitted.values.dtype, np.dtype(np.float32))
        self.assertFalse(fitted.values.flags.writeable)
        if fitted.gradients is not None:
            self.assertEqual(fitted.gradients.dtype, np.dtype(np.float32))
            self.assertFalse(fitted.gradients.flags.writeable)
        if fitted.acf_values is not None:
            self.assertEqual(fitted.acf_values.dtype, np.dtype(np.float32))
            self.assertFalse(fitted.acf_values.flags.writeable)


if __name__ == "__main__":
    unittest.main()
