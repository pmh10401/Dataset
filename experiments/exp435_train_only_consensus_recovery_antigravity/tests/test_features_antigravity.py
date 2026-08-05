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
                RawSeries(1, (0.0, 1.0)),
                RawSeries(2, (1.0, 0.0, 1.0, 0.0)),
                RawSeries(3, (1.0, 0.0)),
            ),
        )
        groups = prepare_train_groups(train)
        self.assertEqual(tuple(group.length for group in groups), (2, 4))
        self.assertEqual(groups[0].instance_indices, (1, 3))
        self.assertEqual(groups[1].instance_indices, (0, 2))
        self.assertEqual(groups[1].values.dtype, np.dtype(np.float32))
        self.assertFalse(groups[1].values.flags.writeable)

    def test_prepare_rejects_nonfinite_train_values(self) -> None:
        train = RawTrainDataset(
            dataset="Demo",
            series=(
                RawSeries(0, (0.0, 1.0)),
                RawSeries(1, (np.nan, 1.0)),
                RawSeries(2, (0.0, 1.0)),
                RawSeries(3, (0.0, 1.0)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prepare_train_groups(train)

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

    def test_acf_respects_requested_lag_cap_and_floor(self) -> None:
        values = np.sin(np.arange(128, dtype=np.float32))
        vector = acf_vector(values, 2, 1e-12)
        self.assertIsNotNone(vector)
        assert vector is not None
        self.assertEqual(vector.shape, (2,))
        self.assertIsNone(acf_vector(values, 32, np.inf))

    def test_short_or_insufficient_groups_are_rejected_with_preregistered_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "insufficient_train_rows"):
            fit_calibrated_group(prepared_group([[0, 1], [1, 0], [0, 1]]))

        with self.assertRaisesRegex(ValueError, "insufficient_series_length"):
            prepare_train_groups(
                RawTrainDataset(
                    "Demo",
                    tuple(RawSeries(index, (float(index),)) for index in range(4)),
                )
            )

    def test_zero_mad_is_rejected_without_fabricating_scale(self) -> None:
        group = prepared_group([[1, 1], [1, 1], [1, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "zero_scale"):
            fit_calibrated_group(group)

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
