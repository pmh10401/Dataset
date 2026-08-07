"""TDD coverage for Exp436's missing-safe TRAIN-only features."""

from __future__ import annotations

from hashlib import sha256
import unittest

import numpy as np

from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    PreparedTrainGroup as Exp435PreparedTrainGroup,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.features_antigravity import (
    fit_calibrated_group as fit_exp435_group,
)

from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    CalibratedTrainGroup,
    MissingSafeRawDataset,
    PreparedTrainGroup,
    RawValueSeries,
    UnavailableTrainGroup,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.features_antigravity import (
    acf_vector,
    fit_calibrated_group,
    prepare_train_groups,
    score_channel_cpu,
)


def _row(index: int, values: object, *, digest: str | None = None) -> RawValueSeries:
    array = np.asarray(values, dtype="<f4")
    return RawValueSeries(
        index,
        tuple(float(value) for value in array),
        sha256(array.tobytes()).hexdigest() if digest is None else digest,
    )


def _train(*rows: RawValueSeries) -> MissingSafeRawDataset:
    return MissingSafeRawDataset("Demo", "TRAIN", tuple(sorted(rows, key=lambda item: item.instance_index)))


class Exp436FeatureTests(unittest.TestCase):
    def test_one_nonfinite_row_excludes_only_it_and_preserves_four_exact_finite_siblings(self) -> None:
        rows = [_row(index, np.arange(16, dtype=np.float32) + index) for index in (0, 1, 3, 4)]
        bad_digest = "e" * 64
        bad = _row(2, (0.0, np.nan, np.inf, *range(13)), digest=bad_digest)
        prepared = prepare_train_groups(_train(*rows, bad))

        self.assertEqual(prepared.train_identity_lengths, ((0, 16), (1, 16), (2, 16), (3, 16), (4, 16)))
        group = prepared.groups[0]
        self.assertIsInstance(group, PreparedTrainGroup)
        assert isinstance(group, PreparedTrainGroup)
        self.assertEqual(group.instance_indices, (0, 1, 3, 4))
        np.testing.assert_array_equal(group.values, np.asarray([row.values for row in rows], dtype=np.float32))
        self.assertFalse(group.values.flags.writeable)
        self.assertEqual(len(prepared.excluded_rows), 1)
        audit = prepared.excluded_rows[0]
        self.assertEqual((audit.instance_index, audit.length, audit.nonfinite_count, audit.raw_blob_sha256), (2, 16, 2, bad_digest))

    def test_three_and_zero_finite_rows_make_only_their_exact_lengths_unavailable(self) -> None:
        finite16 = [_row(index, np.arange(16, dtype=np.float32) + index) for index in range(3)]
        bad16 = _row(3, (np.nan, *range(15)))
        bad17 = [_row(index, (np.inf, *range(16))) for index in range(4, 8)]
        finite18 = [_row(index, np.arange(18, dtype=np.float32) + index) for index in range(8, 12)]
        prepared = prepare_train_groups(_train(*finite16, bad16, *bad17, *finite18))

        self.assertEqual(tuple(group.length for group in prepared.groups), (16, 17, 18))
        self.assertEqual(prepared.groups[0].instance_indices, (0, 1, 2))
        self.assertEqual(prepared.groups[1].instance_indices, ())
        self.assertTrue(all(isinstance(group, UnavailableTrainGroup) for group in prepared.groups[:2]))
        self.assertTrue(all(group.reason == "insufficient_finite_train_rows" for group in prepared.groups[:2]))  # type: ignore[union-attr]
        self.assertIsInstance(prepared.groups[2], PreparedTrainGroup)
        self.assertEqual(tuple((item.length, item.instance_index) for item in prepared.excluded_rows), ((16, 3), (17, 4), (17, 5), (17, 6), (17, 7)))

    def test_four_finite_length_one_rows_remain_preparable_and_calibratable(self) -> None:
        prepared = prepare_train_groups(_train(*[_row(index, (float(index),)) for index in range(4)]))
        self.assertIsInstance(prepared.groups[0], PreparedTrainGroup)
        group = prepared.groups[0]
        assert isinstance(group, PreparedTrainGroup)
        self.assertEqual((group.length, group.instance_indices), (1, (0, 1, 2, 3)))
        fitted = fit_calibrated_group(group)
        self.assertIsInstance(fitted, CalibratedTrainGroup)
        assert isinstance(fitted, CalibratedTrainGroup)
        self.assertTrue(fitted.prototype.available)
        self.assertFalse(fitted.gradient.available)
        self.assertEqual(fitted.gradient.reason, "zero_scale")
        self.assertFalse(fitted.acf.available)
        self.assertEqual(fitted.acf.reason, "short_series")

    def test_prepare_requires_a_frozen_train_container(self) -> None:
        dataset = MissingSafeRawDataset("Demo", "TEST", (_row(0, np.arange(16)),))
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            prepare_train_groups(dataset)
        with self.assertRaisesRegex(ValueError, "MissingSafeRawDataset"):
            prepare_train_groups(object())  # type: ignore[arg-type]

    def test_all_finite_fixture_matches_exp435_shared_calibrations_exactly(self) -> None:
        matrix = np.asarray([
            np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32) * 2 + 1,
            np.arange(16, dtype=np.float32) * 3 - 2, np.arange(16, dtype=np.float32) * -1 + 7,
        ], dtype=np.float32)
        exp436_prepared = prepare_train_groups(_train(*[_row(index, row) for index, row in enumerate(matrix)])).groups[0]
        self.assertIsInstance(exp436_prepared, PreparedTrainGroup)
        exp436 = fit_calibrated_group(exp436_prepared)
        exp435 = fit_exp435_group(Exp435PreparedTrainGroup("Demo", 16, (0, 1, 2, 3), matrix))
        self.assertIsInstance(exp436, CalibratedTrainGroup)
        self.assertEqual(type(exp436), CalibratedTrainGroup)
        for channel in ("prototype", "gradient", "acf"):
            newer = getattr(exp436, channel)
            older = getattr(exp435, channel)
            self.assertEqual(newer.available, older.available)
            self.assertEqual(newer.reference_digest, older.reference_digest)
            self.assertEqual(newer.loo_scores, older.loo_scores)
            self.assertEqual(newer.threshold, older.threshold)
            self.assertLessEqual(abs((newer.center or 0.0) - (older.center or 0.0)), 1e-7)
            self.assertLessEqual(abs((newer.scale or 0.0) - (older.scale or 0.0)), 1e-7)

    def test_prototype_zero_scale_keeps_gradient_and_acf_calculations(self) -> None:
        rows = []
        for multiplier in (1, 2, 3, 4):
            rows.append(np.asarray([0, multiplier, 0, multiplier * 2, 0, multiplier * 3, 0, multiplier * 4,
                                    0, multiplier * 5, 0, multiplier * 6, 0, multiplier * 7, 0, 0], dtype=np.float32))
        fitted = fit_calibrated_group(prepare_train_groups(_train(*[_row(index, row) for index, row in enumerate(rows)])).groups[0])
        self.assertIsInstance(fitted, CalibratedTrainGroup)
        assert isinstance(fitted, CalibratedTrainGroup)
        self.assertFalse(fitted.prototype.available)
        self.assertEqual(fitted.prototype.reason, "zero_scale")
        self.assertIsNone(fitted.values)
        self.assertTrue(fitted.gradient.available)
        self.assertIsNotNone(fitted.gradients)
        self.assertTrue(fitted.acf.available)
        self.assertIsNotNone(fitted.acf_values)

    def test_gradient_zero_scale_keeps_prototype_and_acf_calculations(self) -> None:
        base = np.arange(16, dtype=np.float32)
        fitted = fit_calibrated_group(prepare_train_groups(_train(*[_row(index, base + index * 10) for index in range(4)])).groups[0])
        self.assertIsInstance(fitted, CalibratedTrainGroup)
        assert isinstance(fitted, CalibratedTrainGroup)
        self.assertTrue(fitted.prototype.available)
        self.assertFalse(fitted.gradient.available)
        self.assertEqual(fitted.gradient.reason, "zero_scale")
        self.assertIsNone(fitted.gradients)
        self.assertTrue(fitted.acf.available)

    def test_short_and_zero_energy_acf_are_channel_local_unavailable(self) -> None:
        short = fit_calibrated_group(prepare_train_groups(_train(*[_row(index, (0, index + 1)) for index in range(4)])).groups[0])
        self.assertIsInstance(short, CalibratedTrainGroup)
        assert isinstance(short, CalibratedTrainGroup)
        self.assertFalse(short.acf.available)
        self.assertEqual(short.acf.reason, "short_series")

        constants = fit_calibrated_group(prepare_train_groups(_train(*[_row(index, np.full(16, index, dtype=np.float32)) for index in range(4)])).groups[0])
        self.assertIsInstance(constants, CalibratedTrainGroup)
        assert isinstance(constants, CalibratedTrainGroup)
        self.assertFalse(constants.acf.available)
        self.assertEqual(constants.acf.reason, "zero_acf_energy")
        self.assertTrue(constants.prototype.available)

    def test_nonfinite_numerical_arithmetic_propagates_as_value_error(self) -> None:
        huge = np.asarray([np.float32(3e38), np.float32(-3e38)] * 8, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            acf_vector(huge, 32, 1e-12)
        with self.assertRaisesRegex(ValueError, "score is nonfinite"):
            score_channel_cpu(np.asarray([3e38, 3e38], dtype=np.float32), np.asarray([[-3e38, -3e38]] * 3, dtype=np.float32), 3)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_calibrated_group(prepare_train_groups(_train(*[_row(index, huge + index) for index in range(4)])).groups[0])

    def test_available_terminal_matrices_are_immutable_float32(self) -> None:
        fitted = fit_calibrated_group(prepare_train_groups(_train(*[_row(index, np.arange(16, dtype=np.float32) * (index + 1)) for index in range(4)])).groups[0])
        self.assertIsInstance(fitted, CalibratedTrainGroup)
        assert isinstance(fitted, CalibratedTrainGroup)
        for matrix in (fitted.values, fitted.gradients, fitted.acf_values):
            if matrix is not None:
                self.assertEqual(matrix.dtype, np.dtype(np.float32))
                with self.assertRaises(ValueError):
                    matrix.setflags(write=True)


if __name__ == "__main__":
    unittest.main()
