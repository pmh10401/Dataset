from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    RawSeries,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    build_horizons,
    canonical_waveform_sha256,
    identity_grouped_folds,
    preprocess_length_groups,
)


def synthetic_series(count: int, length: int, *, start: int = 0) -> tuple[RawSeries, ...]:
    """Create finite, varied TRAIN-only waveforms with nonzero raw/diff MADs."""
    positions = np.arange(length, dtype=np.float64)
    return tuple(
        RawSeries(
            instance_index=start + index,
            values=tuple(
                10.0 * index
                + (1.0 + (index % 5) * 0.1) * positions
                + np.sin(positions * (index % 3 + 1) / 7.0)
            ),
        )
        for index in range(count)
    )


class HorizonTests(unittest.TestCase):
    def test_horizons_use_frozen_relative_offsets(self) -> None:
        self.assertEqual(build_horizons(100), (1, 2, 10))
        self.assertEqual(build_horizons(64), (1, 2, 7))

    def test_horizons_reject_duplicate_bool_and_unusable_lengths(self) -> None:
        for length in (True, 0, -1):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "length"):
                build_horizons(length)
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_horizons(10)
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_horizons(1)


class IdentityFoldTests(unittest.TestCase):
    def test_waveform_identity_binds_length_and_canonical_values(self) -> None:
        same_a = RawSeries(instance_index=3, values=(1.0, 2.0, 3.0))
        same_b = RawSeries(instance_index=9, values=(1.0, 2.0, 3.0))
        different_length = RawSeries(instance_index=4, values=(1.0, 2.0, 3.0, 0.0))
        different_value = RawSeries(instance_index=5, values=(1.0, 2.0, 4.0))
        self.assertEqual(canonical_waveform_sha256(same_a), canonical_waveform_sha256(same_b))
        self.assertNotEqual(canonical_waveform_sha256(same_a), canonical_waveform_sha256(different_length))
        self.assertNotEqual(canonical_waveform_sha256(same_a), canonical_waveform_sha256(different_value))

    def test_identity_folds_are_deterministic_complete_and_isolated(self) -> None:
        unique = synthetic_series(160, 64)
        duplicate = RawSeries(instance_index=160, values=unique[0].values)
        series = unique + (duplicate,)
        first = identity_grouped_folds(series)
        second = identity_grouped_folds(series)
        self.assertEqual(first, second)

        validation_indices = tuple(index for fold in first for index in fold.validation_indices)
        self.assertEqual(tuple(sorted(validation_indices)), tuple(range(161)))
        duplicate_folds = [
            fold_index
            for fold_index, fold in enumerate(first)
            if 0 in fold.validation_indices or 160 in fold.validation_indices
        ]
        self.assertEqual(duplicate_folds, [duplicate_folds[0]])
        self.assertTrue(any({0, 160}.issubset(set(fold.validation_indices)) for fold in first))

        identities = {item.instance_index: canonical_waveform_sha256(item) for item in series}
        for fold in first:
            self.assertTrue(set(fold.early_stop_fit_indices).issubset(fold.fit_indices))
            self.assertTrue(set(fold.early_stop_validation_indices).issubset(fold.fit_indices))
            self.assertFalse(set(fold.early_stop_fit_indices) & set(fold.early_stop_validation_indices))
            self.assertFalse(
                {identities[index] for index in fold.early_stop_fit_indices}
                & {identities[index] for index in fold.early_stop_validation_indices}
            )
            self.assertGreaterEqual(
                len({identities[index] for index in fold.early_stop_fit_indices}), 5
            )
            self.assertGreaterEqual(
                len({identities[index] for index in fold.early_stop_validation_indices}), 5
            )

    def test_identity_folds_require_five_early_stop_identities_on_each_side(self) -> None:
        with self.assertRaisesRegex(ValueError, "five identities"):
            identity_grouped_folds(synthetic_series(9, 64))


class PreprocessingTests(unittest.TestCase):
    def test_groups_use_exact_lengths_without_resampling(self) -> None:
        dataset = RawTrainDataset(
            dataset="synthetic",
            series=synthetic_series(50, 64) + synthetic_series(50, 100, start=50),
        )
        groups = preprocess_length_groups(dataset)
        self.assertEqual(tuple(group.group.length for group in groups), (64, 100))
        self.assertEqual(tuple(len(group.group.series) for group in groups), (50, 50))
        self.assertEqual(tuple(group.status for group in groups), ("diagnostic_only", "diagnostic_only"))
        for group in groups:
            self.assertEqual(
                {channels.shape for channels in group.channels.values()},
                {(2, group.group.length)},
            )

    def test_group_scaling_preserves_level_shift_that_per_series_normalization_would_remove(self) -> None:
        dataset = RawTrainDataset(dataset="synthetic", series=synthetic_series(100, 64))
        group = preprocess_length_groups(dataset)[0]
        low_mean = float(np.mean(group.channels[0][0]))
        high_mean = float(np.mean(group.channels[99][0]))
        self.assertLess(low_mean, -0.5)
        self.assertGreater(high_mean, 0.5)

    def test_nonfinite_train_values_are_repaired_without_changing_length(self) -> None:
        series = list(synthetic_series(50, 64))
        values = list(series[0].values)
        values[0] = float("nan")
        values[16] = float("inf")
        series[0] = RawSeries(instance_index=0, values=tuple(values))
        group = preprocess_length_groups(RawTrainDataset(dataset="synthetic", series=tuple(series)))[0]
        repaired = np.asarray(group.group.series[0].values, dtype=np.float32)
        self.assertEqual(repaired.shape, (64,))
        self.assertTrue(np.isfinite(repaired).all())
        self.assertEqual(float(repaired[0]), float(repaired[1]))

    def test_all_missing_and_unrepaired_nonfinite_groups_abstain_fail_closed(self) -> None:
        series = list(synthetic_series(50, 64))
        series[0] = RawSeries(instance_index=0, values=(float("nan"),) * 64)
        group = preprocess_length_groups(RawTrainDataset(dataset="synthetic", series=tuple(series)))[0]
        self.assertEqual(group.status, "nonfinite_abstain")
        self.assertEqual(dict(group.channels), {})

    def test_nonfinite_standardized_residual_abstains_fail_closed(self) -> None:
        positions = np.arange(64, dtype=np.float64)
        series = list(
            RawSeries(
                instance_index=index,
                values=tuple(
                    index * 0.001
                    + (0.0001 + (index % 3) * 0.00001) * positions
                ),
            )
            for index in range(50)
        )
        extreme = list(series[0].values)
        extreme[32] = float(np.finfo(np.float32).max)
        series[0] = RawSeries(instance_index=0, values=tuple(extreme))
        group = preprocess_length_groups(RawTrainDataset(dataset="synthetic", series=tuple(series)))[0]
        self.assertEqual(group.status, "nonfinite_abstain")
        self.assertEqual(dict(group.channels), {})

    def test_zero_raw_and_difference_scales_abstain(self) -> None:
        raw_zero = RawTrainDataset(
            dataset="synthetic",
            series=tuple(RawSeries(instance_index=index, values=(3.0,) * 64) for index in range(50)),
        )
        raw_group = preprocess_length_groups(raw_zero)[0]
        self.assertEqual(raw_group.status, "zero_scale_abstain")
        self.assertLessEqual(raw_group.raw_scale, 1e-12)

        positions = np.arange(64, dtype=float)
        diff_zero = RawTrainDataset(
            dataset="synthetic",
            series=tuple(
                RawSeries(instance_index=index, values=tuple(positions + index * 100.0))
                for index in range(50)
            ),
        )
        diff_group = preprocess_length_groups(diff_zero)[0]
        self.assertEqual(diff_group.status, "zero_scale_abstain")
        self.assertGreater(diff_group.raw_scale, 1e-12)
        self.assertLessEqual(diff_group.difference_scale, 1e-12)

    def test_processed_channels_are_float32_finite_and_immutable(self) -> None:
        group = preprocess_length_groups(
            RawTrainDataset(dataset="synthetic", series=synthetic_series(100, 64))
        )[0]
        self.assertEqual(group.status, "candidate_eligible")
        channels = group.channels[0]
        self.assertEqual(channels.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(channels).all())
        self.assertFalse(channels.flags.writeable)
        with self.assertRaises(ValueError):
            channels[0, 0] = 0.0
        with self.assertRaises(TypeError):
            group.channels[999] = channels  # type: ignore[index]

    def test_eligibility_boundaries_are_exact(self) -> None:
        statuses = {}
        for count in (49, 50, 99, 100):
            dataset = RawTrainDataset(dataset="synthetic", series=synthetic_series(count, 64))
            statuses[count] = preprocess_length_groups(dataset)[0].status
        self.assertEqual(
            statuses,
            {
                49: "insufficient_train_abstain",
                50: "diagnostic_only",
                99: "diagnostic_only",
                100: "candidate_eligible",
            },
        )

    def test_preprocessing_rejects_mutated_frozen_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            preprocess_length_groups(
                RawTrainDataset(dataset="synthetic", series=synthetic_series(50, 64)),
                config=replace(FROZEN_CONFIG, min_train_model=51),
            )


if __name__ == "__main__":
    unittest.main()
