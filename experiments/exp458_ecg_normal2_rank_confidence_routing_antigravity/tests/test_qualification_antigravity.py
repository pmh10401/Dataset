from __future__ import annotations

from collections.abc import Mapping
import unittest
from unittest.mock import patch

import numpy as np


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
SEEDS_ANTIGRAVITY = (20260721, 20261721)


def rows_antigravity(size: int) -> np.ndarray:
    matrix = np.column_stack(
        (
            np.arange(size, dtype=np.float32),
            np.arange(size, dtype=np.float32) % 17,
            np.ones((size, 8), dtype=np.float32),
        )
    )
    matrix.setflags(write=False)
    return matrix


def original_scores_antigravity(size: int) -> np.ndarray:
    return np.arange(size, dtype=float)


def auc_vector_antigravity(size: int, wins: int) -> np.ndarray:
    total_pairs = size * size
    if wins < 0 or wins > total_pairs:
        raise ValueError("wins must stay within the exact pair count")
    per_row, remainder = divmod(wins, size)
    if per_row > size or (per_row == size and remainder):
        raise ValueError("wins cannot exceed one row's exact maximum")
    counts = np.full(size, per_row, dtype=int)
    counts[:remainder] += 1
    if counts.max(initial=0) > size:
        raise ValueError("wins exceed the available normal comparisons")
    return counts.astype(float) - 0.5


def vectors_antigravity(
    size: int,
    *,
    level_shift_wins: int,
    gradual_drift_wins: int,
    amplitude_scaling_wins: int,
    period_change_wins: int,
) -> dict[str, np.ndarray]:
    return {
        "original": original_scores_antigravity(size),
        "level_shift": auc_vector_antigravity(size, level_shift_wins),
        "gradual_drift": auc_vector_antigravity(size, gradual_drift_wins),
        "amplitude_scaling": auc_vector_antigravity(size, amplitude_scaling_wins),
        "period_change": auc_vector_antigravity(size, period_change_wins),
    }


def fold_scorer_from_vectors_antigravity(
    per_seed: Mapping[int, Mapping[str, np.ndarray]],
    *,
    feature_dimension: int = 17,
):
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.qualification_antigravity import (
        FoldScoresAntigravity,
    )

    def score(
        _train: np.ndarray,
        _fit_indices: np.ndarray,
        query_indices: np.ndarray,
        *,
        seed: int,
        fold_index: int,
    ) -> object:
        _ = fold_index
        bundle = per_seed[seed]
        primary = {
            name: np.asarray(bundle[name][query_indices], dtype=float)
            for name in PRIMARY_TYPES_ANTIGRAVITY
        }
        return FoldScoresAntigravity(
            np.asarray(bundle["original"][query_indices], dtype=float),
            primary,
            feature_dimension,
        )

    return score


class FoldScoresAntigravityTest(unittest.TestCase):
    def test_fold_scores_require_exact_primary_order_lengths_dimension_and_finite_values_antigravity(self) -> None:
        """Break caught: a fold can reorder primary types, mismatch lengths, or leak NaN scores."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.qualification_antigravity import (
            FoldScoresAntigravity,
        )

        valid_primary = {
            name: np.asarray((1.0, 2.0), dtype=np.float64)
            for name in PRIMARY_TYPES_ANTIGRAVITY
        }
        fold = FoldScoresAntigravity(np.asarray((0.0, 1.0)), valid_primary, 2)
        self.assertEqual(tuple(fold.primary), PRIMARY_TYPES_ANTIGRAVITY)

        reversed_primary = {
            name: np.asarray((1.0, 2.0), dtype=np.float64)
            for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)
        }
        with self.assertRaisesRegex(ValueError, "ordered"):
            FoldScoresAntigravity(np.asarray((0.0, 1.0)), reversed_primary, 2)

        short_primary = dict(valid_primary)
        short_primary["level_shift"] = np.asarray((1.0,), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "length"):
            FoldScoresAntigravity(np.asarray((0.0, 1.0)), short_primary, 2)

        with self.assertRaisesRegex(ValueError, "feature_dimension"):
            FoldScoresAntigravity(np.asarray((0.0, 1.0)), valid_primary, 0)

        with self.assertRaisesRegex(ValueError, "finite"):
            FoldScoresAntigravity(np.asarray((0.0, np.nan)), valid_primary, 2)

        nonfinite_primary = dict(valid_primary)
        nonfinite_primary["level_shift"] = np.asarray((1.0, np.inf))
        with self.assertRaisesRegex(ValueError, "finite"):
            FoldScoresAntigravity(np.asarray((0.0, 1.0)), nonfinite_primary, 2)


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_rejects_non_frozen_task3_auc_vectors_antigravity(self) -> None:
        """Break caught: synthetic Task 3 vectors can replace the exact approved seed evidence."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.qualification_antigravity import (
            _qualify_normal2_with_dependencies_antigravity,
        )

        size = 1000
        per_seed = {
            20260721: vectors_antigravity(
                size,
                level_shift_wins=850000,
                gradual_drift_wins=900000,
                amplitude_scaling_wins=900000,
                period_change_wins=950000,
            ),
            20261721: vectors_antigravity(
                size,
                level_shift_wins=850000,
                gradual_drift_wins=900000,
                amplitude_scaling_wins=950000,
                period_change_wins=900000,
            ),
        }

        with self.assertRaisesRegex(ValueError, "rank seed evidence"):
            _qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(size),
                fold_scorer=fold_scorer_from_vectors_antigravity(per_seed),
            )

    def test_qualification_rejects_per_type_pooled_wilson_and_single_seed_failures_antigravity(self) -> None:
        """Break caught: one weak gate or one failing seed can still qualify normal_2."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.qualification_antigravity import (
            _qualify_normal2_with_dependencies_antigravity,
        )

        size = 1000
        passing = vectors_antigravity(
            size,
            level_shift_wins=900000,
            gradual_drift_wins=900000,
            amplitude_scaling_wins=900000,
            period_change_wins=900000,
        )
        cases = (
            (
                size,
                {
                    20260721: vectors_antigravity(
                        size,
                        level_shift_wins=849999,
                        gradual_drift_wins=950000,
                        amplitude_scaling_wins=950000,
                        period_change_wins=950000,
                    ),
                    20261721: passing,
                },
            ),
            (
                size,
                {
                    20260721: vectors_antigravity(
                        size,
                        level_shift_wins=900000,
                        gradual_drift_wins=900000,
                        amplitude_scaling_wins=900000,
                        period_change_wins=899996,
                    ),
                    20261721: passing,
                },
            ),
            (
                size,
                {
                    20260721: passing,
                    20261721: vectors_antigravity(
                        size,
                        level_shift_wins=849999,
                        gradual_drift_wins=950000,
                        amplitude_scaling_wins=950000,
                        period_change_wins=950000,
                    ),
                },
            ),
            (
                72,
                {
                    20260721: vectors_antigravity(
                        72,
                        level_shift_wins=5184,
                        gradual_drift_wins=5184,
                        amplitude_scaling_wins=5184,
                        period_change_wins=5184,
                    ),
                    20261721: vectors_antigravity(
                        72,
                        level_shift_wins=5184,
                        gradual_drift_wins=5184,
                        amplitude_scaling_wins=5184,
                        period_change_wins=5184,
                    ),
                },
            ),
        )
        for train_size, per_seed in cases:
            with self.subTest(train_size=train_size), self.assertRaisesRegex(ValueError, "rank seed evidence"):
                _qualify_normal2_with_dependencies_antigravity(
                    loader=lambda train_size=train_size: rows_antigravity(train_size),
                    fold_scorer=fold_scorer_from_vectors_antigravity(per_seed),
                )

    def test_qualification_rejects_nonfinite_scores_and_bad_crossfit_coverage_antigravity(self) -> None:
        """Break caught: missing, duplicated, overlapping, or non-finite fold evidence is accepted."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            qualification_antigravity as qualification,
        )

        size = 20
        passing = {
            seed: vectors_antigravity(
                size,
                level_shift_wins=400,
                gradual_drift_wins=400,
                amplitude_scaling_wins=400,
                period_change_wins=400,
            )
            for seed in SEEDS_ANTIGRAVITY
        }

        broken = dict(passing[20260721])
        broken["original"] = np.array(broken["original"], copy=True)
        broken["original"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            qualification._qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(size),
                fold_scorer=fold_scorer_from_vectors_antigravity(
                    {
                        20260721: broken,
                        20261721: passing[20261721],
                    }
                ),
            )

        missing_plan = (
            (np.asarray([1, 2, 3, 4, 5]), np.asarray([0])),
            (np.asarray([0, 2, 3, 4, 5]), np.asarray([1])),
            (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
            (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
            (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
        )
        with patch.object(qualification, "crossfit_plan_antigravity", return_value=missing_plan), self.assertRaisesRegex(
            ValueError, "exactly once"
        ):
            qualification._qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(6),
                fold_scorer=fold_scorer_from_vectors_antigravity(
                    {
                        20260721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                        20261721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                    }
                ),
            )

        duplicate_plan = (
            (np.asarray([1, 2, 3, 4, 5]), np.asarray([0])),
            (np.asarray([0, 2, 3, 4, 5]), np.asarray([1])),
            (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
            (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
            (np.asarray([0, 1, 2, 4]), np.asarray([3, 5])),
        )
        with patch.object(qualification, "crossfit_plan_antigravity", return_value=duplicate_plan), self.assertRaisesRegex(
            ValueError, "exactly once"
        ):
            qualification._qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(6),
                fold_scorer=fold_scorer_from_vectors_antigravity(
                    {
                        20260721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                        20261721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                    }
                ),
            )

        duplicate_with_full_union_plan = (
            (np.asarray([2, 3, 4, 5]), np.asarray([0, 0])),
            (np.asarray([0, 1, 3, 4, 5]), np.asarray([1])),
            (np.asarray([0, 1, 2, 4, 5]), np.asarray([2])),
            (np.asarray([0, 1, 2, 3, 5]), np.asarray([3])),
            (np.asarray([0, 1, 2, 3, 4]), np.asarray([4, 5])),
        )
        with patch.object(
            qualification,
            "crossfit_plan_antigravity",
            return_value=duplicate_with_full_union_plan,
        ), self.assertRaisesRegex(ValueError, "unique"):
            qualification._qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(6),
                fold_scorer=fold_scorer_from_vectors_antigravity(
                    {
                        20260721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                        20261721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                    }
                ),
            )

        overlap_plan = (
            (np.asarray([1, 2, 3, 4, 5]), np.asarray([0])),
            (np.asarray([0, 2, 3, 4, 5]), np.asarray([1])),
            (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
            (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
            (np.asarray([0, 1, 2, 3, 4]), np.asarray([4, 5])),
        )
        with patch.object(qualification, "crossfit_plan_antigravity", return_value=overlap_plan), self.assertRaisesRegex(
            ValueError, "disjoint"
        ):
            qualification._qualify_normal2_with_dependencies_antigravity(
                loader=lambda: rows_antigravity(6),
                fold_scorer=fold_scorer_from_vectors_antigravity(
                    {
                        20260721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                        20261721: vectors_antigravity(6, level_shift_wins=36, gradual_drift_wins=36, amplitude_scaling_wins=36, period_change_wins=36),
                    }
                ),
            )

    def test_qualification_requires_exact_five_folds_and_valid_index_vectors_antigravity(self) -> None:
        """Break caught: fold count, dtype, bounds, or per-fold uniqueness drifts from the frozen cross-fit contract."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity import (
            qualification_antigravity as qualification,
        )

        passing = {
            seed: vectors_antigravity(
                6,
                level_shift_wins=36,
                gradual_drift_wins=36,
                amplitude_scaling_wins=36,
                period_change_wins=36,
            )
            for seed in SEEDS_ANTIGRAVITY
        }

        cases = (
            (
                "four-folds",
                (
                    (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
                    (np.asarray([0, 1, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 5]), np.asarray([3, 4])),
                    (np.asarray([0, 1, 2, 3]), np.asarray([5])),
                ),
                "five folds",
            ),
            (
                "six-folds",
                (
                    (np.asarray([1, 2, 3, 4, 5]), np.asarray([0])),
                    (np.asarray([0, 2, 3, 4, 5]), np.asarray([1])),
                    (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
                    (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
                    (np.asarray([0, 1, 2, 3, 4]), np.asarray([5])),
                ),
                "five folds",
            ),
            (
                "float-query",
                (
                    (np.asarray([2, 3, 4, 5]), np.asarray([0.0, 1.0])),
                    (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
                    (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
                    (np.asarray([0, 1, 2, 3, 4]), np.asarray([5])),
                ),
                "integer",
            ),
            (
                "bool-fit",
                (
                    (np.asarray([True, False, True, False, True, False]), np.asarray([0])),
                    (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
                    (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
                    (np.asarray([0, 1, 2, 3, 4]), np.asarray([5])),
                ),
                "integer",
            ),
            (
                "out-of-range",
                (
                    (np.asarray([2, 3, 4, 5]), np.asarray([0, 6])),
                    (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
                    (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
                    (np.asarray([0, 1, 2, 3, 4]), np.asarray([5])),
                ),
                "range",
            ),
            (
                "duplicate-fit",
                (
                    (np.asarray([2, 2, 4, 5]), np.asarray([0])),
                    (np.asarray([0, 1, 3, 4, 5]), np.asarray([2])),
                    (np.asarray([0, 1, 2, 4, 5]), np.asarray([3])),
                    (np.asarray([0, 1, 2, 3, 5]), np.asarray([4])),
                    (np.asarray([0, 1, 2, 3, 4]), np.asarray([5])),
                ),
                "unique",
            ),
        )
        for name, plan, message in cases:
            with self.subTest(name=name), patch.object(
                qualification,
                "crossfit_plan_antigravity",
                return_value=plan,
            ), self.assertRaisesRegex(ValueError, message):
                qualification._qualify_normal2_with_dependencies_antigravity(
                    loader=lambda: rows_antigravity(6),
                    fold_scorer=fold_scorer_from_vectors_antigravity(passing),
                )


if __name__ == "__main__":
    unittest.main()
