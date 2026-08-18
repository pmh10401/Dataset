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
    def test_qualification_uses_both_exact_seed_orders_and_collects_full_auc_vectors_antigravity(self) -> None:
        """Break caught: Task 3 reorders seeds/types or summarizes anything but the full OOF vectors."""
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

        qualification = _qualify_normal2_with_dependencies_antigravity(
            loader=lambda: rows_antigravity(size),
            fold_scorer=fold_scorer_from_vectors_antigravity(per_seed),
        )

        self.assertEqual(tuple(seed.seed for seed in qualification.seed_evidence), SEEDS_ANTIGRAVITY)
        self.assertTrue(qualification.passed)
        first = qualification.seed_evidence[0]
        self.assertEqual(tuple(first.per_type_auc), PRIMARY_TYPES_ANTIGRAVITY)
        self.assertEqual(first.normal_trials, size)
        self.assertEqual(first.normal_alerts, 4)
        self.assertLess(first.normal_wilson_upper, 0.05)
        self.assertEqual(first.per_type_auc["level_shift"], 0.85)
        self.assertEqual(first.per_type_auc["gradual_drift"], 0.9)
        self.assertEqual(first.per_type_auc["amplitude_scaling"], 0.9)
        self.assertEqual(first.per_type_auc["period_change"], 0.95)
        self.assertEqual(first.pooled_auc, 0.9)

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
        per_type_fail = _qualify_normal2_with_dependencies_antigravity(
            loader=lambda: rows_antigravity(size),
            fold_scorer=fold_scorer_from_vectors_antigravity(
                {
                    20260721: vectors_antigravity(
                        size,
                        level_shift_wins=849999,
                        gradual_drift_wins=950000,
                        amplitude_scaling_wins=950000,
                        period_change_wins=950000,
                    ),
                    20261721: passing,
                }
            ),
        )
        self.assertFalse(per_type_fail.seed_evidence[0].passed)
        self.assertGreater(per_type_fail.seed_evidence[0].pooled_auc, 0.9)
        self.assertEqual(per_type_fail.seed_evidence[0].per_type_auc["level_shift"], 0.849999)
        self.assertFalse(per_type_fail.passed)

        pooled_fail = _qualify_normal2_with_dependencies_antigravity(
            loader=lambda: rows_antigravity(size),
            fold_scorer=fold_scorer_from_vectors_antigravity(
                {
                    20260721: vectors_antigravity(
                        size,
                        level_shift_wins=900000,
                        gradual_drift_wins=900000,
                        amplitude_scaling_wins=900000,
                        period_change_wins=899996,
                    ),
                    20261721: passing,
                }
            ),
        )
        self.assertFalse(pooled_fail.seed_evidence[0].passed)
        self.assertEqual(pooled_fail.seed_evidence[0].pooled_auc, 0.899999)
        self.assertTrue(all(value > 0.85 for value in pooled_fail.seed_evidence[0].per_type_auc.values()))
        self.assertFalse(pooled_fail.passed)

        one_seed_fail = _qualify_normal2_with_dependencies_antigravity(
            loader=lambda: rows_antigravity(size),
            fold_scorer=fold_scorer_from_vectors_antigravity(
                {
                    20260721: passing,
                    20261721: vectors_antigravity(
                        size,
                        level_shift_wins=849999,
                        gradual_drift_wins=950000,
                        amplitude_scaling_wins=950000,
                        period_change_wins=950000,
                    ),
                }
            ),
        )
        self.assertTrue(one_seed_fail.seed_evidence[0].passed)
        self.assertFalse(one_seed_fail.seed_evidence[1].passed)
        self.assertFalse(one_seed_fail.passed)

        wilson_fail = _qualify_normal2_with_dependencies_antigravity(
            loader=lambda: rows_antigravity(72),
            fold_scorer=fold_scorer_from_vectors_antigravity(
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
                }
            ),
        )
        self.assertFalse(any(seed.passed for seed in wilson_fail.seed_evidence))
        self.assertTrue(all(seed.normal_wilson_upper > 0.05 for seed in wilson_fail.seed_evidence))
        self.assertFalse(wilson_fail.passed)

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
            (np.asarray([3, 4, 5]), np.asarray([0, 1, 2])),
            (np.asarray([0, 1, 2]), np.asarray([3, 4])),
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
            (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
            (np.asarray([0, 1, 4, 5]), np.asarray([2, 2, 3])),
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

        overlap_plan = (
            (np.asarray([0, 1, 2, 3]), np.asarray([3, 4, 5])),
            (np.asarray([0, 1, 2]), np.asarray([0, 1, 2])),
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


if __name__ == "__main__":
    unittest.main()
