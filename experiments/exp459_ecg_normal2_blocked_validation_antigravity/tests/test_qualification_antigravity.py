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


def normal2_rows_antigravity() -> np.ndarray:
    rows = np.column_stack(
        (
            np.arange(1413, dtype=np.float32),
            np.arange(1413, dtype=np.float32) % 17,
            np.ones((1413, 138), dtype=np.float32),
        )
    )
    rows.setflags(write=False)
    return rows


def auc_vector_antigravity(size: int, auc: float) -> np.ndarray:
    wins = int(round(size * size * auc))
    per_row, remainder = divmod(wins, size)
    values = np.full(size, per_row, dtype=float)
    values[:remainder] += 1.0
    return values - 0.5


def fold_scores_antigravity(
    size: int,
    *,
    aucs: Mapping[str, float] | None = None,
    feature_dimension: int = 17,
) -> object:
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
        FoldScoresAntigravity,
    )

    actual_aucs = dict.fromkeys(PRIMARY_TYPES_ANTIGRAVITY, 1.0) if aucs is None else aucs
    original = np.arange(size, dtype=float)
    primary = {
        name: auc_vector_antigravity(size, actual_aucs[name])
        for name in PRIMARY_TYPES_ANTIGRAVITY
    }
    return FoldScoresAntigravity(original, primary, feature_dimension)


class FoldScoresAntigravityTest(unittest.TestCase):
    def test_fold_scores_copy_freeze_and_validate_exact_primary_vectors_antigravity(self) -> None:
        """Break caught: scorer output can be mutated, reordered, nonfinite, or dimensionless."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
            FoldScoresAntigravity,
        )

        original = np.asarray((0.0, 1.0, 2.0), dtype=float)
        primary = {
            name: np.asarray((3.0, 4.0, 5.0), dtype=float)
            for name in PRIMARY_TYPES_ANTIGRAVITY
        }
        scores = FoldScoresAntigravity(original, primary, 11)
        original[0] = 99.0
        primary["level_shift"][0] = 88.0
        np.testing.assert_array_equal(scores.original, np.asarray((0.0, 1.0, 2.0)))
        np.testing.assert_array_equal(scores.primary["level_shift"], np.asarray((3.0, 4.0, 5.0)))
        self.assertFalse(scores.original.flags.writeable)
        self.assertFalse(scores.primary["level_shift"].flags.writeable)

        reversed_primary = {
            name: np.asarray((3.0, 4.0, 5.0), dtype=float)
            for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)
        }
        with self.assertRaisesRegex(ValueError, "ordered"):
            FoldScoresAntigravity(np.asarray((0.0, 1.0, 2.0)), reversed_primary, 11)

        invalid_cases = (
            (np.asarray((0.0, np.nan, 2.0)), primary, 11, "finite"),
            (np.asarray((0.0, 1.0, 2.0)), {**primary, "level_shift": np.asarray((1.0,))}, 11, "length"),
            (np.asarray((0.0, 1.0, 2.0)), primary, 0, "feature_dimension"),
            (np.asarray((True, False, True)), primary, 11, "real numeric"),
        )
        for original_values, primary_values, dimension, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                FoldScoresAntigravity(original_values, primary_values, dimension)


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_calls_ten_ordered_blocked_scorers_and_covers_each_row_per_seed_antigravity(
        self,
    ) -> None:
        """Break caught: qualification uses shuffled folds, wrong fit/query slices, or fewer scorer calls."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
            _qualify_blocked_normal2_with_dependencies_antigravity,
        )

        calls: list[tuple[int, int, list[int], list[int], int, bool]] = []

        def scorer(
            train: np.ndarray,
            fit_indices: np.ndarray,
            query_indices: np.ndarray,
            *,
            seed: int,
            fold_index: int,
        ) -> object:
            calls.append(
                (
                    seed,
                    fold_index,
                    fit_indices.tolist(),
                    query_indices.tolist(),
                    train.shape[1],
                    train.flags.writeable,
                )
            )
            return fold_scores_antigravity(len(query_indices))

        result = _qualify_blocked_normal2_with_dependencies_antigravity(
            loader=normal2_rows_antigravity,
            fold_scorer=scorer,
        )

        self.assertTrue(result.passed)
        self.assertEqual(
            [(record.seed, record.fold_index) for record in result.fold_evidence],
            [(seed, fold_index) for seed in SEEDS_ANTIGRAVITY for fold_index in range(5)],
        )
        self.assertEqual(
            [(seed, fold_index) for seed, fold_index, *_rest in calls],
            [(seed, fold_index) for seed in SEEDS_ANTIGRAVITY for fold_index in range(5)],
        )
        self.assertEqual(
            [len(query) for *_prefix, query, _width, _writeable in calls[:5]],
            [283, 283, 283, 282, 282],
        )
        for seed in SEEDS_ANTIGRAVITY:
            query_union: list[int] = []
            for call_seed, _fold_index, fit, query, width, writeable in calls:
                if call_seed != seed:
                    continue
                self.assertEqual(width, 140)
                self.assertFalse(writeable)
                self.assertFalse(set(fit) & set(query))
                query_union.extend(query)
            self.assertEqual(sorted(query_union), list(range(1413)))
            self.assertEqual(len(set(query_union)), 1413)
        for record in result.fold_evidence:
            self.assertEqual(record.dataset, "ECG5000_normal_2")
            self.assertEqual(record.normal_trials, len(record.query_indices))
            self.assertEqual(record.normal_alerts, 0)
            self.assertEqual(tuple(record.per_type_auc), PRIMARY_TYPES_ANTIGRAVITY)

    def test_qualification_rejects_train_shape_dtype_finite_and_plan_order_attacks_antigravity(
        self,
    ) -> None:
        """Break caught: non-normal2 arrays or malformed blocked plans reach the scorer."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import (
            qualification_antigravity as qualification,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
            blocked_plan_antigravity,
        )

        invalid_train = (
            np.zeros((1413, 139), dtype=np.float32),
            np.zeros((1413, 140), dtype=np.float64),
            np.full((1413, 140), np.nan, dtype=np.float32),
            np.zeros((1412, 140), dtype=np.float32),
        )
        for train in invalid_train:
            with self.subTest(shape=train.shape, dtype=str(train.dtype)), self.assertRaises(ValueError):
                qualification._qualify_blocked_normal2_with_dependencies_antigravity(
                    loader=lambda train=train: train,
                    fold_scorer=lambda *_args, **_kwargs: fold_scores_antigravity(1),
                )

        valid_plan = blocked_plan_antigravity(1413)
        attacks = (
            ("missing", valid_plan[:4], "five"),
            ("reordered", valid_plan[1:] + valid_plan[:1], "order"),
            ("duplicated", valid_plan[:4] + valid_plan[:1], "order"),
        )
        for name, plan, message in attacks:
            with (
                self.subTest(name=name),
                patch.object(qualification, "blocked_plan_antigravity", return_value=plan),
                self.assertRaisesRegex(ValueError, message),
            ):
                qualification._qualify_blocked_normal2_with_dependencies_antigravity(
                    loader=normal2_rows_antigravity,
                    fold_scorer=lambda *_args, **_kwargs: fold_scores_antigravity(283),
                )

    def test_qualification_rejects_bad_scorer_results_and_dimension_drift_antigravity(
        self,
    ) -> None:
        """Break caught: wrong return type, length, nonfinite scores, or fold dimension drift is accepted."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
            _qualify_blocked_normal2_with_dependencies_antigravity,
        )

        def scorer_with_dimension(dimension: int):  # type: ignore[no-untyped-def]
            def scorer(
                _train: np.ndarray,
                _fit_indices: np.ndarray,
                query_indices: np.ndarray,
                *,
                seed: int,
                fold_index: int,
            ) -> object:
                _ = seed
                return fold_scores_antigravity(
                    len(query_indices),
                    feature_dimension=dimension + fold_index,
                )

            return scorer

        with self.assertRaisesRegex(ValueError, "feature dimensions"):
            _qualify_blocked_normal2_with_dependencies_antigravity(
                loader=normal2_rows_antigravity,
                fold_scorer=scorer_with_dimension(10),
            )

        bad_results = (
            ("type", lambda query_size: object(), "invalid"),
            ("length", lambda query_size: fold_scores_antigravity(query_size - 1), "length"),
            ("nonfinite", self._nonfinite_scores_antigravity, "finite"),
        )
        for name, factory, message in bad_results:
            def scorer(
                _train: np.ndarray,
                _fit_indices: np.ndarray,
                query_indices: np.ndarray,
                *,
                seed: int,
                fold_index: int,
                factory=factory,
            ) -> object:
                _ = (seed, fold_index)
                return factory(len(query_indices))

            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                _qualify_blocked_normal2_with_dependencies_antigravity(
                    loader=normal2_rows_antigravity,
                    fold_scorer=scorer,
                )

    def test_qualification_applies_fold_seed_and_inclusive_threshold_gates_antigravity(
        self,
    ) -> None:
        """Break caught: one fold/seed failure is ignored or exact threshold equality is treated as failing."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            FoldEvidenceAntigravity,
            wilson_upper_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
            _qualify_blocked_normal2_with_dependencies_antigravity,
        )

        def one_fold_fail(
            _train: np.ndarray,
            _fit_indices: np.ndarray,
            query_indices: np.ndarray,
            *,
            seed: int,
            fold_index: int,
        ) -> object:
            aucs = dict.fromkeys(PRIMARY_TYPES_ANTIGRAVITY, 1.0)
            if seed == 20260721 and fold_index == 3:
                aucs["level_shift"] = 0.0
            return fold_scores_antigravity(len(query_indices), aucs=aucs)

        failed = _qualify_blocked_normal2_with_dependencies_antigravity(
            loader=normal2_rows_antigravity,
            fold_scorer=one_fold_fail,
        )
        self.assertFalse(failed.seed_summaries[0].passed)
        self.assertTrue(failed.seed_summaries[1].passed)
        self.assertFalse(failed.passed)
        self.assertFalse(failed.fold_evidence[3].passed)

        fold = failed.fold_evidence[0]
        inclusive = FoldEvidenceAntigravity(
            dataset="ECG5000_normal_2",
            seed=20260721,
            fold_index=0,
            fit_indices=fold.fit_indices,
            query_indices=fold.query_indices,
            normal_trials=len(fold.query_indices),
            normal_alerts=0,
            normal_wilson_upper=wilson_upper_antigravity(0, len(fold.query_indices)),
            per_type_auc={
                "level_shift": 0.85,
                "gradual_drift": 0.90,
                "amplitude_scaling": 0.90,
                "period_change": 0.95,
            },
            pooled_auc=0.90,
            passed=True,
        )
        self.assertTrue(inclusive.passed)

    @staticmethod
    def _nonfinite_scores_antigravity(query_size: int) -> object:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
            FoldScoresAntigravity,
        )

        primary = {
            name: np.ones(query_size, dtype=float)
            for name in PRIMARY_TYPES_ANTIGRAVITY
        }
        original = np.arange(query_size, dtype=float)
        original[0] = np.nan
        return FoldScoresAntigravity(original, primary, 17)


if __name__ == "__main__":
    unittest.main()
