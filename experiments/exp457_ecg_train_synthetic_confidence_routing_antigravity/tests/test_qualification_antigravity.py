from __future__ import annotations

from collections.abc import Mapping
import math
import unittest

import numpy as np


class QualificationMathAntigravityTest(unittest.TestCase):
    def test_plus_one_upper_tail_pvalues_and_exact_alpha_are_hand_derived_antigravity(self) -> None:
        """Break caught: p-values drop plus-one, reverse the tail, or exclude alpha equality."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            empirical_pvalues_antigravity,
        )

        actual = empirical_pvalues_antigravity(
            np.asarray((1.0, 2.0, 2.0, 4.0)),
            np.asarray((0.0, 2.0, 3.0, 4.0, 5.0)),
        )
        np.testing.assert_array_equal(actual, np.asarray((1.0, 0.8, 0.4, 0.4, 0.2)))
        exact = empirical_pvalues_antigravity(
            np.arange(199, dtype=float),
            np.asarray((199.0,)),
        )
        self.assertEqual(float(exact[0]), 0.005)
        self.assertTrue(bool(exact[0] <= 0.005))

    def test_score_vectors_require_real_non_boolean_numeric_dtypes_antigravity(self) -> None:
        """Break caught: bool, complex, or object scores are silently coerced into Gate 1 floats."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _FoldScoresAntigravity,
            empirical_pvalues_antigravity,
        )

        integer_pvalues = empirical_pvalues_antigravity(
            np.asarray((1, 2, 3), dtype=np.int64),
            np.asarray((2, 4), dtype=np.int32),
        )
        np.testing.assert_array_equal(integer_pvalues, np.asarray((0.75, 0.25)))
        valid_primary = {
            name: np.asarray((1.0, 2.0), dtype=np.float32)
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        }
        integer_fold = _FoldScoresAntigravity(
            np.asarray((1, 2), dtype=np.int64),
            valid_primary,
            2,
        )
        np.testing.assert_array_equal(integer_fold.original, np.asarray((1.0, 2.0)))

        invalid = (
            np.asarray((True, False), dtype=np.bool_),
            np.asarray((1.0 + 2.0j, 3.0 + 0.0j), dtype=np.complex128),
            np.asarray((1.0, 2.0), dtype=object),
        )
        finite_real = np.asarray((1.0, 2.0), dtype=np.float64)
        for values in invalid:
            with self.subTest(dtype=str(values.dtype), boundary="reference"), self.assertRaisesRegex(
                ValueError, "real numeric dtype"
            ):
                empirical_pvalues_antigravity(values, finite_real)
            with self.subTest(dtype=str(values.dtype), boundary="query"), self.assertRaisesRegex(
                ValueError, "real numeric dtype"
            ):
                empirical_pvalues_antigravity(finite_real, values)
            with self.subTest(dtype=str(values.dtype), boundary="fold-original"), self.assertRaisesRegex(
                ValueError, "real numeric dtype"
            ):
                _FoldScoresAntigravity(values, valid_primary, 2)
            invalid_primary = dict(valid_primary)
            invalid_primary["level_shift"] = values
            with self.subTest(dtype=str(values.dtype), boundary="fold-primary"), self.assertRaisesRegex(
                ValueError, "real numeric dtype"
            ):
                _FoldScoresAntigravity(finite_real, invalid_primary, 2)

    def test_wilson_95_upper_values_and_validation_are_hand_derived_antigravity(self) -> None:
        """Break caught: Gate 1 uses a Wald bound, wrong z, or permissive integer coercion."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            wilson_upper_antigravity,
        )

        self.assertAlmostEqual(wilson_upper_antigravity(0, 100), 0.03699349820698568, places=15)
        self.assertAlmostEqual(wilson_upper_antigravity(5, 100), 0.11175046923191913, places=15)
        self.assertEqual(wilson_upper_antigravity(100, 100), 1.0)
        for successes, trials in ((True, 100), (0, True), (-1, 100), (101, 100), (0, 0)):
            with self.subTest(successes=successes, trials=trials), self.assertRaises(ValueError):
                wilson_upper_antigravity(successes, trials)


class FoldScoringAntigravityTest(unittest.TestCase):
    def test_one_fold_model_scores_original_and_all_four_perturbations_antigravity(self) -> None:
        """Break caught: synthetic queries are scored by refitted or type-specific fold models."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _score_fold_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.perturbations_antigravity import (
            build_primary_perturbations_antigravity,
        )

        train = np.asarray(
            [[float(row) + math.sin(column / 3.0) + column / 10.0 for column in range(16)] for row in range(1105)],
            dtype=np.float32,
        )
        fit_indices = np.arange(1100, dtype=int)
        query_indices = np.arange(1100, 1105, dtype=int)
        factory_calls: list[tuple[str, int]] = []
        model_calls: list[tuple[np.ndarray, int]] = []

        class RepresentationAntigravity:
            def __init__(self) -> None:
                self.fit_rows: list[np.ndarray] = []
                self.transform_rows: list[np.ndarray] = []

            def fit_transform(self, rows: np.ndarray) -> np.ndarray:
                self.fit_rows.append(np.array(rows, copy=True))
                return np.asarray(rows[:, :2], dtype=float)

            def transform(self, rows: np.ndarray) -> np.ndarray:
                self.transform_rows.append(np.array(rows, copy=True))
                return np.asarray(rows[:, :2], dtype=float)

        class ModelAntigravity:
            def __init__(self) -> None:
                self.score_calls: list[np.ndarray] = []

            def score(self, features: np.ndarray) -> np.ndarray:
                self.score_calls.append(np.array(features, copy=True))
                return np.arange(len(features), dtype=float) + len(self.score_calls) * 10.0

        representation = RepresentationAntigravity()
        model = ModelAntigravity()

        def representation_factory(name: str, seed: int) -> RepresentationAntigravity:
            factory_calls.append((name, seed))
            return representation

        def model_factory(reference: np.ndarray, *, neighbors: int) -> ModelAntigravity:
            model_calls.append((np.array(reference, copy=True), neighbors))
            return model

        result = _score_fold_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            representation_factory=representation_factory,
            model_factory=model_factory,
        )

        self.assertEqual(factory_calls, [("hydra_local_gap", 20260721)])
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(model_calls[0][1], 3)
        expected_reference = np.sort(
            np.random.default_rng(20260721 + 1009 + 8).choice(
                fit_indices,
                size=1000,
                replace=False,
            )
        )
        np.testing.assert_array_equal(representation.fit_rows[0], train[expected_reference])
        self.assertEqual(len(representation.transform_rows), 5)
        np.testing.assert_array_equal(representation.transform_rows[0], train[query_indices])
        self.assertEqual(len(model.score_calls), 5)
        np.testing.assert_array_equal(model.score_calls[0], train[query_indices, :2])
        expected_rows = tuple(
            build_primary_perturbations_antigravity(row) for row in train[query_indices]
        )
        for call_index, name in enumerate(FROZEN_CONFIG_ANTIGRAVITY.primary_types, start=1):
            expected = np.stack([row[name] for row in expected_rows])
            np.testing.assert_array_equal(representation.transform_rows[call_index], expected)
            np.testing.assert_array_equal(model.score_calls[call_index], expected[:, :2])
        self.assertEqual(tuple(result.primary), FROZEN_CONFIG_ANTIGRAVITY.primary_types)
        self.assertEqual(result.original.shape, (5,))
        self.assertTrue(all(values.shape == (5,) for values in result.primary.values()))


class QualificationGateAntigravityTest(unittest.TestCase):
    @staticmethod
    def _rows_antigravity(size: int) -> np.ndarray:
        rows = np.column_stack(
            (
                np.arange(size, dtype=np.float32),
                np.arange(size, dtype=np.float32) % 7,
                np.ones((size, 7), dtype=np.float32),
            )
        )
        rows.setflags(write=False)
        return rows

    def _scorer_antigravity(self, mode: str):  # type: ignore[no-untyped-def]
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _FoldScoresAntigravity,
        )

        def score(
            train: np.ndarray,
            _fit_indices: np.ndarray,
            query_indices: np.ndarray,
            *,
            seed: int,
            fold_index: int,
        ) -> object:
            original = query_indices.astype(float)
            high = float(len(train) + 100)
            primary: dict[str, np.ndarray] = {}
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types:
                values = np.full(len(query_indices), high, dtype=float)
                if mode == "second_seed_per_type_fail" and seed == 20261721 and name == "level_shift":
                    values = np.where(query_indices < 99, high, -100.0)
                elif mode == "pooled_fail":
                    values = np.where(query_indices < 120, high, -100.0)
                primary[name] = values
            if mode == "nonfinite" and seed == 20260721 and fold_index == 0:
                original[0] = np.nan
            return _FoldScoresAntigravity(original, primary, 2)

        return score

    def test_exact_threshold_passes_only_when_both_ordered_seeds_pass_antigravity(self) -> None:
        """Break caught: one seed can qualify a dataset or p == alpha is rejected."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _qualify_ecg_dataset_antigravity,
        )

        train = self._rows_antigravity(199)
        passed = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_1",
            train,
            fold_scorer=self._scorer_antigravity("pass"),
        )
        self.assertTrue(passed.passed)
        self.assertEqual(tuple(item.seed for item in passed.seed_evidence), (20260721, 20261721))
        self.assertTrue(all(item.passed for item in passed.seed_evidence))
        self.assertTrue(all(item.pooled_sensitivity == 1.0 for item in passed.seed_evidence))

        failed = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_1",
            train,
            fold_scorer=self._scorer_antigravity("second_seed_per_type_fail"),
        )
        self.assertTrue(failed.seed_evidence[0].passed)
        self.assertFalse(failed.seed_evidence[1].passed)
        self.assertGreater(failed.seed_evidence[1].pooled_sensitivity, 0.70)
        self.assertLess(failed.seed_evidence[1].per_type_sensitivity["level_shift"], 0.50)
        self.assertFalse(failed.passed)

    def test_pooled_per_type_nonfinite_and_small_set_fail_without_relaxation_antigravity(self) -> None:
        """Break caught: Gate 1 drops one criterion or relaxes alpha for a small TRAIN set."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _qualify_ecg_dataset_antigravity,
        )

        pooled = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_1",
            self._rows_antigravity(199),
            fold_scorer=self._scorer_antigravity("pooled_fail"),
        )
        self.assertTrue(all(min(item.per_type_sensitivity.values()) >= 0.50 for item in pooled.seed_evidence))
        self.assertTrue(all(item.pooled_sensitivity < 0.70 for item in pooled.seed_evidence))
        self.assertFalse(pooled.passed)

        nonfinite = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_1",
            self._rows_antigravity(199),
            fold_scorer=self._scorer_antigravity("nonfinite"),
        )
        self.assertFalse(nonfinite.seed_evidence[0].passed)
        self.assertFalse(nonfinite.passed)

        small = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_5",
            self._rows_antigravity(19),
            fold_scorer=self._scorer_antigravity("pass"),
        )
        self.assertTrue(all(item.normal_alerts == 0 for item in small.seed_evidence))
        self.assertTrue(all(item.normal_wilson_upper > 0.05 for item in small.seed_evidence))
        self.assertTrue(all(item.pooled_sensitivity == 0.0 for item in small.seed_evidence))
        self.assertFalse(small.passed)

    def test_all_rows_are_covered_once_and_complete_oof_is_the_only_reference_antigravity(self) -> None:
        """Break caught: decisions use per-fold/synthetic references or omit OOF rows."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _FoldScoresAntigravity,
            _qualify_ecg_dataset_antigravity,
        )

        covered: dict[int, list[int]] = {seed: [] for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds}

        def fold_scorer(
            _train: np.ndarray,
            _fit_indices: np.ndarray,
            query_indices: np.ndarray,
            *,
            seed: int,
            fold_index: int,
        ) -> object:
            covered[seed].extend(query_indices.tolist())
            original = fold_index * 1000.0 + np.arange(len(query_indices), dtype=float)
            just_above_fold = np.full(len(query_indices), float(original[-1] + 0.5))
            primary: Mapping[str, np.ndarray] = {
                name: np.array(just_above_fold, copy=True)
                for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
            }
            return _FoldScoresAntigravity(original, primary, 2)

        result = _qualify_ecg_dataset_antigravity(
            "ECG5000_normal_1",
            self._rows_antigravity(999),
            fold_scorer=fold_scorer,
        )
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            self.assertEqual(sorted(covered[seed]), list(range(999)))
            self.assertEqual(len(set(covered[seed])), 999)
        for evidence in result.seed_evidence:
            self.assertEqual(evidence.pooled_sensitivity, 199 / 999)
            self.assertEqual(tuple(evidence.per_type_sensitivity), FROZEN_CONFIG_ANTIGRAVITY.primary_types)
            self.assertTrue(all(value == 199 / 999 for value in evidence.per_type_sensitivity.values()))
            self.assertFalse(evidence.passed)
        self.assertFalse(result.passed)

    def test_all_dataset_helper_preserves_frozen_order_without_cross_dataset_state_antigravity(self) -> None:
        """Break caught: all-dataset qualification skips, reorders, or reuses another TRAIN matrix."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
            _qualify_all_ecg_antigravity,
        )

        loaded: list[str] = []
        qualified: list[tuple[str, int]] = []

        def loader(dataset: str) -> np.ndarray:
            loaded.append(dataset)
            return np.full((5, 9), len(loaded), dtype=np.float32)

        def qualifier(dataset: str, train: np.ndarray) -> tuple[str, int]:
            token = int(train[0, 0])
            qualified.append((dataset, token))
            return dataset, token

        actual = _qualify_all_ecg_antigravity(loader=loader, qualifier=qualifier)
        self.assertEqual(tuple(loaded), FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets)
        self.assertEqual(
            tuple(dataset for dataset, _ in qualified),
            FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets,
        )
        self.assertEqual(actual, tuple(zip(FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets, range(1, 6), strict=True)))


if __name__ == "__main__":
    unittest.main()
