from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as np


METHOD_ORDER_ANTIGRAVITY = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def independent_ramp_score_antigravity(values: np.ndarray) -> float:
    row = np.asarray(values, dtype=np.float64)
    center = float(np.median(row[:56]))
    scale = float(1.4826 * np.median(np.abs(row[:56] - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("independent prefix scale must be positive")
    basis = np.zeros(140, dtype=np.float64)
    basis[56:] = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis /= float(np.linalg.norm(basis))
    return float(np.dot((row - center) / scale, basis))


def independent_whole_series_ramp_antigravity(values: np.ndarray) -> float:
    row = np.asarray(values, dtype=np.float64)
    center = float(np.median(row))
    scale = float(1.4826 * np.median(np.abs(row - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("independent whole-series scale must be positive")
    basis = np.zeros(140, dtype=np.float64)
    basis[56:] = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis /= float(np.linalg.norm(basis))
    return float(np.dot((row - center) / scale, basis))


def independent_rank_antigravity(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference_values = np.asarray(reference, dtype=np.float64)
    query_values = np.asarray(query, dtype=np.float64)
    return (1.0 + np.count_nonzero(reference_values[:, None] <= query_values[None, :], axis=0)) / (
        len(reference_values) + 1.0
    )


def fake_baseline_antigravity(size: int) -> SimpleNamespace:
    original = np.asarray([0.10, 0.40, 0.20, 0.80], dtype=np.float64)[:size]
    primary = {
        name: original + (index + 1) * 0.05
        for index, name in enumerate(CHANGE_ORDER_ANTIGRAVITY)
    }
    return SimpleNamespace(
        methods=("hydra_local_gap", "period_signature_local_gap", "hydra_period_rank_max"),
        original={
            "hydra_local_gap": original + 10.0,
            "period_signature_local_gap": original + 20.0,
            "hydra_period_rank_max": original,
        },
        primary={
            "hydra_local_gap": primary,
            "period_signature_local_gap": primary,
            "hydra_period_rank_max": primary,
        },
        feature_dimensions={
            "hydra_local_gap": 512,
            "period_signature_local_gap": 120,
            "hydra_period_rank_max": 2,
        },
        provenance={
            "hydra_local_gap": "exp459",
            "period_signature_local_gap": "exp461",
            "hydra_period_rank_max": "exp461_reviewed",
        },
    )


class ScoringAntigravityTest(unittest.TestCase):
    def test_anomaly_rank_uses_frozen_tie_formula_and_read_only_output_antigravity(self) -> None:
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
            anomaly_rank_antigravity,
        )

        reference = np.asarray([0.0, 2.0, 2.0, 5.0], dtype=np.float64)
        query = np.asarray([-1.0, 0.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)
        expected = np.asarray([1.0, 2.0, 4.0, 4.0, 5.0, 5.0], dtype=np.float64) / 5.0
        actual = anomaly_rank_antigravity(reference, query)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual.flags.writeable)
        self.assertTrue(actual.flags.owndata)

    def test_fold_scoring_preserves_order_gate_semantics_and_dependency_flow_antigravity(self) -> None:
        """Break caught: baseline delegation, gate equality, or reviewed replay drift."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
            _score_fold_methods_with_dependencies_antigravity,
            anomaly_rank_antigravity,
        )

        train = (np.arange(9 * 140, dtype=np.float32).reshape(9, 140) / 100.0)
        fit_indices = np.asarray([0, 2, 4, 6], dtype=np.int64)
        query_indices = np.asarray([1, 3, 5, 7], dtype=np.int64)
        baseline_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []
        replay_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

        def baseline(train_arg, fit_arg, query_arg, *, seed, fold_index):
            del train_arg
            baseline_calls.append((tuple(fit_arg), tuple(query_arg), seed, fold_index))
            return fake_baseline_antigravity(len(query_arg))

        def replay(train_arg, fit_arg, query_arg, *, seed, fold_index):
            del train_arg
            replay_calls.append((tuple(fit_arg), tuple(query_arg), seed, fold_index))
            base = np.asarray([0.3, 0.2, 0.4, 0.1], dtype=np.float64)[: len(query_arg)]
            primary = {
                name: base + (index + 1) * 0.03
                for index, name in enumerate(CHANGE_ORDER_ANTIGRAVITY)
            }
            return SimpleNamespace(
                methods=("hydra_period_rank_max", "gradual_signature_local_gap", "hydra_period_gradual_rank_max"),
                original={
                    "hydra_period_rank_max": base + 0.0,
                    "gradual_signature_local_gap": base + 10.0,
                    "hydra_period_gradual_rank_max": base,
                },
                primary={
                    "hydra_period_rank_max": primary,
                    "gradual_signature_local_gap": primary,
                    "hydra_period_gradual_rank_max": primary,
                },
                feature_dimensions={
                    "hydra_period_rank_max": 2,
                    "gradual_signature_local_gap": 24,
                    "hydra_period_gradual_rank_max": 2,
                },
                provenance={
                    "hydra_period_rank_max": "exp461",
                    "gradual_signature_local_gap": "exp462",
                    "hydra_period_gradual_rank_max": "exp462_replay",
                },
            )

        result = _score_fold_methods_with_dependencies_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20261721,
            fold_index=4,
            baseline_fold_scorer=baseline,
            replay_fold_scorer=replay,
        )

        self.assertEqual(result.methods, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(baseline_calls, [((0, 2, 4, 6), (1, 3, 5, 7), 20261721, 4)])
        self.assertEqual(replay_calls, [((0, 2, 4, 6), (1, 3, 5, 7), 20261721, 4)])
        self.assertEqual(tuple(result.original), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.primary), METHOD_ORDER_ANTIGRAVITY)
        for method in METHOD_ORDER_ANTIGRAVITY:
            self.assertFalse(result.original[method].flags.writeable)
            for change_type in CHANGE_ORDER_ANTIGRAVITY:
                self.assertFalse(result.primary[method][change_type].flags.writeable)

        baseline_scores = fake_baseline_antigravity(4).original["hydra_period_rank_max"]
        ramp_fit = np.asarray(
            [
                # The exact values are recomputed through the public feature in the assertion below.
                0.0,
            ],
            dtype=np.float64,
        )
        del ramp_fit
        # M6 is the exact reviewed Exp462 replay control; Exp463 does not combine it again.
        np.testing.assert_array_equal(
            result.original["M6"],
            np.asarray([0.3, 0.2, 0.4, 0.1], dtype=np.float64),
        )
        replay_primary = {
            name: np.asarray([0.3, 0.2, 0.4, 0.1], dtype=np.float64) + (index + 1) * 0.03
            for index, name in enumerate(CHANGE_ORDER_ANTIGRAVITY)
        }
        for change_type in CHANGE_ORDER_ANTIGRAVITY:
            np.testing.assert_array_equal(result.primary["M6"][change_type], replay_primary[change_type])
        # The baseline is passed through exactly for M0.
        np.testing.assert_array_equal(result.original["M0"], baseline_scores)
        # Every method is finite and query-length preserving.
        for method in METHOD_ORDER_ANTIGRAVITY:
            self.assertEqual(len(result.original[method]), len(query_indices))
            self.assertTrue(np.isfinite(result.original[method]).all())

    def test_exact_m2_to_m6_vectors_use_independent_formulas_antigravity(self) -> None:
        """Break caught: any gate, fit-template, whole-series, or replay-control drift."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
            _score_fold_methods_with_dependencies_antigravity,
        )

        rng = np.random.default_rng(4633021)
        time = np.linspace(0.0, 1.0, 140, dtype=np.float64)
        fit_rows = []
        for index in range(199):
            row = (
                0.15 * np.sin(2.0 * np.pi * time)
                + 0.03 * np.cos(5.0 * np.pi * time)
                + 0.0003 * index
            )
            row[56:] += (0.2 + (index / 198.0) * 4.0) * np.linspace(0.0, 1.0, 84)
            fit_rows.append(row + rng.normal(0.0, 0.002, size=140))
        fit_matrix = np.asarray(fit_rows, dtype=np.float32)

        fit_ramp = np.asarray(
            [independent_ramp_score_antigravity(row) for row in fit_matrix],
            dtype=np.float64,
        )
        fit_order = np.argsort(fit_ramp)
        fit_whole_for_selection = np.asarray(
            [independent_whole_series_ramp_antigravity(row) for row in fit_matrix],
            dtype=np.float64,
        )
        chosen_fit_rows = np.concatenate((fit_order[[196, 197, 198]], [np.argsort(fit_whole_for_selection)[-1]]))
        train = np.vstack((fit_matrix, fit_matrix[chosen_fit_rows]))
        fit_indices = np.arange(199, dtype=np.int64)
        query_indices = np.arange(199, 203, dtype=np.int64)
        query_rows = train[query_indices]

        baseline_original = np.asarray([0.97, 0.60, 0.20, 0.90], dtype=np.float64)
        baseline_primary = {
            name: baseline_original + (index + 1) * 0.01
            for index, name in enumerate(CHANGE_ORDER_ANTIGRAVITY)
        }
        replay_original = np.asarray([0.31, 0.21, 0.41, 0.11], dtype=np.float64)
        replay_primary = {
            name: replay_original + (index + 1) * 0.017
            for index, name in enumerate(CHANGE_ORDER_ANTIGRAVITY)
        }

        def baseline(train_arg, fit_arg, query_arg, *, seed, fold_index):
            del train_arg, fit_arg, seed, fold_index
            self.assertEqual(len(query_arg), len(baseline_original))
            return SimpleNamespace(
                methods=("hydra_period_rank_max",),
                original={"hydra_period_rank_max": baseline_original},
                primary={"hydra_period_rank_max": baseline_primary},
                feature_dimensions={"hydra_period_rank_max": 2},
                provenance={"hydra_period_rank_max": "independent_m0"},
            )

        def replay(train_arg, fit_arg, query_arg, *, seed, fold_index):
            del train_arg, fit_arg, seed, fold_index
            self.assertEqual(len(query_arg), len(replay_original))
            return SimpleNamespace(
                methods=("hydra_period_gradual_rank_max",),
                original={"hydra_period_gradual_rank_max": replay_original},
                primary={"hydra_period_gradual_rank_max": replay_primary},
                feature_dimensions={"hydra_period_gradual_rank_max": 24},
                provenance={"hydra_period_gradual_rank_max": "independent_m6_replay"},
            )

        result = _score_fold_methods_with_dependencies_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20261721,
            fold_index=4,
            baseline_fold_scorer=baseline,
            replay_fold_scorer=replay,
        )

        ramp_query = np.asarray(
            [independent_ramp_score_antigravity(row) for row in query_rows],
            dtype=np.float64,
        )
        ramp_rank = independent_rank_antigravity(fit_ramp, ramp_query)
        np.testing.assert_allclose(ramp_rank[:3], np.asarray([0.99, 0.995, 1.0]), rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(
            result.original["M2"],
            np.where(ramp_rank >= 0.995, np.maximum(baseline_original, ramp_rank), baseline_original),
        )
        np.testing.assert_array_equal(result.original["M3"], np.maximum(baseline_original, ramp_rank))

        template = np.median(np.asarray(fit_matrix, dtype=np.float64), axis=0)
        fit_template_ramp = np.asarray(
            [independent_ramp_score_antigravity(row - template) for row in fit_matrix],
            dtype=np.float64,
        )
        query_template_ramp = np.asarray(
            [independent_ramp_score_antigravity(row - template) for row in query_rows],
            dtype=np.float64,
        )
        template_rank = independent_rank_antigravity(fit_template_ramp, query_template_ramp)
        self.assertTrue(np.any(template_rank < 0.995))
        self.assertTrue(np.any(template_rank >= 0.995))
        np.testing.assert_array_equal(
            result.original["M4"],
            np.where(
                template_rank >= 0.995,
                np.maximum(baseline_original, template_rank),
                baseline_original,
            ),
        )

        fit_whole_ramp = np.asarray(
            [independent_whole_series_ramp_antigravity(row) for row in fit_matrix],
            dtype=np.float64,
        )
        query_whole_ramp = np.asarray(
            [independent_whole_series_ramp_antigravity(row) for row in query_rows],
            dtype=np.float64,
        )
        whole_rank = independent_rank_antigravity(fit_whole_ramp, query_whole_ramp)
        self.assertTrue(np.any(whole_rank < 0.995))
        self.assertTrue(np.any(whole_rank >= 0.995))
        np.testing.assert_array_equal(
            result.original["M5"],
            np.where(
                whole_rank >= 0.995,
                np.maximum(baseline_original, whole_rank),
                baseline_original,
            ),
        )

        np.testing.assert_array_equal(result.original["M6"], replay_original)
        for change_type in CHANGE_ORDER_ANTIGRAVITY:
            np.testing.assert_array_equal(result.primary["M6"][change_type], replay_primary[change_type])

    def test_threshold_equality_and_malformed_boundaries_are_rejected_or_exact_antigravity(self) -> None:
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
            _gated_max_antigravity,
        )

        baseline = np.asarray([0.92, 0.99], dtype=np.float64)
        ramp = np.asarray([0.995, 0.994], dtype=np.float64)
        np.testing.assert_array_equal(
            _gated_max_antigravity(baseline, ramp),
            np.asarray([0.995, 0.99], dtype=np.float64),
        )
        with self.assertRaises(ValueError):
            _gated_max_antigravity(np.asarray([0.1]), np.asarray([np.nan]))

    def test_public_boundary_is_closed_and_signature_exact_antigravity(self) -> None:
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity import scoring_antigravity

        self.assertEqual(
            tuple(scoring_antigravity.__all__),
            ("FoldMethodScoresAntigravity", "anomaly_rank_antigravity", "score_fold_methods_antigravity"),
        )
        self.assertEqual(
            tuple(inspect.signature(scoring_antigravity.score_fold_methods_antigravity).parameters),
            ("train", "fit_indices", "query_indices", "seed", "fold_index"),
        )
        source = inspect.getsource(scoring_antigravity)
        for forbidden in ("TEST", "sqlite3", "label", "routing", "official_result_root", "cli_antigravity"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
