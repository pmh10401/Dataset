from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as np


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
METHOD_ORDER_ANTIGRAVITY = (
    "hydra_period_rank_max",
    "gradual_signature_local_gap",
    "hydra_period_gradual_rank_max",
)


def vector_antigravity(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


class RecordingLocalGapAntigravity:
    calls: list[tuple[np.ndarray, int, np.ndarray]] = []

    def __init__(self, reference: np.ndarray, *, neighbors: int = 3) -> None:
        self.reference = np.array(reference, dtype=np.float64, copy=True)
        self.neighbors = neighbors

    def score(self, query: np.ndarray) -> np.ndarray:
        query_copy = np.array(query, dtype=np.float64, copy=True)
        RecordingLocalGapAntigravity.calls.append((self.reference, self.neighbors, query_copy))
        return np.sum(query_copy, axis=1) - float(np.mean(self.reference))


def fake_baseline_scores_antigravity(size: int):
    original = np.asarray([0.1, 0.4, 0.2], dtype=np.float64)[:size]
    primary = {
        name: original + (index + 1) * 0.25
        for index, name in enumerate(PRIMARY_TYPES_ANTIGRAVITY)
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
            "hydra_local_gap": "exp459_hydra_local_gap",
            "period_signature_local_gap": "exp461_period_signature_local_gap",
            "hydra_period_rank_max": "exp461_branch_rank_max",
        },
    )


class ScoringAntigravityTest(unittest.TestCase):
    def test_anomaly_rank_formula_ties_endpoints_and_immutability_antigravity(self) -> None:
        """Break caught: rank uses strict inequality, wrong denominator, or writable output."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.scoring_antigravity import (
            anomaly_rank_antigravity,
        )

        reference = vector_antigravity([0.0, 2.0, 2.0, 5.0])
        query = vector_antigravity([-1.0, 0.0, 2.0, 3.0, 5.0, 8.0])
        expected = np.asarray([1.0, 2.0, 4.0, 4.0, 5.0, 5.0], dtype=np.float64) / 5.0

        actual = anomaly_rank_antigravity(reference, query)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        self.assertTrue(actual.flags.owndata)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertFalse(actual.flags.writeable)
        reference[0] = 100.0
        np.testing.assert_array_equal(actual, expected)

    def test_fold_scoring_delegates_baseline_and_combines_gradual_rank_antigravity(self) -> None:
        """Break caught: scorer stops delegating Exp461, changes LocalGap flow, or uses weighted fusion."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.scoring_antigravity import (
            _score_fold_methods_with_dependencies_antigravity,
            anomaly_rank_antigravity,
        )

        train = np.arange(7 * 140, dtype=np.float32).reshape(7, 140)
        fit_indices = np.asarray([0, 2, 4, 6], dtype=np.int64)
        query_indices = np.asarray([1, 3, 5], dtype=np.int64)
        baseline_calls: list[tuple[bool, tuple[int, ...], tuple[int, ...], int, int]] = []

        def baseline_scorer(train_arg, fit_arg, query_arg, *, seed, fold_index):
            baseline_calls.append((train_arg is train, tuple(fit_arg), tuple(query_arg), seed, fold_index))
            return fake_baseline_scores_antigravity(len(query_arg))

        def feature_fn(row):
            return np.asarray([row[0], row[1]], dtype=np.float64)

        RecordingLocalGapAntigravity.calls = []
        result = _score_fold_methods_with_dependencies_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20261721,
            fold_index=4,
            baseline_fold_scorer=baseline_scorer,
            local_gap_model=RecordingLocalGapAntigravity,
            feature_function=feature_fn,
            reference_indexer=lambda fit, *, seed, fold_index: np.asarray([fit[1], fit[3]], dtype=np.int64),
        )

        self.assertEqual(result.methods, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(baseline_calls, [(True, (0, 2, 4, 6), (1, 3, 5), 20261721, 4)])
        self.assertEqual(len(RecordingLocalGapAntigravity.calls), 5)
        reference, neighbors, original_features = RecordingLocalGapAntigravity.calls[0]
        np.testing.assert_array_equal(reference, train[[2, 6], :2].astype(np.float64))
        self.assertEqual(neighbors, 3)
        np.testing.assert_array_equal(original_features, train[[1, 3, 5], :2].astype(np.float64))

        baseline = fake_baseline_scores_antigravity(3)
        gradual_original = np.sum(train[[1, 3, 5], :2].astype(np.float64), axis=1) - float(np.mean(reference))
        np.testing.assert_array_equal(result.original["hydra_period_rank_max"], baseline.original["hydra_period_rank_max"])
        np.testing.assert_array_equal(result.original["gradual_signature_local_gap"], gradual_original)
        np.testing.assert_array_equal(
            result.original["hydra_period_gradual_rank_max"],
            np.maximum(
                baseline.original["hydra_period_rank_max"],
                gradual_original,
            ),
        )
        self.assertEqual(result.feature_dimensions["gradual_signature_local_gap"], 2)
        self.assertEqual(result.provenance["hydra_period_rank_max"], "exp461_reviewed_delegate")
        self.assertEqual(result.provenance["gradual_signature_local_gap"], "exp462_gradual_signature_local_gap")
        self.assertEqual(result.provenance["hydra_period_gradual_rank_max"], "exp462_rank_max")

    def test_fold_scoring_rejects_malformed_boundaries_antigravity(self) -> None:
        """Break caught: malformed train, indices, baseline, or score containers pass into Task 4."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.scoring_antigravity import (
            FoldMethodScoresAntigravity,
            _score_fold_methods_with_dependencies_antigravity,
        )

        good_original = {method: np.asarray([0.1, 0.2], dtype=np.float64) for method in METHOD_ORDER_ANTIGRAVITY}
        good_primary = {
            method: {
                name: np.asarray([0.3, 0.4], dtype=np.float64)
                for name in PRIMARY_TYPES_ANTIGRAVITY
            }
            for method in METHOD_ORDER_ANTIGRAVITY
        }
        with self.assertRaises(ValueError):
            FoldMethodScoresAntigravity(
                methods=tuple(reversed(METHOD_ORDER_ANTIGRAVITY)),
                original=good_original,
                primary=good_primary,
                feature_dimensions={method: 1 for method in METHOD_ORDER_ANTIGRAVITY},
                provenance={method: "x" for method in METHOD_ORDER_ANTIGRAVITY},
            )

        train = np.zeros((7, 140), dtype=np.float32)
        with self.assertRaises(ValueError):
            _score_fold_methods_with_dependencies_antigravity(
                train,
                np.asarray([0, 1, 1], dtype=np.int64),
                np.asarray([2, 3], dtype=np.int64),
                seed=20260721,
                fold_index=0,
                baseline_fold_scorer=lambda *_args, **_kwargs: fake_baseline_scores_antigravity(2),
                local_gap_model=RecordingLocalGapAntigravity,
                feature_function=lambda row: np.asarray([row[0]], dtype=np.float64),
                reference_indexer=lambda fit, *, seed, fold_index: fit,
            )

    def test_public_score_boundary_exports_no_test_or_routing_access_antigravity(self) -> None:
        """Break caught: Task 3 grows TEST, CLI, routing, or official lifecycle access."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity import (
            scoring_antigravity,
        )

        self.assertEqual(
            tuple(scoring_antigravity.__all__),
            (
                "FoldMethodScoresAntigravity",
                "anomaly_rank_antigravity",
                "score_fold_methods_antigravity",
            ),
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
