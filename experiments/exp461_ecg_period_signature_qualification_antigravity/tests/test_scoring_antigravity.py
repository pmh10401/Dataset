from __future__ import annotations

import inspect
import unittest

import numpy as np


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
METHOD_ORDER_ANTIGRAVITY = (
    "hydra_local_gap",
    "period_signature_local_gap",
    "hydra_period_rank_max",
)


def vector_antigravity(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def fake_branch_scores_antigravity(base: float, size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    original = base + np.arange(size, dtype=np.float64)
    primary = {
        name: original + (index + 1) * 10.0
        for index, name in enumerate(PRIMARY_TYPES_ANTIGRAVITY)
    }
    return original, primary


class ScoringAntigravityTest(unittest.TestCase):
    def test_anomaly_rank_formula_ties_endpoints_and_immutability_antigravity(self) -> None:
        """Break caught: rank uses strict inequality, wrong denominator, or writable output."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
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

    def test_anomaly_rank_rejects_malformed_inputs_antigravity(self) -> None:
        """Break caught: invalid reference or score vectors enter rank fusion."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
            anomaly_rank_antigravity,
        )

        reference = vector_antigravity([0.0, 1.0])
        scores = vector_antigravity([0.5, 1.5])
        invalid_pairs = (
            (np.asarray([True, False]), scores),
            (np.asarray(["a", "b"], dtype=object), scores),
            (np.asarray([0.0, np.nan]), scores),
            (reference.reshape(1, 2), scores),
            (np.asarray([], dtype=np.float64), scores),
            (reference, np.asarray([True, False])),
            (reference, np.asarray([0.5, np.inf])),
            (reference, scores.reshape(1, 2)),
        )
        for bad_reference, bad_scores in invalid_pairs:
            with self.subTest(reference_dtype=bad_reference.dtype, scores_dtype=bad_scores.dtype):
                with self.assertRaises(ValueError):
                    anomaly_rank_antigravity(bad_reference, bad_scores)

    def test_fold_scoring_method_order_rank_fusion_and_dependency_flow_antigravity(self) -> None:
        """Break caught: scorer changes method order, rank fusion, seed/index flow, or accesses source data."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
            _score_fold_methods_with_dependencies_antigravity,
            anomaly_rank_antigravity,
        )

        train = np.arange(7 * 140, dtype=np.float32).reshape(7, 140)
        train.setflags(write=False)
        fit_indices = np.asarray([0, 2, 4, 6], dtype=np.int64)
        query_indices = np.asarray([1, 3, 5], dtype=np.int64)
        calls: list[tuple[str, tuple[int, ...], tuple[int, ...], int, int, bool]] = []

        def hydra_scorer(train_arg, fit_arg, query_arg, *, seed, fold_index):
            calls.append(("hydra", tuple(fit_arg), tuple(query_arg), seed, fold_index, train_arg is train))
            original, primary = fake_branch_scores_antigravity(0.0, len(query_arg))
            return original, primary, 512

        def period_scorer(train_arg, fit_arg, query_arg, *, seed, fold_index):
            calls.append(("period", tuple(fit_arg), tuple(query_arg), seed, fold_index, train_arg is train))
            original, primary = fake_branch_scores_antigravity(100.0, len(query_arg))
            return original, primary, 120

        result = _score_fold_methods_with_dependencies_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=2,
            hydra_fold_scorer=hydra_scorer,
            period_fold_scorer=period_scorer,
        )

        self.assertEqual(result.methods, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(
            calls,
            [
                ("hydra", (0, 2, 4, 6), (1, 3, 5), 20260721, 2, True),
                ("period", (0, 2, 4, 6), (1, 3, 5), 20260721, 2, True),
            ],
        )
        hydra_original, hydra_primary = fake_branch_scores_antigravity(0.0, 3)
        period_original, period_primary = fake_branch_scores_antigravity(100.0, 3)
        np.testing.assert_array_equal(result.original["hydra_local_gap"], hydra_original)
        np.testing.assert_array_equal(result.original["period_signature_local_gap"], period_original)
        np.testing.assert_array_equal(
            result.original["hydra_period_rank_max"],
            np.maximum(
                hydra_original,
                period_original,
            ),
        )
        for change_type in PRIMARY_TYPES_ANTIGRAVITY:
            with self.subTest(change_type=change_type):
                np.testing.assert_array_equal(
                result.primary["hydra_period_rank_max"][change_type],
                np.maximum(
                    hydra_primary[change_type],
                    period_primary[change_type],
                ),
            )
        self.assertEqual(result.feature_dimensions["hydra_local_gap"], 512)
        self.assertEqual(result.feature_dimensions["period_signature_local_gap"], 120)
        self.assertEqual(result.feature_dimensions["hydra_period_rank_max"], 2)
        self.assertEqual(result.provenance["hydra_local_gap"], "exp459_hydra_local_gap")
        self.assertEqual(result.provenance["period_signature_local_gap"], "exp461_period_signature_local_gap")
        for method in result.methods:
            self.assertFalse(result.original[method].flags.writeable)
            for change_type in PRIMARY_TYPES_ANTIGRAVITY:
                self.assertFalse(result.primary[method][change_type].flags.writeable)

    def test_fold_score_contract_rejects_forgery_dtype_and_length_drift_antigravity(self) -> None:
        """Break caught: forged method order, mutable config, wrong dtype, or score length drift is accepted."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
            FoldMethodScoresAntigravity,
            _score_fold_methods_with_dependencies_antigravity,
        )

        good_original = {
            method: np.asarray([0.1, 0.2], dtype=np.float64)
            for method in METHOD_ORDER_ANTIGRAVITY
        }
        good_primary = {
            method: {
                name: np.asarray([1.1, 1.2], dtype=np.float64)
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
        broken_primary = {
            **good_primary,
            "hydra_local_gap": {
                **good_primary["hydra_local_gap"],
                "period_change": np.asarray([1.0], dtype=np.float64),
            },
        }
        with self.assertRaises(ValueError):
            FoldMethodScoresAntigravity(
                methods=METHOD_ORDER_ANTIGRAVITY,
                original=good_original,
                primary=broken_primary,
                feature_dimensions={method: 1 for method in METHOD_ORDER_ANTIGRAVITY},
                provenance={method: "x" for method in METHOD_ORDER_ANTIGRAVITY},
            )

        with self.assertRaises(ValueError):
            _score_fold_methods_with_dependencies_antigravity(
                np.zeros((7, 140), dtype=np.float64),
                np.asarray([0, 2, 4, 6], dtype=np.int64),
                np.asarray([1, 3, 5], dtype=np.int64),
                seed=20260721,
                fold_index=0,
                hydra_fold_scorer=lambda *_args, **_kwargs: fake_branch_scores_antigravity(0.0, 3) + (10,),
                period_fold_scorer=lambda *_args, **_kwargs: fake_branch_scores_antigravity(1.0, 3) + (120,),
            )

    def test_public_score_boundary_exports_no_test_or_routing_access_antigravity(self) -> None:
        """Break caught: Task 3 grows TEST, CLI, routing, or official lifecycle access."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
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
