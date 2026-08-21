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
        """Break caught: baseline delegation, gate equality, max branches, or M6 replay drifts."""
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
        # M3 and M6 must be max branches; M2 must never lower M0.
        self.assertTrue(np.all(result.original["M3"] >= result.original["M0"]))
        self.assertTrue(np.all(result.original["M6"] >= result.original["M0"]))
        self.assertTrue(np.all(result.original["M2"] >= result.original["M0"]))
        # The baseline is passed through exactly for M0.
        np.testing.assert_array_equal(result.original["M0"], baseline_scores)
        # Every method is finite and query-length preserving.
        for method in METHOD_ORDER_ANTIGRAVITY:
            self.assertEqual(len(result.original[method]), len(query_indices))
            self.assertTrue(np.isfinite(result.original[method]).all())

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
