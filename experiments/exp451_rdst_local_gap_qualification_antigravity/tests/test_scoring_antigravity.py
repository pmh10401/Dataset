from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.contracts_antigravity import CandidateScoresAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.scoring_antigravity import (
    AggregatedScoresAntigravity,
    aggregate_seed_scores_antigravity,
    crossfit_plan_antigravity,
    reference_indices_antigravity,
    score_candidate_antigravity,
)


class ScoringAntigravityTest(unittest.TestCase):
    def rows_antigravity(self, n: int, length: int = 16) -> np.ndarray:
        return np.random.default_rng(n * 100 + length).normal(size=(n, length)).astype(np.float64)

    def test_crossfit_is_deterministic_complete_and_reference_is_capped_unique_antigravity(self) -> None:
        plan = crossfit_plan_antigravity(11, 20260721)
        self.assertEqual(len(plan), 5)
        np.testing.assert_array_equal(np.sort(np.concatenate([query for _, query in plan])), np.arange(11))
        for fit, query in plan:
            self.assertEqual(len(np.intersect1d(fit, query)), 0)
        values = np.arange(1201, dtype=int)
        selected = reference_indices_antigravity(values, 9)
        self.assertEqual(len(selected), 1000)
        self.assertEqual(len(np.unique(selected)), 1000)
        np.testing.assert_array_equal(selected, reference_indices_antigravity(values, 9))

    def test_cpu_adapter_matches_direct_exp449_localgap_on_literal_fixture_antigravity(self) -> None:
        from experiments.exp451_rdst_local_gap_qualification_antigravity.backend_gate_antigravity import build_local_gap_backend_antigravity
        reference = np.asarray([[0., 0.], [1., 2.], [2., 1.], [3., 3.]], dtype=float)
        query = np.asarray([[.5, .5], [4., 4.]], dtype=float)
        backend = build_local_gap_backend_antigravity("cpu").fit(reference)
        expected = LocalGapModelAntigravity(reference, neighbors=3).score(query)
        np.testing.assert_allclose(backend.score(query), expected, rtol=0, atol=0)
        calibration = empirical_candidates(np.asarray([-3., -2., -1., 0., 1., 2.]), backend.score(query), .5)
        self.assertEqual(calibration.indices, (1,))
        np.testing.assert_array_equal(calibration.pvalues, np.asarray([4 / 7, 3 / 7]))

    def test_score_uses_six_fresh_models_and_returns_aligned_oof_antigravity(self) -> None:
        train, test = self.rows_antigravity(10), self.rows_antigravity(3)
        import experiments.exp451_rdst_local_gap_qualification_antigravity.scoring_antigravity as module
        original = module.build_local_gap_backend_antigravity
        calls: list[object] = []
        def wrapped(name: str) -> object:
            item = original(name); calls.append(item); return item
        with patch.object(module, "build_local_gap_backend_antigravity", side_effect=wrapped):
            scored = score_candidate_antigravity(train, test, 20260721, "cpu")
        self.assertEqual(len(calls), 6)
        self.assertEqual(len({id(item) for item in calls}), 6)
        self.assertEqual(scored.train_oof.shape, (10,))
        self.assertEqual(scored.test_scores.shape, (3,))
        self.assertTrue(np.isfinite(scored.train_oof).all())

    def test_score_passes_only_fold_or_full_train_rows_to_each_official_fit_antigravity(self) -> None:
        train, test = self.rows_antigravity(10).astype(np.float32), self.rows_antigravity(2).astype(np.float32)
        import experiments.exp451_rdst_local_gap_qualification_antigravity.scoring_antigravity as module
        original = module.make_rdst_antigravity
        fitted: list[np.ndarray] = []
        transformers: list[object] = []
        class CapturingRDSTAntigravity:
            def __init__(self, value: object) -> None: self.value = value
            @property
            def retained_shapelets(self) -> int: return self.value.retained_shapelets
            def fit_transform(self, rows: np.ndarray, y: object = None) -> np.ndarray:
                self.rows = rows.copy(); fitted.append(self.rows[:, 0, :].copy())
                output = self.value.fit_transform(rows, y=y)
                transformers.append(self.value._transformer)
                return output
            def transform(self, rows: np.ndarray) -> np.ndarray: return self.value.transform(rows)
        with patch.object(module, "make_rdst_antigravity", side_effect=lambda seed: CapturingRDSTAntigravity(original(seed))):
            score_candidate_antigravity(train, test, 20260721, "cpu")
        self.assertEqual(len(fitted), 6)
        self.assertEqual(len({id(item) for item in transformers}), 6)
        self.assertEqual([item.random_state for item in transformers], [20260721] * 6)
        for (fit_indices, query_indices), values in zip(crossfit_plan_antigravity(len(train), 20260721), fitted[:5], strict=True):
            self.assertFalse(any(np.array_equal(row, test[0]) for row in values))
            self.assertTrue(all(any(np.allclose(row, source, rtol=0, atol=1e-7) for source in train) for row in values))
            self.assertTrue(all(not any(np.allclose(row, held, rtol=0, atol=1e-7) for row in values) for held in train[query_indices]))
            np.testing.assert_allclose(values, train[fit_indices], rtol=0, atol=1e-7)
        self.assertEqual(len(fitted[-1]), len(train))

    def test_score_uses_exact_fold_and_full_reference_seeds_antigravity(self) -> None:
        train, test = self.rows_antigravity(10), self.rows_antigravity(2)
        import experiments.exp451_rdst_local_gap_qualification_antigravity.scoring_antigravity as module
        original = module.reference_indices_antigravity
        seen: list[int] = []
        def capturing(indices: object, seed: int) -> np.ndarray:
            seen.append(seed)
            return original(indices, seed)
        with patch.object(module, "reference_indices_antigravity", side_effect=capturing):
            score_candidate_antigravity(train, test, 20260721, "cpu")
        self.assertEqual(seen, [20260721 + 1009 * (fold + 1) + 1024 for fold in range(5)] + [20260721 + 7919 + 1024])

    def test_chunks_never_exceed_128_and_reversed_fold_order_preserves_oof_alignment_antigravity(self) -> None:
        train, test = self.rows_antigravity(645), self.rows_antigravity(3)
        import experiments.exp451_rdst_local_gap_qualification_antigravity.scoring_antigravity as module
        transformed_lengths: list[int] = []
        class FixtureRDSTAntigravity:
            retained_shapelets = 1
            def fit_transform(self, rows: np.ndarray, y: object = None) -> np.ndarray:
                return rows[:, 0, :3].astype(float, copy=False)
            def transform(self, rows: np.ndarray) -> np.ndarray:
                transformed_lengths.append(len(rows))
                flat = rows[:, 0, :]
                return flat[:, :3].astype(float, copy=False)
        original_plan = module.crossfit_plan_antigravity
        with patch.object(module, "make_rdst_antigravity", side_effect=lambda seed: FixtureRDSTAntigravity()):
            baseline = score_candidate_antigravity(train, test, 20260721, "cpu")
            transformed_lengths.clear()
            with patch.object(module, "crossfit_plan_antigravity", side_effect=lambda n, seed: tuple(reversed(original_plan(n, seed)))):
                reversed_order = score_candidate_antigravity(train, test, 20260721, "cpu")
        self.assertTrue(transformed_lengths)
        self.assertLessEqual(max(transformed_lengths), 128)
        np.testing.assert_array_equal(baseline.train_oof, reversed_order.train_oof)

    def test_two_seed_median_and_metadata_rejections_antigravity(self) -> None:
        first = CandidateScoresAntigravity(np.asarray([1., 4.]), np.asarray([2., 8.]), 2, 2, 6, 2, 20260721)
        second = CandidateScoresAntigravity(np.asarray([3., 2.]), np.asarray([4., 6.]), 2, 2, 6, 2, 20261721)
        aggregate = aggregate_seed_scores_antigravity((first, second))
        self.assertIs(type(aggregate), AggregatedScoresAntigravity)
        np.testing.assert_array_equal(aggregate.train_oof, np.asarray([2., 3.]))
        np.testing.assert_array_equal(aggregate.test_scores, np.asarray([3., 7.]))
        with self.assertRaises(ValueError):
            aggregate_seed_scores_antigravity((second, first))
        with self.assertRaises(ValueError):
            aggregate_seed_scores_antigravity((first, first))
        for values in ((first.train_oof, first.test_scores, True, 2, 6, 2, (20260721, 20261721)), (first.train_oof, first.test_scores, 2, 2, 0, 2, (20260721, 20261721)), (first.train_oof, first.test_scores, 2, 2, 6, 2, (np.int64(20260721), 20261721))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AggregatedScoresAntigravity(*values)

    def test_one_hundred_train_rows_resolution_blocks_alpha_antigravity(self) -> None:
        result = empirical_candidates(np.arange(100, dtype=float), np.asarray([1000.]), .005)
        self.assertTrue(result.resolution_blocked)
        self.assertEqual(result.attainable_p, 1 / 101)
        self.assertEqual(result.indices, ())


if __name__ == "__main__":
    unittest.main()
