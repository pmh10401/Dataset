from __future__ import annotations

import unittest

import numpy as np

from experiments.exp175_176 import diagnostic_core as core


class DiagnosticCoreTests(unittest.TestCase):
    def test_fixed_grids_are_preregistered(self):
        self.assertEqual(core.ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(core.RANK_DEPTHS, (1, 2, 3, 5, 8, 10))

    def test_descending_ranks_are_stable_for_ties(self):
        ranks = core.descending_ranks(np.array([2.0, 5.0, 5.0, 1.0]))
        np.testing.assert_array_equal(ranks, np.array([3, 1, 2, 4]))

    def test_upper_tail_pvalue_has_finite_sample_correction(self):
        train = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(core.upper_tail_pvalue(train, 4.0), 0.25)
        self.assertAlmostEqual(core.upper_tail_pvalue(train, 2.0), 0.75)

    def test_miss_categories_are_mutually_exclusive(self):
        self.assertEqual(
            core.classify_miss(True, False, {0.05: 4}),
            "already_in_standard_review",
        )
        self.assertEqual(
            core.classify_miss(False, True, {0.05: 4}),
            "already_in_priority_review",
        )
        self.assertEqual(
            core.classify_miss(False, False, {0.05: 2}),
            "score_visible_policy_miss",
        )
        self.assertEqual(
            core.classify_miss(False, False, {0.05: 1}),
            "single_source_only",
        )
        self.assertEqual(
            core.classify_miss(False, False, {0.05: 0}),
            "representation_blind",
        )

    def test_source_diagnostic_reports_rank_and_support(self):
        result = core.source_diagnostic(
            train_scores=np.array([1.0, 2.0, 3.0]),
            test_scores=np.array([4.0, 2.0, 0.0]),
            index=0,
        )
        self.assertEqual(result["rank"], 1)
        self.assertAlmostEqual(result["rank_percentile"], 1 / 3)
        self.assertAlmostEqual(result["pvalue"], 0.25)
        self.assertFalse(any(result[f"support_alpha_{alpha:g}"] for alpha in core.ALPHAS))


if __name__ == "__main__":
    unittest.main()
