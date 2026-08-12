"""Behavior tests for Exp249-compatible LocalGap and five-fold scoring."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.exp249_250_official_mini_multirocket.local_gap import LocalGapModel
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import MethodCandidateAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
    LocalGapModelAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    CandidateScoresAntigravity,
    crossfit_plan_antigravity,
    score_candidate_antigravity,
)


def _candidate_antigravity(name: str) -> MethodCandidateAntigravity:
    return MethodCandidateAntigravity(name, name, (20260721,), (0.005,))


class LocalGapAndScoringAntigravityTests(unittest.TestCase):
    """Tests independently hand-build data; none use TEST labels."""

    def setUp(self) -> None:
        self.train = np.asarray(
            [[float((row * 3 + column * column) % 23) for column in range(16)] for row in range(6)], dtype=np.float32
        )
        self.test = np.asarray(
            [[float((row * 5 + column * 2) % 19) for column in range(16)] for row in range(3)], dtype=np.float32
        )

    def test_local_gap_matches_exp249_on_hand_derived_arrays_antigravity(self) -> None:
        """Catches any semantic change to robust scaling, neighbours, or local radius."""
        reference = np.asarray([[0.0, 0.0], [2.0, 1.0], [4.0, 1.0], [8.0, 3.0]], dtype=float)
        query = np.asarray([[1.0, 0.0], [6.0, 2.0]], dtype=float)
        expected = LocalGapModel(reference, neighbors=3).score(query)
        actual = LocalGapModelAntigravity(reference, neighbors=3).score(query)
        np.testing.assert_array_equal(actual, expected)

    def test_crossfit_plan_is_deterministic_five_fold_partition_antigravity(self) -> None:
        """Catches changed reference sampling/fold order or fewer-than-five folds."""
        folds = crossfit_plan_antigravity(11, seed=20260721)
        self.assertEqual(len(folds), 5)
        self.assertEqual(tuple(query.tolist() for _, query in folds), ([0, 2, 8], [3, 9], [1, 5], [4, 6], [7, 10]))
        self.assertEqual(tuple(query.tolist() for _, query in crossfit_plan_antigravity(11, seed=20260721)), ([0, 2, 8], [3, 9], [1, 5], [4, 6], [7, 10]))
        self.assertEqual(sorted(np.concatenate([query for _, query in folds]).tolist()), list(range(11)))
        with self.assertRaisesRegex(ValueError, "five TRAIN"):
            crossfit_plan_antigravity(4, seed=20260721)

    def test_score_contract_is_immutable_repeatable_and_finite_for_all_fixed_sources_antigravity(self) -> None:
        """Catches changed scoring seed, non-finite scores, or feature-dimension drift."""
        for name in ("minirocket_local_gap", "multirocket_local_gap", "hydra_local_gap", "hydra_multirocket_local_gap"):
            with self.subTest(name=name):
                first = score_candidate_antigravity(self.train, self.test, _candidate_antigravity(name), seed=20260721)
                second = score_candidate_antigravity(self.train, self.test, _candidate_antigravity(name), seed=20260721)
                self.assertIsInstance(first, CandidateScoresAntigravity)
                self.assertEqual(first.seed, 20260721)
                self.assertEqual(first.train_oof.shape, (6,))
                self.assertEqual(first.test_scores.shape, (3,))
                self.assertTrue(np.isfinite(first.train_oof).all())
                self.assertTrue(np.isfinite(first.test_scores).all())
                np.testing.assert_array_equal(first.train_oof, second.train_oof)
                np.testing.assert_array_equal(first.test_scores, second.test_scores)
                with self.assertRaises(AttributeError):
                    first.seed = 1

    def test_score_rejects_boolean_seed_and_mismatched_row_shape_antigravity(self) -> None:
        """Catches bool numeric fields or an inconsistent TRAIN/TEST feature shape."""
        candidate = _candidate_antigravity("minirocket_local_gap")
        with self.assertRaisesRegex(ValueError, "seed"):
            score_candidate_antigravity(self.train, self.test, candidate, seed=True)
        with self.assertRaisesRegex(ValueError, "shape"):
            score_candidate_antigravity(self.train, self.test[:, :-1], candidate, seed=20260721)


if __name__ == "__main__":
    unittest.main()
