"""Synthetic, hand-derived tests for Exp417 horizon scoring."""

from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import HorizonScore, SeedDecision
from experiments.exp417_causal_multihorizon_shadow.scoring import (
    consensus_candidate,
    empirical_p_value,
    longest_true_run_fraction,
    score_horizon,
    seed_decision,
)


def score(horizon: int, *, p_value: float, persistence: float) -> HorizonScore:
    return HorizonScore(
        horizon=horizon,
        score=1.0,
        p_value=p_value,
        persistence=persistence,
    )


def passing_scores() -> dict[int, HorizonScore]:
    return {
        1: score(1, p_value=0.01, persistence=0.05),
        2: score(2, p_value=0.01, persistence=0.05),
        10: score(10, p_value=0.5, persistence=0.0),
    }


class PersistenceAndPValueTests(unittest.TestCase):
    def test_empirical_p_value_has_plus_one_correction(self) -> None:
        self.assertEqual(empirical_p_value([1.0, 2.0, 3.0], 2.5), 0.5)

    def test_persistence_is_longest_consecutive_fraction(self) -> None:
        mask = np.array([0, 1, 1, 0, 1], dtype=bool)
        self.assertEqual(longest_true_run_fraction(mask), 2 / 5)

    def test_empty_and_all_false_masks_have_zero_persistence(self) -> None:
        self.assertEqual(longest_true_run_fraction(np.array([], dtype=bool)), 0.0)
        self.assertEqual(longest_true_run_fraction(np.array([False, False])), 0.0)


class HorizonScoringTests(unittest.TestCase):
    def test_horizon_score_uses_hand_calculated_mad_q95_persistence_and_p_value(self) -> None:
        # The pooled normal OOF center is 2, its MAD is 1, and its robust
        # scale is 1.4826.  The query's standardized Q95 is 2 / 1.4826;
        # its two adjacent threshold exceedances give R=2/5.  Both OOF
        # series have S=(1.8 / 1.4826) + 2/5, so p=(1+0)/(1+2)=1/3.
        oof = (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        )
        result = score_horizon(
            oof,
            np.array([0.0, 3.0, 4.0, 4.0, 0.0]),
            horizon=2,
        )
        self.assertEqual(result.horizon, 2)
        self.assertAlmostEqual(result.persistence, 2 / 5)
        self.assertAlmostEqual(result.score, (2.0 / 1.4826) + (2 * (2 / 5)))
        self.assertAlmostEqual(result.p_value, 1 / 3)

    def test_score_horizon_preserves_exact_aligned_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            score_horizon(
                (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])),
                np.array([0.0, 1.0, 2.0]),
                horizon=1,
            )

    def test_score_horizon_rejects_zero_mad_and_nonfinite_or_negative_residuals(self) -> None:
        fixtures = (
            ((np.array([1.0, 1.0]),), np.array([1.0, 1.0])),
            ((np.array([0.0, np.nan]),), np.array([0.0, 1.0])),
            ((np.array([0.0, 1.0]),), np.array([0.0, np.inf])),
            ((np.array([-1.0, 1.0]),), np.array([0.0, 1.0])),
        )
        for oof, query in fixtures:
            with self.subTest(oof=oof, query=query), self.assertRaises(ValueError):
                score_horizon(oof, query, horizon=1)

    def test_score_horizon_rejects_bool_or_invalid_horizon(self) -> None:
        normal = (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            score_horizon(normal, np.array([0.0, 1.0]), horizon=True)
        with self.assertRaises(ValueError):
            score_horizon(normal, np.array([0.0, 1.0]), horizon=0)


class DecisionTests(unittest.TestCase):
    def test_two_threshold_and_persistent_horizons_make_a_seed_candidate(self) -> None:
        decision = seed_decision(passing_scores(), seed=41701)
        self.assertTrue(decision.candidate)
        self.assertIsInstance(decision.scores, MappingProxyType)
        with self.assertRaises(TypeError):
            decision.scores[3] = score(3, p_value=0.01, persistence=0.05)  # type: ignore[index]

    def test_two_small_p_values_without_persistence_do_not_make_a_candidate(self) -> None:
        scores = passing_scores()
        scores[2] = score(2, p_value=0.01, persistence=0.049999)
        self.assertFalse(seed_decision(scores, seed=41701).candidate)

    def test_seed_decision_rejects_key_record_mismatch_missing_or_extra_horizons(self) -> None:
        mismatched = passing_scores()
        mismatched[2] = score(3, p_value=0.01, persistence=0.05)
        with self.assertRaises(ValueError):
            seed_decision(mismatched, seed=41701)
        with self.assertRaises(ValueError):
            seed_decision({1: passing_scores()[1], 2: passing_scores()[2]}, seed=41701)
        extra = passing_scores()
        extra[20] = score(20, p_value=0.5, persistence=0.0)
        with self.assertRaises(ValueError):
            seed_decision(extra, seed=41701)
        wrong_frozen_triple = passing_scores()
        wrong_frozen_triple[10] = score(11, p_value=0.5, persistence=0.0)
        del wrong_frozen_triple[10]
        wrong_frozen_triple[11] = score(11, p_value=0.5, persistence=0.0)
        with self.assertRaises(ValueError):
            seed_decision(wrong_frozen_triple, seed=41701)
        with self.assertRaises(ValueError):
            seed_decision(passing_scores(), seed=99)

    def test_consensus_requires_exactly_the_frozen_seeds_and_all_true(self) -> None:
        accepted = tuple(seed_decision(passing_scores(), seed=seed) for seed in FROZEN_CONFIG.model_seeds)
        self.assertTrue(consensus_candidate(accepted))
        disagreeing_scores = passing_scores()
        disagreeing_scores[2] = score(2, p_value=0.5, persistence=0.0)
        disagreeing = seed_decision(disagreeing_scores, seed=41703)
        self.assertFalse(consensus_candidate(accepted[:2] + (disagreeing,)))
        with self.assertRaises(ValueError):
            consensus_candidate(accepted[:2])
        with self.assertRaises(ValueError):
            consensus_candidate((accepted[0], accepted[0], accepted[2]))
        wrong_seed = SeedDecision(seed=99, scores=passing_scores(), candidate=True)
        with self.assertRaises(ValueError):
            consensus_candidate((accepted[0], accepted[1], wrong_seed))


class InputValidationTests(unittest.TestCase):
    def test_p_value_rejects_empty_nonfinite_and_bool_values(self) -> None:
        for scores, query in (
            ([], 1.0),
            ([1.0, float("nan")], 1.0),
            ([1.0], float("inf")),
            ([True], 1.0),
            ([1.0], True),
        ):
            with self.subTest(scores=scores, query=query), self.assertRaises(ValueError):
                empirical_p_value(scores, query)

    def test_persistence_rejects_non_boolean_or_non_vector_masks(self) -> None:
        for mask in (np.array([0, 1]), np.array([[True, False]]), [True, False]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                longest_true_run_fraction(mask)  # type: ignore[arg-type]
