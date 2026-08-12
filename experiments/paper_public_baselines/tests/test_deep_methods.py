from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.methods.deep_svdd import score_deep_svdd
from experiments.paper_public_baselines.methods.drocc import score_drocc
from experiments.paper_public_baselines.methods import score_published_method


def train8() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=(8, 16)).astype(np.float32)


def test3() -> np.ndarray:
    rng = np.random.default_rng(12)
    return rng.normal(size=(3, 16)).astype(np.float32)


class DeepMethodTests(unittest.TestCase):
    def test_deep_svdd_same_seed_repeats_scores(self) -> None:
        first = score_deep_svdd(train8(), test3(), seed=11, epochs=2)
        second = score_deep_svdd(train8(), test3(), seed=11, epochs=2)

        self.assertEqual(first.status, "complete")
        np.testing.assert_array_equal(first.oof_scores, second.oof_scores)
        np.testing.assert_array_equal(first.test_scores, second.test_scores)

    def test_drocc_returns_finite_scores_and_no_test_fit(self) -> None:
        result = score_drocc(train8(), test3(), seed=11, epochs=2)

        self.assertEqual(result.status, "complete")
        self.assertTrue(np.isfinite(result.oof_scores).all())
        self.assertTrue(np.isfinite(result.test_scores).all())
        self.assertEqual(result.fit_rows, len(train8()))
        self.assertEqual(result.test_scores.shape, (3,))

    def test_deep_methods_abstain_below_frozen_minimum(self) -> None:
        for scorer in (score_deep_svdd, score_drocc):
            with self.subTest(scorer=scorer.__name__):
                result = scorer(train8()[:7], test3(), seed=11, epochs=2)
                self.assertEqual(result.status, "abstain")
                self.assertEqual(result.reason, "insufficient_train_rows")

    def test_test_values_do_not_change_oof_scores(self) -> None:
        first = score_deep_svdd(train8(), test3(), seed=17, epochs=1)
        altered = score_deep_svdd(train8(), test3() + 1000.0, seed=17, epochs=1)

        np.testing.assert_array_equal(first.oof_scores, altered.oof_scores)

    def test_unified_router_dispatches_both_deep_methods(self) -> None:
        for method_id in ("deep_svdd_1d", "drocc_1d"):
            with self.subTest(method_id=method_id):
                result = score_published_method(
                    method_id, train8(), test3(), seed=19, epochs=1
                )
                self.assertEqual(result.method_id, method_id)
                self.assertEqual(result.status, "complete")


if __name__ == "__main__":
    unittest.main()
