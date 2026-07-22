from __future__ import annotations

import unittest

import numpy as np

from experiments.exp251_sktime_statistical_features.scoring import score_feature_blocks


class StatisticalScoringContractTest(unittest.TestCase):
    def test_all_blocks_receive_oof_and_test_scores(self) -> None:
        base = np.arange(10, dtype=float)[:, None]
        train = {"a": base, "b": np.hstack([base, base**2])}
        test = {
            "a": np.asarray([[4.5], [100.0]]),
            "b": np.asarray([[4.5, 20.25], [100.0, 10000.0]]),
        }
        scores = score_feature_blocks(train, test, seed=17)
        self.assertEqual(set(scores), {"a", "b"})
        for result in scores.values():
            self.assertEqual(result.train_oof.shape, (10,))
            self.assertEqual(result.test_scores.shape, (2,))
            self.assertTrue(np.isfinite(result.train_oof).all())
            self.assertTrue(np.isfinite(result.test_scores).all())
            self.assertGreater(result.test_scores[1], result.test_scores[0])

    def test_train_and_test_block_sets_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "block sets"):
            score_feature_blocks(
                {"a": np.ones((5, 1))},
                {"b": np.ones((2, 1))},
                seed=17,
            )


if __name__ == "__main__":
    unittest.main()
