from __future__ import annotations

import unittest

import numpy as np

from experiments.exp254_sktime_structural_features.cache_pipeline import source_config
from experiments.exp254_sktime_structural_features.scoring import score_feature_blocks


class StructuralScoringTest(unittest.TestCase):
    def test_scores_are_finite_and_deterministic(self) -> None:
        rng = np.random.default_rng(19)
        train = {"demo": rng.normal(size=(12, 5))}
        test = {"demo": rng.normal(size=(4, 5))}
        first = score_feature_blocks(train, test, seed=23)["demo"]
        second = score_feature_blocks(train, test, seed=23)["demo"]
        self.assertEqual(first.train_oof.shape, (12,))
        self.assertEqual(first.test_scores.shape, (4,))
        self.assertTrue(np.isfinite(first.train_oof).all())
        np.testing.assert_allclose(first.train_oof, second.train_oof, rtol=0, atol=0)
        np.testing.assert_allclose(first.test_scores, second.test_scores, rtol=0, atol=0)

    def test_source_config_is_label_free(self) -> None:
        config = source_config()
        self.assertFalse(config["contains_test_labels"])
        self.assertEqual(config["train_calibration"], "five_fold_oof")
        self.assertNotIn("y_test", str(config))


if __name__ == "__main__":
    unittest.main()
