from __future__ import annotations

import unittest

import numpy as np

from experiments.exp255_change_point_features.cache_pipeline import source_config
from experiments.exp255_change_point_features.scoring import (
    crossfit_plan,
    score_feature_blocks,
)


class ChangePointScoringTest(unittest.TestCase):
    def test_scores_are_finite_and_deterministic(self) -> None:
        rng = np.random.default_rng(255)
        train = {"demo": rng.normal(size=(12, 5))}
        test = {"demo": rng.normal(size=(4, 5))}
        first = score_feature_blocks(train, test, seed=20260724)["demo"]
        second = score_feature_blocks(train, test, seed=20260724)["demo"]
        self.assertEqual(first.train_oof.shape, (12,))
        self.assertEqual(first.test_scores.shape, (4,))
        self.assertTrue(np.isfinite(first.train_oof).all())
        self.assertTrue(np.isfinite(first.test_scores).all())
        np.testing.assert_array_equal(first.train_oof, second.train_oof)
        np.testing.assert_array_equal(first.test_scores, second.test_scores)

    def test_crossfit_plan_is_deterministic_and_excludes_query(self) -> None:
        first = crossfit_plan(12, seed=20260724)
        second = crossfit_plan(12, seed=20260724)
        self.assertEqual(len(first), 5)
        for (fit, query), (fit_again, query_again) in zip(first, second):
            np.testing.assert_array_equal(fit, fit_again)
            np.testing.assert_array_equal(query, query_again)
            self.assertFalse(set(fit) & set(query))

    def test_source_config_is_train_only_and_preregistered(self) -> None:
        config = source_config()
        self.assertFalse(config["contains_test_labels"])
        self.assertFalse(config["uses_family_name"])
        self.assertFalse(config["uses_dataset_name_policy"])
        self.assertFalse(config["uses_test_length_budget"])
        self.assertEqual(config["train_calibration"], "five_fold_oof")
        self.assertEqual(config["pelt"]["penalty"], "3.0 * log(working_length)")
        self.assertNotIn("y_test", str(config))


if __name__ == "__main__":
    unittest.main()
