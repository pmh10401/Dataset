from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery.scattering_source import (
    MultiscaleScatteringDeviation,
    scattering_features,
)


class ScatteringSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)

    def test_scattering_features_are_fixed_width_and_finite(self) -> None:
        features = scattering_features(self.rows)

        self.assertEqual(features.shape[0], 8)
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(np.isfinite(features).all())

    def test_scattering_is_deterministic(self) -> None:
        np.testing.assert_array_equal(
            scattering_features(self.rows), scattering_features(self.rows)
        )

    def test_score_records_train_length_config_and_oof_scores(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows,
            test_values=self.rows[:3] + 0.5,
            target_length=128,
        )

        result = MultiscaleScatteringDeviation().score(bundle, 17, Path("."))

        self.assertFalse(result.abstained)
        self.assertEqual(result.train_oof.shape, (8,))
        self.assertEqual(result.test_scores.shape, (3,))
        self.assertTrue(np.isfinite(result.train_oof).all())
        self.assertTrue(np.isfinite(result.test_scores).all())
        self.assertEqual(result.provenance["scattering"]["target_length"], 128)
        self.assertEqual(result.provenance["scattering"]["J"], 5)
        self.assertEqual(
            result.provenance["scattering_feature_dimension"],
            result.provenance["feature_dimension"],
        )
        self.assertTrue(result.provenance["cross_fitted"])
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_score_abstains_below_minimum_train_without_using_labels(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows[:4],
            test_values=self.rows[:2],
            target_length=128,
        )

        result = MultiscaleScatteringDeviation().score(bundle, 17, Path("."))

        self.assertTrue(result.abstained)
        self.assertEqual(result.abstention_reason, "n_train_below_5")
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_score_rejects_test_rows_not_matching_train_target_length(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows,
            test_values=self.rows[:2, :64],
            target_length=128,
        )

        with self.assertRaisesRegex(ValueError, "target length"):
            MultiscaleScatteringDeviation().score(bundle, 17, Path("."))

    def test_score_derives_configuration_from_train_width_not_metadata(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows,
            test_values=self.rows[:2],
            target_length=64,
        )

        result = MultiscaleScatteringDeviation().score(bundle, 17, Path("."))

        self.assertEqual(result.provenance["scattering"]["target_length"], 128)


if __name__ == "__main__":
    unittest.main()
