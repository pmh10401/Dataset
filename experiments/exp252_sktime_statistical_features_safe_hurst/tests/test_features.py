from __future__ import annotations

import unittest

import numpy as np

from experiments.exp252_sktime_statistical_features_safe_hurst.features import (
    FEATURE_DIMENSIONS,
    extract_feature_blocks,
    fit_train_sanitizer,
)


class StatisticalFeatureContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = np.random.default_rng(17).normal(size=(8, 64)).astype("float32")

    def test_exact_feature_names_and_dimensions(self) -> None:
        blocks = extract_feature_blocks(self.rows, target_length=64)
        self.assertEqual(
            FEATURE_DIMENSIONS,
            {
                "catch22_22": 22,
                "summary_9": 9,
                "hurst_3": 3,
                "signature_6": 6,
                "combined_40": 40,
            },
        )
        self.assertEqual(
            {name: values.shape for name, values in blocks.items()},
            {name: (8, width) for name, width in FEATURE_DIMENSIONS.items()},
        )

    def test_feature_extraction_is_deterministic(self) -> None:
        first = extract_feature_blocks(self.rows, target_length=64)
        second = extract_feature_blocks(self.rows, target_length=64)
        for name in FEATURE_DIMENSIONS:
            np.testing.assert_allclose(first[name], second[name], rtol=0.0, atol=0.0)

    def test_test_nonfinite_values_use_frozen_train_replacement(self) -> None:
        train = {
            "block": np.asarray([[1.0, np.nan], [3.0, np.inf], [5.0, np.nan]])
        }
        test = {"block": np.asarray([[np.nan, 99.0], [100.0, np.inf]])}
        sanitizer = fit_train_sanitizer(train)
        clean_train = sanitizer.transform(train)
        clean_test = sanitizer.transform(test)
        np.testing.assert_array_equal(clean_train["block"], [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        np.testing.assert_array_equal(clean_test["block"], [[3.0, 0.0], [100.0, 0.0]])
        self.assertEqual(sanitizer.replacements["block"].tolist(), [3.0, 0.0])
        self.assertEqual(sanitizer.disabled_columns["block"].tolist(), [False, True])

    def test_hurst_failure_is_isolated_to_the_invalid_series(self) -> None:
        rows = np.vstack(
            [
                np.ones(64, dtype="float32"),
                np.random.default_rng(23).normal(size=64).astype("float32"),
            ]
        )
        blocks = extract_feature_blocks(rows, target_length=64)
        self.assertTrue(np.isnan(blocks["hurst_3"][0]).all())
        self.assertTrue(np.isfinite(blocks["hurst_3"][1]).all())
        self.assertTrue(np.isnan(blocks["combined_40"][0, 31:34]).all())

    def test_all_invalid_train_column_is_disabled_for_train_and_test(self) -> None:
        train = {"block": np.asarray([[1.0, np.nan], [3.0, np.inf]])}
        test = {"block": np.asarray([[2.0, 99.0], [4.0, np.nan]])}
        sanitizer = fit_train_sanitizer(train)
        np.testing.assert_array_equal(
            sanitizer.transform(train)["block"], [[1.0, 0.0], [3.0, 0.0]]
        )
        np.testing.assert_array_equal(
            sanitizer.transform(test)["block"], [[2.0, 0.0], [4.0, 0.0]]
        )

    def test_partially_invalid_train_column_uses_train_median(self) -> None:
        train = {"block": np.asarray([[1.0], [np.nan], [5.0]])}
        test = {"block": np.asarray([[np.nan], [99.0]])}
        sanitizer = fit_train_sanitizer(train)
        np.testing.assert_array_equal(sanitizer.transform(train)["block"], [[1.0], [3.0], [5.0]])
        np.testing.assert_array_equal(sanitizer.transform(test)["block"], [[3.0], [99.0]])
        self.assertEqual(sanitizer.disabled_columns["block"].tolist(), [False])


if __name__ == "__main__":
    unittest.main()
