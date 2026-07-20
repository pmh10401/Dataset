from __future__ import annotations

import unittest

import numpy as np

from experiments.exp254_sktime_structural_features.features import (
    FEATURE_DIMENSIONS,
    StructuralFeatureExtractor,
    matrix_profile_windows,
)


class StructuralFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.train = rng.normal(size=(8, 96))
        self.test = rng.normal(size=(3, 96))

    def test_fixed_feature_dimensions(self) -> None:
        extractor = StructuralFeatureExtractor(seed=11, working_length=96)
        train = extractor.fit_transform_train(self.train)
        test = extractor.transform(self.test)
        self.assertEqual(
            FEATURE_DIMENSIONS,
            {
                "random_interval_224": 224,
                "matrix_profile_16": 16,
                "sax_transition_72": 72,
                "sfa_hash_128": 128,
                "structural_combined_440": 440,
            },
        )
        for name, width in FEATURE_DIMENSIONS.items():
            self.assertEqual(train[name].shape, (8, width))
            self.assertEqual(test[name].shape, (3, width))
            self.assertTrue(np.isfinite(train[name]).all())
            self.assertTrue(np.isfinite(test[name]).all())

    def test_same_seed_is_deterministic(self) -> None:
        first = StructuralFeatureExtractor(seed=13, working_length=96)
        second = StructuralFeatureExtractor(seed=13, working_length=96)
        a = first.fit_transform_train(self.train)
        b = second.fit_transform_train(self.train)
        for name in FEATURE_DIMENSIONS:
            np.testing.assert_allclose(a[name], b[name], rtol=0, atol=0)

    def test_matrix_profile_windows_are_valid(self) -> None:
        windows = matrix_profile_windows(32)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(3 <= value < 32 for value in windows))
        self.assertEqual(windows, matrix_profile_windows(32))


if __name__ == "__main__":
    unittest.main()
