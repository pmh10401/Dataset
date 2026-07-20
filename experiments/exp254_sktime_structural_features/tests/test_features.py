from __future__ import annotations

import unittest
import warnings

import numpy as np

from experiments.exp254_sktime_structural_features.features import (
    FEATURE_DIMENSIONS,
    StructuralFeatureExtractor,
    _profile_summary,
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

    def test_profile_summary_uses_only_finite_distances(self) -> None:
        values = np.asarray([[1.0, np.inf, np.nan, 3.0], [np.inf, np.nan, np.inf, np.nan]])
        result = _profile_summary(values)
        self.assertTrue(np.isfinite(result[0]).all())
        np.testing.assert_allclose(result[0, [0, 1, 7]], [1.0, 2.0, 3.0])
        self.assertTrue(np.isnan(result[1]).all())

    def test_constant_matrix_profile_does_not_emit_runtime_warning(self) -> None:
        extractor = StructuralFeatureExtractor(seed=3, working_length=32)
        panel = np.zeros((2, 1, 32), dtype=np.float32)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            result = extractor._matrix_profile(panel)
        self.assertFalse([item for item in seen if issubclass(item.category, RuntimeWarning)])
        self.assertEqual(result.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()
