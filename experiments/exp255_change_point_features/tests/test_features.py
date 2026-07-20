from __future__ import annotations

import unittest

import numpy as np

from experiments.exp255_change_point_features.features import (
    FEATURE_DIMENSIONS,
    ChangePointFeatureExtractor,
    RobustReference,
    detector_parameters,
)


class ChangePointFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train = np.vstack(
            [
                np.zeros(64),
                np.full(64, 0.01),
                np.full(64, -0.01),
                np.linspace(-0.01, 0.01, 64),
                np.linspace(0.01, -0.01, 64),
            ]
        )

    def test_reference_is_fitted_only_from_train(self) -> None:
        reference = RobustReference.fit(self.train)
        frozen_location = reference.location.copy()
        frozen_scale = reference.scale.copy()
        test = np.vstack([np.zeros(64), np.full(64, 1000.0)])
        first = reference.transform(test)
        second = reference.transform(test.copy())
        np.testing.assert_allclose(first, second)
        np.testing.assert_array_equal(reference.location, frozen_location)
        np.testing.assert_array_equal(reference.scale, frozen_scale)

    def test_step_change_exceeds_constant_change_strength(self) -> None:
        extractor = ChangePointFeatureExtractor(working_length=64)
        extractor.fit_transform_train(self.train)
        test = np.vstack([np.zeros(64), np.r_[np.zeros(32), np.ones(32)]])
        blocks = extractor.transform(test)
        self.assertGreater(
            float(np.max(blocks["cusum_mosum_18"][1])),
            float(np.max(blocks["cusum_mosum_18"][0])),
        )

    def test_feature_dimensions_are_fixed_and_finite(self) -> None:
        expected = {
            "cusum_mosum_18": 18,
            "pelt_l1_l2_10": 10,
            "pelt_rbf_5": 5,
            "change_point_combined_33": 33,
        }
        self.assertEqual(FEATURE_DIMENSIONS, expected)
        extractor = ChangePointFeatureExtractor(working_length=64)
        blocks = extractor.fit_transform_train(self.train)
        for name, width in expected.items():
            self.assertEqual(blocks[name].shape, (len(self.train), width))
            self.assertTrue(np.isfinite(blocks[name]).all())

    def test_detector_parameters_are_preregistered(self) -> None:
        self.assertEqual(
            detector_parameters(64),
            {"min_size": 4, "jump": 1, "penalty": 3.0 * np.log(64)},
        )
        self.assertEqual(
            detector_parameters(256),
            {"min_size": 8, "jump": 4, "penalty": 3.0 * np.log(256)},
        )

    def test_same_input_is_deterministic(self) -> None:
        first = ChangePointFeatureExtractor(working_length=64)
        second = ChangePointFeatureExtractor(working_length=64)
        first_train = first.fit_transform_train(self.train)
        second_train = second.fit_transform_train(self.train)
        for name in FEATURE_DIMENSIONS:
            np.testing.assert_allclose(first_train[name], second_train[name])


if __name__ == "__main__":
    unittest.main()
