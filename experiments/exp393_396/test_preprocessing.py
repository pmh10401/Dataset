import unittest

import numpy as np

from experiments.exp393_396.preprocessing import (
    prepare_dataset,
    segment_descriptors,
)


class PreprocessingTests(unittest.TestCase):
    def test_target_length_uses_train_only_and_outputs_are_finite(self):
        train = [
            np.array([0.0, np.nan, 2.0]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        ]
        test = [np.array([100.0] * 20)]

        prepared = prepare_dataset(train, test)

        self.assertEqual(prepared.target_length, 4)
        self.assertTrue(np.isfinite(prepared.train).all())
        self.assertTrue(np.isfinite(prepared.test).all())
        self.assertEqual(prepared.test.shape, (1, 4))

    def test_all_missing_series_is_sanitized_to_zero(self):
        prepared = prepare_dataset(
            [np.array([np.nan, np.inf, np.nan])],
            [np.array([np.nan, np.nan])],
        )

        np.testing.assert_array_equal(prepared.train, np.zeros((1, 3)))
        np.testing.assert_array_equal(prepared.test, np.zeros((1, 3)))

    def test_descriptor_shape_and_values_are_deterministic(self):
        trace = np.linspace(0.0, 1.0, 64)

        first = segment_descriptors(trace, target_length=64)
        second = segment_descriptors(trace, target_length=64)

        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape[1], 6)
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
