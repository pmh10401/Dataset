import unittest

import numpy as np

from experiments.exp285_287_synthetic_shape_verifier.model import synthetic_traces


class SyntheticModelTest(unittest.TestCase):
    def test_four_deterministic_shapes_per_normal(self):
        values = np.vstack([np.linspace(0, 1, 32), np.linspace(1, 0, 32)])
        first, labels = synthetic_traces(values); second, _ = synthetic_traces(values)
        self.assertEqual(first.shape, (8, 32)); np.testing.assert_allclose(first, second)
        self.assertEqual(labels[:4], ("spike", "level_shift", "drift", "frequency"))


if __name__ == "__main__": unittest.main()
