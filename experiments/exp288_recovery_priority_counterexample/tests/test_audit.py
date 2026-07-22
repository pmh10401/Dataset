import unittest
from experiments.exp288_recovery_priority_counterexample.audit import category


class CategoryTest(unittest.TestCase):
    def test_fixed_explanatory_categories(self):
        self.assertEqual(category(0.01, 0.0, 0.0, 0.0), "normal_shape_match")
        self.assertEqual(category(0.10, 0.6, 0.0, 0.0), "amplitude_or_level_only")
        self.assertEqual(category(0.20, 0.0, 0.0, 0.06), "spectral_shift")
        self.assertEqual(category(0.20, 0.0, 0.0, 0.01), "complex_difference")


if __name__ == "__main__": unittest.main()
