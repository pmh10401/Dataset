import unittest

from experiments.exp322_324_exp291_reliability_weighting.pipeline import compare_selected_indices


class ReplayTests(unittest.TestCase):
    def test_index_comparison_ignores_whitespace(self):
        result = compare_selected_indices("1  3 5", "1 3 5")
        self.assertFalse(result["changed"])
        self.assertEqual(result["baseline_indices"], [1, 3, 5])

    def test_index_change_is_explicit(self):
        result = compare_selected_indices("1 3", "1 4")
        self.assertTrue(result["changed"])
        self.assertEqual(result["removed_indices"], [3])
        self.assertEqual(result["added_indices"], [4])


if __name__ == "__main__":
    unittest.main()
