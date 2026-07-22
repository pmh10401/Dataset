from __future__ import annotations

import unittest

from experiments.exp186_200.core import (
    agreement_tiers,
    apply_count_cap,
    conformal_count_cap,
    stable_dataset_guard,
    validate_disjoint_indices,
)
from experiments.exp186_200.run_experiment_186_200 import dataset_count_cap


class Exp186200CoreTest(unittest.TestCase):
    def test_three_tiers_are_disjoint_and_complete(self):
        tiers = agreement_tiers({1, 2, 3}, {2, 3, 4})
        self.assertEqual(tiers.high, {2, 3})
        self.assertEqual(tiers.standard, {1})
        self.assertEqual(tiers.exploratory, {4})
        self.assertEqual(tiers.all_indices, {1, 2, 3, 4})

    def test_count_cap_uses_train_counts_only(self):
        self.assertGreaterEqual(conformal_count_cap([0, 0, 1, 1, 2], alpha=0.01), 0)
        self.assertEqual(conformal_count_cap([10, 20, 30, 40, 50], alpha=0.5), 30)

    def test_dataset_count_cap_uses_that_datasets_fold_counts(self):
        precompute = {
            "candidate_counts_189": [0, 1, 0, 2, 1],
            "candidate_counts_199": [3, 4, 2, 5, 4],
        }
        self.assertEqual(dataset_count_cap(precompute, "189"), 2)
        self.assertEqual(dataset_count_cap(precompute, "199"), 5)

    def test_stability_requires_four_of_five_folds(self):
        self.assertTrue(stable_dataset_guard([True, True, True, True, False]))
        self.assertFalse(stable_dataset_guard([True, True, True, False, False]))

    def test_apply_count_cap_sorting(self):
        candidate_indices = {5, 2, 8, 1}
        # Evidence tuple: smaller p-value is stronger evidence
        evidence = {
            1: (0.1, 0.2),
            2: (0.01, 0.05),
            5: (0.1, 0.05),
            8: (0.001, 0.1),
        }
        # Sorted order should be:
        # 8: (0.001, 0.1) -> smallest first element
        # 2: (0.01, 0.05)
        # 5: (0.1, 0.05)  -> first element 0.1, second element 0.05 (which is < 0.2)
        # 1: (0.1, 0.2)
        # So 8, 2, 5, 1
        capped_2 = apply_count_cap(candidate_indices, 2, evidence)
        self.assertEqual(capped_2, {8, 2})

        capped_3 = apply_count_cap(candidate_indices, 3, evidence)
        self.assertEqual(capped_3, {8, 2, 5})

    def test_validate_disjoint_indices(self):
        validate_disjoint_indices({
            "a": {1, 2},
            "b": {3, 4},
        })
        with self.assertRaises(ValueError):
            validate_disjoint_indices({
                "a": {1, 2},
                "b": {2, 3},
            })


if __name__ == "__main__":
    unittest.main()
