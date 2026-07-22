from __future__ import annotations

import unittest

from experiments.exp317_318_train_reliability_fusion.exp318_prediction import (
    apply_candidate_guards,
    qualifies,
    weighted_group_support,
)


class Exp318PredictionTests(unittest.TestCase):
    def test_representation_sources_count_as_one_group(self):
        result = weighted_group_support(
            weights={"contrastive": 0.4, "svdd": 0.8, "discord": 0.6},
            supporting_sources={"contrastive", "svdd", "discord"},
            groups={
                "contrastive": "representation",
                "svdd": "representation",
                "discord": "local_shape",
            },
        )
        self.assertEqual(result.available_group_count, 2)
        self.assertEqual(result.supporting_group_count, 2)
        self.assertAlmostEqual(result.available_weight, 1.4)
        self.assertAlmostEqual(result.supporting_weight, 1.4)

    def test_qualification_requires_three_available_and_two_supporting_groups(self):
        self.assertTrue(qualifies(0.75, 3, 2, threshold=0.67))
        self.assertFalse(qualifies(0.75, 2, 2, threshold=0.67))
        self.assertFalse(qualifies(0.75, 3, 1, threshold=0.67))
        self.assertFalse(qualifies(0.60, 3, 2, threshold=0.67))

    def test_existing_hard_and_exact_normal_are_removed(self):
        self.assertEqual(
            apply_candidate_guards({1, 2, 3, 4}, hard={2}, exact_normal={4}),
            {1, 3},
        )


if __name__ == "__main__":
    unittest.main()
