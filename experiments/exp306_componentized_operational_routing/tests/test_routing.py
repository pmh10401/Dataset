from __future__ import annotations

import unittest

from experiments.exp306_componentized_operational_routing.routing import build_componentized_row


class Exp306RoutingTests(unittest.TestCase):
    def test_partitions_guarded_benchmark_into_core_and_research_shadows(self):
        row = build_componentized_row(
            dataset="D",
            exp236={"hard_indices": [0, 1], "new_hard_indices": [1]},
            exp253={"promoted_indices": [2], "final_hard_indices": [0, 1, 2]},
            exp290={"train_raw_exact_normal_indices": [1], "guarded_hard_indices": [0, 2]},
            exp291={
                "automatic_hard_indices": [0, 2], "automatic_normal_exact_indices": [1],
                "standard_review_indices": [3], "priority_review_indices": [],
                "shadow_review_indices": [], "recovery_shadow_review_indices": [],
                "research_only_recovery_priority_indices": [], "n_test": 4,
            },
        )
        self.assertEqual(row["core_hard_indices"], [0])
        self.assertEqual(row["exp236_research_shadow_indices"], [])
        self.assertEqual(row["exp253_research_shadow_indices"], [2])
        self.assertEqual(row["benchmark_union_hard_indices"], [0, 2])
        self.assertEqual(row["automatic_normal_exact_indices"], [1])
        self.assertFalse(row["operational_policy_updated"])

    def test_rejects_union_that_does_not_match_exp291(self):
        with self.assertRaisesRegex(ValueError, "benchmark union"):
            build_componentized_row(
                dataset="D",
                exp236={"hard_indices": [0], "new_hard_indices": []},
                exp253={"promoted_indices": [], "final_hard_indices": [0]},
                exp290={"train_raw_exact_normal_indices": [], "guarded_hard_indices": [0]},
                exp291={
                    "automatic_hard_indices": [1], "automatic_normal_exact_indices": [],
                    "standard_review_indices": [], "priority_review_indices": [],
                    "shadow_review_indices": [], "recovery_shadow_review_indices": [],
                    "research_only_recovery_priority_indices": [], "n_test": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
