from __future__ import annotations

import unittest

from experiments.exp308_frozen_six_expert_review_tiers.routing import partition_review_records


class Exp308RoutingTests(unittest.TestCase):
    def test_partitions_ranked_candidates_without_changing_the_universe(self):
        rows = [
            {
                "dataset": "D",
                "candidate_index": index,
                "selected_expert_support_count": support,
                "independent_group_support_count": groups,
                "verified_expert_support_count": support,
                "expert_rank_sum": rank_sum,
            }
            for index, support, groups, rank_sum in (
                (3, 1, 1, 5),
                (1, 3, 3, 1),
                (2, 2, 2, 2),
                (4, 0, 0, 9),
            )
        ]

        partitioned = partition_review_records(rows, strong_count=1, medium_count=1)

        self.assertEqual([row["candidate_index"] for row in partitioned], [1, 2, 3, 4])
        self.assertEqual(
            [row["review_tier"] for row in partitioned],
            ["strong_review", "medium_review", "standard_review", "standard_review"],
        )
        self.assertEqual({row["candidate_index"] for row in partitioned}, {1, 2, 3, 4})

    def test_rejects_duplicate_dataset_index_candidates(self):
        row = {
            "dataset": "D", "candidate_index": 1,
            "selected_expert_support_count": 1,
            "independent_group_support_count": 1,
            "verified_expert_support_count": 1,
            "expert_rank_sum": 1,
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            partition_review_records([row, dict(row)], strong_count=1, medium_count=0)


if __name__ == "__main__":
    unittest.main()
