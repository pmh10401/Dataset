from __future__ import annotations

import unittest

from experiments.exp304_reconstruction_fifth_source_common_support.routing import (
    POLICIES,
    build_five_source_routing,
    policies_for_support,
)


class FifthSourceRoutingTests(unittest.TestCase):
    def test_policy_membership_distinguishes_consensus_and_unique_reconstruction(self) -> None:
        self.assertEqual(
            policies_for_support(existing_support_count=2, reconstruction_support=True),
            [
                "verified_2of4",
                "verified_1of4",
                "five_source_3of5",
                "five_source_2of5",
                "five_source_1of5",
                "reconstruction_and_existing",
            ],
        )
        self.assertEqual(
            policies_for_support(existing_support_count=0, reconstruction_support=True),
            ["five_source_1of5", "reconstruction_only_residual"],
        )
        self.assertEqual(
            policies_for_support(existing_support_count=1, reconstruction_support=False),
            ["verified_1of4", "five_source_1of5"],
        )

    def test_dataset_routing_preserves_standard_universe_and_builds_every_policy(self) -> None:
        result = build_five_source_routing(
            dataset="demo",
            hard_indices=[1],
            standard_indices=[4, 7, 8],
            existing_support_by_index={4: {"statistical", "structural"}, 7: set(), 8: {"change_point"}},
            reconstruction_support={4, 7},
        )

        self.assertEqual(set(result["promoted_indices_by_policy"]), set(POLICIES))
        self.assertEqual(result["promoted_indices_by_policy"]["reconstruction_and_existing"], [4])
        self.assertEqual(result["promoted_indices_by_policy"]["reconstruction_only_residual"], [7])
        self.assertEqual(result["promoted_indices_by_policy"]["five_source_1of5"], [4, 7, 8])
        self.assertEqual(set(result["expanded_hard_indices_by_policy"]["five_source_1of5"]), {1, 4, 7, 8})

    def test_dataset_routing_rejects_missing_existing_support_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "existing support coverage mismatch"):
            build_five_source_routing(
                dataset="demo",
                hard_indices=[1],
                standard_indices=[4, 7],
                existing_support_by_index={4: {"statistical"}},
                reconstruction_support={7},
            )


if __name__ == "__main__":
    unittest.main()
