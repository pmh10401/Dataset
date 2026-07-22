from __future__ import annotations

import unittest

from experiments.exp295_moe_hard_confidence_tiering.routing import (
    build_hard_confidence_records,
)


class HardConfidenceRoutingTests(unittest.TestCase):
    def test_support_counts_map_to_mutually_exclusive_confidence_tiers(self) -> None:
        records = build_hard_confidence_records(
            dataset="demo",
            hard_indices=[3, 7, 9],
            source_support={
                "statistical": {3, 7},
                "structural": {3},
                "change_point": {7},
                "official_multirocket": set(),
            },
            selected_experts=["statistical", "structural"],
        )

        by_index = {record["candidate_index"]: record for record in records}
        self.assertEqual(set(by_index), {3, 7, 9})
        self.assertEqual(by_index[3]["confidence_tier"], "high")
        self.assertEqual(by_index[7]["confidence_tier"], "medium")
        self.assertEqual(by_index[9]["confidence_tier"], "needs_confirmation")
        self.assertEqual(by_index[3]["selected_expert_support_count"], 2)
        self.assertEqual(by_index[7]["verified_expert_support_count"], 2)

    def test_duplicate_hard_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate Hard index"):
            build_hard_confidence_records(
                dataset="demo",
                hard_indices=[4, 4],
                source_support={
                    "statistical": set(),
                    "structural": set(),
                    "change_point": set(),
                    "official_multirocket": set(),
                },
                selected_experts=["statistical", "structural"],
            )

    def test_exactly_two_selected_experts_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two selected experts"):
            build_hard_confidence_records(
                dataset="demo",
                hard_indices=[1],
                source_support={
                    "statistical": {1},
                    "structural": set(),
                    "change_point": set(),
                    "official_multirocket": set(),
                },
                selected_experts=["statistical"],
            )

    def test_gate_abstention_keeps_hard_alert_as_needs_confirmation(self) -> None:
        records = build_hard_confidence_records(
            dataset="small_train",
            hard_indices=[2],
            source_support={
                "statistical": {2},
                "structural": set(),
                "change_point": set(),
                "official_multirocket": set(),
            },
            selected_experts=[],
        )

        self.assertEqual(records[0]["confidence_tier"], "needs_confirmation")
        self.assertEqual(records[0]["selected_expert_support_count"], 0)
        self.assertFalse(records[0]["gate_available"])


if __name__ == "__main__":
    unittest.main()
