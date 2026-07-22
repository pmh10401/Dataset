from __future__ import annotations

import unittest

from experiments.exp302_moe_automatic_alert_expansion_frontier.routing import (
    EXPANSION_POLICIES,
    build_dataset_expansion,
    expansion_policies_for_counts,
)


GATES = ("risk_first_top2", "stability_first_top2", "balanced_rank_top2")


def gate_row(variant: str, counts: dict[int, int]) -> dict[str, object]:
    return {
        "dataset": "demo",
        "n_test": 20,
        "gate_variant": variant,
        "automatic_hard_indices": [1],
        "standard_review_indices": [4, 7],
        "candidate_rank_records": [
            {
                "dataset": "demo",
                "candidate_index": index,
                "selected_expert_support_count": count,
                "verified_expert_support_count": count,
                "supporting_experts": [],
                "expert_rank_sum": 0,
            }
            for index, count in sorted(counts.items())
        ],
    }


class ExpansionRoutingTests(unittest.TestCase):
    def test_nested_policies_expand_from_unanimous_two_to_any_gate(self) -> None:
        self.assertEqual(
            expansion_policies_for_counts([2, 2, 2]),
            list(EXPANSION_POLICIES),
        )
        self.assertEqual(
            expansion_policies_for_counts([1, 1, 1]),
            ["all_gates_any_support", "majority_gates_any_support", "any_gate_any_support"],
        )
        self.assertEqual(
            expansion_policies_for_counts([1, 0, 1]),
            ["majority_gates_any_support", "any_gate_any_support"],
        )
        self.assertEqual(expansion_policies_for_counts([0, 0, 1]), ["any_gate_any_support"])
        self.assertEqual(expansion_policies_for_counts([0, 0, 0]), [])

    def test_dataset_merge_preserves_candidate_universe_and_builds_nested_promotions(self) -> None:
        rows = [
            gate_row(GATES[0], {4: 2, 7: 0}),
            gate_row(GATES[1], {4: 2, 7: 1}),
            gate_row(GATES[2], {4: 2, 7: 0}),
        ]

        result = build_dataset_expansion(rows)

        self.assertEqual(result["automatic_hard_indices"], [1])
        self.assertEqual(result["standard_review_indices"], [4, 7])
        self.assertEqual(result["promoted_indices_by_policy"]["all_gates_two_experts"], [4])
        self.assertEqual(result["promoted_indices_by_policy"]["any_gate_any_support"], [4, 7])
        self.assertEqual(
            set(result["expanded_hard_indices_by_policy"]["any_gate_any_support"]),
            {1, 4, 7},
        )

    def test_dataset_merge_rejects_gate_candidate_mismatch(self) -> None:
        rows = [
            gate_row(GATES[0], {4: 2, 7: 0}),
            gate_row(GATES[1], {4: 2, 7: 1}),
            gate_row(GATES[2], {4: 2}),
        ]
        rows[2]["standard_review_indices"] = [4]

        with self.assertRaisesRegex(ValueError, "Standard candidate mismatch"):
            build_dataset_expansion(rows)


if __name__ == "__main__":
    unittest.main()
