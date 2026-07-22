from __future__ import annotations

import unittest

from experiments.exp307_expanded_six_expert_moe.routing import (
    build_candidate_records,
    select_distinct_experts,
)


def audit(risk, stability, observable=True):
    return {
        "available": True, "provenance_verified": True,
        "risk_upper_95": risk, "stability": stability,
        "stability_observable": observable, "attainable_p": 0.01,
    }


class Exp307RoutingTests(unittest.TestCase):
    def test_selects_at_most_one_expert_per_method_group(self):
        audits = {
            "statistical": audit(0.01, 0.8),
            "ordinal_pattern": audit(0.02, 0.0, False),
            "structural": audit(0.03, 0.7),
            "change_point": audit(0.04, 0.6),
            "official_multirocket": audit(0.05, 0.9),
            "masked_context": audit(0.06, 0.0, False),
        }
        selected = select_distinct_experts(audits, variant="risk_first_distinct3")
        self.assertEqual(selected, ["statistical", "structural", "official_multirocket"])

    def test_candidate_ranking_preserves_universe_and_counts_independent_groups(self):
        records = build_candidate_records(
            dataset="D",
            standard_indices=[2, 1],
            support_by_index={
                1: {"statistical", "ordinal_pattern", "official_multirocket"},
                2: {"structural", "change_point"},
            },
            ordered_experts=["statistical", "structural", "official_multirocket", "masked_context"],
            selected_experts=["statistical", "structural", "official_multirocket"],
        )
        self.assertEqual({row["candidate_index"] for row in records}, {1, 2})
        by_index = {row["candidate_index"]: row for row in records}
        self.assertEqual(by_index[1]["independent_group_support_count"], 2)
        self.assertEqual(by_index[2]["independent_group_support_count"], 1)
        self.assertEqual(records[0]["candidate_index"], 1)


if __name__ == "__main__":
    unittest.main()
