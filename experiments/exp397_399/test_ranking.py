from __future__ import annotations

import unittest

from experiments.exp397_399.ranking import rank_candidates


def candidate(
    dataset: str,
    index: int,
    *,
    tier: int,
    composite_p: float,
    transition_p: float = 1.0,
    dwell_p: float = 1.0,
    recovery_p: float = 1.0,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "candidate_index": index,
        "exp292_tier": tier,
        "p_values": {
            "composite": composite_p,
            "transition": transition_p,
            "dwell": dwell_p,
            "recovery": recovery_p,
        },
    }


class RankingTests(unittest.TestCase):
    def test_tier_preserving_never_promotes_lower_tier(self):
        rows = [
            candidate("A", 0, tier=1, composite_p=0.9),
            candidate(
                "A",
                1,
                tier=2,
                composite_p=0.001,
                transition_p=0.001,
            ),
        ]

        ranked = rank_candidates(rows, "tier_preserving_grammar", 0.005)

        self.assertEqual([row["candidate_index"] for row in ranked], [0, 1])
        self.assertEqual([row["effective_tier"] for row in ranked], [1, 2])

    def test_additive_grammar_promotes_at_most_one_tier(self):
        rows = [
            candidate(
                "A",
                1,
                tier=3,
                composite_p=0.001,
                transition_p=0.001,
                dwell_p=0.001,
                recovery_p=0.001,
            )
        ]

        ranked = rank_candidates(rows, "additive_grammar_family", 0.005)

        self.assertEqual(ranked[0]["effective_tier"], 2)
        self.assertEqual(ranked[0]["grammar_support_count"], 3)

    def test_grammar_only_ignores_exp292_tier(self):
        rows = [
            candidate("A", 0, tier=1, composite_p=0.8),
            candidate("A", 1, tier=3, composite_p=0.001),
        ]

        ranked = rank_candidates(rows, "grammar_only_control", 0.005)

        self.assertEqual([row["candidate_index"] for row in ranked], [1, 0])

    def test_ranking_is_deterministic_under_input_reordering(self):
        rows = [
            candidate("B", 2, tier=3, composite_p=0.2),
            candidate("A", 1, tier=3, composite_p=0.2),
        ]

        first = rank_candidates(rows, "exp292_baseline", 0.005)
        second = rank_candidates(list(reversed(rows)), "exp292_baseline", 0.005)

        self.assertEqual(first, second)
        self.assertEqual([row["rank"] for row in first], [1, 2])


if __name__ == "__main__":
    unittest.main()
