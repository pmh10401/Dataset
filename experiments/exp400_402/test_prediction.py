from __future__ import annotations

import unittest

from experiments.exp400_402.contracts import ALPHAS
from experiments.exp400_402.prediction import build_shadow_predictions


def exp292_fixture() -> list[dict[str, object]]:
    return [
        {
            "dataset": "A",
            "n_test": 6,
            "automatic_hard_indices": [0],
            "priority_review_indices": [5],
            "standard_review_indices": [1, 2],
            "review_priority_1_indices": [1],
            "review_priority_2_indices": [2],
            "review_priority_3_indices": [],
        },
        {
            "dataset": "B",
            "n_test": 4,
            "automatic_hard_indices": [3],
            "priority_review_indices": [],
            "standard_review_indices": [0],
            "review_priority_1_indices": [],
            "review_priority_2_indices": [],
            "review_priority_3_indices": [0],
        },
        {
            "dataset": "C",
            "n_test": 3,
            "automatic_hard_indices": [],
            "priority_review_indices": [],
            "standard_review_indices": [],
            "review_priority_1_indices": [],
            "review_priority_2_indices": [],
            "review_priority_3_indices": [],
        },
    ]


def exp397_fixture() -> list[dict[str, object]]:
    candidates = [("A", 1, 1), ("A", 2, 2), ("B", 0, 3)]
    rows: list[dict[str, object]] = []
    for alpha_offset, alpha in enumerate(ALPHAS):
        for rank, (dataset, index, tier) in enumerate(candidates, start=1):
            common = {
                "dataset": dataset,
                "candidate_index": index,
                "n_test": 6 if dataset == "A" else 4,
                "exp292_tier": tier,
                "p_values": {
                    "transition": 0.001 * rank,
                    "dwell": 0.2,
                    "recovery": 0.3,
                    "composite": 0.01,
                },
                "grammar_support_count": min(3, alpha_offset + 1),
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
            rows.append(
                {
                    **common,
                    "variant": "exp292_baseline",
                    "alpha": alpha,
                    "rank": rank,
                }
            )
            rows.append(
                {
                    **common,
                    "variant": "tier_preserving_grammar",
                    "alpha": alpha,
                    "rank": 4 - rank,
                }
            )
            rows.append(
                {
                    **common,
                    "variant": "additive_grammar_family",
                    "alpha": alpha,
                    "rank": rank,
                }
            )
            rows.append(
                {
                    **common,
                    "variant": "grammar_only_control",
                    "alpha": alpha,
                    "rank": rank,
                }
            )
    return rows


class PredictionTests(unittest.TestCase):
    def test_build_preserves_candidate_universe_and_frozen_lanes(self) -> None:
        rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(len(row["candidates"]) for row in rows), 3)
        self.assertEqual(sum(row["automatic_hard_count"] for row in rows), 2)
        self.assertEqual(sum(row["priority_review_count"] for row in rows), 1)
        self.assertEqual(rows[-1]["dataset"], "C")
        self.assertEqual(rows[-1]["candidates"], [])

    def test_alpha_changes_rank_not_candidate_universe(self) -> None:
        rows = build_shadow_predictions(exp292_fixture(), exp397_fixture())
        candidate = rows[0]["candidates"][0]
        self.assertEqual(
            set(candidate["tier_preserving_rank"]),
            {str(alpha) for alpha in ALPHAS},
        )
        self.assertEqual(candidate["exp292_rank"], 1)

    def test_rejects_missing_variant_alpha_row(self) -> None:
        ranking = exp397_fixture()
        ranking.pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_shadow_predictions(exp292_fixture(), ranking)

    def test_rejects_candidate_universe_mismatch(self) -> None:
        ranking = [
            row
            for row in exp397_fixture()
            if not (
                row["dataset"] == "A"
                and row["candidate_index"] == 2
            )
        ]
        with self.assertRaisesRegex(ValueError, "candidate universe|coverage"):
            build_shadow_predictions(exp292_fixture(), ranking)


if __name__ == "__main__":
    unittest.main()
