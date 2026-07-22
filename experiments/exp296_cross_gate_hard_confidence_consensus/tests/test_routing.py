from __future__ import annotations

import unittest

from experiments.exp296_cross_gate_hard_confidence_consensus.routing import (
    build_consensus_records,
)


VARIANTS = (
    "risk_first_top2",
    "stability_first_top2",
    "balanced_rank_top2",
)


def gate_row(variant: str, counts: dict[int, int]) -> dict[str, object]:
    return {
        "dataset": "demo",
        "gate_variant": variant,
        "automatic_hard_indices": sorted(counts),
        "hard_confidence_records": [
            {
                "candidate_index": index,
                "selected_expert_support_count": count,
                "verified_expert_support_count": count,
            }
            for index, count in sorted(counts.items())
        ],
    }


class ConsensusRoutingTests(unittest.TestCase):
    def test_builds_four_mutually_exclusive_consensus_tiers(self) -> None:
        rows = [
            gate_row(VARIANTS[0], {1: 2, 2: 1, 3: 1, 4: 0}),
            gate_row(VARIANTS[1], {1: 2, 2: 1, 3: 0, 4: 0}),
            gate_row(VARIANTS[2], {1: 2, 2: 2, 3: 0, 4: 0}),
        ]

        records = build_consensus_records(dataset="demo", gate_rows=rows)
        tiers = {record["candidate_index"]: record["consensus_tier"] for record in records}

        self.assertEqual(tiers[1], "unanimous_high")
        self.assertEqual(tiers[2], "unanimous_supported")
        self.assertEqual(tiers[3], "partial_supported")
        self.assertEqual(tiers[4], "no_selected_support")
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["supporting_gate_count"], 3)
        self.assertEqual(records[0]["high_gate_count"], 3)

    def test_rejects_missing_gate_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "Gate coverage"):
            build_consensus_records(
                dataset="demo",
                gate_rows=[
                    gate_row(VARIANTS[0], {1: 1}),
                    gate_row(VARIANTS[1], {1: 1}),
                ],
            )

    def test_rejects_hard_index_mismatch_between_gates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hard index mismatch"):
            build_consensus_records(
                dataset="demo",
                gate_rows=[
                    gate_row(VARIANTS[0], {1: 1}),
                    gate_row(VARIANTS[1], {1: 1}),
                    gate_row(VARIANTS[2], {2: 1}),
                ],
            )


if __name__ == "__main__":
    unittest.main()
