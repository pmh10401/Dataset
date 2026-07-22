from __future__ import annotations

import unittest

from experiments.exp291_final_operational_routing_freeze.routing import route_dataset


def _exp290() -> dict[str, object]:
    return {
        "dataset": "demo", "n_test": 10,
        "baseline_hard_indices": [1, 2], "guarded_hard_indices": [1],
        "train_raw_exact_normal_indices": [2],
    }


def _exp272() -> dict[str, object]:
    return {
        "dataset": "demo", "n_test": 10, "hard_indices": [1, 2],
        "standard_indices": [3], "priority_indices": [4], "shadow_indices": [5],
        "recovery_shadow_indices": [6], "recovery_priority_indices": [7],
    }


class FinalRoutingTest(unittest.TestCase):
    def test_preserves_guarded_hard_and_review_lanes(self) -> None:
        row = route_dataset(_exp290(), _exp272())
        self.assertEqual(row["automatic_hard_indices"], [1])
        self.assertEqual(row["automatic_normal_exact_indices"], [2])
        self.assertEqual(row["standard_review_indices"], [3])
        self.assertEqual(row["priority_review_indices"], [4])
        self.assertEqual(row["shadow_review_indices"], [5])
        self.assertEqual(row["recovery_shadow_review_indices"], [6])
        self.assertEqual(row["research_only_recovery_priority_indices"], [7])

    def test_rejects_exp253_hard_parity_mismatch(self) -> None:
        exp272 = _exp272()
        exp272["hard_indices"] = [1]
        with self.assertRaisesRegex(ValueError, "Hard parity"):
            route_dataset(_exp290(), exp272)

    def test_rejects_operational_lane_overlap(self) -> None:
        exp272 = _exp272()
        exp272["standard_indices"] = [1]
        with self.assertRaisesRegex(ValueError, "overlap"):
            route_dataset(_exp290(), exp272)

    def test_prediction_contains_no_labels_or_metrics(self) -> None:
        row = route_dataset(_exp290(), _exp272())
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        forbidden = {"y_test", "true_label", "hard_tp", "hard_fp", "f1", "precision"}
        self.assertFalse(forbidden & set(row))


if __name__ == "__main__":
    unittest.main()
