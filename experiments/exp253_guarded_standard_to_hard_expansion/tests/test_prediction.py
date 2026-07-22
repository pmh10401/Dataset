from __future__ import annotations

import inspect
import unittest

from experiments.exp253_guarded_standard_to_hard_expansion.inputs import FrozenDatasetInputs
from experiments.exp253_guarded_standard_to_hard_expansion.prediction import predict_all, route_dataset


def fixture(**changes) -> FrozenDatasetInputs:
    values = {
        "dataset": "demo", "n_test": 10,
        "hard": {1}, "standard": {2, 3, 4}, "priority": {5},
        "mini1024": set(), "mini4096": set(),
        "ordinal": set(), "complexity": set(), "fourier": set(), "masked": set(),
    }
    values.update(changes)
    return FrozenDatasetInputs(**values)


class RoutingContractTest(unittest.TestCase):
    def test_broad_promotion_uses_either_mini_scale(self) -> None:
        row = fixture(mini1024={2}, mini4096={3})
        result = route_dataset(row)
        self.assertEqual(result["promoted_indices"], [2, 3])
        self.assertEqual(result["final_hard_indices"], [1, 2, 3])
        self.assertEqual(result["final_standard_indices"], [4])

    def test_outside_and_priority_are_not_promoted(self) -> None:
        row = fixture(standard={2}, priority={5}, mini1024={2, 5, 9})
        result = route_dataset(row)
        self.assertEqual(result["promoted_indices"], [2])
        self.assertEqual(result["final_priority_indices"], [5])

    def test_confidence_grades_are_disjoint(self) -> None:
        row = fixture(
            mini1024={2, 3, 4}, mini4096={2, 3}, ordinal={2}
        )
        result = route_dataset(row)
        self.assertEqual(result["grade_s_indices"], [2])
        self.assertEqual(result["grade_a_indices"], [3])
        self.assertEqual(result["grade_b_indices"], [4])

    def test_prediction_accepts_no_labels(self) -> None:
        parameters = inspect.signature(predict_all).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)
        forbidden = {"tp", "fp", "f1", "true_anomalies", "y_test"}
        self.assertFalse(forbidden & set(route_dataset(fixture())))


if __name__ == "__main__":
    unittest.main()
