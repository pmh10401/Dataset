"""Behavior-first tests for sequential versus 11-worker Policy A parity."""
from __future__ import annotations

import unittest

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import WorkUnitAntigravity


class WorkerParityAntigravityTests(unittest.TestCase):
    def test_policy_a_real_shape_is_exactly_nineteen_of_the_57_ordered_units(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import ordered_work_units_antigravity

        all_units = ordered_work_units_antigravity(tuple(f"dataset_{index:02d}" for index in range(19)))
        policy_a = tuple(unit for unit in all_units if unit.policy == "linear_edge_nearest_control")
        self.assertEqual(len(all_units), 57)
        self.assertEqual(len(policy_a), 19)
        self.assertEqual(tuple(unit.sequence for unit in policy_a), tuple(range(19)))

    def test_parity_requires_exact_semantic_payload_and_records_speed_ratio(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import (
            WorkerParityEvidenceAntigravity,
            worker_parity_antigravity,
        )

        units = tuple(
            WorkUnitAntigravity(index, "linear_edge_nearest_control", f"dataset_{index:02d}")
            for index in range(3)
        )
        payloads = {unit.sequence: {"sequence": unit.sequence, "policy": unit.policy, "dataset": unit.dataset, "score": 0.5} for unit in units}
        result = worker_parity_antigravity(
            units,
            sequential_runner=lambda: payloads,
            parallel_runner=lambda: payloads,
            sequential_elapsed=3.0,
            parallel_elapsed=1.5,
        )
        self.assertIsInstance(result, WorkerParityEvidenceAntigravity)
        self.assertTrue(result.semantic_equal)
        self.assertEqual(result.sequential_payload_sha256, result.parallel_payload_sha256)
        self.assertEqual(result.speed_ratio, 2.0)
        self.assertEqual(result.unit_count, 3)

    def test_parity_rejects_score_or_route_payload_mismatch(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import worker_parity_antigravity

        units = (WorkUnitAntigravity(0, "linear_edge_nearest_control", "dataset_00"),)
        with self.assertRaisesRegex(ValueError, "parity"):
            worker_parity_antigravity(
                units,
                sequential_runner=lambda: {0: {"score": 1.0, "candidates": [0], "route": {"a": 1}}},
                parallel_runner=lambda: {0: {"score": 1.0, "candidates": [1], "route": {"a": 1}}},
                sequential_elapsed=1.0,
                parallel_elapsed=1.0,
            )


if __name__ == "__main__":
    unittest.main()
