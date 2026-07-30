import math
import unittest

from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
)


class Exp419ContractTests(unittest.TestCase):
    def test_valid_cost_and_complete_shard_plan_are_accepted(self):
        self.assertEqual(
            DatasetCost("a", 10, "preprocessing_only").cost,
            10,
        )
        plan = ShardPlan(
            2,
            (
                ShardAssignment(0, ("a",), 10),
                ShardAssignment(1, ("b",), 20),
            ),
            ("a", "b"),
        )
        self.assertEqual(len(plan.sha256), 64)

    def test_duplicate_dataset_in_shard_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ShardPlan(
                2,
                (
                    ShardAssignment(0, ("a",), 10),
                    ShardAssignment(1, ("a",), 20),
                ),
                ("a",),
            )

    def test_missing_dataset_in_shard_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "population"):
            ShardPlan(1, (ShardAssignment(0, ("a",), 10),), ("a", "b"))

    def test_out_of_order_shard_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "indices"):
            ShardPlan(
                2,
                (
                    ShardAssignment(1, ("a",), 10),
                    ShardAssignment(0, ("b",), 20),
                ),
                ("a", "b"),
            )

    def test_negative_or_non_integer_cost_is_rejected(self):
        for cost in (-1, 1.5, True):
            with self.subTest(cost=cost):
                with self.assertRaisesRegex(ValueError, "cost"):
                    DatasetCost("a", cost, "preprocessing_only")

    def test_unknown_stratum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stratum"):
            DatasetCost("a", 10, "unknown")

    def test_non_finite_runtime_timing_is_rejected(self):
        for elapsed_seconds in (math.inf, math.nan):
            with self.subTest(elapsed_seconds=elapsed_seconds):
                with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
                    RuntimeTrial(
                        1,
                        elapsed_seconds,
                        1,
                        True,
                        True,
                        "",
                        ("a",),
                        "a" * 64,
                    )

    def test_non_lowercase_sha256_digest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "evidence_sha256"):
            RuntimeTrial(1, 1.0, 1, True, True, "", ("a",), "A" * 64)
