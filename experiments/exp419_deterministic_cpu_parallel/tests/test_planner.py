import unittest
from dataclasses import fields

from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    TrainLengthGroupMeta,
)
from experiments.exp419_deterministic_cpu_parallel.planner import estimate_dataset_cost


class PlannerTests(unittest.TestCase):
    def test_cost_proxy_uses_frozen_group_weights(self):
        groups = (
            TrainLengthGroupMeta("a", 10, 49),
            TrainLengthGroupMeta("a", 20, 50),
            TrainLengthGroupMeta("a", 30, 100),
        )

        result = estimate_dataset_cost("a", groups)

        self.assertEqual(result.cost, 49 * 10 + 50 * 20 * 30 + 100 * 30 * 81)
        self.assertEqual(result.stratum, "candidate_eligible")

    def test_cost_stratum_uses_largest_training_group_count(self):
        preprocessing = estimate_dataset_cost(
            "pre",
            (
                TrainLengthGroupMeta("pre", 10, 1),
                TrainLengthGroupMeta("pre", 20, 49),
            ),
        )
        diagnostic = estimate_dataset_cost(
            "diag",
            (
                TrainLengthGroupMeta("diag", 10, 50),
                TrainLengthGroupMeta("diag", 20, 99),
            ),
        )

        self.assertEqual(preprocessing.stratum, "preprocessing_only")
        self.assertEqual(diagnostic.stratum, "diagnostic_oof")

    def test_cost_sums_duplicate_lengths(self):
        result = estimate_dataset_cost(
            "a",
            (
                TrainLengthGroupMeta("a", 10, 1),
                TrainLengthGroupMeta("a", 10, 49),
            ),
        )

        self.assertEqual(result.cost, 500)
        self.assertEqual(result.stratum, "preprocessing_only")

    def test_cost_rejects_groups_for_another_dataset(self):
        with self.assertRaisesRegex(ValueError, "dataset"):
            estimate_dataset_cost("a", (TrainLengthGroupMeta("b", 10, 1),))

    def test_zero_length_and_count_are_rejected_at_train_metadata_boundary(self):
        for length, count in ((0, 1), (1, 0)):
            with self.subTest(length=length, count=count):
                with self.assertRaisesRegex(ValueError, "length|series_count"):
                    TrainLengthGroupMeta("a", length, count)

    def test_cost_rejects_signed_64_bit_overflow(self):
        with self.assertRaisesRegex(OverflowError, "64-bit"):
            estimate_dataset_cost(
                "a",
                (TrainLengthGroupMeta("a", 9_223_372_036_854_775_807, 2),),
            )

    def test_benchmark_selection_uses_frozen_positions_and_global_name_order(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import select_benchmark_datasets

        costs = (
            *(DatasetCost(f"pre{index}", cost, "preprocessing_only") for index, cost in enumerate((70, 10, 60, 20, 50, 30, 40))),
            *(DatasetCost(f"diag{index}", cost, "diagnostic_oof") for index, cost in enumerate((700, 100, 600, 200, 500, 300, 400))),
            *(DatasetCost(f"cand{index}", cost, "candidate_eligible") for index, cost in enumerate((7000, 1000, 6000, 2000, 5000, 3000, 4000))),
        )

        selected = select_benchmark_datasets(costs)

        self.assertEqual(
            selected,
            (
                "cand0", "cand1", "cand3", "cand4", "cand6",
                "diag0", "diag1", "diag3", "diag4", "diag6",
                "pre0", "pre1", "pre3", "pre4", "pre6",
            ),
        )

    def test_benchmark_selection_removes_duplicate_quantile_positions(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import select_benchmark_datasets

        selected = select_benchmark_datasets(
            (
                DatasetCost("pre-low", 1, "preprocessing_only"),
                DatasetCost("pre-high", 2, "preprocessing_only"),
            )
        )

        self.assertEqual(selected, ("pre-high", "pre-low"))

    def test_dataset_cost_cannot_accept_test_like_metadata(self):
        self.assertEqual(
            tuple(field.name for field in fields(DatasetCost)),
            ("dataset", "cost", "stratum"),
        )
        with self.assertRaises(TypeError):
            DatasetCost("a", 1, "preprocessing_only", test_length=100)

    def test_shard_plan_assigns_every_dataset_by_largest_first_cost(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import build_shard_plan

        costs = (
            DatasetCost("a", 10, "preprocessing_only"),
            DatasetCost("b", 9, "preprocessing_only"),
            DatasetCost("c", 8, "preprocessing_only"),
            DatasetCost("d", 7, "preprocessing_only"),
            DatasetCost("e", 6, "preprocessing_only"),
        )

        plan = build_shard_plan(costs, 2)

        self.assertEqual(plan.population, ("a", "b", "c", "d", "e"))
        self.assertEqual(plan.assignments[0].datasets, ("a", "d", "e"))
        self.assertEqual(plan.assignments[0].total_cost, 23)
        self.assertEqual(plan.assignments[1].datasets, ("b", "c"))
        self.assertEqual(plan.assignments[1].total_cost, 17)
        self.assertEqual(
            tuple(dataset for shard in plan.assignments for dataset in shard.datasets),
            ("a", "d", "e", "b", "c"),
        )

    def test_shard_plan_breaks_equal_cost_dataset_and_shard_ties_deterministically(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import build_shard_plan

        plan = build_shard_plan(
            (
                DatasetCost("b", 10, "preprocessing_only"),
                DatasetCost("a", 10, "preprocessing_only"),
                DatasetCost("d", 1, "preprocessing_only"),
                DatasetCost("c", 1, "preprocessing_only"),
            ),
            2,
        )

        self.assertEqual(plan.assignments[0].datasets, ("a", "c"))
        self.assertEqual(plan.assignments[1].datasets, ("b", "d"))

    def test_shard_plan_rejects_non_frozen_worker_counts(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import build_shard_plan

        costs = (
            DatasetCost("a", 1, "preprocessing_only"),
            DatasetCost("b", 1, "preprocessing_only"),
        )
        for worker_count in (0, 3, 5):
            with self.subTest(worker_count=worker_count):
                with self.assertRaisesRegex(ValueError, "worker_count"):
                    build_shard_plan(costs, worker_count)

    def test_shard_plan_hash_is_repeatable_for_same_costs(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import build_shard_plan

        costs = (
            DatasetCost("c", 5, "candidate_eligible"),
            DatasetCost("a", 8, "preprocessing_only"),
            DatasetCost("b", 5, "diagnostic_oof"),
        )

        first = build_shard_plan(costs, 2)
        second = build_shard_plan(tuple(reversed(costs)), 2)

        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first, second)
