import unittest
from dataclasses import fields

from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    TrainLengthGroupMeta,
)
from experiments.exp419_deterministic_cpu_parallel.planner import estimate_dataset_cost


class PlannerTests(unittest.TestCase):
    @staticmethod
    def _balanced_selection_costs():
        return tuple(
            reversed(
                (
                    *(DatasetCost(f"pre{index}", 1, "preprocessing_only") for index in range(20)),
                    *(DatasetCost(f"diag{index}", 1, "diagnostic_oof") for index in range(20)),
                    *(
                        DatasetCost(
                            f"cand{index}",
                            2000 if index == 19 else 800 if index >= 15 else 1,
                            "candidate_eligible",
                        )
                        for index in range(20)
                    ),
                )
            )
        )

    def test_balanced_selection_adds_first_cost_ordered_prefix_until_proxy_gate(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import (
            select_balanced_benchmark_datasets,
        )

        proof = select_balanced_benchmark_datasets(self._balanced_selection_costs())

        self.assertEqual(
            proof.anchor_datasets,
            (
                "cand0", "cand12", "cand19", "cand4", "cand9",
                "diag0", "diag12", "diag17", "diag4", "diag9",
                "pre0", "pre12", "pre17", "pre4", "pre9",
            ),
        )
        self.assertEqual(
            proof.added_datasets,
            ("cand15", "cand16", "cand17", "cand18"),
        )
        self.assertEqual(
            proof.selected_datasets,
            (
                "cand0", "cand12", "cand15", "cand16", "cand17", "cand18",
                "cand19", "cand4", "cand9", "diag0", "diag12", "diag17",
                "diag4", "diag9", "pre0", "pre12", "pre17", "pre4", "pre9",
            ),
        )
        self.assertEqual(proof.total_cost, 5214)
        self.assertEqual(
            dict(proof.shard_loads),
            {2: (2800, 2414), 4: (2000, 1600, 807, 807), 6: (2000, 800, 800, 800, 800, 14)},
        )
        self.assertEqual(
            dict(proof.proxy_headrooms),
            {2: 1.862142857142857, 4: 2.607, 6: 2.607},
        )

        from experiments.exp419_deterministic_cpu_parallel.planner import build_shard_plan

        previous_prefix_names = (
            "cand0", "cand12", "cand15", "cand16", "cand17", "cand19",
            "cand4", "cand9", "diag0", "diag12", "diag17", "diag4", "diag9",
            "pre0", "pre12", "pre17", "pre4", "pre9",
        )
        previous_prefix = tuple(
            cost for cost in self._balanced_selection_costs()
            if cost.dataset in previous_prefix_names
        )
        self.assertEqual(sum(cost.cost for cost in previous_prefix), 4414)
        self.assertEqual(
            tuple(assignment.total_cost for assignment in build_shard_plan(previous_prefix, 4).assignments),
            (2000, 805, 805, 804),
        )
        self.assertEqual(
            tuple(assignment.total_cost for assignment in build_shard_plan(previous_prefix, 6).assignments),
            (2000, 800, 800, 800, 7, 7),
        )
        self.assertLess(4414 / 2000, 2.5)

    def test_balanced_selection_fills_single_stratum_to_six_workers_before_planning(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import (
            select_balanced_benchmark_datasets,
        )

        proof = select_balanced_benchmark_datasets(
            (
                DatasetCost("a", 4, "candidate_eligible"),
                DatasetCost("b", 2, "candidate_eligible"),
                DatasetCost("c", 1, "candidate_eligible"),
                DatasetCost("d", 1, "candidate_eligible"),
                DatasetCost("e", 1, "candidate_eligible"),
                DatasetCost("f", 1, "candidate_eligible"),
            )
        )

        self.assertEqual(proof.anchor_datasets, ("a", "c", "d", "e", "f"))
        self.assertEqual(proof.added_datasets, ("b",))
        self.assertEqual(proof.selected_datasets, ("a", "b", "c", "d", "e", "f"))
        self.assertEqual(proof.total_cost, 10)
        self.assertEqual(
            dict(proof.shard_loads),
            {2: (5, 5), 4: (4, 2, 2, 2), 6: (4, 2, 1, 1, 1, 1)},
        )
        self.assertEqual(dict(proof.proxy_headrooms), {2: 2.0, 4: 2.5, 6: 2.5})

    def test_balanced_selection_rejects_invalid_or_unbalanced_cost_populations(self):
        from experiments.exp419_deterministic_cpu_parallel.planner import (
            select_balanced_benchmark_datasets,
        )

        costs = self._balanced_selection_costs()
        with self.assertRaisesRegex(ValueError, "six"):
            select_balanced_benchmark_datasets(costs[:5])
        with self.assertRaisesRegex(ValueError, "DatasetCost"):
            select_balanced_benchmark_datasets(("not-a-cost", *costs[1:]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_balanced_benchmark_datasets(
                (*costs, DatasetCost("pre0", 1, "preprocessing_only"))
            )
        with self.assertRaisesRegex(ValueError, "cannot meet proxy gate"):
            select_balanced_benchmark_datasets(
                (
                    DatasetCost("pre", 1, "preprocessing_only"),
                    DatasetCost("diag", 1, "diagnostic_oof"),
                    DatasetCost("cand", 1000, "candidate_eligible"),
                    DatasetCost("pre-extra", 1, "preprocessing_only"),
                    DatasetCost("diag-extra", 1, "diagnostic_oof"),
                    DatasetCost("cand-extra", 1, "candidate_eligible"),
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot meet proxy gate"):
            select_balanced_benchmark_datasets(
                (
                    DatasetCost("pre", 0, "preprocessing_only"),
                    DatasetCost("diag", 0, "diagnostic_oof"),
                    DatasetCost("cand", 0, "candidate_eligible"),
                    DatasetCost("pre-extra", 0, "preprocessing_only"),
                    DatasetCost("diag-extra", 0, "diagnostic_oof"),
                    DatasetCost("cand-extra", 0, "candidate_eligible"),
                )
            )

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
