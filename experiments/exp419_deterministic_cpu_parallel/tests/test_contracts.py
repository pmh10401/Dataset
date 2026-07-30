import math
import unittest

from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    ParallelRun,
    RuntimeBenchmark,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerCompletion,
)


class Exp419ContractTests(unittest.TestCase):
    def _plan(self):
        return ShardPlan(
            2,
            (
                ShardAssignment(0, ("a",), 10),
                ShardAssignment(1, ("b",), 20),
            ),
            ("a", "b"),
        )

    def _completion(self, shard_index, datasets, *, status="completed", error=""):
        return WorkerCompletion(
            shard_index,
            shard_index + 10,
            datasets,
            {dataset: "a" * 64 for dataset in datasets},
            1.0,
            1,
            {"OMP_NUM_THREADS": "1"},
            status,
            error,
        )

    def _trial(self, worker_count, datasets=("a", "b")):
        return RuntimeTrial(
            worker_count,
            1.0,
            1,
            True,
            True,
            "",
            datasets,
            f"{worker_count:x}" * 64,
        )

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

    def test_qualified_runtime_benchmark_requires_all_frozen_worker_candidates(self):
        with self.assertRaisesRegex(ValueError, "worker_candidates"):
            RuntimeBenchmark(
                (self._trial(1), self._trial(2), self._trial(4)),
                2,
                "qualified_parallel_runtime",
                ("a", "b"),
            )

    def test_qualified_runtime_benchmark_requires_a_parallel_selection(self):
        with self.assertRaisesRegex(ValueError, "selected_worker_count"):
            RuntimeBenchmark(
                tuple(self._trial(count) for count in (1, 2, 4, 6)),
                1,
                "qualified_parallel_runtime",
                ("a", "b"),
            )

    def test_runtime_benchmark_requires_trials_to_cover_exact_benchmark_datasets(self):
        with self.assertRaisesRegex(ValueError, "completed_datasets"):
            RuntimeBenchmark(
                (
                    self._trial(1),
                    self._trial(2, ("a",)),
                    self._trial(4),
                    self._trial(6),
                ),
                2,
                "qualified_parallel_runtime",
                ("a", "b"),
            )

    def test_completed_parallel_run_requires_every_plan_shard_completion(self):
        plan = self._plan()
        with self.assertRaisesRegex(ValueError, "completions"):
            ParallelRun(
                plan,
                (self._completion(0, ("a",)),),
                {"a": "a" * 64, "b": "b" * 64},
                {},
                True,
                "completed",
            )

    def test_completed_parallel_run_requires_correct_shards_and_full_digests(self):
        plan = self._plan()
        with self.assertRaisesRegex(ValueError, "shard"):
            ParallelRun(
                plan,
                (self._completion(0, ("b",)), self._completion(1, ("a",))),
                {"a": "a" * 64, "b": "b" * 64},
                {},
                True,
                "completed",
            )
        with self.assertRaisesRegex(ValueError, "record_digests"):
            ParallelRun(
                plan,
                (self._completion(0, ("a",)), self._completion(1, ("b",))),
                {"a": "a" * 64},
                {},
                True,
                "completed",
            )

    def test_parallel_run_requires_publication_and_status_consistency(self):
        plan = self._plan()
        completions = (self._completion(0, ("a",)), self._completion(1, ("b",)))
        digests = {"a": "a" * 64, "b": "b" * 64}
        for publication_allowed, status in ((True, "failed"), (False, "completed")):
            with self.subTest(publication_allowed=publication_allowed, status=status):
                with self.assertRaisesRegex(ValueError, "publication_allowed"):
                    ParallelRun(plan, completions, digests, {}, publication_allowed, status)
