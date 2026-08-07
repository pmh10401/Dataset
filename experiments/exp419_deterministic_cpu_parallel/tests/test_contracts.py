from copy import deepcopy
import math
from types import MappingProxyType
import unittest

from experiments.exp419_deterministic_cpu_parallel.contracts import (
    BenchmarkSelectionProof,
    DatasetCost,
    ParallelRun,
    RuntimeBenchmark,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerCompletion,
    benchmark_selection_proof_from_payload,
    benchmark_selection_proof_payload,
)


class Exp419ContractTests(unittest.TestCase):
    def _selection_proof(self, **changes):
        values = {
            "anchor_datasets": ("a", "b"),
            "added_datasets": ("c", "d"),
            "selected_datasets": ("a", "b", "c", "d"),
            "proxy_headroom_gate": 2.5,
            "total_cost": 100,
            "shard_loads": {
                2: (50, 50),
                4: (40, 20, 20, 20),
                6: (40, 12, 12, 12, 12, 12),
            },
            "proxy_headrooms": {2: 2.0, 4: 2.5, 6: 2.5},
        }
        values.update(changes)
        return BenchmarkSelectionProof(**values)

    def test_valid_benchmark_selection_proof_is_immutable_and_repeatable(self):
        proof = self._selection_proof()
        self.assertEqual(proof.anchor_datasets, ("a", "b"))
        self.assertEqual(proof.added_datasets, ("c", "d"))
        self.assertEqual(proof.selected_datasets, ("a", "b", "c", "d"))
        self.assertEqual(proof.proxy_headroom_gate, 2.5)
        self.assertEqual(tuple(proof.shard_loads), (2, 4, 6))
        self.assertEqual(tuple(proof.proxy_headrooms), (2, 4, 6))
        self.assertIsInstance(proof.shard_loads, MappingProxyType)
        self.assertIsInstance(proof.proxy_headrooms, MappingProxyType)
        self.assertEqual(len(proof.sha256), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in proof.sha256))
        self.assertEqual(proof.sha256, self._selection_proof().sha256)

    def test_benchmark_selection_proof_requires_lexical_disjoint_exact_datasets(self):
        invalid = (
            {"anchor_datasets": ("b", "a")},
            {"added_datasets": ("c", "c")},
            {"selected_datasets": ("a", "b", "d", "c")},
            {"anchor_datasets": ("a", "b"), "added_datasets": ("b", "c")},
            {"selected_datasets": ("a", "b", "c", "e")},
            {"anchor_datasets": ()},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._selection_proof(**changes)

    def test_benchmark_selection_proof_rejects_invalid_gate_cost_and_numeric_values(self):
        invalid = (
            {"proxy_headroom_gate": 2.0},
            {"proxy_headroom_gate": True},
            {"proxy_headroom_gate": math.inf},
            {"total_cost": -1},
            {"total_cost": True},
            {"shard_loads": {2: (50, 50), 4: (40, 20, 20, 20), 6: (40, 12, 12, 12, 12, -1)}},
            {"shard_loads": {2: (50, 50), 4: (40, 20, 20, 20), 6: (40, 12, 12, 12, 12, True)}},
            {"proxy_headrooms": {2: 2.0, 4: math.nan, 6: 2.5}},
            {"proxy_headrooms": {2: 2.0, 4: True, 6: 2.5}},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._selection_proof(**changes)

    def test_benchmark_selection_proof_requires_exact_ordered_worker_mappings(self):
        invalid = (
            {"shard_loads": {2: (50, 50), 4: (40, 20, 20, 20)}},
            {"shard_loads": {2: (50, 50), 4: (40, 20, 20, 20), 6: (40, 12, 12, 12, 12, 12), 8: (25,) * 8}},
            {"shard_loads": {4: (40, 20, 20, 20), 2: (50, 50), 6: (40, 12, 12, 12, 12, 12)}},
            {"shard_loads": {2.0: (50, 50), 4.0: (40, 20, 20, 20), 6.0: (40, 12, 12, 12, 12, 12)}},
            {"proxy_headrooms": {2: 2.0, 4: 2.5}},
            {"proxy_headrooms": {2: 2.0, 4: 2.5, 6: 2.5, 8: 2.5}},
            {"proxy_headrooms": {4: 2.5, 2: 2.0, 6: 2.5}},
            {"proxy_headrooms": {2.0: 2.0, 4.0: 2.5, 6.0: 2.5}},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._selection_proof(**changes)

    def test_benchmark_selection_proof_requires_matching_loads_and_headrooms(self):
        invalid = (
            {"shard_loads": {2: (50, 50), 4: (40, 20, 20), 6: (40, 12, 12, 12, 12, 12)}},
            {"shard_loads": {2: (50, 50), 4: (39, 20, 20, 20), 6: (40, 12, 12, 12, 12, 12)}},
            {"proxy_headrooms": {2: 2.0, 4: 2.4, 6: 2.5}},
            {"proxy_headrooms": {2: 2.0, 4: 2.5, 6: 2.4}},
            {"proxy_headrooms": {2: 2.1, 4: 2.5, 6: 2.5}},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self._selection_proof(**changes)

    def test_benchmark_selection_proof_payload_round_trip_and_rejects_mutation(self):
        proof = self._selection_proof()
        payload = benchmark_selection_proof_payload(proof)
        self.assertEqual(
            payload,
            {
                "anchor_datasets": ["a", "b"],
                "added_datasets": ["c", "d"],
                "selected_datasets": ["a", "b", "c", "d"],
                "proxy_headroom_gate": 2.5,
                "total_cost": 100,
                "shard_loads": {
                    "2": [50, 50],
                    "4": [40, 20, 20, 20],
                    "6": [40, 12, 12, 12, 12, 12],
                },
                "proxy_headrooms": {"2": 2.0, "4": 2.5, "6": 2.5},
                "sha256": proof.sha256,
            },
        )
        self.assertEqual(benchmark_selection_proof_from_payload(payload), proof)
        for mutation in (
            lambda value: value.pop("sha256"),
            lambda value: value.update({"unexpected": None}),
            lambda value: value["selected_datasets"].__setitem__(0, "z"),
            lambda value: value["shard_loads"].__setitem__("4", [41, 20, 20, 19]),
            lambda value: value.update({"sha256": "0" * 64}),
        ):
            with self.subTest(mutation=mutation):
                mutated = deepcopy(payload)
                mutation(mutated)
                with self.assertRaises(ValueError):
                    benchmark_selection_proof_from_payload(mutated)

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

    def _benchmark_selection_proof(self, **changes):
        values = {
            "anchor_datasets": ("a",),
            "added_datasets": ("b",),
            "selected_datasets": ("a", "b"),
        }
        values.update(changes)
        return self._selection_proof(**values)

    def _runtime_benchmark(self, proof=None):
        proof = proof or self._benchmark_selection_proof()
        return RuntimeBenchmark(
            trials=tuple(self._trial(count) for count in (1, 2, 4, 6)),
            selected_worker_count=2,
            status="qualified_parallel_runtime",
            benchmark_datasets=("a", "b"),
            selection_proof=proof,
        )

    def test_runtime_benchmark_binds_exact_valid_selection_proof(self):
        benchmark = self._runtime_benchmark()
        self.assertEqual(
            benchmark.benchmark_datasets,
            benchmark.selection_proof.selected_datasets,
        )
        different_datasets = self._benchmark_selection_proof(
            anchor_datasets=("a", "b"),
            added_datasets=("c",),
            selected_datasets=("a", "b", "c"),
        )
        with self.assertRaisesRegex(ValueError, "selection proof"):
            self._runtime_benchmark(different_datasets)

        proof = self._benchmark_selection_proof()
        mutations = (
            ("anchor_datasets", ("b",)),
            ("added_datasets", ("a",)),
            (
                "shard_loads",
                MappingProxyType(
                    {
                        2: (51, 49),
                        4: (40, 20, 20, 20),
                        6: (40, 12, 12, 12, 12, 12),
                    }
                ),
            ),
            (
                "proxy_headrooms",
                MappingProxyType({2: 100 / 51, 4: 2.5, 6: 2.5}),
            ),
            ("total_cost", 101),
            ("sha256", "0" * 64),
        )
        for name, value in mutations:
            with self.subTest(name=name):
                mutated = self._benchmark_selection_proof()
                object.__setattr__(mutated, name, value)
                with self.assertRaisesRegex(ValueError, "selection proof"):
                    self._runtime_benchmark(mutated)

        other = self._benchmark_selection_proof(
            anchor_datasets=("b",),
            added_datasets=("a",),
        )
        copied_digest = self._benchmark_selection_proof()
        object.__setattr__(copied_digest, "sha256", other.sha256)
        with self.assertRaisesRegex(ValueError, "selection proof"):
            self._runtime_benchmark(copied_digest)

        with self.assertRaisesRegex(ValueError, "BenchmarkSelectionProof"):
            self._runtime_benchmark(object())

    def test_runtime_benchmark_sha256_includes_full_selection_proof(self):
        first = self._runtime_benchmark()
        second = self._runtime_benchmark(
            self._benchmark_selection_proof(
                anchor_datasets=("b",),
                added_datasets=("a",),
            )
        )
        self.assertNotEqual(first.selection_proof.sha256, second.selection_proof.sha256)
        self.assertNotEqual(first.sha256, second.sha256)

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
                self._benchmark_selection_proof(),
            )

    def test_qualified_runtime_benchmark_requires_a_parallel_selection(self):
        with self.assertRaisesRegex(ValueError, "selected_worker_count"):
            RuntimeBenchmark(
                tuple(self._trial(count) for count in (1, 2, 4, 6)),
                1,
                "qualified_parallel_runtime",
                ("a", "b"),
                self._benchmark_selection_proof(),
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
                self._benchmark_selection_proof(),
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
