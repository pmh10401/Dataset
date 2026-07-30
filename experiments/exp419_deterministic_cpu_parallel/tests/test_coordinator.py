from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch
import unittest

from experiments.exp417_causal_multihorizon_shadow.config import (
    FROZEN_CONFIG as EXP417_CONFIG,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    FrozenLaneSnapshot,
    FrozenRunContext,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    ParallelRoots,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerBootstrap,
    WorkerCompletion,
)
from experiments.exp419_deterministic_cpu_parallel.workers import (
    _context_sha256,
)


def _context() -> FrozenRunContext:
    datasets = tuple(f"synthetic-{number:04d}" for number in range(1117))
    lanes = FrozenLaneSnapshot(
        hard={"lane-hard": tuple(range(2218))},
        standard={"lane-standard": tuple(range(806))},
        priority={"lane-priority": tuple(range(31))},
        automatic_normal={"lane-automatic": tuple(range(35))},
    )
    return FrozenRunContext(
        database_sha256=EXP417_CONFIG.database_sha256,
        datasets=datasets,
        lanes=lanes,
        test_identity_manifest={name: ((0, 100),) for name in datasets},
        code_commit="c" * 40,
        runtime_versions={"python": "synthetic"},
    )


def _plan(context: FrozenRunContext) -> ShardPlan:
    first = context.datasets[::2]
    second = context.datasets[1::2]
    return ShardPlan(
        2,
        (
            ShardAssignment(0, first, len(first)),
            ShardAssignment(1, second, len(second)),
        ),
        context.datasets,
    )


def _digest(dataset: str) -> str:
    return sha256(dataset.encode("utf-8")).hexdigest()


class _Store:
    def __init__(
        self,
        root: Path,
        provenance: object,
        expected_datasets: tuple[str, ...],
        *,
        completed: tuple[str, ...] = (),
        corrupt: bool = False,
        terminal: tuple[str, ...] = (),
    ) -> None:
        self.root = root
        self.provenance = provenance
        self.expected_datasets = tuple(expected_datasets)
        self._digests = {name: _digest(name) for name in completed}
        self._results = {
            name: SimpleNamespace(
                dataset=name,
                decision_evidence=(
                    _TerminalMarker() if name in terminal else object()
                ),
                status="failed" if name in terminal else "qualified",
            )
            for name in completed
        }
        self.corrupt = corrupt
        self.imported: list[str] = []

    def completed_record_digests(self):
        if self.corrupt:
            raise ValueError("corrupt synthetic checkpoint")
        return MappingProxyType(dict(self._digests))

    def load_completed(self):
        if self.corrupt:
            raise ValueError("corrupt synthetic checkpoint")
        return MappingProxyType(dict(self._results))

    def require_complete_snapshot(self):
        if set(self._digests) != set(self.expected_datasets):
            raise ValueError("synthetic checkpoint incomplete")
        return self.load_completed(), self.completed_record_digests()

    def import_validated_dataset(self, source: "_Store", dataset: str) -> str:
        if self.provenance != source.provenance:
            raise ValueError("provenance mismatch")
        if dataset not in source.expected_datasets:
            raise ValueError("wrong shard")
        if dataset not in source.completed_record_digests():
            raise ValueError("missing record")
        if dataset in self._digests:
            raise ValueError("duplicate target record")
        self._digests[dataset] = source._digests[dataset]
        self._results[dataset] = source._results[dataset]
        self.imported.append(dataset)
        return self._digests[dataset]


class _TerminalMarker:
    pass


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _context()
        self.plan = _plan(self.context)
        self.provenance = SimpleNamespace(identity="same")

    def _full_worker_stores(self, root: Path) -> dict[int, _Store]:
        return {
            assignment.shard_index: _Store(
                root / f"worker-{assignment.shard_index}",
                self.provenance,
                assignment.datasets,
                completed=assignment.datasets,
            )
            for assignment in self.plan.assignments
        }

    def test_merge_is_canonical_when_workers_finish_out_of_order(self):
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            merge_worker_checkpoints,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = self._full_worker_stores(root)
            canonical = _Store(
                root / "canonical",
                self.provenance,
                self.context.datasets,
            )

            digests = merge_worker_checkpoints(
                self.context,
                self.plan,
                {1: stores[1], 0: stores[0]},
                canonical,
            )

            self.assertEqual(tuple(digests), self.context.datasets)
            self.assertEqual(canonical.imported, list(self.context.datasets))
            self.assertEqual(
                dict(digests),
                {name: _digest(name) for name in self.context.datasets},
            )

    def test_merge_rejects_missing_record_before_canonical_mutation(self):
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            merge_worker_checkpoints,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = self._full_worker_stores(root)
            missing = self.plan.assignments[0].datasets[-1]
            stores[0]._digests.pop(missing)
            stores[0]._results.pop(missing)
            canonical = _Store(
                root / "canonical", self.provenance, self.context.datasets
            )
            with self.assertRaisesRegex(ValueError, "coverage"):
                merge_worker_checkpoints(
                    self.context, self.plan, stores, canonical
                )
            self.assertEqual(canonical.imported, [])

    def test_merge_rejects_extra_duplicate_and_wrong_shard_records(self):
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            merge_worker_checkpoints,
        )

        mutations = ("extra", "duplicate", "wrong_shard")
        for mutation in mutations:
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                root = Path(temporary)
                stores = self._full_worker_stores(root)
                foreign = self.plan.assignments[0].datasets[0]
                if mutation == "extra":
                    stores[1]._digests["outside"] = "f" * 64
                elif mutation == "duplicate":
                    stores[1]._digests[foreign] = _digest(foreign)
                else:
                    stores[1].expected_datasets += (foreign,)
                    stores[1]._digests[foreign] = _digest(foreign)
                canonical = _Store(
                    root / "canonical", self.provenance, self.context.datasets
                )
                with self.assertRaises(ValueError):
                    merge_worker_checkpoints(
                        self.context, self.plan, stores, canonical
                    )
                self.assertEqual(canonical.imported, [])

    def test_merge_rejects_wrong_plan_provenance_and_corruption(self):
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            merge_worker_checkpoints,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = self._full_worker_stores(root)
            canonical = _Store(
                root / "canonical", self.provenance, self.context.datasets
            )
            wrong_plan = ShardPlan(
                2, self.plan.assignments, tuple(reversed(self.context.datasets))
            )
            with self.assertRaisesRegex(ValueError, "frozen context"):
                merge_worker_checkpoints(
                    self.context, wrong_plan, stores, canonical
                )
            stores[0].provenance = SimpleNamespace(identity="wrong")
            with self.assertRaisesRegex(ValueError, "provenance"):
                merge_worker_checkpoints(
                    self.context, self.plan, stores, canonical
                )
            stores[0].provenance = self.provenance
            stores[0].corrupt = True
            with self.assertRaisesRegex(ValueError, "corrupt"):
                merge_worker_checkpoints(
                    self.context, self.plan, stores, canonical
                )
            self.assertEqual(canonical.imported, [])

    def test_merge_rejects_terminal_record_before_canonical_mutation(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = self._full_worker_stores(root)
            terminal_name = self.plan.assignments[1].datasets[0]
            stores[1]._results[terminal_name] = SimpleNamespace(
                dataset=terminal_name,
                decision_evidence=_TerminalMarker(),
                status="failed",
            )
            canonical = _Store(
                root / "canonical", self.provenance, self.context.datasets
            )
            with (
                patch.object(
                    coordinator,
                    "DatasetTerminalEvidence",
                    _TerminalMarker,
                ),
                self.assertRaisesRegex(ValueError, "terminal"),
            ):
                coordinator.merge_worker_checkpoints(
                    self.context, self.plan, stores, canonical
                )
            self.assertEqual(canonical.imported, [])

    def _roots(self, root: Path) -> ParallelRoots:
        return ParallelRoots(
            root,
            {
                assignment.shard_index: root
                / f"worker-{assignment.shard_index}"
                for assignment in self.plan.assignments
            },
            root / "canonical",
        )

    def _bootstrap(self) -> WorkerBootstrap:
        return WorkerBootstrap(
            SimpleNamespace(resources="synthetic"),
            self.provenance,
            _context_sha256(self.context),
        )

    def _seed_runtime_evidence(
        self,
        coordinator,
        roots: ParallelRoots,
        completions: tuple[WorkerCompletion, ...],
    ) -> None:
        binding = coordinator._freeze_plan_binding(
            roots, self.plan, self.context
        )
        coordinator._append_runtime_evidence(
            roots,
            binding,
            (),
            "seeded-prior-attempt",
            completions,
            50,
        )

    def test_resume_schedules_only_missing_records_in_original_shards(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            stores: dict[Path, _Store] = {}
            canonical = _Store(
                roots.canonical_root,
                self.provenance,
                self.context.datasets,
            )
            stores[roots.canonical_root] = canonical
            expected_missing: dict[int, tuple[str, ...]] = {}
            for assignment in self.plan.assignments:
                completed = assignment.datasets[1::3]
                expected_missing[assignment.shard_index] = tuple(
                    name
                    for name in assignment.datasets
                    if name not in completed
                )
                stores[roots.worker_roots[assignment.shard_index]] = _Store(
                    roots.worker_roots[assignment.shard_index],
                    self.provenance,
                    assignment.datasets,
                    completed=completed,
                )
            self._seed_runtime_evidence(
                coordinator,
                roots,
                tuple(
                    WorkerCompletion(
                        assignment.shard_index,
                        300 + assignment.shard_index,
                        stores[
                            roots.worker_roots[assignment.shard_index]
                        ].expected_datasets[1::3],
                        dict(
                            stores[
                                roots.worker_roots[
                                    assignment.shard_index
                                ]
                            ]._digests
                        ),
                        2.0,
                        50,
                        {
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                        },
                        "failed",
                        "synthetic prior interruption",
                    )
                    for assignment in self.plan.assignments
                ),
            )
            scheduled = []

            def open_store(path, provenance, datasets):
                if path not in stores:
                    stores[path] = _Store(path, provenance, datasets)
                return stores[path]

            def spawn(specs, bootstrap):
                del bootstrap
                scheduled.extend(specs)
                completions = []
                for spec in specs:
                    store = open_store(
                        spec.checkpoint_root,
                        self.provenance,
                        spec.shard.datasets,
                    )
                    for name in spec.shard.datasets:
                        store._digests[name] = _digest(name)
                        store._results[name] = SimpleNamespace(
                            dataset=name,
                            decision_evidence=object(),
                            status="qualified",
                        )
                    completions.append(
                        WorkerCompletion(
                            spec.shard.shard_index,
                            100 + spec.shard.shard_index,
                            spec.shard.datasets,
                            dict(store._digests),
                            1.0,
                            100,
                            {
                                "OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "VECLIB_MAXIMUM_THREADS": "1",
                            },
                            "completed",
                            "",
                        )
                    )
                return tuple(completions)

            summary = SimpleNamespace(
                publication_allowed=True, status="complete"
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    side_effect=open_store,
                ),
                patch.object(coordinator, "spawn_workers", side_effect=spawn),
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=summary,
                ),
            ):
                run = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )

            self.assertTrue(run.publication_allowed)
            self.assertEqual(
                {
                    spec.shard.shard_index: spec.shard.datasets
                    for spec in scheduled
                },
                expected_missing,
            )
            self.assertEqual(set(run.record_digests), set(self.context.datasets))

    def test_complete_canonical_store_skips_workers_and_rebuilds_summary(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            canonical = _Store(
                roots.canonical_root,
                self.provenance,
                self.context.datasets,
                completed=self.context.datasets,
            )
            self._seed_runtime_evidence(
                coordinator,
                roots,
                tuple(
                    WorkerCompletion(
                        assignment.shard_index,
                        400 + assignment.shard_index,
                        assignment.datasets,
                        {
                            name: _digest(name)
                            for name in assignment.datasets
                        },
                        2.0,
                        50,
                        {
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                        },
                        "completed",
                        "",
                    )
                    for assignment in self.plan.assignments
                ),
            )
            summary = SimpleNamespace(
                publication_allowed=True, status="complete"
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    return_value=canonical,
                ),
                patch.object(coordinator, "spawn_workers") as spawn,
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=summary,
                ) as reconstruct,
            ):
                run = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )
            self.assertTrue(run.publication_allowed)
            spawn.assert_not_called()
            reconstruct.assert_called_once_with(self.context, canonical)

    def test_aggregate_worker_memory_failure_returns_nonpublishable_run(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            stores = {
                roots.canonical_root: _Store(
                    roots.canonical_root,
                    self.provenance,
                    self.context.datasets,
                )
            }
            for assignment in self.plan.assignments:
                stores[roots.worker_roots[assignment.shard_index]] = _Store(
                    roots.worker_roots[assignment.shard_index],
                    self.provenance,
                    assignment.datasets,
                )

            def open_store(path, provenance, datasets):
                if path not in stores:
                    stores[path] = _Store(
                        path, provenance, datasets, completed=datasets
                    )
                return stores[path]

            def oversized(specs, bootstrap):
                del bootstrap
                return tuple(
                    WorkerCompletion(
                        spec.shard.shard_index,
                        200 + spec.shard.shard_index,
                        spec.shard.datasets,
                        {name: _digest(name) for name in spec.shard.datasets},
                        1.0,
                        EXP417_CONFIG.expected_datasets
                        * 100_000_000,
                        {
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                        },
                        "completed",
                        "",
                    )
                    for spec in specs
                )

            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    side_effect=open_store,
                ),
                patch.object(coordinator, "spawn_workers", side_effect=oversized),
            ):
                run = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )
            self.assertFalse(run.publication_allowed)
            self.assertEqual(run.status, "failed")

    def test_resume_rejects_a_different_plan_bound_to_the_same_roots(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            canonical = _Store(
                roots.canonical_root,
                self.provenance,
                self.context.datasets,
                completed=self.context.datasets,
            )
            self._seed_runtime_evidence(
                coordinator,
                roots,
                tuple(
                    WorkerCompletion(
                        assignment.shard_index,
                        500 + assignment.shard_index,
                        assignment.datasets,
                        {
                            name: _digest(name)
                            for name in assignment.datasets
                        },
                        2.0,
                        50,
                        {
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                        },
                        "completed",
                        "",
                    )
                    for assignment in self.plan.assignments
                ),
            )
            summary = SimpleNamespace(
                publication_allowed=True, status="complete"
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    return_value=canonical,
                ),
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=summary,
                ),
            ):
                first = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )
                changed_assignments = (
                    ShardAssignment(
                        0,
                        self.plan.assignments[0].datasets,
                        self.plan.assignments[0].total_cost + 1,
                    ),
                    self.plan.assignments[1],
                )
                different_plan = ShardPlan(
                    2, changed_assignments, self.context.datasets
                )
                second = coordinator.resume_or_run_parallel(
                    self.context,
                    different_plan,
                    roots,
                    self._bootstrap(),
                )

            self.assertTrue(first.publication_allowed)
            self.assertFalse(second.publication_allowed)
            self.assertIn("shard-plan", second.summary["error"])

    def test_new_binding_rejects_any_nonempty_checkpoint_root(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        for occupied in ("canonical", "worker"):
            with self.subTest(occupied=occupied), TemporaryDirectory() as temporary:
                root = Path(temporary)
                roots = self._roots(root)
                checkpoint = (
                    roots.canonical_root
                    if occupied == "canonical"
                    else roots.worker_roots[0]
                )
                checkpoint.mkdir()
                (checkpoint / "prior-exp417-evidence").write_text(
                    "unbound", encoding="utf-8"
                )

                with self.assertRaisesRegex(ValueError, "unbound"):
                    coordinator._freeze_plan_binding(
                        roots,
                        self.plan,
                        self.context,
                    )
                self.assertFalse(
                    (root / "exp419_shard_plan.json").exists()
                )

    def test_binding_freezes_full_runtime_and_thread_contract(self):
        from dataclasses import asdict

        from experiments.exp419_deterministic_cpu_parallel import coordinator
        from experiments.exp419_deterministic_cpu_parallel.config import (
            FROZEN_CONFIG,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            coordinator._freeze_plan_binding(
                roots,
                self.plan,
                self.context,
            )
            payload = json.loads(
                (root / "exp419_shard_plan.json").read_text("utf-8")
            )

            self.assertEqual(
                payload["frozen_config"],
                json.loads(json.dumps(asdict(FROZEN_CONFIG))),
            )
            self.assertEqual(
                payload["runtime_versions"],
                dict(self.context.runtime_versions),
            )
            self.assertEqual(payload["start_method"], "spawn")
            self.assertEqual(
                payload["thread_environment"],
                {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            )

    def test_binding_rejects_casefold_aliases_and_symlink_marker(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            aliased = ParallelRoots(
                root,
                {
                    0: root / "Shard",
                    1: root / "shard",
                },
                root / "canonical",
            )
            with self.assertRaisesRegex(ValueError, "alias"):
                coordinator._freeze_plan_binding(
                    aliased,
                    self.plan,
                    self.context,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            coordinator._freeze_plan_binding(
                roots,
                self.plan,
                self.context,
            )
            marker = root / "exp419_shard_plan.json"
            saved = root.parent / f"{root.name}-saved-binding"
            saved.write_bytes(marker.read_bytes())
            marker.unlink()
            marker.symlink_to(saved)
            with self.assertRaisesRegex(ValueError, "regular"):
                coordinator._freeze_plan_binding(
                    roots,
                    self.plan,
                    self.context,
                )

    def test_complete_canonical_without_durable_worker_evidence_fails_closed(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            coordinator._freeze_plan_binding(
                roots,
                self.plan,
                self.context,
            )
            canonical = _Store(
                roots.canonical_root,
                self.provenance,
                self.context.datasets,
                completed=self.context.datasets,
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    return_value=canonical,
                ),
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=SimpleNamespace(
                        publication_allowed=True, status="complete"
                    ),
                ),
                patch.object(
                    coordinator,
                    "_parent_peak_rss_bytes",
                    return_value=77,
                ),
            ):
                run = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )

            self.assertFalse(run.publication_allowed)
            self.assertIn("evidence", run.summary["error"])
            self.assertTrue(
                all(item.pid == os.getpid() for item in run.completions)
            )
            self.assertTrue(
                all(item.peak_rss_bytes == 77 for item in run.completions)
            )

    def test_fresh_worker_evidence_is_durable_reused_and_includes_parent_rss(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            stores: dict[Path, _Store] = {
                roots.canonical_root: _Store(
                    roots.canonical_root,
                    self.provenance,
                    self.context.datasets,
                )
            }
            for assignment in self.plan.assignments:
                stores[roots.worker_roots[assignment.shard_index]] = _Store(
                    roots.worker_roots[assignment.shard_index],
                    self.provenance,
                    assignment.datasets,
                )

            def open_store(path, provenance, datasets):
                if path not in stores:
                    stores[path] = _Store(path, provenance, datasets)
                return stores[path]

            def spawn(specs, bootstrap):
                del bootstrap
                completions = []
                for spec in specs:
                    store = open_store(
                        spec.checkpoint_root,
                        self.provenance,
                        spec.shard.datasets,
                    )
                    for name in spec.shard.datasets:
                        store._digests[name] = _digest(name)
                        store._results[name] = SimpleNamespace(
                            dataset=name,
                            decision_evidence=object(),
                            status="qualified",
                        )
                    completions.append(
                        WorkerCompletion(
                            spec.shard.shard_index,
                            700 + spec.shard.shard_index,
                            spec.shard.datasets,
                            dict(store._digests),
                            3.5 + spec.shard.shard_index,
                            100,
                            {
                                "OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "VECLIB_MAXIMUM_THREADS": "1",
                            },
                            "completed",
                            "",
                        )
                    )
                return tuple(completions)

            summary = SimpleNamespace(
                publication_allowed=True, status="complete"
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    side_effect=open_store,
                ),
                patch.object(coordinator, "spawn_workers", side_effect=spawn),
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=summary,
                ),
                patch.object(
                    coordinator,
                    "_parent_peak_rss_bytes",
                    return_value=50,
                    create=True,
                ),
            ):
                first = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )
                second = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )

            self.assertTrue(first.publication_allowed)
            self.assertTrue(second.publication_allowed)
            self.assertTrue(
                (root / "exp419_runtime_evidence.json").is_file()
            )
            self.assertEqual(
                first.summary["aggregate_peak_rss_bytes"], 250
            )
            self.assertEqual(
                second.summary["aggregate_peak_rss_bytes"], 250
            )
            first_metrics = tuple(
                (
                    item.pid,
                    item.elapsed_seconds,
                    item.peak_rss_bytes,
                    dict(item.thread_environment),
                )
                for item in first.completions
            )
            second_metrics = tuple(
                (
                    item.pid,
                    item.elapsed_seconds,
                    item.peak_rss_bytes,
                    dict(item.thread_environment),
                )
                for item in second.completions
            )
            self.assertEqual(second_metrics, first_metrics)
            self.assertEqual(
                tuple(second.record_digests), self.context.datasets
            )

    def test_partial_canonical_terminal_record_fails_before_worker_scheduling(self):
        from experiments.exp419_deterministic_cpu_parallel import coordinator

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = self._roots(root)
            terminal_name = self.context.datasets[0]
            canonical = _Store(
                roots.canonical_root,
                self.provenance,
                self.context.datasets,
                completed=(terminal_name,),
                terminal=(terminal_name,),
            )
            stores = {roots.canonical_root: canonical}
            for assignment in self.plan.assignments:
                stores[roots.worker_roots[assignment.shard_index]] = _Store(
                    roots.worker_roots[assignment.shard_index],
                    self.provenance,
                    assignment.datasets,
                    completed=assignment.datasets,
                )

            def open_store(path, provenance, datasets):
                del provenance, datasets
                return stores[path]

            summary = SimpleNamespace(
                publication_allowed=True, status="complete"
            )
            with (
                patch.object(
                    coordinator,
                    "_open_checkpoint_store",
                    side_effect=open_store,
                ),
                patch.object(
                    coordinator,
                    "DatasetTerminalEvidence",
                    _TerminalMarker,
                ),
                patch.object(coordinator, "spawn_workers") as spawn,
                patch.object(
                    coordinator,
                    "_reconstruct_global_summary",
                    return_value=summary,
                ),
            ):
                run = coordinator.resume_or_run_parallel(
                    self.context,
                    self.plan,
                    roots,
                    self._bootstrap(),
                )

            self.assertFalse(run.publication_allowed)
            self.assertIn("terminal", run.summary["error"])
            spawn.assert_not_called()

    def test_select_runtime_chooses_fastest_exactly_qualified_parallel_trial(self):
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            select_runtime,
        )

        datasets = ("a", "b")
        trials = (
            RuntimeTrial(1, 100.0, 100, True, True, "", datasets, "1" * 64),
            RuntimeTrial(2, 49.0, 100, True, True, "", datasets, "2" * 64),
            RuntimeTrial(4, 30.0, 100, True, True, "", datasets, "4" * 64),
            RuntimeTrial(6, 25.0, 100, False, True, "", datasets, "6" * 64),
        )
        benchmark = select_runtime(trials)
        self.assertEqual(benchmark.selected_worker_count, 4)
        self.assertEqual(benchmark.status, "qualified_parallel_runtime")

    def test_select_runtime_rejects_parity_memory_finite_fallback_and_speed(self):
        from experiments.exp419_deterministic_cpu_parallel.config import (
            FROZEN_CONFIG,
        )
        from experiments.exp419_deterministic_cpu_parallel.coordinator import (
            select_runtime,
        )

        datasets = ("a", "b")
        trials = (
            RuntimeTrial(1, 100.0, 100, True, True, "", datasets, "1" * 64),
            RuntimeTrial(2, 40.0, 100, False, True, "", datasets, "2" * 64),
            RuntimeTrial(
                4,
                30.0,
                FROZEN_CONFIG.memory_limit_bytes + 1,
                True,
                True,
                "",
                datasets,
                "4" * 64,
            ),
            RuntimeTrial(
                6,
                60.0,
                100,
                True,
                False,
                "cpu fallback",
                datasets,
                "6" * 64,
            ),
        )
        benchmark = select_runtime(trials)
        self.assertIsNone(benchmark.selected_worker_count)
        self.assertEqual(
            benchmark.status, "no_qualified_parallel_runtime"
        )
