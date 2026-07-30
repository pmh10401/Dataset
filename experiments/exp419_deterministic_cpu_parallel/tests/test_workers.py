from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import os
import signal
import sys
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow import checkpoints
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
)
from experiments.exp417_causal_multihorizon_shadow.config import (
    FROZEN_CONFIG as EXP417_CONFIG,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    ShardAssignment,
    WorkerBootstrap,
    WorkerCompletion,
    WorkerSpec,
)


_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_RUN_ORDER: list[str] = []
_OPENED_STORES: list["_SyntheticCheckpointStore"] = []
_SYNTHETIC_CONTEXT: FrozenRunContext | None = None


def _lanes() -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard={"lane-hard": tuple(range(2218))},
        standard={"lane-standard": tuple(range(806))},
        priority={"lane-priority": tuple(range(31))},
        automatic_normal={"lane-automatic": tuple(range(35))},
    )


def _context() -> FrozenRunContext:
    datasets = tuple(
        sorted(
            ("alpha", "beta", "outside")
            + tuple(f"synthetic-{number:04d}" for number in range(1114))
        )
    )
    return FrozenRunContext(
        database_sha256=EXP417_CONFIG.database_sha256,
        datasets=datasets,
        lanes=_lanes(),
        test_identity_manifest={name: ((0, 100),) for name in datasets},
        code_commit="c" * 40,
        runtime_versions=MappingProxyType(
            {
                "numpy": "synthetic",
                "python": "synthetic",
                "torch": "synthetic",
            }
        ),
    )


def _context_sha256(context: FrozenRunContext) -> str:
    return sha256(
        canonical_json_bytes(checkpoints._context_payload(context))
    ).hexdigest()


def _provenance() -> Exp417CheckpointProvenance:
    return Exp417CheckpointProvenance(
        git_commit="c" * 40,
        database_sha256=EXP417_CONFIG.database_sha256,
        exp291_prediction_sha256=EXP417_CONFIG.exp291_prediction_sha256,
        exp291_manifest_sha256=EXP417_CONFIG.exp291_manifest_sha256,
        exp292_prediction_sha256=EXP417_CONFIG.exp292_prediction_sha256,
        exp292_manifest_sha256=EXP417_CONFIG.exp292_manifest_sha256,
        frozen_config_sha256="5" * 64,
        preprocessing_contract_sha256="6" * 64,
        fold_contract_sha256="7" * 64,
        model_contract_sha256="8" * 64,
        model_seeds=EXP417_CONFIG.model_seeds,
        perturbation_seeds=EXP417_CONFIG.perturbation_seeds,
        runtime_versions=MappingProxyType(
            {
                "numpy": "synthetic",
                "python": "synthetic",
                "torch": "synthetic",
            }
        ),
        backend_contract_sha256="9" * 64,
    )


def _resources():
    from experiments.exp417_causal_multihorizon_shadow.runner import (
        RunnerResources,
    )

    paths = tuple(Path(f"/synthetic/input-{index}") for index in range(6))
    return RunnerResources(*paths)


def _bootstrap(context: FrozenRunContext) -> WorkerBootstrap:
    return WorkerBootstrap(_resources(), _provenance(), _context_sha256(context))


def _torch_module() -> ModuleType:
    module = ModuleType("torch")
    module.set_num_threads = Mock()
    module.set_num_interop_threads = Mock()
    return module


class _SyntheticCheckpointStore:
    def __init__(
        self,
        root: Path,
        provenance: Exp417CheckpointProvenance,
        expected_datasets: tuple[str, ...],
    ) -> None:
        self.root = root
        self.provenance = provenance
        self.expected_datasets = tuple(expected_datasets)
        self._digests: dict[str, str] = {}
        root.mkdir(parents=True)

    def complete(self, dataset: str) -> None:
        if dataset not in self.expected_datasets:
            raise ValueError("dataset outside synthetic shard")
        self._digests[dataset] = sha256(dataset.encode("utf-8")).hexdigest()

    def completed_record_digests(self):
        return MappingProxyType(dict(self._digests))


def _synthetic_freeze_context(
    resources: RunnerResources,
    provenance: Exp417CheckpointProvenance,
) -> FrozenRunContext:
    del resources, provenance
    if _SYNTHETIC_CONTEXT is None:
        raise AssertionError("synthetic context is not configured")
    return _SYNTHETIC_CONTEXT


def _synthetic_open_store(
    spec: WorkerSpec,
    provenance: Exp417CheckpointProvenance,
) -> _SyntheticCheckpointStore:
    store = _SyntheticCheckpointStore(
        spec.checkpoint_root, provenance, spec.shard.datasets
    )
    _OPENED_STORES.append(store)
    return store


def _synthetic_build_runtime(resources: RunnerResources) -> object:
    del resources
    return object()


def _result(dataset: str) -> DatasetResult:
    result = Mock(spec=DatasetResult)
    result.dataset = dataset
    return result


def _synthetic_run_success(
    dataset: str,
    context: FrozenRunContext,
    store: _SyntheticCheckpointStore,
    runtime: object,
) -> DatasetResult:
    del context, runtime
    _RUN_ORDER.append(dataset)
    store.complete(dataset)
    return _result(dataset)


def _synthetic_run_outside_shard(
    dataset: str,
    context: FrozenRunContext,
    store: _SyntheticCheckpointStore,
    runtime: object,
) -> DatasetResult:
    del dataset, context, store, runtime
    return _result("outside")


def _synthetic_run_second_failure(
    dataset: str,
    context: FrozenRunContext,
    store: _SyntheticCheckpointStore,
    runtime: object,
) -> DatasetResult:
    del context, runtime
    _RUN_ORDER.append(dataset)
    if dataset == "beta":
        raise RuntimeError("synthetic dataset failure")
    store.complete(dataset)
    return _result(dataset)


def _synthetic_run_untyped_result(
    dataset: str,
    context: FrozenRunContext,
    store: _SyntheticCheckpointStore,
    runtime: object,
) -> object:
    del context, runtime
    store.complete(dataset)
    return SimpleNamespace(dataset=dataset)


def _raise_spawn_timeout(signum: int, frame: object) -> None:
    del signum, frame
    raise TimeoutError("real spawn integration exceeded 15 seconds")


def _spawn_success_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    spec.checkpoint_root.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    (spec.checkpoint_root / "pid.txt").write_text(
        str(pid),
        encoding="utf-8",
    )
    digests = {
        dataset: sha256(
            f"{spec.shard.shard_index}:{dataset}".encode("utf-8")
        ).hexdigest()
        for dataset in spec.shard.datasets
    }
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=pid,
        completed_datasets=spec.shard.datasets,
        record_digests=digests,
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment=_THREAD_ENVIRONMENT,
        status="completed",
        error="",
    )


def _spawn_failed_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    spec.checkpoint_root.mkdir(parents=True, exist_ok=False)
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=(),
        record_digests={},
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment=_THREAD_ENVIRONMENT,
        status="failed",
        error="RuntimeError: synthetic spawned failure",
    )


class WorkerTests(unittest.TestCase):
    def setUp(self):
        global _SYNTHETIC_CONTEXT
        _SYNTHETIC_CONTEXT = _context()
        _RUN_ORDER.clear()
        _OPENED_STORES.clear()

    def test_pins_exact_thread_contract(self):
        from experiments.exp419_deterministic_cpu_parallel.workers import (
            pin_worker_threads,
        )

        torch = _torch_module()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            snapshot = pin_worker_threads()

            self.assertEqual(dict(snapshot), _THREAD_ENVIRONMENT)
            self.assertIsInstance(snapshot, MappingProxyType)
            self.assertEqual(
                {name: os.environ[name] for name in _THREAD_ENVIRONMENT},
                _THREAD_ENVIRONMENT,
            )
            torch.set_num_threads.assert_called_once_with(1)
            torch.set_num_interop_threads.assert_called_once_with(1)
            with self.assertRaises(TypeError):
                snapshot["OMP_NUM_THREADS"] = "2"

    def test_rejects_conflicting_thread_environment_without_mutation(self):
        from experiments.exp419_deterministic_cpu_parallel.workers import (
            pin_worker_threads,
        )

        inherited = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "8",
        }
        torch = _torch_module()
        with (
            patch.dict(os.environ, inherited, clear=True),
            patch.dict(sys.modules, {"torch": torch}),
        ):
            with self.assertRaisesRegex(ValueError, "OPENBLAS_NUM_THREADS"):
                pin_worker_threads()

            self.assertEqual(dict(os.environ), inherited)
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()

    def test_worker_uses_only_its_root_and_runs_assigned_order(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            worker_root = base / "worker-3"
            unassigned_root = base / "worker-4"
            spec = WorkerSpec(
                ShardAssignment(3, ("beta", "alpha"), 2),
                worker_root,
            )
            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers, "_freeze_context", _synthetic_freeze_context
                ),
                patch.object(
                    workers, "_open_checkpoint_store", _synthetic_open_store
                ),
                patch.object(
                    workers, "_build_runtime", _synthetic_build_runtime
                ),
                patch.object(
                    workers,
                    "_run_dataset_checkpointed",
                    _synthetic_run_success,
                ),
            ):
                completion = workers.run_worker(
                    spec, _bootstrap(_SYNTHETIC_CONTEXT)
                )

            self.assertEqual(_RUN_ORDER, ["beta", "alpha"])
            self.assertTrue(worker_root.is_dir())
            self.assertFalse(unassigned_root.exists())
            self.assertEqual(len(_OPENED_STORES), 1)
            self.assertEqual(
                _OPENED_STORES[0].expected_datasets, ("beta", "alpha")
            )
            self.assertEqual(completion.completed_datasets, ("beta", "alpha"))
            self.assertEqual(
                dict(completion.record_digests),
                {
                    "beta": sha256(b"beta").hexdigest(),
                    "alpha": sha256(b"alpha").hexdigest(),
                },
            )
            self.assertEqual(
                dict(completion.thread_environment), _THREAD_ENVIRONMENT
            )
            self.assertEqual(completion.status, "completed")
            self.assertEqual(completion.error, "")

    def test_worker_refuses_result_outside_assigned_shard(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers, "_freeze_context", _synthetic_freeze_context
                ),
                patch.object(
                    workers, "_open_checkpoint_store", _synthetic_open_store
                ),
                patch.object(
                    workers, "_build_runtime", _synthetic_build_runtime
                ),
                patch.object(
                    workers,
                    "_run_dataset_checkpointed",
                    _synthetic_run_outside_shard,
                ),
            ):
                completion = workers.run_worker(
                    spec, _bootstrap(_SYNTHETIC_CONTEXT)
                )

            self.assertEqual(completion.status, "failed")
            self.assertEqual(completion.completed_datasets, ())
            self.assertEqual(dict(completion.record_digests), {})
            self.assertRegex(completion.error, "outside.*shard")

    def test_worker_failure_records_only_completed_prefix(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha", "beta"), 2),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers, "_freeze_context", _synthetic_freeze_context
                ),
                patch.object(
                    workers, "_open_checkpoint_store", _synthetic_open_store
                ),
                patch.object(
                    workers, "_build_runtime", _synthetic_build_runtime
                ),
                patch.object(
                    workers,
                    "_run_dataset_checkpointed",
                    _synthetic_run_second_failure,
                ),
            ):
                completion = workers.run_worker(
                    spec, _bootstrap(_SYNTHETIC_CONTEXT)
                )

            self.assertEqual(_RUN_ORDER, ["alpha", "beta"])
            self.assertEqual(completion.status, "failed")
            self.assertEqual(completion.completed_datasets, ("alpha",))
            self.assertEqual(
                dict(completion.record_digests),
                {"alpha": sha256(b"alpha").hexdigest()},
            )
            self.assertRegex(completion.error, "synthetic dataset failure")
            self.assertNotEqual(completion.error, "")

    def test_worker_rejects_context_digest_before_creating_root(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "worker-0"
            spec = WorkerSpec(ShardAssignment(0, ("alpha",), 1), root)
            bootstrap = WorkerBootstrap(
                _resources(), _provenance(), "f" * 64
            )
            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers, "_freeze_context", _synthetic_freeze_context
                ),
                patch.object(
                    workers, "_open_checkpoint_store", _synthetic_open_store
                ),
            ):
                completion = workers.run_worker(spec, bootstrap)

            self.assertEqual(completion.status, "failed")
            self.assertRegex(completion.error, "context.*SHA-256")
            self.assertFalse(root.exists())

    def test_worker_rejects_untyped_dataset_result(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers, "_freeze_context", _synthetic_freeze_context
                ),
                patch.object(
                    workers, "_open_checkpoint_store", _synthetic_open_store
                ),
                patch.object(
                    workers, "_build_runtime", _synthetic_build_runtime
                ),
                patch.object(
                    workers,
                    "_run_dataset_checkpointed",
                    _synthetic_run_untyped_result,
                ),
            ):
                completion = workers.run_worker(
                    spec, _bootstrap(_SYNTHETIC_CONTEXT)
                )

            self.assertEqual(completion.status, "failed")
            self.assertRegex(completion.error, "DatasetResult")

    def test_worker_rejects_weakly_typed_bootstrap_contract(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            bootstrap = WorkerBootstrap(
                object(), _provenance(), _context_sha256(_SYNTHETIC_CONTEXT)
            )
            with patch.object(
                workers,
                "pin_worker_threads",
                return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
            ):
                completion = workers.run_worker(spec, bootstrap)

            self.assertEqual(completion.status, "failed")
            self.assertRegex(completion.error, "RunnerResources")
            self.assertFalse(spec.checkpoint_root.exists())

    def test_spawn_workers_uses_distinct_processes_and_canonical_order(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            specs = (
                WorkerSpec(
                    ShardAssignment(1, ("beta",), 1),
                    base / "worker-1",
                ),
                WorkerSpec(
                    ShardAssignment(0, ("alpha",), 1),
                    base / "worker-0",
                ),
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_success_worker,
                ),
                patch.dict(
                    os.environ,
                    _THREAD_ENVIRONMENT,
                    clear=False,
                ),
            ):
                previous_handler = signal.signal(
                    signal.SIGALRM,
                    _raise_spawn_timeout,
                )
                signal.setitimer(signal.ITIMER_REAL, 15.0)
                try:
                    completions = workers.spawn_workers(
                        specs,
                        _bootstrap(_SYNTHETIC_CONTEXT),
                    )
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous_handler)

            self.assertEqual(
                tuple(item.shard_index for item in completions),
                (0, 1),
            )
            self.assertEqual(len({item.pid for item in completions}), 2)
            for completion in completions:
                root = base / f"worker-{completion.shard_index}"
                self.assertEqual(
                    int((root / "pid.txt").read_text(encoding="utf-8")),
                    completion.pid,
                )
                self.assertEqual(completion.status, "completed")
            self.assertFalse(
                specs[0].checkpoint_root.is_relative_to(
                    specs[1].checkpoint_root
                )
            )
            self.assertFalse(
                specs[1].checkpoint_root.is_relative_to(
                    specs[0].checkpoint_root
                )
            )

    def test_spawn_workers_propagates_failed_child_exit_code(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_failed_worker,
                ),
                patch.dict(
                    os.environ,
                    _THREAD_ENVIRONMENT,
                    clear=False,
                ),
            ):
                completions = workers.spawn_workers(
                    (spec,),
                    _bootstrap(_SYNTHETIC_CONTEXT),
                )

            self.assertEqual(len(completions), 1)
            self.assertEqual(completions[0].status, "failed")
            self.assertRegex(completions[0].error, r"process exit code 1")

    def test_spawn_workers_rejects_shared_or_nested_checkpoint_roots(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bootstrap = _bootstrap(_SYNTHETIC_CONTEXT)
            for roots in (
                (base / "same", base / "same"),
                (base / "parent", base / "parent" / "child"),
            ):
                with self.subTest(roots=roots):
                    specs = (
                        WorkerSpec(
                            ShardAssignment(0, ("alpha",), 1),
                            roots[0],
                        ),
                        WorkerSpec(
                            ShardAssignment(1, ("beta",), 1),
                            roots[1],
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, "disjoint"):
                        workers.spawn_workers(specs, bootstrap)
                    self.assertFalse(roots[0].exists())
                    self.assertFalse(roots[1].exists())
