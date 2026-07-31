from __future__ import annotations

from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
import multiprocessing
from multiprocessing.connection import Connection
import pickle
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
import threading
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
    BenchmarkTrainBootstrap,
    BenchmarkTrainContext,
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
_ORIGINAL_PROCESS_START = multiprocessing.process.BaseProcess.start
_STARTED_TEST_PIDS: list[int] = []
_TEST_START_ATTEMPTS = 0
_ORIGINAL_CONNECTION_CLOSE = Connection.close
_TEST_RECEIVER: Connection | None = None
_TEST_SENDER: Connection | None = None
_INTERRUPTED_SENDER_CLOSE = False


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
    return WorkerBootstrap(
        _resources(), _provenance(), _context_sha256(context), context
    )


def _benchmark_context(context: FrozenRunContext) -> BenchmarkTrainContext:
    return BenchmarkTrainContext(
        database_sha256=context.database_sha256,
        datasets=context.datasets,
        lanes=context.lanes,
        code_commit=context.code_commit,
        runtime_versions=context.runtime_versions,
    )


def _benchmark_bootstrap(context: FrozenRunContext) -> BenchmarkTrainBootstrap:
    train_context = _benchmark_context(context)
    return BenchmarkTrainBootstrap(_resources(), train_context, train_context.sha256)


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


def _synthetic_build_runtime(
    resources: RunnerResources,
    seed_workers: int,
    on_stage: object,
    *,
    connection: object,
    context: object,
) -> object:
    del resources, seed_workers, on_stage, connection, context
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
    raise TimeoutError("synthetic fault-path watchdog expired")


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


def _large_completion(spec: WorkerSpec) -> WorkerCompletion:
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=spec.shard.datasets,
        record_digests={
            dataset: sha256(dataset.encode("utf-8")).hexdigest()
            for dataset in spec.shard.datasets
        },
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment=_THREAD_ENVIRONMENT,
        status="completed",
        error="",
    )


def _spawn_large_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    spec.checkpoint_root.mkdir(parents=True, exist_ok=False)
    return _large_completion(spec)


def _spawn_incomplete_success_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    completed = spec.shard.datasets[:1]
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=completed,
        record_digests={
            dataset: sha256(dataset.encode("utf-8")).hexdigest()
            for dataset in completed
        },
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment=_THREAD_ENVIRONMENT,
        status="completed",
        error="",
    )


def _spawn_wrong_environment_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=spec.shard.datasets,
        record_digests={
            dataset: sha256(dataset.encode("utf-8")).hexdigest()
            for dataset in spec.shard.datasets
        },
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment={
            **_THREAD_ENVIRONMENT,
            "OMP_NUM_THREADS": "2",
        },
        status="completed",
        error="",
    )


def _spawn_nonprefix_failure_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del bootstrap
    completed = spec.shard.datasets[1:2]
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=completed,
        record_digests={
            dataset: sha256(dataset.encode("utf-8")).hexdigest()
            for dataset in completed
        },
        elapsed_seconds=0.001,
        peak_rss_bytes=1,
        thread_environment=_THREAD_ENVIRONMENT,
        status="failed",
        error="RuntimeError: synthetic non-prefix failure",
    )


def _terminate_test_children() -> None:
    for process in multiprocessing.active_children():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(2.0)


def _recording_process_start(process: object) -> None:
    _ORIGINAL_PROCESS_START(process)
    _STARTED_TEST_PIDS.append(process.pid)


def _fail_second_process_start(process: object) -> None:
    global _TEST_START_ATTEMPTS
    _TEST_START_ATTEMPTS += 1
    if _TEST_START_ATTEMPTS == 2:
        raise RuntimeError("synthetic second start failure")
    _recording_process_start(process)


def _launch_then_interrupt_process_start(process: object) -> None:
    _recording_process_start(process)
    raise KeyboardInterrupt("synthetic interruption after process launch")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _spawn_blocking_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    del spec, bootstrap
    threading.Event().wait()
    raise AssertionError("unreachable")


def _plain_completion_payload(spec: WorkerSpec) -> dict[str, object]:
    from experiments.exp419_deterministic_cpu_parallel import workers

    return workers._completion_payload(
        WorkerCompletion(
            shard_index=spec.shard.shard_index,
            pid=os.getpid(),
            completed_datasets=spec.shard.datasets,
            record_digests={
                dataset: sha256(dataset.encode("utf-8")).hexdigest()
                for dataset in spec.shard.datasets
            },
            elapsed_seconds=0.001,
            peak_rss_bytes=1,
            thread_environment=_THREAD_ENVIRONMENT,
            status="completed",
            error="",
        )
    )


def _spawn_abrupt_or_blocking_entry(
    worker_function: object,
    spec: WorkerSpec,
    bootstrap_payload: object,
    result_connection: object,
) -> None:
    del worker_function, bootstrap_payload
    if spec.shard.shard_index == 0:
        result_connection.close()
        os._exit(7)
    threading.Event().wait()


def _spawn_malformed_entry(
    worker_function: object,
    spec: WorkerSpec,
    bootstrap_payload: object,
    result_connection: object,
) -> None:
    del worker_function, spec, bootstrap_payload
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    result_connection.send({"malformed": True})
    threading.Event().wait()


def _spawn_multiple_entry(
    worker_function: object,
    spec: WorkerSpec,
    bootstrap_payload: object,
    result_connection: object,
) -> None:
    del worker_function, bootstrap_payload
    payload = _plain_completion_payload(spec)
    result_connection.send(payload)
    result_connection.send(payload)
    threading.Event().wait()


def _raise_keyboard_interrupt(waitables: object) -> None:
    del waitables
    raise KeyboardInterrupt("synthetic explicit interruption")


class _RecordingSpawnContext:
    def __init__(self, context: object) -> None:
        self._context = context

    def Pipe(self, *, duplex: bool):
        global _TEST_RECEIVER, _TEST_SENDER
        receiver, sender = self._context.Pipe(duplex=duplex)
        _TEST_RECEIVER = receiver
        _TEST_SENDER = sender
        return receiver, sender

    def Process(self, *args: object, **kwargs: object):
        return self._context.Process(*args, **kwargs)


def _interrupt_first_sender_close(connection: Connection) -> None:
    global _INTERRUPTED_SENDER_CLOSE
    if connection is _TEST_SENDER and not _INTERRUPTED_SENDER_CLOSE:
        _INTERRUPTED_SENDER_CLOSE = True
        raise KeyboardInterrupt("synthetic sender close interruption")
    _ORIGINAL_CONNECTION_CLOSE(connection)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        global _SYNTHETIC_CONTEXT, _TEST_START_ATTEMPTS
        global _TEST_RECEIVER, _TEST_SENDER, _INTERRUPTED_SENDER_CLOSE
        _SYNTHETIC_CONTEXT = _context()
        _RUN_ORDER.clear()
        _OPENED_STORES.clear()
        _STARTED_TEST_PIDS.clear()
        _TEST_START_ATTEMPTS = 0
        _TEST_RECEIVER = None
        _TEST_SENDER = None
        _INTERRUPTED_SENDER_CLOSE = False
        from experiments.exp419_deterministic_cpu_parallel import workers

        self._input_verifier = patch.object(
            workers, "_verify_official_worker_inputs", return_value=None
        )
        self._worker_connection = patch.object(
            workers,
            "_verified_worker_connection",
            return_value=nullcontext(object()),
        )
        self._input_verifier.start()
        self._worker_connection.start()
        self.addCleanup(self._input_verifier.stop)
        self.addCleanup(self._worker_connection.stop)

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
                    workers,
                    "_freeze_context",
                    side_effect=AssertionError("official context rebuild"),
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
            self.assertEqual(completion.status, "completed", completion.error)
            self.assertEqual(completion.error, "")

            status = json.loads(
                (spec.status_root / "exp419_worker_status.json").read_bytes()
            )
            self.assertEqual(status["phase"], "worker_completed")
            self.assertEqual(status["dataset"], "alpha")
            self.assertEqual(status["dataset_index"], 1)
            self.assertEqual(status["completed_datasets"], 2)
            self.assertEqual(status["seed"], None)
            self.assertEqual(status["fold"], None)
            self.assertEqual(status["error"], "")
            self.assertTrue(
                {"prediction", "score", "label", "metric"}.isdisjoint(status)
            )

    def test_bounded_smoke_uses_train_context_without_full_context_freezer(self):
        """The two-batch smoke must not construct Exp417 TEST identity metadata."""
        from experiments.exp417_causal_multihorizon_shadow.runner import (
            RunnerResources,
        )
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "train-only.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.executescript(
                    """
                    CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT);
                    CREATE TABLE instances (
                        dataset_id INTEGER,
                        split TEXT,
                        label TEXT,
                        instance_index INTEGER,
                        values_blob BLOB
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO datasets (id, name) VALUES (1, 'alpha')"
                )
                connection.execute(
                    """
                    INSERT INTO instances
                    (dataset_id, split, label, instance_index, values_blob)
                    VALUES (1, 'TRAIN', '0', 0, ?)
                    """,
                    (struct.pack("<8f", *range(8)),),
                )
                connection.commit()
            finally:
                connection.close()
            context = _benchmark_context(_SYNTHETIC_CONTEXT)
            resources = RunnerResources(
                database,
                Path("/synthetic/exp291-prediction"),
                Path("/synthetic/exp291-manifest"),
                Path("/synthetic/exp292-prediction"),
                Path("/synthetic/exp292-manifest"),
                Path("/synthetic/backend-audit"),
            )
            bootstrap = BenchmarkTrainBootstrap(
                resources, context, context.sha256
            )
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )

            @contextmanager
            def train_only_connection(*_args):
                live = sqlite3.connect(database)
                try:
                    yield live
                finally:
                    live.close()

            with (
                patch.object(
                    workers,
                    "pin_worker_threads",
                    return_value=MappingProxyType(dict(_THREAD_ENVIRONMENT)),
                ),
                patch.object(
                    workers,
                    "_freeze_context",
                    side_effect=AssertionError("full context freezer"),
                ),
                patch.object(workers, "_run_bounded_train_batches"),
                patch(
                    "experiments.exp417_causal_multihorizon_shadow.inputs."
                    "verified_read_only_connection",
                    side_effect=train_only_connection,
                ),
            ):
                completion = workers.run_bounded_train_smoke_worker(
                    spec, bootstrap
                )

            self.assertEqual(completion.status, "completed", completion.error)
            self.assertEqual(completion.completed_datasets, ("alpha",))

    def test_checkpoint_store_uses_canonical_universe_without_changing_run_order(
        self,
    ):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "worker-0"
            spec = WorkerSpec(
                ShardAssignment(0, ("beta", "alpha"), 2),
                root,
            )

            store = workers._open_checkpoint_store(spec, _provenance())

            self.assertEqual(spec.shard.datasets, ("beta", "alpha"))
            self.assertEqual(store.expected_datasets, ("alpha", "beta"))
            self.assertTrue(root.is_dir())

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
                _resources(), _provenance(), "f" * 64, _SYNTHETIC_CONTEXT
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
                object(),
                _provenance(),
                _context_sha256(_SYNTHETIC_CONTEXT),
                _SYNTHETIC_CONTEXT,
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
                completions = workers.spawn_workers(
                    specs,
                    _bootstrap(_SYNTHETIC_CONTEXT),
                )

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

    def test_spawn_workers_drains_oversized_completion_before_join(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            large_datasets = tuple(
                f"synthetic-large-dataset-{index:04d}-{'x' * 32}"
                for index in range(1117)
            )
            spec = WorkerSpec(
                ShardAssignment(
                    0,
                    large_datasets,
                    len(large_datasets),
                ),
                Path(temporary) / "worker-0",
            )
            sample_payload = workers._completion_payload(
                _large_completion(spec)
            )
            self.assertGreater(len(pickle.dumps(sample_payload)), 113_000)
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_large_worker,
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

            self.assertEqual(
                completions[0].completed_datasets,
                spec.shard.datasets,
            )
            self.assertEqual(
                set(completions[0].record_digests),
                set(spec.shard.datasets),
            )

    def test_spawn_workers_rejects_invalid_terminal_contracts(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        cases = (
            (_spawn_incomplete_success_worker, "successful completion"),
            (_spawn_wrong_environment_worker, "thread environment"),
            (_spawn_nonprefix_failure_worker, "assigned prefix"),
        )
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, (worker_function, message) in enumerate(cases):
                with self.subTest(worker=worker_function.__name__):
                    spec = WorkerSpec(
                        ShardAssignment(
                            0,
                            ("alpha", "beta"),
                            2,
                        ),
                        base / f"worker-{index}",
                    )
                    with (
                        patch.object(
                            workers,
                            "run_worker",
                            worker_function,
                        ),
                        patch.dict(
                            os.environ,
                            _THREAD_ENVIRONMENT,
                            clear=False,
                        ),
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            workers.spawn_workers(
                                (spec,),
                                _bootstrap(_SYNTHETIC_CONTEXT),
                            )

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

    def test_spawn_workers_rejects_casefolded_and_symlink_root_aliases(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            actual = base / "actual"
            actual.mkdir()
            alias = base / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            bootstrap = _bootstrap(_SYNTHETIC_CONTEXT)
            cases = (
                (base / "Worker-A", base / "worker-a"),
                (actual, alias),
            )
            for roots in cases:
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

    def test_partial_start_failure_terminates_and_reaps_started_children(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        observed: BaseException | None = None
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            specs = (
                WorkerSpec(
                    ShardAssignment(0, ("alpha",), 1),
                    base / "worker-0",
                ),
                WorkerSpec(
                    ShardAssignment(1, ("beta",), 1),
                    base / "worker-1",
                ),
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_blocking_worker,
                ),
                patch.object(
                    multiprocessing.process.BaseProcess,
                    "start",
                    _fail_second_process_start,
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
                signal.setitimer(signal.ITIMER_REAL, 10.0)
                try:
                    workers.spawn_workers(
                        specs,
                        _bootstrap(_SYNTHETIC_CONTEXT),
                    )
                except BaseException as error:
                    observed = error
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous_handler)
            surviving = tuple(
                pid for pid in _STARTED_TEST_PIDS if _pid_exists(pid)
            )
            _terminate_test_children()

        self.assertIsInstance(observed, RuntimeError)
        self.assertRegex(str(observed), "second start failure")
        self.assertEqual(surviving, ())

    def test_start_interruption_after_launch_cleans_owned_child_and_pipes(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        observed: BaseException | None = None
        real_context = multiprocessing.get_context("spawn")
        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_blocking_worker,
                ),
                patch.object(
                    workers.multiprocessing,
                    "get_context",
                    return_value=_RecordingSpawnContext(real_context),
                ),
                patch.object(
                    multiprocessing.process.BaseProcess,
                    "start",
                    _launch_then_interrupt_process_start,
                ),
                patch.dict(
                    os.environ,
                    _THREAD_ENVIRONMENT,
                    clear=False,
                ),
            ):
                try:
                    workers.spawn_workers(
                        (spec,),
                        _bootstrap(_SYNTHETIC_CONTEXT),
                    )
                except BaseException as error:
                    observed = error

            surviving = tuple(
                pid for pid in _STARTED_TEST_PIDS if _pid_exists(pid)
            )
            closed_before_fallback = (
                _TEST_RECEIVER is not None and _TEST_RECEIVER.closed,
                _TEST_SENDER is not None and _TEST_SENDER.closed,
            )
            _terminate_test_children()
            if _TEST_RECEIVER is not None and not _TEST_RECEIVER.closed:
                _ORIGINAL_CONNECTION_CLOSE(_TEST_RECEIVER)
            if _TEST_SENDER is not None and not _TEST_SENDER.closed:
                _ORIGINAL_CONNECTION_CLOSE(_TEST_SENDER)

        self.assertIsInstance(observed, KeyboardInterrupt)
        self.assertRegex(str(observed), "after process launch")
        self.assertEqual(surviving, ())
        self.assertEqual(closed_before_fallback, (True, True))

    def test_post_start_sender_close_interruption_cleans_owned_child_and_pipes(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        observed: BaseException | None = None
        real_context = multiprocessing.get_context("spawn")
        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_blocking_worker,
                ),
                patch.object(
                    workers.multiprocessing,
                    "get_context",
                    return_value=_RecordingSpawnContext(real_context),
                ),
                patch.object(
                    multiprocessing.process.BaseProcess,
                    "start",
                    _recording_process_start,
                ),
                patch.object(
                    Connection,
                    "close",
                    _interrupt_first_sender_close,
                ),
                patch.dict(
                    os.environ,
                    _THREAD_ENVIRONMENT,
                    clear=False,
                ),
            ):
                try:
                    workers.spawn_workers(
                        (spec,),
                        _bootstrap(_SYNTHETIC_CONTEXT),
                    )
                except BaseException as error:
                    observed = error

            surviving = tuple(
                pid for pid in _STARTED_TEST_PIDS if _pid_exists(pid)
            )
            closed_before_fallback = (
                _TEST_RECEIVER is not None and _TEST_RECEIVER.closed,
                _TEST_SENDER is not None and _TEST_SENDER.closed,
            )
            _terminate_test_children()
            if _TEST_RECEIVER is not None and not _TEST_RECEIVER.closed:
                _ORIGINAL_CONNECTION_CLOSE(_TEST_RECEIVER)
            if _TEST_SENDER is not None and not _TEST_SENDER.closed:
                _ORIGINAL_CONNECTION_CLOSE(_TEST_SENDER)

        self.assertIsInstance(observed, KeyboardInterrupt)
        self.assertRegex(str(observed), "sender close interruption")
        self.assertEqual(surviving, ())
        self.assertEqual(closed_before_fallback, (True, True))

    def test_ipc_faults_terminate_and_reap_all_children(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        cases = (
            (_spawn_abrupt_or_blocking_entry, "without a completion", 2),
            (_spawn_malformed_entry, "invalid completion", 1),
            (_spawn_multiple_entry, "multiple completion", 1),
        )
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            for case_index, (entry, message, worker_count) in enumerate(cases):
                with self.subTest(entry=entry.__name__):
                    _STARTED_TEST_PIDS.clear()
                    specs = tuple(
                        WorkerSpec(
                            ShardAssignment(
                                index,
                                ("alpha",) if index == 0 else ("beta",),
                                1,
                            ),
                            base / f"case-{case_index}-worker-{index}",
                        )
                        for index in range(worker_count)
                    )
                    observed: BaseException | None = None
                    with (
                        patch.object(
                            workers,
                            "_worker_process_entry",
                            entry,
                        ),
                        patch.object(
                            multiprocessing.process.BaseProcess,
                            "start",
                            _recording_process_start,
                        ),
                    ):
                        previous_handler = signal.signal(
                            signal.SIGALRM,
                            _raise_spawn_timeout,
                        )
                        signal.setitimer(signal.ITIMER_REAL, 10.0)
                        try:
                            workers.spawn_workers(
                                specs,
                                _bootstrap(_SYNTHETIC_CONTEXT),
                            )
                        except BaseException as error:
                            observed = error
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0.0)
                            signal.signal(signal.SIGALRM, previous_handler)
                    surviving = tuple(
                        pid
                        for pid in _STARTED_TEST_PIDS
                        if _pid_exists(pid)
                    )
                    _terminate_test_children()
                    self.assertIsInstance(observed, RuntimeError)
                    self.assertRegex(str(observed), message)
                    self.assertEqual(surviving, ())

    def test_explicit_interruption_terminates_and_reaps_children(self):
        from experiments.exp419_deterministic_cpu_parallel import workers

        observed: BaseException | None = None
        with TemporaryDirectory() as temporary:
            spec = WorkerSpec(
                ShardAssignment(0, ("alpha",), 1),
                Path(temporary) / "worker-0",
            )
            with (
                patch.object(
                    workers,
                    "run_worker",
                    _spawn_blocking_worker,
                ),
                patch.object(
                    workers,
                    "wait_connections",
                    _raise_keyboard_interrupt,
                    create=True,
                ),
                patch.object(
                    multiprocessing.process.BaseProcess,
                    "start",
                    _recording_process_start,
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
                signal.setitimer(signal.ITIMER_REAL, 10.0)
                try:
                    workers.spawn_workers(
                        (spec,),
                        _bootstrap(_SYNTHETIC_CONTEXT),
                    )
                except BaseException as error:
                    observed = error
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous_handler)
            surviving = tuple(
                pid for pid in _STARTED_TEST_PIDS if _pid_exists(pid)
            )
            _terminate_test_children()

        self.assertIsInstance(observed, KeyboardInterrupt)
        self.assertEqual(surviving, ())
