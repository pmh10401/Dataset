"""Isolated spawned-worker execution for Exp419."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import multiprocessing
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import resource
import sys
from time import monotonic
from types import MappingProxyType
import unicodedata

from .contracts import (
    BenchmarkTrainBootstrap,
    BenchmarkTrainContext,
    RuntimeLogRecord,
    WorkerBootstrap,
    WorkerCompletion,
    WorkerSpec,
    benchmark_train_context_from_payload,
    benchmark_train_context_payload,
)
from .config import FROZEN_CONFIG, seed_workers_for_outer_workers
from .worker_status import (
    WorkerStatus,
    append_runtime_log,
    write_worker_status,
)


_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_CLEANUP_TERMINATE_SECONDS = 2.0
_CLEANUP_KILL_SECONDS = 2.0


def pin_worker_threads() -> Mapping[str, str]:
    """Apply Exp419's exact one-thread CPU contract without overwriting conflicts."""
    conflicts = {
        name: value
        for name, value in _THREAD_ENVIRONMENT.items()
        if (inherited := os.environ.get(name)) is not None and inherited != value
    }
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"conflicting worker thread environment: {names}")

    os.environ.update(_THREAD_ENVIRONMENT)

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return MappingProxyType(dict(_THREAD_ENVIRONMENT))


def _freeze_context(resources: object, provenance: object) -> object:
    from experiments.exp417_causal_multihorizon_shadow.inputs import (
        freeze_lane_inputs,
        freeze_run_context,
    )

    lanes = freeze_lane_inputs(
        resources.exp291_prediction,
        resources.exp291_manifest,
        resources.exp292_prediction,
        resources.exp292_manifest,
    )
    return freeze_run_context(
        resources.database,
        lanes,
        verified_code_commit=provenance.git_commit,
    )


def _open_checkpoint_store(spec: WorkerSpec, provenance: object) -> object:
    from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
        Exp417CheckpointStore,
    )

    return Exp417CheckpointStore(
        spec.checkpoint_root,
        provenance,
        tuple(sorted(spec.shard.datasets)),
    )


def _build_runtime(
    resources: object,
    seed_workers: int,
    on_stage: Callable[[str, int | None, int | None], None],
    *,
    connection: object,
    context: object,
) -> object:
    """Keep all production stages except deterministic TRAIN seed fitting."""
    from experiments.exp417_causal_multihorizon_shadow.backend import (
        load_frozen_backend_audit,
    )
    from experiments.exp417_causal_multihorizon_shadow.inputs import (
        load_raw_test_values_from_connection,
        load_raw_train_values_from_connection,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerRuntime
    from .seed_parallel import train_models_seed_parallel

    audit = load_frozen_backend_audit(resources.backend_audit_path)

    def train_models(processed: object) -> object:
        return train_models_seed_parallel(
            processed,
            audit,
            seed_workers,
            on_stage=lambda phase, _length, seed: on_stage(phase, seed, None),
        )

    base = RunnerRuntime.production_with_train_models(resources, train_models)

    def verify(current_context: object) -> None:
        if current_context is not context:
            raise ValueError("worker runtime context changed")

    def load_train(dataset: str, current_context: object) -> object:
        on_stage("train_loading", None, None)
        if current_context is not context:
            raise ValueError("worker runtime context changed")
        return load_raw_train_values_from_connection(
            connection,
            dataset,
            database_sha256=context.database_sha256,
            datasets=context.datasets,
        )

    def preprocess(train: object) -> object:
        on_stage("preprocessing", None, None)
        return base.preprocess(train)

    def qualify(dataset: str, processed: object, models: object) -> object:
        on_stage("qualification", None, None)
        return base.qualify(dataset, processed, models)

    def load_test(dataset: str, contract: object, current_context: object) -> object:
        on_stage("test_scoring", None, None)
        if current_context is not context:
            raise ValueError("worker runtime context changed")
        return load_raw_test_values_from_connection(
            connection, dataset, contract, context
        )

    return replace(
        base,
        verify=verify,
        load_train=load_train,
        preprocess=preprocess,
        qualify=qualify,
        load_test=load_test,
    )


class _WorkerStatusReporter:
    """Worker-local, visibility-only snapshots kept outside checkpoint roots."""

    def __init__(self, spec: WorkerSpec) -> None:
        self._spec = spec
        self._started = monotonic()
        self._dataset: str | None = None
        self._dataset_index: int | None = None
        self._completed = 0

    def begin_dataset(self, dataset: str, dataset_index: int) -> None:
        self._dataset = dataset
        self._dataset_index = dataset_index
        self.emit("dataset_starting")

    def mark_checkpointed(self) -> None:
        self._completed += 1
        self.emit("checkpoint")

    def emit(
        self,
        phase: str,
        seed: int | None = None,
        fold: int | None = None,
        error: str = "",
    ) -> None:
        worker_status = WorkerStatus(
            shard_index=self._spec.shard.shard_index,
            worker_pid=os.getpid(),
            expected_datasets=len(self._spec.shard.datasets),
            completed_datasets=self._completed,
            phase=phase,
            dataset=self._dataset,
            dataset_index=self._dataset_index,
            seed=seed,
            fold=fold,
            error=error,
        )
        write_worker_status(self._spec.status_root, worker_status)
        append_runtime_log(
            self._spec.status_root,
            RuntimeLogRecord(
                run_id=self._spec.run_id,
                timestamp=datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z"),
                pid=worker_status.worker_pid,
                shard_index=worker_status.shard_index,
                current_dataset=worker_status.dataset,
                phase=worker_status.phase,
                seed=worker_status.seed,
                fold=worker_status.fold,
                completed_datasets=worker_status.completed_datasets,
                elapsed_seconds=monotonic() - self._started,
                status=(
                    "failed"
                    if phase == "worker_failed"
                    else "completed"
                    if phase == "worker_completed"
                    else "running"
                ),
                error=worker_status.error,
            ),
        )


def _status_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".replace("\n", " ")[:512]


def _run_dataset_checkpointed(
    dataset: str,
    context: object,
    checkpoints: object,
    runtime: object,
) -> object:
    from experiments.exp417_causal_multihorizon_shadow.runner import (
        run_dataset_checkpointed,
    )

    return run_dataset_checkpointed(
        dataset,
        context,
        checkpoints,
        runtime=runtime,
    )


def _validate_dataset_result(result: object, dataset: str) -> None:
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        DatasetResult,
    )

    if not isinstance(result, DatasetResult):
        raise ValueError("worker result must be an Exp417 DatasetResult")
    if result.dataset != dataset:
        raise ValueError(
            "worker result dataset is outside its assigned shard"
        )


def _validate_bootstrap_types(bootstrap: WorkerBootstrap) -> None:
    from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
        Exp417CheckpointProvenance,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(bootstrap.resources, RunnerResources):
        raise ValueError("worker resources must be RunnerResources")
    if not isinstance(bootstrap.provenance, Exp417CheckpointProvenance):
        raise ValueError(
            "worker provenance must be Exp417CheckpointProvenance"
        )
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenRunContext,
    )

    if not isinstance(bootstrap.context, FrozenRunContext):
        raise ValueError("worker context must be FrozenRunContext")


def _validate_benchmark_bootstrap_types(
    bootstrap: BenchmarkTrainBootstrap,
) -> None:
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(bootstrap.resources, RunnerResources):
        raise ValueError("benchmark worker resources must be RunnerResources")
    if type(bootstrap.context) is not BenchmarkTrainContext:
        raise ValueError("benchmark worker context must be BenchmarkTrainContext")
    if bootstrap.expected_context_sha256 != bootstrap.context.sha256:
        raise ValueError("benchmark worker context SHA-256 does not match bootstrap")


def _resources_payload(resources: object) -> dict[str, str]:
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(resources, RunnerResources):
        raise ValueError("worker resources must be RunnerResources")
    return {
        "database": str(resources.database),
        "exp291_prediction": str(resources.exp291_prediction),
        "exp291_manifest": str(resources.exp291_manifest),
        "exp292_prediction": str(resources.exp292_prediction),
        "exp292_manifest": str(resources.exp292_manifest),
        "backend_audit_path": str(resources.backend_audit_path),
    }


def _bootstrap_payload(
    bootstrap: WorkerBootstrap | BenchmarkTrainBootstrap,
) -> dict[str, object]:
    from experiments.exp417_causal_multihorizon_shadow import checkpoints

    if isinstance(bootstrap, WorkerBootstrap):
        _validate_bootstrap_types(bootstrap)
        return {
            "kind": "official",
            "resources": _resources_payload(bootstrap.resources),
            "provenance": bootstrap.provenance.to_payload(),
            "context": checkpoints._context_payload(bootstrap.context),
            "expected_context_sha256": bootstrap.expected_context_sha256,
        }
    if not isinstance(bootstrap, BenchmarkTrainBootstrap):
        raise ValueError("worker bootstrap is invalid")
    _validate_benchmark_bootstrap_types(bootstrap)
    resources = bootstrap.resources
    return {
        "kind": "benchmark_train",
        "resources": _resources_payload(resources),
        "context": benchmark_train_context_payload(bootstrap.context),
        "expected_context_sha256": bootstrap.expected_context_sha256,
    }


def _bootstrap_from_payload(
    payload: object,
) -> WorkerBootstrap | BenchmarkTrainBootstrap:
    from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
        Exp417CheckpointProvenance,
    )
    from experiments.exp417_causal_multihorizon_shadow import checkpoints
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(payload, dict) or not isinstance(payload.get("kind"), str):
        raise ValueError("worker bootstrap payload is invalid")
    resources_payload = payload["resources"]
    if not isinstance(resources_payload, dict) or set(resources_payload) != {
        "database",
        "exp291_prediction",
        "exp291_manifest",
        "exp292_prediction",
        "exp292_manifest",
        "backend_audit_path",
    }:
        raise ValueError("worker resources payload is invalid")
    resources = RunnerResources(
        database=Path(resources_payload["database"]),
        exp291_prediction=Path(resources_payload["exp291_prediction"]),
        exp291_manifest=Path(resources_payload["exp291_manifest"]),
        exp292_prediction=Path(resources_payload["exp292_prediction"]),
        exp292_manifest=Path(resources_payload["exp292_manifest"]),
        backend_audit_path=Path(resources_payload["backend_audit_path"]),
    )
    if payload["kind"] == "official":
        if set(payload) != {
            "kind",
            "resources",
            "provenance",
            "context",
            "expected_context_sha256",
        } or not isinstance(payload["provenance"], dict):
            raise ValueError("official worker bootstrap payload is invalid")
        return WorkerBootstrap(
            resources,
            Exp417CheckpointProvenance(**payload["provenance"]),
            payload["expected_context_sha256"],
            checkpoints._context_from_payload(payload["context"]),
        )
    if payload["kind"] == "benchmark_train":
        if set(payload) != {
            "kind",
            "resources",
            "context",
            "expected_context_sha256",
        }:
            raise ValueError("benchmark worker bootstrap payload is invalid")
        return BenchmarkTrainBootstrap(
            resources,
            benchmark_train_context_from_payload(payload["context"]),
            payload["expected_context_sha256"],
        )
    raise ValueError("worker bootstrap kind is invalid")


def _context_sha256(context: object) -> str:
    from experiments.exp403_408.artifact_io import canonical_json_bytes
    from experiments.exp417_causal_multihorizon_shadow import checkpoints
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenRunContext,
    )

    if not isinstance(context, FrozenRunContext):
        raise ValueError("worker context must be FrozenRunContext")
    payload = checkpoints._context_payload(context)
    return sha256(canonical_json_bytes(payload)).hexdigest()


def _validate_context_provenance(context: object, provenance: object) -> None:
    lanes = context.lanes
    expected = {
        "git_commit": context.code_commit,
        "database_sha256": context.database_sha256,
        "exp291_prediction_sha256": lanes.exp291_prediction_sha256,
        "exp291_manifest_sha256": lanes.exp291_manifest_sha256,
        "exp292_prediction_sha256": lanes.exp292_prediction_sha256,
        "exp292_manifest_sha256": lanes.exp292_manifest_sha256,
        "runtime_versions": dict(context.runtime_versions),
    }
    if any(getattr(provenance, name) != value for name, value in expected.items()):
        raise ValueError(
            "worker provenance does not match reconstructed frozen context"
        )


def _verify_official_worker_inputs(resources: object, context: object) -> None:
    """Verify immutable sources once before a session-backed official shard."""
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenRunContext,
    )
    from experiments.exp417_causal_multihorizon_shadow.inputs import (
        assert_label_free_sql_contract,
        verify_lane_input_snapshots,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(resources, RunnerResources):
        raise ValueError("worker resources must be RunnerResources")
    if not isinstance(context, FrozenRunContext):
        raise ValueError("worker context must be FrozenRunContext")
    assert_label_free_sql_contract()
    verify_lane_input_snapshots(
        resources.exp291_prediction,
        resources.exp291_manifest,
        resources.exp292_prediction,
        resources.exp292_manifest,
    )


def _verified_worker_connection(resources: object, context: object) -> object:
    """Open the one descriptor-pinned SQLite session for an official shard."""
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenRunContext,
    )
    from experiments.exp417_causal_multihorizon_shadow.inputs import (
        verified_read_only_connection,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(resources, RunnerResources):
        raise ValueError("worker resources must be RunnerResources")
    if not isinstance(context, FrozenRunContext):
        raise ValueError("worker context must be FrozenRunContext")
    return verified_read_only_connection(resources.database, context.database_sha256)


def _peak_rss_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise ValueError("worker maximum RSS is invalid")
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(maximum * multiplier)


def _completed_prefix(
    spec: WorkerSpec,
    record_digests: Mapping[str, str],
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    completed: list[str] = []
    for dataset in spec.shard.datasets:
        if dataset not in record_digests:
            break
        completed.append(dataset)
    copied = {
        dataset: record_digests[dataset]
        for dataset in completed
    }
    return tuple(completed), MappingProxyType(copied)


def run_worker(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
) -> WorkerCompletion:
    """Run one shard sequentially with a process-local Exp417 runtime and store."""
    if not isinstance(spec, WorkerSpec):
        raise ValueError("spec must be WorkerSpec")
    if not isinstance(bootstrap, WorkerBootstrap):
        raise ValueError("bootstrap must be WorkerBootstrap")

    started = monotonic()
    thread_environment: Mapping[str, str] = MappingProxyType({})
    status = _WorkerStatusReporter(spec)
    try:
        status.emit("worker_starting")
        thread_environment = pin_worker_threads()
    except Exception as error:
        try:
            status.emit("worker_failed", error=_status_error(error))
        except Exception as status_error:
            error = RuntimeError(
                f"worker status write failed: {_status_error(status_error)}"
            )
        return _failed_completion(
            spec,
            started,
            thread_environment,
            error,
            None,
        )
    return _run_worker_pinned(
        spec,
        bootstrap,
        started,
        thread_environment,
        status,
    )


def _run_worker_pinned(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
    started: float,
    thread_environment: Mapping[str, str],
    status: _WorkerStatusReporter,
) -> WorkerCompletion:
    checkpoint_store: object | None = None
    try:
        _validate_bootstrap_types(bootstrap)
        status.emit("context")
        context = bootstrap.context
        _validate_context_provenance(context, bootstrap.provenance)
        if _context_sha256(context) != bootstrap.expected_context_sha256:
            raise ValueError(
                "worker context SHA-256 does not match bootstrap"
            )
        if any(dataset not in context.datasets for dataset in spec.shard.datasets):
            raise ValueError("worker shard contains a dataset outside context")

        _verify_official_worker_inputs(bootstrap.resources, context)
        checkpoint_store = _open_checkpoint_store(
            spec,
            bootstrap.provenance,
        )
        with _verified_worker_connection(bootstrap.resources, context) as connection:
            runtime = _build_runtime(
                bootstrap.resources,
                seed_workers_for_outer_workers(spec.outer_worker_count),
                status.emit,
                connection=connection,
                context=context,
            )
            for index, dataset in enumerate(spec.shard.datasets):
                status.begin_dataset(dataset, index)
                result = _run_dataset_checkpointed(
                    dataset,
                    context,
                    checkpoint_store,
                    runtime,
                )
                _validate_dataset_result(result, dataset)
                observed = checkpoint_store.completed_record_digests()
                expected_prefix = spec.shard.datasets[: index + 1]
                if (
                    set(observed) != set(expected_prefix)
                    or any(name not in spec.shard.datasets for name in observed)
                ):
                    raise ValueError(
                        "worker checkpoint digests do not match assigned prefix"
                    )
                status.mark_checkpointed()

        digests = checkpoint_store.completed_record_digests()
        completed, frozen_digests = _completed_prefix(spec, digests)
        completion = WorkerCompletion(
            shard_index=spec.shard.shard_index,
            pid=os.getpid(),
            completed_datasets=completed,
            record_digests=frozen_digests,
            elapsed_seconds=monotonic() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            thread_environment=thread_environment,
            status="completed",
            error="",
        )
        status.emit("worker_completed")
        return completion
    except Exception as error:
        try:
            status.emit("worker_failed", error=_status_error(error))
        except Exception as status_error:
            error = RuntimeError(
                f"worker status write failed: {_status_error(status_error)}"
            )
        return _failed_completion(
            spec,
            started,
            thread_environment,
            error,
            checkpoint_store,
        )


def _failed_completion(
    spec: WorkerSpec,
    started: float,
    thread_environment: Mapping[str, str],
    error: Exception,
    checkpoint_store: object | None,
) -> WorkerCompletion:
    digests: Mapping[str, str] = {}
    if checkpoint_store is not None:
        try:
            digests = checkpoint_store.completed_record_digests()
        except Exception:
            digests = {}
    completed, frozen_digests = _completed_prefix(spec, digests)
    return WorkerCompletion(
        shard_index=spec.shard.shard_index,
        pid=os.getpid(),
        completed_datasets=completed,
        record_digests=frozen_digests,
        elapsed_seconds=monotonic() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        thread_environment=thread_environment,
        status="failed",
        error=f"{type(error).__name__}: {error}",
    )


def run_bounded_train_smoke_worker(
    spec: WorkerSpec,
    bootstrap: BenchmarkTrainBootstrap,
) -> WorkerCompletion:
    """Run only the preregistered TRAIN-only scheduler smoke for one shard.

    This does not create a checkpoint, prediction, score, candidate, metric,
    or TEST-value object.  Its record digests bind only the fixed smoke layout
    and dataset names so the benchmark cannot be mistaken for Exp417 evidence.
    """
    if not isinstance(spec, WorkerSpec):
        raise ValueError("spec must be WorkerSpec")
    if not isinstance(bootstrap, BenchmarkTrainBootstrap):
        raise ValueError("bootstrap must be BenchmarkTrainBootstrap")
    started = monotonic()
    thread_environment: Mapping[str, str] = MappingProxyType({})
    completed: list[str] = []
    digests: dict[str, str] = {}
    status = _WorkerStatusReporter(spec)
    try:
        status.emit("worker_starting")
        thread_environment = pin_worker_threads()
        _validate_benchmark_bootstrap_types(bootstrap)
        context = bootstrap.context
        status.emit("context")
        if context.sha256 != bootstrap.expected_context_sha256:
            raise ValueError(
                "benchmark worker context SHA-256 does not match bootstrap"
            )
        if any(dataset not in context.datasets for dataset in spec.shard.datasets):
            raise ValueError("worker shard contains a dataset outside context")

        from experiments.exp417_causal_multihorizon_shadow.inputs import (
            load_raw_train_values_from_connection,
            verified_read_only_connection,
        )
        from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
            preprocess_length_groups,
        )

        seed_workers = seed_workers_for_outer_workers(spec.outer_worker_count)
        with verified_read_only_connection(
            bootstrap.resources.database, context.database_sha256
        ) as connection:
            for index, dataset in enumerate(spec.shard.datasets):
                status.begin_dataset(dataset, index)
                status.emit("train_loading")
                train = load_raw_train_values_from_connection(
                    connection,
                    dataset,
                    database_sha256=context.database_sha256,
                    datasets=context.datasets,
                )
                status.emit("preprocessing")
                processed = tuple(preprocess_length_groups(train))
                status.emit("oof")
                _run_bounded_train_batches(
                    processed,
                    seed_workers,
                    FROZEN_CONFIG.bounded_train_batches_per_seed,
                )
                completed.append(dataset)
                digests[dataset] = _bounded_smoke_digest(
                    dataset,
                    spec.outer_worker_count,
                    seed_workers,
                    FROZEN_CONFIG.bounded_train_batches_per_seed,
                    bootstrap.expected_context_sha256,
                )
                status.mark_checkpointed()
        status.emit("worker_completed")
        return WorkerCompletion(
            shard_index=spec.shard.shard_index,
            pid=os.getpid(),
            completed_datasets=tuple(completed),
            record_digests=digests,
            elapsed_seconds=monotonic() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            thread_environment=thread_environment,
            status="completed",
            error="",
        )
    except Exception as error:
        try:
            status.emit("worker_failed", error=_status_error(error))
        except Exception as status_error:
            error = RuntimeError(
                f"worker status write failed: {_status_error(status_error)}"
            )
        return WorkerCompletion(
            shard_index=spec.shard.shard_index,
            pid=os.getpid(),
            completed_datasets=tuple(completed),
            record_digests=digests,
            elapsed_seconds=monotonic() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            thread_environment=thread_environment,
            status="failed",
            error=_status_error(error),
        )


def _run_bounded_train_batches(
    processed: Sequence[object], seed_workers: int, batches_per_seed: int
) -> None:
    """Run exactly ``batches_per_seed`` optimizer batches for every model group."""
    if type(batches_per_seed) is not int or batches_per_seed != 2:
        raise ValueError("bounded TRAIN smoke batch limit must be frozen at two")
    from experiments.exp417_causal_multihorizon_shadow.config import (
        FROZEN_CONFIG as EXP417_CONFIG,
    )
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        ProcessedGroup,
    )
    from experiments.exp417_causal_multihorizon_shadow.model import (
        _materialize_network_from_state,
        prepare_seed_initial_states,
    )
    from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
        build_horizons,
    )
    import numpy as np
    import torch
    from torch import nn

    groups = tuple(processed)
    if any(not isinstance(group, ProcessedGroup) for group in groups):
        raise ValueError("bounded TRAIN smoke requires processed groups")
    device = torch.device("cpu")
    for group in groups:
        if group.status not in {"candidate_eligible", "diagnostic_only"}:
            continue
        horizons = build_horizons(group.group.length)
        states = prepare_seed_initial_states(horizons, device)

        def fit(seed: int) -> None:
            state = states[seed]
            network = _materialize_network_from_state(state, horizons, device)
            network.train()
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
            loss_function = nn.HuberLoss()
            indices = tuple(sorted(group.channels))
            batch_size = min(32, len(indices))
            for batch_number in range(batches_per_seed):
                offset = (batch_number * batch_size) % len(indices)
                selected = tuple(
                    indices[(offset + position) % len(indices)]
                    for position in range(batch_size)
                )
                inputs = torch.from_numpy(
                    np.stack([group.channels[index] for index in selected]).copy()
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = network(inputs)
                loss = sum(
                    loss_function(
                        predictions[horizon][:, 0], inputs[:, 0, horizon:]
                    )
                    for horizon in horizons
                )
                if not torch.isfinite(loss):
                    raise ValueError("bounded TRAIN smoke loss is non-finite")
                loss.backward()
                optimizer.step()

        with ThreadPoolExecutor(max_workers=seed_workers) as pool:
            futures = tuple(
                pool.submit(fit, seed) for seed in EXP417_CONFIG.model_seeds
            )
            for future in futures:
                future.result()


def _bounded_smoke_digest(
    dataset: str,
    outer_workers: int,
    seed_workers: int,
    batches_per_seed: int,
    context_sha256: str,
) -> str:
    payload = (
        "exp419.bounded-train-smoke.v1",
        dataset,
        outer_workers,
        seed_workers,
        batches_per_seed,
        context_sha256,
    )
    return sha256(repr(payload).encode("utf-8")).hexdigest()


def _worker_process_entry(
    worker_function: Callable[
        [WorkerSpec, WorkerBootstrap | BenchmarkTrainBootstrap], WorkerCompletion
    ],
    spec: WorkerSpec,
    bootstrap_payload: object,
    result_connection: object,
) -> None:
    completion: WorkerCompletion | None = None
    started = monotonic()
    thread_environment: Mapping[str, str] = MappingProxyType({})
    status = _WorkerStatusReporter(spec)
    try:
        try:
            bootstrap = _bootstrap_from_payload(bootstrap_payload)
            if worker_function is run_worker:
                if not isinstance(bootstrap, WorkerBootstrap):
                    raise ValueError("official worker requires WorkerBootstrap")
                status.emit("worker_starting")
                thread_environment = pin_worker_threads()
                completion = _run_worker_pinned(
                    spec,
                    bootstrap,
                    started,
                    thread_environment,
                    status,
                )
            else:
                completion = worker_function(spec, bootstrap)
        except Exception as error:
            try:
                status.emit("worker_failed", error=_status_error(error))
            except Exception as status_error:
                error = RuntimeError(
                    f"worker status write failed: {_status_error(status_error)}"
                )
            completion = _failed_completion(
                spec,
                started,
                thread_environment,
                error,
                None,
            )
        if not isinstance(completion, WorkerCompletion):
            raise ValueError("worker process returned an invalid completion")
        result_connection.send(_completion_payload(completion))
    finally:
        result_connection.close()
    if completion.status == "failed":
        raise SystemExit(1)


def _completion_payload(completion: WorkerCompletion) -> dict[str, object]:
    return {
        "shard_index": completion.shard_index,
        "pid": completion.pid,
        "completed_datasets": completion.completed_datasets,
        "record_digests": dict(completion.record_digests),
        "elapsed_seconds": completion.elapsed_seconds,
        "peak_rss_bytes": completion.peak_rss_bytes,
        "thread_environment": dict(completion.thread_environment),
        "status": completion.status,
        "error": completion.error,
    }


def _completion_from_payload(payload: object) -> WorkerCompletion:
    if not isinstance(payload, dict):
        raise ValueError("worker completion payload must be a dictionary")
    expected = {
        "shard_index",
        "pid",
        "completed_datasets",
        "record_digests",
        "elapsed_seconds",
        "peak_rss_bytes",
        "thread_environment",
        "status",
        "error",
    }
    if set(payload) != expected:
        raise ValueError("worker completion payload fields are invalid")
    return WorkerCompletion(**payload)


def _validated_specs(specs: Sequence[WorkerSpec]) -> tuple[WorkerSpec, ...]:
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise ValueError("specs must be a sequence")
    frozen = tuple(specs)
    if not frozen or any(not isinstance(spec, WorkerSpec) for spec in frozen):
        raise ValueError("specs must contain WorkerSpec records")
    indices = tuple(spec.shard.shard_index for spec in frozen)
    if len(indices) != len(set(indices)):
        raise ValueError("worker shard indices must be unique")

    roots = tuple(
        (
            root.resolve(strict=False),
            _casefolded_path_parts(root),
            _filesystem_identity(root),
        )
        for spec in frozen
        for root in (spec.checkpoint_root, spec.status_root)
    )
    for index, (root, folded, identity) in enumerate(roots):
        for other, other_folded, other_identity in roots[index + 1 :]:
            if (
                root == other
                or root.is_relative_to(other)
                or other.is_relative_to(root)
                or _parts_are_nested(folded, other_folded)
                or (
                    identity is not None
                    and identity == other_identity
                )
            ):
                raise ValueError(
                    "worker checkpoint roots must be disjoint"
                )
    return frozen


def _casefolded_path_parts(path: Path) -> tuple[str, ...]:
    resolved = path.resolve(strict=False)
    return tuple(
        unicodedata.normalize("NFC", part).casefold()
        for part in resolved.parts
    )


def _filesystem_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _parts_are_nested(
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> bool:
    shortest = min(len(first), len(second))
    return (
        first[:shortest] == second[:shortest]
        and (
            len(first) == shortest
            or len(second) == shortest
        )
    )


def _terminate_and_reap(processes: Sequence[object]) -> None:
    started = tuple(
        process
        for process in processes
        if process.pid is not None
    )
    for process in started:
        if process.is_alive():
            process.terminate()

    terminate_deadline = monotonic() + _CLEANUP_TERMINATE_SECONDS
    for process in started:
        process.join(max(0.0, terminate_deadline - monotonic()))

    for process in started:
        if process.is_alive():
            process.kill()

    kill_deadline = monotonic() + _CLEANUP_KILL_SECONDS
    for process in started:
        process.join(max(0.0, kill_deadline - monotonic()))

    survivors = tuple(
        process.pid
        for process in started
        if process.is_alive()
    )
    if survivors:
        raise RuntimeError(
            f"worker cleanup could not reap processes: {survivors}"
        )


def _close_connections(receivers: Sequence[object]) -> None:
    for receiver in receivers:
        try:
            receiver.close()
        except OSError:
            pass


def _validate_completion_contract(
    spec: WorkerSpec,
    completion: WorkerCompletion,
) -> None:
    if dict(completion.thread_environment) != _THREAD_ENVIRONMENT:
        raise RuntimeError(
            f"worker shard {spec.shard.shard_index} thread environment "
            "does not match the pinned contract"
        )
    assigned = spec.shard.datasets
    completed = completion.completed_datasets
    if completion.status == "completed":
        if (
            completed != assigned
            or set(completion.record_digests) != set(assigned)
        ):
            raise RuntimeError(
                f"worker shard {spec.shard.shard_index} successful "
                "completion does not cover exact assigned datasets"
            )
    elif (
        completed != assigned[: len(completed)]
        or set(completion.record_digests) != set(completed)
    ):
        raise RuntimeError(
            f"worker shard {spec.shard.shard_index} failed completion "
            "is not an assigned prefix"
        )


def _drain_worker_messages(
    specs: tuple[WorkerSpec, ...],
    processes: Sequence[object],
    receivers: Sequence[object],
) -> tuple[WorkerCompletion, ...]:
    waitables: dict[object, tuple[str, int]] = {}
    for index, (process, receiver) in enumerate(
        zip(processes, receivers, strict=True)
    ):
        waitables[receiver] = ("pipe", index)
        waitables[process.sentinel] = ("sentinel", index)

    completions: list[WorkerCompletion | None] = [None] * len(specs)
    message_received = [False] * len(specs)
    pipe_closed = [False] * len(specs)
    process_exited = [False] * len(specs)

    while waitables:
        ready = wait_connections(tuple(waitables))
        for waitable in ready:
            kind, index = waitables.get(waitable, (None, None))
            if kind is None or index is None:
                raise RuntimeError("worker IPC returned an unknown waitable")
            if kind == "pipe":
                receiver = receivers[index]
                try:
                    payload = receiver.recv()
                except EOFError:
                    pipe_closed[index] = True
                    del waitables[waitable]
                else:
                    if message_received[index]:
                        raise RuntimeError(
                            f"worker shard {specs[index].shard.shard_index} "
                            "sent multiple completion messages"
                        )
                    message_received[index] = True
                    try:
                        completion = _completion_from_payload(payload)
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            f"worker shard "
                            f"{specs[index].shard.shard_index} sent an "
                            "invalid completion"
                        ) from error
                    _validate_completion_contract(specs[index], completion)
                    if (
                        completion.shard_index
                        != specs[index].shard.shard_index
                        or completion.pid != processes[index].pid
                    ):
                        raise RuntimeError(
                            f"worker shard "
                            f"{specs[index].shard.shard_index} completion "
                            "identity does not match its process"
                        )
                    completions[index] = completion
            else:
                processes[index].join()
                process_exited[index] = True
                del waitables[waitable]

        for index, spec in enumerate(specs):
            if (
                process_exited[index]
                and pipe_closed[index]
                and not message_received[index]
            ):
                raise RuntimeError(
                    f"worker shard {spec.shard.shard_index} exited with "
                    f"code {processes[index].exitcode} without a completion"
                )

    validated: list[WorkerCompletion] = []
    for spec, process, completion in zip(
        specs,
        processes,
        completions,
        strict=True,
    ):
        if completion is None:
            raise RuntimeError(
                f"worker shard {spec.shard.shard_index} has no completion"
            )
        if completion.status == "completed":
            if process.exitcode != 0:
                raise RuntimeError(
                    f"worker shard {spec.shard.shard_index} reported success "
                    f"but exited with code {process.exitcode}"
                )
        elif process.exitcode == 0:
            raise RuntimeError(
                f"worker shard {spec.shard.shard_index} reported failure "
                "but exited with code 0"
            )
        else:
            completion = replace(
                completion,
                error=(
                    f"{completion.error}; process exit code "
                    f"{process.exitcode}"
                ),
            )
        validated.append(completion)
    return tuple(sorted(validated, key=lambda item: item.shard_index))


def _spawn_workers_with(
    specs: Sequence[WorkerSpec],
    bootstrap: WorkerBootstrap | BenchmarkTrainBootstrap,
    worker_function: Callable[
        [WorkerSpec, WorkerBootstrap | BenchmarkTrainBootstrap], WorkerCompletion
    ],
) -> tuple[WorkerCompletion, ...]:
    """Spawn one isolated process and result pipe per typed worker function."""
    frozen_specs = _validated_specs(specs)
    if not isinstance(bootstrap, (WorkerBootstrap, BenchmarkTrainBootstrap)):
        raise ValueError("bootstrap must be an Exp419 worker bootstrap")
    if not callable(worker_function):
        raise ValueError("worker_function must be callable")

    context = multiprocessing.get_context("spawn")
    bootstrap_payload = _bootstrap_payload(bootstrap)
    owned_workers: list[tuple[object, object, object]] = []
    try:
        for spec in frozen_specs:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_worker_process_entry,
                args=(
                    worker_function,
                    spec,
                    bootstrap_payload,
                    sender,
                ),
            )
            owned_workers.append((process, receiver, sender))
            process.start()
            sender.close()
    except BaseException:
        _close_connections(
            tuple(sender for _, _, sender in owned_workers)
        )
        _close_connections(
            tuple(receiver for _, receiver, _ in owned_workers)
        )
        _terminate_and_reap(
            tuple(process for process, _, _ in owned_workers)
        )
        raise

    processes = tuple(
        process for process, _, _ in owned_workers
    )
    receivers = tuple(
        receiver for _, receiver, _ in owned_workers
    )
    try:
        return _drain_worker_messages(
            frozen_specs,
            processes,
            receivers,
        )
    except BaseException:
        _terminate_and_reap(processes)
        raise
    finally:
        _close_connections(
            tuple(sender for _, _, sender in owned_workers)
        )
        _close_connections(receivers)


def spawn_workers(
    specs: Sequence[WorkerSpec],
    bootstrap: WorkerBootstrap,
) -> tuple[WorkerCompletion, ...]:
    """Spawn one isolated process and result pipe per official worker shard."""
    return _spawn_workers_with(specs, bootstrap, run_worker)


def spawn_bounded_train_smoke_workers(
    specs: Sequence[WorkerSpec],
    bootstrap: BenchmarkTrainBootstrap,
    *,
    batches_per_seed: int,
) -> tuple[WorkerCompletion, ...]:
    """Spawn the non-official, two-batch TRAIN-only scheduler smoke."""
    if (
        type(batches_per_seed) is not int
        or batches_per_seed != FROZEN_CONFIG.bounded_train_batches_per_seed
    ):
        raise ValueError("bounded TRAIN smoke batch limit is not frozen")
    return _spawn_workers_with(specs, bootstrap, run_bounded_train_smoke_worker)


__all__ = [
    "pin_worker_threads",
    "run_bounded_train_smoke_worker",
    "run_worker",
    "spawn_bounded_train_smoke_workers",
    "spawn_workers",
]
