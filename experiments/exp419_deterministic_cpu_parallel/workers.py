"""Isolated spawned-worker execution for Exp419."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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

from .contracts import WorkerBootstrap, WorkerCompletion, WorkerSpec


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
        spec.shard.datasets,
    )


def _build_runtime(resources: object) -> object:
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerRuntime

    return RunnerRuntime.production(resources)


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


def _bootstrap_payload(bootstrap: WorkerBootstrap) -> dict[str, object]:
    _validate_bootstrap_types(bootstrap)
    resources = bootstrap.resources
    return {
        "resources": {
            "database": str(resources.database),
            "exp291_prediction": str(resources.exp291_prediction),
            "exp291_manifest": str(resources.exp291_manifest),
            "exp292_prediction": str(resources.exp292_prediction),
            "exp292_manifest": str(resources.exp292_manifest),
            "backend_audit_path": str(resources.backend_audit_path),
        },
        "provenance": bootstrap.provenance.to_payload(),
        "expected_context_sha256": bootstrap.expected_context_sha256,
    }


def _bootstrap_from_payload(payload: object) -> WorkerBootstrap:
    from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
        Exp417CheckpointProvenance,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources

    if not isinstance(payload, dict) or set(payload) != {
        "resources",
        "provenance",
        "expected_context_sha256",
    }:
        raise ValueError("worker bootstrap payload is invalid")
    resources_payload = payload["resources"]
    provenance_payload = payload["provenance"]
    if not isinstance(resources_payload, dict) or set(resources_payload) != {
        "database",
        "exp291_prediction",
        "exp291_manifest",
        "exp292_prediction",
        "exp292_manifest",
        "backend_audit_path",
    }:
        raise ValueError("worker resources payload is invalid")
    if not isinstance(provenance_payload, dict):
        raise ValueError("worker provenance payload is invalid")
    resources = RunnerResources(
        database=Path(resources_payload["database"]),
        exp291_prediction=Path(resources_payload["exp291_prediction"]),
        exp291_manifest=Path(resources_payload["exp291_manifest"]),
        exp292_prediction=Path(resources_payload["exp292_prediction"]),
        exp292_manifest=Path(resources_payload["exp292_manifest"]),
        backend_audit_path=Path(resources_payload["backend_audit_path"]),
    )
    provenance = Exp417CheckpointProvenance(**provenance_payload)
    return WorkerBootstrap(
        resources,
        provenance,
        payload["expected_context_sha256"],
    )


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
    try:
        thread_environment = pin_worker_threads()
    except Exception as error:
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
    )


def _run_worker_pinned(
    spec: WorkerSpec,
    bootstrap: WorkerBootstrap,
    started: float,
    thread_environment: Mapping[str, str],
) -> WorkerCompletion:
    checkpoint_store: object | None = None
    try:
        _validate_bootstrap_types(bootstrap)
        context = _freeze_context(bootstrap.resources, bootstrap.provenance)
        _validate_context_provenance(context, bootstrap.provenance)
        if _context_sha256(context) != bootstrap.expected_context_sha256:
            raise ValueError(
                "reconstructed worker context SHA-256 does not match bootstrap"
            )
        if any(dataset not in context.datasets for dataset in spec.shard.datasets):
            raise ValueError("worker shard contains a dataset outside context")

        checkpoint_store = _open_checkpoint_store(
            spec,
            bootstrap.provenance,
        )
        runtime = _build_runtime(bootstrap.resources)
        for index, dataset in enumerate(spec.shard.datasets):
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

        digests = checkpoint_store.completed_record_digests()
        completed, frozen_digests = _completed_prefix(spec, digests)
        return WorkerCompletion(
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
    except Exception as error:
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


def _worker_process_entry(
    worker_function: Callable[[WorkerSpec, WorkerBootstrap], WorkerCompletion],
    spec: WorkerSpec,
    bootstrap_payload: object,
    result_connection: object,
) -> None:
    completion: WorkerCompletion | None = None
    started = monotonic()
    thread_environment: Mapping[str, str] = MappingProxyType({})
    try:
        try:
            thread_environment = pin_worker_threads()
            bootstrap = _bootstrap_from_payload(bootstrap_payload)
            if worker_function is run_worker:
                completion = _run_worker_pinned(
                    spec,
                    bootstrap,
                    started,
                    thread_environment,
                )
            else:
                completion = worker_function(spec, bootstrap)
        except Exception as error:
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
            spec.checkpoint_root.resolve(strict=False),
            _casefolded_path_parts(spec.checkpoint_root),
            _filesystem_identity(spec.checkpoint_root),
        )
        for spec in frozen
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


def spawn_workers(
    specs: Sequence[WorkerSpec],
    bootstrap: WorkerBootstrap,
) -> tuple[WorkerCompletion, ...]:
    """Spawn one isolated process and result pipe per worker shard."""
    frozen_specs = _validated_specs(specs)
    if not isinstance(bootstrap, WorkerBootstrap):
        raise ValueError("bootstrap must be WorkerBootstrap")

    context = multiprocessing.get_context("spawn")
    worker_function = run_worker
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


__all__ = ["pin_worker_threads", "run_worker", "spawn_workers"]
