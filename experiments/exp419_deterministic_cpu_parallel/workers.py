"""Isolated spawned-worker execution for Exp419."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import multiprocessing
import os
from pathlib import Path
import resource
import sys
from time import monotonic
from types import MappingProxyType

from .contracts import WorkerBootstrap, WorkerCompletion, WorkerSpec


_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


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

    roots = tuple(spec.checkpoint_root.resolve(strict=False) for spec in frozen)
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if (
                root == other
                or root.is_relative_to(other)
                or other.is_relative_to(root)
            ):
                raise ValueError(
                    "worker checkpoint roots must be disjoint"
                )
    return frozen


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
    processes: list[object] = []
    receivers: list[object] = []
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
            try:
                process.start()
            except BaseException:
                receiver.close()
                sender.close()
                raise
            sender.close()
            receivers.append(receiver)
            processes.append(process)
    except BaseException:
        for process in processes:
            process.join()
        for receiver in receivers:
            receiver.close()
        raise

    for process in processes:
        process.join()

    completions: list[WorkerCompletion] = []
    protocol_errors: list[str] = []
    for spec, process, receiver in zip(
        frozen_specs,
        processes,
        receivers,
        strict=True,
    ):
        completion: object | None = None
        try:
            try:
                completion = receiver.recv()
            except EOFError:
                protocol_errors.append(
                    f"worker shard {spec.shard.shard_index} exited with "
                    f"code {process.exitcode} without a completion"
                )
                continue
            try:
                extra = receiver.recv()
            except EOFError:
                pass
            else:
                del extra
                protocol_errors.append(
                    f"worker shard {spec.shard.shard_index} sent multiple "
                    "completion messages"
                )
                continue
        finally:
            receiver.close()

        try:
            completion = _completion_from_payload(completion)
        except (TypeError, ValueError):
            protocol_errors.append(
                f"worker shard {spec.shard.shard_index} sent an invalid "
                "completion"
            )
            continue
        if (
            completion.shard_index != spec.shard.shard_index
            or completion.pid != process.pid
        ):
            protocol_errors.append(
                f"worker shard {spec.shard.shard_index} completion identity "
                "does not match its process"
            )
            continue
        if completion.status == "completed":
            if process.exitcode != 0:
                protocol_errors.append(
                    f"worker shard {spec.shard.shard_index} reported success "
                    f"but exited with code {process.exitcode}"
                )
                continue
        elif process.exitcode == 0:
            protocol_errors.append(
                f"worker shard {spec.shard.shard_index} reported failure "
                "but exited with code 0"
            )
            continue
        else:
            completion = replace(
                completion,
                error=(
                    f"{completion.error}; process exit code "
                    f"{process.exitcode}"
                ),
            )
        completions.append(completion)

    if protocol_errors:
        raise RuntimeError("; ".join(protocol_errors))
    return tuple(sorted(completions, key=lambda item: item.shard_index))


__all__ = ["pin_worker_threads", "run_worker", "spawn_workers"]
