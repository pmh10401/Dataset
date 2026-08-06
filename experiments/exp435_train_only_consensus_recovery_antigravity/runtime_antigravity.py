"""Parent-owned resumable Exp435 runtime.

The database session remains in the parent.  Each dataset's normal TRAIN and
values-only TEST objects are decoded there, then pure records are sent to CPU
workers.  Workers never touch checkpoints, logs, SQLite descriptors, or
labels.  Results are committed in sorted dataset-name order regardless of
completion order.
"""

from __future__ import annotations

import os


# This module is imported by the parent and by spawned workers.  Keep this
# bootstrap immediately after ``os``: even the artifact/checkpoint imports
# below may transitively load NumPy.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _set_numerical_thread_environment() -> None:
    for name in _THREAD_ENV:
        os.environ[name] = "1"


_set_numerical_thread_environment()

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .checkpoint_antigravity import (
    Exp435CheckpointStore,
    context_fingerprint,
    context_payload,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    BackendBatchEvidence,
    BackendDecision,
    Exp435DatasetResult,
    Exp435RunSummary,
)
from .progress_antigravity import ProgressWriter, _validate_output_inventory


_PREFETCH = FROZEN_CONFIG.cpu_workers


@dataclass(frozen=True)
class LaneMembershipSnapshot:
    """Pickle-safe immutable copy of all four frozen lane coordinate sets."""

    hard: frozenset[tuple[str, int]]
    standard: frozenset[tuple[str, int]]
    priority: frozenset[tuple[str, int]]
    automatic_normal: frozenset[tuple[str, int]]

    def contains(self, dataset: str, instance_index: int) -> bool:
        coordinate = (dataset, instance_index)
        return any(coordinate in lane for lane in (
            self.hard, self.standard, self.priority, self.automatic_normal
        ))


def _lane_mapping(lanes: object, name: str) -> Mapping[str, object]:
    mapping = getattr(lanes, name, None)
    if mapping is None and isinstance(lanes, Mapping):
        mapping = lanes.get(name)
    if not isinstance(mapping, Mapping):
        raise ValueError(f"frozen lane {name} mapping is invalid")
    return mapping


def _lane_snapshot(
    lanes: object,
    datasets: tuple[str, ...],
    test_sizes: Mapping[str, int] | None = None,
) -> LaneMembershipSnapshot:
    """Validate and copy a real FrozenLaneSnapshot-shaped object."""

    dataset_set = set(datasets)
    sizes = test_sizes if test_sizes is not None else getattr(lanes, "test_sizes", None)
    if isinstance(sizes, Mapping):
        if set(sizes) != dataset_set or any(type(size) is not int or size <= 0 for size in sizes.values()):
            raise ValueError("frozen TEST sizes are invalid")
    frozen: dict[str, frozenset[tuple[str, int]]] = {}
    all_coordinates: set[tuple[str, int]] = set()
    for name in ("hard", "standard", "priority", "automatic_normal"):
        mapping = _lane_mapping(lanes, name)
        coordinates: set[tuple[str, int]] = set()
        for dataset, indices in mapping.items():
            if not isinstance(dataset, str) or dataset not in dataset_set:
                raise ValueError("frozen lane contains a dataset outside context")
            if not isinstance(indices, (tuple, list, frozenset)):
                raise ValueError("frozen lane coordinates are invalid")
            for index in indices:
                if type(index) is not int or index < 0:
                    raise ValueError("frozen lane coordinate is invalid")
                if isinstance(sizes, Mapping) and dataset in sizes and index >= int(sizes[dataset]):
                    raise ValueError("frozen lane coordinate exceeds TEST bounds")
                coordinate = (dataset, index)
                if coordinate in coordinates or coordinate in all_coordinates:
                    raise ValueError("frozen lanes overlap or contain duplicates")
                coordinates.add(coordinate)
        all_coordinates.update(coordinates)
        frozen[name] = frozenset(coordinates)
    return LaneMembershipSnapshot(**frozen)


def _worker_initializer() -> None:
    _set_numerical_thread_environment()


def _default_dataset_scorer(
    train: object,
    test: object,
    lanes: object,
    backend_decision: BackendDecision,
) -> Exp435DatasetResult:
    """Run the frozen CPU feature/calibration/scoring path in a worker."""

    if backend_decision.adopted:
        raise ValueError("adopted MLX backend requires the runtime MLX scorer")
    # Imports occur after the worker initializer has constrained numerical
    # libraries.  No TEST label or anomaly metadata is imported here.
    from .features_antigravity import fit_calibrated_group, prepare_train_groups
    from .scoring_antigravity import score_dataset_cpu

    groups = tuple(fit_calibrated_group(group) for group in prepare_train_groups(train))
    return score_dataset_cpu(groups, test, lanes)


def _default_mlx_dataset_scorer(
    train: object,
    test: object,
    lanes: object,
    backend_decision: BackendDecision,
    engine: object,
) -> Exp435DatasetResult:
    """Use one approved MLX engine for all channel scores in one process."""

    from .features_antigravity import fit_calibrated_group, prepare_train_groups
    from .mlx_antigravity import score_with_backend
    from .scoring_antigravity import score_dataset_backend

    groups = tuple(fit_calibrated_group(group) for group in prepare_train_groups(train))
    channel_scorer = lambda query, references, k: score_with_backend(  # noqa: E731
        backend_decision,
        query,
        references,
        k,
        engine=engine,
    )
    return score_dataset_backend(groups, test, lanes, channel_scorer)


def _score_worker(
    scorer: Callable[[object, object, object, BackendDecision], Exp435DatasetResult],
    dataset: str,
    train: object,
    test: object,
    lanes: object,
    backend_decision: BackendDecision,
) -> Exp435DatasetResult:
    result = scorer(train, test, lanes, backend_decision)
    if not isinstance(result, Exp435DatasetResult):
        raise ValueError("dataset scorer returned an invalid Exp435DatasetResult")
    if result.dataset != dataset:
        raise ValueError("dataset scorer returned a result for another dataset")
    return result


def _metadata(train: object, test: object) -> dict[str, object]:
    train_series = getattr(train, "series", ())
    test_series = getattr(test, "series", ())
    return {
        "train_size": len(train_series),
        "train_lengths": sorted({len(item.values) for item in train_series}),
        "test_size": len(test_series),
        "test_lengths": sorted({len(item.values) for item in test_series}),
    }


def _backend_or_cpu(value: BackendDecision | None) -> BackendDecision:
    if value is None:
        return BackendDecision.cpu_fallback("backend decision not supplied")
    if not isinstance(value, BackendDecision):
        raise ValueError("backend decision is invalid")
    return value


def _runtime_context(session: object, backend: BackendDecision) -> dict[str, object]:
    context = getattr(session, "context", None)
    if context is None:
        raise ValueError("input session has no verified context")
    projected = context_payload(context)
    projected["schema"] = "exp435.runtime-context.v1"
    projected["experiment_id"] = FROZEN_CONFIG.experiment_id
    projected["frozen_config"] = asdict(FROZEN_CONFIG)
    projected["backend_decision"] = backend.to_dict()
    manifest = projected.get("test_identity_manifest")
    if isinstance(manifest, Mapping):
        manifest_sizes = {
            dataset: len(entries)
            for dataset, entries in manifest.items()
            if isinstance(dataset, str) and isinstance(entries, list)
        }
        if set(manifest_sizes) != set(projected.get("datasets", ())):
            raise ValueError("verified context TEST identity manifest is incomplete")
        supplied_sizes = projected.get("test_sizes")
        if supplied_sizes is not None and supplied_sizes != manifest_sizes:
            raise ValueError("verified context TEST sizes disagree with identity manifest")
        projected["test_sizes"] = manifest_sizes
    elif "test_sizes" not in projected:
        raise ValueError("verified context lacks TEST identity manifest")
    datasets = projected.get("datasets")
    sizes = projected.get("test_sizes")
    if (
        not isinstance(datasets, list)
        or not isinstance(sizes, dict)
        or set(sizes) != set(datasets)
        or any(type(size) is not int or size <= 0 for size in sizes.values())
    ):
        raise ValueError("verified context TEST sizes do not cover the frozen population")
    supplied_identity = projected.get("run_identity")
    if supplied_identity is not None and supplied_identity != FROZEN_CONFIG.run_identity:
        raise ValueError("verified context run identity does not match frozen identity")
    # A run identity is part of the checkpoint context, never caller input or
    # a derivative of the mutable source commit.
    projected["run_identity"] = FROZEN_CONFIG.run_identity
    return projected


def _summary(
    store: Exp435CheckpointStore,
    backend: BackendDecision,
    *,
    status: str,
    failed: int = 0,
    publication_hashes: Mapping[str, str] | None = None,
) -> Exp435RunSummary:
    results = [store.load_record(dataset) for dataset in sorted(store.completed_digests())]
    recorded_failed = sum(result.status == "failed" for result in results)
    reported_failed = max(int(failed), recorded_failed)
    # ``failed`` may include the current terminal attempt that never reached
    # the checkpoint (for example a candidate-budget overflow).  Count that
    # attempt in the summary so successful + abstained + failed remains a
    # truthful conservation law without inventing a checkpoint record.
    completed = len(results) + max(0, reported_failed - recorded_failed)
    successful = sum(result.status == "scored" for result in results)
    abstained = sum(result.status == "abstained" for result in results)
    candidate_count = sum(result.candidate_count for result in results)
    hashes = {} if publication_hashes is None else dict(publication_hashes)
    return Exp435RunSummary(
        run_identity=store.run_identity,
        status=status,
        total_datasets=len(store.datasets),
        completed_datasets=completed,
        successful_datasets=successful,
        abstained_datasets=abstained,
        failed_datasets=reported_failed,
        candidate_count=candidate_count,
        backend=backend,
        prediction_sha256=hashes.get("prediction_sha256"),
        manifest_sha256=hashes.get("manifest_sha256"),
        artifact_sha256=hashes.get("artifact_sha256"),
    )


def _result_event(
    result: Exp435DatasetResult,
    digest: str,
    *,
    completed: int,
    total: int,
    metadata: Mapping[str, object],
    backend: BackendDecision,
) -> dict[str, object]:
    return {
        "kind": "dataset_complete",
        "dataset": result.dataset,
        "status": result.status,
        "record_sha256": digest,
        "backend": backend.backend,
        "train_size": metadata.get("train_size"),
        "train_lengths": metadata.get("train_lengths"),
        "test_size": result.test_size,
        "test_lengths": metadata.get("test_lengths"),
        "candidate_count": result.candidate_count,
        "completed": completed,
        "total": total,
    }


def _counts(store: Exp435CheckpointStore) -> tuple[int, int, int, int]:
    results = [store.load_record(dataset) for dataset in sorted(store.completed_digests())]
    return (
        len(results),
        sum(result.status == "scored" for result in results),
        sum(result.status == "abstained" for result in results),
        sum(result.status == "failed" for result in results),
    )


def _persist_failure(writer: ProgressWriter, error: BaseException) -> None:
    message = " ".join(str(error).split())[:1000]
    current_dataset = (
        writer.progress.get("current_dataset")
        if isinstance(writer.progress.get("current_dataset"), str)
        else None
    )
    stage = writer.failure_stage(current_dataset)
    if "candidate budget" in message.lower():
        writer.append_event({
            "kind": "candidate_budget_exceeded",
            "dataset": current_dataset,
            "error_type": type(error).__name__,
            "error_message": message,
            "observed": getattr(error, "observed", None),
            "budget": FROZEN_CONFIG.global_candidate_budget,
            "stage": stage,
        })
    else:
        writer.append_event({
            "kind": "failure",
            "dataset": current_dataset,
            "error_type": type(error).__name__,
            "error_message": message,
            "stage": stage,
        })
    writer.mark_failed(
        current_dataset,
        error,
    )


def _abort_executor(executor: ProcessPoolExecutor) -> None:
    """Cancel pending work and terminate running children after a hard failure."""

    processes = getattr(executor, "_processes", {})
    if isinstance(processes, Mapping):
        for process in tuple(processes.values()):
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def _reconcile_runtime(
    store: Exp435CheckpointStore,
    writer: ProgressWriter,
    backend: BackendDecision,
    *,
    metadata_by_dataset: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    accounting = store.recover_runtime_accounting(writer.output)
    if not accounting.missing_events:
        return
    metadata_by_dataset = {} if metadata_by_dataset is None else metadata_by_dataset
    completed, successful, abstained, failed = _counts(store)
    for dataset in accounting.missing_events:
        result = store.load_record(dataset)
        metadata = metadata_by_dataset.get(dataset, {"test_size": result.test_size})
        digest = store.completed_digests()[dataset]
        writer.append_event(
            _result_event(
                result,
                digest,
                completed=completed,
                total=len(store.datasets),
                metadata=metadata,
                backend=backend,
            )
        )
    writer.update(
        stage="idle",
        current_dataset=None,
        dataset_metadata=None,
        completed=completed,
        success=successful,
        abstention=abstained,
        error=failed,
        candidate_count=sum(store.load_record(dataset).candidate_count for dataset in store.completed_digests()),
        state="running",
    )


def _load_values(session: object, dataset: str) -> tuple[object, object, Mapping[str, object]]:
    loader_train = getattr(session, "load_train", None)
    loader_test = getattr(session, "load_test_values", None)
    if not callable(loader_train) or not callable(loader_test):
        raise ValueError("input session does not expose label-free TRAIN/TEST values loaders")
    train = loader_train(dataset)
    # The Exp435 input session checks the concrete RawTrainDataset and delegates
    # the descriptor-bound values-only TEST path only after this call.
    test = loader_test(dataset, train)
    return train, test, _metadata(train, test)


def _session_lanes(session: object) -> object:
    context = getattr(session, "context", None)
    lanes = getattr(context, "lanes", None)
    if lanes is not None:
        return lanes
    if isinstance(context, Mapping):
        return context.get("lanes")
    return getattr(session, "lanes", None)


def _run(
    *,
    run_root: Path,
    checkpoint_root: Path,
    input_session: object,
    backend_decision: BackendDecision,
    dataset_scorer: Callable[[object, object, object, BackendDecision], Exp435DatasetResult] | None,
    worker_count: int | None,
    resume: bool,
) -> Exp435RunSummary:
    _set_numerical_thread_environment()
    backend = _backend_or_cpu(backend_decision)
    context = _runtime_context(input_session, backend)
    datasets = tuple(context["datasets"])
    if not datasets:
        raise ValueError("verified context dataset population is empty")
    if worker_count is None:
        worker_count = 1 if backend.adopted else FROZEN_CONFIG.cpu_workers
    if type(worker_count) is not int or worker_count not in {1, FROZEN_CONFIG.cpu_workers}:
        raise ValueError("worker_count must be 1 or the frozen CPU worker count 7")
    if backend.adopted and worker_count != 1:
        raise ValueError("MLX backend must use one process without an outer pool")
    lane_snapshot = _lane_snapshot(
        context.get("lanes"),
        datasets,
        context.get("test_sizes") if isinstance(context.get("test_sizes"), Mapping) else None,
    )
    using_default_scorer = dataset_scorer is None
    if dataset_scorer is None:
        dataset_scorer = _default_dataset_scorer
    if not callable(dataset_scorer):
        raise ValueError("dataset_scorer is invalid")

    if resume:
        # This complete read-only output gate includes failed-state rejection
        # and must run before checkpoint recovery can reconcile crash leftovers.
        ProgressWriter.preflight_existing(
            run_root,
            run_identity=FROZEN_CONFIG.run_identity,
            checkpoint=checkpoint_root,
            backend=backend.backend,
            total_datasets=len(datasets),
            context_fingerprint=context_fingerprint(context),
        )
        store = Exp435CheckpointStore.open_existing(checkpoint_root, context)
        writer = ProgressWriter.open_existing(
            run_root,
            run_identity=store.run_identity,
            checkpoint=checkpoint_root,
            backend=backend.backend,
            total_datasets=len(datasets),
            context_fingerprint=store.context_fingerprint,
        )
    else:
        if Path(run_root).exists() or Path(checkpoint_root).exists():
            raise ValueError("new runtime roots must both be absent")
        store = Exp435CheckpointStore.create(checkpoint_root, context)
        writer = ProgressWriter.create(
            run_root,
            run_identity=store.run_identity,
            checkpoint=checkpoint_root,
            total_datasets=len(datasets),
            backend=backend.backend,
            context_fingerprint=store.context_fingerprint,
        )
    if store.run_identity != FROZEN_CONFIG.run_identity:
        raise ValueError("checkpoint run identity does not match frozen identity")
    failure_persisted = False
    try:
        mlx_engine: object | None = None
        if backend.adopted and using_default_scorer:
            from .mlx_antigravity import MlxEngine

            mlx_engine = MlxEngine()
        _reconcile_runtime(store, writer, backend)
        completed_datasets = set(store.completed_digests())
        missing = [dataset for dataset in datasets if dataset not in completed_datasets]
        completed, successful, abstained, failed = _counts(store)
        candidate_count = sum(store.load_record(dataset).candidate_count for dataset in store.completed_digests())
        if not missing:
            state = "checkpoint_complete" if len(datasets) == FROZEN_CONFIG.expected_datasets else "running"
            if state == "checkpoint_complete":
                writer.mark_complete(candidate_count=candidate_count)
            return _summary(store, backend, status=state, failed=failed)

        writer.start_heartbeat()
        writer.transition(
            "running",
            stage="idle",
            current_dataset=None,
            dataset_metadata=None,
            completed=completed,
            success=successful,
            abstention=abstained,
            error=failed,
            candidate_count=candidate_count,
        )

        # MLX is explicitly serial.  CPU serial mode is useful for the
        # deterministic identity fixture; production defaults to seven.
        if backend.adopted or worker_count == 1:
            ready: dict[str, tuple[Exp435DatasetResult, Mapping[str, object]]] = {}
            for dataset in missing:
                writer.update(
                    stage="loading",
                    current_dataset=dataset,
                    dataset_metadata=None,
                    completed=completed,
                    success=successful,
                    abstention=abstained,
                    error=failed,
                    candidate_count=candidate_count,
                )
                train, test, metadata = _load_values(input_session, dataset)
                writer.update(
                    stage="dispatching",
                    current_dataset=dataset,
                    dataset_metadata=metadata,
                    completed=completed,
                    success=successful,
                    abstention=abstained,
                    error=failed,
                    candidate_count=candidate_count,
                )
                if backend.adopted and using_default_scorer:
                    assert mlx_engine is not None
                    result = _score_worker(
                        lambda train_item, test_item, lane_item, decision: _default_mlx_dataset_scorer(
                            train_item, test_item, lane_item, decision, mlx_engine
                        ),
                        dataset,
                        train,
                        test,
                        lane_snapshot,
                        backend,
                    )
                else:
                    result = _score_worker(dataset_scorer, dataset, train, test, lane_snapshot, backend)
                ready[dataset] = (result, metadata)
                # Serial path is already ordered because ``missing`` follows
                # the frozen dataset order.
                digest = store.commit(result)
                completed += 1
                successful += result.status == "scored"
                abstained += result.status == "abstained"
                failed += result.status == "failed"
                candidate_count += result.candidate_count
                writer.append_event(_result_event(result, digest, completed=completed, total=len(datasets), metadata=metadata, backend=backend))
                writer.update(stage="idle", current_dataset=None, dataset_metadata=None, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
        else:
            # At most one submitted/in-flight task per CPU worker.  Only the
            # parent loads values and writes durable files; commits are still
            # held for deterministic dataset-name order.
            lanes = lane_snapshot
            executor: ProcessPoolExecutor | None = ProcessPoolExecutor(
                max_workers=FROZEN_CONFIG.cpu_workers,
                initializer=_worker_initializer,
            )
            pending: dict[Future[Exp435DatasetResult], tuple[str, Mapping[str, object]]] = {}
            completed_results: dict[str, tuple[Exp435DatasetResult, Mapping[str, object]]] = {}
            submit_index = 0
            commit_index = 0
            try:
                while submit_index < len(missing) or pending:
                    while submit_index < len(missing) and len(pending) < _PREFETCH:
                        dataset = missing[submit_index]
                        writer.update(
                            stage="loading",
                            current_dataset=dataset,
                            dataset_metadata=None,
                            completed=completed,
                            success=successful,
                            abstention=abstained,
                            error=failed,
                            candidate_count=candidate_count,
                        )
                        train, test, metadata = _load_values(input_session, dataset)
                        writer.update(stage="dispatching", current_dataset=dataset, dataset_metadata=metadata, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
                        assert executor is not None
                        future = executor.submit(_score_worker, dataset_scorer, dataset, train, test, lanes, backend)
                        pending[future] = (dataset, metadata)
                        writer.update(stage="awaiting_worker_result", current_dataset=dataset, dataset_metadata=metadata, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
                        submit_index += 1
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        dataset, metadata = pending.pop(future)
                        try:
                            completed_results[dataset] = (future.result(), metadata)
                        except BaseException:
                            # Preserve the actual failing dataset before the
                            # outer handler persists its durable error event.
                            writer.update(
                                stage="awaiting_worker_result",
                                current_dataset=dataset,
                                dataset_metadata=metadata,
                                completed=completed,
                                success=successful,
                                abstention=abstained,
                                error=failed,
                                candidate_count=candidate_count,
                            )
                            raise
                    if pending:
                        active_future = next(iter(pending))
                        active_dataset, active_metadata = pending[active_future]
                        writer.update(
                            stage="awaiting_worker_result",
                            current_dataset=active_dataset,
                            dataset_metadata=active_metadata,
                            completed=completed,
                            success=successful,
                            abstention=abstained,
                            error=failed,
                            candidate_count=candidate_count,
                        )
                    while commit_index < len(missing) and missing[commit_index] in completed_results:
                        dataset = missing[commit_index]
                        result, metadata = completed_results.pop(dataset)
                        writer.update(
                            stage="committing",
                            current_dataset=dataset,
                            dataset_metadata=metadata,
                            completed=completed,
                            success=successful,
                            abstention=abstained,
                            error=failed,
                            candidate_count=candidate_count,
                        )
                        digest = store.commit(result)
                        completed += 1
                        successful += result.status == "scored"
                        abstained += result.status == "abstained"
                        failed += result.status == "failed"
                        candidate_count += result.candidate_count
                        writer.append_event(_result_event(result, digest, completed=completed, total=len(datasets), metadata=metadata, backend=backend))
                        commit_index += 1
                    if pending:
                        active_future = next(iter(pending))
                        active_dataset, active_metadata = pending[active_future]
                        writer.update(
                            stage="awaiting_worker_result",
                            current_dataset=active_dataset,
                            dataset_metadata=active_metadata,
                            completed=completed,
                            success=successful,
                            abstention=abstained,
                            error=failed,
                            candidate_count=candidate_count,
                        )
                    else:
                        writer.update(
                            stage="idle",
                            current_dataset=None,
                            dataset_metadata=None,
                            completed=completed,
                            success=successful,
                            abstention=abstained,
                            error=failed,
                            candidate_count=candidate_count,
                        )
            except KeyboardInterrupt as error:
                writer.mark_interrupted(
                    writer.progress.get("current_dataset")
                    if isinstance(writer.progress.get("current_dataset"), str)
                    else None,
                    error,
                )
                failure_persisted = True
                for future in pending:
                    future.cancel()
                assert executor is not None
                _abort_executor(executor)
                executor = None
                raise
            except BaseException as error:
                # Persist evidence while the parent is still responsive, then
                # cancel work that has not started and avoid waiting for a slow
                # sibling process before surfacing the failure.
                _persist_failure(writer, error)
                failure_persisted = True
                for future in pending:
                    future.cancel()
                assert executor is not None
                _abort_executor(executor)
                executor = None
                raise
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        if len(store.completed_digests()) != len(datasets) or failed:
            raise ValueError("runtime population is incomplete")
        final_status = "checkpoint_complete" if len(datasets) == FROZEN_CONFIG.expected_datasets else "running"
        if final_status == "checkpoint_complete":
            writer.mark_complete(candidate_count=candidate_count)
        return _summary(store, backend, status=final_status, failed=failed)
    except KeyboardInterrupt as error:
        if not failure_persisted:
            writer.mark_interrupted(
                writer.progress.get("current_dataset")
                if isinstance(writer.progress.get("current_dataset"), str)
                else None,
                error,
            )
            failure_persisted = True
        raise
    except BaseException as error:
        # Candidate-budget overflow is persisted as an immutable event before
        # the exception escapes; no publication artifact is created here.
        try:
            if not failure_persisted:
                _persist_failure(writer, error)
        finally:
            raise
    finally:
        writer.stop_heartbeat()
        writer.close()


def run_exp435_new(
    *,
    run_root: Path,
    checkpoint_root: Path,
    input_session: object,
    backend_decision: BackendDecision,
    dataset_scorer: Callable[[object, object, object, BackendDecision], Exp435DatasetResult] | None = None,
    worker_count: int | None = None,
) -> Exp435RunSummary:
    return _run(
        run_root=Path(run_root),
        checkpoint_root=Path(checkpoint_root),
        input_session=input_session,
        backend_decision=backend_decision,
        dataset_scorer=dataset_scorer,
        worker_count=worker_count,
        resume=False,
    )


def resume_exp435(
    *,
    run_root: Path,
    checkpoint_root: Path,
    input_session: object,
    backend_decision: BackendDecision,
    dataset_scorer: Callable[[object, object, object, BackendDecision], Exp435DatasetResult] | None = None,
    worker_count: int | None = None,
) -> Exp435RunSummary:
    return _run(
        run_root=Path(run_root),
        checkpoint_root=Path(checkpoint_root),
        input_session=input_session,
        backend_decision=backend_decision,
        dataset_scorer=dataset_scorer,
        worker_count=worker_count,
        resume=True,
    )


def _backend_from_payload(payload: object) -> BackendDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("stored backend decision is absent")
    raw = dict(payload)
    timings_raw = raw.get("timings", ())
    evidence_raw = raw.get("batch_evidence", ())
    if not isinstance(timings_raw, list) or not isinstance(evidence_raw, list):
        raise ValueError("stored backend timing evidence is invalid")
    try:
        raw["timings"] = tuple(tuple(item) for item in timings_raw)
        raw["batch_evidence"] = tuple(
            BackendBatchEvidence(**dict(item)) for item in evidence_raw
        )
        return BackendDecision(**{
            key: raw[key]
            for key in (
                "backend", "adopted", "smoke_seed", "finite_outputs",
                "candidate_identity_match", "parity_passed", "normalized_error_max",
                "absolute_error_max", "speedup", "batch_size", "selected_batch_size",
                "cpu_median_seconds", "mlx_median_seconds", "mlx_version",
                "mlx_metal_version", "device", "device_info", "metal_available",
                "timings", "batch_evidence", "reason",
            )
            if key in raw
        })
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("stored backend decision is invalid") from error


def _inspect_progress_accounting(
    store: Exp435CheckpointStore,
    progress: Mapping[str, object],
) -> tuple[str, int]:
    """Bind progress counters to durable records and preserve terminal state."""

    records = [store.load_record(dataset) for dataset in sorted(store.completed_digests())]
    recorded_completed = len(records)
    recorded_success = sum(result.status == "scored" for result in records)
    recorded_abstention = sum(result.status == "abstained" for result in records)
    recorded_failed = sum(result.status == "failed" for result in records)
    recorded_candidates = sum(result.candidate_count for result in records)

    integer_fields = ("completed", "completed_datasets", "success", "abstention", "error", "candidate_count")
    if any(type(progress.get(field)) is not int for field in integer_fields):
        raise ValueError("runtime progress accounting is invalid")
    completed = int(progress["completed"])
    if progress["completed_datasets"] != completed or completed > len(store.datasets):
        raise ValueError("runtime progress completed count is invalid")
    success = int(progress["success"])
    abstention = int(progress["abstention"])
    errors = int(progress["error"])
    candidates = int(progress["candidate_count"])
    if any(value < 0 for value in (completed, success, abstention, errors, candidates)):
        raise ValueError("runtime progress accounting is invalid")
    if success + abstention + errors != completed:
        raise ValueError("runtime progress count conservation is invalid")
    if success != recorded_success or abstention != recorded_abstention or candidates != recorded_candidates:
        raise ValueError("runtime progress does not match checkpoint records")
    terminal_failed = errors - recorded_failed
    if terminal_failed < 0 or completed != recorded_completed + terminal_failed:
        raise ValueError("runtime progress failure accounting does not match checkpoint records")

    state = str(progress.get("state", "running"))
    if state not in {
        "initializing", "running", "interrupted", "failed",
        "checkpoint_complete", "complete", "complete_review_only",
    }:
        state = "running"
    if state == "failed" and errors <= 0:
        raise ValueError("failed runtime progress has no failed dataset")
    if state in {"checkpoint_complete", "complete", "complete_review_only"}:
        if completed != len(store.datasets) or errors != 0:
            raise ValueError("complete runtime progress is inconsistent with checkpoint")
    return state, recorded_failed + terminal_failed


def inspect_exp435(
    *,
    run_root: Path,
    checkpoint_root: Path,
    expected_context: object,
) -> Exp435RunSummary:
    """Inspect only canonical files; this function opens neither DB nor workers."""

    expected_base = context_payload(expected_context)
    if expected_base.get("run_identity") != FROZEN_CONFIG.run_identity:
        raise ValueError("expected context run identity does not match frozen identity")
    stored_context = Exp435CheckpointStore.read_context_projection(checkpoint_root)
    if stored_context.get("run_identity") != FROZEN_CONFIG.run_identity:
        raise ValueError("stored context run identity does not match frozen identity")
    backend = _backend_from_payload(stored_context.get("backend_decision"))
    expected_runtime = _runtime_context(type("_Session", (), {"context": expected_base})(), backend)
    if dict(stored_context) != expected_runtime:
        raise ValueError("context fingerprint does not match verified context")
    store = Exp435CheckpointStore.open_read_only(
        checkpoint_root,
        expected_run_identity=FROZEN_CONFIG.run_identity,
        expected_context=expected_runtime,
    )
    output = Path(run_root)
    _validate_output_inventory(output, mutable=False)
    progress_path = output / "progress_antigravity.json"
    progress_raw = progress_path.read_bytes()
    progress = json.loads(progress_raw)
    if (
        not isinstance(progress, dict)
        or canonical_json_bytes(progress) != progress_raw
        or progress.get("run_identity") != store.run_identity
    ):
        raise ValueError("runtime progress is not canonical")
    from .progress_antigravity import (
        _assert_label_free,
        _canonical_checkpoint,
        _require_persisted_stage,
        _validate_binding,
        _validate_events,
    )

    _assert_label_free(progress)
    _require_persisted_stage(progress.get("stage"), source="runtime progress stage")
    expected_checkpoint = _canonical_checkpoint(checkpoint_root)
    _validate_binding(
        progress,
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(store.datasets),
        run_identity=store.run_identity,
        context_fingerprint=store.context_fingerprint,
        source="runtime progress",
        require_total=True,
    )
    _validate_events(
        output / "events_antigravity.jsonl",
        store.run_identity,
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(store.datasets),
        context_fingerprint=store.context_fingerprint,
    )
    pid_raw = (output / "run_antigravity.pid").read_bytes()
    pid = json.loads(pid_raw)
    if not isinstance(pid, dict) or canonical_json_bytes(pid) != pid_raw:
        raise ValueError("runtime PID is not canonical")
    _validate_binding(
        pid,
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(store.datasets),
        run_identity=store.run_identity,
        context_fingerprint=store.context_fingerprint,
        source="runtime PID",
    )
    status, failed = _inspect_progress_accounting(store, progress)
    return _summary(store, backend, status=status, failed=failed)


__all__ = ["inspect_exp435", "resume_exp435", "run_exp435_new"]
