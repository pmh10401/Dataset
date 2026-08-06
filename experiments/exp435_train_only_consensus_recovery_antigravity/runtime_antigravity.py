"""Parent-owned resumable Exp435 runtime.

The database session remains in the parent.  Each dataset's normal TRAIN and
values-only TEST objects are decoded there, then pure records are sent to CPU
workers.  Workers never touch checkpoints, logs, SQLite descriptors, or
labels.  Results are committed in sorted dataset-name order regardless of
completion order.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict
import os
from pathlib import Path
from typing import Callable, Mapping

from .checkpoint_antigravity import (
    Exp435CheckpointStore,
    context_payload,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    BackendDecision,
    Exp435DatasetResult,
    Exp435RunSummary,
)
from .progress_antigravity import ProgressWriter


_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
_PREFETCH = 14


def _set_numerical_thread_environment() -> None:
    for name in _THREAD_ENV:
        os.environ[name] = "1"


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
        raise ValueError("adopted MLX backend requires a single-process scorer")
    # Imports occur after the worker initializer has constrained numerical
    # libraries.  No TEST label or anomaly metadata is imported here.
    from .features_antigravity import fit_calibrated_group, prepare_train_groups
    from .scoring_antigravity import score_dataset_cpu

    groups = tuple(fit_calibrated_group(group) for group in prepare_train_groups(train))
    return score_dataset_cpu(groups, test, lanes)


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
    # A run identity is part of the checkpoint context, never caller input.
    code_commit = projected.get("code_commit")
    if isinstance(code_commit, str) and code_commit:
        projected["run_identity"] = f"exp435-{code_commit[:12]}"
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
    completed = len(results)
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
        failed_datasets=failed,
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
        stage="recovered",
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
    if dataset_scorer is None:
        dataset_scorer = _default_dataset_scorer
    if not callable(dataset_scorer):
        raise ValueError("dataset_scorer is invalid")

    if resume:
        store = Exp435CheckpointStore.open_existing(checkpoint_root, context)
        writer = ProgressWriter.open_existing(
            run_root,
            run_identity=store.run_identity,
            checkpoint=checkpoint_root,
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
        )
    try:
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
            stage="scoring",
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
                train, test, metadata = _load_values(input_session, dataset)
                writer.update(
                    stage="scoring",
                    current_dataset=dataset,
                    dataset_metadata=metadata,
                    completed=completed,
                    success=successful,
                    abstention=abstained,
                    error=failed,
                    candidate_count=candidate_count,
                )
                result = _score_worker(dataset_scorer, dataset, train, test, _session_lanes(input_session), backend)
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
                writer.update(stage="scoring", current_dataset=dataset, dataset_metadata=metadata, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
        else:
            # Bounded 14-task prefetch; only the parent loads values and writes
            # durable files.  Futures are retained until their name-order slot
            # can be committed.
            lanes = _session_lanes(input_session)
            executor = ProcessPoolExecutor(max_workers=FROZEN_CONFIG.cpu_workers, initializer=_worker_initializer)
            pending: dict[Future[Exp435DatasetResult], tuple[str, Mapping[str, object]]] = {}
            completed_results: dict[str, tuple[Exp435DatasetResult, Mapping[str, object]]] = {}
            submit_index = 0
            commit_index = 0
            try:
                while submit_index < len(missing) or pending:
                    while submit_index < len(missing) and len(pending) < _PREFETCH:
                        dataset = missing[submit_index]
                        train, test, metadata = _load_values(input_session, dataset)
                        writer.update(stage="scoring", current_dataset=dataset, dataset_metadata=metadata, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
                        future = executor.submit(_score_worker, dataset_scorer, dataset, train, test, lanes, backend)
                        pending[future] = (dataset, metadata)
                        submit_index += 1
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        dataset, metadata = pending.pop(future)
                        completed_results[dataset] = (future.result(), metadata)
                    while commit_index < len(missing) and missing[commit_index] in completed_results:
                        dataset = missing[commit_index]
                        result, metadata = completed_results.pop(dataset)
                        digest = store.commit(result)
                        completed += 1
                        successful += result.status == "scored"
                        abstained += result.status == "abstained"
                        failed += result.status == "failed"
                        candidate_count += result.candidate_count
                        writer.append_event(_result_event(result, digest, completed=completed, total=len(datasets), metadata=metadata, backend=backend))
                        writer.update(stage="scoring", current_dataset=dataset, dataset_metadata=metadata, completed=completed, success=successful, abstention=abstained, error=failed, candidate_count=candidate_count)
                        commit_index += 1
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        if len(store.completed_digests()) != len(datasets) or failed:
            raise ValueError("runtime population is incomplete")
        final_status = "checkpoint_complete" if len(datasets) == FROZEN_CONFIG.expected_datasets else "running"
        if final_status == "checkpoint_complete":
            writer.mark_complete(candidate_count=candidate_count)
        return _summary(store, backend, status=final_status, failed=failed)
    except KeyboardInterrupt as error:
        writer.mark_interrupted(writer.progress.get("current_dataset") if isinstance(writer.progress.get("current_dataset"), str) else None, error)
        raise
    except BaseException as error:
        # Candidate-budget overflow is persisted as an immutable event before
        # the exception escapes; no publication artifact is created here.
        try:
            message = " ".join(str(error).split())[:1000]
            if "candidate budget" in message.lower():
                writer.append_event({
                    "kind": "candidate_budget_exceeded",
                    "error_type": type(error).__name__,
                    "error_message": message,
                    "observed": getattr(error, "observed", None),
                    "budget": FROZEN_CONFIG.global_candidate_budget,
                    "stage": writer.progress.get("stage", "scoring"),
                })
            else:
                writer.append_event({"kind": "failure", "error_type": type(error).__name__, "error_message": message, "stage": writer.progress.get("stage", "scoring")})
            writer.mark_failed(writer.progress.get("current_dataset") if isinstance(writer.progress.get("current_dataset"), str) else None, error)
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


def inspect_exp435(
    *,
    run_root: Path,
    checkpoint_root: Path,
    expected_context: object,
) -> Exp435RunSummary:
    """Inspect only canonical files; this function opens neither DB nor workers."""

    context = context_payload(expected_context)
    backend_payload = context.get("backend_decision")
    backend = BackendDecision.cpu_fallback("inspect")
    if isinstance(backend_payload, Mapping):
        backend_name = backend_payload.get("backend")
        if backend_name == "cpu":
            backend = BackendDecision.cpu_fallback(str(backend_payload.get("reason", "inspect")))
    store = Exp435CheckpointStore.open_existing(checkpoint_root, context)
    progress_path = Path(run_root) / "progress_antigravity.json"
    progress = {}
    if progress_path.is_file():
        import json

        progress = json.loads(progress_path.read_bytes())
    status = str(progress.get("state", "running"))
    if status not in {"checkpoint_complete", "complete", "complete_review_only"}:
        status = "running"
    return _summary(store, backend, status=status, failed=int(progress.get("error", 0) or 0))


__all__ = ["inspect_exp435", "resume_exp435", "run_exp435_new"]
