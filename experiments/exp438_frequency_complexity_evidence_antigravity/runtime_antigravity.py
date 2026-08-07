"""Deterministic parent-owned runtime scheduling for Exp438."""

from __future__ import annotations

import os

# This must happen before imports below reach NumPy in production workers.
_THREAD_VARIABLES_ANTIGRAVITY = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
for _thread_variable_antigravity in _THREAD_VARIABLES_ANTIGRAVITY:
    os.environ[_thread_variable_antigravity] = "1"

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .checkpoint_antigravity import Exp438CheckpointStore
from .config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
)
from .contracts_antigravity import (
    CHANNELS_ANTIGRAVITY,
    AbstentionEvidence,
    BackendBatchEvidence,
    BackendDecision,
    Exp438DatasetResult,
    Exp438RunSummary,
)
from .inputs_antigravity import (
    Exp438RawDataset,
    partition_train_rows_antigravity,
    test_row_abstention_reason_antigravity,
)
from .scoring_antigravity import (
    CandidateBudgetExceededAntigravity,
    fit_calibrated_group_antigravity,
    score_test_row_cpu_antigravity,
)
from .progress_antigravity import ProgressWriterAntigravity


def worker_count_antigravity(logical_cpus: int | None = None) -> int:
    """Return the frozen max-minus-one dataset-worker policy."""
    return cpu_worker_count_antigravity(logical_cpus)


def _worker_initializer_antigravity() -> None:
    for name in _THREAD_VARIABLES_ANTIGRAVITY:
        os.environ[name] = "1"


@dataclass(frozen=True)
class LaneMembershipSnapshotAntigravity:
    hard: frozenset[tuple[str, int]]
    standard: frozenset[tuple[str, int]]
    priority: frozenset[tuple[str, int]]
    automatic_normal: frozenset[tuple[str, int]]

    def __post_init__(self) -> None:
        values = (self.hard, self.standard, self.priority, self.automatic_normal)
        if any(type(value) is not frozenset for value in values):
            raise ValueError("lane snapshot values must be frozen sets")
        seen: set[tuple[str, int]] = set()
        for lane in values:
            for coordinate in lane:
                if (
                    not isinstance(coordinate, tuple)
                    or len(coordinate) != 2
                    or type(coordinate[0]) is not str
                    or not coordinate[0]
                    or type(coordinate[1]) is not int
                    or coordinate[1] < 0
                    or coordinate in seen
                ):
                    raise ValueError("lane snapshot contains an invalid or overlapping coordinate")
                seen.add(coordinate)

    @classmethod
    def empty_antigravity(cls) -> "LaneMembershipSnapshotAntigravity":
        return cls(frozenset(), frozenset(), frozenset(), frozenset())

    def contains_antigravity(self, dataset: str, instance_index: int) -> bool:
        coordinate = (dataset, instance_index)
        return any(
            coordinate in lane
            for lane in (self.hard, self.standard, self.priority, self.automatic_normal)
        )


@dataclass(frozen=True)
class CpuDatasetTaskAntigravity:
    dataset: str
    train: Exp438RawDataset
    test: Exp438RawDataset
    lanes: LaneMembershipSnapshotAntigravity

    def __post_init__(self) -> None:
        if (
            type(self.dataset) is not str
            or not self.dataset
            or type(self.train) is not Exp438RawDataset
            or type(self.test) is not Exp438RawDataset
            or self.train.dataset != self.dataset
            or self.test.dataset != self.dataset
            or self.train.split != "TRAIN"
            or self.test.split != "TEST"
            or type(self.lanes) is not LaneMembershipSnapshotAntigravity
        ):
            raise ValueError("Exp438 CPU task is invalid")


def _score_cpu_task_antigravity(task: CpuDatasetTaskAntigravity) -> Exp438DatasetResult:
    """Score one picklable label-free dataset task using canonical NumPy CPU code."""
    if type(task) is not CpuDatasetTaskAntigravity:
        raise ValueError("Exp438 CPU task is invalid")
    prepared = partition_train_rows_antigravity(task.train)
    calibrated = {
        group.raw_length: fit_calibrated_group_antigravity(group)
        for group in prepared.groups
    }
    evidence = []
    abstentions = []
    for row in task.test.series:
        reason = test_row_abstention_reason_antigravity(row)
        group = calibrated.get(row.raw_length)
        if reason is None and group is None:
            reason = "no_exact_finite_train_group"
        if reason is not None:
            abstentions.append(
                AbstentionEvidence(row.dataset, row.instance_index, row.raw_length, reason)
            )
            continue
        evidence.append(
            score_test_row_cpu_antigravity(
                row,
                group,
                outside_all_frozen_lanes=not task.lanes.contains_antigravity(
                    row.dataset, row.instance_index
                ),
            )
        )
    result = Exp438DatasetResult(
        dataset=task.dataset,
        evidence=tuple(evidence),
        abstentions=tuple(abstentions),
        train_exclusions=prepared.exclusions,
    )
    if len(result.evidence) + len(result.abstentions) != len(task.test.series):
        raise ValueError("Exp438 CPU task did not conserve TEST rows")
    return result


def _score_accelerated_task_antigravity(
    task: CpuDatasetTaskAntigravity,
    *,
    backend: str,
    batch_size: int | None,
    engine: object,
) -> Exp438DatasetResult:
    """Use only the readiness-approved exact stage and keep all rules unchanged."""
    if type(task) is not CpuDatasetTaskAntigravity or backend not in {"numba", "mlx", "mps"}:
        raise ValueError("Exp438 accelerated task binding is invalid")
    if getattr(engine, "backend", None) != backend:
        raise ValueError("Exp438 accelerated engine does not match the selected backend")
    if backend == "numba":
        compute = getattr(engine, "compute", None)
        if (
            not callable(compute)
            or type(batch_size) is not int
            or batch_size < 1
        ):
            raise ValueError("Exp438 Numba engine is invalid")
        aspe_compute = compute
        dense_score_batch = None
    else:
        score_batch = getattr(engine, "score_batch", None)
        if not callable(score_batch) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Exp438 dense accelerator is invalid")
        aspe_compute = None

        def dense_score_batch(queries, references, k):
            return score_batch(
                queries, references, k, batch_size=batch_size
            )

    prepared = partition_train_rows_antigravity(task.train)
    fit_keywords = {}
    score_keywords = {}
    if aspe_compute is not None:
        fit_keywords["aspe_compute"] = aspe_compute
        score_keywords["aspe_compute"] = aspe_compute
    if dense_score_batch is not None:
        fit_keywords["dense_score_batch"] = dense_score_batch
        score_keywords["dense_score_batch"] = dense_score_batch
    calibrated = {
        group.raw_length: fit_calibrated_group_antigravity(group, **fit_keywords)
        for group in prepared.groups
    }
    evidence = []
    abstentions = []
    for row in task.test.series:
        reason = test_row_abstention_reason_antigravity(row)
        group = calibrated.get(row.raw_length)
        if reason is None and group is None:
            reason = "no_exact_finite_train_group"
        if reason is not None:
            abstentions.append(
                AbstentionEvidence(row.dataset, row.instance_index, row.raw_length, reason)
            )
            continue
        evidence.append(
            score_test_row_cpu_antigravity(
                row,
                group,
                outside_all_frozen_lanes=not task.lanes.contains_antigravity(
                    row.dataset, row.instance_index
                ),
                **score_keywords,
            )
        )
    return Exp438DatasetResult(
        task.dataset, tuple(evidence), tuple(abstentions), prepared.exclusions
    )


def _length_histogram_antigravity(dataset: Exp438RawDataset) -> dict[str, int]:
    counts: dict[int, int] = {}
    for row in dataset.series:
        counts[row.raw_length] = counts.get(row.raw_length, 0) + 1
    return {str(length): counts[length] for length in sorted(counts)}


def _dataset_metadata_antigravity(
    train: Exp438RawDataset, test: Exp438RawDataset
) -> dict[str, object]:
    return {
        "train_count": len(train.series),
        "test_count": len(test.series),
        "train_length_histogram": _length_histogram_antigravity(train),
        "test_length_histogram": _length_histogram_antigravity(test),
        "train_value_count": sum(row.raw_length for row in train.series),
        "test_value_count": sum(row.raw_length for row in test.series),
        "train_bytes": sum(row.raw_length * 4 for row in train.series),
        "test_bytes": sum(row.raw_length * 4 for row in test.series),
    }


def _load_task_antigravity(
    session: object,
    dataset: str,
    lanes: LaneMembershipSnapshotAntigravity,
) -> tuple[CpuDatasetTaskAntigravity, dict[str, object]]:
    load_train = getattr(session, "load_train", None)
    load_test = getattr(session, "load_test_values", None)
    if not callable(load_train) or not callable(load_test):
        raise ValueError("Exp438 input session lacks label-free value loaders")
    train = load_train(dataset)
    test = load_test(dataset, train)
    if type(train) is not Exp438RawDataset or type(test) is not Exp438RawDataset:
        raise ValueError("Exp438 input session returned an invalid raw dataset")
    return (
        CpuDatasetTaskAntigravity(dataset, train, test, lanes),
        _dataset_metadata_antigravity(train, test),
    )


def run_ordered_cpu_schedule_antigravity(
    *,
    datasets: tuple[str, ...],
    session: object,
    lanes: LaneMembershipSnapshotAntigravity,
    workers: int,
    commit: Callable[[Exp438DatasetResult, Mapping[str, object]], None],
    progress: Callable[..., None],
    score_task: Callable[[CpuDatasetTaskAntigravity], Exp438DatasetResult] = _score_cpu_task_antigravity,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    observer: Callable[[int, int], None] | None = None,
) -> None:
    """Compute concurrently while committing only the frozen ordered prefix."""
    if (
        type(datasets) is not tuple
        or not datasets
        or datasets != tuple(sorted(datasets))
        or len(set(datasets)) != len(datasets)
        or type(workers) is not int
        or workers < 1
        or type(lanes) is not LaneMembershipSnapshotAntigravity
        or not all(callable(item) for item in (commit, progress, score_task, executor_factory))
    ):
        raise ValueError("Exp438 CPU schedule binding is invalid")
    capacity = min(workers, len(datasets))
    executor = executor_factory(
        max_workers=capacity,
        initializer=_worker_initializer_antigravity,
    )
    pending: dict[Future[Exp438DatasetResult], tuple[str, Mapping[str, object]]] = {}
    buffered: dict[str, tuple[Exp438DatasetResult, Mapping[str, object]]] = {}
    submit_index = 0
    commit_index = 0
    try:
        while submit_index < len(datasets) or pending:
            while (
                submit_index < len(datasets)
                and len(pending) + len(buffered) < capacity
            ):
                dataset = datasets[submit_index]
                progress(
                    stage="loading",
                    current_dataset=dataset,
                    dataset_metadata=None,
                    in_flight=tuple(name for name, _ in pending.values()),
                )
                task, metadata = _load_task_antigravity(session, dataset, lanes)
                progress(
                    stage="dispatching",
                    current_dataset=dataset,
                    dataset_metadata=metadata,
                    in_flight=tuple(name for name, _ in pending.values()),
                )
                future = executor.submit(score_task, task)
                pending[future] = (dataset, metadata)
                submit_index += 1
            if observer is not None:
                observer(len(pending), len(buffered))
            if not pending:
                raise RuntimeError("Exp438 ordered scheduler lost its next frozen result")
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                dataset, metadata = pending.pop(future)
                progress(
                    stage="awaiting_worker_result",
                    current_dataset=dataset,
                    dataset_metadata=metadata,
                    in_flight=tuple(name for name, _ in pending.values()),
                )
                result = future.result()
                if type(result) is not Exp438DatasetResult or result.dataset != dataset:
                    raise ValueError("Exp438 worker returned the wrong dataset")
                buffered[dataset] = (result, metadata)
            if observer is not None:
                observer(len(pending), len(buffered))
            while commit_index < len(datasets) and datasets[commit_index] in buffered:
                dataset = datasets[commit_index]
                result, metadata = buffered.pop(dataset)
                progress(
                    stage="committing",
                    current_dataset=dataset,
                    dataset_metadata=metadata,
                    in_flight=tuple(name for name, _ in pending.values()),
                )
                commit(result, metadata)
                commit_index += 1
            if observer is not None:
                observer(len(pending), len(buffered))
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run_serial_accelerator_schedule_antigravity(
    *,
    datasets: tuple[str, ...],
    session: object,
    lanes: LaneMembershipSnapshotAntigravity,
    commit: Callable[[Exp438DatasetResult, Mapping[str, object]], None],
    progress: Callable[..., None],
    score_task: Callable[[CpuDatasetTaskAntigravity], Exp438DatasetResult],
) -> None:
    """Run one adopted accelerator in the parent process with no outer parallelism."""
    if (
        type(datasets) is not tuple
        or not datasets
        or datasets != tuple(sorted(datasets))
        or len(set(datasets)) != len(datasets)
        or type(lanes) is not LaneMembershipSnapshotAntigravity
        or not all(callable(item) for item in (commit, progress, score_task))
    ):
        raise ValueError("Exp438 accelerator schedule binding is invalid")
    for dataset in datasets:
        progress(
            stage="loading",
            current_dataset=dataset,
            dataset_metadata=None,
            in_flight=(),
        )
        task, metadata = _load_task_antigravity(session, dataset, lanes)
        progress(
            stage="dispatching",
            current_dataset=dataset,
            dataset_metadata=metadata,
            in_flight=(dataset,),
        )
        result = score_task(task)
        if type(result) is not Exp438DatasetResult or result.dataset != dataset:
            raise ValueError("Exp438 accelerator returned the wrong dataset")
        progress(
            stage="committing",
            current_dataset=dataset,
            dataset_metadata=metadata,
            in_flight=(),
        )
        commit(result, metadata)


def _lane_mapping_antigravity(lanes: object, name: str) -> Mapping[str, object]:
    value = getattr(lanes, name, None)
    if value is None and isinstance(lanes, Mapping):
        value = lanes.get(name)
    if not isinstance(value, Mapping):
        raise ValueError("Exp438 frozen lane mapping is invalid")
    return value


def _lane_snapshot_antigravity(
    lanes: object, datasets: tuple[str, ...]
) -> tuple[LaneMembershipSnapshotAntigravity, dict[str, dict[str, list[int]]]]:
    copied: dict[str, frozenset[tuple[str, int]]] = {}
    payload: dict[str, dict[str, list[int]]] = {}
    for name in ("hard", "standard", "priority", "automatic_normal"):
        coordinates: set[tuple[str, int]] = set()
        mapping_payload: dict[str, list[int]] = {}
        for dataset, raw_indices in sorted(_lane_mapping_antigravity(lanes, name).items()):
            if dataset not in datasets or not isinstance(raw_indices, (tuple, list, frozenset)):
                raise ValueError("Exp438 frozen lane mapping is invalid")
            indices = sorted(raw_indices)
            if len(indices) != len(set(indices)) or any(type(index) is not int or index < 0 for index in indices):
                raise ValueError("Exp438 frozen lane mapping is invalid")
            mapping_payload[str(dataset)] = indices
            coordinates.update((str(dataset), index) for index in indices)
        copied[name] = frozenset(coordinates)
        payload[name] = mapping_payload
    return LaneMembershipSnapshotAntigravity(**copied), payload


def _decision_payload_antigravity(decision: BackendDecision) -> dict[str, object]:
    if type(decision) is not BackendDecision:
        raise ValueError("Exp438 backend decision is invalid")
    return {
        "selected_backend": decision.selected_backend,
        "selected_batch_size": decision.selected_batch_size,
        "evidence": [asdict(item) for item in decision.evidence],
        "fallback_reason": decision.fallback_reason,
    }


def _decision_from_payload_antigravity(payload: object) -> BackendDecision:
    if not isinstance(payload, Mapping):
        raise ValueError("Exp438 stored backend decision is invalid")
    try:
        return BackendDecision(
            selected_backend=payload["selected_backend"],
            selected_batch_size=payload["selected_batch_size"],
            evidence=tuple(BackendBatchEvidence(**item) for item in payload["evidence"]),
            fallback_reason=payload["fallback_reason"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp438 stored backend decision is invalid") from error


@contextmanager
def _session_from_factory_antigravity(factory: Callable[[], object]):
    if not callable(factory):
        raise ValueError("Exp438 input session factory is required")
    owned = factory()
    if hasattr(owned, "__enter__") and hasattr(owned, "__exit__"):
        with owned as session:
            yield session
        return
    try:
        yield owned
    finally:
        close = getattr(owned, "_close_antigravity", None)
        if callable(close):
            close()


def _build_context_antigravity(
    *,
    session: object,
    reviewed_commit: str,
    decision: BackendDecision,
    output_root: Path,
    checkpoint_root: Path,
    readiness_payload: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], LaneMembershipSnapshotAntigravity]:
    source = getattr(session, "context", None)
    datasets = tuple(getattr(source, "datasets", ()))
    if (
        type(reviewed_commit) is not str
        or len(reviewed_commit) != 40
        or any(character not in "0123456789abcdef" for character in reviewed_commit)
        or not datasets
        or datasets != tuple(sorted(datasets))
        or len(set(datasets)) != len(datasets)
        or getattr(source, "code_commit", None) != reviewed_commit
        or getattr(source, "database_sha256", None) != FROZEN_CONFIG.database_sha256
    ):
        raise ValueError("Exp438 input session does not match the reviewed identity")
    test_manifest = getattr(source, "test_identity_manifest", None)
    if not isinstance(test_manifest, Mapping) or set(test_manifest) != set(datasets):
        raise ValueError("Exp438 TEST identity manifest is invalid")
    manifest_payload = {
        dataset: [[int(index), int(length)] for index, length in test_manifest[dataset]]
        for dataset in datasets
    }
    lanes, lane_payload = _lane_snapshot_antigravity(
        getattr(source, "lanes", None), datasets
    )
    source_paths = getattr(getattr(session, "source_paths", None), "to_payload", None)
    if not callable(source_paths):
        raise ValueError("Exp438 canonical source paths are unavailable")
    path_payload = source_paths()
    if not isinstance(path_payload, Mapping):
        raise ValueError("Exp438 canonical source paths are invalid")
    context: dict[str, object] = {
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "code_commit": reviewed_commit,
        "database_sha256": FROZEN_CONFIG.database_sha256,
        "datasets": list(datasets),
        "backend": decision.selected_backend,
        "backend_decision": _decision_payload_antigravity(decision),
        "frozen_config": frozen_config_payload_antigravity(),
        "test_identity_manifest": manifest_payload,
        "lane_mappings": lane_payload,
        "source_paths": dict(path_payload),
        "output_root": str(Path(output_root).resolve()),
        "checkpoint_root": str(Path(checkpoint_root).resolve()),
        "official_cpu_workers": worker_count_antigravity(),
        "prediction_label_access_count": 0,
    }
    if readiness_payload is not None:
        if (
            readiness_payload.get("schema") != "exp438.readiness.v1"
            or readiness_payload.get("reviewed_commit") != reviewed_commit
            or readiness_payload.get("database_sha256") != FROZEN_CONFIG.database_sha256
            or readiness_payload.get("datasets") != list(datasets)
            or readiness_payload.get("backend_decision") != _decision_payload_antigravity(decision)
            or not isinstance(readiness_payload.get("backend_provenance"), Mapping)
            or not isinstance(readiness_payload.get("source_sha256"), Mapping)
        ):
            raise ValueError("Exp438 readiness does not match the runtime context")
        context["readiness_sha256"] = sha256(
            canonical_json_bytes(dict(readiness_payload))
        ).hexdigest()
        context["backend_provenance"] = dict(readiness_payload["backend_provenance"])
        context["source_sha256"] = dict(readiness_payload["source_sha256"])
    return context, lanes


def _counts_antigravity(
    results: tuple[Exp438DatasetResult, ...], *, imported: int
) -> dict[str, object]:
    channels = {channel: 0 for channel in CHANNELS_ANTIGRAVITY}
    for result in results:
        for channel, coordinates in result.candidate_coordinates.items():
            channels[channel] += len(coordinates)
    return {
        "completed": len(results),
        "import_count": imported,
        "recompute_count": 0,
        "success_count": sum(bool(result.evidence) for result in results),
        "abstention_count": sum(not result.evidence for result in results),
        "error_count": 0,
        "per_channel_candidate_counts": channels,
        "checkpoint": results[-1].dataset if results else None,
    }


def _summary_antigravity(
    store: Exp438CheckpointStore, decision: BackendDecision
) -> Exp438RunSummary:
    results = store.results()
    counts = _counts_antigravity(results, imported=len(results))
    return Exp438RunSummary(
        total_datasets=len(store.context_payload["datasets"]),
        completed_datasets=len(results),
        successful_datasets=int(counts["success_count"]),
        abstained_datasets=int(counts["abstention_count"]),
        failed_datasets=0,
        per_channel_candidate_counts=counts["per_channel_candidate_counts"],
        backend=decision,
        prediction_label_access_count=0,
    )


def _create_accelerator_engine_antigravity(backend: str) -> object:
    if backend == "numba":
        from .numba_antigravity import create_numba_aspe_engine_antigravity

        return create_numba_aspe_engine_antigravity()
    if backend == "mlx":
        from .mlx_antigravity import create_mlx_dense_engine_antigravity

        return create_mlx_dense_engine_antigravity()
    if backend == "mps":
        from .mps_antigravity import create_mps_dense_engine_antigravity

        return create_mps_dense_engine_antigravity()
    raise ValueError("CPU does not create an accelerator engine")


def _run_antigravity(
    *,
    mode: str,
    output_root: Path,
    checkpoint_root: Path,
    reviewed_commit: str,
    decision: BackendDecision,
    input_session_factory: Callable[[], object],
    workers: int,
    executor_factory: Callable[..., object] = ProcessPoolExecutor,
    score_task: Callable[[CpuDatasetTaskAntigravity], Exp438DatasetResult] | None = None,
    accelerator_engine_factory: Callable[[str], object] = _create_accelerator_engine_antigravity,
    readiness_payload: Mapping[str, object] | None = None,
) -> Exp438RunSummary:
    """Execute only a verified new or interrupted-resume Exp438 root."""
    if mode not in {"new", "resume"} or type(decision) is not BackendDecision:
        raise ValueError("Exp438 runtime mode or decision is invalid")
    output_root = Path(output_root)
    checkpoint_root = Path(checkpoint_root)
    with _session_from_factory_antigravity(input_session_factory) as session:
        context, lanes = _build_context_antigravity(
            session=session,
            reviewed_commit=reviewed_commit,
            decision=decision,
            output_root=output_root,
            checkpoint_root=checkpoint_root,
            readiness_payload=readiness_payload,
        )
        if getattr(session, "label_access_count", None) != 0:
            raise ValueError("Exp438 prediction label access must remain zero")
        if mode == "new":
            if output_root.exists() or checkpoint_root.exists():
                raise ValueError("Exp438 new roots must be absent")
            store = Exp438CheckpointStore.create(checkpoint_root, context)
            writer = ProgressWriterAntigravity.create(
                output_root,
                store.context_hash,
                len(context["datasets"]),
                decision.selected_backend,
                pid=os.getpid(),
            )
        else:
            store = Exp438CheckpointStore.open_existing(
                checkpoint_root, context, mutable=True
            )
            if store.status != "interrupted":
                raise ValueError("Exp438 resume requires an interrupted checkpoint")
            writer = ProgressWriterAntigravity.open_existing(
                output_root, store.context_hash, mutable=True
            )
            if writer.snapshot["state"] != "interrupted":
                raise ValueError("Exp438 resume requires interrupted progress")
        existing = store.results()
        counters = _counts_antigravity(existing, imported=len(existing))
        datasets = tuple(context["datasets"])
        missing = datasets[len(existing):]

        def progress(**payload) -> None:
            writer.update(
                stage=payload["stage"],
                current_dataset=payload["current_dataset"],
                dataset_metadata=payload["dataset_metadata"],
                completed=int(counters["completed"]),
                import_count=int(counters["import_count"]),
                recompute_count=int(counters["recompute_count"]),
                success_count=int(counters["success_count"]),
                abstention_count=int(counters["abstention_count"]),
                error_count=int(counters["error_count"]),
                checkpoint=counters["checkpoint"],
                per_channel_candidate_counts=counters["per_channel_candidate_counts"],
            )

        def commit(result: Exp438DatasetResult, metadata: Mapping[str, object]) -> None:
            proposed = dict(counters["per_channel_candidate_counts"])
            for channel, coordinates in result.candidate_coordinates.items():
                proposed[channel] += len(coordinates)
                if proposed[channel] > FROZEN_CONFIG.channel_candidate_budget:
                    raise CandidateBudgetExceededAntigravity(
                        f"{channel} candidate budget exceeded before checkpoint commit"
                    )
            if sum(proposed.values()) > FROZEN_CONFIG.total_candidate_budget:
                raise CandidateBudgetExceededAntigravity(
                    "total candidate budget exceeded before checkpoint commit"
                )
            digest = store.commit(result, dataset_metadata=metadata)
            counters["completed"] = int(counters["completed"]) + 1
            counters["recompute_count"] = int(counters["recompute_count"]) + 1
            counters["success_count"] = int(counters["success_count"]) + bool(result.evidence)
            counters["abstention_count"] = int(counters["abstention_count"]) + (not result.evidence)
            counters["per_channel_candidate_counts"] = proposed
            counters["checkpoint"] = result.dataset
            writer.append_event({
                "event": "dataset_committed",
                "dataset": result.dataset,
                "record_sha256": digest,
                "completed": counters["completed"],
            })
            progress(
                stage="idle",
                current_dataset=None,
                dataset_metadata=None,
                in_flight=(),
            )

        writer.start_heartbeat(interval_seconds=120.0)
        try:
            if decision.selected_backend == "cpu":
                run_ordered_cpu_schedule_antigravity(
                    datasets=missing,
                    session=session,
                    lanes=lanes,
                    workers=workers,
                    commit=commit,
                    progress=progress,
                    score_task=score_task or _score_cpu_task_antigravity,
                    executor_factory=executor_factory,
                )
            else:
                engine = accelerator_engine_factory(decision.selected_backend)
                accelerated = score_task or partial(
                    _score_accelerated_task_antigravity,
                    backend=decision.selected_backend,
                    batch_size=decision.selected_batch_size,
                    engine=engine,
                )
                run_serial_accelerator_schedule_antigravity(
                    datasets=missing,
                    session=session,
                    lanes=lanes,
                    commit=commit,
                    progress=progress,
                    score_task=accelerated,
                )
            if getattr(session, "label_access_count", None) != 0:
                raise ValueError("Exp438 prediction label access must remain zero")
            store.mark_complete()
            writer.mark_complete(completed=len(datasets))
            return _summary_antigravity(store, decision)
        except KeyboardInterrupt:
            store.mark_interrupted()
            writer.mark_interrupted()
            raise
        except BaseException as error:
            counters["error_count"] = 1
            writer.update(
                stage="idle",
                current_dataset=None,
                completed=int(counters["completed"]),
                import_count=int(counters["import_count"]),
                recompute_count=int(counters["recompute_count"]),
                success_count=int(counters["success_count"]),
                abstention_count=int(counters["abstention_count"]),
                error_count=1,
                checkpoint=counters["checkpoint"],
                per_channel_candidate_counts=counters["per_channel_candidate_counts"],
                latest_error={
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            store.mark_failed(type(error).__name__)
            writer.mark_failed(type(error).__name__)
            raise
        finally:
            writer.stop_heartbeat()


def inspect_exp438_antigravity(
    output_root: Path, checkpoint_root: Path
) -> Exp438RunSummary:
    """Inspect completed Exp438 artifacts without opening the DB or workers."""
    store = Exp438CheckpointStore.inspect_existing(checkpoint_root)
    writer = ProgressWriterAntigravity.open_existing(
        output_root, store.context_hash, mutable=False
    )
    if store.status != "complete" or writer.snapshot["state"] != "complete":
        raise ValueError("Exp438 inspection requires matching complete roots")
    decision = _decision_from_payload_antigravity(
        store.context_payload.get("backend_decision")
    )
    return _summary_antigravity(store, decision)


def run_exp438_new_antigravity(
    *,
    output_root: Path,
    checkpoint_root: Path,
    reviewed_commit: str,
    decision: BackendDecision,
    input_session_factory: Callable[[], object],
    readiness_payload: Mapping[str, object],
) -> Exp438RunSummary:
    return _run_antigravity(
        mode="new",
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        reviewed_commit=reviewed_commit,
        decision=decision,
        input_session_factory=input_session_factory,
        workers=worker_count_antigravity(),
        readiness_payload=readiness_payload,
    )


def resume_exp438_antigravity(
    *,
    output_root: Path,
    checkpoint_root: Path,
    reviewed_commit: str,
    decision: BackendDecision,
    input_session_factory: Callable[[], object],
    readiness_payload: Mapping[str, object],
) -> Exp438RunSummary:
    return _run_antigravity(
        mode="resume",
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        reviewed_commit=reviewed_commit,
        decision=decision,
        input_session_factory=input_session_factory,
        workers=worker_count_antigravity(),
        readiness_payload=readiness_payload,
    )


__all__ = [
    "CpuDatasetTaskAntigravity",
    "LaneMembershipSnapshotAntigravity",
    "run_ordered_cpu_schedule_antigravity",
    "run_serial_accelerator_schedule_antigravity",
    "worker_count_antigravity",
    "inspect_exp438_antigravity",
    "resume_exp438_antigravity",
    "run_exp438_new_antigravity",
]
