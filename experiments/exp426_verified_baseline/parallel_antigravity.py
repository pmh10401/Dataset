"""Fixed twelve-process, bounded-prefetch executor for Exp426."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import time
from typing import Iterable, Iterator

from .contracts_antigravity import Exp426DatasetRecord, WorkerEvidence


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Exp426SourceWorkerTask:
    dataset: str
    raw: bytes
    expected_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("source worker dataset is invalid")
        if not isinstance(self.raw, bytes) or not self.raw:
            raise ValueError("source worker bytes are invalid")
        _lower_sha256(self.expected_digest, "source worker digest")


@dataclass(frozen=True)
class Exp426TrainWorkerTask:
    dataset: str
    train: object
    fold_contract_sha256: str
    source_record_sha256: str
    source_result_sha256: str

    def __post_init__(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.contracts import RawTrainDataset

        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("TRAIN worker dataset is invalid")
        if not isinstance(self.train, RawTrainDataset) or self.train.dataset != self.dataset:
            raise ValueError("TRAIN worker payload dataset does not match")
        _lower_sha256(self.fold_contract_sha256, "fold contract digest")
        _lower_sha256(self.source_record_sha256, "source record digest")
        _lower_sha256(self.source_result_sha256, "source result digest")


@dataclass(frozen=True)
class _Exp426ProbeTask:
    """Private deterministic scheduling probe; rejected by the public runner."""

    dataset: str
    record: Exp426DatasetRecord
    delay_seconds: float
    fail: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dataset, str)
            or not self.dataset
            or not isinstance(self.record, Exp426DatasetRecord)
            or self.record.dataset != self.dataset
            or isinstance(self.delay_seconds, bool)
            or not isinstance(self.delay_seconds, (int, float))
            or self.delay_seconds < 0
            or type(self.fail) is not bool
        ):
            raise ValueError("parallel test probe is invalid")


WorkerTask = Exp426SourceWorkerTask | Exp426TrainWorkerTask | _Exp426ProbeTask


@dataclass(frozen=True)
class Exp426OrderedWorkerResult:
    ordinal: int
    dataset: str
    value: object
    worker_pid: int
    thread_limits: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("worker result ordinal is invalid")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("worker result dataset is invalid")
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise ValueError("worker result PID is invalid")

    @property
    def scientific_sha256(self) -> str:
        value = self.value
        digest = getattr(value, "scientific_sha256", None)
        if not isinstance(digest, str):
            raise ValueError("worker result has no scientific digest")
        return digest


@dataclass(frozen=True)
class Exp426ParallelAudit:
    worker_pids: tuple[int, ...]
    thread_limits: tuple[tuple[tuple[str, str], ...], ...]
    max_in_flight: int


def _lower_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _limit_worker_threads() -> None:
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"


def _execute_task(ordinal: int, task: WorkerTask) -> Exp426OrderedWorkerResult:
    limits = tuple((name, os.environ.get(name, "")) for name in _THREAD_ENVIRONMENT)
    if isinstance(task, _Exp426ProbeTask):
        if task.delay_seconds:
            time.sleep(task.delay_seconds)
        if task.fail:
            raise ValueError("injected worker failure")
        value: object = WorkerEvidence(ordinal, task.dataset, task.record)
    elif isinstance(task, Exp426SourceWorkerTask):
        from .source_antigravity import classify_exp424_record_bytes

        classification = classify_exp424_record_bytes(
            task.dataset, task.raw, task.expected_digest
        )
        value = (
            classification
            if classification.imported_record is None
            else WorkerEvidence(ordinal, task.dataset, classification.imported_record)
        )
    elif isinstance(task, Exp426TrainWorkerTask):
        from .source_antigravity import verify_train_abstention

        record = verify_train_abstention(
            task.dataset,
            task.train,  # type: ignore[arg-type]
            task.fold_contract_sha256,
            source_record_sha256=task.source_record_sha256,
            source_result_sha256=task.source_result_sha256,
        )
        value = WorkerEvidence(ordinal, task.dataset, record)
    else:
        raise ValueError("worker task type is invalid")
    return Exp426OrderedWorkerResult(
        ordinal, task.dataset, value, os.getpid(), limits
    )


class _ExecutionMetrics:
    def __init__(self) -> None:
        self.max_in_flight = 0
        self.limits_by_pid: dict[int, tuple[tuple[str, str], ...]] = {}


def _validated_task(
    task: object, *, previous_dataset: str | None, allow_test_probes: bool,
) -> WorkerTask:
    allowed = (Exp426SourceWorkerTask, Exp426TrainWorkerTask, _Exp426ProbeTask)
    if type(task) not in allowed:
        raise ValueError("worker task type is invalid")
    if not allow_test_probes and type(task) is _Exp426ProbeTask:
        raise ValueError("test probe tasks are forbidden in the public executor")
    if previous_dataset is not None and task.dataset <= previous_dataset:
        raise ValueError("worker task datasets must be sorted and unique")
    return task  # type: ignore[return-value]


def _iter_execute_ordered(
    tasks: Iterable[WorkerTask], *, workers: int, prefetch: int,
    allow_test_probes: bool, metrics: _ExecutionMetrics,
) -> Iterator[Exp426OrderedWorkerResult]:
    if type(workers) is not int or workers <= 0 or workers > 12:
        raise ValueError("workers must be between 1 and 12")
    if type(prefetch) is not int or prefetch <= 0 or prefetch > 24:
        raise ValueError("prefetch must be between 1 and 24")
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Iterable):
        raise ValueError("worker tasks must be iterable")
    task_iterator = iter(tasks)
    context = multiprocessing.get_context("spawn")
    futures: dict[int, tuple[WorkerTask, Future[Exp426OrderedWorkerResult]]] = {}
    next_submit = 0
    next_emit = 0
    previous_dataset: str | None = None
    exhausted = False
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_limit_worker_threads,
    ) as pool:
        while futures or not exhausted:
            while not exhausted and len(futures) < prefetch:
                try:
                    candidate = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                task = _validated_task(
                    candidate,
                    previous_dataset=previous_dataset,
                    allow_test_probes=allow_test_probes,
                )
                previous_dataset = task.dataset
                futures[next_submit] = (
                    task, pool.submit(_execute_task, next_submit, task)
                )
                next_submit += 1
                metrics.max_in_flight = max(
                    metrics.max_in_flight, len(futures)
                )
            if not futures:
                break
            task, future = futures.pop(next_emit)
            try:
                result = future.result()
            except Exception as error:
                for _pending_task, pending in futures.values():
                    pending.cancel()
                raise RuntimeError(
                    f"Exp426 worker failed for dataset {task.dataset}"
                ) from error
            if result.ordinal != next_emit or result.dataset != task.dataset:
                raise ValueError("worker result order or dataset is invalid")
            metrics.limits_by_pid[result.worker_pid] = result.thread_limits
            next_emit += 1
            yield result


def _audit_from_metrics(metrics: _ExecutionMetrics) -> Exp426ParallelAudit:
    limits_by_pid = metrics.limits_by_pid
    audit = Exp426ParallelAudit(
        tuple(sorted(limits_by_pid)),
        tuple(limits_by_pid[pid] for pid in sorted(limits_by_pid)),
        metrics.max_in_flight,
    )
    return audit


def run_ordered_workers(
    tasks: Iterable[Exp426SourceWorkerTask | Exp426TrainWorkerTask],
    *, workers: int = 12, prefetch: int = 24,
) -> Iterator[Exp426OrderedWorkerResult]:
    """Run the production Exp426 worker pool with exactly twelve processes."""
    if type(workers) is not int or workers != 12:
        raise ValueError("workers must equal the frozen value 12")
    metrics = _ExecutionMetrics()
    yield from _iter_execute_ordered(
        tasks, workers=workers, prefetch=prefetch,
        allow_test_probes=False, metrics=metrics,
    )


def _iter_ordered_workers_for_test(
    tasks: Iterable[WorkerTask], *, workers: int, prefetch: int,
) -> Iterator[Exp426OrderedWorkerResult]:
    """Private streaming seam used to prove bounded input consumption."""
    metrics = _ExecutionMetrics()
    yield from _iter_execute_ordered(
        tasks, workers=workers, prefetch=prefetch,
        allow_test_probes=True, metrics=metrics,
    )


def _run_ordered_workers_for_test(
    tasks: Iterable[WorkerTask], *, workers: int, prefetch: int,
) -> tuple[tuple[Exp426OrderedWorkerResult, ...], Exp426ParallelAudit]:
    """Private readiness/test seam for fixed 1-versus-12 parity probes."""
    metrics = _ExecutionMetrics()
    results = tuple(_iter_execute_ordered(
        tasks, workers=workers, prefetch=prefetch,
        allow_test_probes=True, metrics=metrics,
    ))
    return results, _audit_from_metrics(metrics)
