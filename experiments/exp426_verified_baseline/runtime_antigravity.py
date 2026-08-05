"""Parent-owned resumable orchestration and self-written progress for Exp426."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Event, RLock, Thread
import time
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .checkpoint_antigravity import (
    Exp426CheckpointInspection,
    Exp426CheckpointStore,
    _atomic_replace,
    _fsync_directory,
    _write_new,
)
from .contracts_antigravity import Exp426Counts, Exp426Progress, Exp426Provenance, WorkerEvidence
from .parallel_antigravity import (
    Exp426OrderedWorkerResult,
    Exp426SourceWorkerTask,
    Exp426TrainWorkerTask,
    run_ordered_workers,
)
from .source_antigravity import (
    Exp424SourceClassification,
    read_verified_source_tasks,
    revalidate_verified_source,
)


_OUTPUT_FILES = ("progress.json", "events.jsonl", "parent.log", "run.pid")


@dataclass(frozen=True)
class _RuntimeDependencies:
    source_tasks: Callable[[object], Iterable[object]]
    source_revalidate: Callable[[object], None]
    worker_runner: Callable[..., Iterable[object]]
    load_train: Callable[[object, str, object], object]
    verify_database: Callable[[object, str], None]
    heartbeat_seconds: float | None = 30.0
    process_is_live: Callable[[int], bool] = lambda pid: _process_is_live(pid)

    def __post_init__(self) -> None:
        for name in (
            "source_tasks", "source_revalidate", "worker_runner",
            "load_train", "verify_database", "process_is_live",
        ):
            if not callable(getattr(self, name)):
                raise ValueError(f"runtime dependency {name} is invalid")
        if self.heartbeat_seconds is not None and (
            isinstance(self.heartbeat_seconds, bool)
            or not isinstance(self.heartbeat_seconds, (int, float))
            or self.heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat interval is invalid")


@dataclass(frozen=True)
class Exp426RuntimeResult:
    state: str
    counts: Exp426Counts
    output: Path
    checkpoint: Path


@dataclass(frozen=True)
class Exp426RuntimeInspection:
    progress: Mapping[str, object]
    checkpoint: Exp426CheckpointInspection

    def __post_init__(self) -> None:
        object.__setattr__(self, "progress", MappingProxyType(dict(self.progress)))


def _production_dependencies() -> _RuntimeDependencies:
    def source_tasks(source: object) -> Iterator[Exp426SourceWorkerTask]:
        for item in read_verified_source_tasks(source):  # type: ignore[arg-type]
            yield Exp426SourceWorkerTask(item.dataset, item.raw, item.expected_digest)

    def load_train(database: object, dataset: str, context: object) -> object:
        return database.load_train(dataset, context)  # type: ignore[attr-defined]

    def verify_database(database: object, name: str) -> None:
        database.verify_boundary(name)  # type: ignore[attr-defined]

    return _RuntimeDependencies(
        source_tasks=source_tasks,
        source_revalidate=lambda source: revalidate_verified_source(source),  # type: ignore[arg-type]
        worker_runner=run_ordered_workers,
        load_train=load_train,
        verify_database=verify_database,
    )


def _process_is_live(pid: int) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _append_durable(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_events(output: Path) -> tuple[dict[str, object], ...]:
    raw = (output / "events.jsonl").read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("runtime events contain a partial record")
    result: list[dict[str, object]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("runtime event is invalid") from error
        if not isinstance(value, dict) or canonical_json_bytes(value) != line:
            raise ValueError("runtime event is not canonical JSON")
        result.append(value)
    datasets = [item.get("dataset") for item in result if item.get("kind") == "dataset_complete"]
    if len(datasets) != len(set(datasets)):
        raise ValueError("runtime dataset completion event is duplicated")
    return tuple(result)


def _counts_from_store(store: Exp426CheckpointStore) -> Exp426Counts:
    digests = store.completed_digests()
    imported = verified = 0
    for dataset in sorted(digests):
        record = store.load_record(dataset)
        if record.origin == "exp424_import":
            imported += 1
        elif record.origin == "verified_train_abstention":
            verified += 1
        else:
            raise ValueError("checkpoint contains an unknown record origin")
    return Exp426Counts(
        total=len(store.provenance.expected_datasets),
        completed=len(digests),
        imported=imported,
        verified_train_abstentions=verified,
        abstained=verified,
        failed=0,
    )


def _counts_payload(counts: Exp426Counts, errors: int) -> dict[str, int]:
    return {
        "total": counts.total,
        "completed": counts.completed,
        "remaining": counts.remaining,
        "imported": counts.imported,
        "verified_train_abstentions": counts.verified_train_abstentions,
        "abstained": counts.abstained,
        "failed": counts.failed,
        "success": counts.completed,
        "error": errors,
    }


class _RuntimeLogger:
    def __init__(
        self, output: Path, run_identity: str, checkpoint: Path,
        *, heartbeat_seconds: float | None,
    ) -> None:
        self.output = output
        self.run_identity = run_identity
        self.checkpoint = checkpoint
        self.heartbeat_seconds = heartbeat_seconds
        self.started = time.monotonic()
        self.elapsed_offset = 0.0
        self.lock = RLock()
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.progress: dict[str, object] = {}
        self.errors = 0

    @classmethod
    def create(
        cls, output: Path, run_identity: str, checkpoint: Path,
        *, heartbeat_seconds: float | None,
    ) -> "_RuntimeLogger":
        output.mkdir(mode=0o700)
        _fsync_directory(output.parent)
        _write_new(output / "events.jsonl", b"")
        _write_new(output / "parent.log", b"")
        logger = cls(output, run_identity, checkpoint, heartbeat_seconds=heartbeat_seconds)
        _write_new(output / "run.pid", canonical_json_bytes(logger._pid_payload(True)))
        return logger

    @classmethod
    def open(
        cls, output: Path, run_identity: str, checkpoint: Path,
        *, heartbeat_seconds: float | None, process_is_live: Callable[[int], bool],
    ) -> "_RuntimeLogger":
        if not output.is_dir() or any((output / name).is_symlink() for name in _OUTPUT_FILES):
            raise ValueError("runtime output root is invalid")
        if not all((output / name).is_file() for name in _OUTPUT_FILES):
            raise ValueError("runtime output inventory is incomplete")
        progress = _canonical_object(output / "progress.json", "runtime progress")
        pid_payload = _canonical_object(output / "run.pid", "runtime PID")
        if progress.get("run_identity") != run_identity or pid_payload.get("run_identity") != run_identity:
            raise ValueError("runtime output run identity does not match")
        old_pid = pid_payload.get("pid")
        if pid_payload.get("active") is True and type(old_pid) is int and process_is_live(old_pid):
            raise ValueError("matching Exp426 runtime process is still active")
        _load_events(output)
        logger = cls(output, run_identity, checkpoint, heartbeat_seconds=heartbeat_seconds)
        logger.progress = progress
        elapsed = progress.get("elapsed_seconds", 0.0)
        logger.elapsed_offset = float(elapsed) if isinstance(elapsed, (int, float)) else 0.0
        counts = progress.get("counts")
        if isinstance(counts, dict) and type(counts.get("error")) is int:
            logger.errors = counts["error"]
        _atomic_replace(output / "run.pid", canonical_json_bytes(logger._pid_payload(True)))
        return logger

    def _now(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _elapsed(self) -> float:
        return self.elapsed_offset + max(0.0, time.monotonic() - self.started)

    def _pid_payload(self, active: bool) -> dict[str, object]:
        return {
            "schema": "exp426.runtime-pid.v1",
            "run_identity": self.run_identity,
            "pid": os.getpid(),
            "active": active,
            "updated_at": self._now(),
        }

    def write_progress(
        self, state: str, counts: Exp426Counts,
        *, current_dataset: str | None = None,
        metadata: Mapping[str, object] | None = None,
        latest_error: Mapping[str, object] | None = None,
    ) -> None:
        Exp426Progress(state, counts, current_dataset, 12)
        elapsed = self._elapsed()
        throughput = counts.completed / elapsed if elapsed > 0 else 0.0
        eta = counts.remaining / throughput if throughput > 0 else None
        payload = {
            "schema": "exp426.runtime-progress.v1",
            "run_identity": self.run_identity,
            "updated_at": self._now(),
            "pid": os.getpid(),
            "state": state,
            "current_dataset": current_dataset,
            "dataset_metadata": None if metadata is None else dict(metadata),
            "backend": "cpu",
            "workers": 12,
            "counts": _counts_payload(counts, self.errors),
            "throughput_datasets_per_second": throughput,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "checkpoint": str(self.checkpoint),
            "latest_error": None if latest_error is None else dict(latest_error),
        }
        with self.lock:
            _atomic_replace(self.output / "progress.json", canonical_json_bytes(payload))
            self.progress = payload

    def append_event(self, payload: Mapping[str, object]) -> None:
        event = dict(payload)
        event["schema"] = "exp426.runtime-event.v1"
        event["run_identity"] = self.run_identity
        event["recorded_at"] = self._now()
        with self.lock:
            _append_durable(self.output / "events.jsonl", canonical_json_bytes(event) + b"\n")

    def append_log(self, message: str) -> None:
        if not isinstance(message, str) or "\n" in message:
            raise ValueError("runtime log message is invalid")
        with self.lock:
            _append_durable(
                self.output / "parent.log",
                f"{self._now()} {message}\n".encode("utf-8"),
            )

    def transition(
        self, state: str, counts: Exp426Counts,
        *, current_dataset: str | None = None,
        metadata: Mapping[str, object] | None = None,
        latest_error: Mapping[str, object] | None = None,
    ) -> None:
        self.write_progress(
            state, counts, current_dataset=current_dataset,
            metadata=metadata, latest_error=latest_error,
        )
        self.append_event({"kind": "state", "state": state})
        self.append_log(f"state={state} completed={counts.completed}/{counts.total}")

    def start_heartbeat(self) -> None:
        if self.heartbeat_seconds is None:
            return

        def heartbeat() -> None:
            assert self.heartbeat_seconds is not None
            while not self.stop_event.wait(self.heartbeat_seconds):
                with self.lock:
                    if not self.progress or self.progress.get("state") in {"complete_review_only", "failed"}:
                        continue
                    payload = dict(self.progress)
                    payload["updated_at"] = self._now()
                    payload["elapsed_seconds"] = self._elapsed()
                    _atomic_replace(self.output / "progress.json", canonical_json_bytes(payload))
                    self.progress = payload
                    _atomic_replace(self.output / "run.pid", canonical_json_bytes(self._pid_payload(True)))

        self.thread = Thread(target=heartbeat, name="exp426-heartbeat", daemon=True)
        self.thread.start()

    def stop_heartbeat(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def close(self) -> None:
        self.stop_heartbeat()
        with self.lock:
            _atomic_replace(self.output / "run.pid", canonical_json_bytes(self._pid_payload(False)))


def _metadata_from_classification(item: Exp424SourceClassification) -> dict[str, object]:
    return {
        "train_size": item.train_size,
        "train_lengths": list(item.train_lengths),
        "test_size": item.test_size,
        "test_lengths": list(item.test_lengths),
    }


def _metadata_from_train(item: object, source: Exp424SourceClassification) -> dict[str, object]:
    series = getattr(item, "series", None)
    if not isinstance(series, tuple) or not series:
        raise ValueError("TRAIN worker input has no series metadata")
    lengths = tuple(sorted({len(value.values) for value in series}))
    return {
        "train_size": len(series),
        "train_lengths": list(lengths),
        "test_size": source.test_size,
        "test_lengths": list(source.test_lengths),
    }


def _dataset_event(
    record: object, digest: str, counts: Exp426Counts,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "kind": "dataset_complete",
        "dataset": record.dataset,
        "origin": record.origin,
        "status": record.status,
        "record_sha256": digest,
        "backend": "cpu",
        "workers": 12,
        **dict(metadata),
        "completed": counts.completed,
        "total": counts.total,
    }


def _existing_event_datasets(output: Path) -> set[str]:
    return {
        item["dataset"]
        for item in _load_events(output)
        if item.get("kind") == "dataset_complete" and isinstance(item.get("dataset"), str)
    }


def _hook_call(
    hook: Callable[[str, str | None], None] | None,
    name: str, dataset: str | None,
) -> None:
    if hook is not None:
        hook(name, dataset)


def _commit_with_logs(
    store: Exp426CheckpointStore,
    logger: _RuntimeLogger,
    record: object,
    metadata: Mapping[str, object],
    state: str,
    hook: Callable[[str, str | None], None] | None,
    counts: Exp426Counts,
) -> Exp426Counts:
    logger.write_progress(
        state, counts,
        current_dataset=record.dataset, metadata=metadata,
    )
    digest = store.commit(record)
    _hook_call(hook, "after_record_commit", record.dataset)
    if record.origin == "exp424_import":
        imported = counts.imported + 1
        verified = counts.verified_train_abstentions
    elif record.origin == "verified_train_abstention":
        imported = counts.imported
        verified = counts.verified_train_abstentions + 1
    else:
        raise ValueError("checkpoint record has an unknown origin")
    updated_counts = Exp426Counts(
        total=counts.total,
        completed=counts.completed + 1,
        imported=imported,
        verified_train_abstentions=verified,
        abstained=verified,
        failed=counts.failed,
    )
    logger.append_event(_dataset_event(record, digest, updated_counts, metadata))
    _hook_call(hook, "after_event_append", record.dataset)
    logger.write_progress(
        state, updated_counts, current_dataset=record.dataset, metadata=metadata,
    )
    _hook_call(hook, "after_progress_replace", record.dataset)
    return updated_counts


def _reconcile_completed_event(
    store: Exp426CheckpointStore, logger: _RuntimeLogger,
) -> None:
    completed = set(store.completed_digests())
    event_datasets = _existing_event_datasets(logger.output)
    missing = sorted(completed - event_datasets)
    if not missing:
        return
    if len(missing) != 1 or logger.progress.get("current_dataset") != missing[0]:
        raise ValueError("completed checkpoint has no recoverable runtime event")
    metadata = logger.progress.get("dataset_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("completed checkpoint runtime metadata is absent")
    record = store.load_record(missing[0])
    counts = _counts_from_store(store)
    digest = store.completed_digests()[missing[0]]
    logger.append_event(_dataset_event(record, digest, counts, metadata))


def _run_session(
    store: Exp426CheckpointStore,
    logger: _RuntimeLogger,
    frozen_context: object,
    source: object,
    database: object,
    dependencies: _RuntimeDependencies,
    hook: Callable[[str, str | None], None] | None,
    *, is_new: bool,
) -> Exp426RuntimeResult:
    counts = _counts_from_store(store)
    if not is_new:
        _reconcile_completed_event(store, logger)
    if counts.completed == counts.total:
        if logger.progress.get("state") != "complete_review_only":
            logger.transition("finalizing", counts)
            logger.stop_heartbeat()
            logger.transition("complete_review_only", counts)
        return Exp426RuntimeResult("complete_review_only", counts, logger.output, store.root)

    if is_new:
        logger.transition("initializing", counts)
    logger.start_heartbeat()
    current_dataset: str | None = None
    current_metadata: Mapping[str, object] | None = None
    try:
        logger.transition("validating_source", counts)
        completed = set(store.completed_digests())

        def missing_source_tasks() -> Iterator[object]:
            for task in dependencies.source_tasks(source):
                dataset = getattr(task, "dataset", None)
                if not isinstance(dataset, str):
                    raise ValueError("source task dataset is invalid")
                if dataset not in completed:
                    yield task

        classifications: list[Exp424SourceClassification] = []
        for result in dependencies.worker_runner(
            missing_source_tasks(), workers=12, prefetch=24
        ):
            value = getattr(result, "value", None)
            if not isinstance(value, Exp424SourceClassification):
                raise ValueError("source worker returned an invalid classification")
            current_dataset = value.dataset
            current_metadata = _metadata_from_classification(value)
            classifications.append(value)
            _hook_call(hook, "after_worker_result", value.dataset)
        dependencies.source_revalidate(source)

        logger.transition("importing", counts)
        terminal: list[Exp424SourceClassification] = []
        for item in classifications:
            if item.imported_record is None:
                terminal.append(item)
                continue
            current_dataset = item.dataset
            current_metadata = _metadata_from_classification(item)
            counts = _commit_with_logs(
                store, logger, item.imported_record, current_metadata,
                "importing", hook, counts,
            )

        logger.transition("verifying_train", counts)
        metadata_by_dataset: dict[str, Mapping[str, object]] = {}

        def train_tasks() -> Iterator[Exp426TrainWorkerTask]:
            nonlocal current_dataset, current_metadata
            for item in terminal:
                dependencies.verify_database(database, f"train:{item.dataset}:before")
                train = dependencies.load_train(database, item.dataset, frozen_context)
                dependencies.verify_database(database, f"train:{item.dataset}:after")
                metadata = _metadata_from_train(train, item)
                metadata_by_dataset[item.dataset] = metadata
                current_dataset = item.dataset
                current_metadata = metadata
                logger.write_progress(
                    "verifying_train", counts,
                    current_dataset=item.dataset, metadata=metadata,
                )
                yield Exp426TrainWorkerTask(
                    item.dataset, train, store.provenance.fold_contract_sha256,
                    item.source_record_sha256, item.source_result_sha256,
                )

        train_results: list[WorkerEvidence] = []
        for result in dependencies.worker_runner(train_tasks(), workers=12, prefetch=24):
            value = getattr(result, "value", None)
            if not isinstance(value, WorkerEvidence):
                raise ValueError("TRAIN worker returned invalid evidence")
            current_dataset = value.dataset
            current_metadata = metadata_by_dataset[value.dataset]
            train_results.append(value)
            _hook_call(hook, "after_worker_result", value.dataset)
        dependencies.source_revalidate(source)
        for evidence in train_results:
            current_dataset = evidence.dataset
            current_metadata = metadata_by_dataset[evidence.dataset]
            counts = _commit_with_logs(
                store, logger, evidence.record, current_metadata,
                "verifying_train", hook, counts,
            )

        if counts.completed != counts.total or counts.failed:
            raise ValueError("runtime population is incomplete")
        logger.transition("finalizing", counts)
        logger.stop_heartbeat()
        logger.transition("complete_review_only", counts)
        return Exp426RuntimeResult("complete_review_only", counts, logger.output, store.root)
    except BaseException as error:
        logger.stop_heartbeat()
        logger.errors += 1
        latest = {"type": type(error).__name__, "state": logger.progress.get("state")}
        logger.transition(
            "failed", _counts_from_store(store),
            current_dataset=current_dataset,
            metadata=current_metadata,
            latest_error=latest,
        )
        raise


def run_exp426_new(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
    frozen_context: object,
    database: object,
    source: object,
    *,
    _dependencies: _RuntimeDependencies | None = None,
    _hook: Callable[[str, str | None], None] | None = None,
) -> Exp426RuntimeResult:
    output = Path(output)
    checkpoint = Path(checkpoint)
    if output.exists() or output.is_symlink() or checkpoint.exists() or checkpoint.is_symlink():
        raise ValueError("new runtime roots must both be absent")
    dependencies = _production_dependencies() if _dependencies is None else _dependencies
    if not isinstance(dependencies, _RuntimeDependencies):
        raise ValueError("runtime dependencies are invalid")
    store = Exp426CheckpointStore.create(checkpoint, provenance, checkpoint_context)
    logger = _RuntimeLogger.create(
        output, store.provenance.run_identity, checkpoint,
        heartbeat_seconds=dependencies.heartbeat_seconds,
    )
    try:
        return _run_session(
            store, logger, frozen_context, source, database,
            dependencies, _hook, is_new=True,
        )
    finally:
        logger.close()


def run_exp426_resume(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
    frozen_context: object,
    database: object,
    source: object,
    *,
    _dependencies: _RuntimeDependencies | None = None,
    _hook: Callable[[str, str | None], None] | None = None,
) -> Exp426RuntimeResult:
    output = Path(output)
    checkpoint = Path(checkpoint)
    dependencies = _production_dependencies() if _dependencies is None else _dependencies
    if not isinstance(dependencies, _RuntimeDependencies):
        raise ValueError("runtime dependencies are invalid")
    store = Exp426CheckpointStore.open(checkpoint, provenance, checkpoint_context)
    progress = _canonical_object(output / "progress.json", "runtime progress")
    counts = _counts_from_store(store)
    if counts.completed == counts.total and progress.get("state") == "complete_review_only":
        return Exp426RuntimeResult("complete_review_only", counts, output, checkpoint)
    logger = _RuntimeLogger.open(
        output, store.provenance.run_identity, checkpoint,
        heartbeat_seconds=dependencies.heartbeat_seconds,
        process_is_live=dependencies.process_is_live,
    )
    try:
        return _run_session(
            store, logger, frozen_context, source, database,
            dependencies, _hook, is_new=False,
        )
    finally:
        logger.close()


def inspect_exp426(
    output: Path, checkpoint: Path, *, expected_run_identity: str | None = None,
) -> Exp426RuntimeInspection:
    output = Path(output)
    checkpoint_inspection = Exp426CheckpointStore.inspect_read_only(
        checkpoint, expected_run_identity=expected_run_identity
    )
    if not output.is_dir() or any((output / name).is_symlink() for name in _OUTPUT_FILES):
        raise ValueError("runtime output root is invalid")
    if not all((output / name).is_file() for name in _OUTPUT_FILES):
        raise ValueError("runtime output inventory is incomplete")
    progress = _canonical_object(output / "progress.json", "runtime progress")
    _load_events(output)
    _canonical_object(output / "run.pid", "runtime PID")
    if (
        progress.get("run_identity") != checkpoint_inspection.run_identity
        or not isinstance(progress.get("counts"), dict)
        or progress["counts"].get("completed") != checkpoint_inspection.completed
        or progress["counts"].get("total") != checkpoint_inspection.total
    ):
        raise ValueError("runtime progress and checkpoint disagree")
    return Exp426RuntimeInspection(progress, checkpoint_inspection)
