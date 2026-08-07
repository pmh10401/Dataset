"""Local-only, atomic progress status for an Exp419 worker."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import ClassVar

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .contracts import RuntimeLogRecord


_SCHEMA = "exp419.worker-status.v1"
_STATUS_FILENAME = "exp419_worker_status.json"
_PHASES = frozenset(
    {
        "worker_starting",
        "context",
        "dataset_starting",
        "train_loading",
        "preprocessing",
        "oof",
        "final_fit",
        "qualification",
        "test_scoring",
        "checkpoint",
        "worker_completed",
        "worker_failed",
    }
)


def _non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class WorkerStatus:
    """A visibility-only worker snapshot with no data-derived payloads."""

    shard_index: int
    worker_pid: int
    expected_datasets: int
    completed_datasets: int
    phase: str
    dataset: str | None
    dataset_index: int | None
    seed: int | None
    fold: int | None
    error: str

    _PAYLOAD_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "shard_index",
            "worker_pid",
            "expected_datasets",
            "completed_datasets",
            "phase",
            "dataset",
            "dataset_index",
            "seed",
            "fold",
            "error",
        }
    )

    def __post_init__(self) -> None:
        shard_index = _non_negative_integer(self.shard_index, "shard_index")
        worker_pid = _non_negative_integer(self.worker_pid, "worker_pid")
        expected = _non_negative_integer(self.expected_datasets, "expected_datasets")
        completed = _non_negative_integer(self.completed_datasets, "completed_datasets")
        if expected == 0:
            raise ValueError("expected_datasets must be positive")
        if completed > expected:
            raise ValueError("completed_datasets must not exceed expected_datasets")
        if not isinstance(self.phase, str) or self.phase not in _PHASES:
            raise ValueError("phase is not an Exp419 worker status phase")
        if self.dataset is None:
            if self.dataset_index is not None:
                raise ValueError("dataset_index requires dataset")
        else:
            if not isinstance(self.dataset, str) or not self.dataset:
                raise ValueError("dataset must be a non-empty string or None")
            if type(self.dataset_index) is not int or not 0 <= self.dataset_index < expected:
                raise ValueError("dataset_index must name an expected dataset")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a non-negative integer or None")
        if self.fold is not None and (type(self.fold) is not int or not 0 <= self.fold < 5):
            raise ValueError("fold must be an OOF fold index or None")
        if not isinstance(self.error, str) or "\n" in self.error or len(self.error) > 512:
            raise ValueError("error must be a one-line string at most 512 characters")
        object.__setattr__(self, "shard_index", shard_index)
        object.__setattr__(self, "worker_pid", worker_pid)
        object.__setattr__(self, "expected_datasets", expected)
        object.__setattr__(self, "completed_datasets", completed)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "shard_index": self.shard_index,
            "worker_pid": self.worker_pid,
            "expected_datasets": self.expected_datasets,
            "completed_datasets": self.completed_datasets,
            "phase": self.phase,
            "dataset": self.dataset,
            "dataset_index": self.dataset_index,
            "seed": self.seed,
            "fold": self.fold,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "WorkerStatus":
        if type(payload) is not dict or set(payload) != cls._PAYLOAD_FIELDS:
            raise ValueError("worker status field set is invalid")
        if payload["schema"] != _SCHEMA:
            raise ValueError("worker status schema is invalid")
        return cls(
            shard_index=payload["shard_index"],
            worker_pid=payload["worker_pid"],
            expected_datasets=payload["expected_datasets"],
            completed_datasets=payload["completed_datasets"],
            phase=payload["phase"],
            dataset=payload["dataset"],
            dataset_index=payload["dataset_index"],
            seed=payload["seed"],
            fold=payload["fold"],
            error=payload["error"],
        )


def write_worker_status(root: Path, status: WorkerStatus) -> Path:
    """Atomically replace the visibility snapshot or raise an I/O error."""
    if not isinstance(root, Path):
        raise ValueError("worker status root must be a Path")
    if not isinstance(status, WorkerStatus):
        raise ValueError("status must be a WorkerStatus")
    root.mkdir(parents=True, exist_ok=True)
    target = root / _STATUS_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_STATUS_FILENAME}.{os.getpid()}.", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(status.to_payload()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def append_runtime_log(root: Path, record: RuntimeLogRecord) -> Path:
    """Durably append one safe worker observation without reading any log."""
    if not isinstance(root, Path):
        raise ValueError("runtime log root must be a Path")
    if not isinstance(record, RuntimeLogRecord):
        raise ValueError("record must be a RuntimeLogRecord")
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"worker-{record.shard_index}.runtime.jsonl"
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as error:
        raise ValueError("runtime log must be a regular non-symlink file") from error
    try:
        payload = record.canonical_json_line()
        count = os.write(descriptor, payload)
        if count != len(payload):
            raise OSError("short runtime log append")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


__all__ = ["WorkerStatus", "append_runtime_log", "write_worker_status"]
