"""Atomic progress, durable events, and PID lifecycle for Exp435."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event, RLock, Thread
import time
from typing import Callable, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes


_OUTPUT_FILES = frozenset(
    {
        "progress_antigravity.json",
        "events_antigravity.jsonl",
        "parent_antigravity.log",
        "run_antigravity.pid",
    }
)
_FORBIDDEN_KEYS = ("label", "truth", "ground_truth", "anomaly_position", "actual_values", "raw_values")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_new(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalise_error(error: object | None, stage: str | None = None) -> dict[str, object] | None:
    if error is None:
        return None
    if isinstance(error, Mapping):
        error_type = str(error.get("type", "Error"))
        message = str(error.get("message", error_type))
    elif isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error) or error_type
    else:
        error_type = type(error).__name__
        message = str(error)
    message = " ".join(message.split()).strip()[:1000] or error_type
    payload: dict[str, object] = {"type": error_type, "message": message}
    if stage is not None:
        payload["stage"] = stage
    return payload


def _assert_label_free(value: object, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in _FORBIDDEN_KEYS) or key_text in {"values", "series", "targets", "y_true"}:
                raise ValueError(f"{path} contains forbidden label/value field")
            _assert_label_free(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_label_free(item, path=f"{path}[{index}]")


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


class ProgressWriter:
    """Parent-owned writer for progress JSON, JSONL events and text logs."""

    def __init__(
        self,
        output: Path,
        *,
        run_identity: str,
        checkpoint: Path,
        total_datasets: int,
        backend: str,
        heartbeat_seconds: float | None = 30.0,
        process_is_live: Callable[[int], bool] = _process_is_live,
    ) -> None:
        if not isinstance(run_identity, str) or not run_identity:
            raise ValueError("run identity is invalid")
        if type(total_datasets) is not int or total_datasets <= 0:
            raise ValueError("total datasets is invalid")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend is invalid")
        if heartbeat_seconds is not None and (
            isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat interval is invalid")
        self.output = Path(output)
        self.run_identity = run_identity
        self.checkpoint = Path(checkpoint)
        self.total_datasets = total_datasets
        self.backend = backend
        self.heartbeat_seconds = heartbeat_seconds
        self.process_is_live = process_is_live
        self.started = time.monotonic()
        self.elapsed_offset = 0.0
        self.lock = RLock()
        self.stop_event = Event()
        self.heartbeat_thread: Thread | None = None
        self.progress: dict[str, object] = {}
        self._closed = False

    @classmethod
    def create(
        cls,
        output: Path,
        *,
        run_identity: str,
        checkpoint: Path,
        total_datasets: int,
        backend: str,
        heartbeat_seconds: float | None = 30.0,
    ) -> "ProgressWriter":
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise ValueError("runtime output root already exists")
        output.mkdir(mode=0o700)
        _fsync_directory(output.parent)
        _write_new(output / "events_antigravity.jsonl", b"")
        _write_new(output / "parent_antigravity.log", b"")
        writer = cls(
            output,
            run_identity=run_identity,
            checkpoint=checkpoint,
            total_datasets=total_datasets,
            backend=backend,
            heartbeat_seconds=heartbeat_seconds,
        )
        initial = writer._make_progress(
            state="initializing",
            stage="initializing",
            current_dataset=None,
            metadata=None,
            completed=0,
            success=0,
            abstention=0,
            error=0,
            candidate_count=0,
            latest_error=None,
        )
        _write_new(output / "progress_antigravity.json", canonical_json_bytes(initial))
        _write_new(output / "run_antigravity.pid", canonical_json_bytes(writer._pid_payload(True)))
        writer.progress = initial
        writer.append_event({"kind": "state", "state": "initializing"})
        return writer

    @classmethod
    def open_existing(
        cls,
        output: Path,
        *,
        run_identity: str,
        checkpoint: Path,
        process_is_live: Callable[[int], bool] = _process_is_live,
        heartbeat_seconds: float | None = 30.0,
    ) -> "ProgressWriter":
        output = Path(output)
        if not output.is_dir() or set(item.name for item in output.iterdir()) != set(_OUTPUT_FILES):
            raise ValueError("runtime output inventory is invalid")
        progress = _canonical_object(output / "progress_antigravity.json", "runtime progress")
        pid_payload = _canonical_object(output / "run_antigravity.pid", "runtime PID")
        if progress.get("run_identity") != run_identity or pid_payload.get("run_identity") != run_identity:
            raise ValueError("runtime run identity does not match")
        old_pid = pid_payload.get("pid")
        if pid_payload.get("active") is True and type(old_pid) is int and process_is_live(old_pid):
            raise ValueError("runtime process is still active")
        total = progress.get("total_datasets", progress.get("total"))
        backend = progress.get("backend")
        if type(total) is not int or not isinstance(backend, str):
            raise ValueError("runtime progress accounting is invalid")
        writer = cls(
            output,
            run_identity=run_identity,
            checkpoint=checkpoint,
            total_datasets=total,
            backend=backend,
            heartbeat_seconds=heartbeat_seconds,
            process_is_live=process_is_live,
        )
        elapsed = progress.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            writer.elapsed_offset = max(0.0, float(elapsed))
        writer.progress = progress
        writer._atomic_pid(True)
        return writer

    # ``open`` is intentionally a readable spelling for callers that model the
    # checkpoint store's open_existing operation.
    open = open_existing

    def _elapsed(self) -> float:
        return self.elapsed_offset + max(0.0, time.monotonic() - self.started)

    def _pid_payload(self, active: bool) -> dict[str, object]:
        return {
            "schema": "exp435.runtime-pid.v1",
            "run_identity": self.run_identity,
            "pid": os.getpid(),
            "active": active,
            "updated_at": _now(),
        }

    def _atomic_pid(self, active: bool) -> None:
        _atomic_replace(self.output / "run_antigravity.pid", canonical_json_bytes(self._pid_payload(active)))

    def _make_progress(
        self,
        *,
        state: str,
        stage: str,
        current_dataset: str | None,
        metadata: Mapping[str, object] | None,
        completed: int,
        success: int,
        abstention: int,
        error: int,
        candidate_count: int,
        latest_error: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if type(completed) is not int or completed < 0 or completed > self.total_datasets:
            raise ValueError("completed dataset count is invalid")
        if any(type(value) is not int or value < 0 for value in (success, abstention, error, candidate_count)):
            raise ValueError("runtime counts are invalid")
        if success + abstention + error != completed:
            raise ValueError("runtime count conservation is invalid")
        if metadata is not None:
            _assert_label_free(metadata)
            metadata = dict(metadata)
        elapsed = self._elapsed()
        throughput = completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total_datasets - completed
        eta = remaining / throughput if throughput > 0 else None
        return {
            "schema": "exp435.runtime-progress.v1",
            "run_identity": self.run_identity,
            "updated_at": _now(),
            "pid": os.getpid(),
            "heartbeat": _now(),
            "state": state,
            "stage": stage,
            "backend": self.backend,
            "current_dataset": current_dataset,
            "dataset_metadata": metadata,
            "completed": completed,
            "total": self.total_datasets,
            "completed_datasets": completed,
            "total_datasets": self.total_datasets,
            "success": success,
            "abstention": abstention,
            "error": error,
            "candidate_count": candidate_count,
            "throughput": throughput,
            "throughput_datasets_per_second": throughput,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "checkpoint": str(self.checkpoint),
            "latest_error": None if latest_error is None else dict(latest_error),
        }

    def update(
        self,
        *,
        stage: str,
        current_dataset: str | None,
        dataset_metadata: Mapping[str, object] | None,
        completed: int,
        success: int,
        abstention: int,
        error: int,
        candidate_count: int,
        latest_error: Mapping[str, object] | None = None,
        state: str = "running",
    ) -> None:
        if not isinstance(stage, str) or not stage:
            raise ValueError("runtime stage is invalid")
        payload = self._make_progress(
            state=state,
            stage=stage,
            current_dataset=current_dataset,
            metadata=dataset_metadata,
            completed=completed,
            success=success,
            abstention=abstention,
            error=error,
            candidate_count=candidate_count,
            latest_error=latest_error,
        )
        with self.lock:
            _atomic_replace(self.output / "progress_antigravity.json", canonical_json_bytes(payload))
            self.progress = payload
            self._atomic_pid(True)

    write_progress = update

    def append_event(self, payload: Mapping[str, object]) -> None:
        _assert_label_free(payload)
        event = dict(payload)
        event.setdefault("schema", "exp435.runtime-event.v1")
        event.setdefault("run_identity", self.run_identity)
        event.setdefault("recorded_at", _now())
        with self.lock:
            _append_durable(self.output / "events_antigravity.jsonl", canonical_json_bytes(event) + b"\n")

    def append_log(self, message: str) -> None:
        if not isinstance(message, str) or "\n" in message:
            raise ValueError("runtime log message is invalid")
        # Raw series and labels must never be interpolated into the human log.
        if any(token in message.lower() for token in ("label=", "values=", "anomaly_position")):
            raise ValueError("runtime log contains forbidden label/value field")
        with self.lock:
            _append_durable(self.output / "parent_antigravity.log", f"{_now()} {message}\n".encode())

    def transition(
        self,
        state: str,
        *,
        stage: str,
        current_dataset: str | None,
        dataset_metadata: Mapping[str, object] | None,
        completed: int,
        success: int,
        abstention: int,
        error: int,
        candidate_count: int,
        latest_error: Mapping[str, object] | None = None,
    ) -> None:
        self.update(
            state=state,
            stage=stage,
            current_dataset=current_dataset,
            dataset_metadata=dataset_metadata,
            completed=completed,
            success=success,
            abstention=abstention,
            error=error,
            candidate_count=candidate_count,
            latest_error=latest_error,
        )
        self.append_event({"kind": "state", "state": state, "stage": stage, "dataset": current_dataset})
        self.append_log(f"state={state} stage={stage} completed={completed}/{self.total_datasets}")

    def mark_interrupted(self, current_dataset: str | None, error: object | None = None) -> None:
        current = self.progress
        self.transition(
            "interrupted",
            stage=str(current.get("stage", "running")),
            current_dataset=current_dataset,
            dataset_metadata=current.get("dataset_metadata") if isinstance(current.get("dataset_metadata"), Mapping) else None,
            completed=int(current.get("completed", 0)),
            success=int(current.get("success", 0)),
            abstention=int(current.get("abstention", 0)),
            error=int(current.get("error", 0)),
            candidate_count=int(current.get("candidate_count", 0)),
            latest_error=_normalise_error(error, str(current.get("stage", "running"))),
        )

    def mark_failed(self, current_dataset: str | None, error: object) -> None:
        current = self.progress
        current_completed = int(current.get("completed", 0))
        current_errors = int(current.get("error", 0))
        # The failed dataset is terminal accounting even though no checkpoint
        # record is published.  Keeping conservation explicit prevents a
        # failed root from looking resumable.
        failed_completed = min(self.total_datasets, current_completed + 1)
        self.transition(
            "failed",
            stage=str(current.get("stage", "running")),
            current_dataset=current_dataset,
            dataset_metadata=current.get("dataset_metadata") if isinstance(current.get("dataset_metadata"), Mapping) else None,
            completed=failed_completed,
            success=int(current.get("success", 0)),
            abstention=int(current.get("abstention", 0)),
            error=max(1, current_errors + 1),
            candidate_count=int(current.get("candidate_count", 0)),
            latest_error=_normalise_error(error, str(current.get("stage", "running"))),
        )

    def mark_complete(self, *, candidate_count: int | None = None) -> None:
        current = self.progress
        self.transition(
            "checkpoint_complete",
            stage="finalizing",
            current_dataset=None,
            dataset_metadata=None,
            completed=self.total_datasets,
            success=int(current.get("success", 0)),
            abstention=int(current.get("abstention", 0)),
            error=int(current.get("error", 0)),
            candidate_count=int(current.get("candidate_count", 0) if candidate_count is None else candidate_count),
            latest_error=None,
        )

    def start_heartbeat(self) -> None:
        if self.heartbeat_seconds is None or self.heartbeat_thread is not None:
            return
        self.stop_event.clear()

        def loop() -> None:
            while not self.stop_event.wait(self.heartbeat_seconds):
                with self.lock:
                    if self._closed:
                        return
                    payload = dict(self.progress)
                    payload["updated_at"] = _now()
                    payload["heartbeat"] = _now()
                    payload["elapsed_seconds"] = self._elapsed()
                    _atomic_replace(self.output / "progress_antigravity.json", canonical_json_bytes(payload))
                    self.progress = payload
                    self._atomic_pid(True)

        self.heartbeat_thread = Thread(target=loop, name="exp435-heartbeat", daemon=True)
        self.heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self.stop_event.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=max(1.0, (self.heartbeat_seconds or 1.0) + 1.0))
            self.heartbeat_thread = None

    def close(self) -> None:
        if self._closed:
            return
        self.stop_heartbeat()
        with self.lock:
            self._closed = True
            self._atomic_pid(False)


__all__ = ["ProgressWriter"]
