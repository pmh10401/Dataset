"""Atomic, self-written Exp439 monitoring artifacts for long-running execution."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from threading import Event, RLock, Thread
from typing import Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .contracts_antigravity import CHANNELS_ANTIGRAVITY, reject_label_like_fields_antigravity


_SHA256_ANTIGRAVITY = re.compile(r"[0-9a-f]{64}\Z")
_FILES_ANTIGRAVITY = {
    "progress_antigravity.json",
    "events_antigravity.jsonl",
    "parent_antigravity.log",
    "run_antigravity.pid",
}
_STAGES_ANTIGRAVITY = {
    "idle",
    "loading",
    "dispatching",
    "awaiting_worker_result",
    "committing",
}
_METADATA_KEYS_ANTIGRAVITY = {
    "train_count",
    "test_count",
    "train_length_histogram",
    "test_length_histogram",
    "train_value_count",
    "test_value_count",
    "train_bytes",
    "test_bytes",
}


def _now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_antigravity(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_antigravity(path.parent)


def _replace_antigravity(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".antigravity_{path.name}_{uuid4().hex}.tmp")
    _write_new_antigravity(temporary, raw)
    try:
        os.replace(temporary, path)
        _fsync_directory_antigravity(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append_antigravity(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_progress_antigravity(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp439 progress is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("Exp439 progress is not canonical JSON")
    return value


def _nonnegative_integer_antigravity(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is invalid")
    return value


def _histogram_antigravity(value: object, expected_count: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("length histogram is invalid")
    result: dict[str, int] = {}
    for key, count in value.items():
        if (
            type(key) is not str
            or not key.isdecimal()
            or int(key) < 1
            or type(count) is not int
            or count < 0
        ):
            raise ValueError("length histogram is invalid")
        result[key] = count
    if sum(result.values()) != expected_count:
        raise ValueError("length histogram count is inconsistent")
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _metadata_antigravity(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _METADATA_KEYS_ANTIGRAVITY:
        raise ValueError("dataset metadata is invalid")
    result = dict(value)
    for name in (
        "train_count", "test_count", "train_value_count", "test_value_count",
        "train_bytes", "test_bytes",
    ):
        result[name] = _nonnegative_integer_antigravity(result[name], name)
    result["train_length_histogram"] = _histogram_antigravity(
        result["train_length_histogram"], int(result["train_count"])
    )
    result["test_length_histogram"] = _histogram_antigravity(
        result["test_length_histogram"], int(result["test_count"])
    )
    return result


def _channel_counts_antigravity(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(CHANNELS_ANTIGRAVITY):
        raise ValueError("per-channel candidate counts are invalid")
    result = {
        channel: _nonnegative_integer_antigravity(value[channel], channel)
        for channel in CHANNELS_ANTIGRAVITY
    }
    return result


def _validate_payload_antigravity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Exp439 progress schema is invalid")
    payload = dict(value)
    required = {
        "schema", "context_hash", "state", "stage", "backend", "pid",
        "current_dataset", "dataset_metadata", "completed", "total",
        "import_count", "recompute_count", "success_count", "abstention_count",
        "error_count", "datasets_per_second", "elapsed_seconds", "eta_seconds",
        "checkpoint", "per_channel_candidate_counts", "latest_error",
        "created_at", "updated_at",
    }
    if (
        set(payload) != required
        or payload.get("schema") != "exp439.runtime-progress.v1"
        or not isinstance(payload.get("context_hash"), str)
        or _SHA256_ANTIGRAVITY.fullmatch(str(payload["context_hash"])) is None
        or payload.get("state") not in {"running", "interrupted", "complete", "failed"}
        or payload.get("stage") not in _STAGES_ANTIGRAVITY
        or payload.get("backend") not in {"cpu", "numba", "mlx", "mps"}
        or type(payload.get("pid")) is not int
        or int(payload["pid"]) < 1
        or not isinstance(payload.get("created_at"), str)
        or not isinstance(payload.get("updated_at"), str)
    ):
        raise ValueError("Exp439 progress schema is invalid")
    total = _nonnegative_integer_antigravity(payload["total"], "total")
    completed = _nonnegative_integer_antigravity(payload["completed"], "completed")
    if total < 1 or completed > total:
        raise ValueError("Exp439 progress counters are invalid")
    for name in (
        "import_count", "recompute_count", "success_count", "abstention_count", "error_count"
    ):
        _nonnegative_integer_antigravity(payload[name], name)
    if int(payload["success_count"]) + int(payload["abstention_count"]) > completed:
        raise ValueError("Exp439 successful outcomes exceed completed datasets")
    if int(payload["error_count"]) > total - completed:
        raise ValueError("Exp439 error count exceeds unfinished datasets")
    current = payload["current_dataset"]
    if current is not None and (type(current) is not str or not current):
        raise ValueError("current dataset is invalid")
    metadata = _metadata_antigravity(payload["dataset_metadata"])
    payload["dataset_metadata"] = metadata
    if current is None and metadata is not None:
        raise ValueError("dataset metadata requires a current dataset")
    if payload["state"] in {"interrupted", "complete", "failed"} and (
        payload["stage"] != "idle" or current is not None or metadata is not None
    ):
        raise ValueError("terminal progress must be idle")
    for name in ("datasets_per_second", "elapsed_seconds"):
        raw = payload[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) < 0.0:
            raise ValueError("progress timing is invalid")
    eta = payload["eta_seconds"]
    if eta is not None and (
        isinstance(eta, bool)
        or not isinstance(eta, (int, float))
        or not math.isfinite(float(eta))
        or float(eta) < 0.0
    ):
        raise ValueError("progress ETA is invalid")
    if payload["checkpoint"] is not None and (
        type(payload["checkpoint"]) is not str or not payload["checkpoint"]
    ):
        raise ValueError("progress checkpoint is invalid")
    payload["per_channel_candidate_counts"] = _channel_counts_antigravity(
        payload["per_channel_candidate_counts"]
    )
    reject_label_like_fields_antigravity(payload)
    return payload


def _synchronized_antigravity(method):
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class ProgressWriterAntigravity:
    """The parent process's only writer for progress, events, log, and PID."""

    def __init__(self, output: Path, payload: dict[str, object], *, mutable: bool) -> None:
        self.output = Path(output)
        self._payload = payload
        self._mutable = mutable
        self._started = time.monotonic()
        self._elapsed_offset = float(payload["elapsed_seconds"])
        self._lock = RLock()
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None

    @classmethod
    def create(
        cls,
        output: Path,
        context_hash: str,
        total_datasets: int,
        backend: str,
        *,
        pid: int,
    ) -> "ProgressWriterAntigravity":
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise ValueError("Exp439 progress root already exists")
        if (
            type(context_hash) is not str
            or _SHA256_ANTIGRAVITY.fullmatch(context_hash) is None
            or type(total_datasets) is not int
            or total_datasets < 1
            or backend not in {"cpu", "numba", "mlx", "mps"}
            or type(pid) is not int
            or pid < 1
        ):
            raise ValueError("Exp439 progress binding is invalid")
        output.mkdir(mode=0o700, parents=False)
        _fsync_directory_antigravity(output.parent)
        now = _now_antigravity()
        payload: dict[str, object] = {
            "schema": "exp439.runtime-progress.v1",
            "context_hash": context_hash,
            "state": "running",
            "stage": "idle",
            "backend": backend,
            "pid": pid,
            "current_dataset": None,
            "dataset_metadata": None,
            "completed": 0,
            "total": total_datasets,
            "import_count": 0,
            "recompute_count": 0,
            "success_count": 0,
            "abstention_count": 0,
            "error_count": 0,
            "datasets_per_second": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "checkpoint": None,
            "per_channel_candidate_counts": {channel: 0 for channel in CHANNELS_ANTIGRAVITY},
            "latest_error": None,
            "created_at": now,
            "updated_at": now,
        }
        _validate_payload_antigravity(payload)
        _write_new_antigravity(output / "progress_antigravity.json", canonical_json_bytes(payload))
        _write_new_antigravity(output / "events_antigravity.jsonl", b"")
        _write_new_antigravity(output / "parent_antigravity.log", b"")
        _write_new_antigravity(output / "run_antigravity.pid", f"{pid}\n".encode("ascii"))
        writer = cls(output, payload, mutable=True)
        writer.append_event({"event": "runtime_started", "backend": backend, "pid": pid})
        writer._append_log_antigravity("runtime_started")
        return writer

    @classmethod
    def open_existing(
        cls, output: Path, context_hash: str, *, mutable: bool
    ) -> "ProgressWriterAntigravity":
        output = Path(output)
        try:
            names = {path.name for path in output.iterdir()}
        except OSError as error:
            raise ValueError("Exp439 progress root is invalid") from error
        if names != _FILES_ANTIGRAVITY:
            raise ValueError("Exp439 progress inventory is invalid")
        payload = _validate_payload_antigravity(
            _read_progress_antigravity(output / "progress_antigravity.json")
        )
        if payload["context_hash"] != context_hash:
            raise ValueError("Exp439 progress context does not match")
        try:
            expected_pid = f"{payload['pid']}\n".encode("ascii")
            if (output / "run_antigravity.pid").read_bytes() != expected_pid:
                raise ValueError("Exp439 progress PID does not match")
            events = (output / "events_antigravity.jsonl").read_bytes()
            if events and not events.endswith(b"\n"):
                raise ValueError("Exp439 progress events contain a partial record")
            for line in events.splitlines():
                event = json.loads(line)
                if canonical_json_bytes(event) != line:
                    raise ValueError("Exp439 progress event is not canonical")
                reject_label_like_fields_antigravity(event)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp439 progress artifacts are invalid") from error
        if mutable and payload["state"] == "failed":
            raise ValueError("failed progress is immutable")
        if mutable and payload["state"] == "complete":
            raise ValueError("complete progress is immutable")
        return cls(output, payload, mutable=mutable)

    def _require_mutable_antigravity(self) -> None:
        if not self._mutable:
            raise ValueError("progress is read-only")
        if self._payload["state"] in {"complete", "failed"}:
            raise ValueError(f"{self._payload['state']} progress is immutable")

    @property
    def snapshot(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self._payload))

    def _write_payload_antigravity(self, payload: dict[str, object]) -> None:
        validated = _validate_payload_antigravity(payload)
        _replace_antigravity(
            self.output / "progress_antigravity.json",
            canonical_json_bytes(validated),
        )
        self._payload = validated

    @_synchronized_antigravity
    def update(
        self,
        *,
        stage: str,
        current_dataset: str | None = None,
        dataset_metadata: Mapping[str, object] | None = None,
        completed: int | None = None,
        import_count: int | None = None,
        recompute_count: int | None = None,
        success_count: int | None = None,
        abstention_count: int | None = None,
        error_count: int | None = None,
        checkpoint: str | None = None,
        per_channel_candidate_counts: Mapping[str, int] | None = None,
        latest_error: Mapping[str, object] | None = None,
    ) -> None:
        self._require_mutable_antigravity()
        if stage not in _STAGES_ANTIGRAVITY:
            raise ValueError("progress stage is invalid")
        proposed = dict(self._payload)
        proposed["state"] = "running"
        proposed["stage"] = stage
        proposed["current_dataset"] = current_dataset
        proposed["dataset_metadata"] = _metadata_antigravity(dataset_metadata)
        for name, value in (
            ("completed", completed),
            ("import_count", import_count),
            ("recompute_count", recompute_count),
            ("success_count", success_count),
            ("abstention_count", abstention_count),
            ("error_count", error_count),
        ):
            if value is not None:
                proposed[name] = _nonnegative_integer_antigravity(value, name)
        if checkpoint is not None:
            proposed["checkpoint"] = checkpoint
        if per_channel_candidate_counts is not None:
            proposed["per_channel_candidate_counts"] = _channel_counts_antigravity(
                per_channel_candidate_counts
            )
        if latest_error is not None:
            reject_label_like_fields_antigravity(latest_error)
            proposed["latest_error"] = dict(latest_error)
        elapsed = self._elapsed_offset + max(0.0, time.monotonic() - self._started)
        rate = float(proposed["completed"]) / elapsed if elapsed > 0.0 else 0.0
        remaining = int(proposed["total"]) - int(proposed["completed"])
        proposed["elapsed_seconds"] = elapsed
        proposed["datasets_per_second"] = rate
        proposed["eta_seconds"] = remaining / rate if rate > 0.0 else None
        proposed["updated_at"] = _now_antigravity()
        self._write_payload_antigravity(proposed)

    @_synchronized_antigravity
    def append_event(self, event: Mapping[str, object]) -> None:
        self._require_mutable_antigravity()
        if not isinstance(event, Mapping):
            raise ValueError("progress event must be an object")
        payload = dict(event)
        reject_label_like_fields_antigravity(payload)
        payload.setdefault("at", _now_antigravity())
        raw = canonical_json_bytes(payload) + b"\n"
        _append_antigravity(self.output / "events_antigravity.jsonl", raw)

    def _append_log_antigravity(self, message: str) -> None:
        _append_antigravity(
            self.output / "parent_antigravity.log",
            f"{_now_antigravity()} {message}\n".encode("utf-8"),
        )

    @_synchronized_antigravity
    def _mark_terminal_antigravity(
        self, state: str, *, completed: int | None = None, failure: str | None = None
    ) -> None:
        self._require_mutable_antigravity()
        proposed = dict(self._payload)
        if completed is not None:
            proposed["completed"] = _nonnegative_integer_antigravity(completed, "completed")
        if state == "complete" and proposed["completed"] != proposed["total"]:
            raise ValueError("complete progress must cover every dataset")
        if state == "failed":
            if type(failure) is not str or not failure:
                raise ValueError("failure code is invalid")
            details = (
                dict(proposed["latest_error"])
                if isinstance(proposed["latest_error"], Mapping)
                else {}
            )
            details["code"] = failure
            proposed["latest_error"] = details
        proposed["state"] = state
        proposed["stage"] = "idle"
        proposed["current_dataset"] = None
        proposed["dataset_metadata"] = None
        elapsed = self._elapsed_offset + max(0.0, time.monotonic() - self._started)
        proposed["elapsed_seconds"] = elapsed
        proposed["datasets_per_second"] = (
            float(proposed["completed"]) / elapsed if elapsed > 0.0 else 0.0
        )
        remaining = int(proposed["total"]) - int(proposed["completed"])
        proposed["eta_seconds"] = (
            remaining / float(proposed["datasets_per_second"])
            if proposed["datasets_per_second"] > 0.0 and state == "running"
            else None
        )
        proposed["updated_at"] = _now_antigravity()
        self._heartbeat_stop.set()
        self._write_payload_antigravity(proposed)
        terminal_event = {
            "event": state,
            "completed": proposed["completed"],
            "failure": failure,
            "at": _now_antigravity(),
        }
        _append_antigravity(
            self.output / "events_antigravity.jsonl",
            canonical_json_bytes(terminal_event) + b"\n",
        )
        log_message = state if failure is None else f"{state} {failure}"
        if failure is not None and isinstance(proposed["latest_error"], Mapping):
            dataset = proposed["latest_error"].get("dataset")
            cause_message = proposed["latest_error"].get("cause_message")
            if type(dataset) is str:
                log_message += f" dataset={dataset}"
            if type(cause_message) is str:
                log_message += f" cause={cause_message}"
        self._append_log_antigravity(log_message)

    def mark_interrupted(self) -> None:
        self._mark_terminal_antigravity("interrupted")

    def mark_complete(self, *, completed: int) -> None:
        self._mark_terminal_antigravity("complete", completed=completed)

    def mark_failed(self, failure: str) -> None:
        self._mark_terminal_antigravity("failed", failure=failure)

    def heartbeat(self) -> None:
        with self._lock:
            if not self._mutable or self._payload["state"] != "running":
                return
            current = self.snapshot
            self.update(
                stage=str(current["stage"]),
                current_dataset=current["current_dataset"],
                dataset_metadata=current["dataset_metadata"],
                completed=int(current["completed"]),
                import_count=int(current["import_count"]),
                recompute_count=int(current["recompute_count"]),
                success_count=int(current["success_count"]),
                abstention_count=int(current["abstention_count"]),
                error_count=int(current["error_count"]),
                checkpoint=current["checkpoint"],
                per_channel_candidate_counts=current["per_channel_candidate_counts"],
                latest_error=current["latest_error"],
            )

    def start_heartbeat(self, *, interval_seconds: float = 120.0) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0.0
        ):
            raise ValueError("heartbeat interval is invalid")
        with self._lock:
            self._require_mutable_antigravity()
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                raise ValueError("heartbeat is already running")
            self._heartbeat_stop.clear()

            def run() -> None:
                while not self._heartbeat_stop.wait(float(interval_seconds)):
                    self.heartbeat()

            self._heartbeat_thread = Thread(
                target=run,
                name="exp439-heartbeat-antigravity",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        with self._lock:
            thread = self._heartbeat_thread
            self._heartbeat_stop.set()
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._heartbeat_thread = None
