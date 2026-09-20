"""Strict parent-owned Exp436 v1 progress/event durability.

The checkpoint remains the scientific authority.  This module is deliberately
small: it persists only a replayable event prefix and a snapshot of that prefix
under one directory-FD flock; it never creates an auxiliary lock file.
"""
from __future__ import annotations

import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from threading import Event, RLock, Thread
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes


_ARTIFACTS = frozenset({"progress_antigravity.json", "events_antigravity.jsonl", "parent_antigravity.log", "run_antigravity.pid"})
_TEMP = re.compile(r"^\.(?:progress_antigravity\.json|run_antigravity\.pid)\.[0-9a-f]{32}\.tmp$")
_ACTIVE = frozenset({"loading", "dispatching", "awaiting_worker_result", "committing"})
_TERMINAL = frozenset({"interrupted", "failed", "checkpoint_complete"})
_ZERO = "0" * 64
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROGRESS_KEYS = frozenset({"schema", "run_identity", "context_sha256", "pid", "pid_active", "heartbeat", "updated_at", "state", "stage", "backend", "current_dataset", "dataset_metadata", "in_flight_datasets", "completed_datasets", "total_datasets", "recomputed_datasets", "success", "abstention", "error", "excluded_train_rows", "nonfinite_test_rows", "candidate_count", "datasets_per_second", "elapsed_seconds", "eta_seconds", "checkpoint_path", "latest_error", "last_event_sequence", "last_event_sha256", "progress_sha256"})
_PID_KEYS = frozenset({"schema", "run_identity", "context_sha256", "checkpoint_path", "backend", "total_datasets", "pid", "pid_active", "heartbeat", "state", "pid_sha256"})
_EVENT_KEYS = frozenset({"schema", "sequence", "previous_event_sha256", "run_identity", "context_sha256", "timestamp", "kind", "payload", "event_sha256"})
_COUNTERS = ("completed_datasets", "recomputed_datasets", "success", "abstention", "error", "excluded_train_rows", "nonfinite_test_rows", "candidate_count")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _digest(payload: Mapping[str, object]) -> str:
    return sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _canonical(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _open_directory(path: Path) -> int:
    try:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("runtime output root is invalid")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("runtime output root is invalid")
        return fd
    except OSError as error:
        raise ValueError("runtime output root is invalid") from error


def _write_all(fd: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        count = os.write(fd, raw[offset:])
        if count <= 0:
            raise OSError("short durable write")
        offset += count


def _atomic_replace(output_fd: int, name: str, raw: bytes) -> None:
    temporary = f".{name}.{uuid4().hex}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=output_fd)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, name, src_dir_fd=output_fd, dst_dir_fd=output_fd)
        os.fsync(output_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=output_fd)
        except FileNotFoundError:
            pass


def _append(output_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), dir_fd=output_fd)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def _create(output_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=output_fd)
    try:
        _write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(output_fd)


def _validate_metadata(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping) or set(metadata) != {"train_count", "test_count", "train_length_histogram", "test_length_histogram"}:
        raise ValueError("dataset metadata is invalid")
    result = dict(metadata)
    for count in ("train_count", "test_count"):
        if type(result[count]) is not int or result[count] < 0 or (count == "train_count" and result[count] < 1):
            raise ValueError("dataset metadata is invalid")
    for name, count in (("train_length_histogram", result["train_count"]), ("test_length_histogram", result["test_count"])):
        histogram = result[name]
        if not isinstance(histogram, Mapping):
            raise ValueError("dataset metadata is invalid")
        pairs = list(histogram.items())
        if any(
            type(key) is not str
            or not key.isdecimal()
            or key.startswith("0")
            or int(key) < 1
            or type(item) is not int
            or item < 0
            for key, item in pairs
        ):
            raise ValueError("dataset metadata is invalid")
        if sum(item for _, item in pairs) != count:
            raise ValueError("dataset metadata is invalid")
        result[name] = dict(sorted(pairs))
    return result


def _validate_error(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"dataset", "prior_stage", "type", "message"}:
        raise ValueError("latest error is invalid")
    result = dict(value)
    if result["dataset"] is not None and (not isinstance(result["dataset"], str) or not result["dataset"]):
        raise ValueError("latest error is invalid")
    if result["prior_stage"] not in _ACTIVE | {"idle"} or not isinstance(result["type"], str) or not isinstance(result["message"], str):
        raise ValueError("latest error is invalid")
    return result


def _validate_counters(values: Mapping[str, object], *, total: int) -> None:
    if any(type(values[name]) is not int or values[name] < 0 for name in _COUNTERS):
        raise ValueError("progress counters are invalid")
    if values["completed_datasets"] > total or values["recomputed_datasets"] != values["completed_datasets"] or values["success"] + values["abstention"] != values["completed_datasets"] or values["candidate_count"] > 100:
        raise ValueError("progress counters are invalid")


def _validate_event(event: object, *, previous: str, sequence: int, run_identity: str, context_sha256: str) -> dict[str, object]:
    if not isinstance(event, Mapping) or set(event) != _EVENT_KEYS:
        raise ValueError("runtime event schema is invalid")
    value = dict(event)
    unsigned = dict(value)
    digest = unsigned.pop("event_sha256")
    if value["schema"] != "exp436.runtime-event.v1" or value["sequence"] != sequence or value["previous_event_sha256"] != previous or value["run_identity"] != run_identity or value["context_sha256"] != context_sha256 or not isinstance(value["timestamp"], str) or not _SHA.fullmatch(str(digest)) or _digest(unsigned) != digest:
        raise ValueError("runtime event chain is invalid")
    kind, payload = value["kind"], value["payload"]
    if kind not in {"runtime_started", "dataset_complete", "reconciliation", "interrupted", "failed", "checkpoint_complete"} or not isinstance(payload, Mapping):
        raise ValueError("runtime event schema is invalid")
    if kind == "runtime_started":
        if set(payload) != {"backend", "checkpoint_path", "total_datasets"}:
            raise ValueError("runtime event payload is invalid")
    elif kind == "dataset_complete":
        required = {"sequence_index", "dataset", "record_sha256", "status", "dataset_metadata", *_COUNTERS}
        if set(payload) != required or type(payload["sequence_index"]) is not int or payload["sequence_index"] < 0 or not isinstance(payload["dataset"], str) or not _SHA.fullmatch(str(payload["record_sha256"])) or payload["status"] not in {"scored", "abstained"}:
            raise ValueError("runtime event payload is invalid")
        _validate_metadata(payload["dataset_metadata"]); _validate_counters(payload, total=1117)
    elif kind == "reconciliation":
        if set(payload) != {"from_completed", "to_completed", "datasets"} or type(payload["from_completed"]) is not int or type(payload["to_completed"]) is not int or not isinstance(payload["datasets"], list) or any(not isinstance(item, str) for item in payload["datasets"]):
            raise ValueError("runtime event payload is invalid")
    else:
        required = {"offending_dataset", "prior_stage", "error_type", "error_message", *_COUNTERS}
        if set(payload) != required or payload["offending_dataset"] is not None and not isinstance(payload["offending_dataset"], str) or payload["prior_stage"] not in _ACTIVE | {"idle"} or not isinstance(payload["error_type"], str) or not isinstance(payload["error_message"], str):
            raise ValueError("runtime event payload is invalid")
        _validate_counters(payload, total=1117)
    return value


def _read_events_raw(raw: bytes, *, run_identity: str, context_sha256: str) -> tuple[dict[str, object], ...]:
    if raw and not raw.endswith(b"\n"):
        raise ValueError("runtime events contain a partial record")
    values: list[dict[str, object]] = []
    previous = _ZERO
    terminal_seen = False
    resumable_terminal = False
    checkpoint_complete_seen = False
    for sequence, line in enumerate(raw.splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("runtime event is invalid") from error
        if canonical_json_bytes(record) != line:
            raise ValueError("runtime event is noncanonical")
        value = _validate_event(record, previous=previous, sequence=sequence, run_identity=run_identity, context_sha256=context_sha256)
        if terminal_seen:
            if value["kind"] != "runtime_started" or not resumable_terminal or checkpoint_complete_seen:
                raise ValueError("runtime event follows a terminal lifecycle")
            terminal_seen = False; resumable_terminal = False
        if value["kind"] in _TERMINAL:
            if value["kind"] == "checkpoint_complete" and checkpoint_complete_seen:
                raise ValueError("checkpoint complete event is duplicated")
            checkpoint_complete_seen = checkpoint_complete_seen or value["kind"] == "checkpoint_complete"
            terminal_seen = True; resumable_terminal = value["kind"] == "interrupted"
        previous = str(value["event_sha256"]); values.append(value)
    if not values or values[0]["kind"] != "runtime_started":
        raise ValueError("runtime events lack start record")
    return tuple(values)


def _read_events(output: Path, *, run_identity: str, context_sha256: str) -> tuple[dict[str, object], ...]:
    return _read_events_raw((output / "events_antigravity.jsonl").read_bytes(), run_identity=run_identity, context_sha256=context_sha256)


def _canonical_raw(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _checkpoint_prefix(inspection: object) -> tuple[tuple[str, ...], Mapping[str, str]]:
    if not isinstance(inspection, Mapping):
        raise ValueError("checkpoint inspection is invalid")
    completed, datasets, digests = inspection.get("completed"), inspection.get("datasets"), inspection.get("record_digests")
    if type(completed) is not int or completed < 0 or not isinstance(datasets, (tuple, list)) or not isinstance(digests, Mapping):
        raise ValueError("checkpoint inspection is invalid")
    names = tuple(datasets)
    if len(names) != completed or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names) or set(digests) != set(names) or any(not isinstance(digests[name], str) or not _SHA.fullmatch(digests[name]) for name in names):
        raise ValueError("checkpoint inspection is invalid")
    return names, MappingProxyType(dict(digests))


def _validate_progress(payload: object, *, run_identity: str, checkpoint: str, context_sha256: str, events: Sequence[Mapping[str, object]], event_tail: tuple[int, str] | None = None) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _PROGRESS_KEYS:
        raise ValueError("runtime progress schema is invalid")
    value = dict(payload); unsigned = dict(value); digest = unsigned.pop("progress_sha256")
    if value["schema"] != "exp436.runtime-progress.v1" or value["run_identity"] != run_identity or value["context_sha256"] != context_sha256 or value["checkpoint_path"] != checkpoint or not _SHA.fullmatch(str(digest)) or _digest(unsigned) != digest:
        raise ValueError("runtime progress binding is invalid")
    if type(value["total_datasets"]) is not int or value["total_datasets"] != 1117 or value["backend"] not in {"cpu", "mlx", "mps"} or type(value["pid"]) is not int or type(value["pid_active"]) is not bool or not isinstance(value["heartbeat"], str) or not isinstance(value["updated_at"], str):
        raise ValueError("runtime progress schema is invalid")
    _validate_counters(value, total=int(value["total_datasets"]))
    if value["stage"] not in _ACTIVE | {"idle"} or value["state"] not in {"running", *_TERMINAL}:
        raise ValueError("runtime progress stage is invalid")
    active = value["stage"] in _ACTIVE
    if active != (value["current_dataset"] is not None) or active != (value["dataset_metadata"] is not None):
        raise ValueError("runtime progress stage invariant is invalid")
    if value["current_dataset"] is not None and (not isinstance(value["current_dataset"], str) or not value["current_dataset"]):
        raise ValueError("runtime progress stage invariant is invalid")
    if value["dataset_metadata"] is not None:
        _validate_metadata(value["dataset_metadata"])
    inflight = value["in_flight_datasets"]
    if not isinstance(inflight, list) or len(inflight) > 11 or any(not isinstance(item, str) or not item for item in inflight) or len(set(inflight)) != len(inflight) or inflight != sorted(inflight):
        raise ValueError("runtime progress in-flight datasets are invalid")
    if value["state"] in _TERMINAL and (value["stage"] != "idle" or value["current_dataset"] is not None or value["dataset_metadata"] is not None or inflight or value["pid_active"]):
        raise ValueError("terminal progress is invalid")
    _validate_error(value["latest_error"])
    for name in ("datasets_per_second", "elapsed_seconds"):
        if type(value[name]) not in {int, float} or isinstance(value[name], bool) or not math.isfinite(float(value[name])) or value[name] < 0:
            raise ValueError("runtime progress timing is invalid")
    if value["eta_seconds"] is not None and (type(value["eta_seconds"]) not in {int, float} or isinstance(value["eta_seconds"], bool) or not math.isfinite(float(value["eta_seconds"])) or value["eta_seconds"] < 0):
        raise ValueError("runtime progress timing is invalid")
    sequence = value["last_event_sequence"]
    event_count, tail = (len(events), str(events[sequence - 1]["event_sha256"])) if event_tail is None and type(sequence) is int and 1 <= sequence <= len(events) else event_tail or (-1, "")
    if type(sequence) is not int or sequence < 1 or sequence > event_count or value["last_event_sha256"] != tail:
        raise ValueError("runtime progress event binding is invalid")
    return value


def _validate_pid(payload: object, *, run_identity: str, checkpoint: str, context_sha256: str, progress: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _PID_KEYS:
        raise ValueError("runtime PID schema is invalid")
    value = dict(payload); unsigned = dict(value); digest = unsigned.pop("pid_sha256")
    if value["schema"] != "exp436.runtime-pid.v1" or value["run_identity"] != run_identity or value["context_sha256"] != context_sha256 or value["checkpoint_path"] != checkpoint or value["backend"] != progress["backend"] or value["total_datasets"] != progress["total_datasets"] or type(value["pid"]) is not int or type(value["pid_active"]) is not bool or value["pid_active"] != progress["pid_active"] or value["state"] != progress["state"] or not isinstance(value["heartbeat"], str) or not _SHA.fullmatch(str(digest)) or _digest(unsigned) != digest:
        raise ValueError("runtime PID binding is invalid")
    return value


def _inventory(output: Path, *, allow_temps: bool = False) -> tuple[str, ...]:
    try:
        if os.path.islink(output) or not output.is_dir():
            raise ValueError("runtime output inventory is invalid")
        names = tuple(sorted(path.name for path in output.iterdir()))
    except OSError as error:
        raise ValueError("runtime output inventory is invalid") from error
    extras = set(names) - _ARTIFACTS
    if extras and (not allow_temps or any(not _TEMP.fullmatch(name) for name in extras)):
        raise ValueError("runtime output inventory is invalid")
    for name in _ARTIFACTS:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime output inventory is invalid")
    return tuple(sorted(extras))


def _validate_output_inventory(output: Path, *, mutable: bool = False) -> None:
    _inventory(Path(output), allow_temps=mutable)


def inspect_progress_bytes(runtime_bytes: Mapping[str, bytes], *, run_identity: str, checkpoint: Path, context_fingerprint: str, checkpoint_inspection: Mapping[str, object], process_is_live: Callable[[int], bool] | None = None) -> Mapping[str, object]:
    """Validate exactly the four runtime artifacts without filesystem writes."""
    if set(runtime_bytes) != _ARTIFACTS or any(not isinstance(raw, bytes) for raw in runtime_bytes.values()):
        raise ValueError("runtime output inventory is invalid")
    checkpoint_path = str(Path(checkpoint).resolve())
    if not _SHA.fullmatch(context_fingerprint):
        raise ValueError("runtime progress binding is invalid")
    events = _read_events_raw(runtime_bytes["events_antigravity.jsonl"], run_identity=run_identity, context_sha256=context_fingerprint)
    progress = _validate_progress(_canonical_raw(runtime_bytes["progress_antigravity.json"], "runtime progress"), run_identity=run_identity, checkpoint=checkpoint_path, context_sha256=context_fingerprint, events=events)
    pid = _validate_pid(_canonical_raw(runtime_bytes["run_antigravity.pid"], "runtime PID"), run_identity=run_identity, checkpoint=checkpoint_path, context_sha256=context_fingerprint, progress=progress)
    datasets, digests = _checkpoint_prefix(checkpoint_inspection)
    completions = [event for event in events if event["kind"] == "dataset_complete"]
    if len(completions) > len(datasets):
        raise ValueError("event prefix is ahead of checkpoint")
    for index, event in enumerate(completions):
        payload = event["payload"]
        if payload["sequence_index"] != index or payload["dataset"] != datasets[index] or payload["record_sha256"] != digests[datasets[index]] or payload["completed_datasets"] != index + 1 or payload["recomputed_datasets"] != index + 1:
            raise ValueError("event dataset prefix is invalid")
    if progress["completed_datasets"] > len(completions):
        raise ValueError("progress is ahead of event prefix")
    if progress["completed_datasets"] > len(datasets):
        raise ValueError("progress is ahead of checkpoint")
    if int(progress["completed_datasets"]):
        counters = completions[int(progress["completed_datasets"]) - 1]["payload"]
        if any(progress[name] != counters[name] for name in _COUNTERS):
            raise ValueError("progress counters do not match event prefix")
    if process_is_live is not None and pid["pid_active"] and process_is_live(int(pid["pid"])):
        raise ValueError("runtime process is still active")
    missing = tuple(datasets[len(completions):])
    lifecycle_state = events[-1]["kind"] if events[-1]["kind"] in _TERMINAL else "running"
    completion_payloads = tuple(MappingProxyType(dict(event["payload"])) for event in completions)
    terminal_payload = MappingProxyType(dict(events[-1]["payload"])) if lifecycle_state in _TERMINAL else None
    return MappingProxyType({"progress": MappingProxyType(progress), "pid": MappingProxyType(pid), "event_completed": len(completions), "checkpoint_completed": len(datasets), "reconciliation_required": missing, "progress_reconciliation_required": int(progress["completed_datasets"]) < len(completions), "lifecycle_state": lifecycle_state, "terminal_payload": terminal_payload, "completion_payloads": completion_payloads})


def inspect_progress(output: Path, *, run_identity: str, checkpoint: Path, context_fingerprint: str, checkpoint_inspection: Mapping[str, object], process_is_live: Callable[[int], bool] | None = None, _allow_owned_temps: bool = False) -> Mapping[str, object]:
    """Strictly inspect four artifacts without changing a byte."""
    output = Path(output)
    _inventory(output, allow_temps=_allow_owned_temps)
    return inspect_progress_bytes({name: (output / name).read_bytes() for name in _ARTIFACTS}, run_identity=run_identity, checkpoint=checkpoint, context_fingerprint=context_fingerprint, checkpoint_inspection=checkpoint_inspection, process_is_live=process_is_live)


class ProgressWriter:
    """One process-owned writer bound to a locked output directory FD."""
    def __init__(self, output: Path, *, run_identity: str, checkpoint: Path, total_datasets: int, backend: str, context_fingerprint: str, heartbeat_seconds: float | None = 30.0, heartbeat_observer: Callable[[], None] | None = None) -> None:
        if not isinstance(run_identity, str) or not run_identity or total_datasets != 1117 or backend not in {"cpu", "mlx", "mps"} or not _SHA.fullmatch(context_fingerprint):
            raise ValueError("progress binding is invalid")
        if heartbeat_seconds is not None and (not isinstance(heartbeat_seconds, (int, float)) or heartbeat_seconds <= 0):
            raise ValueError("heartbeat interval is invalid")
        self.output = Path(output); self.run_identity = run_identity; self.checkpoint = str(Path(checkpoint).resolve()); self.total_datasets = total_datasets; self.backend = backend; self.context_fingerprint = context_fingerprint
        self.heartbeat_seconds = heartbeat_seconds; self.heartbeat_observer = heartbeat_observer; self.creator_pid = os.getpid(); self._directory_fd: int | None = None; self._lock_held = False; self._closed = False; self._terminal = False; self._lock = RLock(); self._stop = Event(); self._thread: Thread | None = None; self._started = time.monotonic(); self._elapsed_offset = 0.0; self.progress: dict[str, object] = {}; self._event_count: int | None = None; self._event_tail_sha256: str | None = None; self._completion_count: int | None = None

    @classmethod
    def create(cls, output: Path, *, run_identity: str, checkpoint: Path, total_datasets: int, backend: str, context_fingerprint: str, heartbeat_seconds: float | None = 30.0, heartbeat_observer: Callable[[], None] | None = None) -> "ProgressWriter":
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise ValueError("runtime output root already exists")
        output.mkdir(mode=0o700); parent_fd = _open_directory(output.parent)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        writer = cls(output, run_identity=run_identity, checkpoint=checkpoint, total_datasets=total_datasets, backend=backend, context_fingerprint=context_fingerprint, heartbeat_seconds=heartbeat_seconds, heartbeat_observer=heartbeat_observer)
        writer._claim()
        try:
            _create(writer._fd(), "events_antigravity.jsonl", b""); _create(writer._fd(), "parent_antigravity.log", b"")
            writer._append_event("runtime_started", {"backend": backend, "checkpoint_path": writer.checkpoint, "total_datasets": total_datasets})
            writer._persist(writer._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None))
        except BaseException:
            writer.close(); raise
        return writer

    @classmethod
    def open_existing(cls, output: Path, *, run_identity: str, checkpoint: Path, backend: str | None = None, total_datasets: int | None = None, context_fingerprint: str | None = None, checkpoint_inspection: Mapping[str, object] | None = None, heartbeat_seconds: float | None = 30.0, heartbeat_observer: Callable[[], None] | None = None, process_is_live: Callable[[int], bool] | None = None) -> "ProgressWriter":
        if backend is None or total_datasets is None or context_fingerprint is None or checkpoint_inspection is None:
            raise ValueError("strict checkpoint inspection is required")
        writer = cls(Path(output), run_identity=run_identity, checkpoint=checkpoint, total_datasets=total_datasets, backend=backend, context_fingerprint=context_fingerprint, heartbeat_seconds=heartbeat_seconds, heartbeat_observer=heartbeat_observer)
        writer._claim()
        try:
            extras = _inventory(writer.output, allow_temps=True)
            # Validate durable authorities before removing an owned temporary.
            if extras:
                for name in extras:
                    raw = _canonical(writer.output / name, "runtime temporary")
                    if name.startswith(".progress"):
                        _validate_progress(raw, run_identity=run_identity, checkpoint=writer.checkpoint, context_sha256=context_fingerprint, events=_read_events(writer.output, run_identity=run_identity, context_sha256=context_fingerprint))
                    else:
                        _validate_pid(raw, run_identity=run_identity, checkpoint=writer.checkpoint, context_sha256=context_fingerprint, progress=_canonical(writer.output / "progress_antigravity.json", "runtime progress"))
            state = inspect_progress(writer.output, run_identity=run_identity, checkpoint=checkpoint, context_fingerprint=context_fingerprint, checkpoint_inspection=checkpoint_inspection, process_is_live=process_is_live, _allow_owned_temps=True)
            for name in extras:
                os.unlink(name, dir_fd=writer._fd())
            if extras:
                os.fsync(writer._fd())
            writer.progress = dict(state["progress"]); writer._elapsed_offset = float(writer.progress["elapsed_seconds"]); writer._started = time.monotonic()
            writer._cache_events()
            if state["lifecycle_state"] == "interrupted":
                # Interrupted roots are resumable: claim a fresh parent PID and
                # continue from the checkpoint authority, never reusing an engine.
                writer._append_event("runtime_started", {"backend": writer.backend, "checkpoint_path": writer.checkpoint, "total_datasets": writer.total_datasets})
                writer._persist(writer._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None))
            elif state["lifecycle_state"] == "running":
                # A crash just after appending a fresh runtime_started event can
                # leave an interrupted snapshot behind; event lifecycle wins.
                writer._persist(writer._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None))
            else:
                writer._terminal = True
            return writer
        except BaseException:
            writer._release(); raise

    @classmethod
    def preflight_existing(cls, output: Path, *, run_identity: str, checkpoint: Path, backend: str | None = None, total_datasets: int | None = None, context_fingerprint: str | None = None, checkpoint_inspection: Mapping[str, object] | None = None, process_is_live: Callable[[int], bool] | None = None) -> Mapping[str, object]:
        if context_fingerprint is None or checkpoint_inspection is None:
            raise ValueError("strict checkpoint inspection is required")
        state = inspect_progress(output, run_identity=run_identity, checkpoint=checkpoint, context_fingerprint=context_fingerprint, checkpoint_inspection=checkpoint_inspection, process_is_live=process_is_live)
        progress = state["progress"]
        if backend is not None and progress["backend"] != backend or total_datasets is not None and progress["total_datasets"] != total_datasets:
            raise ValueError("runtime progress binding is invalid")
        if state["lifecycle_state"] == "failed":
            raise ValueError("failed runtime cannot resume")
        if state["lifecycle_state"] == "checkpoint_complete":
            terminal = state["terminal_payload"]
            if (progress["state"] != "checkpoint_complete" or int(state["checkpoint_completed"]) != total_datasets
                    or int(state["event_completed"]) != total_datasets or int(progress["completed_datasets"]) != total_datasets
                    or bool(state["reconciliation_required"]) or bool(state["progress_reconciliation_required"])
                    or not isinstance(terminal, Mapping) or any(terminal[name] != progress[name] for name in _COUNTERS)):
                raise ValueError("checkpoint complete runtime is incomplete or inconsistent")
        return state

    def _fd(self) -> int:
        if self._directory_fd is None:
            raise ValueError("runtime writer owner is invalid")
        return self._directory_fd

    def _claim(self) -> None:
        self._directory_fd = _open_directory(self.output)
        try:
            fcntl.flock(self._directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self._directory_fd); self._directory_fd = None
            raise ValueError("runtime writer is already owned") from error
        self._lock_held = True

    def _release(self) -> None:
        fd, self._directory_fd = self._directory_fd, None
        self._lock_held = False
        if fd is not None:
            try: fcntl.flock(fd, fcntl.LOCK_UN)
            finally: os.close(fd)

    def _assert_owner(self) -> None:
        if self._closed or self._directory_fd is None or not self._lock_held or os.getpid() != self.creator_pid:
            raise ValueError("runtime writer owner is invalid")
        try:
            if not stat.S_ISDIR(os.fstat(self._directory_fd).st_mode):
                raise ValueError("runtime writer owner is invalid")
        except OSError as error:
            raise ValueError("runtime writer owner is invalid") from error

    def _elapsed(self) -> float:
        return float(self._elapsed_offset + max(0.0, time.monotonic() - self._started))

    def _event_tail(self) -> tuple[int, str]:
        self._cache_events()
        if self._event_count is None or self._event_tail_sha256 is None:
            raise ValueError("runtime event cache is invalid")
        return self._event_count, self._event_tail_sha256

    def _cache_events(self) -> None:
        if self._event_count is not None:
            return
        events = _read_events(self.output, run_identity=self.run_identity, context_sha256=self.context_fingerprint) if (self.output / "events_antigravity.jsonl").stat().st_size else ()
        self._event_count = len(events); self._event_tail_sha256 = str(events[-1]["event_sha256"]) if events else _ZERO
        self._completion_count = sum(event["kind"] == "dataset_complete" for event in events)

    def _snapshot(self, *, state: str, stage: str, current_dataset: str | None, dataset_metadata: Mapping[str, object] | None, in_flight_datasets: Sequence[str], pid_active: bool, latest_error: Mapping[str, object] | None, **overrides: object) -> dict[str, object]:
        base = self.progress
        sequence, tail = self._event_tail()
        completed = int(overrides.get("completed_datasets", base.get("completed_datasets", 0)))
        elapsed = self._elapsed(); rate = completed / elapsed if elapsed else 0.0
        payload: dict[str, object] = {"schema": "exp436.runtime-progress.v1", "run_identity": self.run_identity, "context_sha256": self.context_fingerprint, "pid": self.creator_pid, "pid_active": pid_active, "heartbeat": _timestamp(), "updated_at": _timestamp(), "state": state, "stage": stage, "backend": self.backend, "current_dataset": current_dataset, "dataset_metadata": None if dataset_metadata is None else _validate_metadata(dataset_metadata), "in_flight_datasets": sorted(in_flight_datasets), "completed_datasets": completed, "total_datasets": self.total_datasets, "recomputed_datasets": int(overrides.get("recomputed_datasets", base.get("recomputed_datasets", completed))), "success": int(overrides.get("success", base.get("success", 0))), "abstention": int(overrides.get("abstention", base.get("abstention", 0))), "error": int(overrides.get("error", base.get("error", 0))), "excluded_train_rows": int(overrides.get("excluded_train_rows", base.get("excluded_train_rows", 0))), "nonfinite_test_rows": int(overrides.get("nonfinite_test_rows", base.get("nonfinite_test_rows", 0))), "candidate_count": int(overrides.get("candidate_count", base.get("candidate_count", 0))), "datasets_per_second": rate, "elapsed_seconds": elapsed, "eta_seconds": None if rate == 0.0 else (self.total_datasets - completed) / rate, "checkpoint_path": self.checkpoint, "latest_error": _validate_error(latest_error), "last_event_sequence": sequence, "last_event_sha256": tail}
        unsigned = dict(payload); payload["progress_sha256"] = _digest(unsigned)
        _validate_progress(payload, run_identity=self.run_identity, checkpoint=self.checkpoint, context_sha256=self.context_fingerprint, events=(), event_tail=(sequence, tail))
        return payload

    def _pid_payload(self, progress: Mapping[str, object]) -> dict[str, object]:
        value: dict[str, object] = {"schema": "exp436.runtime-pid.v1", "run_identity": self.run_identity, "context_sha256": self.context_fingerprint, "checkpoint_path": self.checkpoint, "backend": self.backend, "total_datasets": self.total_datasets, "pid": self.creator_pid, "pid_active": progress["pid_active"], "heartbeat": progress["heartbeat"], "state": progress["state"]}
        value["pid_sha256"] = _digest(value); return value

    def _persist(self, payload: Mapping[str, object]) -> None:
        self._assert_owner()
        _atomic_replace(self._fd(), "progress_antigravity.json", canonical_json_bytes(dict(payload)))
        _atomic_replace(self._fd(), "run_antigravity.pid", canonical_json_bytes(self._pid_payload(payload)))
        self.progress = dict(payload)

    def _append_event(self, kind: str, payload: Mapping[str, object]) -> None:
        self._assert_owner()
        self._cache_events()
        if self._event_count is None or self._event_tail_sha256 is None:
            raise ValueError("runtime event cache is invalid")
        previous = self._event_tail_sha256
        record: dict[str, object] = {"schema": "exp436.runtime-event.v1", "sequence": self._event_count + 1, "previous_event_sha256": previous, "run_identity": self.run_identity, "context_sha256": self.context_fingerprint, "timestamp": _timestamp(), "kind": kind, "payload": dict(payload)}
        record["event_sha256"] = _digest(record)
        _validate_event(record, previous=previous, sequence=self._event_count + 1, run_identity=self.run_identity, context_sha256=self.context_fingerprint)
        _append(self._fd(), "events_antigravity.jsonl", canonical_json_bytes(record) + b"\n")
        self._event_count += 1; self._event_tail_sha256 = str(record["event_sha256"])
        if kind == "dataset_complete":
            if self._completion_count is None: raise ValueError("runtime event cache is invalid")
            self._completion_count += 1
        self.append_log(f"event={kind} stage={self.progress.get('stage', 'idle')} dataset={payload.get('dataset', payload.get('offending_dataset'))} completed={payload.get('completed_datasets', self.progress.get('completed_datasets', 0))}")

    def append_log(self, message: str) -> None:
        self._assert_owner()
        if not isinstance(message, str) or "\n" in message:
            raise ValueError("runtime log message is invalid")
        _append(self._fd(), "parent_antigravity.log", f"{_timestamp()} {message}\n".encode())

    def update(self, *, stage: str, current_dataset: str | None, dataset_metadata: Mapping[str, object] | None, completed: int | None = None, success: int | None = None, abstention: int | None = None, error: int | None = None, candidates: int | None = None, candidate_count: int | None = None, excluded_train_rows: int | None = None, nonfinite_test_rows: int | None = None, latest_error: Mapping[str, object] | None = None, state: str = "running", in_flight_datasets: Sequence[str] = ()) -> None:
        self._assert_owner()
        if state in _TERMINAL:
            raise ValueError("terminal state requires a terminal lifecycle method")
        if candidate_count is not None and candidates is not None and candidate_count != candidates:
            raise ValueError("candidate counter is ambiguous")
        with self._lock:
            payload = self._snapshot(state=state, stage=stage, current_dataset=current_dataset, dataset_metadata=dataset_metadata, in_flight_datasets=in_flight_datasets, pid_active=True, latest_error=latest_error, completed_datasets=self.progress.get("completed_datasets", 0) if completed is None else completed, recomputed_datasets=self.progress.get("recomputed_datasets", 0) if completed is None else completed, success=self.progress.get("success", 0) if success is None else success, abstention=self.progress.get("abstention", 0) if abstention is None else abstention, error=self.progress.get("error", 0) if error is None else error, candidate_count=self.progress.get("candidate_count", 0) if candidate_count is None and candidates is None else (candidate_count if candidate_count is not None else candidates), excluded_train_rows=self.progress.get("excluded_train_rows", 0) if excluded_train_rows is None else excluded_train_rows, nonfinite_test_rows=self.progress.get("nonfinite_test_rows", 0) if nonfinite_test_rows is None else nonfinite_test_rows)
            self._persist(payload); self.append_log(f"state={state} stage={stage} dataset={current_dataset} completed={payload['completed_datasets']}")

    write_progress = update

    def record_dataset_complete(self, *, sequence_index: int, dataset: str, record_sha256: str, status: str, dataset_metadata: Mapping[str, object], completed_datasets: int, success: int, abstention: int, error: int, excluded_train_rows: int, nonfinite_test_rows: int, candidate_count: int) -> None:
        self._assert_owner()
        with self._lock:
            if self._terminal:
                raise ValueError("runtime is terminal")
            metadata = _validate_metadata(dataset_metadata)
            payload = {"sequence_index": sequence_index, "dataset": dataset, "record_sha256": record_sha256, "status": status, "dataset_metadata": metadata, "completed_datasets": completed_datasets, "recomputed_datasets": completed_datasets, "success": success, "abstention": abstention, "error": error, "excluded_train_rows": excluded_train_rows, "nonfinite_test_rows": nonfinite_test_rows, "candidate_count": candidate_count}
            self._cache_events()
            if self._completion_count is None: raise ValueError("runtime event cache is invalid")
            prior = self._completion_count
            if sequence_index != prior or completed_datasets != prior + 1:
                raise ValueError("dataset completion sequence is invalid")
            self._append_event("dataset_complete", payload)
            counters = {name: payload[name] for name in _COUNTERS}
            self._persist(self._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None, **counters))

    def append_event(self, payload: Mapping[str, object]) -> None:
        """Compatibility adapter for the unfinished runtime; v1 callers use record_dataset_complete."""
        self._assert_owner()
        if not isinstance(payload, Mapping) or payload.get("kind") != "dataset_complete":
            raise ValueError("only dataset completion events are accepted")
        current = self.progress
        self.record_dataset_complete(sequence_index=int(payload.get("sequence_index", int(current["completed_datasets"]))), dataset=str(payload.get("dataset")), record_sha256=str(payload.get("record_sha256")), status=str(payload.get("status")), dataset_metadata=payload.get("dataset_metadata") if isinstance(payload.get("dataset_metadata"), Mapping) else {}, completed_datasets=int(payload.get("completed", int(current["completed_datasets"]) + 1)), success=int(current["success"]) + (payload.get("status") == "scored"), abstention=int(current["abstention"]) + (payload.get("status") == "abstained"), error=int(current["error"]), excluded_train_rows=int(current["excluded_train_rows"]), nonfinite_test_rows=int(current["nonfinite_test_rows"]), candidate_count=int(current["candidate_count"]) + int(payload.get("candidate_count", 0)))

    def reconcile_checkpoint_tail(self, bindings: Sequence[tuple[int, str, str, str, Mapping[str, object], Mapping[str, int]]], *, checkpoint_inspection: Mapping[str, object]) -> None:
        self._assert_owner()
        state = inspect_progress(self.output, run_identity=self.run_identity, checkpoint=Path(self.checkpoint), context_fingerprint=self.context_fingerprint, checkpoint_inspection=checkpoint_inspection)
        missing = tuple(state["reconciliation_required"])
        if len(bindings) != len(missing) or tuple(binding[1] for binding in bindings) != missing:
            raise ValueError("reconciliation tail is not the authoritative prefix")
        initial = int(self.progress["completed_datasets"])
        for index, dataset, digest, status, metadata, counters in bindings:
            self.record_dataset_complete(sequence_index=index, dataset=dataset, record_sha256=digest, status=status, dataset_metadata=metadata, completed_datasets=index + 1, success=int(counters["success"]), abstention=int(counters["abstention"]), error=int(counters.get("error", 0)), excluded_train_rows=int(counters["excluded_train_rows"]), nonfinite_test_rows=int(counters["nonfinite_test_rows"]), candidate_count=int(counters["candidate_count"]))
        self._append_event("reconciliation", {"from_completed": initial, "to_completed": int(self.progress["completed_datasets"]), "datasets": list(missing)})
        self._persist(self._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None))

    def synchronize_event_prefix(self, *, checkpoint_inspection: Mapping[str, object]) -> None:
        """Advance a stale snapshot to its already-durable event prefix only."""
        self._assert_owner()
        state = inspect_progress(self.output, run_identity=self.run_identity, checkpoint=Path(self.checkpoint), context_fingerprint=self.context_fingerprint, checkpoint_inspection=checkpoint_inspection)
        if not state["progress_reconciliation_required"]:
            return
        if state["lifecycle_state"] != "running":
            raise ValueError("terminal event prefix requires terminal recovery")
        payload = state["completion_payloads"][-1]
        with self._lock:
            self._persist(self._snapshot(state="running", stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=True, latest_error=None, **{name: int(payload[name]) for name in _COUNTERS}))

    def _terminal_event(self, kind: str, dataset: str | None, prior_stage: str, error: BaseException | None) -> dict[str, object]:
        current = self.progress
        return {"offending_dataset": dataset, "prior_stage": prior_stage, "error_type": "" if error is None else type(error).__name__, "error_message": "" if error is None else str(error), **{name: int(current[name]) for name in _COUNTERS}}

    def _mark_terminal(self, kind: str, dataset: str | None, prior_stage: str, error: BaseException | None) -> None:
        self._assert_owner()
        if self._terminal:
            raise ValueError("terminal lifecycle already recorded")
        with self._lock:
            event = self._terminal_event(kind, dataset, prior_stage, error); self._append_event(kind, event)
            latest = None if kind == "checkpoint_complete" else {"dataset": dataset, "prior_stage": prior_stage, "type": event["error_type"], "message": event["error_message"]}
            self._persist(self._snapshot(state=kind, stage="idle", current_dataset=None, dataset_metadata=None, in_flight_datasets=(), pid_active=False, latest_error=latest))
            self._terminal = True

    def mark_interrupted(self, current_dataset: str | None, prior_stage: str | BaseException = "idle", error: BaseException | None = None) -> None:
        if isinstance(prior_stage, BaseException):
            error, prior_stage = prior_stage, str(self.progress.get("stage", "idle"))
        self._mark_terminal("interrupted", current_dataset, str(prior_stage), error)

    def mark_failed(self, current_dataset: str | None, prior_stage: str | BaseException, error: BaseException | None = None) -> None:
        if isinstance(prior_stage, BaseException):
            error, prior_stage = prior_stage, str(self.progress.get("stage", "idle"))
        if error is None:
            raise ValueError("failure requires an error")
        self._mark_terminal("failed", current_dataset, str(prior_stage), error)

    def mark_checkpoint_complete(self) -> None:
        if int(self.progress.get("completed_datasets", 0)) != self.total_datasets:
            raise ValueError("checkpoint complete requires all datasets")
        self._mark_terminal("checkpoint_complete", None, "idle", None)

    def start_heartbeat(self) -> None:
        self._assert_owner()
        if self.heartbeat_seconds is None or self._thread is not None:
            return
        def run() -> None:
            while not self._stop.wait(float(self.heartbeat_seconds)):
                with self._lock:
                    if self._closed or self._terminal:
                        return
                    try:
                        self._persist(self._snapshot(state=str(self.progress["state"]), stage=str(self.progress["stage"]), current_dataset=self.progress["current_dataset"] if isinstance(self.progress["current_dataset"], str) else None, dataset_metadata=self.progress["dataset_metadata"] if isinstance(self.progress["dataset_metadata"], Mapping) else None, in_flight_datasets=self.progress["in_flight_datasets"] if isinstance(self.progress["in_flight_datasets"], list) else (), pid_active=True, latest_error=self.progress["latest_error"] if isinstance(self.progress["latest_error"], Mapping) else None))
                        self.append_log(f"heartbeat stage={self.progress['stage']} completed={self.progress['completed_datasets']}")
                        if self.heartbeat_observer is not None:
                            self.heartbeat_observer()
                    except ValueError:
                        return
        self._thread = Thread(target=run, name="exp436-progress-heartbeat", daemon=True); self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def close(self) -> None:
        if self._closed:
            return
        self.stop_heartbeat()
        if self._directory_fd is not None and self._lock_held and os.getpid() == self.creator_pid:
            # A raw close never invents an interrupted/failed scientific state;
            # it only releases the operating-system liveness lease.
            if not self._terminal and self.progress:
                with self._lock:
                    self._persist(self._snapshot(state=str(self.progress["state"]), stage=str(self.progress["stage"]), current_dataset=self.progress["current_dataset"] if isinstance(self.progress["current_dataset"], str) else None, dataset_metadata=self.progress["dataset_metadata"] if isinstance(self.progress["dataset_metadata"], Mapping) else None, in_flight_datasets=self.progress["in_flight_datasets"] if isinstance(self.progress["in_flight_datasets"], list) else (), pid_active=False, latest_error=self.progress["latest_error"] if isinstance(self.progress["latest_error"], Mapping) else None))
        self._closed = True; self._release()


__all__ = ["ProgressWriter", "inspect_progress", "_validate_output_inventory"]
