"""Parent-owned atomic progress, PID, event, and Korean text evidence."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import re

from .artifacts_antigravity import canonical_json_bytes_antigravity, fsync_directory_antigravity
from .checkpoint_antigravity import ProcessIdentityAntigravity, current_process_identity_antigravity
from .config_antigravity import FROZEN_CONFIG


_TERMINAL = frozenset({"complete", "error", "failed"})
_RESUMABLE = frozenset({"interrupted", "interrupted_resumable"})
_PROGRESS_KEYS = frozenset({
    "pid", "start_identity", "active", "current_dataset", "current_candidate_index",
    "current_view", "view_count", "excluded_count", "train_count", "train_missing",
    "train_used", "train_excluded", "test_count", "test_rows", "raw_length",
    "target_length", "stage", "backend", "completed", "total", "imported",
    "recomputed", "success", "abstention", "error", "throughput", "elapsed_seconds",
    "eta_seconds", "checkpoint", "heartbeat", "latest_error", "total",
})
_PID_KEYS = frozenset({"pid", "start_identity", "active", "heartbeat"})
_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _finite_nonnegative_antigravity(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} is invalid")
    return float(value)


def _exact_nonnegative_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is invalid")
    return value


def _write_atomic_json_antigravity(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ValueError(f"{path.name} is a symlink")
    encoded = canonical_json_bytes_antigravity(payload)
    _discard_stale_atomic_temps_antigravity(path)
    temporary = path.with_name(
        f".{path.name}.antigravity-{os.getpid()}-{time.time_ns()}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("progress write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        fsync_directory_antigravity(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _discard_stale_atomic_temps_antigravity(path: Path) -> None:
    """Delete abandoned write temps; only canonical JSON is authoritative."""
    legacy = path.with_name(path.name.replace(".json", ".tmp"))
    candidates = {legacy, *path.parent.glob(f".{path.name}.antigravity-*.tmp")}
    removed = False
    for temporary in sorted(candidates, key=lambda item: item.name):
        if not temporary.exists() and not temporary.is_symlink():
            continue
        if temporary.is_symlink() or not temporary.is_file():
            raise ValueError(f"stale atomic temp is invalid: {temporary.name}")
        temporary.unlink()
        removed = True
    if removed:
        fsync_directory_antigravity(path.parent)


def _read_canonical_antigravity(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"progress file is invalid: {path.name}")
    try:
        payload = json.loads(path.read_text())
        encoded = canonical_json_bytes_antigravity(payload)
        if encoded != path.read_bytes() or not isinstance(payload, Mapping):
            raise ValueError("noncanonical")
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"progress file is invalid: {path.name}") from error


def _pid_probe_default_antigravity(process: ProcessIdentityAntigravity) -> bool:
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProgressWriterAntigravity:
    """Only the parent owns this object; worker processes receive no writer."""

    def __init__(
        self,
        attempt_path: Path,
        *,
        total: int,
        process: ProcessIdentityAntigravity,
        checkpoint_path: Path,
        elapsed_offset: float = 0.0,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("progress total is invalid")
        attempt = Path(attempt_path)
        if "antigravity" not in attempt.name or not attempt.is_dir() or attempt.is_symlink():
            raise ValueError("progress attempt path is invalid")
        if type(process) is not ProcessIdentityAntigravity:
            raise ValueError("progress process identity is invalid")
        self.attempt_path = attempt
        self.progress_path = attempt / "progress_antigravity.json"
        self.pid_path = attempt / "pid_antigravity.json"
        self.events_path = attempt / "events_antigravity.jsonl"
        self.text_path = attempt / "text_antigravity.log"
        self.checkpoint_path = str(checkpoint_path)
        self.total = total
        self.process = process
        self._started = time.monotonic() - max(0.0, elapsed_offset)
        self._stage = "initializing"
        self._active = True
        self._backend = "cpu"
        self._current_dataset: str | None = None
        self._current_candidate_index: int | None = None
        self._current_view: int | None = None
        self._view_count: int | None = None
        self._excluded_count: int | None = None
        self._dimensions: dict[str, int | None] = {
            "train_count": None, "train_missing": None, "train_used": None,
            "train_excluded": None, "test_count": None, "test_rows": None,
            "raw_length": None, "target_length": None,
        }
        initial = {name: 0 for name in ("completed", "imported", "recomputed", "success", "abstention", "error")}
        if counters is not None:
            for name in initial:
                value = counters.get(name, 0)
                _exact_nonnegative_antigravity(value, name)
                initial[name] = value
        self._counters = initial
        self._latest_error: str | None = None
        self._local_owner_pid = os.getpid()
        self._ensure_append_files_antigravity()

    @classmethod
    def create(
        cls,
        attempt_path: Path,
        *,
        total: int = FROZEN_CONFIG.source_candidate_count,
        pid: int | None = None,
        start_identity: str | None = None,
        process: ProcessIdentityAntigravity | None = None,
        checkpoint_path: Path | None = None,
        backend: str = "cpu",
    ) -> "ProgressWriterAntigravity":
        attempt = Path(attempt_path)
        if process is None:
            if pid is None:
                process = current_process_identity_antigravity()
            else:
                process = ProcessIdentityAntigravity(pid, start_identity or f"{pid}:{time.time_ns()}")
        elif pid is not None and pid != process.pid:
            raise ValueError("progress PID does not match process")
        if backend != "cpu":
            raise ValueError("Exp444 progress backend must start on CPU")
        if not attempt.is_dir():
            raise ValueError("progress attempt path is invalid")
        for filename in (
            "progress_antigravity.json", "pid_antigravity.json",
            "events_antigravity.jsonl", "text_antigravity.log",
        ):
            path = attempt / filename
            if path.exists() or path.is_symlink():
                raise ValueError(f"progress destination already exists: {filename}")
        writer = cls(
            attempt, total=total, process=process,
            checkpoint_path=checkpoint_path or attempt / "checkpoint_antigravity/state_antigravity.json",
        )
        writer._write_pid_antigravity(True)
        writer._write_progress_antigravity()
        writer._append_event_antigravity("start", "initializing", {"completed": 0})
        writer._append_text_antigravity(f"시작 pid={process.pid} 완료=0/{total}\n")
        return writer

    def _ensure_append_files_antigravity(self) -> None:
        for path in (self.events_path, self.text_path):
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"progress log is invalid: {path.name}")
                continue
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(descriptor)
        fsync_directory_antigravity(self.attempt_path)

    def _write_pid_antigravity(self, active: bool) -> None:
        payload = {
            "pid": self.process.pid,
            "start_identity": self.process.start_identity,
            "active": bool(active),
            "heartbeat": _utc_now_antigravity(),
        }
        _write_atomic_json_antigravity(self.pid_path, payload)

    def _elapsed_antigravity(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _throughput_antigravity(self) -> float:
        elapsed = self._elapsed_antigravity()
        return float(self._counters["completed"]) / elapsed if elapsed > 0 else 0.0

    def _eta_antigravity(self) -> float | None:
        if not self._active:
            return None
        rate = self._throughput_antigravity()
        if rate <= 0:
            # Before the first completed candidate there is no measured
            # throughput.  Keep an explicit finite estimate in the live state;
            # terminal states alone are required to carry JSON ``null``.
            return float(self.total - self._counters["completed"])
        return max(0.0, float(self.total - self._counters["completed"]) / rate)

    def _progress_payload_antigravity(self) -> dict[str, object]:
        elapsed = self._elapsed_antigravity()
        payload: dict[str, object] = {
            "pid": self.process.pid,
            "start_identity": self.process.start_identity,
            "active": self._active,
            "current_dataset": self._current_dataset,
            "current_candidate_index": self._current_candidate_index,
            "current_view": self._current_view,
            "view_count": self._view_count,
            "excluded_count": self._excluded_count,
            **self._dimensions,
            "stage": self._stage,
            "backend": self._backend,
            **self._counters,
            "total": self.total,
            "throughput": self._throughput_antigravity(),
            "elapsed_seconds": elapsed,
            "eta_seconds": self._eta_antigravity(),
            "checkpoint": self.checkpoint_path,
            "heartbeat": _utc_now_antigravity(),
            "latest_error": self._latest_error,
        }
        if set(payload) != _PROGRESS_KEYS:
            raise AssertionError("progress schema drift")
        return payload

    def _write_progress_antigravity(self) -> None:
        _write_atomic_json_antigravity(self.progress_path, self._progress_payload_antigravity())

    def _append_event_antigravity(self, event: str, stage: str, extra: Mapping[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            "event": event,
            "stage": stage,
            "timestamp": _utc_now_antigravity(),
            "pid": self.process.pid,
            "completed": self._counters["completed"],
            "total": self.total,
        }
        if extra:
            payload.update(dict(extra))
        # ``canonical_json_bytes_antigravity`` already carries one newline.
        encoded = canonical_json_bytes_antigravity(payload)
        descriptor = os.open(self.events_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("event append made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _append_text_antigravity(self, line: str) -> None:
        if not isinstance(line, str) or not line.endswith("\n"):
            raise ValueError("text log line is invalid")
        descriptor = os.open(self.text_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            encoded = line.encode("utf-8")
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("text log append made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _require_active_antigravity(self) -> None:
        if os.getpid() != self._local_owner_pid:
            raise ValueError("progress mutation requires the parent owner process")
        if not self._active or self._stage in _TERMINAL:
            raise ValueError("progress is terminal")

    def start_candidate_antigravity(
        self,
        dataset: str,
        candidate_index: int,
        *,
        train_count: int,
        test_count: int,
        raw_length: int,
        train_missing: int | None = None,
        train_used: int | None = None,
        train_excluded: int | None = None,
        test_rows: int | None = None,
        target_length: int | None = None,
        backend: str = "cpu",
    ) -> None:
        self._require_active_antigravity()
        if type(dataset) is not str or not dataset:
            raise ValueError("progress dataset is invalid")
        if type(candidate_index) is not int or candidate_index < 0 or candidate_index >= self.total:
            raise ValueError("progress candidate index is invalid")
        for value, name in ((train_count, "train_count"), (test_count, "test_count"), (raw_length, "raw_length")):
            _exact_nonnegative_antigravity(value, name)
        for value, name in ((train_missing, "train_missing"), (train_used, "train_used"), (train_excluded, "train_excluded"), (test_rows, "test_rows"), (target_length, "target_length")):
            if value is not None:
                _exact_nonnegative_antigravity(value, name)
        if backend not in {"cpu", "mlx", "numba"}:
            raise ValueError("progress backend is invalid")
        self._current_dataset = dataset
        self._current_candidate_index = candidate_index
        self._current_view = None
        self._view_count = None
        self._excluded_count = None
        self._dimensions.update({
            "train_count": train_count,
            "train_missing": train_missing,
            "train_used": train_used,
            "train_excluded": train_excluded,
            "test_count": test_count,
            "test_rows": test_rows if test_rows is not None else test_count,
            "raw_length": raw_length,
            "target_length": target_length if target_length is not None else raw_length,
        })
        self._backend = backend
        self._stage = "candidate"
        self._write_progress_antigravity()
        self._append_event_antigravity("candidate_start", self._stage, {"dataset": dataset, "candidate_index": candidate_index})
        self._append_text_antigravity(f"후보 시작 dataset={dataset} index={candidate_index} train={train_count} test={test_count} raw_length={raw_length}\n")

    def start_view_antigravity(self, *, view_index: int, view_count: int, excluded_count: int) -> None:
        self._require_active_antigravity()
        if self._current_candidate_index is None:
            raise ValueError("progress candidate is not active")
        if type(view_index) is not int or view_index < 0 or type(view_count) is not int or view_count < 1 or view_index >= view_count:
            raise ValueError("progress view dimensions are invalid")
        _exact_nonnegative_antigravity(excluded_count, "excluded_count")
        self._current_view = view_index
        self._view_count = view_count
        self._excluded_count = excluded_count
        self._stage = "view"
        self._write_progress_antigravity()
        self._append_event_antigravity("view_start", self._stage, {"view_index": view_index, "view_count": view_count, "excluded_count": excluded_count})
        self._append_text_antigravity(f"뷰 시작 dataset={self._current_dataset} view={view_index}/{view_count} 제외={excluded_count}\n")

    def complete_candidate_antigravity(
        self,
        status: str = "success",
        *,
        imported: bool = False,
        recomputed: bool = True,
        abstention: bool = False,
        error: BaseException | str | None = None,
    ) -> None:
        self._require_active_antigravity()
        if status not in {"success", "scored", "abstained", "error", "failed"}:
            raise ValueError("progress candidate status is invalid")
        if type(imported) is not bool or type(recomputed) is not bool or type(abstention) is not bool:
            raise ValueError("progress candidate counters are invalid")
        if self._current_candidate_index is None:
            raise ValueError("progress candidate is not active")
        if self._counters["completed"] >= self.total:
            raise ValueError("progress is complete")
        self._counters["completed"] += 1
        self._counters["imported"] += int(imported)
        self._counters["recomputed"] += int(recomputed)
        self._counters["abstention"] += int(abstention or status == "abstained")
        if status in {"error", "failed"} or error is not None:
            self._counters["error"] += 1
            self._latest_error = str(error) if error is not None else status
        else:
            self._counters["success"] += 1
        dataset = self._current_dataset
        self._current_dataset = None
        self._current_candidate_index = None
        self._current_view = None
        self._view_count = None
        self._excluded_count = None
        self._stage = "running"
        self._write_progress_antigravity()
        self._append_event_antigravity("candidate_complete", self._stage, {"dataset": dataset, "status": status})
        self._append_text_antigravity(f"후보 완료 dataset={dataset} status={status} 완료={self._counters['completed']}/{self.total}\n")

    def reconcile_committed_success_antigravity(
        self,
        sequence: int,
        *,
        record_sha256: str,
        abstention: bool = False,
    ) -> None:
        """Advance progress once for an already-durable result after its normal write failed.

        The checkpoint is the durable authority at this boundary.  Re-reading
        the current progress file makes recovery independent of any partially
        changed in-memory counters in a failed normal completion call.
        """
        self._require_active_antigravity()
        if type(sequence) is not int or sequence < 0 or sequence >= self.total:
            raise ValueError("progress committed sequence is invalid")
        if type(record_sha256) is not str or _SHA256_ANTIGRAVITY.fullmatch(record_sha256) is None:
            raise ValueError("progress committed record hash is invalid")
        persisted = _read_canonical_antigravity(self.progress_path)
        if (
            persisted.get("active") is not True
            or persisted.get("completed") != sequence
            or persisted.get("current_candidate_index") != sequence
            or persisted.get("pid") != self.process.pid
            or persisted.get("start_identity") != self.process.start_identity
        ):
            raise ValueError("progress committed recovery authority is invalid")
        self._counters = {name: int(persisted[name]) for name in (
            "completed", "imported", "recomputed", "success", "abstention", "error",
        )}
        self._counters["completed"] += 1
        self._counters["recomputed"] += 1
        self._counters["abstention"] += int(abstention)
        self._counters["success"] += 1
        dataset = persisted["current_dataset"]
        self._current_dataset = None
        self._current_candidate_index = None
        self._current_view = None
        self._view_count = None
        self._excluded_count = None
        self._stage = "running"
        self._write_progress_antigravity()
        self._append_event_antigravity(
            "committed_success_reconciled", self._stage,
            {"dataset": dataset, "candidate_index": sequence, "record_sha256": record_sha256},
        )
        self._append_text_antigravity(
            f"커밋 성공 복구 dataset={dataset} index={sequence} sha256={record_sha256} "
            f"완료={self._counters['completed']}/{self.total}\n"
        )

    @classmethod
    def close_setup_failure_antigravity(
        cls,
        attempt_path: Path,
        *,
        total: int,
        process: ProcessIdentityAntigravity,
        checkpoint_path: Path,
        error: BaseException,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        """Create or deactivate parent-owned progress after a setup failure."""
        writer = cls(
            Path(attempt_path), total=total, process=process,
            checkpoint_path=checkpoint_path, counters=counters,
        )
        writer.close_antigravity("error", str(error) or type(error).__name__)

    def reconcile_orphan_antigravity(
        self,
        dataset: str,
        candidate_index: int,
        *,
        record_sha256: str,
        status: str,
        train_count: int,
        test_count: int,
        raw_length: int,
        test_rows: int | None = None,
        target_length: int | None = None,
        backend: str = "cpu",
        abstention: bool = False,
        error: str | None = None,
    ) -> None:
        """Atomically advance progress for one already-durable exact-next record."""
        self._require_active_antigravity()
        if type(dataset) is not str or not dataset:
            raise ValueError("progress orphan dataset is invalid")
        if type(candidate_index) is not int or candidate_index != self._counters["completed"]:
            raise ValueError("progress orphan is not the exact next candidate")
        if candidate_index < 0 or candidate_index >= self.total:
            raise ValueError("progress orphan candidate index is invalid")
        if type(record_sha256) is not str or _SHA256_ANTIGRAVITY.fullmatch(record_sha256) is None:
            raise ValueError("progress orphan record hash is invalid")
        if status not in {"success", "abstained", "error"}:
            raise ValueError("progress orphan status is invalid")
        if type(abstention) is not bool or abstention != (status == "abstained"):
            raise ValueError("progress orphan abstention is invalid")
        if status == "error":
            if type(error) is not str or not error:
                raise ValueError("progress orphan error detail is invalid")
        elif error is not None:
            raise ValueError("progress orphan success cannot contain an error")
        for value, name in (
            (train_count, "train_count"), (test_count, "test_count"),
            (raw_length, "raw_length"),
        ):
            _exact_nonnegative_antigravity(value, name)
        for value, name in ((test_rows, "test_rows"), (target_length, "target_length")):
            if value is not None:
                _exact_nonnegative_antigravity(value, name)
        if backend not in {"cpu", "mlx", "numba"}:
            raise ValueError("progress orphan backend is invalid")
        if any(value is not None for value in (
            self._current_dataset, self._current_candidate_index, self._current_view,
            self._view_count, self._excluded_count,
        )):
            raise ValueError("progress orphan cannot replace an active candidate")

        self._dimensions.update({
            "train_count": train_count,
            "test_count": test_count,
            "test_rows": test_count if test_rows is None else test_rows,
            "raw_length": raw_length,
            "target_length": raw_length if target_length is None else target_length,
        })
        self._backend = backend
        self._counters["completed"] += 1
        self._counters["recomputed"] += 1
        self._counters["abstention"] += int(abstention)
        if status == "error":
            self._counters["error"] += 1
            self._latest_error = error
        else:
            self._counters["success"] += 1
        self._stage = "resuming"
        self._write_progress_antigravity()
        self._append_event_antigravity(
            "orphan_reconciled",
            self._stage,
            {
                "dataset": dataset,
                "candidate_index": candidate_index,
                "status": status,
                "record_sha256": record_sha256,
            },
        )
        self._append_text_antigravity(
            f"고아 레코드 복구 dataset={dataset} index={candidate_index} "
            f"status={status} sha256={record_sha256} "
            f"완료={self._counters['completed']}/{self.total}\n"
        )

    def close_antigravity(self, status: str, latest_error: str | None = None) -> None:
        if os.getpid() != self._local_owner_pid:
            raise ValueError("progress mutation requires the parent owner process")
        if not self._active or self._stage in _TERMINAL:
            raise ValueError("progress is terminal")
        if status not in _TERMINAL | _RESUMABLE:
            raise ValueError("progress terminal status is invalid")
        if status == "complete" and self._counters["completed"] != self.total:
            raise ValueError("progress cannot complete before all candidates")
        self._latest_error = latest_error if latest_error is not None else self._latest_error
        if status in {"error", "failed"} and not self._latest_error:
            self._latest_error = "terminal failure"
        self._active = False
        self._stage = status
        self._current_dataset = None
        self._current_candidate_index = None
        self._current_view = None
        self._view_count = None
        self._excluded_count = None
        self._write_progress_antigravity()
        self._write_pid_antigravity(False)
        self._append_event_antigravity("terminal", status, {"latest_error": self._latest_error})
        self._append_text_antigravity(f"종료 stage={status} 오류={self._latest_error or '-'} 완료={self._counters['completed']}/{self.total}\n")

    @classmethod
    def open_interrupted(
        cls,
        attempt_path: Path,
        *,
        total: int = FROZEN_CONFIG.source_candidate_count,
        pid: int | None = None,
        start_identity: str | None = None,
        process: ProcessIdentityAntigravity | None = None,
        checkpoint_path: Path | None = None,
        pid_probe: Callable[[ProcessIdentityAntigravity], bool] | None = None,
        authority_preflight: Callable[[], None] | None = None,
    ) -> "ProgressWriterAntigravity":
        attempt = Path(attempt_path)
        if not attempt.is_dir() or attempt.is_symlink():
            raise ValueError("progress attempt path is invalid")
        if authority_preflight is not None and not callable(authority_preflight):
            raise ValueError("progress authority preflight is invalid")
        progress = _read_canonical_antigravity(attempt / "progress_antigravity.json")
        pid_payload = _read_canonical_antigravity(attempt / "pid_antigravity.json")
        if set(progress) != _PROGRESS_KEYS or set(pid_payload) != _PID_KEYS:
            raise ValueError("progress schema is invalid")
        if progress["total"] != total or progress["stage"] not in _RESUMABLE or progress["active"] is not False:
            raise ValueError("progress is not resumable")
        if pid_payload["pid"] != progress["pid"] or pid_payload["start_identity"] != progress["start_identity"] or pid_payload["active"] is not False:
            raise ValueError("progress PID state is invalid")
        if progress["backend"] != "cpu":
            raise ValueError("progress backend changed")
        if progress["eta_seconds"] is not None:
            raise ValueError("progress inactive ETA is invalid")
        if any(progress[name] is not None for name in (
            "current_dataset", "current_candidate_index", "current_view",
            "view_count", "excluded_count",
        )):
            raise ValueError("progress inactive current fields are invalid")
        for name in (
            "train_count", "train_missing", "train_used", "train_excluded",
            "test_count", "test_rows", "raw_length", "target_length",
        ):
            if progress[name] is not None:
                _exact_nonnegative_antigravity(progress[name], name)
        for name in ("heartbeat",):
            if type(progress[name]) is not str:
                raise ValueError(f"progress {name} is invalid")
            try:
                datetime.fromisoformat(progress[name].replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"progress {name} is invalid") from error
        latest_error = progress["latest_error"]
        if latest_error is not None and (type(latest_error) is not str or not latest_error):
            raise ValueError("progress latest error is invalid")
        stored = ProcessIdentityAntigravity(pid_payload["pid"], pid_payload["start_identity"])
        if process is None:
            if pid is None:
                process = current_process_identity_antigravity()
            else:
                process = ProcessIdentityAntigravity(pid, start_identity or f"{pid}:{time.time_ns()}")
        elif pid is not None and pid != process.pid:
            raise ValueError("progress PID does not match process")
        if process.pid == stored.pid:
            raise ValueError("replacement process numeric PID must be distinct")
        if process == stored:
            raise ValueError("replacement process must be distinct")
        probe = _pid_probe_default_antigravity if pid_probe is None else pid_probe
        try:
            if bool(probe(stored)):
                raise ValueError("progress stored process is active")
        except ValueError:
            raise
        except BaseException as error:
            raise ValueError("progress PID probe failed") from error
        counters = {name: progress[name] for name in ("completed", "imported", "recomputed", "success", "abstention", "error")}
        for name, value in counters.items():
            _exact_nonnegative_antigravity(value, name)
        if counters["completed"] > total:
            raise ValueError("progress completed count is invalid")
        if counters["imported"] + counters["recomputed"] != counters["completed"]:
            raise ValueError("progress source counter mismatch")
        if counters["success"] + counters["error"] != counters["completed"]:
            raise ValueError("progress outcome counter mismatch")
        if counters["abstention"] > counters["success"]:
            raise ValueError("progress abstention counter mismatch")
        if counters["completed"] and any(progress[name] is None for name in ("train_count", "test_count", "raw_length")):
            raise ValueError("progress completed dimensions are invalid")
        if counters["error"] and latest_error is None:
            raise ValueError("progress error detail is invalid")
        elapsed = _finite_nonnegative_antigravity(progress["elapsed_seconds"], "elapsed_seconds")
        _finite_nonnegative_antigravity(progress["throughput"], "throughput")
        if type(pid_payload["heartbeat"]) is not str:
            raise ValueError("progress PID heartbeat is invalid")
        try:
            datetime.fromisoformat(pid_payload["heartbeat"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("progress PID heartbeat is invalid") from error
        for filename in ("events_antigravity.jsonl", "text_antigravity.log"):
            path = attempt / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"progress log is invalid: {filename}")
        expected_checkpoint = attempt / "checkpoint_antigravity/state_antigravity.json"
        if type(progress["checkpoint"]) is not str or progress["checkpoint"] != str(expected_checkpoint):
            raise ValueError("progress checkpoint path changed")
        if checkpoint_path is not None and str(checkpoint_path) != str(expected_checkpoint):
            raise ValueError("progress checkpoint path changed")
        checkpoint = _read_canonical_antigravity(expected_checkpoint)
        checkpoint_keys = {
            "context_sha256", "status", "completed_units", "total_units", "error_count",
            "current_unit", "current_view", "payload_sha256", "record_sha256_prefix",
            "process", "error", "updated_at",
        }
        if set(checkpoint) != checkpoint_keys:
            raise ValueError("progress checkpoint schema is invalid")
        checkpoint_process = checkpoint["process"]
        if not isinstance(checkpoint_process, Mapping) or set(checkpoint_process) != {"pid", "start_identity"}:
            raise ValueError("progress checkpoint identity is invalid")
        if checkpoint_process["pid"] != stored.pid or checkpoint_process["start_identity"] != stored.start_identity:
            raise ValueError("progress checkpoint identity mismatch")
        if checkpoint["status"] not in _RESUMABLE or checkpoint["current_unit"] is not None or checkpoint["current_view"] is not None:
            raise ValueError("progress checkpoint state is invalid")
        for name in ("context_sha256", "payload_sha256"):
            value = checkpoint[name]
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"progress checkpoint {name} is invalid")
        if type(checkpoint["updated_at"]) is not str:
            raise ValueError("progress checkpoint update time is invalid")
        try:
            datetime.fromisoformat(checkpoint["updated_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("progress checkpoint update time is invalid") from error
        if checkpoint["completed_units"] != counters["completed"] or checkpoint["total_units"] != total or checkpoint["error_count"] != counters["error"]:
            raise ValueError("progress checkpoint counter mismatch")
        prefix = checkpoint["record_sha256_prefix"]
        if not isinstance(prefix, list) or len(prefix) != counters["completed"] or any(
            type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in prefix
        ):
            raise ValueError("progress checkpoint prefix is invalid")
        expected_payload_sha = hashlib.sha256(canonical_json_bytes_antigravity(prefix)).hexdigest()
        if checkpoint["payload_sha256"] != expected_payload_sha:
            raise ValueError("progress checkpoint payload hash changed")
        if checkpoint["error"] is not None:
            raise ValueError("progress checkpoint error state is invalid")
        if authority_preflight is not None:
            try:
                authority_preflight()
            except ValueError:
                raise
            except BaseException as error:
                raise ValueError("progress authority preflight failed") from error
        writer = cls(
            attempt, total=total, process=process,
            checkpoint_path=expected_checkpoint,
            elapsed_offset=elapsed, counters=counters,
        )
        writer._stage = "resuming"
        writer._backend = progress["backend"]
        writer._latest_error = progress["latest_error"]
        for name in writer._dimensions:
            writer._dimensions[name] = progress[name]
        writer._write_pid_antigravity(True)
        writer._write_progress_antigravity()
        writer._append_event_antigravity("resume", "resuming", {"replaced_pid": stored.pid})
        writer._append_text_antigravity(f"재개 pid={process.pid} 이전_pid={stored.pid} 완료={writer._counters['completed']}/{total}\n")
        return writer

    open_for_resume_antigravity = open_interrupted
    reopen_antigravity = open_interrupted

    @property
    def completed_count_antigravity(self) -> int:
        return self._counters["completed"]

    @property
    def error_count_antigravity(self) -> int:
        return self._counters["error"]

    @property
    def stage_antigravity(self) -> str:
        return self._stage


__all__ = ["ProgressWriterAntigravity"]
