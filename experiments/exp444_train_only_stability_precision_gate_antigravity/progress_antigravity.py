"""Parent-owned atomic progress, PID, event, and Korean text evidence."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

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
    temporary = path.with_name(path.name.replace(".json", ".tmp"))
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"{temporary.name} already exists")
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

    def close_antigravity(self, status: str, latest_error: str | None = None) -> None:
        if self._stage in _TERMINAL or not self._active and self._stage not in _RESUMABLE:
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
    ) -> "ProgressWriterAntigravity":
        attempt = Path(attempt_path)
        if not attempt.is_dir() or attempt.is_symlink():
            raise ValueError("progress attempt path is invalid")
        progress = _read_canonical_antigravity(attempt / "progress_antigravity.json")
        pid_payload = _read_canonical_antigravity(attempt / "pid_antigravity.json")
        if set(progress) != _PROGRESS_KEYS or set(pid_payload) != _PID_KEYS:
            raise ValueError("progress schema is invalid")
        if progress["total"] != total or progress["stage"] not in _RESUMABLE or progress["active"] is not False:
            raise ValueError("progress is not resumable")
        if pid_payload["pid"] != progress["pid"] or pid_payload["start_identity"] != progress["start_identity"] or pid_payload["active"] is not False:
            raise ValueError("progress PID state is invalid")
        if progress["eta_seconds"] is not None:
            _finite_nonnegative_antigravity(progress["eta_seconds"], "eta_seconds")
        stored = ProcessIdentityAntigravity(pid_payload["pid"], pid_payload["start_identity"])
        if process is None:
            if pid is None:
                process = current_process_identity_antigravity()
            else:
                process = ProcessIdentityAntigravity(pid, start_identity or f"{pid}:{time.time_ns()}")
        elif pid is not None and pid != process.pid:
            raise ValueError("progress PID does not match process")
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
        elapsed = _finite_nonnegative_antigravity(progress["elapsed_seconds"], "elapsed_seconds")
        for filename in ("events_antigravity.jsonl", "text_antigravity.log"):
            path = attempt / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"progress log is invalid: {filename}")
        if checkpoint_path is not None and str(checkpoint_path) != str(progress["checkpoint"]):
            raise ValueError("progress checkpoint path changed")
        writer = cls(
            attempt, total=total, process=process,
            checkpoint_path=checkpoint_path or Path(progress["checkpoint"]),
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
