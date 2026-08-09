"""Parent-owned atomic progress and append-only logs for Exp443."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
)
from .checkpoint_antigravity import ProcessIdentityAntigravity, WorkUnitAntigravity
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import PolicyDatasetResultAntigravity


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_all_antigravity(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("progress write made no progress")
        offset += written


def _normalize_active_antigravity(active: Iterable[WorkUnitAntigravity]) -> tuple[WorkUnitAntigravity, ...]:
    if isinstance(active, (str, bytes)):
        raise ValueError("active work units are invalid")
    units = tuple(active)
    if any(type(unit) is not WorkUnitAntigravity for unit in units):
        raise ValueError("active work units have the wrong identity")
    sequences = tuple(unit.sequence for unit in units)
    if len(sequences) != len(set(sequences)):
        raise ValueError("active work units are not unique")
    return tuple(sorted(units, key=lambda unit: unit.sequence))


_TERMINAL_PROGRESS_STAGES = frozenset({
    "complete", "error", "failed", "interrupted_resumable", "interrupted",
})


class ProgressWriterAntigravity:
    """The parent process owns all shared progress files; workers get no writer."""

    def __init__(
        self,
        attempt_path: Path,
        *,
        total: int = FROZEN_CONFIG.recompute_units,
        pid: int | None = None,
        checkpoint_path: Path | None = None,
        process: ProcessIdentityAntigravity | None = None,
    ) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("progress total is invalid")
        self._attempt_path = Path(attempt_path)
        if not self._attempt_path.is_dir() or self._attempt_path.is_symlink():
            raise ValueError("attempt path is invalid")
        self._total = total
        if process is not None:
            if type(process) is not ProcessIdentityAntigravity:
                raise ValueError("progress process identity is invalid")
            if pid is not None and pid != process.pid:
                raise ValueError("progress PID does not match process identity")
            self._pid = process.pid
            self._process_start_identity: str | None = process.start_identity
        else:
            self._pid = os.getpid() if pid is None else pid
            self._process_start_identity = None
        if type(self._pid) is not int or self._pid < 1:
            raise ValueError("progress PID is invalid")
        self._checkpoint_path = str(
            checkpoint_path if checkpoint_path is not None
            else self._attempt_path / "checkpoint_antigravity" / "state_antigravity.json"
        )
        self._started = time.monotonic()
        self._active: tuple[WorkUnitAntigravity, ...] = ()
        self._completed = 0
        self._imported = 0
        self._recomputed = 0
        self._success = 0
        self._abstention = 0
        self._error = 0
        self._latest_error: str | None = None
        self._stage = "initializing"
        self._backend = "cpu"
        self._dimensions: dict[str, int | None] = {
            "train_total": None, "train_missing": None, "train_used": None,
            "train_excluded": None, "test_rows": None, "target_length": None,
        }
        self._active_file_antigravity("events_antigravity.jsonl")
        self._active_file_antigravity("text_antigravity.log")
        self._write_pid_antigravity(True)
        self._write_progress_antigravity()

    @classmethod
    def open_for_resume_antigravity(
        cls,
        attempt_path: Path,
        *,
        total: int = FROZEN_CONFIG.recompute_units,
        pid: int | ProcessIdentityAntigravity | None = None,
        process: ProcessIdentityAntigravity | None = None,
        checkpoint_path: Path | None = None,
        checkpoint_store: object | None = None,
        pid_probe: object | None = None,
    ) -> "ProgressWriterAntigravity":
        """Reopen one interrupted attempt without truncating its evidence."""
        if type(total) is not int or total < 1:
            raise ValueError("progress total is invalid")
        if process is None and type(pid) is ProcessIdentityAntigravity:
            process = pid
            pid = None
        if process is not None and type(process) is not ProcessIdentityAntigravity:
            raise ValueError("progress process identity is invalid")
        attempt = Path(attempt_path)
        if not attempt.is_dir() or attempt.is_symlink():
            raise ValueError("attempt path is invalid")

        def read_payload(path: Path, label: str) -> Mapping[str, object]:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"progress {label} is invalid")
            try:
                payload = parse_canonical_json_bytes_antigravity(path.read_bytes())
            except (OSError, ValueError) as error:
                raise ValueError(f"progress {label} is invalid") from error
            if not isinstance(payload, Mapping):
                raise ValueError(f"progress {label} is invalid")
            return payload

        progress = read_payload(attempt / "progress_antigravity.json", "state")
        expected_progress = {
            "pid", "active", "current_policy", "current_dataset", "active_datasets",
            "train_total", "train_missing", "train_used", "train_excluded", "test_rows",
            "target_length", "stage", "backend", "completed", "total", "imported",
            "recomputed", "success", "abstention", "error", "throughput", "elapsed_seconds",
            "eta_seconds", "checkpoint", "heartbeat", "latest_error",
        }
        if set(progress) != expected_progress:
            raise ValueError("progress state schema is invalid")
        if progress["total"] != total:
            raise ValueError("progress total does not match persisted state")
        stage = progress["stage"]
        if stage not in {"interrupted_resumable", "interrupted"} or progress["active"] is not False:
            raise ValueError("progress attempt is not resumable")
        if type(progress["pid"]) is not int or progress["pid"] < 1:
            raise ValueError("progress persisted PID is invalid")
        if not isinstance(progress["active_datasets"], list) or any(type(item) is not str for item in progress["active_datasets"]):
            raise ValueError("progress active dataset list is invalid")
        for field in ("completed", "imported", "recomputed", "success", "abstention", "error"):
            value = progress[field]
            if type(value) is not int or value < 0:
                raise ValueError(f"progress {field} is invalid")
        if progress["completed"] > total or progress["recomputed"] > progress["completed"]:
            raise ValueError("progress counters are invalid")
        if progress["success"] + progress["abstention"] + progress["error"] > progress["completed"]:
            raise ValueError("progress outcome counters are invalid")
        for field in ("throughput", "elapsed_seconds"):
            value = progress[field]
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"progress {field} is invalid")
        if progress["eta_seconds"] is not None:
            raise ValueError("progress terminal ETA is invalid")
        if type(progress["checkpoint"]) is not str or not progress["checkpoint"]:
            raise ValueError("progress checkpoint path is invalid")
        if checkpoint_path is not None and str(checkpoint_path) != progress["checkpoint"]:
            raise ValueError("progress checkpoint path changed")
        if type(progress["backend"]) is not str or not progress["backend"]:
            raise ValueError("progress backend is invalid")
        if type(progress["heartbeat"]) is not str or not progress["heartbeat"].endswith("Z"):
            raise ValueError("progress heartbeat is invalid")
        latest_error = progress["latest_error"]
        if latest_error is not None and (type(latest_error) is not str or not latest_error):
            raise ValueError("progress latest error is invalid")
        dimension_values: dict[str, int | None] = {}
        for field in ("train_total", "train_missing", "train_used", "train_excluded", "test_rows", "target_length"):
            value = progress[field]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"progress dimension {field} is invalid")
            dimension_values[field] = value

        pid_payload = read_payload(attempt / "pid_antigravity.json", "PID")
        if set(pid_payload) not in ({"pid", "active", "heartbeat"}, {"pid", "active", "heartbeat", "start_identity"}):
            raise ValueError("progress PID schema is invalid")
        if type(pid_payload["pid"]) is not int or pid_payload["pid"] < 1 or pid_payload["pid"] != progress["pid"]:
            raise ValueError("progress PID identity is invalid")
        if pid_payload["active"] is not False:
            raise ValueError("progress process is active")
        stored_process: ProcessIdentityAntigravity | None = None
        if "start_identity" in pid_payload:
            stored_process = ProcessIdentityAntigravity(pid_payload["pid"], pid_payload["start_identity"])
        if pid_probe is not None:
            if not callable(pid_probe) or stored_process is None:
                raise ValueError("progress PID probe identity is invalid")
            try:
                if bool(pid_probe(stored_process)):
                    raise ValueError("progress process is active")
            except ValueError:
                raise
            except BaseException as error:
                raise ValueError("progress PID probe failed") from error

        if checkpoint_store is not None:
            checkpoint_completed = getattr(checkpoint_store, "completed_count_antigravity", None)
            if callable(checkpoint_completed):
                checkpoint_completed = checkpoint_completed()
            if checkpoint_completed != progress["completed"]:
                raise ValueError("progress checkpoint count does not match")

        for filename in ("events_antigravity.jsonl", "text_antigravity.log"):
            path = attempt / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"progress log is invalid: {filename}")
        if process is None:
            resumed_pid = os.getpid() if pid is None else pid
            if type(resumed_pid) is not int or resumed_pid < 1:
                raise ValueError("progress PID is invalid")
            process_start_identity: str | None = None
        else:
            if pid is not None and pid != process.pid:
                raise ValueError("progress PID does not match process identity")
            resumed_pid = process.pid
            process_start_identity = process.start_identity

        resumed = cls.__new__(cls)
        resumed._attempt_path = attempt
        resumed._total = total
        resumed._pid = resumed_pid
        resumed._process_start_identity = process_start_identity
        resumed._checkpoint_path = progress["checkpoint"]
        resumed._started = time.monotonic() - float(progress["elapsed_seconds"])
        resumed._active = ()
        resumed._completed = progress["completed"]
        resumed._imported = progress["imported"]
        resumed._recomputed = progress["recomputed"]
        resumed._success = progress["success"]
        resumed._abstention = progress["abstention"]
        resumed._error = progress["error"]
        resumed._latest_error = latest_error
        resumed._stage = "resuming"
        resumed._backend = progress["backend"]
        resumed._dimensions = dimension_values
        # The progress JSON is replaced atomically; both append-only logs are
        # left untouched.  The PID record is then bound to the new owner.
        resumed._write_progress_antigravity()
        resumed._write_pid_antigravity(True)
        return resumed

    @classmethod
    def reopen_antigravity(cls, *args: object, **kwargs: object) -> "ProgressWriterAntigravity":
        """Alias for callers that describe the operation as a reopen."""
        return cls.open_for_resume_antigravity(*args, **kwargs)

    def _active_file_antigravity(self, filename: str) -> None:
        path = self._attempt_path / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)

    def _write_pid_antigravity(self, active: bool) -> None:
        path = self._attempt_path / "pid_antigravity.json"
        temporary = self._attempt_path / "pid_antigravity.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("PID temporary already exists")
        payload_fields: dict[str, object] = {
            "pid": self._pid,
            "active": active,
            "heartbeat": _utc_now_antigravity(),
        }
        if self._process_start_identity is not None:
            payload_fields["start_identity"] = self._process_start_identity
        payload = canonical_json_bytes_antigravity(payload_fields)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            _write_all_antigravity(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, path)
            fsync_directory_antigravity(self._attempt_path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _append_antigravity(self, filename: str, payload: bytes) -> None:
        descriptor = os.open(self._attempt_path / filename, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            _write_all_antigravity(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _current_antigravity(self) -> WorkUnitAntigravity | None:
        return self._active[0] if self._active else None

    def _timing_antigravity(self, active: bool) -> tuple[float, float, float | None]:
        elapsed = max(0.0, time.monotonic() - self._started)
        throughput = self._completed / elapsed if elapsed else 0.0
        eta = None if not active or throughput <= 0.0 else (self._total - self._completed) / throughput
        if not all(math.isfinite(value) for value in (elapsed, throughput)) or (eta is not None and not math.isfinite(eta)):
            raise ValueError("progress timing is not finite")
        return elapsed, throughput, eta

    def _payload_antigravity(self) -> dict[str, object]:
        active = self._stage not in {"complete", "error", "failed", "interrupted_resumable", "interrupted"}
        elapsed, throughput, eta = self._timing_antigravity(active)
        current = self._current_antigravity()
        return {
            "pid": self._pid,
            "active": active,
            "current_policy": current.policy if current else None,
            "current_dataset": current.dataset if current else None,
            "active_datasets": [unit.dataset for unit in self._active],
            "train_total": self._dimensions["train_total"],
            "train_missing": self._dimensions["train_missing"],
            "train_used": self._dimensions["train_used"],
            "train_excluded": self._dimensions["train_excluded"],
            "test_rows": self._dimensions["test_rows"],
            "target_length": self._dimensions["target_length"],
            "stage": self._stage,
            "backend": self._backend,
            "completed": self._completed,
            "total": self._total,
            "imported": self._imported,
            "recomputed": self._recomputed,
            "success": self._success,
            "abstention": self._abstention,
            "error": self._error,
            "throughput": throughput,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "checkpoint": self._checkpoint_path,
            "heartbeat": _utc_now_antigravity(),
            "latest_error": self._latest_error,
        }

    def _write_progress_antigravity(self) -> None:
        final = self._attempt_path / "progress_antigravity.json"
        temporary = self._attempt_path / "progress_antigravity.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("progress temporary already exists")
        payload = canonical_json_bytes_antigravity(self._payload_antigravity())
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            _write_all_antigravity(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, final)
            fsync_directory_antigravity(self._attempt_path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _event_antigravity(self, event: str, **fields: object) -> None:
        payload = {"event": event, "heartbeat": _utc_now_antigravity(), **fields}
        self._append_antigravity(
            "events_antigravity.jsonl",
            canonical_json_bytes_antigravity(payload),
        )

    def _text_antigravity(self, message: str) -> None:
        self._append_antigravity("text_antigravity.log", (message + "\n").encode("utf-8"))

    def begin_units_antigravity(self, active: tuple[WorkUnitAntigravity, ...]) -> None:
        self._active = _normalize_active_antigravity(active)
        self._stage = "scoring"
        self._event_antigravity(
            "begin_units",
            units=[unit.to_payload_antigravity() for unit in self._active],
        )
        self._text_antigravity(
            f"시작 정책단위={len(self._active)} 현재={self._current_antigravity().policy if self._active else '-'}:{self._current_antigravity().dataset if self._active else '-'}"
        )
        self._write_progress_antigravity()
        self._write_pid_antigravity(True)

    def accept_unit_antigravity(self, result: PolicyDatasetResultAntigravity) -> None:
        if type(result) is not PolicyDatasetResultAntigravity:
            raise ValueError("progress result identity is invalid")
        matches = [unit for unit in self._active if unit.policy == result.policy and unit.dataset == result.dataset]
        if len(matches) != 1:
            raise ValueError("progress result is not an active unit")
        unit = matches[0]
        self._active = tuple(item for item in self._active if item != unit)
        self._completed += 1
        self._recomputed += 1
        if result.status == "scored":
            self._success += 1
        elif result.status == "abstained":
            self._abstention += 1
        else:
            self._error += 1
        self._dimensions.update({
            "train_total": result.train_total,
            "train_missing": result.train_missing,
            "train_used": result.train_used,
            "train_excluded": result.train_excluded,
            "test_rows": result.test_rows,
            "target_length": result.target_length,
        })
        self._stage = "scoring" if self._active else "finalizing"
        self._event_antigravity(
            "accept_unit",
            sequence=unit.sequence,
            policy=unit.policy,
            dataset=unit.dataset,
            status=result.status,
            completed=self._completed,
        )
        self._text_antigravity(
            f"수락 sequence={unit.sequence} 정책={unit.policy} 데이터셋={unit.dataset} 완료={self._completed}/{self._total}"
        )
        self._write_progress_antigravity()
        self._write_pid_antigravity(True)

    def accept_unit_and_align_antigravity(
        self,
        result: PolicyDatasetResultAntigravity,
        *,
        next_dimensions: Mapping[str, int | None] | None = None,
    ) -> None:
        """Advance the ordered prefix and publish the next dimensions once.

        ``accept_unit_antigravity`` remains the Task 7-compatible primitive and
        records the just-finished result dimensions.  Task 8's production
        runtime uses this seam so the current dataset and its dimensions are
        replaced in one atomic progress JSON write; no intermediate state can
        expose dataset B with dataset A's dimensions.
        """
        if type(result) is not PolicyDatasetResultAntigravity:
            raise ValueError("progress result identity is invalid")
        matches = [unit for unit in self._active if unit.policy == result.policy and unit.dataset == result.dataset]
        if len(matches) != 1:
            raise ValueError("progress result is not an active unit")
        if result.status not in {"scored", "abstained"}:
            raise ValueError("progress result status is invalid")
        unit = matches[0]
        self._active = tuple(item for item in self._active if item != unit)
        self._completed += 1
        self._recomputed += 1
        if result.status == "scored":
            self._success += 1
        else:
            self._abstention += 1
        dimensions = {
            "train_total": result.train_total,
            "train_missing": result.train_missing,
            "train_used": result.train_used,
            "train_excluded": result.train_excluded,
            "test_rows": result.test_rows,
            "target_length": result.target_length,
        }
        if next_dimensions is not None:
            if not isinstance(next_dimensions, Mapping) or set(next_dimensions) != set(dimensions):
                raise ValueError("progress next dimensions are invalid")
            dimensions = dict(next_dimensions)
        self._validate_dimensions_antigravity(dimensions)
        self._dimensions.update(dimensions)
        self._stage = "scoring" if self._active else "finalizing"
        self._event_antigravity(
            "accept_unit",
            sequence=unit.sequence,
            policy=unit.policy,
            dataset=unit.dataset,
            status=result.status,
            completed=self._completed,
            current_policy=self._active[0].policy if self._active else None,
            current_dataset=self._active[0].dataset if self._active else None,
            **dimensions,
        )
        self._text_antigravity(
            f"수락 sequence={unit.sequence} 정책={unit.policy} 데이터셋={unit.dataset} 완료={self._completed}/{self._total}"
        )
        self._write_progress_antigravity()
        self._write_pid_antigravity(True)

    def _validate_dimensions_antigravity(self, dimensions: Mapping[str, int | None]) -> None:
        expected = {"train_total", "train_missing", "train_used", "train_excluded", "test_rows", "target_length"}
        if not isinstance(dimensions, Mapping) or set(dimensions) != expected:
            raise ValueError("progress dimensions are invalid")
        values = tuple(dimensions[name] for name in ("train_total", "train_missing", "test_rows", "target_length"))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("progress dimensions are invalid")
        if dimensions["train_missing"] > dimensions["train_total"]:
            raise ValueError("progress missing count exceeds train total")
        for name in ("train_used", "train_excluded"):
            value = dimensions[name]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"progress {name} is invalid")
            if value is not None and value > dimensions["train_total"]:
                raise ValueError(f"progress {name} exceeds train total")

    def align_current_dimensions_antigravity(
        self,
        *,
        train_total: int,
        train_missing: int,
        train_used: int | None,
        train_excluded: int | None,
        test_rows: int,
        target_length: int,
    ) -> None:
        """Bind dimensions to the lowest active dataset after an ordered commit.

        Task 7's generic ``accept_unit`` keeps the just-completed result for
        backwards-compatible simulations.  Task 8 calls this parent-only seam
        when the active current dataset advances, preventing a progress line
        whose dataset name belongs to the next unit but whose dimensions belong
        to the previous one.
        """
        if self._stage in _TERMINAL_PROGRESS_STAGES:
            raise ValueError("terminal progress is closed")
        if not self._active:
            raise ValueError("no current active unit")
        dimensions = {
            "train_total": train_total, "train_missing": train_missing,
            "train_used": train_used, "train_excluded": train_excluded,
            "test_rows": test_rows, "target_length": target_length,
        }
        self._validate_dimensions_antigravity(dimensions)
        self._dimensions.update(dimensions)
        self._event_antigravity(
            "align_dimensions",
            sequence=self._active[0].sequence,
            policy=self._active[0].policy,
            dataset=self._active[0].dataset,
            **dimensions,
        )
        self._write_progress_antigravity()
        self._write_pid_antigravity(True)

    def heartbeat_antigravity(self, active: tuple[WorkUnitAntigravity, ...]) -> None:
        if self._stage in _TERMINAL_PROGRESS_STAGES:
            raise ValueError("terminal progress is closed")
        self._active = _normalize_active_antigravity(active)
        if self._stage not in {"complete", "error", "failed", "interrupted_resumable", "interrupted"}:
            self._stage = "waiting_for_results"
        self._event_antigravity(
            "heartbeat",
            active_datasets=[unit.dataset for unit in self._active],
            completed=self._completed,
        )
        self._text_antigravity(
            f"하트비트 현재={self._current_antigravity().policy if self._active else '-'}:{self._current_antigravity().dataset if self._active else '-'} 완료={self._completed}/{self._total}"
        )
        self._write_progress_antigravity()
        self._write_pid_antigravity(True)

    def terminal_antigravity(self, stage: str, latest_error: str | None) -> None:
        if stage not in {"complete", "error", "failed", "interrupted_resumable", "interrupted"}:
            raise ValueError("terminal progress stage is invalid")
        if latest_error is not None and (type(latest_error) is not str or not latest_error):
            raise ValueError("latest error is invalid")
        self._active = ()
        self._stage = stage
        self._latest_error = latest_error
        if stage in {"error", "failed"}:
            self._error = max(1, self._error)
        self._event_antigravity("terminal", stage=stage, latest_error=latest_error)
        self._text_antigravity(f"종료 stage={stage} 오류={latest_error or '-'} 완료={self._completed}/{self._total}")
        self._write_progress_antigravity()
        self._write_pid_antigravity(False)

    @property
    def completed_count_antigravity(self) -> int:
        return self._completed

    @property
    def stage_antigravity(self) -> str:
        """Expose the immutable current stage for runtime failure routing."""
        return self._stage


__all__ = ["ProgressWriterAntigravity"]
