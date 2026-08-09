"""Parent-owned atomic progress and append-only logs for Exp443."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    fsync_directory_antigravity,
)
from .checkpoint_antigravity import WorkUnitAntigravity
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


class ProgressWriterAntigravity:
    """The parent process owns all shared progress files; workers get no writer."""

    def __init__(
        self,
        attempt_path: Path,
        *,
        total: int = FROZEN_CONFIG.recompute_units,
        pid: int | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("progress total is invalid")
        self._attempt_path = Path(attempt_path)
        if not self._attempt_path.is_dir() or self._attempt_path.is_symlink():
            raise ValueError("attempt path is invalid")
        self._total = total
        self._pid = os.getpid() if pid is None else pid
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

    def _active_file_antigravity(self, filename: str) -> None:
        path = self._attempt_path / filename
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(descriptor)

    def _write_pid_antigravity(self, active: bool) -> None:
        path = self._attempt_path / "pid_antigravity.json"
        temporary = self._attempt_path / "pid_antigravity.tmp"
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("PID temporary already exists")
        payload = canonical_json_bytes_antigravity({
            "pid": self._pid,
            "active": active,
            "heartbeat": _utc_now_antigravity(),
        })
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

    def heartbeat_antigravity(self, active: tuple[WorkUnitAntigravity, ...]) -> None:
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


__all__ = ["ProgressWriterAntigravity"]
