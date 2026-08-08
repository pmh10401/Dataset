"""Process-owned, atomic progress and append-only evidence for Exp441."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Iterable

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .config_antigravity import FROZEN_CONFIG


_DIMENSION_STATUS_ANTIGRAVITY = "not_available_in_frozen_selector_source"


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_all_antigravity(descriptor: int, payload: bytes) -> None:
    """Handle POSIX short writes so evidence bytes are never silently truncated."""
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("progress write made no progress")
        offset += written


class ProgressWriterAntigravity:
    """Own one attempt's observable state; workers never write these artifacts."""

    def __init__(self, attempt_path: Path, *, total: int, pid: int | None = None) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("progress total is invalid")
        self._attempt_path = Path(attempt_path)
        if not self._attempt_path.is_dir() or self._attempt_path.is_symlink():
            raise ValueError("attempt path is invalid")
        self._total = total
        self._pid = os.getpid() if pid is None else pid
        if type(self._pid) is not int or self._pid < 1:
            raise ValueError("progress PID is invalid")
        self._started = time.monotonic()
        self._completed = 0
        self._latest_error: str | None = None
        self._state = self._base_state_antigravity(active=True, stage="allocated")
        write_create_only_antigravity(
            self._attempt_path / "pid_antigravity",
            canonical_json_bytes_antigravity({"pid": self._pid, "active": True, "heartbeat": _utc_now_antigravity()}),
        )
        write_create_only_antigravity(self._attempt_path / "events_antigravity.jsonl", b"")
        write_create_only_antigravity(self._attempt_path / "text_antigravity.log", b"")
        self._write_progress_antigravity()

    def _base_state_antigravity(self, *, active: bool, stage: str) -> dict[str, object]:
        return {
            "pid": self._pid, "active": active, "current_dataset": None,
            "current_record_name": None, "active_datasets": [],
            "train_row_count": None, "test_row_count": None, "raw_length": None,
            "dimension_status": _DIMENSION_STATUS_ANTIGRAVITY, "stage": stage,
            "backend": "cpu", "completed": self._completed, "total": self._total,
            "imported": self._completed, "recomputed": 0, "success": 0, "error": 0,
            "provisional_candidates": None, "outside_hard_candidates": None,
            "eligible_candidates": None, "selected_candidates": None,
            "throughput": 0.0, "elapsed_seconds": 0.0, "eta_seconds": None,
            "source_checkpoint": FROZEN_CONFIG.checkpoint_records_path, "heartbeat": _utc_now_antigravity(),
            "latest_error": self._latest_error,
        }

    @property
    def completed_count_antigravity(self) -> int:
        """Return the parent-accepted prefix length, never a worker-completion count."""
        return self._completed

    def _refresh_timing_antigravity(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._started)
        throughput = self._completed / elapsed if elapsed else 0.0
        self._state["elapsed_seconds"] = elapsed
        self._state["throughput"] = throughput
        self._state["eta_seconds"] = ((self._total - self._completed) / throughput if throughput else None)
        self._state["heartbeat"] = _utc_now_antigravity()
        self._state["completed"] = self._completed
        self._state["imported"] = self._completed

    def _write_progress_antigravity(self) -> None:
        self._refresh_timing_antigravity()
        final = self._attempt_path / "progress_antigravity.json"
        temporary = self._attempt_path / "progress_antigravity.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            payload = canonical_json_bytes_antigravity(self._state)
            _write_all_antigravity(descriptor, payload)
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
            os.replace(temporary, final)
        except BaseException:
            try: temporary.unlink()
            except FileNotFoundError: pass
            raise

    def _append_antigravity(self, filename: str, payload: bytes) -> None:
        descriptor = os.open(self._attempt_path / filename, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            _write_all_antigravity(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def begin_record_antigravity(self, *, dataset: str, record_name: str, active_datasets: Iterable[str]) -> None:
        if not isinstance(dataset, str) or not dataset or not isinstance(record_name, str) or not record_name:
            raise ValueError("progress record context is invalid")
        active = tuple(active_datasets)
        if not active or dataset not in active or any(not isinstance(value, str) or not value for value in active):
            raise ValueError("active datasets are invalid")
        self._state.update({
            "active": True, "stage": "verifying_frozen_sources", "current_dataset": dataset,
            "current_record_name": record_name, "active_datasets": list(active),
        })
        self._write_progress_antigravity()

    def accept_record_antigravity(
        self, *, dataset: str, record_name: str, active_datasets: Iterable[str],
        next_dataset: str | None = None, next_record_name: str | None = None,
        test_row_count: int, raw_length: int | None,
    ) -> None:
        if self._state["current_dataset"] != dataset or self._state["current_record_name"] != record_name:
            raise ValueError("record acceptance is out of order")
        active = tuple(active_datasets)
        if any(not isinstance(value, str) or not value for value in active) or dataset in active:
            raise ValueError("accepted active dataset set is invalid")
        if active:
            if (not isinstance(next_dataset, str) or not next_dataset or next_dataset != active[0] or
                    not isinstance(next_record_name, str) or not next_record_name):
                raise ValueError("next record context is invalid")
        elif next_dataset is not None or next_record_name is not None:
            raise ValueError("terminal record context is invalid")
        self._completed += 1
        if type(test_row_count) is not int or test_row_count < 0 or (raw_length is not None and (type(raw_length) is not int or raw_length < 1)):
            raise ValueError("accepted dimensions are invalid")
        event = {"sequence": self._completed - 1, "dataset": dataset, "record_name": record_name, "test_row_count": test_row_count, "raw_length": raw_length, "accepted": True}
        self._append_antigravity("events_antigravity.jsonl", canonical_json_bytes_antigravity(event).rstrip(b"\n") + b"\n")
        self._append_antigravity("text_antigravity.log", f"accepted dataset={dataset} record={record_name} test_row_count={test_row_count} raw_length={raw_length}\n".encode("utf-8"))
        self._state.update({
            "active_datasets": list(active), "success": self._completed,
            "current_dataset": next_dataset, "current_record_name": next_record_name,
            "test_row_count": None if next_dataset else None, "raw_length": None,
            "stage": "verifying_frozen_sources" if next_dataset else "verifying_source_summary",
        })
        self._write_progress_antigravity()

    def terminal_antigravity(self, *, stage: str, latest_error: str | None) -> None:
        if stage not in {"complete", "failed", "interrupted"}:
            raise ValueError("terminal progress stage is invalid")
        if latest_error is not None and (not isinstance(latest_error, str) or not latest_error):
            raise ValueError("latest error is invalid")
        self._latest_error = latest_error
        self._state.update({
            "active": False, "stage": stage, "current_dataset": None, "current_record_name": None,
            "active_datasets": [], "error": int(stage == "failed"), "latest_error": latest_error,
        })
        self._write_progress_antigravity()
        final_pid = self._attempt_path / "pid_antigravity"
        temporary = self._attempt_path / "pid_terminal_antigravity.tmp"
        write_create_only_antigravity(temporary, canonical_json_bytes_antigravity({"pid": self._pid, "active": False, "heartbeat": _utc_now_antigravity()}))
        os.replace(temporary, final_pid)

    def source_replay_complete_antigravity(self) -> None:
        """Leave verification after the complete parent-accepted prefix before selection."""
        if self._completed != self._total:
            raise ValueError("source replay is incomplete")
        self._state.update({"stage": "selecting", "current_dataset": None, "current_record_name": None, "active_datasets": []})
        self._write_progress_antigravity()

    def selection_counts_antigravity(self, *, provisional: int, outside_hard: int, eligible: int, selected: int) -> None:
        for value in (provisional, outside_hard, eligible, selected):
            if type(value) is not int or value < 0:
                raise ValueError("selection count is invalid")
        self._state.update({"provisional_candidates": provisional, "outside_hard_candidates": outside_hard, "eligible_candidates": eligible, "selected_candidates": selected, "stage": "finalizing"})
        self._write_progress_antigravity()


__all__ = ["ProgressWriterAntigravity"]
