"""Create-only ordered checkpoint authority for Exp444.

The checkpoint is deliberately small: only the frozen thirty candidate
coordinates are committed, in sequence ``0..29``.  A result file is fsynced
before the state prefix is atomically replaced.  Consequently a process
crash can leave at most one exact-next record orphan; resume validates and
reconciles that record, while every other inventory mismatch fails closed.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
from typing import Any

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD = re.compile(r"^(\d{4})_record_antigravity\.json$")
_TERMINAL = frozenset({"error", "failed", "complete"})
_RESUMABLE = frozenset({"interrupted", "interrupted_resumable"})


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_bytes_antigravity(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha_antigravity(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _write_atomic_json_antigravity(path: Path, payload: Mapping[str, object]) -> bytes:
    """Write canonical JSON through an fsynced same-directory temporary file."""
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
                raise OSError("checkpoint atomic write made no progress")
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
    return encoded


def _read_canonical_antigravity(path: Path) -> tuple[Mapping[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint file is invalid: {path.name}")
    try:
        encoded = path.read_bytes()
        payload = parse_canonical_json_bytes_antigravity(encoded)
    except (OSError, ValueError) as error:
        raise ValueError(f"checkpoint file is invalid: {path.name}") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes_antigravity(payload) != encoded:
        raise ValueError(f"checkpoint file is not canonical: {path.name}")
    return payload, encoded


@dataclass(frozen=True)
class ProcessIdentityAntigravity:
    """PID plus a caller-owned start token; PID alone is never sufficient."""

    pid: int
    start_identity: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("process PID is invalid")
        if type(self.start_identity) is not str or not self.start_identity:
            raise ValueError("process start identity is invalid")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {"pid": self.pid, "start_identity": self.start_identity}

    @classmethod
    def from_payload_antigravity(cls, payload: Mapping[str, object]) -> "ProcessIdentityAntigravity":
        if not isinstance(payload, Mapping) or set(payload) != {"pid", "start_identity"}:
            raise ValueError("process identity payload is invalid")
        return cls(payload["pid"], payload["start_identity"])


def current_process_identity_antigravity() -> ProcessIdentityAntigravity:
    """Return a stable-enough local process token for production callers."""
    pid = os.getpid()
    # ``/proc`` is not guaranteed on macOS; a monotonic wall-clock token is
    # still distinct for every owner and is persisted for later PID checks.
    return ProcessIdentityAntigravity(pid, f"{pid}:{os.stat(__file__).st_ino}:{time_ns_antigravity()}")


def time_ns_antigravity() -> int:
    return __import__("time").time_ns()


def _default_pid_probe_antigravity(process: ProcessIdentityAntigravity) -> bool:
    """Best-effort liveness probe used only when a runtime did not inject one."""
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _context_payload_antigravity(context: CheckpointContextAntigravity) -> dict[str, object]:
    if type(context) is not CheckpointContextAntigravity:
        raise ValueError("checkpoint context identity is invalid")
    context.validate_antigravity()
    return context.to_payload()


def _context_sha_antigravity(context_bytes: bytes) -> str:
    return _sha256_bytes_antigravity(context_bytes)


def _record_path_antigravity(root: Path, sequence: int) -> Path:
    return root / "records_antigravity" / f"{sequence:04d}_record_antigravity.json"


def _payload_hash_antigravity(prefix: tuple[str, ...]) -> str:
    return _sha256_bytes_antigravity(canonical_json_bytes_antigravity(list(prefix)))


def _validate_result_antigravity(sequence: int, payload: object) -> CandidateStabilityResultAntigravity:
    if type(sequence) is not int or sequence < 0 or sequence >= FROZEN_CONFIG.source_candidate_count:
        raise ValueError("checkpoint sequence is invalid")
    if not isinstance(payload, CandidateStabilityResultAntigravity):
        raise ValueError("checkpoint result identity is invalid")
    if payload.candidate.original_rank != sequence + 1:
        raise ValueError("checkpoint result rank does not match sequence")
    return payload


def _validate_record_payload_antigravity(
    sequence: int,
    payload: object,
    encoded: bytes,
) -> CandidateStabilityResultAntigravity | None:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint record payload is invalid")
    keys = set(payload)
    if keys == {"sequence", "result"}:
        if payload["sequence"] != sequence or not isinstance(payload["result"], Mapping):
            raise ValueError("checkpoint record identity is invalid")
        try:
            result = CandidateStabilityResultAntigravity.from_payload(payload["result"])
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint record result is invalid") from error
        _validate_result_antigravity(sequence, result)
        return result
    if keys == {"sequence", "result", "status", "error"}:
        if payload["sequence"] != sequence or payload["status"] != "error" or payload["result"] is not None:
            raise ValueError("checkpoint error record is invalid")
        error = payload["error"]
        if not isinstance(error, Mapping) or set(error) != {"type", "message"}:
            raise ValueError("checkpoint error record is invalid")
        if type(error["type"]) is not str or not error["type"] or type(error["message"]) is not str or not error["message"]:
            raise ValueError("checkpoint error record is invalid")
        return None
    raise ValueError("checkpoint record schema is invalid")


def _scan_records_antigravity(
    root: Path,
    expected_prefix: tuple[str, ...],
    total: int,
) -> tuple[tuple[str, ...], int | None, CandidateStabilityResultAntigravity | None]:
    records = root / "records_antigravity"
    if records.is_symlink() or not records.is_dir():
        raise ValueError("checkpoint records directory is invalid")
    found: list[int] = []
    for entry in records.iterdir():
        match = _RECORD.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_file():
            raise ValueError("checkpoint record inventory is invalid")
        found.append(int(match.group(1)))
    count = len(expected_prefix)
    names = sorted(found)
    if names == list(range(count)):
        orphan = None
    elif count < total and names == list(range(count + 1)):
        orphan = count
    else:
        raise ValueError("checkpoint record inventory is invalid")
    prefix = list(expected_prefix)
    orphan_result: CandidateStabilityResultAntigravity | None = None
    if orphan is not None:
        path = _record_path_antigravity(root, orphan)
        payload, encoded = _read_canonical_antigravity(path)
        orphan_result = _validate_record_payload_antigravity(orphan, payload, encoded)
        prefix.append(_sha256_bytes_antigravity(encoded))
    for sequence, digest in enumerate(prefix[:count]):
        path = _record_path_antigravity(root, sequence)
        payload, encoded = _read_canonical_antigravity(path)
        if _sha256_bytes_antigravity(encoded) != digest:
            raise ValueError("checkpoint record bytes changed")
        _validate_record_payload_antigravity(sequence, payload, encoded)
    return tuple(prefix), orphan, orphan_result


class Exp444CheckpointStoreAntigravity:
    """Durable ordered prefix for exactly thirty Exp444 candidates."""

    def __init__(
        self,
        root: Path,
        context: CheckpointContextAntigravity,
        process: ProcessIdentityAntigravity,
        prefix: tuple[str, ...],
        status: str,
        error_count: int,
        *,
        finalization_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.context = context
        self.process = process
        self._prefix = list(prefix)
        self._status = status
        self._error_count = error_count
        self._finalization_only = finalization_only

    @classmethod
    def create(
        cls,
        root: Path,
        context: CheckpointContextAntigravity,
        process: ProcessIdentityAntigravity,
    ) -> "Exp444CheckpointStoreAntigravity":
        context_payload = _context_payload_antigravity(context)
        if type(process) is not ProcessIdentityAntigravity:
            raise ValueError("checkpoint process identity is invalid")
        root = Path(root)
        if "antigravity" not in root.name or root.exists() or root.is_symlink():
            raise ValueError("checkpoint root already exists or is invalid")
        root.mkdir(mode=0o700)
        records = root / "records_antigravity"
        records.mkdir(mode=0o700)
        context_bytes = canonical_json_bytes_antigravity(context_payload)
        write_create_only_antigravity(root / "context_antigravity.json", context_bytes)
        fsync_directory_antigravity(records)
        fsync_directory_antigravity(root)
        state_payload = cls._state_payload(
            context_sha=_context_sha_antigravity(context_bytes),
            context=context,
            process=process,
            prefix=(),
            status="running",
            error_count=0,
            error=None,
        )
        _write_atomic_json_antigravity(root / "state_antigravity.json", state_payload)
        return cls(root, context, process, (), "running", 0)

    # Explicit aliases make the boundary convenient for runtime wrappers and
    # retain the naming style used by the older Exp440/443 stores.
    create_antigravity = create

    @staticmethod
    def _state_payload(
        *, context_sha: str, context: CheckpointContextAntigravity,
        process: ProcessIdentityAntigravity, prefix: tuple[str, ...],
        status: str, error_count: int, error: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return {
            "context_sha256": context_sha,
            "status": status,
            "completed_units": len(prefix),
            "total_units": context.total_units,
            "error_count": error_count,
            "current_unit": None,
            "current_view": None,
            "payload_sha256": _payload_hash_antigravity(prefix),
            "record_sha256_prefix": list(prefix),
            "process": process.to_payload_antigravity(),
            "error": dict(error) if error is not None else None,
            "updated_at": _utc_now_antigravity(),
        }

    @property
    def status_antigravity(self) -> str:
        return self._status

    @property
    def completed_count_antigravity(self) -> int:
        return len(self._prefix)

    @property
    def error_count_antigravity(self) -> int:
        return self._error_count

    @property
    def record_sha256_prefix_antigravity(self) -> tuple[str, ...]:
        return tuple(self._prefix)

    @property
    def finalization_only_antigravity(self) -> bool:
        return self._finalization_only

    def _require_writable_antigravity(self) -> None:
        if self._finalization_only:
            raise ValueError("checkpoint is finalization-only")
        if self._status in _TERMINAL:
            raise ValueError("checkpoint is terminal")
        if self._status not in {"running", "interrupted", "interrupted_resumable"}:
            raise ValueError("checkpoint status is invalid")

    def _write_state_antigravity(self, *, error: Mapping[str, object] | None = None) -> None:
        context_path = self.root / "context_antigravity.json"
        _, context_bytes = _read_canonical_antigravity(context_path)
        payload = self._state_payload(
            context_sha=_context_sha_antigravity(context_bytes),
            context=self.context, process=self.process,
            prefix=tuple(self._prefix), status=self._status,
            error_count=self._error_count, error=error,
        )
        _write_atomic_json_antigravity(self.root / "state_antigravity.json", payload)

    def commit_antigravity(self, sequence: int, result: CandidateStabilityResultAntigravity) -> None:
        self._require_writable_antigravity()
        if type(sequence) is not int or sequence != len(self._prefix) or sequence < 0 or sequence >= self.context.total_units:
            raise ValueError("checkpoint commit is not the next sequence")
        result = _validate_result_antigravity(sequence, result)
        encoded = canonical_json_bytes_antigravity({"sequence": sequence, "result": result.to_payload()})
        path = _record_path_antigravity(self.root, sequence)
        try:
            write_create_only_antigravity(path, encoded)
        except FileExistsError as error:
            raise ValueError("checkpoint record already exists") from error
        fsync_directory_antigravity(path.parent)
        self._prefix.append(_sha256_bytes_antigravity(encoded))
        self._status = "complete" if len(self._prefix) == self.context.total_units else "running"
        self._write_state_antigravity()

    def commit_error_antigravity(self, sequence: int, error: BaseException) -> None:
        self._require_writable_antigravity()
        if type(sequence) is not int or sequence != len(self._prefix) or sequence < 0 or sequence >= self.context.total_units:
            raise ValueError("checkpoint commit is not the next sequence")
        if not isinstance(error, BaseException):
            raise ValueError("checkpoint error identity is invalid")
        message = str(error) or type(error).__name__
        error_payload = {"type": type(error).__name__, "message": message}
        encoded = canonical_json_bytes_antigravity({"sequence": sequence, "result": None, "status": "error", "error": error_payload})
        path = _record_path_antigravity(self.root, sequence)
        try:
            write_create_only_antigravity(path, encoded)
        except FileExistsError as error:
            raise ValueError("checkpoint record already exists") from error
        fsync_directory_antigravity(path.parent)
        self._prefix.append(_sha256_bytes_antigravity(encoded))
        self._error_count += 1
        self._status = "error"
        self._write_state_antigravity(error=error_payload)

    def mark_interrupted_antigravity(self) -> None:
        self._require_writable_antigravity()
        self._status = "interrupted"
        self._write_state_antigravity()

    mark_interrupted_resumable_antigravity = mark_interrupted_antigravity
    interrupt_antigravity = mark_interrupted_antigravity

    @classmethod
    def open_interrupted(
        cls,
        root: Path,
        context: CheckpointContextAntigravity,
        replacement_process: ProcessIdentityAntigravity,
        pid_probe: Callable[[ProcessIdentityAntigravity], bool] | None = None,
        *,
        allow_complete_finalization: bool = False,
    ) -> "Exp444CheckpointStoreAntigravity":
        expected_context = _context_payload_antigravity(context)
        if type(replacement_process) is not ProcessIdentityAntigravity:
            raise ValueError("checkpoint replacement process identity is invalid")
        if type(allow_complete_finalization) is not bool:
            raise ValueError("checkpoint finalization flag is invalid")
        root = Path(root)
        if "antigravity" not in root.name or root.is_symlink() or not root.is_dir():
            raise ValueError("checkpoint root is invalid")
        context_payload, context_bytes = _read_canonical_antigravity(root / "context_antigravity.json")
        if dict(context_payload) != expected_context:
            raise ValueError("checkpoint context changed")
        state_payload, _ = _read_canonical_antigravity(root / "state_antigravity.json")
        expected_keys = {
            "context_sha256", "status", "completed_units", "total_units", "error_count",
            "current_unit", "current_view", "payload_sha256", "record_sha256_prefix",
            "process", "error", "updated_at",
        }
        if set(state_payload) != expected_keys:
            raise ValueError("checkpoint state schema is invalid")
        if state_payload["context_sha256"] != _context_sha_antigravity(context_bytes):
            raise ValueError("checkpoint context hash changed")
        status = state_payload["status"]
        # A process can die between the initial ``running`` state replacement
        # and its interruption heartbeat.  A stale running owner is therefore
        # accepted as an interrupted prefix, but a live owner is rejected
        # below.  This is also the crash window in which one exact-next record
        # may need reconciliation.
        if status not in _RESUMABLE | _TERMINAL | {"running"}:
            raise ValueError("checkpoint is not resumable")
        stored_process = ProcessIdentityAntigravity.from_payload_antigravity(state_payload["process"])
        if stored_process == replacement_process:
            raise ValueError("replacement process must be distinct")
        probe = _default_pid_probe_antigravity if pid_probe is None else pid_probe
        if not callable(probe):
            raise ValueError("checkpoint PID probe is invalid")
        try:
            if bool(probe(stored_process)):
                raise ValueError("checkpoint stored process is active")
        except ValueError:
            raise
        except BaseException as error:
            raise ValueError("checkpoint PID probe failed") from error
        total = context.total_units
        completed = state_payload["completed_units"]
        prefix_payload = state_payload["record_sha256_prefix"]
        error_count = state_payload["error_count"]
        if type(error_count) is not int or error_count < 0:
            raise ValueError("checkpoint error count is invalid")
        if type(state_payload["total_units"]) is not int or state_payload["total_units"] != total:
            raise ValueError("checkpoint total is invalid")
        _valid_sha_antigravity(state_payload["payload_sha256"], "checkpoint payload hash")
        if type(completed) is not int or completed < 0 or completed > total or not isinstance(prefix_payload, list):
            raise ValueError("checkpoint prefix is invalid")
        if len(prefix_payload) != completed or any(_SHA256.fullmatch(value or "") is None for value in prefix_payload):
            raise ValueError("checkpoint prefix is invalid")
        prefix, orphan, orphan_result = _scan_records_antigravity(root, tuple(prefix_payload), total)
        if status in _TERMINAL and status != "complete":
            raise ValueError("checkpoint terminal error is not resumable")
        if status == "complete":
            if not allow_complete_finalization or completed != total or orphan is not None or tuple(prefix) != tuple(prefix_payload):
                raise ValueError("checkpoint is complete")
            if state_payload["payload_sha256"] != _payload_hash_antigravity(tuple(prefix_payload)):
                raise ValueError("checkpoint payload hash changed")
            return cls(root, context, stored_process, tuple(prefix_payload), "complete", error_count, finalization_only=True)
        orphan_error: Mapping[str, object] | None = None
        if orphan is not None and orphan_result is None:
            # A terminal orphan is valid only if it is the exact next error
            # record; _scan_records validates the payload shape above.
            orphan_path = _record_path_antigravity(root, orphan)
            orphan_payload, _ = _read_canonical_antigravity(orphan_path)
            if orphan_payload.get("status") != "error":
                raise ValueError("checkpoint orphan result is invalid")
            orphan_error = orphan_payload.get("error")
        if orphan_error is not None:
            # A durable error record is itself terminal even if the process
            # died before the state replacement.  Reconcile it once, then
            # retain the terminal error instead of reopening a failed prefix.
            resumed = cls(root, context, replacement_process, tuple(prefix), "error", error_count + 1)
            resumed._error_count = error_count + 1
            resumed._write_state_antigravity(error=orphan_error)
            return resumed
        resumed = cls(root, context, replacement_process, tuple(prefix), "running", error_count)
        if orphan is not None:
            # The exact-next record was already durable; only the prefix and
            # owner binding move forward in the atomic state replacement.
            resumed._write_state_antigravity()
        else:
            resumed._write_state_antigravity()
        return resumed

    open_for_resume_antigravity = open_interrupted

    def results_antigravity(self) -> dict[int, CandidateStabilityResultAntigravity]:
        results: dict[int, CandidateStabilityResultAntigravity] = {}
        for sequence in range(len(self._prefix)):
            payload, encoded = _read_canonical_antigravity(_record_path_antigravity(self.root, sequence))
            if _sha256_bytes_antigravity(encoded) != self._prefix[sequence]:
                raise ValueError("checkpoint record bytes changed")
            result = _validate_record_payload_antigravity(sequence, payload, encoded)
            if result is not None:
                results[sequence] = result
        return results


__all__ = [
    "Exp444CheckpointStoreAntigravity", "ProcessIdentityAntigravity",
    "current_process_identity_antigravity",
]
