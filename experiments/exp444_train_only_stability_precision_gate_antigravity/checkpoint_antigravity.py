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
import stat
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
    ProfileCheckpointContextAntigravity,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD = re.compile(r"^(\d{4})_record_antigravity\.json$")
_TERMINAL = frozenset({"error", "failed", "complete"})
_RESUMABLE = frozenset({"interrupted", "interrupted_resumable"})
_CHECKPOINT_STATE_KEYS_ANTIGRAVITY = frozenset({
    "context_sha256", "status", "completed_units", "total_units", "error_count",
    "current_unit", "current_view", "payload_sha256", "record_sha256_prefix",
    "process", "error", "updated_at",
})
_CHECKPOINT_DIRECTORY_FLAGS_ANTIGRAVITY = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_CHECKPOINT_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


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
    _discard_stale_atomic_temps_antigravity(path)
    temporary = path.with_name(
        f".{path.name}.antigravity-{os.getpid()}-{time_ns_antigravity()}.tmp"
    )
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


def _discard_stale_atomic_temps_antigravity(path: Path) -> None:
    """Remove abandoned temp files without ever parsing or promoting them."""
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


@dataclass(frozen=True)
class CheckpointResumeInspectionAntigravity:
    """Immutable, read-only description of the prefix accepted for resume."""

    stored_status: str
    stored_completed_units: int
    accepted_completed_units: int
    accepted_record_sha256_prefix: tuple[str, ...]
    orphan_sequence: int | None
    orphan_result: CandidateStabilityResultAntigravity | None
    orphan_error_type: str | None
    orphan_error_message: str | None

    @property
    def has_orphan_antigravity(self) -> bool:
        return self.orphan_sequence is not None

    @property
    def orphan_is_error_antigravity(self) -> bool:
        return self.orphan_error_type is not None


@dataclass(frozen=True)
class CommittedPrefixAuthorityAntigravity:
    """Exact immutable checkpoint prefix handed to Exp445 progress recovery only."""

    sequence: int
    record_sha256: str
    record_sha256_prefix: tuple[str, ...]
    context_sha256: str
    total_units: int
    status: str
    process: ProcessIdentityAntigravity
    result: CandidateStabilityResultAntigravity

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("committed prefix sequence is invalid")
        if type(self.record_sha256) is not str or _SHA256.fullmatch(self.record_sha256) is None:
            raise ValueError("committed prefix record SHA256 is invalid")
        if (
            type(self.record_sha256_prefix) is not tuple
            or len(self.record_sha256_prefix) != self.sequence + 1
            or any(type(value) is not str or _SHA256.fullmatch(value) is None for value in self.record_sha256_prefix)
            or self.record_sha256_prefix[-1] != self.record_sha256
        ):
            raise ValueError("committed prefix SHA256 values are invalid")
        if type(self.context_sha256) is not str or _SHA256.fullmatch(self.context_sha256) is None:
            raise ValueError("committed prefix context SHA256 is invalid")
        if type(self.total_units) is not int or self.total_units < self.sequence + 1:
            raise ValueError("committed prefix total is invalid")
        if self.status not in {"running", "complete"}:
            raise ValueError("committed prefix status is invalid")
        if type(self.process) is not ProcessIdentityAntigravity:
            raise ValueError("committed prefix process is invalid")
        if type(self.result) is not CandidateStabilityResultAntigravity:
            raise ValueError("committed prefix result is invalid")


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _open_pinned_directory_antigravity(
    parent_fd: int | None, name: str | os.PathLike[str],
) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _CHECKPOINT_DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("checkpoint committed authority directory is invalid") from error
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or _identity_antigravity(before) != _identity_antigravity(after):
            raise ValueError("checkpoint committed authority directory changed")
        return descriptor, _identity_antigravity(after)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_pinned_directory_antigravity(
    parent_fd: int | None, name: str | os.PathLike[str], descriptor: int,
    expected: tuple[int, int, int, int, int, int, int],
) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError("checkpoint committed authority directory changed") from error
    if (
        not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(current.st_mode)
        or _identity_antigravity(opened) != expected or _identity_antigravity(current) != expected
    ):
        raise ValueError("checkpoint committed authority directory changed")


def _read_pinned_canonical_leaf_antigravity(
    directory_fd: int, name: str,
) -> tuple[Mapping[str, object], bytes]:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, _CHECKPOINT_LEAF_FLAGS_ANTIGRAVITY, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError("checkpoint committed authority leaf is invalid") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or _identity_antigravity(before) != _identity_antigravity(opened)
        ):
            raise ValueError("checkpoint committed authority leaf changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode) or after.st_nlink != 1
            or not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
            or _identity_antigravity(after) != _identity_antigravity(opened)
            or _identity_antigravity(current) != _identity_antigravity(opened)
        ):
            raise ValueError("checkpoint committed authority leaf changed")
        payload = parse_canonical_json_bytes_antigravity(encoded)
        if not isinstance(payload, Mapping) or canonical_json_bytes_antigravity(payload) != encoded:
            raise ValueError("checkpoint committed authority leaf is not canonical")
        return payload, encoded
    except ValueError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("checkpoint committed authority leaf is invalid") from error
    finally:
        os.close(descriptor)


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


_CheckpointContextLikeAntigravity = CheckpointContextAntigravity | ProfileCheckpointContextAntigravity


def _context_payload_antigravity(context: _CheckpointContextLikeAntigravity) -> dict[str, object]:
    if type(context) not in {CheckpointContextAntigravity, ProfileCheckpointContextAntigravity}:
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
        context: _CheckpointContextLikeAntigravity,
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
        self._local_owner_pid = os.getpid()

    @classmethod
    def create(
        cls,
        root: Path,
        context: _CheckpointContextLikeAntigravity,
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
        *, context_sha: str, context: _CheckpointContextLikeAntigravity,
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

    def committed_prefix_authority_antigravity(
        self, sequence: int,
    ) -> CommittedPrefixAuthorityAntigravity:
        """Return a descriptor-pinned successful prefix for Exp445 recovery.

        This deliberately re-reads every authoritative checkpoint byte through
        no-follow descriptors instead of trusting caller-supplied paths.  It
        is read-only and compares the durable state with this store's complete
        in-memory prefix before handing any value to progress recovery.
        """
        if type(sequence) is not int or sequence < 0 or sequence >= len(self._prefix):
            raise ValueError("checkpoint committed authority sequence is invalid")
        if self._status not in {"running", "complete"} or self._error_count != 0:
            raise ValueError("checkpoint committed authority status is invalid")
        expected_context = _context_payload_antigravity(self.context)
        root_fd, root_identity = _open_pinned_directory_antigravity(None, self.root)
        try:
            try:
                if tuple(sorted(os.listdir(root_fd))) != (
                    "context_antigravity.json", "records_antigravity", "state_antigravity.json",
                ):
                    raise ValueError("checkpoint committed authority inventory is invalid")
            except OSError as error:
                raise ValueError("checkpoint committed authority inventory is invalid") from error
            context_payload, context_bytes = _read_pinned_canonical_leaf_antigravity(
                root_fd, "context_antigravity.json"
            )
            if dict(context_payload) != expected_context:
                raise ValueError("checkpoint committed authority context changed")
            context_sha = _context_sha_antigravity(context_bytes)
            state, _state_bytes = _read_pinned_canonical_leaf_antigravity(
                root_fd, "state_antigravity.json"
            )
            if set(state) != _CHECKPOINT_STATE_KEYS_ANTIGRAVITY:
                raise ValueError("checkpoint committed authority state schema is invalid")
            prefix_payload = state["record_sha256_prefix"]
            if (
                state["context_sha256"] != context_sha
                or state["status"] != self._status
                or state["total_units"] != self.context.total_units
                or state["completed_units"] != sequence + 1
                or state["completed_units"] != len(self._prefix)
                or type(prefix_payload) is not list
                or tuple(prefix_payload) != tuple(self._prefix)
                or state["payload_sha256"] != _payload_hash_antigravity(tuple(self._prefix))
                or state["process"] != self.process.to_payload_antigravity()
                or state["error_count"] != 0
                or state["error"] is not None
                or state["current_unit"] is not None
                or state["current_view"] is not None
                or type(state["updated_at"]) is not str
            ):
                raise ValueError("checkpoint committed authority state changed")
            try:
                datetime.fromisoformat(state["updated_at"].replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("checkpoint committed authority timestamp is invalid") from error
            if (
                any(type(value) is not str or _SHA256.fullmatch(value) is None for value in prefix_payload)
                or (state["status"] == "complete") != (len(self._prefix) == self.context.total_units)
            ):
                raise ValueError("checkpoint committed authority state changed")
            records_fd, records_identity = _open_pinned_directory_antigravity(root_fd, "records_antigravity")
            try:
                expected_names = tuple(
                    f"{index:04d}_record_antigravity.json" for index in range(sequence + 1)
                )
                if tuple(sorted(os.listdir(records_fd))) != expected_names:
                    raise ValueError("checkpoint committed authority record inventory is invalid")
                result: CandidateStabilityResultAntigravity | None = None
                for index, expected_sha in enumerate(self._prefix):
                    record, encoded = _read_pinned_canonical_leaf_antigravity(
                        records_fd, f"{index:04d}_record_antigravity.json"
                    )
                    if _sha256_bytes_antigravity(encoded) != expected_sha:
                        raise ValueError("checkpoint committed authority record changed")
                    parsed = _validate_record_payload_antigravity(index, record, encoded)
                    if parsed is None:
                        raise ValueError("checkpoint committed authority record is not a success")
                    if index == sequence:
                        result = parsed
                _assert_pinned_directory_antigravity(root_fd, "records_antigravity", records_fd, records_identity)
                if result is None:
                    raise AssertionError("checkpoint committed authority result is unavailable")
            finally:
                os.close(records_fd)
            _assert_pinned_directory_antigravity(None, self.root, root_fd, root_identity)
            return CommittedPrefixAuthorityAntigravity(
                sequence, self._prefix[sequence], tuple(self._prefix), context_sha,
                self.context.total_units, self._status, self.process, result,
            )
        finally:
            os.close(root_fd)

    def _require_writable_antigravity(self) -> None:
        if os.getpid() != self._local_owner_pid:
            raise ValueError("checkpoint mutation requires the parent owner process")
        if self._finalization_only:
            raise ValueError("checkpoint is finalization-only")
        if self._status in _TERMINAL:
            raise ValueError("checkpoint is terminal")
        if self._status != "running":
            raise ValueError("checkpoint is interrupted and no longer writable")

    def _write_state_antigravity(self, *, error: Mapping[str, object] | None = None) -> None:
        if os.getpid() != self._local_owner_pid:
            raise ValueError("checkpoint mutation requires the parent owner process")
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
        context: _CheckpointContextLikeAntigravity,
        replacement_process: ProcessIdentityAntigravity,
        pid_probe: Callable[[ProcessIdentityAntigravity], bool] | None = None,
        *,
        allow_complete_finalization: bool = False,
        _read_only_preflight: bool = False,
    ) -> "Exp444CheckpointStoreAntigravity" | CheckpointResumeInspectionAntigravity:
        expected_context = _context_payload_antigravity(context)
        if type(replacement_process) is not ProcessIdentityAntigravity:
            raise ValueError("checkpoint replacement process identity is invalid")
        if type(allow_complete_finalization) is not bool:
            raise ValueError("checkpoint finalization flag is invalid")
        if type(_read_only_preflight) is not bool:
            raise ValueError("checkpoint preflight flag is invalid")
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
        if stored_process.pid == replacement_process.pid:
            raise ValueError("replacement process numeric PID must be distinct")
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
        if len(prefix_payload) != completed or any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in prefix_payload
        ):
            raise ValueError("checkpoint prefix is invalid")
        if state_payload["payload_sha256"] != _payload_hash_antigravity(tuple(prefix_payload)):
            raise ValueError("checkpoint payload hash changed")
        if state_payload["current_unit"] is not None or state_payload["current_view"] is not None:
            raise ValueError("checkpoint current fields are invalid")
        if type(state_payload["updated_at"]) is not str:
            raise ValueError("checkpoint update time is invalid")
        try:
            datetime.fromisoformat(state_payload["updated_at"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("checkpoint update time is invalid") from error
        state_error = state_payload["error"]
        if status == "error":
            if error_count != 1 or not isinstance(state_error, Mapping) or set(state_error) != {"type", "message"}:
                raise ValueError("checkpoint terminal error state is invalid")
            if any(type(state_error[name]) is not str or not state_error[name] for name in ("type", "message")):
                raise ValueError("checkpoint terminal error state is invalid")
        elif error_count != 0 or state_error is not None:
            raise ValueError("checkpoint error state is invalid")
        if status == "complete" and completed != total:
            raise ValueError("checkpoint complete count is invalid")
        if status != "complete" and completed == total:
            raise ValueError("checkpoint nonterminal count is invalid")
        prefix, orphan, orphan_result = _scan_records_antigravity(root, tuple(prefix_payload), total)
        if status in _TERMINAL and status != "complete":
            raise ValueError("checkpoint terminal error is not resumable")
        if status == "complete":
            if not allow_complete_finalization or completed != total or orphan is not None or tuple(prefix) != tuple(prefix_payload):
                raise ValueError("checkpoint is complete")
            if _read_only_preflight:
                return CheckpointResumeInspectionAntigravity(
                    status, completed, completed, tuple(prefix_payload), None,
                    None, None, None,
                )
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
        if _read_only_preflight:
            return CheckpointResumeInspectionAntigravity(
                status,
                completed,
                len(prefix),
                tuple(prefix),
                orphan,
                orphan_result,
                None if orphan_error is None else str(orphan_error["type"]),
                None if orphan_error is None else str(orphan_error["message"]),
            )
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

    @classmethod
    def inspect_interrupted_antigravity(
        cls,
        root: Path,
        context: _CheckpointContextLikeAntigravity,
        replacement_process: ProcessIdentityAntigravity,
        pid_probe: Callable[[ProcessIdentityAntigravity], bool] | None = None,
    ) -> CheckpointResumeInspectionAntigravity:
        """Validate the complete resumable authority without changing any bytes."""
        result = cls.open_interrupted(
            root,
            context,
            replacement_process,
            pid_probe=pid_probe,
            _read_only_preflight=True,
        )
        if not isinstance(result, CheckpointResumeInspectionAntigravity):
            raise AssertionError("checkpoint read-only preflight returned a writable store")
        return result

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
    "CheckpointResumeInspectionAntigravity", "CommittedPrefixAuthorityAntigravity", "Exp444CheckpointStoreAntigravity",
    "ProcessIdentityAntigravity",
    "current_process_identity_antigravity",
]
