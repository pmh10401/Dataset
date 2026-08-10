"""Descriptor-pinned, read-only authentication of Exp444's terminal attempt."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from .profile_antigravity import (
    EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY,
    EXP444_CONTEXT_ANTIGRAVITY,
    EXP444_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY,
    EXP444_RECORD_SHA256_PREFIX_ANTIGRAVITY,
    EXP444_REVIEWED_COMMIT_ANTIGRAVITY,
    EXP444_ROOT_ANTIGRAVITY,
    EXP444_TERMINAL_SHA256_ANTIGRAVITY,
    SOURCE_DATABASE_PATH_ANTIGRAVITY,
)


_EXP444_ROOT_ANTIGRAVITY = EXP444_ROOT_ANTIGRAVITY
_SOURCE_DATABASE_PATH_ANTIGRAVITY = SOURCE_DATABASE_PATH_ANTIGRAVITY
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_ATTEMPT_FILES_ANTIGRAVITY = (
    "progress_antigravity.json",
    "pid_antigravity.json",
    "events_antigravity.jsonl",
    "text_antigravity.log",
)
_CHECKPOINT_FILES_ANTIGRAVITY = ("context_antigravity.json", "state_antigravity.json")
_RECORD_NAMES_ANTIGRAVITY = tuple(f"{index:04d}_record_antigravity.json" for index in range(30))


@dataclass(frozen=True)
class Exp444PredecessorAuthorityAntigravity:
    """Closed terminal authority; it deliberately contains no writable handle."""

    reviewed_commit: str
    attempt_root: Path
    record_sha256_prefix: tuple[str, ...]
    record_prefix_payload_sha256: str


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_nlink, info.st_size


def _checked_directory_antigravity(parent_fd: int | None, name: str | os.PathLike[str]) -> tuple[int, tuple[int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("Exp444 predecessor directory is absent or unsafe") from error
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or _identity_antigravity(before) != _identity_antigravity(after):
            raise ValueError("Exp444 predecessor directory changed during verification")
        return descriptor, _identity_antigravity(after)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_antigravity(parent_fd: int | None, name: str | os.PathLike[str], expected: tuple[int, int, int, int, int]) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp444 predecessor directory disappeared during verification") from error
    if not stat.S_ISDIR(current.st_mode) or _identity_antigravity(current) != expected:
        raise ValueError("Exp444 predecessor directory changed during verification")


def _assert_exact_names_antigravity(directory_fd: int, expected: tuple[str, ...]) -> None:
    try:
        actual = tuple(sorted(os.listdir(directory_fd)))
    except OSError as error:
        raise ValueError("Exp444 predecessor inventory cannot be read") from error
    if actual != tuple(sorted(expected)):
        raise ValueError("Exp444 predecessor inventory is not exact")


def _read_leaf_antigravity(directory_fd: int, name: str, expected_sha256: str) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError("Exp444 predecessor terminal leaf is absent or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _identity_antigravity(before) != _identity_antigravity(opened):
            raise ValueError("Exp444 predecessor terminal leaf is not a single regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or _identity_antigravity(current) != _identity_antigravity(opened):
            raise ValueError("Exp444 predecessor terminal leaf changed during verification")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Exp444 predecessor terminal SHA256 changed")
        return payload
    finally:
        os.close(descriptor)


def _strict_json_antigravity(payload: bytes, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Exp444 predecessor {name} contains a nonfinite constant: {value}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(decoded, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Exp444 predecessor {name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Exp444 predecessor {name} is not a JSON object")
    return value


def _canonical_prefix_payload_sha256_antigravity(prefix: tuple[str, ...]) -> str:
    encoded = json.dumps(list(prefix), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    return hashlib.sha256(encoded).hexdigest()


def _assert_source_database_antigravity() -> None:
    try:
        metadata = os.stat(_SOURCE_DATABASE_PATH_ANTIGRAVITY, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp444 source database is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("Exp444 source database must have exactly one link")


def _assert_terminal_semantics_antigravity(
    context_bytes: bytes,
    state_bytes: bytes,
    progress_bytes: bytes,
    pid_bytes: bytes,
    record_prefix: tuple[str, ...],
) -> None:
    context = _strict_json_antigravity(context_bytes, "context")
    state = _strict_json_antigravity(state_bytes, "state")
    progress = _strict_json_antigravity(progress_bytes, "progress")
    pid = _strict_json_antigravity(pid_bytes, "PID")
    if context != dict(EXP444_CONTEXT_ANTIGRAVITY):
        raise ValueError("Exp444 predecessor context binding changed")
    if state.get("context_sha256") != EXP444_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/context_antigravity.json"]:
        raise ValueError("Exp444 predecessor state context binding changed")
    if (
        state.get("status"), state.get("completed_units"), state.get("total_units"),
        state.get("error_count"), state.get("error"), state.get("current_unit"), state.get("current_view"),
    ) != ("complete", 30, 30, 0, None, None, None):
        raise ValueError("Exp444 predecessor state is not exact complete 30/30")
    if tuple(state.get("record_sha256_prefix", ())) != record_prefix:
        raise ValueError("Exp444 predecessor record prefix changed")
    if state.get("payload_sha256") != EXP444_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY:
        raise ValueError("Exp444 predecessor aggregate prefix payload changed")
    if (
        progress.get("stage"), progress.get("active"), progress.get("completed"), progress.get("total"),
        progress.get("error"), progress.get("backend"),
    ) != ("complete", False, 30, 30, 0, "cpu"):
        raise ValueError("Exp444 predecessor progress is not complete and inactive")
    if pid.get("active") is not False or type(pid.get("pid")) is not int or pid.get("pid") <= 0:
        raise ValueError("Exp444 predecessor PID state is invalid")
    try:
        os.kill(pid["pid"], 0)
    except ProcessLookupError:
        return
    except PermissionError:
        raise ValueError("Exp444 predecessor PID cannot be proven inactive") from None
    except OSError as error:
        if error.errno == errno.ESRCH:
            return
        raise ValueError("Exp444 predecessor PID cannot be proven inactive") from error
    raise ValueError("Exp444 predecessor PID is still running")


def verify_exp444_predecessor_antigravity() -> Exp444PredecessorAuthorityAntigravity:
    """Authenticate only the fixed Exp444 terminal bytes through pinned descriptors."""
    _assert_source_database_antigravity()
    root_fd, root_identity = _checked_directory_antigravity(None, _EXP444_ROOT_ANTIGRAVITY)
    attempts_fd = checkpoint_fd = records_fd = -1
    try:
        _assert_exact_names_antigravity(root_fd, (EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY[0],))
        attempts_fd, attempts_identity = _checked_directory_antigravity(root_fd, EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY[0])
        _assert_exact_names_antigravity(attempts_fd, (EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY[1],))
        attempt_name = EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY[1]
        attempt_fd, attempt_identity = _checked_directory_antigravity(attempts_fd, attempt_name)
        try:
            _assert_exact_names_antigravity(attempt_fd, ("checkpoint_antigravity", *_ATTEMPT_FILES_ANTIGRAVITY))
            checkpoint_fd, checkpoint_identity = _checked_directory_antigravity(attempt_fd, "checkpoint_antigravity")
            _assert_exact_names_antigravity(checkpoint_fd, ("records_antigravity", *_CHECKPOINT_FILES_ANTIGRAVITY))
            records_fd, records_identity = _checked_directory_antigravity(checkpoint_fd, "records_antigravity")
            _assert_exact_names_antigravity(records_fd, _RECORD_NAMES_ANTIGRAVITY)
            context = _read_leaf_antigravity(checkpoint_fd, "context_antigravity.json", EXP444_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/context_antigravity.json"])
            state = _read_leaf_antigravity(checkpoint_fd, "state_antigravity.json", EXP444_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/state_antigravity.json"])
            progress = _read_leaf_antigravity(attempt_fd, "progress_antigravity.json", EXP444_TERMINAL_SHA256_ANTIGRAVITY["progress_antigravity.json"])
            pid = _read_leaf_antigravity(attempt_fd, "pid_antigravity.json", EXP444_TERMINAL_SHA256_ANTIGRAVITY["pid_antigravity.json"])
            _read_leaf_antigravity(attempt_fd, "events_antigravity.jsonl", EXP444_TERMINAL_SHA256_ANTIGRAVITY["events_antigravity.jsonl"])
            _read_leaf_antigravity(attempt_fd, "text_antigravity.log", EXP444_TERMINAL_SHA256_ANTIGRAVITY["text_antigravity.log"])
            for sequence, name in enumerate(_RECORD_NAMES_ANTIGRAVITY):
                payload = _read_leaf_antigravity(records_fd, name, EXP444_RECORD_SHA256_PREFIX_ANTIGRAVITY[sequence])
                wrapper = _strict_json_antigravity(payload, f"record {sequence}")
                if set(wrapper) != {"sequence", "result"} or wrapper.get("sequence") != sequence or not isinstance(wrapper.get("result"), dict):
                    raise ValueError("Exp444 predecessor record wrapper changed")
            prefix = tuple(EXP444_RECORD_SHA256_PREFIX_ANTIGRAVITY)
            if _canonical_prefix_payload_sha256_antigravity(prefix) != EXP444_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY:
                raise ValueError("Exp444 predecessor aggregate prefix authority is invalid")
            _assert_terminal_semantics_antigravity(context, state, progress, pid, prefix)
            _assert_directory_antigravity(checkpoint_fd, "records_antigravity", records_identity)
            _assert_directory_antigravity(attempt_fd, "checkpoint_antigravity", checkpoint_identity)
            _assert_directory_antigravity(attempts_fd, attempt_name, attempt_identity)
            _assert_directory_antigravity(root_fd, EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY[0], attempts_identity)
            _assert_directory_antigravity(None, _EXP444_ROOT_ANTIGRAVITY, root_identity)
            return Exp444PredecessorAuthorityAntigravity(
                reviewed_commit=EXP444_REVIEWED_COMMIT_ANTIGRAVITY,
                attempt_root=_EXP444_ROOT_ANTIGRAVITY.joinpath(*EXP444_ATTEMPT_SEGMENTS_ANTIGRAVITY),
                record_sha256_prefix=prefix,
                record_prefix_payload_sha256=EXP444_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY,
            )
        finally:
            os.close(attempt_fd)
    finally:
        for descriptor in (records_fd, checkpoint_fd, attempts_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


__all__ = ["Exp444PredecessorAuthorityAntigravity", "verify_exp444_predecessor_antigravity"]
