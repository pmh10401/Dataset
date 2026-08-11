"""Descriptor-pinned, read-only authentication of Exp445's failed publication."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
import stat
from types import MappingProxyType
from typing import Any, Mapping

from .profile_antigravity import (
    EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY,
    EXP445_CONTEXT_ANTIGRAVITY,
    EXP445_PUBLICATION_FAILURE_ANTIGRAVITY,
    EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY,
    EXP445_RECORD_SHA256_PREFIX_ANTIGRAVITY,
    EXP445_REVIEWED_COMMIT_ANTIGRAVITY,
    EXP445_ROOT_ANTIGRAVITY,
    EXP445_TERMINAL_SHA256_ANTIGRAVITY,
    SOURCE_DATABASE_PATH_ANTIGRAVITY,
)


_EXP445_ROOT_ANTIGRAVITY = EXP445_ROOT_ANTIGRAVITY
_SOURCE_DATABASE_PATH_ANTIGRAVITY = SOURCE_DATABASE_PATH_ANTIGRAVITY
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_RECORD_NAMES_ANTIGRAVITY = tuple(f"{index:04d}_record_antigravity.json" for index in range(30))
_ATTEMPT_NAMES_ANTIGRAVITY = (
    "checkpoint_antigravity", "events_antigravity.jsonl", "pid_antigravity.json",
    "progress_antigravity.json", "publication_failure_antigravity.json", "text_antigravity.log",
)
_CHECKPOINT_NAMES_ANTIGRAVITY = ("context_antigravity.json", "records_antigravity", "state_antigravity.json")


@dataclass(frozen=True)
class Exp445PredecessorAuthorityAntigravity:
    """Values-only authority; it exposes neither paths nor live file descriptors."""

    reviewed_commit: str
    record_sha256_prefix: tuple[str, ...]
    record_prefix_payload_sha256: str
    terminal_sha256: Mapping[str, str]


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _checked_directory_antigravity(parent_fd: int | None, name: str | os.PathLike[str]) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("Exp445 predecessor directory is absent or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode) or _identity_antigravity(before) != _identity_antigravity(opened):
            raise ValueError("Exp445 predecessor directory changed during verification")
        return descriptor, _identity_antigravity(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _assert_directory_antigravity(parent_fd: int | None, name: str | os.PathLike[str], descriptor: int, expected: tuple[int, int, int, int, int, int, int]) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp445 predecessor directory disappeared during verification") from error
    if any(not stat.S_ISDIR(item.st_mode) or _identity_antigravity(item) != expected for item in (opened, current)):
        raise ValueError("Exp445 predecessor directory changed during verification")


def _assert_exact_names_antigravity(directory_fd: int, expected: tuple[str, ...]) -> None:
    try:
        actual = tuple(sorted(os.listdir(directory_fd)))
    except OSError as error:
        raise ValueError("Exp445 predecessor inventory cannot be read") from error
    if actual != tuple(sorted(expected)):
        raise ValueError("Exp445 predecessor inventory is not exact")


def _read_leaf_antigravity(directory_fd: int, name: str, expected_sha256: str) -> bytes:
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError("Exp445 predecessor leaf is absent or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _identity_antigravity(before) != _identity_antigravity(opened):
            raise ValueError("Exp445 predecessor leaf is not a single regular file")
        payload = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
        after = os.fstat(descriptor)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if any(
            not stat.S_ISREG(item.st_mode) or item.st_nlink != 1 or _identity_antigravity(item) != _identity_antigravity(opened)
            for item in (after, current)
        ):
            raise ValueError("Exp445 predecessor leaf changed during verification")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Exp445 predecessor leaf SHA256 changed")
        return payload
    finally:
        os.close(descriptor)


def _strict_json_antigravity(payload: bytes, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Exp445 predecessor {name} contains nonfinite constant {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Exp445 predecessor {name} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Exp445 predecessor {name} is not a JSON object")
    return value


def _canonical_prefix_payload_sha256_antigravity(prefix: tuple[str, ...]) -> str:
    payload = json.dumps(list(prefix), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _assert_source_database_antigravity() -> None:
    try:
        metadata = os.stat(_SOURCE_DATABASE_PATH_ANTIGRAVITY, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp445 source database is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("Exp445 source database must be a single-link regular file")


def _assert_terminal_semantics_antigravity(context_bytes: bytes, state_bytes: bytes, progress_bytes: bytes, pid_bytes: bytes, failure_bytes: bytes, record_prefix: tuple[str, ...]) -> None:
    context = _strict_json_antigravity(context_bytes, "context")
    state = _strict_json_antigravity(state_bytes, "state")
    progress = _strict_json_antigravity(progress_bytes, "progress")
    pid = _strict_json_antigravity(pid_bytes, "PID")
    failure = _strict_json_antigravity(failure_bytes, "publication failure")
    if context != dict(EXP445_CONTEXT_ANTIGRAVITY):
        raise ValueError("Exp445 predecessor context binding changed")
    if (
        state.get("context_sha256"), state.get("status"), state.get("completed_units"), state.get("total_units"),
        state.get("error_count"), state.get("error"), state.get("current_unit"), state.get("current_view"),
    ) != (EXP445_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/context_antigravity.json"], "complete", 30, 30, 0, None, None, None):
        raise ValueError("Exp445 predecessor state is not exact complete 30/30")
    if tuple(state.get("record_sha256_prefix", ())) != record_prefix or state.get("payload_sha256") != EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY:
        raise ValueError("Exp445 predecessor record authority changed")
    if (
        progress.get("stage"), progress.get("active"), progress.get("completed"), progress.get("total"), progress.get("error"),
        progress.get("backend"), progress.get("abstention"), progress.get("elapsed_seconds"),
    ) != ("complete", False, 30, 30, 0, "cpu", 4, 913.3950564998668):
        raise ValueError("Exp445 predecessor progress is not exact complete/inactive")
    if pid.get("active") is not False or type(pid.get("pid")) is not int or pid["pid"] <= 0:
        raise ValueError("Exp445 predecessor PID state is invalid")
    if state.get("process", {}).get("pid") != pid["pid"] or progress.get("pid") != pid["pid"]:
        raise ValueError("Exp445 predecessor PID binding changed")
    if failure != {"error": dict(EXP445_PUBLICATION_FAILURE_ANTIGRAVITY["error"]), "schema": "exp445.publication-failure.v1", "stage": "publication"}:
        raise ValueError("Exp445 predecessor publication failure marker changed")
    try:
        os.kill(pid["pid"], 0)
    except ProcessLookupError:
        return
    except PermissionError:
        raise ValueError("Exp445 predecessor PID cannot be proven inactive") from None
    except OSError as error:
        if error.errno == errno.ESRCH:
            return
        raise ValueError("Exp445 predecessor PID cannot be proven inactive") from error
    raise ValueError("Exp445 predecessor PID is still running")


def verify_exp445_predecessor_antigravity() -> Exp445PredecessorAuthorityAntigravity:
    """Authenticate immutable Exp445 evidence by descriptors without mutating it."""
    _assert_source_database_antigravity()
    root_fd, root_identity = _checked_directory_antigravity(None, _EXP445_ROOT_ANTIGRAVITY)
    attempts_fd = attempt_fd = checkpoint_fd = records_fd = -1
    try:
        _assert_exact_names_antigravity(root_fd, (EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[0],))
        attempts_fd, attempts_identity = _checked_directory_antigravity(root_fd, EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[0])
        _assert_exact_names_antigravity(attempts_fd, (EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[1],))
        attempt_fd, attempt_identity = _checked_directory_antigravity(attempts_fd, EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[1])
        _assert_exact_names_antigravity(attempt_fd, _ATTEMPT_NAMES_ANTIGRAVITY)
        checkpoint_fd, checkpoint_identity = _checked_directory_antigravity(attempt_fd, "checkpoint_antigravity")
        _assert_exact_names_antigravity(checkpoint_fd, _CHECKPOINT_NAMES_ANTIGRAVITY)
        records_fd, records_identity = _checked_directory_antigravity(checkpoint_fd, "records_antigravity")
        _assert_exact_names_antigravity(records_fd, _RECORD_NAMES_ANTIGRAVITY)
        context = _read_leaf_antigravity(checkpoint_fd, "context_antigravity.json", EXP445_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/context_antigravity.json"])
        state = _read_leaf_antigravity(checkpoint_fd, "state_antigravity.json", EXP445_TERMINAL_SHA256_ANTIGRAVITY["checkpoint/state_antigravity.json"])
        progress = _read_leaf_antigravity(attempt_fd, "progress_antigravity.json", EXP445_TERMINAL_SHA256_ANTIGRAVITY["progress_antigravity.json"])
        pid = _read_leaf_antigravity(attempt_fd, "pid_antigravity.json", EXP445_TERMINAL_SHA256_ANTIGRAVITY["pid_antigravity.json"])
        failure = _read_leaf_antigravity(attempt_fd, "publication_failure_antigravity.json", EXP445_TERMINAL_SHA256_ANTIGRAVITY["publication_failure_antigravity.json"])
        _read_leaf_antigravity(attempt_fd, "events_antigravity.jsonl", EXP445_TERMINAL_SHA256_ANTIGRAVITY["events_antigravity.jsonl"])
        _read_leaf_antigravity(attempt_fd, "text_antigravity.log", EXP445_TERMINAL_SHA256_ANTIGRAVITY["text_antigravity.log"])
        for sequence, name in enumerate(_RECORD_NAMES_ANTIGRAVITY):
            wrapper = _strict_json_antigravity(_read_leaf_antigravity(records_fd, name, EXP445_RECORD_SHA256_PREFIX_ANTIGRAVITY[sequence]), f"record {sequence}")
            if set(wrapper) != {"sequence", "result"} or wrapper.get("sequence") != sequence or not isinstance(wrapper.get("result"), dict):
                raise ValueError("Exp445 predecessor record wrapper changed")
        prefix = tuple(EXP445_RECORD_SHA256_PREFIX_ANTIGRAVITY)
        if _canonical_prefix_payload_sha256_antigravity(prefix) != EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY:
            raise ValueError("Exp445 predecessor aggregate authority is invalid")
        _assert_terminal_semantics_antigravity(context, state, progress, pid, failure, prefix)
        _assert_directory_antigravity(checkpoint_fd, "records_antigravity", records_fd, records_identity)
        _assert_directory_antigravity(attempt_fd, "checkpoint_antigravity", checkpoint_fd, checkpoint_identity)
        _assert_directory_antigravity(attempts_fd, EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[1], attempt_fd, attempt_identity)
        _assert_directory_antigravity(root_fd, EXP445_ATTEMPT_SEGMENTS_ANTIGRAVITY[0], attempts_fd, attempts_identity)
        _assert_directory_antigravity(None, _EXP445_ROOT_ANTIGRAVITY, root_fd, root_identity)
        _assert_source_database_antigravity()
        return Exp445PredecessorAuthorityAntigravity(
            reviewed_commit=EXP445_REVIEWED_COMMIT_ANTIGRAVITY,
            record_sha256_prefix=prefix,
            record_prefix_payload_sha256=EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY,
            terminal_sha256=MappingProxyType(dict(EXP445_TERMINAL_SHA256_ANTIGRAVITY)),
        )
    finally:
        for descriptor in (records_fd, checkpoint_fd, attempt_fd, attempts_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)


__all__ = ["Exp445PredecessorAuthorityAntigravity", "verify_exp445_predecessor_antigravity"]
