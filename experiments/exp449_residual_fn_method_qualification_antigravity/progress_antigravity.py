"""Create-only candidate leaves and verified contiguous-prefix helpers for Exp449."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity


_ATTEMPT_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "progress_antigravity.json", "progress_antigravity.jsonl",
    "progress_antigravity.txt", "pid_antigravity.json", "candidates_antigravity",
)


def write_failure_audit_antigravity(destination: Path, primary_error: BaseException, cleanup_error: BaseException | None, *, exact_next_sequence: int | None = None) -> None:
    """Write the independent fallback once, retaining primary and cleanup channel failures."""
    if not isinstance(destination, Path) or not isinstance(primary_error, BaseException) or (cleanup_error is not None and not isinstance(cleanup_error, BaseException)) or (exact_next_sequence is not None and (type(exact_next_sequence) is not int or exact_next_sequence < 0)):
        raise ValueError("failure audit inputs are invalid")
    payload = {
        "schema": "exp449_failure_audit", "primary_error": f"{type(primary_error).__name__}: {primary_error}",
        "cleanup_error": None if cleanup_error is None else f"{type(cleanup_error).__name__}: {cleanup_error}",
        "exact_next_sequence": exact_next_sequence,
    }
    write_create_only_antigravity(destination, canonical_json_bytes_antigravity(payload))


def initialise_attempt_antigravity(root: Path, context: object) -> None:
    """Create the exact non-terminal attempt namespace without an authority manifest."""
    if not isinstance(root, Path) or (root.exists() and any(root.iterdir())):
        raise ValueError("attempt root must be new or empty")
    root.mkdir(mode=0o700, exist_ok=True)
    (root / "candidates_antigravity").mkdir(mode=0o700)
    write_create_only_antigravity(root / "context_antigravity.json", canonical_json_bytes_antigravity(context))
    write_create_only_antigravity(root / "progress_antigravity.json", canonical_json_bytes_antigravity({"dataset_unit_completed": 0, "next_sequence": 0, "terminal": False}))
    write_create_only_antigravity(root / "progress_antigravity.jsonl", b"")
    write_create_only_antigravity(root / "progress_antigravity.txt", b"")
    write_create_only_antigravity(root / "pid_antigravity.json", canonical_json_bytes_antigravity({"active": True}))


def _write_progress_json_antigravity(progress: Path, payload: bytes, fault_hook, sequence: int) -> None:
    if fault_hook is not None: fault_hook("progress_json", sequence)
    fd = os.open(progress, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.write(fd, payload); os.fsync(fd)
    finally: os.close(fd)


def _append_progress_jsonl_antigravity(path: Path, payload: bytes, fault_hook, sequence: int) -> None:
    if fault_hook is not None: fault_hook("progress_jsonl", sequence)
    with path.open("ab") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def _append_progress_text_antigravity(path: Path, payload: bytes, fault_hook, sequence: int) -> None:
    if fault_hook is not None: fault_hook("progress_text", sequence)
    with path.open("ab") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def record_verified_progress_antigravity(root: Path, inventory: tuple[str, ...], completed: int, *, fault_hook=None) -> None:
    """Advance mutable observability only after the independently verified prefix reaches it."""
    if type(completed) is not int or completed < 0 or completed > len(inventory):
        raise ValueError("progress count is invalid")
    if verified_prefix_antigravity(root / "candidates_antigravity", inventory) != completed:
        raise ValueError("progress cannot outrun the verified candidate prefix")
    progress = root / "progress_antigravity.json"
    current = strict_json_bytes_antigravity(progress.read_bytes())
    if type(current) is not dict or current.get("dataset_unit_completed") is None or current["dataset_unit_completed"] > completed:
        raise ValueError("progress state is invalid")
    payload = canonical_json_bytes_antigravity({"dataset_unit_completed": completed, "next_sequence": completed, "terminal": completed == len(inventory)})
    _write_progress_json_antigravity(progress, payload, fault_hook, completed - 1)
    _append_progress_jsonl_antigravity(root / "progress_antigravity.jsonl", canonical_json_bytes_antigravity({"dataset_unit_completed": completed}), fault_hook, completed - 1)
    _append_progress_text_antigravity(root / "progress_antigravity.txt", f"{completed}\n".encode(), fault_hook, completed - 1)


def commit_candidate_leaf_antigravity(directory: Path, name: str, value: object, *, fault_hook=None, sequence: int = 0) -> str:
    """Commit a candidate only after file and parent fsync plus canonical no-follow reread."""
    if not isinstance(directory, Path) or type(name) is not str or "/" in name or not name.endswith(".json"):
        raise ValueError("candidate destination is invalid")
    directory.mkdir(mode=0o700, exist_ok=True)
    payload = canonical_json_bytes_antigravity(value)
    destination = directory / name
    parent_fd = -1
    created_identity: tuple[int, int] | None = None
    try:
        parent_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        if fault_hook is not None: fault_hook("cache_create", sequence)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("candidate leaf is unsafe")
            created_identity = (info.st_dev, info.st_ino)
            view = memoryview(payload)
            while view: view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        if fault_hook is not None: fault_hook("cache_fsync", sequence)
        os.fsync(parent_fd)
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            if fault_hook is not None: fault_hook("cache_reread", sequence)
            reread = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
            if fault_hook is not None: fault_hook("cache_sha_mismatch", sequence)
            if reread != payload or canonical_json_bytes_antigravity(strict_json_bytes_antigravity(reread)) != reread: raise ValueError("candidate leaf reread is not canonical")
            opened = os.fstat(fd)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != created_identity or (current.st_dev, current.st_ino) != created_identity: raise ValueError("candidate leaf identity changed")
            return hashlib.sha256(reread).hexdigest()
        finally: os.close(fd)
    except BaseException:
        if created_identity is not None and parent_fd >= 0:
            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(name, dir_fd=parent_fd); os.fsync(parent_fd)
            except OSError: pass
        raise
    finally:
        if parent_fd >= 0: os.close(parent_fd)


def verified_prefix_antigravity(directory: Path, inventory: tuple[str, ...]) -> int:
    """Derive the completed count exclusively from present canonical candidate leaves."""
    if not isinstance(directory, Path) or type(inventory) is not tuple:
        raise ValueError("prefix inputs are invalid")
    if directory.exists() and set(os.listdir(directory)) - set(inventory):
        raise ValueError("candidate directory has unexpected leaf")
    completed = 0
    for name in inventory:
        path = directory / name
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            break
        try:
            payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
            if canonical_json_bytes_antigravity(strict_json_bytes_antigravity(payload)) != payload:
                break
        except ValueError:
            break
        finally:
            os.close(fd)
        completed += 1
    return completed
