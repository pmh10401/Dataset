"""Create-only candidate leaves and verified contiguous-prefix helpers for Exp449."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity


_ATTEMPT_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "progress_antigravity.json", "progress_antigravity.jsonl",
    "progress_antigravity.txt", "pid_antigravity.json", "candidates_antigravity",
)


@dataclass(frozen=True)
class CandidateCommitReceiptAntigravity:
    """A parent-held proof that one exact candidate create returned successfully."""
    sequence: int
    name: str
    sha256: str
    identity: tuple[int, int]


def write_failure_audit_antigravity(destination: Path, primary_error: BaseException, cleanup_error: BaseException | None, *, exact_next_sequence: int | None = None) -> None:
    """Write the independent fallback once, retaining primary and cleanup channel failures."""
    if not isinstance(destination, Path) or not isinstance(primary_error, BaseException) or (cleanup_error is not None and not isinstance(cleanup_error, BaseException)) or (exact_next_sequence is not None and (type(exact_next_sequence) is not int or exact_next_sequence < 0)):
        raise ValueError("failure audit inputs are invalid")
    payload = {
        "schema": "exp449_failure_audit", "primary_error": f"{type(primary_error).__name__}: {primary_error}",
        "cleanup_error": None if cleanup_error is None else f"{type(cleanup_error).__name__}: {cleanup_error}",
        "primary_error_notes": list(getattr(primary_error, "__notes__", ())),
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


def record_verified_progress_antigravity(root: Path, inventory: tuple[str, ...], receipts: tuple[CandidateCommitReceiptAntigravity, ...], completed: int, *, fault_hook=None) -> None:
    """Advance mutable observability only after the independently verified prefix reaches it."""
    if type(completed) is not int or completed < 0 or completed > len(inventory):
        raise ValueError("progress count is invalid")
    if verified_prefix_antigravity(root / "candidates_antigravity", inventory, receipts) != completed:
        raise ValueError("progress cannot outrun the verified candidate prefix")
    progress = root / "progress_antigravity.json"
    current = strict_json_bytes_antigravity(progress.read_bytes())
    if type(current) is not dict or current.get("dataset_unit_completed") is None or current["dataset_unit_completed"] > completed:
        raise ValueError("progress state is invalid")
    payload = canonical_json_bytes_antigravity({"dataset_unit_completed": completed, "next_sequence": completed, "terminal": completed == len(inventory)})
    _write_progress_json_antigravity(progress, payload, fault_hook, completed - 1)
    _append_progress_jsonl_antigravity(root / "progress_antigravity.jsonl", canonical_json_bytes_antigravity({"dataset_unit_completed": completed}), fault_hook, completed - 1)
    _append_progress_text_antigravity(root / "progress_antigravity.txt", f"{completed}\n".encode(), fault_hook, completed - 1)


def _syscall_fault_antigravity(hook, operation: str, sequence: int) -> None:
    if hook is not None: hook(operation, sequence)


def _cleanup_created_leaf_antigravity(parent_fd: int, name: str, created_fd: int, created_identity: tuple[int, int] | None, primary_error: BaseException) -> None:
    """Remove a failed create only when the opened descriptor proves its exact inode."""
    identity = created_identity
    if identity is None:
        try:
            opened = os.fstat(created_fd)
            identity = (opened.st_dev, opened.st_ino)
        except BaseException:
            primary_error.add_note("candidate_cleanup_error=OSError: unable to prove created inode after fstat failure")
            return
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == identity:
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    except BaseException as cleanup_error:
        primary_error.add_note(f"candidate_cleanup_error={type(cleanup_error).__name__}: {cleanup_error}")


def commit_candidate_leaf_antigravity(directory: Path, name: str, value: object, *, fault_hook=None, sequence: int = 0, fault_prefix: str = "", return_created_identity: bool = False, return_commit_receipt: bool = False) -> str | tuple[str, tuple[int, int]] | CandidateCommitReceiptAntigravity:
    """Commit a candidate only after file and parent fsync plus canonical no-follow reread."""
    if not isinstance(directory, Path) or type(name) is not str or "/" in name or not name.endswith(".json"):
        raise ValueError("candidate destination is invalid")
    if type(fault_prefix) is not str:
        raise ValueError("candidate fault prefix is invalid")
    if type(return_created_identity) is not bool or type(return_commit_receipt) is not bool or (return_created_identity and return_commit_receipt):
        raise ValueError("candidate identity return option is invalid")
    directory.mkdir(mode=0o700, exist_ok=True)
    payload = canonical_json_bytes_antigravity(value)
    destination = directory / name
    parent_fd = -1
    fd = -1
    created_identity: tuple[int, int] | None = None
    try:
        _syscall_fault_antigravity(fault_hook, fault_prefix + "open_parent", sequence)
        parent_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        _syscall_fault_antigravity(fault_hook, fault_prefix + "cache_create", sequence); _syscall_fault_antigravity(fault_hook, fault_prefix + "open_create", sequence)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        try:
            # The descriptor is the only creation authority.  A pathname can be
            # replaced between O_EXCL and stat, so do not use it to identify this
            # transaction's inode.
            info = os.fstat(fd)
            created_identity = (info.st_dev, info.st_ino)
            if (info.st_dev, info.st_ino) != created_identity:
                raise ValueError("candidate leaf identity changed during creation")
            _syscall_fault_antigravity(fault_hook, fault_prefix + "fstat_create", sequence)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("candidate leaf is unsafe")
            view = memoryview(payload)
            while view:
                _syscall_fault_antigravity(fault_hook, fault_prefix + "write", sequence); view = view[os.write(fd, view):]
            _syscall_fault_antigravity(fault_hook, fault_prefix + "fsync_leaf", sequence)
            os.fsync(fd)
        finally:
            if fd >= 0:
                os.close(fd)
                fd = -1
        _syscall_fault_antigravity(fault_hook, fault_prefix + "cache_fsync", sequence); _syscall_fault_antigravity(fault_hook, fault_prefix + "fsync_parent", sequence)
        os.fsync(parent_fd)
        _syscall_fault_antigravity(fault_hook, fault_prefix + "open_read", sequence); fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            _syscall_fault_antigravity(fault_hook, fault_prefix + "cache_reread", sequence)
            chunks=[]
            while True:
                _syscall_fault_antigravity(fault_hook, fault_prefix + "read", sequence); chunk=os.read(fd, 1 << 20)
                if not chunk: break
                chunks.append(chunk)
            reread = b"".join(chunks)
            if fault_hook is not None: fault_hook(fault_prefix + "cache_sha_mismatch", sequence)
            if reread != payload or canonical_json_bytes_antigravity(strict_json_bytes_antigravity(reread)) != reread: raise ValueError("candidate leaf reread is not canonical")
            _syscall_fault_antigravity(fault_hook, fault_prefix + "fstat_read", sequence); opened = os.fstat(fd)
            _syscall_fault_antigravity(fault_hook, fault_prefix + "stat_name", sequence)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != created_identity or (current.st_dev, current.st_ino) != created_identity: raise ValueError("candidate leaf identity changed")
            digest = hashlib.sha256(reread).hexdigest()
            if return_commit_receipt:
                return CandidateCommitReceiptAntigravity(sequence, name, digest, created_identity)
            return (digest, created_identity) if return_created_identity else digest
        finally:
            if fd >= 0:
                os.close(fd)
                fd = -1
    except BaseException as primary_error:
        if parent_fd >= 0:
            _cleanup_created_leaf_antigravity(parent_fd, name, fd, created_identity, primary_error)
        raise
    finally:
        if fd >= 0: os.close(fd)
        if parent_fd >= 0: os.close(parent_fd)


def _validate_receipts_antigravity(inventory: tuple[str, ...], receipts: tuple[CandidateCommitReceiptAntigravity, ...]) -> None:
    """Reject any receipt not returned in the exact parent-owned inventory order."""
    if type(inventory) is not tuple or type(receipts) is not tuple or len(receipts) > len(inventory):
        raise ValueError("receipt prefix inputs are invalid")
    for sequence, receipt in enumerate(receipts):
        if (type(receipt) is not CandidateCommitReceiptAntigravity or receipt.sequence != sequence or
                receipt.name != inventory[sequence] or type(receipt.sha256) is not str or
                len(receipt.sha256) != 64 or any(character not in "0123456789abcdef" for character in receipt.sha256) or
                type(receipt.identity) is not tuple or len(receipt.identity) != 2 or
                any(type(part) is not int or part < 0 for part in receipt.identity)):
            raise ValueError("candidate commit receipt differs")


def verified_prefix_antigravity(directory: Path, inventory: tuple[str, ...], receipts: tuple[CandidateCommitReceiptAntigravity, ...]) -> int:
    """Revalidate only parent-returned commit receipts; pathnames never discover progress."""
    if not isinstance(directory, Path):
        raise ValueError("prefix inputs are invalid")
    _validate_receipts_antigravity(inventory, receipts)
    if not directory.is_dir() and receipts:
        raise ValueError("candidate receipt directory is unavailable")
    parent_fd = -1
    try:
        if receipts:
            parent_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        for receipt in receipts:
            fd = -1
            try:
                fd = os.open(receipt.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
                opened = os.fstat(fd)
                if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                        (opened.st_dev, opened.st_ino) != receipt.identity):
                    raise ValueError("received candidate receipt no longer matches descriptor")
                payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
                if (hashlib.sha256(payload).hexdigest() != receipt.sha256 or
                        canonical_json_bytes_antigravity(strict_json_bytes_antigravity(payload)) != payload):
                    raise ValueError("received candidate receipt no longer matches bytes")
                current = os.stat(receipt.name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != receipt.identity:
                    raise ValueError("received candidate receipt no longer matches namespace")
            except FileNotFoundError:
                return receipt.sequence
            finally:
                if fd >= 0:
                    os.close(fd)
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
    return len(receipts)
