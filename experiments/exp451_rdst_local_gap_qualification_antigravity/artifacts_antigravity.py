"""Canonical, descriptor-checked durable files for Exp451 prediction."""
from __future__ import annotations
import json, os, stat, tempfile
from pathlib import Path

def _pairs_antigravity(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate JSON key")
        out[key] = value
    return out

def canonical_json_bytes_antigravity(value: object) -> bytes:
    try: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error: raise ValueError("value is not canonical JSON") from error

def strict_json_bytes_antigravity(payload: bytes) -> object:
    if type(payload) is not bytes: raise ValueError("payload must be exact bytes")
    try: value = json.loads(payload.decode(), object_pairs_hook=_pairs_antigravity)
    except Exception as error: raise ValueError("invalid JSON bytes") from error
    if canonical_json_bytes_antigravity(value) != payload: raise ValueError("noncanonical JSON bytes")
    return value

def _safe_parent_antigravity(path: Path) -> int:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise ValueError("unsafe parent")
    return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))

def _write_all_antigravity(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if type(written) is not int or written <= 0:
            raise OSError("short artifact write")
        view = view[written:]

def write_create_only_antigravity(destination: Path, payload: bytes, *, fault_hook=None, sequence: int = 0) -> None:
    if not isinstance(destination, Path) or type(payload) is not bytes: raise ValueError("invalid writer inputs")
    parent_fd = _safe_parent_antigravity(destination.parent); fd = -1
    try:
        if fault_hook: fault_hook("writer", sequence)
        fd = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("unsafe new leaf")
        _write_all_antigravity(fd, payload)
        if fault_hook: fault_hook("fsync", sequence)
        os.fsync(fd)
        if os.fstat(fd).st_size != len(payload):
            raise OSError("artifact byte count differs")
        os.close(fd); fd = -1
        os.fsync(parent_fd)
    finally:
        if fd >= 0: os.close(fd)
        os.close(parent_fd)

def write_atomic_antigravity(destination: Path, payload: bytes, *, fault_hook=None, sequence: int = 0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = _safe_parent_antigravity(destination.parent)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_antigravity_", dir=destination.parent)
    try:
        _write_all_antigravity(fd, payload)
        if fault_hook: fault_hook("progress_fsync", sequence)
        os.fsync(fd)
        if os.fstat(fd).st_size != len(payload):
            raise OSError("atomic byte count differs")
        os.close(fd); fd = -1
        if fault_hook: fault_hook("rename", sequence)
        os.replace(temporary, destination)
        os.fsync(parent_fd)
    finally:
        if fd >= 0: os.close(fd)
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        os.close(parent_fd)

def read_canonical_file_antigravity(path: Path) -> object:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1: raise ValueError("leaf must be single-link regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino): raise ValueError("leaf identity changed")
        payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
        after = os.stat(path, follow_symlinks=False)
        if (after.st_dev, after.st_ino, after.st_nlink) != (opened.st_dev, opened.st_ino, 1): raise ValueError("leaf changed")
        return strict_json_bytes_antigravity(payload)
    finally: os.close(fd)

__all__ = ["canonical_json_bytes_antigravity", "strict_json_bytes_antigravity", "write_create_only_antigravity", "write_atomic_antigravity", "read_canonical_file_antigravity"]
