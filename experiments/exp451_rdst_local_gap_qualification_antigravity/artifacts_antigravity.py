"""Canonical, descriptor-checked durable files for Exp451 prediction."""
from __future__ import annotations
import json, os, stat, secrets
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

def _identity_antigravity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

def _read_fd_antigravity(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk: return b"".join(chunks)
        chunks.append(chunk)

def _verify_namespace_antigravity(parent_fd: int, name: str, expected: os.stat_result) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if _identity_antigravity(current) != _identity_antigravity(expected):
        raise ValueError("artifact namespace or bytes changed")

def _verify_namespace_reopen_antigravity(parent_fd: int, name: str, expected: os.stat_result, payload: bytes | None = None) -> None:
    fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if _identity_antigravity(opened) != _identity_antigravity(expected):
            raise ValueError("artifact namespace reopen differs")
        if payload is not None and _read_fd_antigravity(fd) != payload:
            raise ValueError("artifact namespace bytes differ")
    finally:
        os.close(fd)

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
        fd = os.open(destination.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("unsafe new leaf")
        _write_all_antigravity(fd, payload)
        if fault_hook: fault_hook("fsync", sequence)
        os.fsync(fd)
        written_info = os.fstat(fd)
        if written_info.st_size != len(payload) or written_info.st_nlink != 1 or not stat.S_ISREG(written_info.st_mode):
            raise OSError("artifact byte count differs")
        if _read_fd_antigravity(fd) != payload:
            raise OSError("artifact canonical reread differs")
        _verify_namespace_reopen_antigravity(parent_fd, destination.name, os.fstat(fd), payload)
        os.fsync(parent_fd)
    finally:
        if fd >= 0: os.close(fd)
        os.close(parent_fd)

def write_atomic_antigravity(destination: Path, payload: bytes, *, fault_hook=None, sequence: int = 0) -> None:
    parent_fd = _safe_parent_antigravity(destination.parent)
    temporary = f".tmp_antigravity_{secrets.token_hex(12)}"
    fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
    try:
        _write_all_antigravity(fd, payload)
        if fault_hook: fault_hook("progress_fsync", sequence)
        os.fsync(fd)
        if os.fstat(fd).st_size != len(payload):
            raise OSError("atomic byte count differs")
        temporary_info = os.fstat(fd)
        if _read_fd_antigravity(fd) != payload:
            raise OSError("atomic temporary bytes differ")
        _verify_namespace_reopen_antigravity(parent_fd, temporary, temporary_info, payload)
        if fault_hook: fault_hook("rename", sequence)
        os.replace(temporary, destination.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
        destination_fd = os.open(destination.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            destination_info = os.fstat(destination_fd)
            if not stat.S_ISREG(destination_info.st_mode) or destination_info.st_nlink != 1 or _read_fd_antigravity(destination_fd) != payload:
                raise ValueError("atomic destination differs")
            _verify_namespace_reopen_antigravity(parent_fd, destination.name, destination_info, payload)
        finally: os.close(destination_fd)
    finally:
        if fd >= 0: os.close(fd)
        try: os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError: pass
        os.close(parent_fd)

def read_exact_bytes_antigravity(path: Path) -> bytes:
    parent_fd = _safe_parent_antigravity(path.parent)
    before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1: os.close(parent_fd); raise ValueError("leaf must be single-link regular file")
    fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if _identity_antigravity(opened) != _identity_antigravity(before): raise ValueError("leaf identity changed")
        payload = _read_fd_antigravity(fd)
        if _identity_antigravity(os.fstat(fd)) != _identity_antigravity(opened): raise ValueError("leaf changed during read")
        _verify_namespace_antigravity(parent_fd, path.name, opened)
        return payload
    finally: os.close(fd); os.close(parent_fd)

def read_canonical_file_antigravity(path: Path) -> object:
    return strict_json_bytes_antigravity(read_exact_bytes_antigravity(path))

def append_text_antigravity(destination: Path, payload: bytes) -> None:
    if type(payload) is not bytes: raise ValueError("append payload must be bytes")
    parent_fd = _safe_parent_antigravity(destination.parent)
    fd = os.open(destination.name, os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1: raise ValueError("unsafe append target")
        previous_size = before.st_size
        _write_all_antigravity(fd, payload); os.fsync(fd)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, stat.S_IFMT(after.st_mode), after.st_nlink) != (before.st_dev, before.st_ino, stat.S_IFMT(before.st_mode), 1):
            raise ValueError("append target changed")
        if after.st_size != previous_size + len(payload):
            raise OSError("append size differs")
        os.lseek(fd, previous_size, os.SEEK_SET)
        appended = b""
        while len(appended) < len(payload):
            chunk = os.read(fd, len(payload) - len(appended))
            if not chunk: break
            appended += chunk
        if appended != payload:
            raise OSError("append bytes differ")
        _verify_namespace_reopen_antigravity(parent_fd, destination.name, after)
        os.fsync(parent_fd)
    finally: os.close(fd); os.close(parent_fd)

__all__ = ["append_text_antigravity", "canonical_json_bytes_antigravity", "strict_json_bytes_antigravity", "write_create_only_antigravity", "write_atomic_antigravity", "read_canonical_file_antigravity", "read_exact_bytes_antigravity"]
