"""Canonical, descriptor-checked artifact byte helpers for Exp449."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable


def _reject_duplicate_keys_antigravity(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes_antigravity(value: object) -> bytes:
    """Encode a JSON value in the sole byte representation accepted by Task 2."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical JSON") from error
    return text.encode("utf-8") + b"\n"


def canonical_jsonl_bytes_antigravity(records: Iterable[object]) -> bytes:
    if isinstance(records, (str, bytes, bytearray)):
        raise ValueError("JSONL records must be an iterable of JSON values")
    return b"".join(canonical_json_bytes_antigravity(record) for record in records)


def strict_json_bytes_antigravity(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise ValueError("JSON payload must be exact bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys_antigravity)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("duplicate JSON key:"):
            raise
        raise ValueError("invalid JSON bytes") from error
    if canonical_json_bytes_antigravity(value) != payload:
        raise ValueError("non-canonical JSON bytes")
    return value


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns)


def _require_regular_single_link_antigravity(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"symlink is forbidden: {path}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"artifact leaf must be a single-link regular file: {path}")
    return info


def _require_safe_relative_antigravity(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if type(relative) is not str or candidate.is_absolute() or not candidate.parts or any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError("artifact inventory path is not a safe relative path")
    current = root
    root_info = os.lstat(current)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("artifact root must be a non-symlink directory")
    for part in candidate.parts[:-1]:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"artifact directory is unsafe: {current}")
    return root / candidate


def write_create_only_antigravity(destination: Path, payload: bytes) -> None:
    """Atomically write one new regular leaf, refusing symlink and namespace races."""
    if not isinstance(destination, Path) or type(payload) is not bytes:
        raise ValueError("destination and payload require Path and exact bytes")
    parent = destination.parent
    parent_info = os.lstat(parent)
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("destination parent is unsafe")
    try:
        prior = os.lstat(destination)
    except FileNotFoundError:
        prior = None
    if prior is not None:
        if stat.S_ISLNK(prior.st_mode):
            raise ValueError("destination is a symlink")
        raise FileExistsError("destination already exists")
    parent_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    parent_fd = os.open(parent, parent_flags)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(destination.name, flags, 0o600, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("new destination is not a single-link regular file")
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short artifact write")
            view = view[written:]
        os.fsync(fd)
        namespace_info = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity_antigravity(namespace_info) != _identity_antigravity(os.fstat(fd)):
            raise ValueError("destination namespace changed during write")
    finally:
        os.close(fd)
        os.close(parent_fd)


def artifact_map_bytes_antigravity(root: Path, inventory: tuple[str, ...]) -> bytes:
    """Return a canonical path-to-SHA256 map only for an exact safe inventory."""
    if not isinstance(root, Path) or type(inventory) is not tuple:
        raise ValueError("root and inventory require Path and exact tuple")
    if len(set(inventory)) != len(inventory):
        raise ValueError("duplicate artifact inventory key")
    expected = set(inventory)
    files: set[str] = set()
    root_info = os.lstat(root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("artifact root must be a non-symlink directory")
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            info = os.lstat(current_path / directory)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("symlink is forbidden in artifact root")
        for name in names:
            path = current_path / name
            _require_regular_single_link_antigravity(path)
            files.add(path.relative_to(root).as_posix())
    extra = sorted(files - expected)
    missing = sorted(expected - files)
    if extra:
        raise ValueError(f"unexpected artifact leaf: {extra[0]}")
    if missing:
        raise ValueError(f"missing artifact leaf: {missing[0]}")
    digests: dict[str, str] = {}
    for relative in sorted(inventory):
        path = _require_safe_relative_antigravity(root, relative)
        before = _require_regular_single_link_antigravity(path)
        read_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            read_flags |= os.O_NOFOLLOW
        fd = os.open(path, read_flags)
        try:
            after_fd = os.fstat(fd)
            if not stat.S_ISREG(after_fd.st_mode) or after_fd.st_nlink != 1:
                raise ValueError("artifact leaf changed to an unsafe type during read")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(fd)
        after_path = _require_regular_single_link_antigravity(path)
        if _identity_antigravity(before) != _identity_antigravity(after_fd) or _identity_antigravity(before) != _identity_antigravity(after_path):
            raise ValueError("artifact leaf changed during read")
        digests[relative] = hashlib.sha256(payload).hexdigest()
    return canonical_json_bytes_antigravity(digests)
