"""Descriptor-pinned, create-only artifact primitives for Exp441."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_bytes


def canonical_json_bytes_antigravity(value: object) -> bytes:
    """Encode one canonical JSON document with its required trailing newline."""
    return canonical_json_bytes(value) + b"\n"


def canonical_jsonl_bytes_antigravity(rows: Iterable[Mapping[str, object]]) -> bytes:
    """Encode canonical JSONL without accepting mutable row aliases."""
    return b"".join(canonical_json_bytes_antigravity(dict(row)) for row in rows)


def parse_canonical_json_bytes_antigravity(payload: bytes) -> object:
    """Decode exactly one canonical JSON document while rejecting duplicate keys."""
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"nonfinite JSON constant: {constant}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source is not JSON") from error
    if canonical_json_bytes_antigravity(value) != payload:
        raise ValueError("source is not canonical JSON")
    return value


def read_pinned_regular_bytes_antigravity(path: Path) -> tuple[bytes, str]:
    """Read one stable regular file through a descriptor that cannot follow links."""
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source is not a regular file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        stable_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if stable_before != stable_after or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("source changed while pinned")
        payload = b"".join(blocks)
        return payload, sha256_bytes(payload)
    finally:
        os.close(descriptor)


def write_create_only_antigravity(path: Path, payload: bytes) -> None:
    """Write all bytes to a new private file or fail without replacing a winner."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written == 0:
                raise OSError(errno.EIO, "create-only write made no progress", str(path))
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory_antigravity(path: Path) -> None:
    """Durably persist a directory entry without traversing a symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_rename_antigravity(source: Path, destination: Path) -> None:
    """Atomically rename on Darwin only when the destination does not exist."""
    renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), ctypes.c_uint(0x00000004)) != 0:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, os.strerror(code), str(destination))


__all__ = [
    "canonical_json_bytes_antigravity", "canonical_jsonl_bytes_antigravity",
    "parse_canonical_json_bytes_antigravity",
    "read_pinned_regular_bytes_antigravity", "write_create_only_antigravity",
    "fsync_directory_antigravity", "exclusive_rename_antigravity",
]
