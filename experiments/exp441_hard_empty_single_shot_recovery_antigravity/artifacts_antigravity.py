"""Descriptor-pinned, create-only artifact primitives for Exp441."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
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


@dataclass(frozen=True)
class PinnedFileIdentityAntigravity:
    """Mutation-relevant identity captured from one open regular-file descriptor."""

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _pinned_file_identity_antigravity(status: os.stat_result) -> PinnedFileIdentityAntigravity:
    return PinnedFileIdentityAntigravity(
        status.st_dev, status.st_ino, status.st_mode, status.st_size,
        status.st_mtime_ns, status.st_ctime_ns,
    )


@dataclass
class PinnedRegularInventoryAntigravity:
    """Own descriptors for one complete verified inventory until the caller closes it."""

    payloads: Mapping[str, bytes]
    metadata: Mapping[str, PinnedFileIdentityAntigravity]
    _descriptors_antigravity: dict[str, int]

    def close_antigravity(self) -> None:
        """Close every retained descriptor exactly once, preserving the first close error."""
        primary: BaseException | None = None
        for name, descriptor in tuple(self._descriptors_antigravity.items()):
            del self._descriptors_antigravity[name]
            try:
                os.close(descriptor)
            except BaseException as error:
                if primary is None:
                    primary = error
        if primary is not None:
            raise primary

    def __enter__(self) -> PinnedRegularInventoryAntigravity:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        if exc_value is None:
            self.close_antigravity()
        else:
            try:
                self.close_antigravity()
            except BaseException as close_error:
                if hasattr(exc_value, "add_note"):
                    exc_value.add_note(f"pinned descriptor cleanup failed: {type(close_error).__name__}: {close_error}")
        return False


def _exact_regular_inventory_entries_antigravity(root: Path, names: tuple[str, ...], *, directory_mode: int, file_mode: int) -> dict[str, Path]:
    root_status = root.lstat()
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode) or stat.S_IMODE(root_status.st_mode) != directory_mode:
        raise ValueError("pinned inventory root is invalid")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != set(names):
        raise ValueError("pinned inventory is invalid")
    for name in names:
        status = entries[name].stat(follow_symlinks=False)
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode) or stat.S_IMODE(status.st_mode) != file_mode:
            raise ValueError("pinned inventory entry is invalid")
    return entries


def pin_exact_regular_inventory_antigravity(
    root: Path, names: Iterable[str], *, directory_mode: int = 0o700, file_mode: int = 0o600,
) -> PinnedRegularInventoryAntigravity:
    """Open, read, and revalidate one exact regular-file inventory while all FDs stay pinned."""
    root = Path(root)
    ordered_names = tuple(names)
    if (not ordered_names or len(ordered_names) != len(set(ordered_names)) or
            any(not isinstance(name, str) or not name or "/" in name for name in ordered_names)):
        raise ValueError("pinned inventory names are invalid")
    descriptors: dict[str, int] = {}
    identities: dict[str, PinnedFileIdentityAntigravity] = {}
    try:
        entries = _exact_regular_inventory_entries_antigravity(root, ordered_names, directory_mode=directory_mode, file_mode=file_mode)
        for name in ordered_names:
            namespace_status = entries[name].stat(follow_symlinks=False)
            descriptor = os.open(entries[name], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            descriptors[name] = descriptor
            opened_status = os.fstat(descriptor)
            opened_identity = _pinned_file_identity_antigravity(opened_status)
            if (not stat.S_ISREG(opened_status.st_mode) or stat.S_IMODE(opened_status.st_mode) != file_mode or
                    opened_identity != _pinned_file_identity_antigravity(namespace_status)):
                raise ValueError("pinned inventory entry changed before open")
            identities[name] = opened_identity
        payloads: dict[str, bytes] = {}
        for name in ordered_names:
            blocks: list[bytes] = []
            while block := os.read(descriptors[name], 1024 * 1024):
                blocks.append(block)
            payloads[name] = b"".join(blocks)
        entries_after = _exact_regular_inventory_entries_antigravity(root, ordered_names, directory_mode=directory_mode, file_mode=file_mode)
        for name in ordered_names:
            descriptor_identity = _pinned_file_identity_antigravity(os.fstat(descriptors[name]))
            namespace_identity = _pinned_file_identity_antigravity(entries_after[name].stat(follow_symlinks=False))
            if descriptor_identity != identities[name] or namespace_identity != identities[name]:
                raise ValueError("pinned inventory changed while open")
        return PinnedRegularInventoryAntigravity(
            MappingProxyType(payloads), MappingProxyType(identities), descriptors,
        )
    except BaseException:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


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
    "read_pinned_regular_bytes_antigravity", "PinnedFileIdentityAntigravity",
    "PinnedRegularInventoryAntigravity", "pin_exact_regular_inventory_antigravity",
    "write_create_only_antigravity",
    "fsync_directory_antigravity", "exclusive_rename_antigravity",
]
