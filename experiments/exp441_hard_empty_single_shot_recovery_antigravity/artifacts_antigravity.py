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
    _root_antigravity: Path
    _names_antigravity: tuple[str, ...]
    _directory_mode_antigravity: int
    _file_mode_antigravity: int
    _root_metadata_antigravity: PinnedFileIdentityAntigravity
    _root_descriptor_antigravity: int

    def revalidate_antigravity(self) -> None:
        """Fail closed if the retained root or any named entry changed after initial pinning."""
        self.revalidate_at_antigravity(self._root_antigravity)

    def revalidate_at_antigravity(self, namespace_root: Path, *, renamed_antigravity: bool = False) -> None:
        """Bind retained descriptors to an exact current directory pathname after a safe rename."""
        try:
            root_descriptor_identity = _pinned_file_identity_antigravity(os.fstat(self._root_descriptor_antigravity))
            root_namespace = Path(namespace_root)
            root_namespace_identity = _pinned_file_identity_antigravity(root_namespace.lstat())
            root_identity_matches = (
                root_descriptor_identity == self._root_metadata_antigravity and
                root_namespace_identity == self._root_metadata_antigravity
            )
            if renamed_antigravity:
                root_identity_matches = (
                    root_descriptor_identity.device == self._root_metadata_antigravity.device and
                    root_descriptor_identity.inode == self._root_metadata_antigravity.inode and
                    root_descriptor_identity.mode == self._root_metadata_antigravity.mode and
                    root_descriptor_identity.size == self._root_metadata_antigravity.size and
                    root_descriptor_identity.mtime_ns == self._root_metadata_antigravity.mtime_ns and
                    root_namespace_identity.device == root_descriptor_identity.device and
                    root_namespace_identity.inode == root_descriptor_identity.inode and
                    root_namespace_identity.mode == root_descriptor_identity.mode and
                    root_namespace_identity.size == root_descriptor_identity.size and
                    root_namespace_identity.mtime_ns == root_descriptor_identity.mtime_ns
                )
            if (not root_identity_matches or
                    not stat.S_ISDIR(root_descriptor_identity.mode) or
                    stat.S_IMODE(root_descriptor_identity.mode) != self._directory_mode_antigravity):
                raise ValueError("pinned inventory root changed")
            if set(os.listdir(self._root_descriptor_antigravity)) != set(self._names_antigravity):
                raise ValueError("pinned inventory changed")
            for name in self._names_antigravity:
                descriptor_identity = _pinned_file_identity_antigravity(os.fstat(self._descriptors_antigravity[name]))
                namespace_identity = _pinned_file_identity_antigravity((root_namespace / name).stat(follow_symlinks=False))
                if (descriptor_identity != self.metadata[name] or namespace_identity != self.metadata[name] or
                        not stat.S_ISREG(descriptor_identity.mode) or stat.S_IMODE(descriptor_identity.mode) != self._file_mode_antigravity):
                    raise ValueError("pinned inventory changed")
        except OSError as error:
            raise ValueError("pinned inventory changed") from error

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
        root_descriptor = self._root_descriptor_antigravity
        self._root_descriptor_antigravity = -1
        if root_descriptor >= 0:
            try:
                os.close(root_descriptor)
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
    root_descriptor = -1
    identities: dict[str, PinnedFileIdentityAntigravity] = {}
    try:
        root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        root_identity = _pinned_file_identity_antigravity(os.fstat(root_descriptor))
        if (not stat.S_ISDIR(root_identity.mode) or stat.S_IMODE(root_identity.mode) != directory_mode or
                root_identity != _pinned_file_identity_antigravity(root.lstat())):
            raise ValueError("pinned inventory root is invalid")
        if set(os.listdir(root_descriptor)) != set(ordered_names):
            raise ValueError("pinned inventory is invalid")
        for name in ordered_names:
            namespace_status = os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
            if (not stat.S_ISREG(namespace_status.st_mode) or stat.S_IMODE(namespace_status.st_mode) != file_mode):
                raise ValueError("pinned inventory entry is invalid")
            descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_descriptor)
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
        pinned = PinnedRegularInventoryAntigravity(
            MappingProxyType(payloads), MappingProxyType(identities), descriptors, root,
            ordered_names, directory_mode, file_mode, root_identity, root_descriptor,
        )
        pinned.revalidate_antigravity()
        return pinned
    except BaseException:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if root_descriptor >= 0:
            try:
                os.close(root_descriptor)
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
