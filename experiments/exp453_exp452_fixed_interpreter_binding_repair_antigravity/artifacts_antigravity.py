"""Canonical, descriptor-checked artifact byte helpers for Exp453."""

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


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _relative_parts_antigravity(relative: str) -> tuple[str, ...]:
    candidate = Path(relative)
    if type(relative) is not str or candidate.is_absolute() or not candidate.parts or any(part in ("", ".", "..") for part in candidate.parts):
        raise ValueError("artifact inventory path is not a safe relative path")
    return candidate.parts


def _directory_flags_antigravity() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _after_inventory_antigravity() -> None:
    """Deliberate test seam between descriptor inventory and descriptor reads."""


def _before_leaf_read_antigravity(relative: str) -> None:
    """Deliberate test seam after opening one retained leaf descriptor."""


def _read_descriptor_antigravity(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


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
    """Hash only an exact inventory rooted at one retained no-follow descriptor."""
    if not isinstance(root, Path) or type(inventory) is not tuple:
        raise ValueError("root and inventory require Path and exact tuple")
    if len(set(inventory)) != len(inventory):
        raise ValueError("duplicate artifact inventory key")
    expected_parts = {relative: _relative_parts_antigravity(relative) for relative in inventory}
    root_path_info = os.stat(root, follow_symlinks=False)
    if stat.S_ISLNK(root_path_info.st_mode) or not stat.S_ISDIR(root_path_info.st_mode):
        raise ValueError("artifact root must be a non-symlink directory")
    root_fd = os.open(root, _directory_flags_antigravity())
    directory_descriptors: dict[tuple[str, ...], int] = {(): root_fd}
    try:
        root_info = os.fstat(root_fd)
        if _identity_antigravity(root_path_info) != _identity_antigravity(root_info):
            raise ValueError("root namespace changed during open")
        directory_identities: dict[tuple[str, ...], tuple[int, int, int, int, int, int, int]] = {(): _identity_antigravity(root_info)}
        leaf_identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}

        def inventory_directory_antigravity(directory_fd: int, prefix: tuple[str, ...]) -> None:
            for name in sorted(os.listdir(directory_fd)):
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative_parts = prefix + (name,)
                relative = "/".join(relative_parts)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(f"symlink is forbidden in artifact root: {relative}")
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(name, _directory_flags_antigravity(), dir_fd=directory_fd)
                    child_info = os.fstat(child_fd)
                    if _identity_antigravity(info) != _identity_antigravity(child_info):
                        os.close(child_fd)
                        raise ValueError("artifact namespace changed during inventory")
                    directory_identities[relative_parts] = _identity_antigravity(child_info)
                    directory_descriptors[relative_parts] = child_fd
                    inventory_directory_antigravity(child_fd, relative_parts)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    leaf_identities[relative] = _identity_antigravity(info)
                else:
                    raise ValueError(f"artifact leaf must be a single-link regular file: {relative}")

        inventory_directory_antigravity(root_fd, ())
        expected = set(expected_parts)
        extra = sorted(set(leaf_identities) - expected)
        missing = sorted(expected - set(leaf_identities))
        if extra:
            raise ValueError(f"unexpected artifact leaf: {extra[0]}")
        if missing:
            raise ValueError(f"missing artifact leaf: {missing[0]}")
        _after_inventory_antigravity()
        root_after_hook = os.fstat(root_fd)
        root_namespace = os.stat(root, follow_symlinks=False)
        if _identity_antigravity(root_namespace) != _identity_antigravity(root_after_hook):
            raise ValueError("root namespace changed after inventory")
        if _identity_antigravity(root_after_hook) != directory_identities[()]:
            raise ValueError("artifact namespace changed after inventory")

        def revalidate_retained_tree_antigravity() -> None:
            for prefix in sorted(directory_descriptors, key=lambda item: (len(item), item)):
                directory_fd = directory_descriptors[prefix]
                if _identity_antigravity(os.fstat(directory_fd)) != directory_identities[prefix]:
                    raise ValueError("artifact namespace changed during read")
                if prefix:
                    parent_fd = directory_descriptors[prefix[:-1]]
                    namespace_info = os.stat(prefix[-1], dir_fd=parent_fd, follow_symlinks=False)
                    if _identity_antigravity(namespace_info) != directory_identities[prefix]:
                        raise ValueError("artifact namespace changed during read")
                else:
                    namespace_info = os.stat(root, follow_symlinks=False)
                    if _identity_antigravity(namespace_info) != directory_identities[()]:
                        raise ValueError("artifact namespace changed during read")
                expected_names = {
                    child[-1] for child in directory_identities
                    if len(child) == len(prefix) + 1 and child[:-1] == prefix
                } | {
                    _relative_parts_antigravity(relative)[-1] for relative in leaf_identities
                    if len(_relative_parts_antigravity(relative)) == len(prefix) + 1
                    and _relative_parts_antigravity(relative)[:-1] == prefix
                }
                observed_names = set(os.listdir(directory_fd))
                if observed_names != expected_names:
                    raise ValueError("artifact namespace changed during read")
                for name in observed_names:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    child_prefix = prefix + (name,)
                    relative = "/".join(child_prefix)
                    if child_prefix in directory_identities:
                        if _identity_antigravity(info) != directory_identities[child_prefix]:
                            raise ValueError("artifact namespace changed during read")
                    elif relative in leaf_identities:
                        if _identity_antigravity(info) != leaf_identities[relative]:
                            raise ValueError("artifact namespace changed during read")
                    else:
                        raise ValueError("artifact namespace changed during read")

        revalidate_retained_tree_antigravity()
        digests: dict[str, str] = {}
        for relative in sorted(inventory):
            parts = expected_parts[relative]
            parent_fd = directory_descriptors[parts[:-1]]
            leaf_name = parts[-1]
            before = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity_antigravity(before) != leaf_identities[relative]:
                raise ValueError("artifact namespace changed during read")
            read_flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            leaf_fd = os.open(leaf_name, read_flags, dir_fd=parent_fd)
            try:
                opened = os.fstat(leaf_fd)
                if _identity_antigravity(opened) != leaf_identities[relative] or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise ValueError("artifact namespace changed during read")
                _before_leaf_read_antigravity(relative)
                revalidate_retained_tree_antigravity()
                payload = _read_descriptor_antigravity(leaf_fd)
                if _identity_antigravity(os.fstat(leaf_fd)) != leaf_identities[relative]:
                    raise ValueError("artifact leaf changed during read")
            finally:
                os.close(leaf_fd)
            after = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity_antigravity(after) != leaf_identities[relative]:
                raise ValueError("artifact namespace changed during read")
            digests[relative] = hashlib.sha256(payload).hexdigest()
        revalidate_retained_tree_antigravity()
        return canonical_json_bytes_antigravity(digests)
    finally:
        for directory_fd in reversed(tuple(directory_descriptors.values())):
            os.close(directory_fd)
