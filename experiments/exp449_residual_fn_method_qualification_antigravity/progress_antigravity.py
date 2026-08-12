"""Create-only candidate leaves and verified contiguous-prefix helpers for Exp449."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity


def commit_candidate_leaf_antigravity(directory: Path, name: str, value: object) -> str:
    """Commit a candidate only after file and parent fsync plus canonical no-follow reread."""
    if not isinstance(directory, Path) or type(name) is not str or "/" in name or not name.endswith(".json"):
        raise ValueError("candidate destination is invalid")
    directory.mkdir(mode=0o700, exist_ok=True)
    payload = canonical_json_bytes_antigravity(value)
    destination = directory / name
    write_create_only_antigravity(destination, payload)
    parent_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(parent_fd)
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
            reread = b"".join(chunks)
            if reread != payload or canonical_json_bytes_antigravity(strict_json_bytes_antigravity(reread)) != reread:
                raise ValueError("candidate leaf reread is not canonical")
            return hashlib.sha256(reread).hexdigest()
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def verified_prefix_antigravity(directory: Path, inventory: tuple[str, ...]) -> int:
    """Derive the completed count exclusively from present canonical candidate leaves."""
    if not isinstance(directory, Path) or type(inventory) is not tuple:
        raise ValueError("prefix inputs are invalid")
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
