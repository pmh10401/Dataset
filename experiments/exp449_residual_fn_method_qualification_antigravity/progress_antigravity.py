"""Create-only candidate leaves and verified contiguous-prefix helpers for Exp449."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity


_ATTEMPT_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "progress_antigravity.json", "progress_antigravity.jsonl",
    "progress_antigravity.txt", "pid_antigravity.json", "candidates_antigravity",
)


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


def record_verified_progress_antigravity(root: Path, inventory: tuple[str, ...], completed: int) -> None:
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
    fd = os.open(progress, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    with (root / "progress_antigravity.jsonl").open("ab") as handle:
        handle.write(canonical_json_bytes_antigravity({"dataset_unit_completed": completed}))
        handle.flush(); os.fsync(handle.fileno())
    with (root / "progress_antigravity.txt").open("ab") as handle:
        handle.write(f"{completed}\n".encode())
        handle.flush(); os.fsync(handle.fileno())


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
