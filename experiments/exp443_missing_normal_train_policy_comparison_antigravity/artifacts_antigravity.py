"""Exp443 artifact boundary.

The descriptor, canonical-byte, fsync, and Darwin no-clobber implementations
are intentionally reused from the already-tested Exp441 primitives.  This
module only narrows their names/paths to Exp443's antigravity artifact surface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    PinnedFileIdentityAntigravity,
    PinnedRegularInventoryAntigravity,
    canonical_json_bytes_antigravity as _canonical_json_bytes_base,
    canonical_jsonl_bytes_antigravity as _canonical_jsonl_bytes_base,
    exclusive_rename_antigravity as _exclusive_rename_base,
    fsync_directory_antigravity as _fsync_directory_base,
    parse_canonical_json_bytes_antigravity as _parse_canonical_json_bytes_base,
    pin_exact_regular_inventory_antigravity as _pin_exact_inventory_base,
    read_pinned_regular_bytes_antigravity as _read_pinned_regular_bytes_base,
    write_create_only_antigravity as _write_create_only_base,
)


def canonical_json_bytes_antigravity(value: object) -> bytes:
    return _canonical_json_bytes_base(value)


def canonical_jsonl_bytes_antigravity(rows: Iterable[Mapping[str, object]]) -> bytes:
    return _canonical_jsonl_bytes_base(rows)


def parse_canonical_json_bytes_antigravity(payload: bytes) -> object:
    return _parse_canonical_json_bytes_base(payload)


def _require_antigravity_name_antigravity(path: Path) -> Path:
    path = Path(path)
    if "antigravity" not in path.name:
        raise ValueError("Exp443 artifacts must contain antigravity in their filename")
    return path


def read_pinned_regular_bytes_antigravity(path: Path) -> tuple[bytes, str]:
    return _read_pinned_regular_bytes_base(_require_antigravity_name_antigravity(path))


def pin_exact_regular_inventory_antigravity(
    root: Path, names: Iterable[str], *, directory_mode: int = 0o700, file_mode: int = 0o600,
) -> PinnedRegularInventoryAntigravity:
    root = Path(root)
    if "antigravity" not in root.name:
        raise ValueError("Exp443 inventory root must contain antigravity in its filename")
    if directory_mode != 0o700 or file_mode != 0o600:
        raise ValueError("Exp443 inventory modes are frozen")
    names = tuple(names)
    if any("antigravity" not in name for name in names):
        raise ValueError("Exp443 inventory entries must contain antigravity")
    return _pin_exact_inventory_base(root, names, directory_mode=directory_mode, file_mode=file_mode)


def write_create_only_antigravity(path: Path, payload: bytes) -> None:
    _write_create_only_base(_require_antigravity_name_antigravity(path), payload)


def fsync_directory_antigravity(path: Path) -> None:
    _fsync_directory_base(Path(path))


def exclusive_rename_antigravity(source: Path, destination: Path) -> None:
    _exclusive_rename_base(_require_antigravity_name_antigravity(source), _require_antigravity_name_antigravity(destination))


def write_canonical_json_create_only_antigravity(path: Path, value: object) -> None:
    """Canonical JSON create-only publication with the private 0600 mode."""
    write_create_only_antigravity(path, canonical_json_bytes_antigravity(value))


def write_canonical_jsonl_create_only_antigravity(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Canonical JSONL create-only publication with the private 0600 mode."""
    write_create_only_antigravity(path, canonical_jsonl_bytes_antigravity(rows))


__all__ = [
    "PinnedFileIdentityAntigravity", "PinnedRegularInventoryAntigravity",
    "canonical_json_bytes_antigravity", "canonical_jsonl_bytes_antigravity",
    "parse_canonical_json_bytes_antigravity", "read_pinned_regular_bytes_antigravity",
    "pin_exact_regular_inventory_antigravity", "write_create_only_antigravity",
    "write_canonical_json_create_only_antigravity", "write_canonical_jsonl_create_only_antigravity",
    "fsync_directory_antigravity", "exclusive_rename_antigravity",
]
