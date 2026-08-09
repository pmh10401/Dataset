"""Thin Exp444 boundary around Exp441's canonical/create-only primitives.

Exp444 deliberately does not fork or duplicate the descriptor and artifact
implementation.  Re-exporting these narrow wrappers makes the consuming
modules' import boundary explicit while preserving Exp441's no-clobber and
canonical-byte semantics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    PinnedFileIdentityAntigravity,
    PinnedRegularInventoryAntigravity,
    canonical_json_bytes_antigravity as _canonical_json_bytes_exp441,
    canonical_jsonl_bytes_antigravity as _canonical_jsonl_bytes_exp441,
    exclusive_rename_antigravity as _exclusive_rename_exp441,
    fsync_directory_antigravity as _fsync_directory_exp441,
    parse_canonical_json_bytes_antigravity as _parse_canonical_json_bytes_exp441,
    pin_exact_regular_inventory_antigravity as _pin_exact_regular_inventory_exp441,
    read_pinned_regular_bytes_antigravity as _read_pinned_regular_bytes_exp441,
    write_create_only_antigravity as _write_create_only_exp441,
)


def canonical_json_bytes_antigravity(value: object) -> bytes:
    return _canonical_json_bytes_exp441(value)


def canonical_jsonl_bytes_antigravity(rows: Iterable[Mapping[str, object]]) -> bytes:
    return _canonical_jsonl_bytes_exp441(rows)


def parse_canonical_json_bytes_antigravity(payload: bytes) -> object:
    return _parse_canonical_json_bytes_exp441(payload)


def read_pinned_regular_bytes_antigravity(path: Path) -> tuple[bytes, str]:
    return _read_pinned_regular_bytes_exp441(Path(path))


def write_create_only_antigravity(path: Path, payload: bytes) -> None:
    _write_create_only_exp441(Path(path), payload)


def fsync_directory_antigravity(path: Path) -> None:
    _fsync_directory_exp441(Path(path))


def exclusive_rename_antigravity(source: Path, destination: Path) -> None:
    _exclusive_rename_exp441(Path(source), Path(destination))


def pin_exact_regular_inventory_antigravity(
    root: Path,
    names: Iterable[str],
    *,
    directory_mode: int = 0o700,
    file_mode: int = 0o600,
) -> PinnedRegularInventoryAntigravity:
    return _pin_exact_regular_inventory_exp441(Path(root), names, directory_mode=directory_mode, file_mode=file_mode)


__all__ = [
    "canonical_json_bytes_antigravity", "canonical_jsonl_bytes_antigravity",
    "parse_canonical_json_bytes_antigravity", "read_pinned_regular_bytes_antigravity",
    "write_create_only_antigravity", "fsync_directory_antigravity",
    "exclusive_rename_antigravity", "pin_exact_regular_inventory_antigravity",
    "PinnedFileIdentityAntigravity", "PinnedRegularInventoryAntigravity",
]
