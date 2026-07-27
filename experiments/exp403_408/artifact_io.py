"""Atomic, hash-addressed artifact helpers for Exp403-408."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for an artifact without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> str:
    """Verify an artifact against its frozen SHA256 and return the observed digest."""
    actual = sha256_file(Path(path))
    if actual != str(expected):
        raise ValueError(f"SHA256 mismatch: path={path}")
    return actual


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    """Write JSONL through a sibling temporary file and return its SHA256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    _replace_bytes(destination, payload)
    return hashlib.sha256(payload).hexdigest()


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write a compressed NPZ atomically and return its SHA256."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **dict(arrays))
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return sha256_file(destination)


def _replace_bytes(destination: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
