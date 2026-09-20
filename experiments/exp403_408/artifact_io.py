"""Atomic, hash-addressed artifact helpers for Exp403-408."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from io import StringIO
from typing import Mapping, Sequence
import csv

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


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value deterministically for frozen artifact hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 digest for an in-memory canonical payload."""
    return hashlib.sha256(payload).hexdigest()


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


def write_json_atomic(path: Path, value: object) -> str:
    """Write canonical JSON through a sibling temporary file and return its hash."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    _replace_bytes(destination, payload)
    return sha256_bytes(payload)


def write_csv_atomic(
    path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> str:
    """Write a canonical UTF-8 CSV through a sibling temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    payload = buffer.getvalue().encode("utf-8")
    _replace_bytes(destination, payload)
    return sha256_bytes(payload)


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
