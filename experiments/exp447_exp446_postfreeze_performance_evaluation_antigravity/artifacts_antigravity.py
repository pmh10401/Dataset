"""Canonical JSON and no-clobber output primitives for Exp447."""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Mapping, Sequence


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {key: _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_antigravity(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_antigravity(item) for item in value]
    raise ValueError("canonical JSON value has an unsupported type")


def canonical_json_bytes_antigravity(value: object) -> bytes:
    """Encode deterministic UTF-8 JSON with an explicit trailing newline."""
    try:
        return json.dumps(
            _json_safe_antigravity(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("canonical JSON value is invalid") from error


def write_create_only_antigravity(path: Path, payload: bytes) -> None:
    """Atomically reserve one new regular file without ever replacing evidence."""
    if not isinstance(payload, bytes):
        raise ValueError("artifact payload must be bytes")
    target = Path(path)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if type(written) is not int or written <= 0:
                raise OSError("create-only artifact write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["canonical_json_bytes_antigravity", "write_create_only_antigravity"]
