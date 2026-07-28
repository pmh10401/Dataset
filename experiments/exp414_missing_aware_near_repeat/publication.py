"""Fail-closed atomic publication for Exp414 qualification artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import uuid

from experiments.exp403_408.artifact_io import sha256_file

from .config import FROZEN_CONFIG
from .reporting import MANIFEST_FILE


def publish_payloads(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    provenance: Mapping[str, object],
    *,
    pre_publish_verify: Callable[[], None] | None = None,
) -> Mapping[str, str]:
    """Stage, verify, and atomically publish eight payloads plus a manifest."""
    FROZEN_CONFIG.validate()
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp414 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp414 output directory must not exist")
    if (
        not isinstance(payloads, Mapping)
        or MANIFEST_FILE in payloads
        or len(payloads) != 8
    ):
        raise ValueError(
            "Exp414 publication requires eight non-manifest artifacts"
        )

    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        artifact_hashes: dict[str, str] = {}
        for name, payload in sorted(payloads.items()):
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name
                or not isinstance(payload, bytes)
            ):
                raise ValueError("Exp414 staged artifact is invalid")
            path = staging / name
            _write_payload(path, payload)
            digest = sha256(payload).hexdigest()
            if sha256_file(path) != digest or path.read_bytes() != payload:
                raise ValueError(
                    f"Exp414 staged artifact verification failed: {name}"
                )
            artifact_hashes[name] = digest

        manifest = {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "provenance": _plain(provenance),
        }
        manifest_payload = _json_bytes(manifest)
        manifest_path = staging / MANIFEST_FILE
        _write_payload(manifest_path, manifest_payload)
        if (
            sha256_file(manifest_path)
            != sha256(manifest_payload).hexdigest()
            or manifest_path.read_bytes() != manifest_payload
        ):
            raise ValueError("Exp414 staged manifest verification failed")
        if {path.name for path in staging.iterdir()} != (
            set(payloads) | {MANIFEST_FILE}
        ):
            raise ValueError("Exp414 staged artifact coverage mismatch")
        if pre_publish_verify is not None:
            pre_publish_verify()
        if output.exists():
            raise FileExistsError(
                "Exp414 output directory appeared during staging"
            )
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return dict(sorted(artifact_hashes.items()))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_payload(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(child)
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


__all__ = ["publish_payloads"]
