"""Fail-closed atomic publication for Exp411 qualification artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import os
from pathlib import Path
import shutil
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file

from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .reporting import MANIFEST_FILE


def publish_payloads(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    provenance: Mapping[str, object],
    *,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
    pre_publish_verify: Callable[[], None] | None = None,
) -> Mapping[str, str]:
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp411 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp411 output directory must not exist")
    if MANIFEST_FILE in payloads or len(payloads) != 7:
        raise ValueError("Exp411 publication requires seven non-manifest artifacts")

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        artifact_hashes: dict[str, str] = {}
        for name, payload in sorted(payloads.items()):
            if Path(name).name != name or not isinstance(payload, bytes):
                raise ValueError("Exp411 staged artifact is invalid")
            path = staging / name
            _write_payload(path, payload)
            digest = sha256(payload).hexdigest()
            if sha256_file(path) != digest or path.read_bytes() != payload:
                raise ValueError(f"Exp411 staged artifact verification failed: {name}")
            artifact_hashes[name] = digest

        manifest = {
            "experiment_id": config.experiment_id,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "provenance": _plain(provenance),
        }
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        manifest_path = staging / MANIFEST_FILE
        _write_payload(manifest_path, manifest_payload)
        if sha256_file(manifest_path) != sha256(manifest_payload).hexdigest():
            raise ValueError("Exp411 staged manifest verification failed")
        if {path.name for path in staging.iterdir()} != set(payloads) | {MANIFEST_FILE}:
            raise ValueError("Exp411 staged artifact coverage mismatch")
        if pre_publish_verify is not None:
            pre_publish_verify()
        if output.exists():
            raise FileExistsError("Exp411 output directory appeared during staging")
        os.replace(staging, output)
        return dict(sorted(artifact_hashes.items()))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_payload(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


__all__ = ["publish_payloads"]
