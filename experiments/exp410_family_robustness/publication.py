"""Fail-closed atomic directory publication for Exp410."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import os
from pathlib import Path
import shutil
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file

from .config import FROZEN_CONFIG, Exp410FrozenConfig
from .contracts import AuditResult
from .reporting import MANIFEST_FILE, render_payloads


def publish_audit(
    result: AuditResult,
    output_dir: Path,
    provenance: Mapping[str, object],
    config: Exp410FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, str]:
    """Publish a complete audit directory or leave no output directory."""
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp410 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp410 output directory already exists")

    payloads = render_payloads(result, provenance, config=config)
    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        artifact_hashes: dict[str, str] = {}
        for name, payload in sorted(payloads.items()):
            path = staging / name
            _write_payload(path, payload)
            digest = sha256(payload).hexdigest()
            if sha256_file(path) != digest or path.read_bytes() != payload:
                raise ValueError(f"Exp410 staged artifact verification failed: {name}")
            artifact_hashes[name] = digest

        manifest = {
            "experiment_id": config.experiment_id,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "provenance": _plain(provenance),
        }
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        manifest_path = staging / MANIFEST_FILE
        _write_payload(manifest_path, manifest_payload)
        manifest_hash = sha256(manifest_payload).hexdigest()
        if (
            sha256_file(manifest_path) != manifest_hash
            or manifest_path.read_bytes() != manifest_payload
        ):
            raise ValueError("Exp410 staged manifest verification failed")

        expected_names = set(payloads) | {MANIFEST_FILE}
        if {path.name for path in staging.iterdir()} != expected_names:
            raise ValueError("Exp410 staged artifact coverage mismatch")
        if output.exists():
            raise FileExistsError("Exp410 output directory appeared during staging")
        os.replace(staging, output)
        return dict(
            sorted({**artifact_hashes, MANIFEST_FILE: manifest_hash}.items())
        )
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
