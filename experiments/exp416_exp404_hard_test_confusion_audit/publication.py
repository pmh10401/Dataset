"""Fail-closed atomic publication for verified Exp416 audit payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

from .config import Exp416FrozenConfig, FROZEN_CONFIG
from .reporting import (
    DATASET_FILE,
    MANIFEST_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    verify_rendered_payloads,
)


_EXPECTED_PAYLOAD_FILES = frozenset(
    {SUMMARY_FILE, DATASET_FILE, REPORT_FILE}
)


def publish_payloads(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    provenance: Mapping[str, object],
    *,
    pre_publish_verify: Callable[[], None] | None = None,
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, str]:
    """Stage, revalidate, and atomically publish three payloads plus manifest."""
    config.validate()
    output = Path(output_dir)
    if output.name in {"", ".", ".."}:
        raise ValueError("Exp416 output directory name is invalid")
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp416 output parent does not exist")
    if _lexists(output):
        raise FileExistsError("Exp416 output directory must not exist")
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != _EXPECTED_PAYLOAD_FILES
        or len(payloads) != 3
    ):
        raise ValueError(
            "Exp416 publication requires exact three non-manifest payloads"
        )
    if pre_publish_verify is not None and not callable(pre_publish_verify):
        raise ValueError("Exp416 pre-publish verifier must be callable")
    if not isinstance(provenance, Mapping):
        raise ValueError("Exp416 publication provenance must be a mapping")
    verify_rendered_payloads(payloads, config=config)

    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    published = False
    staging_identity: tuple[int, int] | None = None
    try:
        staging.mkdir(parents=False, exist_ok=False)
        staging_identity = _directory_identity(staging)
        artifact_hashes: dict[str, str] = {}
        for name, payload in sorted(payloads.items()):
            if (
                type(name) is not str
                or not name
                or Path(name).name != name
                or type(payload) is not bytes
            ):
                raise ValueError("Exp416 staged payload is invalid")
            path = staging / name
            _write_payload(path, payload)
            digest = sha256(payload).hexdigest()
            _verify_file(path, payload, digest, name)
            artifact_hashes[name] = digest

        manifest = {
            "experiment_id": config.experiment_id,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "provenance": _plain(provenance),
        }
        manifest_payload = _json_bytes(manifest)
        _write_payload(staging / MANIFEST_FILE, manifest_payload)
        _verify_staging(
            staging,
            staging_identity,
            payloads,
            artifact_hashes,
            manifest_payload,
            config,
        )

        if pre_publish_verify is not None:
            pre_publish_verify()
        if _lexists(output):
            raise FileExistsError(
                "Exp416 output directory appeared during staging"
            )

        _fsync_directory(staging)
        _verify_staging(
            staging,
            staging_identity,
            payloads,
            artifact_hashes,
            manifest_payload,
            config,
        )
        if _lexists(output):
            raise FileExistsError(
                "Exp416 output directory appeared during staging"
            )

        result = dict(sorted(artifact_hashes.items()))
        os.replace(staging, output)
        published = True
        if _directory_identity(output) != staging_identity:
            raise ValueError("Exp416 published output identity changed")
        _fsync_directory(output.parent)
        published = False
        return result
    except Exception:
        if _is_owned_directory(output, staging_identity):
            _rollback_publication(output, staging, staging_identity)
        elif published:
            raise RuntimeError(
                "Exp416 publication rollback could not identify its output"
            )
        _remove_staging(staging)
        raise


def _write_payload(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_staging(
    staging: Path,
    staging_identity: tuple[int, int],
    payloads: Mapping[str, bytes],
    artifact_hashes: Mapping[str, str],
    manifest_payload: bytes,
    config: Exp416FrozenConfig,
) -> None:
    if _is_symlink(staging) or not _is_real_directory(staging):
        raise ValueError("Exp416 staging root must be a real directory")
    if _directory_identity(staging) != staging_identity:
        raise ValueError("Exp416 staging root identity changed")
    expected = _EXPECTED_PAYLOAD_FILES | {MANIFEST_FILE}
    observed = {entry.name for entry in os.scandir(staging)}
    if observed != expected:
        raise ValueError("Exp416 staged artifact coverage mismatch")

    staged_payloads: dict[str, bytes] = {}
    for name, payload in sorted(payloads.items()):
        path = staging / name
        _verify_file(path, payload, artifact_hashes[name], name)
        staged_payloads[name] = path.read_bytes()
    verify_rendered_payloads(staged_payloads, config=config)

    manifest_path = staging / MANIFEST_FILE
    _verify_file(
        manifest_path,
        manifest_payload,
        sha256(manifest_payload).hexdigest(),
        MANIFEST_FILE,
    )
    try:
        observed_manifest = json.loads(manifest_path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Exp416 staged manifest is invalid") from error
    if _json_bytes(observed_manifest) != manifest_payload:
        raise ValueError("Exp416 staged manifest is not canonical")


def _verify_file(
    path: Path,
    expected_payload: bytes,
    expected_digest: str,
    name: str,
) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(
            f"Exp416 staged payload verification failed: {name}"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.read_bytes() != expected_payload
        or _sha256_file(path) != expected_digest
    ):
        raise ValueError(
            f"Exp416 staged payload verification failed or symlinked: {name}"
        )


def _rollback_publication(
    output: Path,
    staging: Path,
    staging_identity: tuple[int, int] | None,
) -> None:
    if (
        staging_identity is None
        or not _is_real_directory(output)
        or _directory_identity(output) != staging_identity
        or _lexists(staging)
    ):
        raise RuntimeError(
            "Exp416 publication rollback could not identify its output"
        )
    os.replace(output, staging)


def _is_owned_directory(
    path: Path,
    identity: tuple[int, int] | None,
) -> bool:
    return (
        identity is not None
        and _is_real_directory(path)
        and _directory_identity(path) == identity
    )


def _remove_staging(staging: Path) -> None:
    if not _lexists(staging):
        return
    if _is_symlink(staging):
        staging.unlink()
        return
    shutil.rmtree(staging)


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Exp416 staging root must be a real directory")
    return (metadata.st_dev, metadata.st_ino)


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Exp416 provenance keys must be strings")
        return {
            key: _plain(child)
            for key, child in sorted(value.items())
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
