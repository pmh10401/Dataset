"""Fail-closed atomic publication for verified Exp416 audit payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
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
_PROVENANCE_FIELDS = frozenset(
    {
        "database_reference_sha256",
        "exp415_prediction_created",
        "frozen_test_labels_sha256",
        "git_commit",
        "operational_policy_changed",
        "prediction_identity_sha256",
        "prediction_manifest_sha256",
        "prediction_modified",
        "prediction_sha256",
        "test_informed_reselection",
        "test_labels_loaded_after_prediction_freeze",
    }
)
_LINUX_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
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
    verified_provenance = _verified_summary_provenance(payloads, provenance)

    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    published = False
    staging_identity: tuple[int, int] | None = None
    staging_descriptor: int | None = None
    output_descriptor: int | None = None
    try:
        staging.mkdir(parents=False, exist_ok=False)
        staging_descriptor = _open_directory(staging)
        staging_identity = _directory_fd_identity(staging_descriptor)
        artifact_hashes: dict[str, str] = {}
        for name, payload in sorted(payloads.items()):
            if (
                type(name) is not str
                or not name
                or Path(name).name != name
                or type(payload) is not bytes
            ):
                raise ValueError("Exp416 staged payload is invalid")
            _write_payload(staging_descriptor, name, payload)
            digest = sha256(payload).hexdigest()
            artifact_hashes[name] = digest

        manifest = {
            "experiment_id": config.experiment_id,
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
            "provenance": verified_provenance,
        }
        manifest_payload = _json_bytes(manifest)
        _write_payload(staging_descriptor, MANIFEST_FILE, manifest_payload)
        _verify_publication_directory(
            staging,
            staging_identity,
            staging_descriptor,
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

        os.fsync(staging_descriptor)
        staged_file_identities = _verify_publication_directory(
            staging,
            staging_identity,
            staging_descriptor,
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
        _atomic_rename_noreplace(staging, output)
        published = True
        output_descriptor = _open_directory(output)
        if _directory_fd_identity(output_descriptor) != staging_identity:
            raise ValueError("Exp416 published output identity changed")
        _verify_publication_directory(
            output,
            staging_identity,
            output_descriptor,
            payloads,
            artifact_hashes,
            manifest_payload,
            config,
            expected_file_identities=staged_file_identities,
        )
        _fsync_directory(output.parent)
        published = False
        return result
    except Exception as publication_error:
        if output_descriptor is not None:
            os.close(output_descriptor)
            output_descriptor = None
        if staging_descriptor is not None:
            os.close(staging_descriptor)
            staging_descriptor = None
        cleanup_errors: list[Exception] = []
        owned_output_removed = False
        for candidate in (output, staging):
            try:
                removed = _remove_owned_path(candidate, staging_identity)
                if candidate == output:
                    owned_output_removed = removed
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if published and not owned_output_removed and not cleanup_errors:
            cleanup_errors.append(
                RuntimeError(
                    "Exp416 published output could not be quarantined"
                )
            )
        if cleanup_errors:
            raise ExceptionGroup(
                "Exp416 publication failed and cleanup also failed",
                [publication_error, *cleanup_errors],
            )
        raise
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def _write_payload(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    descriptor = os.open(
        name,
        _FILE_WRITE_FLAGS,
        0o666,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_publication_directory(
    directory: Path,
    directory_identity: tuple[int, int],
    directory_descriptor: int,
    payloads: Mapping[str, bytes],
    artifact_hashes: Mapping[str, str],
    manifest_payload: bytes,
    config: Exp416FrozenConfig,
    *,
    expected_file_identities: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    _verify_directory_path_identity(directory, directory_identity)
    if _directory_fd_identity(directory_descriptor) != directory_identity:
        raise ValueError("Exp416 publication directory descriptor changed")
    expected = _EXPECTED_PAYLOAD_FILES | {MANIFEST_FILE}
    with os.scandir(directory_descriptor) as entries:
        observed = tuple(entry.name for entry in entries)
    if len(observed) != 4 or set(observed) != expected:
        raise ValueError("Exp416 publication artifact coverage mismatch")

    staged_payloads: dict[str, bytes] = {}
    file_identities: dict[str, tuple[int, int]] = {}
    for name, payload in sorted(payloads.items()):
        observed_payload, identity = _read_verified_file(
            directory_descriptor,
            name,
            payload,
            artifact_hashes[name],
            expected_identity=(
                None
                if expected_file_identities is None
                else expected_file_identities[name]
            ),
        )
        staged_payloads[name] = observed_payload
        file_identities[name] = identity
    verify_rendered_payloads(staged_payloads, config=config)

    observed_manifest_payload, manifest_identity = _read_verified_file(
        directory_descriptor,
        MANIFEST_FILE,
        manifest_payload,
        sha256(manifest_payload).hexdigest(),
        expected_identity=(
            None
            if expected_file_identities is None
            else expected_file_identities[MANIFEST_FILE]
        ),
    )
    file_identities[MANIFEST_FILE] = manifest_identity
    try:
        observed_manifest = json.loads(observed_manifest_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Exp416 publication manifest is invalid") from error
    if _json_bytes(observed_manifest) != manifest_payload:
        raise ValueError("Exp416 publication manifest is not canonical")
    return file_identities


def _read_verified_file(
    directory_descriptor: int,
    name: str,
    expected_payload: bytes,
    expected_digest: str,
    *,
    expected_identity: tuple[int, int] | None,
) -> tuple[bytes, tuple[int, int]]:
    try:
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            f"Exp416 publication payload verification failed: {name}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            "Exp416 publication payload must be a regular file, not a "
            f"symlink: {name}"
        )
    try:
        descriptor = os.open(
            name,
            _FILE_READ_FLAGS,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ValueError(
            "Exp416 publication payload verification failed or symlinked: "
            f"{name}"
        ) from error
    try:
        opened_metadata = os.fstat(descriptor)
        identity = (opened_metadata.st_dev, opened_metadata.st_ino)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or identity != (metadata.st_dev, metadata.st_ino)
            or (
                expected_identity is not None
                and identity != expected_identity
            )
        ):
            raise ValueError(
                "Exp416 publication payload inode or type changed: "
                f"{name}"
            )
        digest = sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
            digest.update(chunk)
        observed_payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        current_metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            f"Exp416 publication payload verification failed: {name}"
        ) from error
    if (
        not stat.S_ISREG(current_metadata.st_mode)
        or (current_metadata.st_dev, current_metadata.st_ino) != identity
        or observed_payload != expected_payload
        or digest.hexdigest() != expected_digest
    ):
        raise ValueError(
            "Exp416 publication payload verification failed or symlinked: "
            f"{name}"
        )
    return observed_payload, identity


def _is_owned_path(
    path: Path,
    identity: tuple[int, int] | None,
) -> bool:
    if identity is None or not _lexists(path):
        return False
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        return (metadata.st_dev, metadata.st_ino) == identity
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = path.stat()
        except (FileNotFoundError, OSError):
            return False
        return stat.S_ISDIR(target.st_mode) and (
            target.st_dev,
            target.st_ino,
        ) == identity
    return False


def _remove_owned_path(
    path: Path,
    identity: tuple[int, int] | None,
) -> bool:
    if identity is None or not _lexists(path):
        return False
    quarantine = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.quarantine"
    )
    try:
        _atomic_rename_noreplace(path, quarantine)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError(
            f"Exp416 failed to quarantine cleanup candidate: {path}: {error}"
        ) from error
    if not _is_owned_path(quarantine, identity):
        try:
            _atomic_rename_noreplace(quarantine, path)
        except OSError as restore_error:
            raise RuntimeError(
                "Exp416 moved a foreign cleanup candidate and could not "
                f"restore it; preserved at {quarantine}: {restore_error}"
            ) from restore_error
        return False
    if _is_symlink(quarantine):
        try:
            quarantine.unlink()
        except OSError as error:
            raise RuntimeError(
                "Exp416 failed to remove quarantined owned symlink: "
                f"{quarantine}: {error}"
            ) from error
        return True
    try:
        descriptor = os.open(quarantine, _DIRECTORY_OPEN_FLAGS)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != identity
            ):
                raise RuntimeError(
                    "Exp416 quarantined cleanup inode identity changed"
                )
            _clear_owned_directory_fd(descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        quarantine.rmdir()
    except Exception as error:
        raise RuntimeError(
            f"Exp416 quarantined owned cleanup failed at {quarantine}: {error}"
        ) from error
    return True


def _clear_owned_directory_fd(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        try:
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            try:
                child = os.open(
                    name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError(
                    f"Exp416 failed to open quarantined child: {name}: {error}"
                ) from error
            try:
                child_metadata = os.fstat(child)
                child_identity = (
                    child_metadata.st_dev,
                    child_metadata.st_ino,
                )
                if (
                    not stat.S_ISDIR(child_metadata.st_mode)
                    or child_identity
                    != (metadata.st_dev, metadata.st_ino)
                ):
                    continue
                _clear_owned_directory_fd(child)
                os.fsync(child)
            finally:
                os.close(child)
            try:
                current = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if (
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == child_identity
            ):
                try:
                    os.rmdir(name, dir_fd=descriptor)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise RuntimeError(
                        "Exp416 failed to remove quarantined child directory: "
                        f"{name}: {error}"
                    ) from error
            continue
        try:
            current = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
        ):
            continue
        try:
            os.unlink(name, dir_fd=descriptor)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"Exp416 failed to unlink quarantined child: {name}: {error}"
            ) from error


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Rename exclusively, failing closed where no safe primitive exists."""
    source_bytes = os.fsencode(os.path.abspath(source))
    destination_bytes = os.fsencode(os.path.abspath(destination))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renamex_np
        rename.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_bytes,
            destination_bytes,
            _RENAME_EXCL,
        )
        if result != 0:
            _raise_rename_error(destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable",
                str(destination),
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            _LINUX_AT_FDCWD,
            source_bytes,
            _LINUX_AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
        if result != 0:
            _raise_rename_error(destination)
        return
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is unavailable",
        str(destination),
    )


def _raise_rename_error(destination: Path) -> None:
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "Exp416 output directory appeared during atomic rename",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _open_directory(path: Path) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(
            "Exp416 staging root or published output must be a real directory"
        ) from error
    try:
        _directory_fd_identity(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _directory_fd_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(
            "Exp416 publication directory descriptor is not a directory"
        )
    return (metadata.st_dev, metadata.st_ino)


def _verify_directory_path_identity(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            "Exp416 staging root or published output is unavailable"
        ) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise ValueError(
            "Exp416 staging root or published output identity changed"
        )


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verified_summary_provenance(
    payloads: Mapping[str, bytes],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    try:
        summary = json.loads(payloads[SUMMARY_FILE])
        inputs = summary["input_sha256"]
        expected = {
            "database_reference_sha256": inputs["database_reference"],
            "exp415_prediction_created": summary[
                "exp415_prediction_created"
            ],
            "frozen_test_labels_sha256": inputs["frozen_test_labels"],
            "git_commit": summary["git_commit"],
            "operational_policy_changed": summary[
                "operational_policy_changed"
            ],
            "prediction_identity_sha256": summary[
                "prediction_identity_sha256"
            ],
            "prediction_manifest_sha256": inputs["manifest"],
            "prediction_modified": summary["prediction_modified"],
            "prediction_sha256": inputs["prediction"],
            "test_informed_reselection": summary[
                "test_informed_reselection"
            ],
            "test_labels_loaded_after_prediction_freeze": summary[
                "test_labels_loaded_after_prediction_freeze"
            ],
        }
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            "Exp416 verified summary provenance could not be derived"
        ) from error
    if (
        set(provenance) != _PROVENANCE_FIELDS
        or len(provenance) != len(_PROVENANCE_FIELDS)
    ):
        raise ValueError("Exp416 publication provenance schema mismatch")
    for name, expected_value in expected.items():
        observed_value = provenance[name]
        if (
            type(observed_value) is not type(expected_value)
            or observed_value != expected_value
        ):
            raise ValueError(
                f"Exp416 publication provenance mismatch: {name}"
            )
    return dict(sorted(expected.items()))


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
