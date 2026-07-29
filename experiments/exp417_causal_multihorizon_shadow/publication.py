"""Fail-closed atomic publication for verified Exp417 Shadow payloads."""

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

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .reporting import (
    CONFIG_FILE,
    MANIFEST_FILE,
    NON_MANIFEST_FILES,
    OFFICIAL_FILES,
    REPORT_FILE,
    verify_exp417_payloads,
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


def publish_exp417(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    *,
    pre_publish_verify: Callable[[], None],
) -> Mapping[str, str]:
    """Stage and publish exactly seven verified payloads plus one manifest."""
    output = Path(output_dir)
    if output.name in {"", ".", ".."}:
        raise ValueError("Exp417 output directory name is invalid")
    _require_real_parent(output.parent)
    if _lexists(output):
        raise FileExistsError("Exp417 output directory must not exist")
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != NON_MANIFEST_FILES
        or len(payloads) != 7
        or any(type(name) is not str or type(payload) is not bytes for name, payload in payloads.items())
    ):
        raise ValueError("Exp417 publication requires the exact seven non-manifest payloads")
    if not callable(pre_publish_verify):
        raise ValueError("Exp417 pre-publish verifier must be callable")
    verify_exp417_payloads(payloads)
    artifact_hashes = {
        name: sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    manifest_payload = _render_manifest(payloads, artifact_hashes)

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging_identity: tuple[int, int] | None = None
    staging_descriptor: int | None = None
    output_descriptor: int | None = None
    published = False
    try:
        staging.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(staging, 0o700, follow_symlinks=False)
        staging_descriptor = _open_directory(staging)
        staging_identity = _directory_fd_identity(staging_descriptor)
        if stat.S_IMODE(os.fstat(staging_descriptor).st_mode) != 0o700:
            raise ValueError("Exp417 staging directory mode is not 0700")

        for name, payload in sorted(payloads.items()):
            _write_payload(staging_descriptor, name, payload)
        _write_payload(staging_descriptor, MANIFEST_FILE, manifest_payload)

        _verify_publication_directory(
            staging,
            staging_identity,
            staging_descriptor,
            payloads,
            artifact_hashes,
            manifest_payload,
        )
        pre_publish_verify()
        if _lexists(output):
            raise FileExistsError("Exp417 output directory appeared during staging")

        staged_identities = _verify_publication_directory(
            staging,
            staging_identity,
            staging_descriptor,
            payloads,
            artifact_hashes,
            manifest_payload,
        )
        os.fsync(staging_descriptor)
        if _lexists(output):
            raise FileExistsError("Exp417 output directory appeared during staging")

        _atomic_rename_noreplace(staging, output)
        published = True
        output_descriptor = _open_directory(output)
        if _directory_fd_identity(output_descriptor) != staging_identity:
            raise ValueError("Exp417 published output identity changed")
        _verify_publication_directory(
            output,
            staging_identity,
            output_descriptor,
            payloads,
            artifact_hashes,
            manifest_payload,
            expected_file_identities=staged_identities,
        )
        _fsync_directory(output.parent)
        published = False
        return dict(sorted(artifact_hashes.items()))
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
                RuntimeError("Exp417 published output could not be quarantined")
            )
        if staging_identity is not None:
            try:
                _fsync_directory(output.parent)
            except Exception as cleanup_fsync_error:
                cleanup_errors.append(cleanup_fsync_error)
        if cleanup_errors:
            raise ExceptionGroup(
                "Exp417 publication failed and cleanup also failed",
                [publication_error, *cleanup_errors],
            )
        raise
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def _render_manifest(
    payloads: Mapping[str, bytes], artifact_hashes: Mapping[str, str]
) -> bytes:
    try:
        config = json.loads(payloads[CONFIG_FILE])
        frozen_config_sha256 = config["frozen_config_sha256"]
        provenance = config["provenance"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Exp417 manifest provenance could not be derived") from error
    manifest = {
        "experiment_id": "Exp417",
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "frozen_config_sha256": frozen_config_sha256,
        "report_sha256": sha256(payloads[REPORT_FILE]).hexdigest(),
        "provenance": provenance,
    }
    return canonical_json_bytes(manifest) + b"\n"


def _write_payload(directory_descriptor: int, name: str, payload: bytes) -> None:
    if Path(name).name != name or name not in OFFICIAL_FILES:
        raise ValueError("Exp417 publication payload name is invalid")
    descriptor = os.open(
        name,
        _FILE_WRITE_FLAGS,
        0o600,
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
    *,
    expected_file_identities: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    _verify_directory_path_identity(directory, directory_identity)
    if _directory_fd_identity(directory_descriptor) != directory_identity:
        raise ValueError("Exp417 publication directory descriptor identity changed")
    with os.scandir(directory_descriptor) as scan:
        observed = tuple(entry.name for entry in scan)
    if len(observed) != 8 or set(observed) != OFFICIAL_FILES:
        raise ValueError("Exp417 publication artifact coverage mismatch")

    observed_payloads: dict[str, bytes] = {}
    identities: dict[str, tuple[int, int]] = {}
    for name, expected in sorted(payloads.items()):
        raw, identity = _read_verified_file(
            directory_descriptor,
            name,
            expected,
            artifact_hashes[name],
            expected_identity=(
                None
                if expected_file_identities is None
                else expected_file_identities[name]
            ),
        )
        observed_payloads[name] = raw
        identities[name] = identity
    verify_exp417_payloads(observed_payloads)

    raw_manifest, manifest_identity = _read_verified_file(
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
    if raw_manifest != _render_manifest(observed_payloads, artifact_hashes):
        raise ValueError("Exp417 publication manifest provenance mismatch")
    identities[MANIFEST_FILE] = manifest_identity
    return identities


def _read_verified_file(
    directory_descriptor: int,
    name: str,
    expected_payload: bytes,
    expected_digest: str,
    *,
    expected_identity: tuple[int, int] | None,
) -> tuple[bytes, tuple[int, int]]:
    try:
        before = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"Exp417 publication payload unavailable: {name}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Exp417 publication payload is not a regular file: {name}")
    try:
        descriptor = os.open(name, _FILE_READ_FLAGS, dir_fd=directory_descriptor)
    except OSError as error:
        raise ValueError(
            f"Exp417 publication payload open failed or symlinked: {name}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or identity != (before.st_dev, before.st_ino)
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise ValueError(f"Exp417 publication payload inode identity changed: {name}")
        digest = sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        after = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        )
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"Exp417 publication payload disappeared: {name}") from error
    if (
        not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != identity
        or payload != expected_payload
        or digest.hexdigest() != expected_digest
    ):
        raise ValueError(
            f"Exp417 publication payload content or inode verification failed: {name}"
        )
    return payload, identity


def _remove_owned_path(
    path: Path, identity: tuple[int, int] | None
) -> bool:
    if identity is None or not _lexists(path):
        return False
    quarantine = path.with_name(f".{path.name}.{uuid.uuid4().hex}.quarantine")
    try:
        _atomic_rename_noreplace(path, quarantine)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError(
            f"Exp417 failed to quarantine cleanup candidate: {path}: {error}"
        ) from error
    if not _is_owned_path(quarantine, identity):
        try:
            _atomic_rename_noreplace(quarantine, path)
        except OSError as restore_error:
            raise RuntimeError(
                "Exp417 moved a foreign cleanup candidate and could not restore "
                f"it; preserved at {quarantine}: {restore_error}"
            ) from restore_error
        return False
    if _is_symlink(quarantine):
        # A root symlink can be removed only through a pathname, which admits
        # an unavoidable replacement race.  Keeping the uniquely quarantined
        # link is the fail-closed choice; it no longer occupies the official
        # output or staging name and its target is never traversed.
        return True
    try:
        descriptor = os.open(quarantine, _DIRECTORY_OPEN_FLAGS)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != identity
            ):
                raise RuntimeError("Exp417 quarantined cleanup inode identity changed")
            # POSIX has no portable unlink-if-inode-matches primitive.  Never
            # recurse or unlink names in a directory another process may still
            # hold open: stat→unlink would be a TOCTOU deletion of a foreign
            # replacement.  The owned root remains at its unique quarantine
            # name after descriptor/inode verification.
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception as error:
        raise RuntimeError(
            f"Exp417 quarantined owned cleanup failed at {quarantine}: {error}"
        ) from error
    return True


def _is_owned_path(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISDIR(metadata.st_mode):
        return (metadata.st_dev, metadata.st_ino) == identity
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = path.stat()
        except (FileNotFoundError, OSError):
            return False
        return stat.S_ISDIR(target.st_mode) and (
            target.st_dev, target.st_ino
        ) == identity
    return False


def _is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    source_bytes = os.fsencode(os.path.abspath(source))
    destination_bytes = os.fsencode(os.path.abspath(destination))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(source_bytes, destination_bytes, _RENAME_EXCL) != 0:
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
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(
            _LINUX_AT_FDCWD,
            source_bytes,
            _LINUX_AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        ) != 0:
            _raise_rename_error(destination)
        return
    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is unavailable",
        str(destination),
    )


def _raise_rename_error(destination: Path) -> None:
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            number,
            "Exp417 output directory appeared during atomic rename",
            str(destination),
        )
    raise OSError(number, os.strerror(number), str(destination))


def _open_directory(path: Path) -> int:
    try:
        descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(
            "Exp417 staging root or published output must be a real directory"
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
        raise ValueError("Exp417 publication descriptor is not a directory")
    return (metadata.st_dev, metadata.st_ino)


def _verify_directory_path_identity(
    path: Path, expected_identity: tuple[int, int]
) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("Exp417 staging root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or (
        metadata.st_dev, metadata.st_ino
    ) != expected_identity:
        raise ValueError("Exp417 staging root identity changed")


def _require_real_parent(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError("Exp417 output parent does not exist") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Exp417 output parent must be a real directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = ["publish_exp417"]
