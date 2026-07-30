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

from .checkpoints import Exp417CheckpointStore
from .reporting import (
    CONFIG_FILE,
    MANIFEST_FILE,
    NON_MANIFEST_FILES,
    OFFICIAL_FILES,
    REPORT_FILE,
    TrustedCheckpointProvenance,
    verify_exp417_payloads,
)


_LINUX_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_RENAME_SWAP = 0x00000002
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
    checkpoints: Exp417CheckpointStore,
    pre_publish_verify: Callable[[], None],
) -> Mapping[str, str]:
    """Stage and publish exactly seven verified payloads plus one manifest."""
    output = Path(output_dir)
    if output.name in {"", ".", ".."}:
        raise ValueError("Exp417 output directory name is invalid")
    _require_real_parent(output.parent)
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != NON_MANIFEST_FILES
        or len(payloads) != 7
        or any(type(name) is not str or type(payload) is not bytes for name, payload in payloads.items())
    ):
        raise ValueError("Exp417 publication requires the exact seven non-manifest payloads")
    if not callable(pre_publish_verify):
        raise ValueError("Exp417 pre-publish verifier must be callable")
    trusted_checkpoint = TrustedCheckpointProvenance.from_store(checkpoints)
    _require_disjoint_checkpoint_output(checkpoints.root, output)
    if _lexists(output):
        raise FileExistsError("Exp417 output directory must not exist")
    _require_atomic_exchange(output.parent)
    verify_exp417_payloads(
        payloads, trusted_checkpoint=trusted_checkpoint
    )
    artifact_hashes = {
        name: sha256(payload).hexdigest()
        for name, payload in sorted(payloads.items())
    }
    manifest_payload = _render_manifest(
        payloads, artifact_hashes, trusted_checkpoint
    )

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
            trusted_checkpoint,
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
            trusted_checkpoint,
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
            trusted_checkpoint,
            expected_file_identities=staged_identities,
        )
        _fsync_directory(output.parent)
        published = False
        return dict(sorted(artifact_hashes.items()))
    except BaseException as publication_error:
        cleanup_errors: list[BaseException] = []
        if output_descriptor is not None:
            try:
                os.close(output_descriptor)
            except BaseException as close_error:
                cleanup_errors.append(close_error)
            finally:
                output_descriptor = None
        if staging_descriptor is not None:
            try:
                os.close(staging_descriptor)
            except BaseException as close_error:
                cleanup_errors.append(close_error)
            finally:
                staging_descriptor = None
        owned_output_removed = False
        for candidate in (output, staging):
            try:
                removed = _remove_owned_path(candidate, staging_identity)
                if candidate == output:
                    owned_output_removed = removed
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if published and not owned_output_removed and not cleanup_errors:
            cleanup_errors.append(
                RuntimeError("Exp417 published output could not be quarantined")
            )
        if staging_identity is not None:
            try:
                _fsync_directory(output.parent)
            except BaseException as cleanup_fsync_error:
                cleanup_errors.append(cleanup_fsync_error)
        if cleanup_errors:
            raise BaseExceptionGroup(
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
    payloads: Mapping[str, bytes],
    artifact_hashes: Mapping[str, str],
    trusted_checkpoint: TrustedCheckpointProvenance,
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
        "trusted_checkpoint_provenance_sha256":
            trusted_checkpoint.checkpoint_provenance_sha256,
        "trusted_result_map_sha256": sha256(
            canonical_json_bytes(dict(trusted_checkpoint.result_evidence_sha256))
        ).hexdigest(),
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
    trusted_checkpoint: TrustedCheckpointProvenance,
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
    verify_exp417_payloads(
        observed_payloads, trusted_checkpoint=trusted_checkpoint
    )

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
    if raw_manifest != _render_manifest(
        observed_payloads, artifact_hashes, trusted_checkpoint
    ):
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
    capture = path.with_name(f".{path.name}.{uuid.uuid4().hex}.capture")
    capture.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(capture, 0o700, follow_symlinks=False)
    placeholder = capture.lstat()
    placeholder_identity = (placeholder.st_dev, placeholder.st_ino)
    try:
        _atomic_exchange(path, capture)
    except FileNotFoundError:
        # The empty owned capture slot is intentionally retained: deleting it
        # by pathname would recreate the same conditional-unlink problem.
        return False
    except OSError as error:
        raise RuntimeError(
            f"Exp417 failed to atomically capture cleanup candidate: {path}: {error}"
        ) from error

    captured = capture.lstat()
    captured_identity = (captured.st_dev, captured.st_ino)
    if not stat.S_ISDIR(captured.st_mode) or captured_identity != identity:
        # Restore the exact atomically captured foreign entry.  If another
        # foreign owner replaced the placeholder at `path` before this swap,
        # the first entry returns to its official name and the second remains
        # intact in the unique capture slot.  Nothing is deleted or mistaken
        # for the owned inode.
        try:
            _atomic_exchange(path, capture)
        except (FileNotFoundError, OSError) as restore_error:
            raise RuntimeError(
                "Exp417 captured a foreign cleanup candidate and could not "
                f"atomically restore it; preserved at {capture}: {restore_error}"
            ) from restore_error
        conflict = capture.lstat()
        conflict_identity = (conflict.st_dev, conflict.st_ino)
        if (
            not stat.S_ISDIR(conflict.st_mode)
            or conflict_identity != placeholder_identity
        ):
            raise RuntimeError(
                "Exp417 cleanup ownership mismatch; first foreign entry was "
                f"restored and a second foreign entry is preserved at {capture}"
            )
        raise RuntimeError(
            f"Exp417 cleanup path was foreign and was atomically restored: {path}"
        )

    # The owned directory is now durably quarantined at `capture`; the
    # official name contains only our known empty placeholder.  Move whatever
    # currently occupies that name to a unique release slot atomically.  If a
    # foreign replacement raced in, inspect it after capture and restore it to
    # the now-empty official name when possible.
    release = path.with_name(f".{path.name}.{uuid.uuid4().hex}.released")
    try:
        _atomic_rename_noreplace(path, release)
    except OSError as error:
        raise RuntimeError(
            f"Exp417 failed to release owned cleanup placeholder: {path}: {error}"
        ) from error
    released = release.lstat()
    released_identity = (released.st_dev, released.st_ino)
    if (
        not stat.S_ISDIR(released.st_mode)
        or released_identity != placeholder_identity
    ):
        try:
            _atomic_rename_noreplace(release, path)
        except OSError as restore_error:
            raise RuntimeError(
                "Exp417 captured a racing foreign replacement; it remains "
                f"preserved at {release} because its official name is occupied: "
                f"{restore_error}"
            ) from restore_error
        raise RuntimeError(
            "Exp417 captured and restored a racing foreign replacement while "
            "quarantining the verified owned directory"
        )

    quarantine = capture
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


def _require_atomic_exchange(parent: Path) -> None:
    """Prove atomic exchange on the publication filesystem before staging."""
    probe_root = (
        parent / f".exp417.exchange-probe.retained.{uuid.uuid4().hex}"
    )
    probe_a = probe_root / "a"
    probe_b = probe_root / "b"
    probe_root_identity: tuple[int, int] | None = None
    identity_a: tuple[int, int] | None = None
    identity_b: tuple[int, int] | None = None
    child_provenance: dict[str, dict[str, object]] = {
        name: {
            "name": name,
            "state": "not_created",
            "identity": None,
            "error_type": None,
            "error_errno": None,
            "error_message": None,
        }
        for name in ("a", "b")
    }
    probe_descriptor: int | None = None
    probe_error: Exception | None = None
    retention_errors: list[Exception] = []
    exchange_verified = False
    try:
        probe_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(probe_root, 0o700, follow_symlinks=False)
        root_metadata = probe_root.lstat()
        probe_root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        probe_descriptor = os.open(probe_root, _DIRECTORY_OPEN_FLAGS)
        if _directory_fd_identity(probe_descriptor) != probe_root_identity:
            raise RuntimeError("Exp417 atomic-exchange probe root identity changed")

        identities: list[tuple[int, int]] = []
        for name in ("a", "b"):
            try:
                os.mkdir(name, mode=0o700, dir_fd=probe_descriptor)
                child_provenance[name]["state"] = "created_unresolved"
                child_descriptor: int | None = None
                try:
                    child_descriptor = os.open(
                        name, _DIRECTORY_OPEN_FLAGS, dir_fd=probe_descriptor
                    )
                    metadata = os.fstat(child_descriptor)
                    identity = (metadata.st_dev, metadata.st_ino)
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise RuntimeError(
                            f"Exp417 atomic-exchange probe child is not a "
                            f"directory: {name}"
                        )
                    if identity in identities:
                        raise RuntimeError(
                            "Exp417 atomic-exchange probe inode identities "
                            "collided"
                        )
                    os.fsync(child_descriptor)
                    if name == "a":
                        identity_a = identity
                    else:
                        identity_b = identity
                    identities.append(identity)
                    child_provenance[name]["identity"] = identity
                    child_provenance[name]["state"] = "descriptor_verified"
                finally:
                    if child_descriptor is not None:
                        os.close(child_descriptor)
            except Exception as child_error:
                child_provenance[name]["error_type"] = type(child_error).__name__
                child_provenance[name]["error_errno"] = (
                    child_error.errno
                    if isinstance(child_error, OSError)
                    else None
                )
                child_provenance[name]["error_message"] = str(child_error)
                raise
        os.fsync(probe_descriptor)
        _fsync_directory(parent)

        if (_path_identity(probe_a), _path_identity(probe_b)) != (
            identity_a,
            identity_b,
        ):
            raise RuntimeError(
                "Exp417 atomic-exchange probe identities changed before swap"
            )
        _atomic_exchange(probe_a, probe_b)
        if (
            _path_identity(probe_a),
            _path_identity(probe_b),
        ) != (identity_b, identity_a):
            raise RuntimeError(
                "Exp417 atomic-exchange capability probe did not swap inodes"
            )
        _atomic_exchange(probe_a, probe_b)
        if (
            _path_identity(probe_a),
            _path_identity(probe_b),
        ) != (identity_a, identity_b):
            raise RuntimeError(
                "Exp417 atomic-exchange capability probe did not restore inodes"
            )
        exchange_verified = True
        os.fsync(probe_descriptor)
        _fsync_directory(parent)
    except Exception as error:
        probe_error = error
    finally:
        if probe_descriptor is not None:
            try:
                _write_probe_provenance(
                    probe_descriptor,
                    probe_root_identity=probe_root_identity,
                    identity_a=identity_a,
                    identity_b=identity_b,
                    child_provenance=child_provenance,
                    exchange_verified=exchange_verified,
                    probe_error=probe_error,
                )
                os.fsync(probe_descriptor)
            except Exception as retention_error:
                retention_errors.append(retention_error)
            try:
                os.close(probe_descriptor)
            except OSError as retention_error:
                retention_errors.append(retention_error)
        try:
            _fsync_directory(parent)
        except Exception as retention_error:
            retention_errors.append(retention_error)

    if retention_errors:
        errors = ([probe_error] if probe_error is not None else []) + retention_errors
        raise ExceptionGroup(
            "Exp417 retained atomic-exchange probe finalization failed", errors
        )
    if probe_error is not None:
        raise probe_error


def _write_probe_provenance(
    probe_descriptor: int,
    *,
    probe_root_identity: tuple[int, int] | None,
    identity_a: tuple[int, int] | None,
    identity_b: tuple[int, int] | None,
    child_provenance: Mapping[str, Mapping[str, object]],
    exchange_verified: bool,
    probe_error: Exception | None,
) -> None:
    payload = canonical_json_bytes(
        {
            "schema_version": 2,
            "experiment_id": "Exp417",
            "purpose": "atomic_exchange_filesystem_capability_probe",
            "capsule_identity": probe_root_identity,
            "probe_inode_provenance": {
                "a": identity_a,
                "b": identity_b,
            },
            "probe_child_provenance": {
                name: dict(child_provenance[name])
                for name in ("a", "b")
            },
            "atomic_exchange_supported": exchange_verified,
            "operation_errno": (
                probe_error.errno
                if isinstance(probe_error, OSError)
                else None
            ),
            "operation_error_type": (
                type(probe_error).__name__ if probe_error is not None else None
            ),
            "automatic_cleanup": False,
            "cleanup_policy": "manual_offline_identity_verification_only",
        }
    ) + b"\n"
    descriptor = os.open(
        "PROBE_PROVENANCE.json",
        _FILE_WRITE_FLAGS,
        0o600,
        dir_fd=probe_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Exp417 expected a probe directory: {path}")
    return metadata.st_dev, metadata.st_ino


def _atomic_exchange(first: Path, second: Path) -> None:
    """Atomically swap two directory entries without an ownership check gap."""
    first_bytes = os.fsencode(os.path.abspath(first))
    second_bytes = os.fsencode(os.path.abspath(second))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            exchange = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic directory exchange is unavailable"
            ) from error
        exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        if exchange(first_bytes, second_bytes, _RENAME_SWAP) != 0:
            _raise_exchange_error(first)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            exchange = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic directory exchange is unavailable"
            ) from error
        exchange.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint,
        ]
        exchange.restype = ctypes.c_int
        if exchange(
            _LINUX_AT_FDCWD,
            first_bytes,
            _LINUX_AT_FDCWD,
            second_bytes,
            _RENAME_EXCHANGE,
        ) != 0:
            _raise_exchange_error(first)
        return
    raise OSError(errno.ENOTSUP, "atomic directory exchange is unavailable")


def _raise_exchange_error(path: Path) -> None:
    number = ctypes.get_errno()
    if number == errno.ENOENT:
        raise FileNotFoundError(number, os.strerror(number), str(path))
    raise OSError(number, os.strerror(number), str(path))


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


def _require_disjoint_checkpoint_output(
    checkpoint_root: Path, output_dir: Path
) -> None:
    checkpoint = Path(checkpoint_root).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    if (
        checkpoint == output
        or checkpoint in output.parents
        or output in checkpoint.parents
    ):
        raise ValueError(
            "Exp417 checkpoint and output paths must have disjoint ancestry"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


__all__ = ["publish_exp417"]
