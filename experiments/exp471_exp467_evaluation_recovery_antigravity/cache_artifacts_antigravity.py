"""Descriptor-bound create-only persistence for Exp471 score-cache artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat

from .config_antigravity import METHOD_ORDER_ANTIGRAVITY
from .contracts_antigravity import (
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
    canonical_json_bytes_exp471_antigravity,
)


_SCORE_SHARD_DIRECTORY_ANTIGRAVITY = "score_shards_antigravity"
_CACHE_MANIFEST_NAME_ANTIGRAVITY = "score_cache_manifest_antigravity.json"
_SCORE_SHARD_SCHEMA_ANTIGRAVITY = "exp471-score-cache-shard-antigravity.v2"
_CACHE_MANIFEST_SCHEMA_ANTIGRAVITY = (
    "exp471-score-cache-manifest-antigravity.v2"
)
_CACHE_MANIFEST_RECOVERY_SCHEMA_ANTIGRAVITY = (
    "exp471-score-cache-manifest-recovery-antigravity.v1"
)
_CACHE_MANIFEST_RECOVERY_PREFIX_ANTIGRAVITY = (
    "score_cache_manifest_recovery_"
)
_CACHE_MANIFEST_RECOVERY_SUFFIX_ANTIGRAVITY = "_antigravity"
_CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY = (
    "score_cache_manifest_quarantined_antigravity.json"
)
_CACHE_MANIFEST_RECOVERY_METADATA_NAME_ANTIGRAVITY = (
    "score_cache_manifest_recovery_metadata_antigravity.json"
)
if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
    raise RuntimeError("Exp471 score cache requires descriptor-relative no-follow support")
_READ_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _race_hook_antigravity(event: str, relative: str) -> None:
    """No-op seam for deterministic descriptor-swap lifecycle tests."""


def _same_identity_antigravity(
    left: os.stat_result, right: os.stat_result
) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _safe_leaf_name_antigravity(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
        and value not in {".", ".."}
    )


@dataclass(slots=True)
class _DirectoryAnchorAntigravity:
    descriptor: int
    descriptors: list[int]
    validators: list[tuple[int, str, os.stat_result]]

    def validate(self) -> None:
        for parent, name, expected in self.validators:
            try:
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as error:
                raise ValueError("Exp471 cache directory path changed") from error
            if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(
                expected, observed
            ):
                raise ValueError("Exp471 cache directory path changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_absolute_directory_antigravity(
    path: Path, *, create: bool
) -> _DirectoryAnchorAntigravity:
    path = Path(path)
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or PurePosixPath(path.as_posix()).is_absolute() is False
    ):
        raise ValueError("Exp471 cache root must be absolute and non-escaping")
    descriptors: list[int] = []
    validators: list[tuple[int, str, os.stat_result]] = []
    try:
        current = os.open("/", _DIRECTORY_FLAGS_ANTIGRAVITY)
        descriptors.append(current)
        relative_parts: list[str] = []
        for part in path.parts[1:]:
            relative_parts.append(part)
            relative = "/".join(relative_parts)
            _race_hook_antigravity("cache:before_directory_open", relative)
            try:
                child = os.open(
                    part, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=current
                )
            except FileNotFoundError:
                if not create:
                    raise ValueError("Exp471 cache root is missing")
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
                try:
                    child = os.open(
                        part, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=current
                    )
                except OSError as error:
                    raise ValueError(
                        "Exp471 cache root contains an unsafe component"
                    ) from error
            except OSError as error:
                raise ValueError(
                    "Exp471 cache root contains an unsafe component"
                ) from error
            descriptors.append(child)
            opened = os.fstat(child)
            _race_hook_antigravity("cache:after_directory_open", relative)
            try:
                observed = os.stat(part, dir_fd=current, follow_symlinks=False)
            except OSError as error:
                raise ValueError("Exp471 cache root component changed") from error
            if not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(
                observed.st_mode
            ) or not _same_identity_antigravity(opened, observed):
                raise ValueError("Exp471 cache root component changed")
            validators.append((current, part, opened))
            current = child
        anchor = _DirectoryAnchorAntigravity(current, descriptors, validators)
        anchor.validate()
        return anchor
    except Exception:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _open_child_directory_antigravity(
    parent: int, name: str, *, create: bool
) -> tuple[int, os.stat_result]:
    if not _safe_leaf_name_antigravity(name):
        raise ValueError("Exp471 cache child directory name is unsafe")
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent)
    except FileNotFoundError:
        if not create:
            raise ValueError("Exp471 score-shard directory is missing")
        try:
            os.mkdir(name, 0o755, dir_fd=parent)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(
                name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=parent
            )
        except OSError as error:
            raise ValueError("Exp471 score-shard directory is unsafe") from error
    except OSError as error:
        raise ValueError("Exp471 score-shard directory is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        _race_hook_antigravity("cache:after_directory_open", name)
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(observed.st_mode)
            or not _same_identity_antigravity(opened, observed)
        ):
            raise ValueError("Exp471 score-shard directory changed")
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _validate_child_directory_antigravity(
    parent: int, name: str, expected: os.stat_result
) -> None:
    try:
        observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise ValueError("Exp471 score-shard directory changed") from error
    if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(
        expected, observed
    ):
        raise ValueError("Exp471 score-shard directory changed")


def _read_descriptor_antigravity(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _read_leaf_at_antigravity(
    directory: int, name: str, *, missing_ok: bool = False
) -> bytes | None:
    if not _safe_leaf_name_antigravity(name):
        raise ValueError("Exp471 cache leaf name is unsafe")
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(name, _READ_FLAGS_ANTIGRAVITY, dir_fd=directory)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ValueError("Exp471 cache leaf is missing")
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Exp471 cache leaf is not regular")
        _race_hook_antigravity("cache:after_leaf_open", name)
        raw = _read_descriptor_antigravity(descriptor)
        after = os.fstat(descriptor)
        observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or not _same_identity_antigravity(before, after)
            or not _same_identity_antigravity(before, observed)
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError("Exp471 cache leaf changed while open")
        return raw
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("Exp471 cache leaf is unavailable or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_all_antigravity(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Exp471 cache temporary write made no progress")
        offset += written


def _read_existing_after_link_race_antigravity(
    directory: int, destination: str, *, missing_ok: bool = False
) -> bytes | None:
    last_error: ValueError | None = None
    for _attempt in range(16):
        try:
            existing = _read_leaf_at_antigravity(
                directory, destination, missing_ok=missing_ok
            )
        except ValueError as error:
            last_error = error
            continue
        if existing is None and not missing_ok:
            raise ValueError("Exp471 concurrent cache destination disappeared")
        return existing
    if last_error is not None:
        raise last_error
    raise ValueError("Exp471 concurrent cache destination is unavailable")


def _require_owned_regular_path_antigravity(
    directory: int,
    name: str,
    expected: os.stat_result,
    *,
    scope: str,
) -> os.stat_result:
    try:
        observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Exp471 {scope} pathname is unavailable") from error
    if not stat.S_ISREG(observed.st_mode) or not _same_identity_antigravity(
        expected, observed
    ):
        raise ValueError(f"Exp471 {scope} pathname changed")
    return observed


def _unlink_owned_path_antigravity(
    directory: int, name: str, expected: os.stat_result
) -> bool:
    try:
        observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError:
        return False
    if not _same_identity_antigravity(expected, observed):
        return False
    try:
        os.unlink(name, dir_fd=directory)
    except OSError:
        return False
    return True


def _validate_linked_payload_antigravity(
    directory: int,
    destination: str,
    temporary_identity: os.stat_result,
    payload: bytes,
    *,
    temporary_scope: str,
) -> None:
    _require_owned_regular_path_antigravity(
        directory,
        destination,
        temporary_identity,
        scope=f"{temporary_scope} destination",
    )
    observed = _read_leaf_at_antigravity(directory, destination)
    if observed != payload:
        raise ValueError(f"Exp471 linked {temporary_scope} bytes differ")
    _require_owned_regular_path_antigravity(
        directory,
        destination,
        temporary_identity,
        scope=f"{temporary_scope} destination",
    )


def _publish_bytes_at_antigravity(
    directory: int,
    destination: str,
    payload: bytes,
    *,
    temporary_scope: str,
    post_link_validate_antigravity: Callable[[], None] | None = None,
    linked_failure_recover_antigravity: (
        Callable[[os.stat_result, Exception], None] | None
    ) = None,
) -> None:
    existing = _read_existing_after_link_race_antigravity(
        directory, destination, missing_ok=True
    )
    if existing is not None:
        if existing != payload:
            raise ValueError(f"Exp471 existing {temporary_scope} bytes differ")
        if post_link_validate_antigravity is not None:
            post_link_validate_antigravity()
        return

    temporary = (
        f".{temporary_scope}_temporary_antigravity_{secrets.token_hex(16)}"
    )
    temporary_descriptor: int | None = None
    temporary_identity: os.stat_result | None = None
    linked = False
    try:
        temporary_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        opened = os.fstat(temporary_descriptor)
        temporary_identity = opened
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("Exp471 cache temporary leaf is unsafe")
        _write_all_antigravity(temporary_descriptor, payload)
        os.fchmod(temporary_descriptor, 0o444)
        os.fsync(temporary_descriptor)
        final_descriptor_identity = os.fstat(temporary_descriptor)
        if (
            not stat.S_ISREG(final_descriptor_identity.st_mode)
            or not _same_identity_antigravity(
                temporary_identity, final_descriptor_identity
            )
            or final_descriptor_identity.st_size != len(payload)
        ):
            raise ValueError("Exp471 cache temporary descriptor changed")
        _require_owned_regular_path_antigravity(
            directory,
            temporary,
            temporary_identity,
            scope=f"{temporary_scope} temporary",
        )
        try:
            os.link(
                temporary,
                destination,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            _require_owned_regular_path_antigravity(
                directory,
                temporary,
                temporary_identity,
                scope=f"{temporary_scope} temporary",
            )
            existing = _read_existing_after_link_race_antigravity(
                directory, destination
            )
            if existing != payload:
                raise ValueError(
                    f"Exp471 existing {temporary_scope} bytes differ"
                )
        else:
            _require_owned_regular_path_antigravity(
                directory,
                temporary,
                temporary_identity,
                scope=f"{temporary_scope} temporary",
            )
            _validate_linked_payload_antigravity(
                directory,
                destination,
                temporary_identity,
                payload,
                temporary_scope=temporary_scope,
            )
        if post_link_validate_antigravity is not None:
            post_link_validate_antigravity()
        if not _unlink_owned_path_antigravity(
            directory, temporary, temporary_identity
        ):
            raise ValueError(
                f"Exp471 {temporary_scope} temporary pathname changed"
            )
        if linked:
            _validate_linked_payload_antigravity(
                directory,
                destination,
                temporary_identity,
                payload,
                temporary_scope=temporary_scope,
            )
        os.fsync(directory)
    except Exception as error:
        changed = False
        if linked and temporary_identity is not None:
            if linked_failure_recover_antigravity is None:
                changed = _unlink_owned_path_antigravity(
                    directory, destination, temporary_identity
                )
            else:
                linked_failure_recover_antigravity(temporary_identity, error)
        if changed:
            try:
                os.fsync(directory)
            except OSError:
                pass
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_identity is not None and _unlink_owned_path_antigravity(
            directory, temporary, temporary_identity
        ):
            try:
                os.fsync(directory)
            except OSError:
                pass


def _recovery_identity_antigravity(
    value: tuple[int, int] | os.stat_result | None,
) -> dict[str, int] | None:
    if value is None:
        return None
    if isinstance(value, os.stat_result):
        device, inode = value.st_dev, value.st_ino
    elif (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(item) is int and item >= 0 for item in value)
    ):
        device, inode = value
    else:
        raise ValueError("Exp471 manifest recovery identity differs")
    return {"device": int(device), "inode": int(inode)}


def _manifest_recovery_record_antigravity(
    *,
    original_path: str,
    reason: str,
    expected_identity: dict[str, int] | None,
    observed_identity: dict[str, int] | None,
    recovery_path: str | None,
) -> dict[str, object]:
    bare: dict[str, object] = {
        "schema": _CACHE_MANIFEST_RECOVERY_SCHEMA_ANTIGRAVITY,
        "original_path": original_path,
        "reason": reason,
        "expected_identity": expected_identity,
        "observed_identity": observed_identity,
        "recovery_path": recovery_path,
    }
    return {
        **bare,
        "recovery_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    }


def _validated_manifest_recovery_antigravity(
    raw: bytes,
    *,
    original_path: str,
    reason: str,
    expected_identity: dict[str, int] | None,
    recovery_path: str | None,
    recovery_directory: int,
) -> dict[str, object]:
    record = _json_object_antigravity(raw, scope="manifest recovery metadata")
    required = {
        "schema",
        "original_path",
        "reason",
        "expected_identity",
        "observed_identity",
        "recovery_path",
        "recovery_sha256",
    }
    if (
        set(record) != required
        or record["schema"] != _CACHE_MANIFEST_RECOVERY_SCHEMA_ANTIGRAVITY
        or record["original_path"] != original_path
        or record["reason"] != reason
        or record["expected_identity"] != expected_identity
        or record["recovery_path"] != recovery_path
    ):
        raise ValueError("Exp471 manifest recovery metadata differs")
    observed = record["observed_identity"]
    if recovery_path is None:
        if observed is not None:
            raise ValueError("Exp471 manifest recovery identity differs")
    else:
        if (
            not isinstance(observed, dict)
            or set(observed) != {"device", "inode"}
            or any(type(observed[field]) is not int for field in observed)
        ):
            raise ValueError("Exp471 manifest recovery identity differs")
        recovered = os.stat(
            _CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY,
            dir_fd=recovery_directory,
            follow_symlinks=False,
        )
        if observed != _recovery_identity_antigravity(recovered):
            raise ValueError("Exp471 recovered manifest identity differs")
    bare = {key: value for key, value in record.items() if key != "recovery_sha256"}
    if (
        type(record["recovery_sha256"]) is not str
        or record["recovery_sha256"]
        != sha256(canonical_json_bytes_exp471_antigravity(bare)).hexdigest()
        or raw != canonical_json_bytes_exp471_antigravity(record)
    ):
        raise ValueError("Exp471 manifest recovery metadata is noncanonical")
    return dict(record)


def _quarantine_cache_manifest_in_anchor_antigravity(
    root: Path,
    anchor: _DirectoryAnchorAntigravity,
    *,
    reason: str,
    expected_identity: tuple[int, int] | os.stat_result | None,
) -> dict[str, object] | None:
    if type(reason) is not str or not reason:
        raise ValueError("Exp471 manifest recovery reason differs")
    expected_record = _recovery_identity_antigravity(expected_identity)
    original_path = str(Path(root) / _CACHE_MANIFEST_NAME_ANTIGRAVITY)
    operation = {
        "schema": _CACHE_MANIFEST_RECOVERY_SCHEMA_ANTIGRAVITY,
        "original_path": original_path,
        "reason": reason,
        "expected_identity": expected_record,
    }
    operation_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(operation)
    ).hexdigest()
    directory_name = (
        f"{_CACHE_MANIFEST_RECOVERY_PREFIX_ANTIGRAVITY}"
        f"{operation_sha256}{_CACHE_MANIFEST_RECOVERY_SUFFIX_ANTIGRAVITY}"
    )
    try:
        os.stat(directory_name, dir_fd=anchor.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.stat(
                _CACHE_MANIFEST_NAME_ANTIGRAVITY,
                dir_fd=anchor.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        try:
            os.mkdir(directory_name, 0o700, dir_fd=anchor.descriptor)
            created = True
        except FileExistsError:
            created = False
    else:
        created = False

    recovery_directory: int | None = None
    try:
        recovery_directory, directory_identity = _open_child_directory_antigravity(
            anchor.descriptor, directory_name, create=False
        )
        metadata_raw = _read_leaf_at_antigravity(
            recovery_directory,
            _CACHE_MANIFEST_RECOVERY_METADATA_NAME_ANTIGRAVITY,
            missing_ok=True,
        )
        recovery_path = str(
            Path(root)
            / directory_name
            / _CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY
        )
        if metadata_raw is not None:
            metadata_value = _json_object_antigravity(
                metadata_raw, scope="manifest recovery metadata"
            )
            recorded_path = metadata_value.get("recovery_path")
            if recorded_path not in {None, recovery_path}:
                raise ValueError("Exp471 manifest recovery path differs")
            return _validated_manifest_recovery_antigravity(
                metadata_raw,
                original_path=original_path,
                reason=reason,
                expected_identity=expected_record,
                recovery_path=recorded_path,
                recovery_directory=recovery_directory,
            )
        recovered_identity: os.stat_result | None = None
        try:
            recovered_identity = os.stat(
                _CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY,
                dir_fd=recovery_directory,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            if not created:
                raise ValueError("Exp471 manifest recovery is incomplete")
            _race_hook_antigravity(
                "cache:before_manifest_quarantine_rename",
                _CACHE_MANIFEST_NAME_ANTIGRAVITY,
            )
            try:
                os.rename(
                    _CACHE_MANIFEST_NAME_ANTIGRAVITY,
                    _CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY,
                    src_dir_fd=anchor.descriptor,
                    dst_dir_fd=recovery_directory,
                )
            except FileNotFoundError:
                recovered_identity = None
                recovery_path = None
            else:
                recovered_identity = os.stat(
                    _CACHE_MANIFEST_RECOVERED_NAME_ANTIGRAVITY,
                    dir_fd=recovery_directory,
                    follow_symlinks=False,
                )
                os.fsync(recovery_directory)
                os.fsync(anchor.descriptor)
        record = _manifest_recovery_record_antigravity(
            original_path=original_path,
            reason=reason,
            expected_identity=expected_record,
            observed_identity=_recovery_identity_antigravity(recovered_identity),
            recovery_path=recovery_path,
        )
        _publish_bytes_at_antigravity(
            recovery_directory,
            _CACHE_MANIFEST_RECOVERY_METADATA_NAME_ANTIGRAVITY,
            canonical_json_bytes_exp471_antigravity(record),
            temporary_scope="score_cache_manifest_recovery_metadata",
        )
        reopened = _read_leaf_at_antigravity(
            recovery_directory,
            _CACHE_MANIFEST_RECOVERY_METADATA_NAME_ANTIGRAVITY,
        )
        if reopened is None:
            raise ValueError("Exp471 manifest recovery metadata is missing")
        validated = _validated_manifest_recovery_antigravity(
            reopened,
            original_path=original_path,
            reason=reason,
            expected_identity=expected_record,
            recovery_path=recovery_path,
            recovery_directory=recovery_directory,
        )
        _validate_child_directory_antigravity(
            anchor.descriptor, directory_name, directory_identity
        )
        anchor.validate()
        return validated
    finally:
        if recovery_directory is not None:
            os.close(recovery_directory)


def _quarantine_cache_manifest_antigravity(
    root: Path,
    *,
    reason: str,
    expected_identity: tuple[int, int] | os.stat_result | None,
) -> dict[str, object] | None:
    anchor = _open_absolute_directory_antigravity(Path(root), create=False)
    try:
        return _quarantine_cache_manifest_in_anchor_antigravity(
            Path(root),
            anchor,
            reason=reason,
            expected_identity=expected_identity,
        )
    finally:
        anchor.close()


def _cache_manifest_identity_antigravity(root: Path) -> tuple[int, int] | None:
    anchor = _open_absolute_directory_antigravity(Path(root), create=False)
    try:
        try:
            observed = os.stat(
                _CACHE_MANIFEST_NAME_ANTIGRAVITY,
                dir_fd=anchor.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        anchor.validate()
        return observed.st_dev, observed.st_ino
    finally:
        anchor.close()


def _reject_json_constant_antigravity(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json_object_antigravity(raw: bytes, *, scope: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_json_constant_antigravity
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Exp471 {scope} JSON is malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"Exp471 {scope} schema differs")
    return value


def _shard_payload_record_antigravity(
    shard: Exp471ScoreShardAntigravity,
) -> dict[str, object]:
    return {
        "schema": _SCORE_SHARD_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": shard.source_authority_sha256,
        "cache_context_sha256": shard.cache_context_sha256,
        "method": shard.method,
        "dataset": shard.dataset,
        "status": shard.status,
        "test_count": shard.test_count,
        "test_indices": list(shard.test_indices),
        "raw_scores": list(shard.raw_scores),
        "micro_scores": list(shard.micro_scores),
        "abstention_reason": shard.abstention_reason,
        "source_manifest_sha256": shard.source_manifest_sha256,
    }


def _canonical_shard_bytes_antigravity(
    shard: Exp471ScoreShardAntigravity,
) -> bytes:
    if not isinstance(shard, Exp471ScoreShardAntigravity):
        raise ValueError("Exp471 score shard has the wrong type")
    payload = _shard_payload_record_antigravity(shard)
    payload_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(payload)
    ).hexdigest()
    if shard.payload_sha256 != payload_sha256:
        raise ValueError("Exp471 score shard payload digest differs")
    envelope = dict(payload)
    envelope["payload_sha256"] = shard.payload_sha256
    envelope_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(envelope)
    ).hexdigest()
    if shard.envelope_sha256 != envelope_sha256:
        raise ValueError("Exp471 score shard envelope digest differs")
    envelope["envelope_sha256"] = shard.envelope_sha256
    return canonical_json_bytes_exp471_antigravity(envelope)


def _score_shard_from_bytes_antigravity(
    raw: bytes,
) -> Exp471ScoreShardAntigravity:
    value = _json_object_antigravity(raw, scope="score shard")
    required = {
        "schema",
        "source_authority_sha256",
        "cache_context_sha256",
        "method",
        "dataset",
        "status",
        "test_count",
        "test_indices",
        "raw_scores",
        "micro_scores",
        "abstention_reason",
        "source_manifest_sha256",
        "payload_sha256",
        "envelope_sha256",
    }
    if set(value) != required:
        raise ValueError("Exp471 score shard schema differs")
    if value.pop("schema") != _SCORE_SHARD_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 score shard schema differs")
    try:
        shard = Exp471ScoreShardAntigravity(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 score shard fields are invalid") from error
    if raw != _canonical_shard_bytes_antigravity(shard):
        raise ValueError("Exp471 score shard bytes are noncanonical")
    return shard


def _manifest_record_antigravity(
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> dict[str, object]:
    return {
        "schema": _CACHE_MANIFEST_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": manifest.source_authority_sha256,
        "cache_context_sha256": manifest.cache_context_sha256,
        "method_order": list(manifest.method_order),
        "dataset_order": list(manifest.dataset_order),
        "shard_sha256": dict(manifest.shard_sha256),
        "total_shards": manifest.total_shards,
        "scored_shards": manifest.scored_shards,
        "abstained_shards": manifest.abstained_shards,
        "failed_shards": manifest.failed_shards,
    }


def _canonical_manifest_bytes_antigravity(
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> bytes:
    if not isinstance(manifest, Exp471ScoreCacheManifestAntigravity):
        raise ValueError("Exp471 score-cache manifest has the wrong type")
    record = _manifest_record_antigravity(manifest)
    manifest_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(record)
    ).hexdigest()
    if manifest.manifest_sha256 != manifest_sha256:
        raise ValueError("Exp471 score-cache manifest digest differs")
    record["manifest_sha256"] = manifest.manifest_sha256
    return canonical_json_bytes_exp471_antigravity(record)


def _manifest_from_bytes_antigravity(
    raw: bytes,
) -> Exp471ScoreCacheManifestAntigravity:
    value = _json_object_antigravity(raw, scope="score-cache manifest")
    required = {
        "schema",
        "source_authority_sha256",
        "cache_context_sha256",
        "method_order",
        "dataset_order",
        "shard_sha256",
        "total_shards",
        "scored_shards",
        "abstained_shards",
        "failed_shards",
        "manifest_sha256",
    }
    if set(value) != required:
        raise ValueError("Exp471 score-cache manifest schema differs")
    if value.pop("schema") != _CACHE_MANIFEST_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 score-cache manifest schema differs")
    try:
        manifest = Exp471ScoreCacheManifestAntigravity(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 score-cache manifest fields are invalid") from error
    if raw != _canonical_manifest_bytes_antigravity(manifest):
        raise ValueError("Exp471 score-cache manifest bytes are noncanonical")
    return manifest


def score_shard_relative_path_antigravity(method: str, dataset: str) -> Path:
    """Return the contained canonical shard path for one fixed identity."""

    if method not in METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 score shard method is outside the fixed order")
    if type(dataset) is not str or not dataset:
        raise ValueError("Exp471 score shard dataset must be a non-empty string")
    identity = canonical_json_bytes_exp471_antigravity(
        {"method": method, "dataset": dataset}
    )
    digest = sha256(identity).hexdigest()
    return Path(
        _SCORE_SHARD_DIRECTORY_ANTIGRAVITY,
        f"{digest}_antigravity.json",
    )


def _read_score_shard_from_directory_antigravity(
    directory: int, method: str, dataset: str
) -> Exp471ScoreShardAntigravity:
    relative = score_shard_relative_path_antigravity(method, dataset)
    raw = _read_leaf_at_antigravity(directory, relative.name)
    if raw is None:
        raise ValueError("Exp471 score shard is missing")
    shard = _score_shard_from_bytes_antigravity(raw)
    if shard.method != method or shard.dataset != dataset:
        raise ValueError("Exp471 score shard identity differs from its path")
    return shard


def publish_score_shard_antigravity(
    root: Path, shard: Exp471ScoreShardAntigravity
) -> Path:
    payload = _canonical_shard_bytes_antigravity(shard)
    relative = score_shard_relative_path_antigravity(shard.method, shard.dataset)
    anchor = _open_absolute_directory_antigravity(Path(root), create=True)
    shard_directory: int | None = None
    try:
        shard_directory, expected = _open_child_directory_antigravity(
            anchor.descriptor, relative.parts[0], create=True
        )
        _publish_bytes_at_antigravity(
            shard_directory,
            relative.name,
            payload,
            temporary_scope="score_shard",
        )
        _validate_child_directory_antigravity(
            anchor.descriptor, relative.parts[0], expected
        )
        anchor.validate()
        return Path(root) / relative
    finally:
        if shard_directory is not None:
            os.close(shard_directory)
        anchor.close()


def read_score_shard_antigravity(
    root: Path, method: str, dataset: str
) -> Exp471ScoreShardAntigravity:
    relative = score_shard_relative_path_antigravity(method, dataset)
    anchor = _open_absolute_directory_antigravity(Path(root), create=False)
    shard_directory: int | None = None
    try:
        shard_directory, expected = _open_child_directory_antigravity(
            anchor.descriptor, relative.parts[0], create=False
        )
        shard = _read_score_shard_from_directory_antigravity(
            shard_directory, method, dataset
        )
        _validate_child_directory_antigravity(
            anchor.descriptor, relative.parts[0], expected
        )
        anchor.validate()
        return shard
    finally:
        if shard_directory is not None:
            os.close(shard_directory)
        anchor.close()


def _verify_manifest_coverage_in_directory_antigravity(
    shard_directory: int,
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> None:
    expected_keys = {
        f"{method}:{dataset}"
        for method in manifest.method_order
        for dataset in manifest.dataset_order
    }
    if set(manifest.shard_sha256) != expected_keys:
        raise ValueError("Exp471 score-cache manifest coverage identities differ")
    expected_names = {
        score_shard_relative_path_antigravity(method, dataset).name
        for method in manifest.method_order
        for dataset in manifest.dataset_order
    }

    def require_exact_inventory_antigravity() -> None:
        observed_names = tuple(os.listdir(shard_directory))
        if (
            len(observed_names) != len(expected_names)
            or set(observed_names) != expected_names
        ):
            raise ValueError(
                "Exp471 score-cache shard inventory differs from manifest coverage"
            )

    require_exact_inventory_antigravity()
    status_counts = {"scored": 0, "abstained": 0}
    for method in manifest.method_order:
        for dataset in manifest.dataset_order:
            shard = _read_score_shard_from_directory_antigravity(
                shard_directory, method, dataset
            )
            key = f"{method}:{dataset}"
            if (
                shard.source_authority_sha256
                != manifest.source_authority_sha256
                or shard.cache_context_sha256 != manifest.cache_context_sha256
                or shard.envelope_sha256 != manifest.shard_sha256[key]
            ):
                raise ValueError(
                    "Exp471 score-cache shard differs from manifest coverage"
                )
            status_counts[shard.status] += 1
    require_exact_inventory_antigravity()
    if (
        status_counts["scored"] != manifest.scored_shards
        or status_counts["abstained"] != manifest.abstained_shards
        or manifest.failed_shards != 0
    ):
        raise ValueError("Exp471 score-cache manifest status counts differ")


def _open_verified_manifest_coverage_antigravity(
    anchor: _DirectoryAnchorAntigravity,
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> tuple[int, os.stat_result]:
    shard_directory: int | None = None
    try:
        shard_directory, expected = _open_child_directory_antigravity(
            anchor.descriptor, _SCORE_SHARD_DIRECTORY_ANTIGRAVITY, create=False
        )
        _verify_manifest_coverage_in_directory_antigravity(
            shard_directory, manifest
        )
        _validate_child_directory_antigravity(
            anchor.descriptor, _SCORE_SHARD_DIRECTORY_ANTIGRAVITY, expected
        )
        anchor.validate()
        return shard_directory, expected
    except Exception:
        if shard_directory is not None:
            os.close(shard_directory)
        raise


def _revalidate_manifest_coverage_antigravity(
    anchor: _DirectoryAnchorAntigravity,
    shard_directory: int,
    expected: os.stat_result,
    manifest: Exp471ScoreCacheManifestAntigravity,
) -> None:
    _verify_manifest_coverage_in_directory_antigravity(
        shard_directory, manifest
    )
    _validate_child_directory_antigravity(
        anchor.descriptor, _SCORE_SHARD_DIRECTORY_ANTIGRAVITY, expected
    )
    anchor.validate()


def publish_cache_manifest_antigravity(
    root: Path, manifest: Exp471ScoreCacheManifestAntigravity
) -> Path:
    payload = _canonical_manifest_bytes_antigravity(manifest)
    anchor = _open_absolute_directory_antigravity(Path(root), create=True)
    shard_directory: int | None = None
    try:
        shard_directory, expected = _open_verified_manifest_coverage_antigravity(
            anchor, manifest
        )

        def revalidate_coverage_antigravity() -> None:
            _revalidate_manifest_coverage_antigravity(
                anchor,
                shard_directory,
                expected,
                manifest,
            )

        def recover_linked_manifest_antigravity(
            expected_identity: os.stat_result,
            error: Exception,
        ) -> None:
            _quarantine_cache_manifest_in_anchor_antigravity(
                Path(root),
                anchor,
                reason=(
                    "terminal manifest linked validation failed: "
                    f"{type(error).__name__}: {error}"
                ),
                expected_identity=expected_identity,
            )

        _publish_bytes_at_antigravity(
            anchor.descriptor,
            _CACHE_MANIFEST_NAME_ANTIGRAVITY,
            payload,
            temporary_scope="score_cache_manifest",
            post_link_validate_antigravity=revalidate_coverage_antigravity,
            linked_failure_recover_antigravity=recover_linked_manifest_antigravity,
        )
        anchor.validate()
        return Path(root) / _CACHE_MANIFEST_NAME_ANTIGRAVITY
    finally:
        if shard_directory is not None:
            os.close(shard_directory)
        anchor.close()


def read_cache_manifest_antigravity(
    root: Path,
) -> Exp471ScoreCacheManifestAntigravity:
    anchor = _open_absolute_directory_antigravity(Path(root), create=False)
    shard_directory: int | None = None
    try:
        raw = _read_leaf_at_antigravity(
            anchor.descriptor, _CACHE_MANIFEST_NAME_ANTIGRAVITY
        )
        if raw is None:
            raise ValueError("Exp471 score-cache manifest is missing")
        manifest = _manifest_from_bytes_antigravity(raw)
        shard_directory, expected = _open_verified_manifest_coverage_antigravity(
            anchor, manifest
        )
        _revalidate_manifest_coverage_antigravity(
            anchor, shard_directory, expected, manifest
        )
        anchor.validate()
        return manifest
    finally:
        if shard_directory is not None:
            os.close(shard_directory)
        anchor.close()


__all__ = [
    "publish_cache_manifest_antigravity",
    "publish_score_shard_antigravity",
    "read_cache_manifest_antigravity",
    "read_score_shard_antigravity",
    "score_shard_relative_path_antigravity",
]
