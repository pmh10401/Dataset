"""Verified, create-only Exp471 paper-report publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
from dataclasses import dataclass
import errno
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import stat
import subprocess
import sys
import threading
from time import monotonic
from typing import Any

from .config_antigravity import (
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    Exp471ReportAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from .typed_abstain_reporting_antigravity import (
    Exp471TypedReportArtifactAntigravity,
    expected_typed_exp471_report_leaves_antigravity,
    write_typed_exp471_report_antigravity,
)


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE_MARKER_NAME_ANTIGRAVITY = (
    ".exp471_report_complete_antigravity.json"
)
_ARTIFACT_MAP_NAME_ANTIGRAVITY = "artifact_sha256_antigravity.json"
_PROVENANCE_NAME_ANTIGRAVITY = "exp471_report_provenance_antigravity.json"
_REPORT_SCHEMA_ANTIGRAVITY = "exp471-report-publication-antigravity.v2"
_ARTIFACT_MAP_SCHEMA_ANTIGRAVITY = "exp471-report-artifact-map-antigravity.v2"
_PROVENANCE_SCHEMA_ANTIGRAVITY = "exp471-report-provenance-antigravity.v2"
_LOCK_SCHEMA_ANTIGRAVITY = "exp471-report-lock-antigravity.v1"
_RENAME_NOREPLACE_ANTIGRAVITY = 1
_RENAME_EXCL_ANTIGRAVITY = 0x00000004

_REPORT_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp450_resampling_image_ablation_antigravity/"
    "evaluation_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/"
    "reporting_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "config_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "contracts_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "evaluation_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "score_cache_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "reporting_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "typed_abstain_reporting_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "cli_antigravity.py",
    "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py",
)


@dataclass(frozen=True, slots=True)
class _LockTokenAntigravity:
    path: Path
    payload: bytes
    device: int
    inode: int
    descriptor: int
    parent_path: Path
    parent_descriptor: int
    parent_device: int
    parent_inode: int


@dataclass(frozen=True, slots=True)
class _StageTokenAntigravity:
    path: Path
    descriptor: int
    device: int
    inode: int


_LOCAL_PUBLICATION_CONDITION_ANTIGRAVITY = threading.Condition()
_LOCAL_PUBLICATION_ROOTS_ANTIGRAVITY: set[Path] = set()
_TASK5_SCORE_CACHE_ROOT_ANTIGRAVITY = Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)


def _repository_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_all_antigravity(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Exp471 report write made no progress")
        view = view[written:]


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_anchor_antigravity(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    linked = path.lstat()
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or opened.st_dev != linked.st_dev
        or opened.st_ino != linked.st_ino
    ):
        os.close(descriptor)
        raise ValueError("Exp471 report directory identity differs")
    return descriptor, opened


def _validate_directory_anchor_antigravity(
    path: Path, descriptor: int, device: int, inode: int
) -> None:
    opened = os.fstat(descriptor)
    try:
        linked = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp471 report parent identity changed") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or opened.st_dev != device
        or opened.st_ino != inode
        or linked.st_dev != device
        or linked.st_ino != inode
    ):
        raise ValueError("Exp471 report parent identity changed")


def _entry_stat_antigravity(parent_descriptor: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)


def _require_entry_identity_antigravity(
    parent_descriptor: int, name: str, device: int, inode: int, scope: str
) -> os.stat_result:
    try:
        linked = _entry_stat_antigravity(parent_descriptor, name)
    except FileNotFoundError as error:
        raise ValueError(f"Exp471 report {scope} identity changed") from error
    if linked.st_dev != device or linked.st_ino != inode:
        raise ValueError(f"Exp471 report {scope} identity changed")
    return linked


def _require_safe_parent_antigravity(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or "antigravity" not in root.name:
        raise ValueError("Exp471 report root must be an absolute antigravity path")
    parent = root.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise ValueError("Exp471 report parent is unsafe")
    current = parent
    while current != current.parent:
        details = current.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp471 report parent contains a symlink")
        current = current.parent
    return parent


def _report_lock_path_antigravity(root: Path) -> Path:
    root = Path(root)
    return root.parent / f".{root.name}.publish_lock_antigravity.json"


def _process_start_identity_antigravity(pid: int) -> str | None:
    if type(pid) is not int or pid < 1:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown-live-process-antigravity"
    identity = " ".join(result.stdout.split())
    if result.returncode != 0 or not identity:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
        return "unknown-live-process-antigravity"
    return identity


def _lock_record_antigravity(pid: int, process_start_identity: str) -> dict[str, object]:
    if (
        type(pid) is not int
        or pid < 1
        or type(process_start_identity) is not str
        or not process_start_identity
    ):
        raise ValueError("Exp471 report lock identity is invalid")
    return {
        "schema": _LOCK_SCHEMA_ANTIGRAVITY,
        "pid": pid,
        "process_start_identity": process_start_identity,
    }


def _read_descriptor_antigravity(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_lock_antigravity(path: Path) -> tuple[dict[str, object], bytes, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Exp471 report lock is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_nlink != 1
        ):
            raise ValueError("Exp471 report lock is unsafe")
        raw = _read_descriptor_antigravity(descriptor)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp471 report lock record is foreign") from error
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "pid", "process_start_identity"}
        or value.get("schema") != _LOCK_SCHEMA_ANTIGRAVITY
        or canonical_json_bytes_exp471_antigravity(value) != raw
        or type(value.get("pid")) is not int
        or value["pid"] < 1
        or type(value.get("process_start_identity")) is not str
        or not value["process_start_identity"]
    ):
        raise ValueError("Exp471 report lock record is foreign")
    return value, raw, opened


def _atomic_rename_noreplace_at_antigravity(
    parent_descriptor: int, source_name: str, destination_name: str
) -> None:
    if "/" in source_name or "/" in destination_name:
        raise ValueError("Exp471 report rename names must be direct children")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "Exp471 atomic no-replace rename is unavailable"
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
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            ctypes.c_uint(_RENAME_EXCL_ANTIGRAVITY),
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "Exp471 atomic no-replace rename is unavailable"
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
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            ctypes.c_uint(_RENAME_NOREPLACE_ANTIGRAVITY),
        )
    else:
        raise OSError(
            errno.ENOTSUP, "Exp471 atomic no-replace rename is unavailable"
        )
    if result == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(code, "Exp471 report destination already exists")
    raise OSError(code, "Exp471 atomic no-replace rename failed")


def _atomic_rename_noreplace_antigravity(source: Path, destination: Path) -> None:
    if source.parent != destination.parent:
        raise ValueError("Exp471 report rename source is not a sibling")
    descriptor, _ = _open_directory_anchor_antigravity(source.parent)
    try:
        _atomic_rename_noreplace_at_antigravity(
            descriptor, source.name, destination.name
        )
    finally:
        os.close(descriptor)


def _postcheck_namespace_hook_antigravity(
    scope: str, parent_descriptor: int, name: str
) -> None:
    """Deterministic test seam at a checked-name/namespace-operation boundary."""

    del scope, parent_descriptor, name


def _final_claim_namespace_hook_antigravity(
    scope: str, parent_descriptor: int, name: str
) -> None:
    """Deterministic seam after the final claimed-name identity check."""

    del scope, parent_descriptor, name


def _claim_entry_antigravity(
    parent_descriptor: int,
    source_name: str,
    expected_device: int,
    expected_inode: int,
    scope: str,
    *,
    retained_quarantine: bool = False,
) -> tuple[str, int, os.stat_result]:
    """Move a mutable candidate to a unique non-authoritative name and reopen it."""

    for _ in range(128):
        prefix = (
            ".retained_quarantine_"
            if retained_quarantine
            else ".claim_"
        )
        claim_name = f"{prefix}{secrets.token_hex(12)}_antigravity"
        try:
            _atomic_rename_noreplace_at_antigravity(
                parent_descriptor, source_name, claim_name
            )
        except FileExistsError:
            continue
        except FileNotFoundError as error:
            raise ValueError(f"Exp471 report {scope} identity changed") from error
        break
    else:
        raise FileExistsError("Exp471 report claim name space is exhausted")

    try:
        linked = _entry_stat_antigravity(parent_descriptor, claim_name)
    except FileNotFoundError as error:
        raise ValueError(f"Exp471 report {scope} claim disappeared") from error
    if linked.st_dev != expected_device or linked.st_ino != expected_inode:
        raise ValueError(f"Exp471 report {scope} claim identity changed")
    if not (stat.S_ISREG(linked.st_mode) or stat.S_ISDIR(linked.st_mode)):
        raise ValueError(f"Exp471 report {scope} claim is unsafe")
    flags = os.O_RDONLY
    if stat.S_ISDIR(linked.st_mode) and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(claim_name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(f"Exp471 report {scope} claim is unsafe") from error
    opened = os.fstat(descriptor)
    if (
        linked.st_dev != expected_device
        or linked.st_ino != expected_inode
        or opened.st_dev != expected_device
        or opened.st_ino != expected_inode
        or stat.S_IFMT(linked.st_mode) != stat.S_IFMT(opened.st_mode)
    ):
        os.close(descriptor)
        raise ValueError(f"Exp471 report {scope} claim identity changed")
    return claim_name, descriptor, opened


def _delete_claimed_entry_antigravity(
    parent_descriptor: int,
    claim_name: str,
    descriptor: int,
    expected_device: int,
    expected_inode: int,
    scope: str,
) -> None:
    """Evacuate a verified claim; pathname deletion cannot be made inode-targeted."""

    opened = os.fstat(descriptor)
    linked = _require_entry_identity_antigravity(
        parent_descriptor,
        claim_name,
        expected_device,
        expected_inode,
        f"{scope} owned claim",
    )
    if opened.st_dev != expected_device or opened.st_ino != expected_inode:
        raise ValueError(f"Exp471 report {scope} owned claim identity changed")
    _final_claim_namespace_hook_antigravity(
        f"{scope} delete claim", parent_descriptor, claim_name
    )
    retained_name, retained_descriptor, retained = _claim_entry_antigravity(
        parent_descriptor,
        claim_name,
        expected_device,
        expected_inode,
        f"{scope} retained quarantine",
        retained_quarantine=True,
    )
    del retained_name
    try:
        if (
            retained.st_dev != opened.st_dev
            or retained.st_ino != opened.st_ino
            or os.fstat(retained_descriptor).st_dev != opened.st_dev
            or os.fstat(retained_descriptor).st_ino != opened.st_ino
        ):
            raise ValueError(
                f"Exp471 report {scope} retained quarantine identity changed"
            )
        os.fsync(parent_descriptor)
    finally:
        os.close(retained_descriptor)


def _evacuate_canonical_entry_antigravity(
    parent_descriptor: int, canonical_name: str
) -> str:
    """Move an uncertain canonical object to observable retained quarantine."""

    try:
        before = _entry_stat_antigravity(parent_descriptor, canonical_name)
    except FileNotFoundError as error:
        raise ValueError("Exp471 report canonical entry disappeared") from error
    for _ in range(128):
        retained_name = (
            f".retained_quarantine_{secrets.token_hex(12)}_antigravity"
        )
        try:
            _atomic_rename_noreplace_at_antigravity(
                parent_descriptor, canonical_name, retained_name
            )
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(
            "Exp471 report retained quarantine name space is exhausted"
        )
    try:
        moved = _entry_stat_antigravity(parent_descriptor, retained_name)
        if moved.st_dev != before.st_dev or moved.st_ino != before.st_ino:
            raise ValueError("Exp471 report canonical evacuation identity changed")
        if stat.S_ISREG(moved.st_mode) or stat.S_ISDIR(moved.st_mode):
            flags = os.O_RDONLY
            if stat.S_ISDIR(moved.st_mode) and hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(
                retained_name, flags, dir_fd=parent_descriptor
            )
            try:
                reopened = os.fstat(descriptor)
                if (
                    reopened.st_dev != before.st_dev
                    or reopened.st_ino != before.st_ino
                ):
                    raise ValueError(
                        "Exp471 report canonical evacuation identity changed"
                    )
            finally:
                os.close(descriptor)
        os.fsync(parent_descriptor)
        return retained_name
    except Exception:
        os.fsync(parent_descriptor)
        raise


def _recover_dead_lock_antigravity(
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
    opened: os.stat_result,
    parent_descriptor: int,
    parent_device: int,
    parent_inode: int,
) -> None:
    current_identity = _process_start_identity_antigravity(int(value["pid"]))
    if current_identity == value["process_start_identity"]:
        raise ValueError("Exp471 report lock has an active owner")
    if current_identity == "unknown-live-process-antigravity":
        raise ValueError("Exp471 report lock owner cannot be proven dead")
    _validate_directory_anchor_antigravity(
        path.parent, parent_descriptor, parent_device, parent_inode
    )
    _require_entry_identity_antigravity(
        parent_descriptor,
        path.name,
        opened.st_dev,
        opened.st_ino,
        "lock during recovery",
    )
    _postcheck_namespace_hook_antigravity(
        "dead lock initial claim", parent_descriptor, path.name
    )
    quarantine = path.with_name(
        f".{path.name}.{secrets.token_hex(12)}.dead_antigravity"
    )
    _atomic_rename_noreplace_at_antigravity(
        parent_descriptor, path.name, quarantine.name
    )
    try:
        moved_value, moved_raw, moved = _read_lock_antigravity(quarantine)
        if (
            moved.st_dev != opened.st_dev
            or moved.st_ino != opened.st_ino
            or moved_raw != raw
            or moved_value != value
        ):
            raise ValueError("Exp471 report dead lock identity changed")
        _validate_directory_anchor_antigravity(
            path.parent, parent_descriptor, parent_device, parent_inode
        )
        _require_entry_identity_antigravity(
            parent_descriptor,
            quarantine.name,
            opened.st_dev,
            opened.st_ino,
            "dead lock quarantine",
        )
        _postcheck_namespace_hook_antigravity(
            "dead lock quarantine", parent_descriptor, quarantine.name
        )
        claim_name, claim_descriptor, claim_opened = _claim_entry_antigravity(
            parent_descriptor,
            quarantine.name,
            opened.st_dev,
            opened.st_ino,
            "dead lock quarantine",
        )
        try:
            claim_path = path.parent / claim_name
            claimed_value, claimed_raw, claimed = _read_lock_antigravity(claim_path)
            if (
                claim_opened.st_dev != claimed.st_dev
                or claim_opened.st_ino != claimed.st_ino
                or claimed_raw != raw
                or claimed_value != value
            ):
                raise ValueError("Exp471 report dead lock claim identity changed")
            _delete_claimed_entry_antigravity(
                parent_descriptor,
                claim_name,
                claim_descriptor,
                opened.st_dev,
                opened.st_ino,
                "dead lock quarantine",
            )
        finally:
            os.close(claim_descriptor)
        os.fsync(parent_descriptor)
    except Exception:
        # A record whose identity changed is preserved under quarantine.
        raise


def _acquire_report_lock_antigravity(root: Path) -> _LockTokenAntigravity:
    parent = _require_safe_parent_antigravity(root)
    parent_descriptor, parent_opened = _open_directory_anchor_antigravity(parent)
    path = _report_lock_path_antigravity(root)
    try:
        if os.path.lexists(path):
            if path.is_symlink():
                raise ValueError("Exp471 report lock is a symlink")
            value, raw, opened = _read_lock_antigravity(path)
            _recover_dead_lock_antigravity(
                path,
                value,
                raw,
                opened,
                parent_descriptor,
                parent_opened.st_dev,
                parent_opened.st_ino,
            )
        identity = _process_start_identity_antigravity(os.getpid())
        if identity is None or identity == "unknown-live-process-antigravity":
            raise ValueError("Exp471 current process start identity is unavailable")
        payload = canonical_json_bytes_exp471_antigravity(
            _lock_record_antigravity(os.getpid(), identity)
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError as error:
            raise ValueError("Exp471 report lock has an active owner") from error
        try:
            _write_all_antigravity(descriptor, payload)
            os.fsync(descriptor)
            opened = os.fstat(descriptor)
            _validate_directory_anchor_antigravity(
                parent,
                parent_descriptor,
                parent_opened.st_dev,
                parent_opened.st_ino,
            )
            _require_entry_identity_antigravity(
                parent_descriptor,
                path.name,
                opened.st_dev,
                opened.st_ino,
                "owned lock",
            )
            os.fsync(parent_descriptor)
        except Exception:
            retained = os.fstat(descriptor)
            os.close(descriptor)
            try:
                _require_entry_identity_antigravity(
                    parent_descriptor,
                    path.name,
                    retained.st_dev,
                    retained.st_ino,
                    "failed lock",
                )
                _postcheck_namespace_hook_antigravity(
                    "failed lock", parent_descriptor, path.name
                )
                claim_name, claim_descriptor, _ = _claim_entry_antigravity(
                    parent_descriptor,
                    path.name,
                    retained.st_dev,
                    retained.st_ino,
                    "failed lock",
                )
                try:
                    _delete_claimed_entry_antigravity(
                        parent_descriptor,
                        claim_name,
                        claim_descriptor,
                        retained.st_dev,
                        retained.st_ino,
                        "failed lock",
                    )
                finally:
                    os.close(claim_descriptor)
            except (NameError, OSError, ValueError):
                pass
            raise
        return _LockTokenAntigravity(
            path,
            payload,
            opened.st_dev,
            opened.st_ino,
            descriptor,
            parent,
            parent_descriptor,
            parent_opened.st_dev,
            parent_opened.st_ino,
        )
    except Exception:
        os.close(parent_descriptor)
        raise


def _release_report_lock_antigravity(token: _LockTokenAntigravity) -> None:
    try:
        _validate_directory_anchor_antigravity(
            token.parent_path,
            token.parent_descriptor,
            token.parent_device,
            token.parent_inode,
        )
        value, raw, opened = _read_lock_antigravity(token.path)
        del value
        retained = os.fstat(token.descriptor)
        if not (
            opened.st_dev == token.device == retained.st_dev
            and opened.st_ino == token.inode == retained.st_ino
            and raw == token.payload
        ):
            raise ValueError("Exp471 report lock identity changed")
        _require_entry_identity_antigravity(
            token.parent_descriptor,
            token.path.name,
            token.device,
            token.inode,
            "lock release",
        )
        _postcheck_namespace_hook_antigravity(
            "lock release", token.parent_descriptor, token.path.name
        )
        claim_name, claim_descriptor, claim_opened = _claim_entry_antigravity(
            token.parent_descriptor,
            token.path.name,
            token.device,
            token.inode,
            "lock release",
        )
        try:
            claim_path = token.parent_path / claim_name
            claimed_value, claimed_raw, claimed = _read_lock_antigravity(claim_path)
            del claimed_value
            if (
                claim_opened.st_dev != claimed.st_dev
                or claim_opened.st_ino != claimed.st_ino
                or claimed_raw != token.payload
            ):
                raise ValueError("Exp471 report lock claim identity changed")
            _delete_claimed_entry_antigravity(
                token.parent_descriptor,
                claim_name,
                claim_descriptor,
                token.device,
                token.inode,
                "lock release",
            )
        finally:
            os.close(claim_descriptor)
        os.fsync(token.parent_descriptor)
    except FileNotFoundError:
        return
    finally:
        os.close(token.descriptor)
        os.close(token.parent_descriptor)


def _reserve_local_publication_antigravity(root: Path) -> bool:
    deadline = monotonic() + 30.0
    with _LOCAL_PUBLICATION_CONDITION_ANTIGRAVITY:
        while root in _LOCAL_PUBLICATION_ROOTS_ANTIGRAVITY:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise ValueError("Exp471 report lock has an active owner")
            _LOCAL_PUBLICATION_CONDITION_ANTIGRAVITY.wait(
                timeout=min(0.1, remaining)
            )
            if os.path.lexists(root):
                return False
        _LOCAL_PUBLICATION_ROOTS_ANTIGRAVITY.add(root)
        return True


def _release_local_publication_antigravity(root: Path) -> None:
    with _LOCAL_PUBLICATION_CONDITION_ANTIGRAVITY:
        _LOCAL_PUBLICATION_ROOTS_ANTIGRAVITY.discard(root)
        _LOCAL_PUBLICATION_CONDITION_ANTIGRAVITY.notify_all()


def _create_stage_token_antigravity(
    root: Path, token: _LockTokenAntigravity
) -> _StageTokenAntigravity:
    _validate_directory_anchor_antigravity(
        token.parent_path,
        token.parent_descriptor,
        token.parent_device,
        token.parent_inode,
    )
    for _ in range(128):
        name = f".{root.name}.stage_antigravity_{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=token.parent_descriptor)
        except FileExistsError:
            continue
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(name, flags, dir_fd=token.parent_descriptor)
        opened = os.fstat(descriptor)
        _require_entry_identity_antigravity(
            token.parent_descriptor,
            name,
            opened.st_dev,
            opened.st_ino,
            "staging",
        )
        return _StageTokenAntigravity(
            token.parent_path / name,
            descriptor,
            opened.st_dev,
            opened.st_ino,
        )
    raise FileExistsError("Exp471 report staging name space is exhausted")


def _cleanup_stage_antigravity(
    stage: _StageTokenAntigravity, token: _LockTokenAntigravity
) -> None:
    _validate_directory_anchor_antigravity(
        token.parent_path,
        token.parent_descriptor,
        token.parent_device,
        token.parent_inode,
    )
    _require_entry_identity_antigravity(
        token.parent_descriptor,
        stage.path.name,
        stage.device,
        stage.inode,
        "staging cleanup",
    )
    retained = os.fstat(stage.descriptor)
    if retained.st_dev != stage.device or retained.st_ino != stage.inode:
        raise ValueError("Exp471 report staging cleanup identity changed")

    def clear_directory(descriptor: int) -> None:
        entries = tuple(os.scandir(descriptor))
        for entry in entries:
            details = os.stat(
                entry.name, dir_fd=descriptor, follow_symlinks=False
            )
            if stat.S_ISREG(details.st_mode):
                _postcheck_namespace_hook_antigravity(
                    "staging regular child", descriptor, entry.name
                )
                claim_name, claim_descriptor, _ = _claim_entry_antigravity(
                    descriptor,
                    entry.name,
                    details.st_dev,
                    details.st_ino,
                    "staging regular child",
                )
                try:
                    _delete_claimed_entry_antigravity(
                        descriptor,
                        claim_name,
                        claim_descriptor,
                        details.st_dev,
                        details.st_ino,
                        "staging regular child",
                    )
                finally:
                    os.close(claim_descriptor)
                continue
            if not stat.S_ISDIR(details.st_mode):
                _postcheck_namespace_hook_antigravity(
                    "staging unsafe child", descriptor, entry.name
                )
                _claim_entry_antigravity(
                    descriptor,
                    entry.name,
                    details.st_dev,
                    details.st_ino,
                    "staging unsafe child",
                )
                raise ValueError("Exp471 report staging cleanup found an unsafe entry")
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            child = os.open(entry.name, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if opened.st_dev != details.st_dev or opened.st_ino != details.st_ino:
                    raise ValueError(
                        "Exp471 report staging child identity changed"
                    )
                clear_directory(child)
                os.fsync(child)
                _require_entry_identity_antigravity(
                    descriptor,
                    entry.name,
                    details.st_dev,
                    details.st_ino,
                    "staging directory child",
                )
                _postcheck_namespace_hook_antigravity(
                    "staging directory child", descriptor, entry.name
                )
                claim_name, claim_descriptor, _ = _claim_entry_antigravity(
                    descriptor,
                    entry.name,
                    details.st_dev,
                    details.st_ino,
                    "staging directory child",
                )
                try:
                    claimed = os.fstat(claim_descriptor)
                    retained = os.fstat(child)
                    if (
                        claimed.st_dev != retained.st_dev
                        or claimed.st_ino != retained.st_ino
                    ):
                        raise ValueError(
                            "Exp471 report staging child identity changed"
                        )
                    _delete_claimed_entry_antigravity(
                        descriptor,
                        claim_name,
                        claim_descriptor,
                        details.st_dev,
                        details.st_ino,
                        "staging directory child",
                    )
                finally:
                    os.close(claim_descriptor)
            finally:
                os.close(child)

    clear_directory(stage.descriptor)
    os.fsync(stage.descriptor)
    _require_entry_identity_antigravity(
        token.parent_descriptor,
        stage.path.name,
        stage.device,
        stage.inode,
        "staging cleanup",
    )
    _postcheck_namespace_hook_antigravity(
        "staging cleanup", token.parent_descriptor, stage.path.name
    )
    claim_name, claim_descriptor, _ = _claim_entry_antigravity(
        token.parent_descriptor,
        stage.path.name,
        stage.device,
        stage.inode,
        "staging cleanup",
    )
    try:
        claimed = os.fstat(claim_descriptor)
        retained = os.fstat(stage.descriptor)
        if claimed.st_dev != retained.st_dev or claimed.st_ino != retained.st_ino:
            raise ValueError("Exp471 report staging cleanup identity changed")
        _delete_claimed_entry_antigravity(
            token.parent_descriptor,
            claim_name,
            claim_descriptor,
            stage.device,
            stage.inode,
            "staging cleanup",
        )
    finally:
        os.close(claim_descriptor)
    os.fsync(token.parent_descriptor)


def _read_report_for_worker_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    root: Path,
    worker: Mapping[str, object],
) -> Exp471ReportAuthorityAntigravity:
    authority = read_exp471_report_antigravity(artifact, root)
    provenance, _ = _read_json_leaf_antigravity(
        root, _PROVENANCE_NAME_ANTIGRAVITY
    )
    typed_leaves = expected_typed_exp471_report_leaves_antigravity(artifact)
    expected = _provenance_antigravity(
        artifact,
        worker,
        {
            relative: sha256(payload).hexdigest()
            for relative, payload in typed_leaves.items()
        },
    )
    if provenance != expected:
        raise ValueError("Exp471 report existing worker provenance differs")
    return authority


def _dependency_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _report_closure_antigravity(
    *, read_bytes: Callable[[Path], bytes] | None = None
) -> dict[str, object]:
    """Bind only source modules that execute while deriving typed report bytes."""

    repository = _repository_root_antigravity()
    reader = Path.read_bytes if read_bytes is None else read_bytes
    files: dict[str, str] = {}
    for relative in _REPORT_CLOSURE_PATHS_ANTIGRAVITY:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Exp471 report code closure is incomplete")
        payload = reader(path)
        if not isinstance(payload, bytes):
            raise ValueError("Exp471 report code closure member bytes differ")
        files[relative] = sha256(payload).hexdigest()
    bare = {
        "schema": "exp471-report-code-closure-antigravity.v1",
        "file_sha256": files,
    }
    return {
        **bare,
        "closure_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    }


def _safe_relative_antigravity(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        value
        and not path.is_absolute()
        and "\\" not in value
        and "\x00" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def _regular_leaf_bytes_antigravity(root: Path, relative: str) -> bytes:
    if not _safe_relative_antigravity(relative):
        raise ValueError("Exp471 report artifact path escapes its root")
    path = root.joinpath(*relative.split("/"))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Exp471 report leaf is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_dev != linked.st_dev
            or opened.st_ino != linked.st_ino
            or opened.st_nlink != 1
        ):
            raise ValueError("Exp471 report leaf is unsafe")
        return _read_descriptor_antigravity(descriptor)
    finally:
        os.close(descriptor)


def _leaf_names_antigravity(root: Path) -> tuple[str, ...]:
    names: list[str] = []
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(subdirectories):
            path = directory_path / name
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise ValueError("Exp471 report contains an unsafe directory")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            _regular_leaf_bytes_antigravity(root, relative)
            names.append(relative)
    if len(names) != len(set(names)):
        raise ValueError("Exp471 report contains duplicate leaves")
    return tuple(sorted(names))


def _create_leaf_antigravity(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise ValueError("Exp471 report staging leaf already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all_antigravity(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _provenance_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    worker_provenance: Mapping[str, object],
    rendered_leaf_sha256: Mapping[str, str],
) -> dict[str, object]:
    source = artifact.payload["source_authority"]
    cache = artifact.payload["score_cache"]
    implementation = artifact.payload["implementation"]
    if not all(
        isinstance(value, Mapping)
        for value in (source, cache, implementation, worker_provenance)
    ):
        raise ValueError("Exp471 report provenance inputs differ")
    return {
        "schema": _PROVENANCE_SCHEMA_ANTIGRAVITY,
        "experiment": "Exp471",
        "typed_reporter": (
            "experiments.exp471_exp467_evaluation_recovery_antigravity."
            "typed_abstain_reporting_antigravity"
        ),
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "curve_axes": {
            "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
            "precision_recall": {"x": "recall", "y": "precision"},
        },
        "source_authority": {
            "source_context_sha256": source["source_context_sha256"],
            "inventory_sha256": source["inventory_sha256"],
            "authority_sha256": source["authority_sha256"],
        },
        "score_cache": {
            "cache_context_sha256": cache["cache_context_sha256"],
            "manifest_sha256": cache["manifest_sha256"],
            "total_shards": cache["total_shards"],
            "scored_shards": cache["scored_shards"],
            "abstained_shards": cache["abstained_shards"],
            "failed_shards": cache["failed_shards"],
        },
        "evaluation": {
            "evaluation_sha256": artifact.evaluation_sha256,
            "code_closure_sha256": artifact.code_closure_sha256,
        },
        "report_code_closure": _report_closure_antigravity(),
        "interpreter": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "dependencies": {
            "numpy": _dependency_version_antigravity("numpy"),
            "scikit-learn": _dependency_version_antigravity("scikit-learn"),
            "scipy": _dependency_version_antigravity("scipy"),
        },
        "evaluation_implementation": implementation,
        "support_authority": json.loads(
            canonical_json_bytes_exp471_antigravity(
                artifact.payload["support_authority"]
            ).decode("utf-8")
        ),
        "worker_provenance": worker_provenance,
        "rendered_leaf_sha256": dict(sorted(rendered_leaf_sha256.items())),
    }


def _authority_record_antigravity(
    authority: Exp471ReportAuthorityAntigravity,
) -> dict[str, object]:
    return {
        "schema": _REPORT_SCHEMA_ANTIGRAVITY,
        "source_authority_sha256": authority.source_authority_sha256,
        "cache_manifest_sha256": authority.cache_manifest_sha256,
        "evaluation_sha256": authority.evaluation_sha256,
        "report_file_sha256": dict(authority.report_file_sha256),
        "report_manifest_sha256": authority.report_manifest_sha256,
        "artifact_map": _ARTIFACT_MAP_NAME_ANTIGRAVITY,
    }


def _artifact_map_bytes_antigravity(files: Mapping[str, str]) -> bytes:
    return canonical_json_bytes_exp471_antigravity(
        {
            "schema": _ARTIFACT_MAP_SCHEMA_ANTIGRAVITY,
            "self_excluded": _ARTIFACT_MAP_NAME_ANTIGRAVITY,
            "files": dict(sorted(files.items())),
        }
    )


def _validate_worker_provenance_antigravity(
    value: object, artifact: Exp471EvaluationArtifactAntigravity
) -> Mapping[str, object]:
    from . import score_cache_antigravity as score_cache
    from .config_antigravity import (
        ACCELERATION_ASSESSMENT_ANTIGRAVITY,
        CPU_WORKERS_ANTIGRAVITY,
        INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        LOGICAL_CPU_COUNT_ANTIGRAVITY,
    )

    if not isinstance(value, Mapping):
        raise ValueError("Exp471 worker provenance is missing")
    raw = canonical_json_bytes_exp471_antigravity(value)
    plain = json.loads(raw.decode("utf-8"))
    required = {
        "schema",
        "benchmark_decision",
        "execution_decision",
        "detected_logical_cpus",
        "default_cpu_workers",
        "inner_numerical_threads",
        "acceleration",
    }
    if (
        not isinstance(plain, dict)
        or set(plain) != required
        or plain.get("schema") != "exp471-worker-provenance-antigravity.v1"
    ):
        raise ValueError("Exp471 worker provenance is invalid")
    cache = artifact.payload.get("score_cache")
    if not isinstance(cache, Mapping):
        raise ValueError("Exp471 worker provenance cache binding is missing")
    benchmark = score_cache._validated_benchmark_decision_antigravity(
        plain["benchmark_decision"],
        expected_context_sha256=cache.get("cache_context_sha256"),
    )
    execution = plain["execution_decision"]
    execution_required = {
        "schema",
        "cache_context_sha256",
        "benchmark_decision_sha256",
        "selected_workers",
        "fallback_occurred",
        "events",
    }
    if (
        not isinstance(execution, dict)
        or set(execution) != execution_required
        or execution.get("schema")
        != "exp471-score-preparation-execution-antigravity.v1"
        or execution.get("cache_context_sha256")
        != benchmark["cache_context_sha256"]
        or execution.get("benchmark_decision_sha256")
        != benchmark["decision_sha256"]
        or not isinstance(execution.get("events"), list)
        or not execution["events"]
    ):
        raise ValueError("Exp471 worker provenance execution decision differs")
    checked_events: list[dict[str, object]] = []
    prior: str | None = None
    for sequence, event in enumerate(execution["events"]):
        checked = score_cache._validated_execution_event_antigravity(
            event,
            benchmark,
            expected_sequence=sequence,
            expected_prior_sha256=prior,
        )
        checked_events.append(checked)
        prior = checked["execution_sha256"]
    expected_execution = json.loads(
        canonical_json_bytes_exp471_antigravity(
            score_cache._execution_summary_antigravity(
                benchmark, checked_events
            )
        ).decode("utf-8")
    )
    if execution != expected_execution:
        raise ValueError("Exp471 worker provenance execution summary differs")
    if (
        plain["detected_logical_cpus"] != LOGICAL_CPU_COUNT_ANTIGRAVITY
        or plain["default_cpu_workers"] != CPU_WORKERS_ANTIGRAVITY
        or plain["inner_numerical_threads"]
        != INNER_NUMERICAL_THREADS_ANTIGRAVITY
        or plain["acceleration"] != dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY)
        or plain["detected_logical_cpus"]
        != benchmark["detected_logical_cpus"]
        or plain["default_cpu_workers"] != benchmark["default_cpu_workers"]
        or plain["inner_numerical_threads"]
        != benchmark["inner_numerical_threads"]
        or plain["acceleration"] != benchmark["acceleration"]
    ):
        raise ValueError("Exp471 worker provenance policy differs")
    return plain


def _durable_worker_provenance_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> Mapping[str, object]:
    """Reopen Task 5 benchmark/event authority independently of report bytes."""

    from .score_cache_antigravity import status_score_cache_antigravity

    status_value = status_score_cache_antigravity(
        _TASK5_SCORE_CACHE_ROOT_ANTIGRAVITY
    )
    cache = artifact.payload.get("score_cache")
    if not isinstance(cache, Mapping):
        raise ValueError("Exp471 durable worker authority cache binding is missing")
    if (
        status_value.get("complete") is not True
        or status_value.get("manifest_sha256") != artifact.cache_manifest_sha256
        or status_value.get("manifest_sha256") != cache.get("manifest_sha256")
        or status_value.get("total_shards") != cache.get("total_shards")
        or status_value.get("benchmark_decision") is None
        or status_value.get("execution_decision") is None
    ):
        raise ValueError("Exp471 durable worker authority differs")
    worker = {
        "schema": "exp471-worker-provenance-antigravity.v1",
        "benchmark_decision": status_value["benchmark_decision"],
        "execution_decision": status_value["execution_decision"],
        "detected_logical_cpus": status_value["detected_logical_cpus"],
        "default_cpu_workers": status_value["default_cpu_workers"],
        "inner_numerical_threads": status_value["inner_numerical_threads"],
        "acceleration": status_value["acceleration"],
    }
    return _validate_worker_provenance_antigravity(worker, artifact)


def _read_json_leaf_antigravity(root: Path, relative: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular_leaf_bytes_antigravity(root, relative)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp471 report JSON leaf is malformed") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes_exp471_antigravity(value):
        raise ValueError("Exp471 report JSON leaf is noncanonical")
    return value, raw


def read_exp471_report_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    root: Path,
) -> Exp471ReportAuthorityAntigravity:
    """Reopen every final leaf and verify its complete authority map."""

    expected_typed_leaves = expected_typed_exp471_report_leaves_antigravity(artifact)
    root = Path(root)
    _require_safe_parent_antigravity(root)
    if root.is_symlink():
        raise ValueError("Exp471 report root is a symlink")
    if not root.is_dir():
        raise FileNotFoundError("Exp471 report is absent")
    before = root.lstat()
    marker, marker_raw = _read_json_leaf_antigravity(
        root, _COMPLETE_MARKER_NAME_ANTIGRAVITY
    )
    del marker_raw
    required_marker = {
        "schema",
        "source_authority_sha256",
        "cache_manifest_sha256",
        "evaluation_sha256",
        "report_file_sha256",
        "report_manifest_sha256",
        "artifact_map",
    }
    if (
        set(marker) != required_marker
        or marker.get("schema") != _REPORT_SCHEMA_ANTIGRAVITY
        or marker.get("artifact_map") != _ARTIFACT_MAP_NAME_ANTIGRAVITY
    ):
        raise ValueError("Exp471 report complete marker differs")
    try:
        authority = Exp471ReportAuthorityAntigravity(
            source_authority_sha256=marker["source_authority_sha256"],
            cache_manifest_sha256=marker["cache_manifest_sha256"],
            evaluation_sha256=marker["evaluation_sha256"],
            report_file_sha256=marker["report_file_sha256"],
            report_manifest_sha256=marker["report_manifest_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 report complete marker fields differ") from error
    if (
        authority.source_authority_sha256 != artifact.source_authority_sha256
        or authority.cache_manifest_sha256 != artifact.cache_manifest_sha256
        or authority.evaluation_sha256 != artifact.evaluation_sha256
    ):
        raise ValueError("Exp471 report evaluation authority differs")
    artifact_map, artifact_map_raw = _read_json_leaf_antigravity(
        root, _ARTIFACT_MAP_NAME_ANTIGRAVITY
    )
    if (
        set(artifact_map) != {"schema", "self_excluded", "files"}
        or artifact_map.get("schema") != _ARTIFACT_MAP_SCHEMA_ANTIGRAVITY
        or artifact_map.get("self_excluded") != _ARTIFACT_MAP_NAME_ANTIGRAVITY
        or not isinstance(artifact_map.get("files"), dict)
        or artifact_map["files"] != dict(authority.report_file_sha256)
        or sha256(artifact_map_raw).hexdigest()
        != authority.report_manifest_sha256
    ):
        raise ValueError("Exp471 report artifact map digest differs")
    actual_names = set(_leaf_names_antigravity(root))
    expected_names = set(authority.report_file_sha256) | {
        _ARTIFACT_MAP_NAME_ANTIGRAVITY,
        _COMPLETE_MARKER_NAME_ANTIGRAVITY,
    }
    if actual_names != expected_names:
        raise ValueError("Exp471 report leaf inventory differs")
    if set(authority.report_file_sha256) != {
        *expected_typed_leaves,
        _PROVENANCE_NAME_ANTIGRAVITY,
    }:
        raise ValueError("Exp471 report fixed leaf inventory differs")
    for relative, expected in authority.report_file_sha256.items():
        if not _safe_relative_antigravity(relative):
            raise ValueError("Exp471 report artifact path escapes its root")
        observed = sha256(_regular_leaf_bytes_antigravity(root, relative)).hexdigest()
        if observed != expected:
            raise ValueError(f"Exp471 report leaf digest differs: {relative}")
    for relative, expected_bytes in expected_typed_leaves.items():
        if _regular_leaf_bytes_antigravity(root, relative) != expected_bytes:
            raise ValueError(
                f"Exp471 typed report leaf semantic bytes differ: {relative}"
            )
    provenance, _ = _read_json_leaf_antigravity(root, _PROVENANCE_NAME_ANTIGRAVITY)
    stored_worker = _validate_worker_provenance_antigravity(
        provenance.get("worker_provenance"), artifact
    )
    worker = _durable_worker_provenance_antigravity(artifact)
    if stored_worker != worker:
        raise ValueError("Exp471 report durable worker authority differs")
    expected_provenance = _provenance_antigravity(
        artifact,
        worker,
        {
            relative: sha256(payload).hexdigest()
            for relative, payload in expected_typed_leaves.items()
        },
    )
    if provenance != expected_provenance:
        raise ValueError("Exp471 report provenance differs")
    after = root.lstat()
    if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
        raise ValueError("Exp471 report root changed during verification")
    return authority


def _fsync_tree_antigravity(root: Path) -> None:
    directories: list[Path] = []
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories.append(directory_path)
        for name in subdirectories:
            path = directory_path / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Exp471 report staging directory is unsafe")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Exp471 report staging leaf is unsafe")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory_antigravity(directory)


def write_exp471_report_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    root: Path,
    *,
    worker_provenance: Mapping[str, object] | None = None,
    renderer: Callable[[Exp471EvaluationArtifactAntigravity, Path], Exp471TypedReportArtifactAntigravity] | None = None,
    fault: Callable[[str], None] | None = None,
) -> Exp471ReportAuthorityAntigravity:
    """Render once, verify all leaves, and atomically publish a complete report."""

    durable_worker = _durable_worker_provenance_antigravity(artifact)
    if worker_provenance is None:
        worker = durable_worker
    else:
        supplied_worker = _validate_worker_provenance_antigravity(
            worker_provenance, artifact
        )
        if supplied_worker != durable_worker:
            raise ValueError(
                "Exp471 supplied worker provenance differs from durable authority"
            )
        worker = durable_worker
    root = Path(root)
    _require_safe_parent_antigravity(root)
    if os.path.lexists(root):
        if root.is_symlink():
            raise ValueError("Exp471 report root is a symlink")
        try:
            return _read_report_for_worker_antigravity(
                artifact, root, worker
            )
        except FileNotFoundError:
            raise
        except ValueError as error:
            if not (root / _COMPLETE_MARKER_NAME_ANTIGRAVITY).is_file():
                raise ValueError(
                    "Exp471 report destination is markerless or foreign"
                ) from error
            raise

    local_owner = _reserve_local_publication_antigravity(root)
    if not local_owner:
        return _read_report_for_worker_antigravity(artifact, root, worker)
    token: _LockTokenAntigravity | None = None
    stage: _StageTokenAntigravity | None = None
    published = False
    try:
        token = _acquire_report_lock_antigravity(root)
        if os.path.lexists(root):
            return _read_report_for_worker_antigravity(artifact, root, worker)
        stage = _create_stage_token_antigravity(root, token)
        active_renderer = (
            write_typed_exp471_report_antigravity if renderer is None else renderer
        )
        rendered = active_renderer(artifact, stage.path)
        if (
            not isinstance(rendered, Exp471TypedReportArtifactAntigravity)
            or rendered.root.parent != stage.path
            or rendered.root.name != "exp471_typed_report_artifacts_antigravity"
        ):
            raise ValueError("Exp471 typed reporter returned an unsafe root")
        rendered_names = _leaf_names_antigravity(stage.path)
        if not rendered_names:
            raise ValueError("Exp471 typed reporter produced no leaves")
        rendered_digests = {
            relative: sha256(
                _regular_leaf_bytes_antigravity(stage.path, relative)
            ).hexdigest()
            for relative in rendered_names
        }
        expected_typed_leaves = expected_typed_exp471_report_leaves_antigravity(
            artifact
        )
        expected_digests = {
            relative: sha256(payload).hexdigest()
            for relative, payload in expected_typed_leaves.items()
        }
        if rendered_digests != expected_digests or dict(rendered.leaves) != expected_digests:
            raise ValueError("Exp471 typed reporter leaf digest differs")
        provenance = _provenance_antigravity(
            artifact, worker, rendered_digests
        )
        _create_leaf_antigravity(
            stage.path / _PROVENANCE_NAME_ANTIGRAVITY,
            canonical_json_bytes_exp471_antigravity(provenance),
        )
        report_files = dict(rendered_digests)
        report_files[_PROVENANCE_NAME_ANTIGRAVITY] = sha256(
            _regular_leaf_bytes_antigravity(
                stage.path, _PROVENANCE_NAME_ANTIGRAVITY
            )
        ).hexdigest()
        artifact_map_raw = _artifact_map_bytes_antigravity(report_files)
        _create_leaf_antigravity(
            stage.path / _ARTIFACT_MAP_NAME_ANTIGRAVITY, artifact_map_raw
        )
        authority = Exp471ReportAuthorityAntigravity(
            source_authority_sha256=artifact.source_authority_sha256,
            cache_manifest_sha256=artifact.cache_manifest_sha256,
            evaluation_sha256=artifact.evaluation_sha256,
            report_file_sha256=report_files,
            report_manifest_sha256=sha256(artifact_map_raw).hexdigest(),
        )
        _create_leaf_antigravity(
            stage.path / _COMPLETE_MARKER_NAME_ANTIGRAVITY,
            canonical_json_bytes_exp471_antigravity(
                _authority_record_antigravity(authority)
            ),
        )
        _fsync_tree_antigravity(stage.path)
        staged = _read_report_for_worker_antigravity(
            artifact, stage.path, worker
        )
        if staged != authority:
            raise ValueError("Exp471 staged report authority differs")
        if fault is not None:
            fault("before_rename")
        _validate_directory_anchor_antigravity(
            token.parent_path,
            token.parent_descriptor,
            token.parent_device,
            token.parent_inode,
        )
        retained = os.fstat(stage.descriptor)
        if retained.st_dev != stage.device or retained.st_ino != stage.inode:
            raise ValueError("Exp471 report staging identity changed")
        _require_entry_identity_antigravity(
            token.parent_descriptor,
            stage.path.name,
            stage.device,
            stage.inode,
            "staging before rename",
        )
        _postcheck_namespace_hook_antigravity(
            "staging before rename", token.parent_descriptor, stage.path.name
        )
        renamed_to_canonical = False
        try:
            claim_name, claim_descriptor, claim_opened = _claim_entry_antigravity(
                token.parent_descriptor,
                stage.path.name,
                stage.device,
                stage.inode,
                "staging before rename",
            )
            try:
                retained = os.fstat(stage.descriptor)
                if (
                    claim_opened.st_dev != retained.st_dev
                    or claim_opened.st_ino != retained.st_ino
                ):
                    raise ValueError("Exp471 report staging claim identity changed")
            finally:
                os.close(claim_descriptor)
            stage = _StageTokenAntigravity(
                token.parent_path / claim_name,
                stage.descriptor,
                stage.device,
                stage.inode,
            )
            _require_entry_identity_antigravity(
                token.parent_descriptor,
                claim_name,
                stage.device,
                stage.inode,
                "staging publication claim",
            )
            _final_claim_namespace_hook_antigravity(
                "staging publication claim",
                token.parent_descriptor,
                claim_name,
            )
            _atomic_rename_noreplace_at_antigravity(
                token.parent_descriptor, claim_name, root.name
            )
            renamed_to_canonical = True
            _require_entry_identity_antigravity(
                token.parent_descriptor,
                root.name,
                stage.device,
                stage.inode,
                "published root",
            )
            published = True
        except BaseException:
            if renamed_to_canonical:
                try:
                    _entry_stat_antigravity(token.parent_descriptor, root.name)
                except FileNotFoundError:
                    pass
                else:
                    _evacuate_canonical_entry_antigravity(
                        token.parent_descriptor, root.name
                    )
                try:
                    _entry_stat_antigravity(token.parent_descriptor, root.name)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError(
                        "Exp471 report canonical entry remained after evacuation"
                    )
                os.fsync(token.parent_descriptor)
                raise
            if os.path.lexists(root):
                return _read_report_for_worker_antigravity(
                    artifact, root, worker
                )
            raise
        os.fsync(token.parent_descriptor)
        try:
            if fault is not None:
                fault("after_rename")
        except Exception:
            return _read_report_for_worker_antigravity(
                artifact, root, worker
            )
        return _read_report_for_worker_antigravity(artifact, root, worker)
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error: Exception | None = None
        if stage is not None:
            try:
                if not published:
                    _cleanup_stage_antigravity(stage, token)
            except Exception as error:
                cleanup_error = error
            finally:
                os.close(stage.descriptor)
        if token is not None:
            try:
                _release_report_lock_antigravity(token)
            except Exception as error:
                if cleanup_error is None:
                    cleanup_error = error
        _release_local_publication_antigravity(root)
        if cleanup_error is not None and not active_error:
            raise cleanup_error


__all__ = [
    "read_exp471_report_antigravity",
    "write_exp471_report_antigravity",
]
