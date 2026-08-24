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
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
    Exp450ReportArtifactAntigravity,
    write_exp450_report_antigravity,
)

from .config_antigravity import METHOD_ORDER_ANTIGRAVITY
from .contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    Exp471ReportAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from .evaluation_antigravity import (
    reconstruct_frozen_evaluation_exp471_antigravity,
)


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE_MARKER_NAME_ANTIGRAVITY = (
    ".exp471_report_complete_antigravity.json"
)
_ARTIFACT_MAP_NAME_ANTIGRAVITY = "artifact_sha256_antigravity.json"
_PROVENANCE_NAME_ANTIGRAVITY = "exp471_report_provenance_antigravity.json"
_REPORT_SCHEMA_ANTIGRAVITY = "exp471-report-publication-antigravity.v1"
_ARTIFACT_MAP_SCHEMA_ANTIGRAVITY = "exp471-report-artifact-map-antigravity.v1"
_PROVENANCE_SCHEMA_ANTIGRAVITY = "exp471-report-provenance-antigravity.v1"
_LOCK_SCHEMA_ANTIGRAVITY = "exp471-report-lock-antigravity.v1"
_RENAME_NOREPLACE_ANTIGRAVITY = 1
_RENAME_EXCL_ANTIGRAVITY = 0x00000004

_REPORT_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp450_resampling_image_ablation_antigravity/"
    "reporting_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "reporting_antigravity.py",
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


def _atomic_rename_noreplace_antigravity(source: Path, destination: Path) -> None:
    if source.parent != destination.parent:
        raise ValueError("Exp471 report rename source is not a sibling")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "Exp471 atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(
            source_bytes,
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
            -100,
            source_bytes,
            -100,
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


def _recover_dead_lock_antigravity(
    path: Path,
    value: Mapping[str, object],
    raw: bytes,
    opened: os.stat_result,
) -> None:
    current_identity = _process_start_identity_antigravity(int(value["pid"]))
    if current_identity == value["process_start_identity"]:
        raise ValueError("Exp471 report lock has an active owner")
    if current_identity == "unknown-live-process-antigravity":
        raise ValueError("Exp471 report lock owner cannot be proven dead")
    linked = path.lstat()
    if linked.st_dev != opened.st_dev or linked.st_ino != opened.st_ino:
        raise ValueError("Exp471 report lock changed during recovery")
    quarantine = path.with_name(
        f".{path.name}.{secrets.token_hex(12)}.dead_antigravity"
    )
    _atomic_rename_noreplace_antigravity(path, quarantine)
    try:
        moved_value, moved_raw, moved = _read_lock_antigravity(quarantine)
        if (
            moved.st_dev != opened.st_dev
            or moved.st_ino != opened.st_ino
            or moved_raw != raw
            or moved_value != value
        ):
            raise ValueError("Exp471 report dead lock identity changed")
        quarantine.unlink()
        _fsync_directory_antigravity(path.parent)
    except Exception:
        # A record whose identity changed is preserved under quarantine.
        raise


def _acquire_report_lock_antigravity(root: Path) -> _LockTokenAntigravity:
    parent = _require_safe_parent_antigravity(root)
    path = _report_lock_path_antigravity(root)
    if os.path.lexists(path):
        if path.is_symlink():
            raise ValueError("Exp471 report lock is a symlink")
        value, raw, opened = _read_lock_antigravity(path)
        _recover_dead_lock_antigravity(path, value, raw, opened)
    identity = _process_start_identity_antigravity(os.getpid())
    if identity is None or identity == "unknown-live-process-antigravity":
        raise ValueError("Exp471 current process start identity is unavailable")
    payload = canonical_json_bytes_exp471_antigravity(
        _lock_record_antigravity(os.getpid(), identity)
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError("Exp471 report lock has an active owner") from error
    try:
        _write_all_antigravity(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            linked = path.lstat()
            if linked.st_dev == opened.st_dev and linked.st_ino == opened.st_ino:
                path.unlink()
        except (NameError, OSError):
            pass
        raise
    else:
        os.close(descriptor)
    _fsync_directory_antigravity(parent)
    return _LockTokenAntigravity(path, payload, opened.st_dev, opened.st_ino)


def _release_report_lock_antigravity(token: _LockTokenAntigravity) -> None:
    try:
        linked = token.path.lstat()
        if linked.st_dev != token.device or linked.st_ino != token.inode:
            return
        value, raw, opened = _read_lock_antigravity(token.path)
        del value
        if (
            opened.st_dev == token.device
            and opened.st_ino == token.inode
            and raw == token.payload
        ):
            token.path.unlink()
            _fsync_directory_antigravity(token.path.parent)
    except FileNotFoundError:
        return


def _dependency_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _report_closure_antigravity() -> dict[str, object]:
    repository = _repository_root_antigravity()
    files: dict[str, str] = {}
    for relative in _REPORT_CLOSURE_PATHS_ANTIGRAVITY:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Exp471 report code closure is incomplete")
        files[relative] = sha256(path.read_bytes()).hexdigest()
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
        "frozen_reporter": (
            "experiments.exp450_resampling_image_ablation_antigravity."
            "reporting_antigravity"
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


def _validate_worker_provenance_antigravity(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("Exp471 worker provenance is missing")
    # Round-trip through the strict canonical encoder to reject non-finite or exotic data.
    raw = canonical_json_bytes_exp471_antigravity(value)
    plain = json.loads(raw.decode("utf-8"))
    if not isinstance(plain, dict) or not plain:
        raise ValueError("Exp471 worker provenance is invalid")
    return plain


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

    reconstruct_frozen_evaluation_exp471_antigravity(artifact)
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
    for relative, expected in authority.report_file_sha256.items():
        if not _safe_relative_antigravity(relative):
            raise ValueError("Exp471 report artifact path escapes its root")
        observed = sha256(_regular_leaf_bytes_antigravity(root, relative)).hexdigest()
        if observed != expected:
            raise ValueError(f"Exp471 report leaf digest differs: {relative}")
    provenance, _ = _read_json_leaf_antigravity(root, _PROVENANCE_NAME_ANTIGRAVITY)
    if (
        provenance.get("schema") != _PROVENANCE_SCHEMA_ANTIGRAVITY
        or provenance.get("method_order") != list(METHOD_ORDER_ANTIGRAVITY)
        or provenance.get("source_authority", {}).get("authority_sha256")
        != artifact.source_authority_sha256
        or provenance.get("score_cache", {}).get("manifest_sha256")
        != artifact.cache_manifest_sha256
        or provenance.get("evaluation", {}).get("evaluation_sha256")
        != artifact.evaluation_sha256
        or provenance.get("report_code_closure") != _report_closure_antigravity()
    ):
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
    renderer: Callable[[object, Path], Exp450ReportArtifactAntigravity] | None = None,
    fault: Callable[[str], None] | None = None,
) -> Exp471ReportAuthorityAntigravity:
    """Render once, verify all leaves, and atomically publish a complete report."""

    frozen = reconstruct_frozen_evaluation_exp471_antigravity(artifact)
    if worker_provenance is None:
        from .config_antigravity import EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY
        from .score_cache_antigravity import status_score_cache_antigravity

        status = status_score_cache_antigravity(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
        worker_provenance = {
            "benchmark_decision": status["benchmark_decision"],
            "execution_decision": status["execution_decision"],
            "detected_logical_cpus": status["detected_logical_cpus"],
            "default_cpu_workers": status["default_cpu_workers"],
            "inner_numerical_threads": status["inner_numerical_threads"],
            "acceleration": status["acceleration"],
        }
    worker = _validate_worker_provenance_antigravity(worker_provenance)
    root = Path(root)
    parent = _require_safe_parent_antigravity(root)
    if os.path.lexists(root):
        if root.is_symlink():
            raise ValueError("Exp471 report root is a symlink")
        try:
            authority = read_exp471_report_antigravity(artifact, root)
        except FileNotFoundError:
            raise
        except ValueError as error:
            if not (root / _COMPLETE_MARKER_NAME_ANTIGRAVITY).is_file():
                raise ValueError(
                    "Exp471 report destination is markerless or foreign"
                ) from error
            raise
        provenance, _ = _read_json_leaf_antigravity(
            root, _PROVENANCE_NAME_ANTIGRAVITY
        )
        if provenance.get("worker_provenance") != worker:
            raise ValueError("Exp471 report existing worker provenance differs")
        return authority

    token = _acquire_report_lock_antigravity(root)
    stage: Path | None = None
    published = False
    try:
        if os.path.lexists(root):
            return read_exp471_report_antigravity(artifact, root)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{root.name}.stage_antigravity_", dir=parent
            )
        )
        active_renderer = (
            write_exp450_report_antigravity if renderer is None else renderer
        )
        rendered = active_renderer(frozen, stage)
        if (
            not isinstance(rendered, Exp450ReportArtifactAntigravity)
            or rendered.root.parent != stage
            or rendered.root.name != "exp450_report_artifacts_antigravity"
        ):
            raise ValueError("Exp471 frozen reporter returned an unsafe root")
        rendered_names = _leaf_names_antigravity(stage)
        if not rendered_names:
            raise ValueError("Exp471 frozen reporter produced no leaves")
        rendered_digests = {
            relative: sha256(
                _regular_leaf_bytes_antigravity(stage, relative)
            ).hexdigest()
            for relative in rendered_names
        }
        provenance = _provenance_antigravity(
            artifact, worker, rendered_digests
        )
        _create_leaf_antigravity(
            stage / _PROVENANCE_NAME_ANTIGRAVITY,
            canonical_json_bytes_exp471_antigravity(provenance),
        )
        report_files = dict(rendered_digests)
        report_files[_PROVENANCE_NAME_ANTIGRAVITY] = sha256(
            _regular_leaf_bytes_antigravity(stage, _PROVENANCE_NAME_ANTIGRAVITY)
        ).hexdigest()
        artifact_map_raw = _artifact_map_bytes_antigravity(report_files)
        _create_leaf_antigravity(
            stage / _ARTIFACT_MAP_NAME_ANTIGRAVITY, artifact_map_raw
        )
        authority = Exp471ReportAuthorityAntigravity(
            source_authority_sha256=artifact.source_authority_sha256,
            cache_manifest_sha256=artifact.cache_manifest_sha256,
            evaluation_sha256=artifact.evaluation_sha256,
            report_file_sha256=report_files,
            report_manifest_sha256=sha256(artifact_map_raw).hexdigest(),
        )
        _create_leaf_antigravity(
            stage / _COMPLETE_MARKER_NAME_ANTIGRAVITY,
            canonical_json_bytes_exp471_antigravity(
                _authority_record_antigravity(authority)
            ),
        )
        _fsync_tree_antigravity(stage)
        staged = read_exp471_report_antigravity(artifact, stage)
        if staged != authority:
            raise ValueError("Exp471 staged report authority differs")
        if fault is not None:
            fault("before_rename")
        try:
            _atomic_rename_noreplace_antigravity(stage, root)
            published = True
        except BaseException:
            if os.path.lexists(root):
                reopened = read_exp471_report_antigravity(artifact, root)
                published = True
                return reopened
            raise
        _fsync_directory_antigravity(parent)
        try:
            if fault is not None:
                fault("after_rename")
        except Exception:
            return read_exp471_report_antigravity(artifact, root)
        return read_exp471_report_antigravity(artifact, root)
    finally:
        if stage is not None and not published and stage.exists():
            shutil.rmtree(stage)
            _fsync_directory_antigravity(parent)
        _release_report_lock_antigravity(token)


__all__ = [
    "read_exp471_report_antigravity",
    "write_exp471_report_antigravity",
]
