"""Create-only readiness → evaluate → inspect lifecycle for Exp454."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Callable, TypeVar

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    strict_json_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import (
    EXP454_EVALUATION_ROOT_ANTIGRAVITY,
    EXP454_FAILURE_AUDIT_ANTIGRAVITY,
    EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY,
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
    RUNTIME_VERSIONS_ANTIGRAVITY,
)


_SCHEMA_ANTIGRAVITY = "exp454_lifecycle_receipt"
_RUN_IDENTITY_ANTIGRAVITY = "exp454-exp453-exp449-authority-boundary-repair-antigravity-20260814"
_SUCCESS_LEAVES_ANTIGRAVITY = (
    "readiness_complete_antigravity.json",
    "evaluate_started_antigravity.json",
    "evaluate_complete_antigravity.json",
    "inspect_started_antigravity.json",
    "inspect_complete_antigravity.json",
)
_SUCCESS_EVENTS_ANTIGRAVITY = (
    ("readiness", "complete"),
    ("evaluate", "started"),
    ("evaluate", "complete"),
    ("inspect", "started"),
    ("inspect", "complete"),
)
_STOPPED_LEAF_ANTIGRAVITY = "stopped_antigravity.json"
_PREDECESSOR_STATE_ANTIGRAVITY = {"readiness": "S0", "evaluate": "S1", "inspect": "S3"}
_STABLE_STATES_ANTIGRAVITY = {0, 1, 3, 5}
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_T = TypeVar("_T")


def _lower_hex_antigravity(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class _LifecyclePathsAntigravity:
    receipt_directory: Path
    audit_path: Path
    evaluation_root: Path
    executable: str
    prefix: str
    python_version: str
    numpy_version: str
    aeon_version: str
    sklearn_version: str

    def __post_init__(self) -> None:
        versions = (self.python_version, self.numpy_version, self.aeon_version, self.sklearn_version)
        if any(
            not isinstance(path, Path)
            for path in (self.receipt_directory, self.audit_path, self.evaluation_root)
        ) or self.executable != FIXED_EXECUTABLE_ANTIGRAVITY or self.prefix != FIXED_PREFIX_ANTIGRAVITY or versions != RUNTIME_VERSIONS_ANTIGRAVITY:
            raise ValueError("lifecycle paths are invalid")


def _observe_exact_runtime_antigravity(*, distribution_version=None) -> tuple[str, str, str, str, str, str]:
    version = importlib.metadata.version if distribution_version is None else distribution_version
    if not callable(version):
        raise ValueError("lifecycle paths are invalid")
    try:
        numpy_version = version("numpy")
        aeon_version = version("aeon")
        sklearn_version = version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("lifecycle paths are invalid") from error
    observed = (
        sys.executable,
        sys.prefix,
        platform.python_version(),
        numpy_version,
        aeon_version,
        sklearn_version,
    )
    if (
        any(type(item) is not str or not item for item in observed)
        or observed[0] != FIXED_EXECUTABLE_ANTIGRAVITY
        or observed[1] != FIXED_PREFIX_ANTIGRAVITY
        or observed[2:] != RUNTIME_VERSIONS_ANTIGRAVITY
    ):
        raise ValueError("lifecycle paths are invalid")
    return observed


def _production_paths_antigravity() -> _LifecyclePathsAntigravity:
    executable, prefix, python_version, numpy_version, aeon_version, sklearn_version = (
        _observe_exact_runtime_antigravity()
    )
    return _LifecyclePathsAntigravity(
        Path(EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY),
        Path(EXP454_FAILURE_AUDIT_ANTIGRAVITY),
        Path(EXP454_EVALUATION_ROOT_ANTIGRAVITY),
        executable,
        prefix,
        python_version,
        numpy_version,
        aeon_version,
        sklearn_version,
    )


@dataclass(frozen=True, slots=True)
class LifecycleReceiptAntigravity:
    schema: str
    experiment_id: str
    run_identity: str
    reviewed_commit: str
    phase: str
    event: str
    ordinal: int
    previous_leaf_sha256: str | None
    evaluation_root_identity: str | None
    permanent_stop: bool
    executable: str
    prefix: str
    python_version: str
    numpy_version: str
    aeon_version: str
    sklearn_version: str

    def __post_init__(self) -> None:
        versions = (self.python_version, self.numpy_version, self.aeon_version, self.sklearn_version)
        if (
            self.schema != _SCHEMA_ANTIGRAVITY
            or self.experiment_id != "Exp454"
            or self.run_identity != _RUN_IDENTITY_ANTIGRAVITY
            or not _lower_hex_antigravity(self.reviewed_commit, 40)
            or self.phase not in _PREDECESSOR_STATE_ANTIGRAVITY
            or self.event not in {"started", "complete", "stopped"}
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or (
                self.previous_leaf_sha256 is not None
                and not _lower_hex_antigravity(self.previous_leaf_sha256, 64)
            )
            or (
                self.evaluation_root_identity is not None
                and not _lower_hex_antigravity(self.evaluation_root_identity, 64)
            )
            or type(self.permanent_stop) is not bool
            or self.executable != FIXED_EXECUTABLE_ANTIGRAVITY
            or self.prefix != FIXED_PREFIX_ANTIGRAVITY
            or versions != RUNTIME_VERSIONS_ANTIGRAVITY
        ):
            raise ValueError("lifecycle receipt fields are invalid")

    def payload_antigravity(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "run_identity": self.run_identity,
            "reviewed_commit": self.reviewed_commit,
            "phase": self.phase,
            "event": self.event,
            "ordinal": self.ordinal,
            "previous_leaf_sha256": self.previous_leaf_sha256,
            "evaluation_root_identity": self.evaluation_root_identity,
            "permanent_stop": self.permanent_stop,
            "executable": self.executable,
            "prefix": self.prefix,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "aeon_version": self.aeon_version,
            "sklearn_version": self.sklearn_version,
        }


@dataclass(frozen=True, slots=True)
class ReceiptClassificationAntigravity:
    state: str
    stopped: bool
    invalid: bool
    reason: str | None
    reviewed_commit: str | None = None
    previous_leaf_sha256: str | None = None
    evaluation_root_identity: str | None = None

    def __post_init__(self) -> None:
        if (
            self.state not in {"S0", "S1", "S2", "S3", "S4", "S5", "STOPPED", "INVALID"}
            or type(self.stopped) is not bool
            or type(self.invalid) is not bool
            or (self.reason is not None and type(self.reason) is not str)
            or (
                self.reviewed_commit is not None
                and not _lower_hex_antigravity(self.reviewed_commit, 40)
            )
            or (
                self.previous_leaf_sha256 is not None
                and not _lower_hex_antigravity(self.previous_leaf_sha256, 64)
            )
            or (
                self.evaluation_root_identity is not None
                and not _lower_hex_antigravity(self.evaluation_root_identity, 64)
            )
        ):
            raise ValueError("receipt classification is invalid")


def _receipt_from_bytes_antigravity(payload: bytes) -> LifecycleReceiptAntigravity:
    value = strict_json_bytes_antigravity(payload)
    fields = {
        "schema", "experiment_id", "run_identity", "reviewed_commit", "phase", "event",
        "ordinal", "previous_leaf_sha256", "evaluation_root_identity", "permanent_stop",
        "executable", "prefix", "python_version", "numpy_version", "aeon_version", "sklearn_version",
    }
    if type(value) is not dict or set(value) != fields or canonical_json_bytes_antigravity(value) != payload:
        raise ValueError("receipt schema differs")
    return LifecycleReceiptAntigravity(**value)


def _read_receipt_antigravity(directory_fd: int, name: str) -> tuple[bytes, LifecycleReceiptAntigravity]:
    descriptor = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("receipt leaf is unsafe")
        payload = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink) != (
            current.st_dev, current.st_ino, current.st_mode, current.st_nlink
        ):
            raise ValueError("receipt leaf changed")
        return payload, _receipt_from_bytes_antigravity(payload)
    finally:
        os.close(descriptor)


def _classify_receipts_antigravity(
    receipt_directory: Path,
    *,
    just_created: bool = False,
    allow_partial: bool = False,
) -> ReceiptClassificationAntigravity:
    try:
        path_info = os.stat(receipt_directory, follow_symlinks=False)
    except FileNotFoundError:
        return ReceiptClassificationAntigravity("S0", False, False, None)
    except OSError as error:
        return ReceiptClassificationAntigravity("INVALID", False, True, f"{type(error).__name__}: {error}")
    descriptor = -1
    try:
        descriptor = os.open(receipt_directory, _DIRECTORY_FLAGS_ANTIGRAVITY)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (path_info.st_dev, path_info.st_ino, path_info.st_mode, path_info.st_nlink)
            != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
        ):
            raise ValueError("receipt directory is unsafe")
        names = set(os.listdir(descriptor))
        if not names:
            if just_created:
                return ReceiptClassificationAntigravity("S0", False, False, None)
            raise ValueError("occupied empty receipt namespace")
        success_count = 0
        while success_count < len(_SUCCESS_LEAVES_ANTIGRAVITY) and _SUCCESS_LEAVES_ANTIGRAVITY[success_count] in names:
            success_count += 1
        expected = set(_SUCCESS_LEAVES_ANTIGRAVITY[:success_count])
        stopped = _STOPPED_LEAF_ANTIGRAVITY in names
        if names != expected | ({_STOPPED_LEAF_ANTIGRAVITY} if stopped else set()):
            raise ValueError("receipt inventory is not an exact contiguous prefix")
        if success_count not in _STABLE_STATES_ANTIGRAVITY and not stopped and not allow_partial:
            raise ValueError("partial receipt namespace is invalid")
        previous: str | None = None
        reviewed: str | None = None
        root_identity: str | None = None
        for ordinal, name in enumerate(_SUCCESS_LEAVES_ANTIGRAVITY[:success_count]):
            payload, receipt = _read_receipt_antigravity(descriptor, name)
            phase, event = _SUCCESS_EVENTS_ANTIGRAVITY[ordinal]
            expected_root = ordinal >= 2
            if (
                (receipt.phase, receipt.event) != (phase, event)
                or receipt.ordinal != ordinal
                or receipt.previous_leaf_sha256 != previous
                or receipt.permanent_stop
                or (expected_root and receipt.evaluation_root_identity is None)
                or (not expected_root and receipt.evaluation_root_identity is not None)
            ):
                raise ValueError("receipt transition differs")
            if reviewed is None:
                reviewed = receipt.reviewed_commit
            elif receipt.reviewed_commit != reviewed:
                raise ValueError("receipt reviewed commit differs")
            if receipt.evaluation_root_identity is not None:
                if root_identity is None:
                    root_identity = receipt.evaluation_root_identity
                elif receipt.evaluation_root_identity != root_identity:
                    raise ValueError("receipt root identity differs")
            previous = hashlib.sha256(payload).hexdigest()
        if stopped:
            payload, terminal = _read_receipt_antigravity(descriptor, _STOPPED_LEAF_ANTIGRAVITY)
            if (
                terminal.event != "stopped"
                or not terminal.permanent_stop
                or terminal.ordinal != success_count
                or terminal.previous_leaf_sha256 != previous
                or (reviewed is not None and terminal.reviewed_commit != reviewed)
                or (
                    root_identity is not None
                    and terminal.evaluation_root_identity != root_identity
                )
            ):
                raise ValueError("terminal receipt differs")
            reviewed = terminal.reviewed_commit
            previous = hashlib.sha256(payload).hexdigest()
            root_identity = terminal.evaluation_root_identity
        current = os.stat(receipt_directory, follow_symlinks=False)
        if (
            (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
            != (current.st_dev, current.st_ino, current.st_mode, current.st_nlink)
            or set(os.listdir(descriptor)) != names
        ):
            raise ValueError("receipt directory changed during classification")
        return ReceiptClassificationAntigravity(
            "STOPPED" if stopped else f"S{success_count}",
            stopped,
            False,
            None,
            reviewed,
            previous,
            root_identity,
        )
    except BaseException as error:
        return ReceiptClassificationAntigravity(
            "INVALID", False, True, f"{type(error).__name__}: {error}"
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _root_identity_antigravity(root: Path) -> str | None:
    try:
        before = os.stat(root, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError("evaluation root is unsafe") from error
    descriptor = -1
    pinned: list[tuple[str, int, tuple[int, int, int, int], bytes]] = []
    try:
        try:
            descriptor = os.open(root, _DIRECTORY_FLAGS_ANTIGRAVITY)
        except OSError as error:
            raise ValueError("evaluation root is unsafe") from error
        opened = os.fstat(descriptor)
        directory_identity = (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink)
        if not stat.S_ISDIR(opened.st_mode) or directory_identity != (
            before.st_dev, before.st_ino, before.st_mode, before.st_nlink
        ):
            raise ValueError("evaluation root is unsafe")
        names = sorted(os.listdir(descriptor))
        if not names:
            raise ValueError("evaluation root is empty")
        digest = hashlib.sha256()
        for name in names:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            leaf = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=descriptor)
            leaf_info = os.fstat(leaf)
            identity = (leaf_info.st_dev, leaf_info.st_ino, leaf_info.st_mode, leaf_info.st_nlink)
            if not stat.S_ISREG(leaf_info.st_mode) or leaf_info.st_nlink != 1 or identity != (
                info.st_dev, info.st_ino, info.st_mode, info.st_nlink
            ):
                os.close(leaf)
                raise ValueError("evaluation root leaf is unsafe")
            payload = b"".join(iter(lambda: os.read(leaf, 1 << 20), b""))
            pinned.append((name, leaf, identity, payload))
            digest.update(name.encode("utf-8") + b"\0" + str(len(payload)).encode("ascii") + b"\0" + payload + b"\0")
        for name, leaf, identity, payload in pinned:
            os.lseek(leaf, 0, os.SEEK_SET)
            if (
                (lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink))(os.fstat(leaf)) != identity
                or (lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink))(os.stat(name, dir_fd=descriptor, follow_symlinks=False)) != identity
                or b"".join(iter(lambda: os.read(leaf, 1 << 20), b"")) != payload
            ):
                raise ValueError("evaluation root changed during capture")
        current = os.stat(root, follow_symlinks=False)
        if directory_identity != (current.st_dev, current.st_ino, current.st_mode, current.st_nlink) or set(os.listdir(descriptor)) != set(names):
            raise ValueError("evaluation root changed during capture")
        return digest.hexdigest()
    finally:
        for _name, leaf, _identity, _payload in reversed(pinned):
            os.close(leaf)
        if descriptor >= 0:
            os.close(descriptor)


def validate_receipt_inventory_antigravity(
    receipt_directory: Path = Path(EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY),
    *,
    evaluation_root: Path | None = None,
) -> ReceiptClassificationAntigravity:
    """Classify a complete owned prefix; partial and unsafe namespaces are invalid."""
    if not isinstance(receipt_directory, Path) or (
        evaluation_root is not None and not isinstance(evaluation_root, Path)
    ):
        raise ValueError("receipt validation paths are invalid")
    classification = _classify_receipts_antigravity(receipt_directory)
    if (
        not classification.invalid
        and evaluation_root is not None
        and classification.evaluation_root_identity is not None
    ):
        try:
            current = _root_identity_antigravity(evaluation_root)
        except ValueError as error:
            return ReceiptClassificationAntigravity(
                "INVALID", False, True, f"{type(error).__name__}: {error}"
            )
        if current != classification.evaluation_root_identity:
            return ReceiptClassificationAntigravity(
                "INVALID", False, True, "ValueError: evaluation root identity differs"
            )
    return classification


def receipt_reviewed_commit_antigravity(
    receipt_directory: Path = Path(EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY),
) -> str:
    classification = validate_receipt_inventory_antigravity(receipt_directory)
    if classification.invalid or classification.stopped or classification.reviewed_commit is None:
        raise ValueError("receipt reviewed commit is unavailable")
    return classification.reviewed_commit


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_FLAGS_ANTIGRAVITY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _leaf_name_antigravity(phase: str, event: str) -> str:
    return _STOPPED_LEAF_ANTIGRAVITY if event == "stopped" else f"{phase}_{event}_antigravity.json"


def _append_receipt_antigravity(
    paths: _LifecyclePathsAntigravity,
    classification: ReceiptClassificationAntigravity,
    phase: str,
    event: str,
    reviewed_commit: str,
    *,
    evaluation_root_identity: str | None,
) -> ReceiptClassificationAntigravity:
    if classification.invalid or classification.stopped:
        raise ValueError("cannot append to invalid or stopped receipts")
    ordinal = int(classification.state[1:])
    receipt = LifecycleReceiptAntigravity(
        _SCHEMA_ANTIGRAVITY,
        "Exp454",
        _RUN_IDENTITY_ANTIGRAVITY,
        reviewed_commit,
        phase,
        event,
        ordinal,
        classification.previous_leaf_sha256,
        evaluation_root_identity,
        event == "stopped",
        paths.executable,
        paths.prefix,
        paths.python_version,
        paths.numpy_version,
        paths.aeon_version,
        paths.sklearn_version,
    )
    write_create_only_antigravity(
        paths.receipt_directory / _leaf_name_antigravity(phase, event),
        canonical_json_bytes_antigravity(receipt.payload_antigravity()),
    )
    _fsync_directory_antigravity(paths.receipt_directory)
    updated = _classify_receipts_antigravity(
        paths.receipt_directory,
        allow_partial=event == "started",
    )
    if updated.invalid:
        raise ValueError("receipt commit could not be revalidated")
    return updated


def _write_failure_audit_antigravity(
    paths: _LifecyclePathsAntigravity,
    phase: str,
    reviewed_commit: str,
    primary: BaseException,
) -> None:
    payload = {
        "schema": "exp454.lifecycle.failure",
        "phase": phase,
        "reviewed_commit": reviewed_commit,
        "primary_error": f"{type(primary).__name__}: {primary}",
        "evaluation_root": str(paths.evaluation_root),
        "recovery_state": "stopped",
        "permanent_stop": True,
    }
    write_create_only_antigravity(paths.audit_path, canonical_json_bytes_antigravity(payload))
    _fsync_directory_antigravity(paths.audit_path.parent)


def _stop_antigravity(
    paths: _LifecyclePathsAntigravity,
    classification: ReceiptClassificationAntigravity,
    phase: str,
    reviewed_commit: str,
    primary: BaseException,
    *,
    audit_occupied: bool,
) -> None:
    try:
        _append_receipt_antigravity(
            paths,
            classification,
            phase,
            "stopped",
            reviewed_commit,
            evaluation_root_identity=classification.evaluation_root_identity,
        )
    except BaseException as error:
        primary.add_note(f"stopped_receipt_error={type(error).__name__}: {error}")
    if not audit_occupied:
        try:
            _write_failure_audit_antigravity(paths, phase, reviewed_commit, primary)
        except BaseException as error:
            primary.add_note(f"failure_audit_error={type(error).__name__}: {error}")


def _create_receipt_directory_antigravity(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as error:
        raise ValueError("receipt namespace is already occupied") from error
    _fsync_directory_antigravity(path.parent)


def _path_exists_antigravity(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("lifecycle path state cannot be verified") from error


def _run_phase_with_paths_antigravity(
    phase: str,
    reviewed_commit: str | None,
    operation: Callable[..., _T],
    paths: _LifecyclePathsAntigravity,
    *,
    path_precondition: Callable[[], None] | None = None,
) -> _T:
    """Run one private-root phase; failures create a terminal receipt and return nonzero."""
    bare_inspect = phase == "inspect" and reviewed_commit is None
    if (
        phase not in _PREDECESSOR_STATE_ANTIGRAVITY
        or (phase == "inspect" and not bare_inspect)
        or (phase != "inspect" and not _lower_hex_antigravity(reviewed_commit, 40))
        or not callable(operation)
        or type(paths) is not _LifecyclePathsAntigravity
        or (path_precondition is not None and not callable(path_precondition))
    ):
        raise ValueError("lifecycle phase inputs are invalid")

    precondition_error: BaseException | None = None
    if path_precondition is not None:
        try:
            path_precondition()
        except BaseException as error:
            precondition_error = error

    receipt_existed = _path_exists_antigravity(paths.receipt_directory)
    classification = _classify_receipts_antigravity(paths.receipt_directory)
    if classification.invalid:
        raise ValueError("receipt inventory is invalid")
    if classification.stopped:
        raise ValueError("receipt lifecycle is permanently stopped")
    if not receipt_existed:
        _create_receipt_directory_antigravity(paths.receipt_directory)
        classification = _classify_receipts_antigravity(
            paths.receipt_directory,
            just_created=True,
        )
    if classification.invalid:
        raise ValueError("receipt inventory is invalid")

    chain_commit = classification.reviewed_commit or reviewed_commit
    if chain_commit is None or not _lower_hex_antigravity(chain_commit, 40):
        chain_commit = "0" * 40
    audit_occupied = _path_exists_antigravity(paths.audit_path)
    primary = precondition_error
    if primary is None and classification.state != _PREDECESSOR_STATE_ANTIGRAVITY[phase]:
        primary = ValueError("command does not have the exact predecessor")
    if (
        primary is None
        and reviewed_commit is not None
        and classification.reviewed_commit is not None
        and reviewed_commit != classification.reviewed_commit
    ):
        primary = ValueError("reviewed commit does not match receipt chain")
    if primary is None and audit_occupied:
        primary = ValueError("audit destination is occupied")
    try:
        root_before = _root_identity_antigravity(paths.evaluation_root)
    except BaseException as error:
        root_before = None
        if primary is None:
            primary = error
    if primary is None:
        if phase in {"readiness", "evaluate"} and root_before is not None:
            primary = ValueError("evaluation root must be absent")
        elif phase == "inspect" and (
            root_before is None or root_before != classification.evaluation_root_identity
        ):
            primary = ValueError("evaluation root does not match receipt authority")
    if primary is not None:
        _stop_antigravity(
            paths,
            classification,
            phase,
            chain_commit,
            primary,
            audit_occupied=audit_occupied,
        )
        raise primary

    try:
        if phase in {"evaluate", "inspect"}:
            classification = _append_receipt_antigravity(
                paths,
                classification,
                phase,
                "started",
                chain_commit,
                evaluation_root_identity=(root_before if phase == "inspect" else None),
            )
        result = operation(chain_commit) if bare_inspect else operation()
        root_after = _root_identity_antigravity(paths.evaluation_root)
        if phase == "readiness" and root_after is not None:
            raise ValueError("evaluation root appeared during readiness")
        if phase == "evaluate" and root_after is None:
            raise ValueError("evaluation root was not created")
        if phase == "inspect" and root_after != root_before:
            raise ValueError("evaluation root changed during inspect")
        classification = _append_receipt_antigravity(
            paths,
            classification,
            phase,
            "complete",
            chain_commit,
            evaluation_root_identity=(root_after if phase in {"evaluate", "inspect"} else None),
        )
        return result
    except BaseException as error:
        current = _classify_receipts_antigravity(
            paths.receipt_directory,
            allow_partial=True,
        )
        if current.invalid:
            current = classification
        _stop_antigravity(
            paths,
            current,
            phase,
            chain_commit,
            error,
            audit_occupied=audit_occupied,
        )
        raise


def run_phase_antigravity(
    phase: str,
    reviewed_commit: str | None,
    operation: Callable[..., _T],
) -> _T:
    """Public fixed-path lifecycle wrapper with no path override surface."""
    return _run_phase_with_paths_antigravity(
        phase,
        reviewed_commit,
        operation,
        _production_paths_antigravity(),
    )


__all__ = [
    "LifecycleReceiptAntigravity",
    "ReceiptClassificationAntigravity",
    "receipt_reviewed_commit_antigravity",
    "run_phase_antigravity",
    "validate_receipt_inventory_antigravity",
]
