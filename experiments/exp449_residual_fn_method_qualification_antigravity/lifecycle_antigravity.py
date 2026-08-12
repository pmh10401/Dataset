"""Descriptor-safe receipt state machine shared by Exp449 public phase wrappers.

This module deliberately owns no scientific dependency.  Its only durable
state is the receipt prefix and the create-only forensic failure audit.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity


_EXPERIMENT_ID_ANTIGRAVITY = "Exp449"
_SCHEMA_ANTIGRAVITY = "exp449_lifecycle_receipt"
_RUN_IDENTITY_ANTIGRAVITY = "exp449-residual-fn-method-qualification-antigravity-20260812"
_PREDICTION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812"
_EVALUATION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_evaluation_antigravity_20260812"
_RECEIPT_DIR_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_lifecycle_receipts_antigravity_20260812")
_AUDIT_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_failure_audit_antigravity.json")
_SUCCESS_LEAVES_ANTIGRAVITY = (
    "readiness_complete_antigravity.json", "predict_started_antigravity.json", "predict_complete_antigravity.json",
    "evaluate_started_antigravity.json", "evaluate_complete_antigravity.json", "inspect_started_antigravity.json",
    "inspect_complete_antigravity.json",
)
_STOPPED_LEAF_ANTIGRAVITY = "stopped_antigravity.json"
_LEAF_EVENTS_ANTIGRAVITY = (
    ("readiness", "complete"), ("predict", "started"), ("predict", "complete"),
    ("evaluate", "started"), ("evaluate", "complete"), ("inspect", "started"), ("inspect", "complete"),
)
_PREDECESSOR_ANTIGRAVITY = {"readiness": "S0", "predict": "S1", "evaluate": "S3", "inspect": "S5"}
_T = TypeVar("_T")


def _is_lower_hex_antigravity(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


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
    prediction_root_identity: str | None
    evaluation_root_identity: str | None
    permanent_stop: bool

    def __post_init__(self) -> None:
        if (self.schema != _SCHEMA_ANTIGRAVITY or self.experiment_id != _EXPERIMENT_ID_ANTIGRAVITY
                or self.run_identity != _RUN_IDENTITY_ANTIGRAVITY or not _is_lower_hex_antigravity(self.reviewed_commit, 40)
                or self.phase not in _PREDECESSOR_ANTIGRAVITY or self.event not in ("started", "complete", "stopped")
                or type(self.ordinal) is not int or self.ordinal < 0
                or (self.previous_leaf_sha256 is not None and not _is_lower_hex_antigravity(self.previous_leaf_sha256, 64))
                or (self.prediction_root_identity is not None and not _is_lower_hex_antigravity(self.prediction_root_identity, 64))
                or (self.evaluation_root_identity is not None and not _is_lower_hex_antigravity(self.evaluation_root_identity, 64))
                or type(self.permanent_stop) is not bool):
            raise ValueError("lifecycle receipt fields are invalid")

    def payload_antigravity(self) -> dict[str, object]:
        return {"schema": self.schema, "experiment_id": self.experiment_id, "run_identity": self.run_identity,
                "reviewed_commit": self.reviewed_commit, "phase": self.phase, "event": self.event,
                "ordinal": self.ordinal, "previous_leaf_sha256": self.previous_leaf_sha256,
                "prediction_root_identity": self.prediction_root_identity,
                "evaluation_root_identity": self.evaluation_root_identity, "permanent_stop": self.permanent_stop}


@dataclass(frozen=True, slots=True)
class ReceiptClassificationAntigravity:
    state: str
    stopped: bool
    invalid: bool
    reason: str | None
    reviewed_commit: str | None = None
    previous_leaf_sha256: str | None = None

    def __post_init__(self) -> None:
        if (self.state not in {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "STOPPED", "INVALID"}
                or type(self.stopped) is not bool or type(self.invalid) is not bool
                or (self.reason is not None and type(self.reason) is not str)
                or (self.reviewed_commit is not None and not _is_lower_hex_antigravity(self.reviewed_commit, 40))
                or (self.previous_leaf_sha256 is not None and not _is_lower_hex_antigravity(self.previous_leaf_sha256, 64))):
            raise ValueError("receipt classification is invalid")


def _receipt_from_bytes_antigravity(payload: bytes) -> LifecycleReceiptAntigravity:
    value = strict_json_bytes_antigravity(payload)
    fields = {"schema", "experiment_id", "run_identity", "reviewed_commit", "phase", "event", "ordinal", "previous_leaf_sha256", "prediction_root_identity", "evaluation_root_identity", "permanent_stop"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("receipt schema differs")
    return LifecycleReceiptAntigravity(**value)


def _read_leaf_antigravity(directory_fd: int, name: str) -> tuple[bytes, tuple[int, int]]:
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("receipt leaf is unsafe")
        payload = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("receipt leaf was replaced")
        return payload, (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)


def validate_receipt_inventory_antigravity(receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY) -> ReceiptClassificationAntigravity:
    """Classify the complete receipt namespace without treating pathname shape as authority."""
    if not isinstance(receipt_dir, Path):
        raise ValueError("receipt directory must be a Path")
    try:
        info = os.lstat(receipt_dir)
    except FileNotFoundError:
        return ReceiptClassificationAntigravity("S0", False, False, None)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return ReceiptClassificationAntigravity("INVALID", False, True, "receipt directory is unsafe")
    try:
        directory_fd = os.open(receipt_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(directory_fd)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("receipt directory was replaced")
            names = set(os.listdir(directory_fd))
            stopped = _STOPPED_LEAF_ANTIGRAVITY in names
            success = [name for name in _SUCCESS_LEAVES_ANTIGRAVITY if name in names]
            expected_names = set(success) | ({_STOPPED_LEAF_ANTIGRAVITY} if stopped else set())
            if names != expected_names or success != list(_SUCCESS_LEAVES_ANTIGRAVITY[:len(success)]):
                raise ValueError("receipt inventory is not a contiguous prefix")
            if not success and not stopped:
                raise ValueError("receipt directory is an invalid empty namespace")
            previous: str | None = None
            reviewed: str | None = None
            ordinal = 0
            for index, name in enumerate(success):
                payload, _identity = _read_leaf_antigravity(directory_fd, name)
                receipt = _receipt_from_bytes_antigravity(payload)
                if ((receipt.phase, receipt.event) != _LEAF_EVENTS_ANTIGRAVITY[index] or receipt.permanent_stop
                        or receipt.ordinal != ordinal or receipt.previous_leaf_sha256 != previous):
                    raise ValueError("receipt transition differs")
                reviewed = receipt.reviewed_commit if reviewed is None else reviewed
                if receipt.reviewed_commit != reviewed:
                    raise ValueError("receipt reviewed commit differs")
                previous = hashlib.sha256(payload).hexdigest(); ordinal += 1
            if stopped:
                payload, _identity = _read_leaf_antigravity(directory_fd, _STOPPED_LEAF_ANTIGRAVITY)
                receipt = _receipt_from_bytes_antigravity(payload)
                if (receipt.event != "stopped" or not receipt.permanent_stop or receipt.ordinal != ordinal
                        or receipt.previous_leaf_sha256 != previous or (reviewed is not None and receipt.reviewed_commit != reviewed)):
                    raise ValueError("terminal receipt differs")
                current_root = os.stat(receipt_dir, follow_symlinks=False)
                if (current_root.st_dev, current_root.st_ino) != (opened.st_dev, opened.st_ino) or set(os.listdir(directory_fd)) != names:
                    raise ValueError("receipt directory changed during validation")
                return ReceiptClassificationAntigravity("STOPPED", True, False, None, receipt.reviewed_commit, hashlib.sha256(payload).hexdigest())
            current_root = os.stat(receipt_dir, follow_symlinks=False)
            if (current_root.st_dev, current_root.st_ino) != (opened.st_dev, opened.st_ino) or set(os.listdir(directory_fd)) != names:
                raise ValueError("receipt directory changed during validation")
            return ReceiptClassificationAntigravity(f"S{len(success)}", False, False, None, reviewed, previous)
        finally:
            os.close(directory_fd)
    except BaseException as error:
        return ReceiptClassificationAntigravity("INVALID", False, True, f"{type(error).__name__}: {error}")


def receipt_reviewed_commit_antigravity(receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY) -> str:
    """Return the descriptor-validated run commit for the bare inspect command."""
    classification = validate_receipt_inventory_antigravity(receipt_dir)
    if classification.invalid or classification.state == "S0" or classification.stopped:
        raise ValueError("receipt run identity is unavailable")
    if classification.reviewed_commit is None:
        raise ValueError("receipt run identity is unavailable")
    return classification.reviewed_commit


def _fsync_parent_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _root_identity_antigravity(root: Path) -> str | None:
    """Return null for an absent root or a descriptor-derived stable content identity."""
    try:
        path_info = os.lstat(root)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode):
        raise ValueError("root identity path is unsafe")
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (path_info.st_dev, path_info.st_ino):
            raise ValueError("root identity path was replaced")
        digest = hashlib.sha256()
        def add_directory(directory: int, prefix: str) -> None:
            for name in sorted(os.listdir(directory)):
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                relative = prefix + name
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
                    try:
                        child_info = os.fstat(child)
                        if (child_info.st_dev, child_info.st_ino) != (info.st_dev, info.st_ino):
                            raise ValueError("root identity directory was replaced")
                        add_directory(child, relative + "/")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    leaf = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
                    try:
                        leaf_info = os.fstat(leaf)
                        if (leaf_info.st_dev, leaf_info.st_ino) != (info.st_dev, info.st_ino):
                            raise ValueError("root identity leaf was replaced")
                        digest.update(relative.encode("utf-8") + b"\0")
                        digest.update(b"".join(iter(lambda: os.read(leaf, 1 << 20), b"")))
                    finally:
                        os.close(leaf)
                else:
                    raise ValueError("root identity inventory is unsafe")
        add_directory(descriptor, "")
        if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("root identity directory was replaced")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _fault_antigravity(fault_hook, point: str) -> None:
    if fault_hook is not None:
        fault_hook(point)


def _append_leaf_at_descriptor_antigravity(directory: Path, directory_fd: int, name: str, payload: bytes, fault_hook) -> None:
    """Create, fsync, reread, and namespace-revalidate a receipt leaf beneath one retained fd."""
    _fault_antigravity(fault_hook, "leaf_create")
    leaf = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
    created: tuple[int, int] | None = None
    primary: BaseException | None = None
    try:
        info = os.fstat(leaf); created = (info.st_dev, info.st_ino)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("receipt leaf is unsafe")
        view = memoryview(payload)
        while view:
            _fault_antigravity(fault_hook, "leaf_write")
            written = os.write(leaf, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        _fault_antigravity(fault_hook, "leaf_fsync"); os.fsync(leaf)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _fault_antigravity(fault_hook, "leaf_close")
            os.close(leaf)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"leaf_close_error={type(error).__name__}: {error}")
            else:
                raise
    _fault_antigravity(fault_hook, "parent_fsync"); os.fsync(directory_fd)
    _fault_antigravity(fault_hook, "leaf_reread")
    reader = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    primary = None
    try:
        opened = os.fstat(reader)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if created is None or (opened.st_dev, opened.st_ino) != created or (current.st_dev, current.st_ino) != created or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("receipt leaf was rebound")
        reread = b"".join(iter(lambda: os.read(reader, 1 << 20), b""))
        if reread != payload or canonical_json_bytes_antigravity(strict_json_bytes_antigravity(reread)) != reread:
            raise ValueError("receipt leaf reread differs")
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _fault_antigravity(fault_hook, "reread_close")
            os.close(reader)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"reread_close_error={type(error).__name__}: {error}")
            else:
                raise
    current_directory = os.stat(directory, follow_symlinks=False)
    retained = os.fstat(directory_fd)
    if (current_directory.st_dev, current_directory.st_ino) != (retained.st_dev, retained.st_ino):
        raise ValueError("receipt directory was rebound")


def append_receipt_antigravity(receipt_dir: Path, phase: str, event: str, reviewed_commit: str, *, prediction_root: Path = Path(_PREDICTION_ROOT_ANTIGRAVITY), evaluation_root: Path = Path(_EVALUATION_ROOT_ANTIGRAVITY), fault_hook=None) -> LifecycleReceiptAntigravity:
    """Append the one next governed receipt leaf and revalidate the complete prefix."""
    if not isinstance(receipt_dir, Path) or not isinstance(prediction_root, Path) or not isinstance(evaluation_root, Path) or phase not in _PREDECESSOR_ANTIGRAVITY or event not in ("started", "complete", "stopped") or not _is_lower_hex_antigravity(reviewed_commit, 40):
        raise ValueError("receipt append inputs are invalid")
    classification = validate_receipt_inventory_antigravity(receipt_dir)
    if classification.invalid or classification.stopped:
        raise ValueError("receipt namespace cannot be appended")
    state_number = int(classification.state[1:])
    if event == "stopped":
        name = _STOPPED_LEAF_ANTIGRAVITY
    else:
        if state_number >= len(_SUCCESS_LEAVES_ANTIGRAVITY) or _LEAF_EVENTS_ANTIGRAVITY[state_number] != (phase, event):
            raise ValueError("receipt transition is not the next governed leaf")
        name = _SUCCESS_LEAVES_ANTIGRAVITY[state_number]
    previous = classification.previous_leaf_sha256
    if state_number:
        if classification.reviewed_commit != reviewed_commit or previous is None:
            raise ValueError("receipt reviewed commit differs")
    receipt = LifecycleReceiptAntigravity(_SCHEMA_ANTIGRAVITY, _EXPERIMENT_ID_ANTIGRAVITY, _RUN_IDENTITY_ANTIGRAVITY, reviewed_commit, phase, event, state_number, previous, _root_identity_antigravity(prediction_root), _root_identity_antigravity(evaluation_root), event == "stopped")
    # S0 is the only state allowed to create this directory.  Re-open it
    # no-follow and retain that descriptor for the complete leaf transaction.
    if classification.state == "S0":
        _fault_antigravity(fault_hook, "directory_create")
        receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_fd = os.open(receipt_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    primary: BaseException | None = None
    try:
        _append_leaf_at_descriptor_antigravity(receipt_dir, directory_fd, name, canonical_json_bytes_antigravity(receipt.payload_antigravity()), fault_hook)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _fault_antigravity(fault_hook, "directory_close")
            os.close(directory_fd)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"directory_close_error={type(error).__name__}: {error}")
            else:
                raise
    result = validate_receipt_inventory_antigravity(receipt_dir)
    expected_state = "STOPPED" if event == "stopped" else f"S{state_number + 1}"
    if result.state != expected_state:
        raise ValueError("receipt append failed revalidation")
    return receipt


def write_failure_audit_antigravity(destination: Path, phase: str, root_destinations: tuple[str, ...], primary_error: BaseException, *, recovery_state: str = "stopped", cleanup_notes: tuple[str, ...] = ()) -> None:
    """Create exactly one deterministic audit; occupied paths are deliberately untouched."""
    if (not isinstance(destination, Path) or phase not in _PREDECESSOR_ANTIGRAVITY or type(root_destinations) is not tuple
            or any(type(value) is not str for value in root_destinations) or not isinstance(primary_error, BaseException)
            or type(recovery_state) is not str or type(cleanup_notes) is not tuple or any(type(note) is not str for note in cleanup_notes)):
        raise ValueError("failure audit inputs are invalid")
    payload = {"phase": phase, "root_destinations": list(root_destinations), "recovery_state": recovery_state,
               "primary_error": f"{type(primary_error).__name__}: {primary_error}", "cleanup_notes": list(cleanup_notes),
               "permanent_stop": True}
    write_create_only_antigravity(destination, canonical_json_bytes_antigravity(payload))
    _fsync_parent_antigravity(destination.parent)


def _root_destinations_antigravity(phase: str) -> tuple[str, ...]:
    if phase == "readiness":
        return ()
    if phase == "predict":
        return (_PREDICTION_ROOT_ANTIGRAVITY,)
    return (_PREDICTION_ROOT_ANTIGRAVITY, _EVALUATION_ROOT_ANTIGRAVITY)


def _stop_and_audit_antigravity(receipt_dir: Path, audit_path: Path, phase: str, reviewed_commit: str, primary: BaseException, prediction_root: Path, evaluation_root: Path) -> None:
    try:
        append_receipt_antigravity(receipt_dir, phase, "stopped", reviewed_commit, prediction_root=prediction_root, evaluation_root=evaluation_root)
    except BaseException as secondary:
        primary.add_note(f"stopped_receipt_error={type(secondary).__name__}: {secondary}")
    try:
        write_failure_audit_antigravity(audit_path, phase, _root_destinations_antigravity(phase), primary)
    except BaseException as secondary:
        primary.add_note(f"failure_audit_error={type(secondary).__name__}: {secondary}")


def run_phase_antigravity(phase: str, reviewed_commit: str, operation: Callable[[], _T], *, receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY, audit_path: Path = _AUDIT_PATH_ANTIGRAVITY, prediction_root: Path = Path(_PREDICTION_ROOT_ANTIGRAVITY), evaluation_root: Path = Path(_EVALUATION_ROOT_ANTIGRAVITY), path_precondition: Callable[[], None] | None = None) -> _T:
    """Run one reusable lifecycle phase; synthetic tests inject only operation and paths."""
    if phase not in _PREDECESSOR_ANTIGRAVITY or not _is_lower_hex_antigravity(reviewed_commit, 40) or not callable(operation) or not isinstance(receipt_dir, Path) or not isinstance(audit_path, Path) or not isinstance(prediction_root, Path) or not isinstance(evaluation_root, Path) or (path_precondition is not None and not callable(path_precondition)):
        raise ValueError("lifecycle phase inputs are invalid")
    classification = validate_receipt_inventory_antigravity(receipt_dir)
    if classification.invalid:
        raise ValueError("receipt inventory is invalid")
    if classification.stopped:
        raise ValueError("receipt lifecycle is permanently stopped")
    audit_occupied = audit_path.exists() or audit_path.is_symlink()
    primary: BaseException | None = None
    chain_commit = classification.reviewed_commit or reviewed_commit
    if classification.state != _PREDECESSOR_ANTIGRAVITY[phase]:
        primary = ValueError("command does not have the exact predecessor")
    elif classification.reviewed_commit is not None and classification.reviewed_commit != reviewed_commit:
        primary = ValueError("reviewed commit does not match receipt chain")
    elif audit_occupied:
        primary = ValueError("failure audit destination is occupied")
    else:
        try:
            if path_precondition is not None:
                path_precondition()
        except BaseException as error:
            primary = error
    if primary is not None:
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, primary, prediction_root, evaluation_root)
        raise primary
    try:
        if phase != "readiness":
            append_receipt_antigravity(receipt_dir, phase, "started", chain_commit, prediction_root=prediction_root, evaluation_root=evaluation_root)
        result = operation()
    except BaseException as error:
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, error, prediction_root, evaluation_root)
        raise
    append_receipt_antigravity(receipt_dir, phase, "complete", chain_commit, prediction_root=prediction_root, evaluation_root=evaluation_root)
    return result


__all__ = ["LifecycleReceiptAntigravity", "ReceiptClassificationAntigravity", "append_receipt_antigravity", "receipt_reviewed_commit_antigravity", "run_phase_antigravity", "validate_receipt_inventory_antigravity", "write_failure_audit_antigravity"]
