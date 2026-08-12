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
_NO_CHAIN_INSPECT_COMMIT_ANTIGRAVITY = "0" * 40
_LEAF_EVENTS_ANTIGRAVITY = (
    ("readiness", "complete"), ("predict", "started"), ("predict", "complete"),
    ("evaluate", "started"), ("evaluate", "complete"), ("inspect", "started"), ("inspect", "complete"),
)
_PREDECESSOR_ANTIGRAVITY = {"readiness": "S0", "predict": "S1", "evaluate": "S3", "inspect": "S5"}
_STAGE_ROOT_EXPECTATION_ANTIGRAVITY = {
    ("readiness", "complete"): (False, False), ("predict", "started"): (False, False),
    ("predict", "complete"): (True, False), ("evaluate", "started"): (True, False),
    ("evaluate", "complete"): (True, True), ("inspect", "started"): (True, True),
    ("inspect", "complete"): (True, True),
}
_T = TypeVar("_T")


def _is_lower_hex_antigravity(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _stage_root_fields_antigravity(phase: str, event: str, prediction: str | None, evaluation: str | None) -> tuple[str | None, str | None]:
    """Freeze which root identities a successful stage is permitted to bind."""
    if event == "stopped": return prediction, evaluation
    require_prediction, require_evaluation = _STAGE_ROOT_EXPECTATION_ANTIGRAVITY[(phase, event)]
    if require_prediction and prediction is None: raise ValueError("stage requires prediction root identity")
    if require_evaluation and evaluation is None: raise ValueError("stage requires evaluation root identity")
    return (prediction if require_prediction else None, evaluation if require_evaluation else None)


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


class _LifecycleTransactionAntigravity:
    """One command-wide pinned receipt namespace; it is never pathname-reopened."""
    __slots__ = ("path", "parent_fd", "directory_fd", "directory_identity", "leaves", "owned_leaf_fds", "classification")

    def __init__(self, path: Path, *, create_s0: bool) -> None:
        self.path = path; self.parent_fd = self.directory_fd = -1; self.directory_identity = None
        self.leaves: list[tuple[str, int, tuple[int, int], bytes, LifecycleReceiptAntigravity]] = []
        self.owned_leaf_fds: list[int] = []
        self.classification: ReceiptClassificationAntigravity | None = None
        parent = path.parent
        self.parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            try:
                info = os.stat(path.name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not create_s0:
                    self.classification = ReceiptClassificationAntigravity("S0", False, False, None)
                    return
                os.mkdir(path.name, 0o700, dir_fd=self.parent_fd)
                os.fsync(self.parent_fd)
                info = os.stat(path.name, dir_fd=self.parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError("receipt directory is unsafe")
            self.directory_fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.parent_fd)
            opened = os.fstat(self.directory_fd)
            if _namespace_identity_antigravity(opened) != _namespace_identity_antigravity(info):
                raise ValueError("receipt directory was replaced")
            self.directory_identity = _namespace_identity_antigravity(opened)
            self.classification = self._classify()
        except BaseException:
            self.close(None); raise

    def _classify(self) -> ReceiptClassificationAntigravity:
        if self.directory_fd < 0:
            return ReceiptClassificationAntigravity("S0", False, False, None)
        names = set(os.listdir(self.directory_fd)); stopped = _STOPPED_LEAF_ANTIGRAVITY in names
        success = [name for name in _SUCCESS_LEAVES_ANTIGRAVITY if name in names]
        if not names:
            return ReceiptClassificationAntigravity("S0", False, False, None)
        if names != set(success) | ({_STOPPED_LEAF_ANTIGRAVITY} if stopped else set()) or success != list(_SUCCESS_LEAVES_ANTIGRAVITY[:len(success)]):
            raise ValueError("receipt inventory is not a contiguous prefix")
        previous = None; reviewed = None
        for ordinal, name in enumerate(success):
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
            # Own the descriptor before any validation can fail.
            self.owned_leaf_fds.append(fd)
            info = os.fstat(fd); identity = _namespace_identity_antigravity(info)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1: raise ValueError("receipt leaf is unsafe")
            payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b"")); receipt = _receipt_from_bytes_antigravity(payload)
            if (receipt.phase, receipt.event) != _LEAF_EVENTS_ANTIGRAVITY[ordinal] or receipt.permanent_stop or receipt.ordinal != ordinal or receipt.previous_leaf_sha256 != previous:
                raise ValueError("receipt transition differs")
            require_prediction, require_evaluation = _STAGE_ROOT_EXPECTATION_ANTIGRAVITY[(receipt.phase, receipt.event)]
            if ((not require_prediction and receipt.prediction_root_identity is not None)
                    or (not require_evaluation and receipt.evaluation_root_identity is not None)
                    or (require_prediction and receipt.prediction_root_identity is None)
                    or (require_evaluation and receipt.evaluation_root_identity is None)):
                raise ValueError("receipt stage root identities differ")
            reviewed = receipt.reviewed_commit if reviewed is None else reviewed
            if receipt.reviewed_commit != reviewed: raise ValueError("receipt reviewed commit differs")
            self.owned_leaf_fds.remove(fd); self.leaves.append((name, fd, identity, payload, receipt)); previous = hashlib.sha256(payload).hexdigest()
        if stopped:
            fd = os.open(_STOPPED_LEAF_ANTIGRAVITY, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
            self.owned_leaf_fds.append(fd)
            info = os.fstat(fd); identity = _namespace_identity_antigravity(info); payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b"")); receipt = _receipt_from_bytes_antigravity(payload)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or receipt.event != "stopped" or not receipt.permanent_stop or receipt.ordinal != len(success) or receipt.previous_leaf_sha256 != previous or (reviewed is not None and receipt.reviewed_commit != reviewed):
                raise ValueError("terminal receipt differs")
            self.owned_leaf_fds.remove(fd); self.leaves.append((_STOPPED_LEAF_ANTIGRAVITY, fd, identity, payload, receipt)); return ReceiptClassificationAntigravity("STOPPED", True, False, None, receipt.reviewed_commit, hashlib.sha256(payload).hexdigest())
        return ReceiptClassificationAntigravity(f"S{len(success)}", False, False, None, reviewed, previous)

    def end_validate(self) -> None:
        if self.directory_fd < 0: return
        retained = os.fstat(self.directory_fd)
        current = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if _namespace_identity_antigravity(retained) != self.directory_identity or _namespace_identity_antigravity(current) != self.directory_identity:
            raise ValueError("receipt directory was rebound")
        if set(os.listdir(self.directory_fd)) != {item[0] for item in self.leaves}:
            raise ValueError("receipt inventory changed during transaction")
        for name, fd, identity, payload, _receipt in self.leaves:
            info = os.fstat(fd); current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            if _namespace_identity_antigravity(info) != identity or _namespace_identity_antigravity(current) != identity:
                raise ValueError("receipt leaf was replaced")
            os.lseek(fd, 0, os.SEEK_SET)
            if b"".join(iter(lambda: os.read(fd, 1 << 20), b"")) != payload:
                raise ValueError("receipt leaf changed")

    def close(self, primary: BaseException | None) -> None:
        first: BaseException | None = None
        for _name, fd, _identity, _payload, _receipt in reversed(self.leaves):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"receipt_leaf_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        self.leaves.clear()
        for fd in reversed(self.owned_leaf_fds):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"receipt_unvalidated_leaf_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        self.owned_leaf_fds.clear()
        for descriptor in (self.directory_fd, self.parent_fd):
            if descriptor >= 0:
                try: os.close(descriptor)
                except BaseException as error:
                    if primary is not None: primary.add_note(f"receipt_directory_close_error={type(error).__name__}: {error}")
                    elif first is None: first = error
        self.directory_fd = self.parent_fd = -1
        if primary is None and first is not None:
            raise first


# Compatibility alias for existing synthetic tests; command code uses the explicit
# lifecycle name so the ownership boundary is visible at each public phase.
_ReceiptTransactionAntigravity = _LifecycleTransactionAntigravity


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


def _validate_bound_root_identities_antigravity(
    transaction: _LifecycleTransactionAntigravity, prediction_root: Path, evaluation_root: Path,
) -> None:
    """Bind every recorded non-null root digest to current descriptor-captured bytes."""
    prediction = _RootIdentityTransactionAntigravity(prediction_root)
    evaluation = _RootIdentityTransactionAntigravity(evaluation_root)
    primary: BaseException | None = None
    try:
        for _name, _fd, _identity, _payload, receipt in transaction.leaves:
            if receipt.prediction_root_identity is not None and receipt.prediction_root_identity != prediction.identity:
                raise ValueError("prediction root identity differs from receipt chain")
            if receipt.evaluation_root_identity is not None and receipt.evaluation_root_identity != evaluation.identity:
                raise ValueError("evaluation root identity differs from receipt chain")
        prediction.end_validate(); evaluation.end_validate()
    except BaseException as error:
        primary = error
        raise
    finally:
        prediction.close(primary); evaluation.close(primary)


def validate_receipt_inventory_antigravity(receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY, *, prediction_root: Path | None = None, evaluation_root: Path | None = None) -> ReceiptClassificationAntigravity:
    """Classify the complete receipt namespace without treating pathname shape as authority."""
    if not isinstance(receipt_dir, Path):
        raise ValueError("receipt directory must be a Path")
    try:
        transaction = _LifecycleTransactionAntigravity(receipt_dir, create_s0=False)
        result = transaction.classification
        transaction.end_validate()
        if (prediction_root is None) != (evaluation_root is None):
            raise ValueError("root identity validation requires both roots")
        if prediction_root is not None and evaluation_root is not None:
            _validate_bound_root_identities_antigravity(transaction, prediction_root, evaluation_root)
        transaction.close(None)
        if result is None: raise ValueError("receipt transaction did not classify")
        return result
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


def _namespace_identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int]:
    """The descriptor-bound identity used for every retained directory and leaf."""
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink)


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
                    digest.update(b"D\0" + relative.encode("utf-8") + b"\0")
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
                        digest.update(b"F\0" + relative.encode("utf-8") + b"\0")
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


class _RootIdentityTransactionAntigravity:
    """Retain every descriptor used for a root identity until a receipt commits."""
    __slots__ = ("root", "root_name", "parent_fd", "root_fd", "descriptors", "records", "identity", "absence")

    def __init__(self, root: Path, fault_hook=None) -> None:
        self.root = root; self.root_name = root.name; self.parent_fd = self.root_fd = -1; self.descriptors: list[tuple[int, tuple[int, int, int, int], tuple[str, ...]]] = []; self.records: list[tuple[int, tuple[int, int, int, int], int, str, bytes]] = []; self.identity = None; self.absence = None
        self.parent_fd = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            # Everything after parent open is owned by this constructor; any
            # failure below closes every partial root/leaf/parent descriptor.
            try:
                path_info = os.stat(self.root_name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                os.close(self.parent_fd); self.parent_fd = -1
                self.absence = _AbsentPathPinAntigravity(root)
                return
            if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode):
                raise ValueError("root identity path is unsafe")
            self.root_fd = os.open(self.root_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.parent_fd)
            root_info = os.fstat(self.root_fd)
            if _namespace_identity_antigravity(root_info) != _namespace_identity_antigravity(path_info):
                raise ValueError("root identity path was replaced")
            digest = hashlib.sha256()
            def walk(fd: int, prefix: tuple[str, ...]) -> None:
                info = os.fstat(fd); self.descriptors.append((fd, _namespace_identity_antigravity(info), prefix))
                for name in sorted(os.listdir(fd)):
                    item = os.stat(name, dir_fd=fd, follow_symlinks=False); relative = "/".join(prefix + (name,))
                    if stat.S_ISDIR(item.st_mode):
                        digest.update(b"D\0" + relative.encode() + b"\0")
                        child = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
                        if _namespace_identity_antigravity(os.fstat(child)) != _namespace_identity_antigravity(item): raise ValueError("root directory was replaced")
                        walk(child, prefix + (name,)); _fault_antigravity(fault_hook, "after_child_capture")
                    elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                        child = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd); child_info = os.fstat(child)
                        if _namespace_identity_antigravity(child_info) != _namespace_identity_antigravity(item): raise ValueError("root leaf was replaced")
                        payload = b"".join(iter(lambda: os.read(child, 1 << 20), b"")); digest.update(b"F\0" + relative.encode() + b"\0" + payload)
                        self.records.append((child, _namespace_identity_antigravity(child_info), fd, name, payload)); _fault_antigravity(fault_hook, "after_leaf_read")
                    else: raise ValueError("root identity inventory is unsafe")
            walk(self.root_fd, ())
        except BaseException as error:
            self.close(error)
            raise
        self.identity = digest.hexdigest()

    def end_validate(self) -> None:
        if self.absence is not None:
            self.absence.end_validate()
            return
        if self.root_fd < 0: return
        path_info = os.stat(self.root_name, dir_fd=self.parent_fd, follow_symlinks=False); info = os.fstat(self.root_fd)
        if _namespace_identity_antigravity(path_info) != _namespace_identity_antigravity(info): raise ValueError("root namespace was replaced")
        for fd, identity, prefix in self.descriptors:
            current = os.fstat(fd)
            if _namespace_identity_antigravity(current) != identity: raise ValueError("root descriptor changed")
            if prefix:
                parent_fd = next(parent for parent, _parent_identity, parent_prefix in self.descriptors if parent_prefix == prefix[:-1])
                namespace = os.stat(prefix[-1], dir_fd=parent_fd, follow_symlinks=False)
                if _namespace_identity_antigravity(namespace) != identity: raise ValueError("root directory was replaced")
            expected = {path[-1] for _fd, _id, path in self.descriptors if path and path[:-1] == prefix} | {name for _fd, _id, parent_fd, name, _payload in self.records if parent_fd == fd}
            if set(os.listdir(fd)) != expected: raise ValueError("root inventory changed")
        for fd, identity, parent_fd, name, payload in self.records:
            info = os.fstat(fd)
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            os.lseek(fd, 0, os.SEEK_SET)
            if (_namespace_identity_antigravity(info) != identity or _namespace_identity_antigravity(current) != identity
                    or b"".join(iter(lambda: os.read(fd, 1 << 20), b"")) != payload):
                raise ValueError("root leaf was replaced")

    def close(self, primary: BaseException | None) -> None:
        first: BaseException | None = None
        for fd, *_ in reversed(self.records):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_leaf_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        for fd, *_ in reversed(self.descriptors):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_directory_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        self.records.clear(); self.descriptors.clear()
        if self.absence is not None:
            try: self.absence.close(primary)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_absence_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
            self.absence = None
        if self.parent_fd >= 0:
            try: os.close(self.parent_fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_parent_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        self.root_fd = self.parent_fd = -1
        if primary is None and first is not None:
            raise first


class _AbsentPathPinAntigravity:
    """Pin an exact absent child name plus its parent identity and inventory."""
    __slots__ = ("path", "name", "parent_fd", "parent_identity", "parent_inventory", "permitted_names")

    def __init__(self, path: Path, *, permitted_names: tuple[str, ...] = ()) -> None:
        self.path = path; self.name = path.name; self.parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        parent = os.fstat(self.parent_fd)
        self.parent_identity = (parent.st_dev, parent.st_ino, parent.st_mode)
        self.parent_inventory = frozenset(os.listdir(self.parent_fd))
        self.permitted_names = frozenset(permitted_names)
        try:
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        error = ValueError("absent root path destination is occupied")
        self.close(error)
        raise error

    def end_validate(self) -> None:
        if self.parent_fd < 0:
            return
        self._validate_parent_identity()
        try:
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            self._validate_parent_inventory(False)
            return
        raise ValueError("absent root path appeared during command")

    def _validate_parent_identity(self) -> None:
        if self.parent_fd < 0:
            return
        parent = os.fstat(self.parent_fd)
        if (parent.st_dev, parent.st_ino, parent.st_mode) != self.parent_identity:
            raise ValueError("absent root path parent was replaced")

    def _validate_parent_inventory(self, permit_name: bool) -> None:
        if self.parent_fd < 0:
            return
        current = frozenset(os.listdir(self.parent_fd))
        allowed = self.parent_inventory | self.permitted_names | ({self.name} if permit_name else set())
        if not self.parent_inventory <= current <= allowed:
            raise ValueError("absent root path parent inventory changed")

    def validate_parent(self, *, permit_name: bool = False) -> None:
        self._validate_parent_identity(); self._validate_parent_inventory(permit_name)

    def validate_produced(self) -> None:
        self.validate_parent(permit_name=True)
        try:
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("absent root path production is missing") from error

    def close(self, primary: BaseException | None) -> None:
        if self.parent_fd < 0:
            return
        descriptor, self.parent_fd = self.parent_fd, -1
        try:
            os.close(descriptor)
        except BaseException as error:
            if primary is not None:
                primary.add_note(f"absent_root_parent_close_error={type(error).__name__}: {error}")
            else:
                raise


# Compatibility alias for the Round-3 command authority naming.
_AbsentRootPinAntigravity = _AbsentPathPinAntigravity


class _CommandRootAuthorityAntigravity:
    """Command-wide root authority retained from precondition through final receipt commit."""
    __slots__ = ("phase", "prediction_root", "evaluation_root", "prediction", "evaluation", "prediction_absent", "evaluation_absent", "identities")

    def __init__(self, phase: str, prediction_root: Path, evaluation_root: Path) -> None:
        self.phase, self.prediction_root, self.evaluation_root = phase, prediction_root, evaluation_root
        self.prediction = self.evaluation = self.prediction_absent = self.evaluation_absent = None
        require_prediction, require_evaluation = {"readiness": (False, False), "predict": (False, False), "evaluate": (True, False), "inspect": (True, True)}[phase]
        try:
            self.prediction = _RootIdentityTransactionAntigravity(prediction_root) if require_prediction else None
            self.evaluation = _RootIdentityTransactionAntigravity(evaluation_root) if require_evaluation else None
            self.prediction_absent = None if require_prediction else _AbsentPathPinAntigravity(prediction_root)
            self.evaluation_absent = None if require_evaluation else _AbsentPathPinAntigravity(
                evaluation_root, permitted_names=(prediction_root.name,) if phase == "predict" else (),
            )
        except BaseException as error:
            self.close(error)
            raise
        self.identities = (self.prediction.identity if self.prediction is not None else None, self.evaluation.identity if self.evaluation is not None else None)

    def verify_receipts(self, transaction: _LifecycleTransactionAntigravity) -> None:
        for _name, _fd, _identity, _payload, receipt in transaction.leaves:
            if receipt.prediction_root_identity is not None and receipt.prediction_root_identity != self.identities[0]:
                raise ValueError("prediction root identity differs from receipt chain")
            if receipt.evaluation_root_identity is not None and receipt.evaluation_root_identity != self.identities[1]:
                raise ValueError("evaluation root identity differs from receipt chain")

    def capture_produced(self) -> None:
        if self.phase == "predict":
            self.prediction_absent.validate_produced(); self.prediction_absent.close(None); self.prediction_absent = None
            self.prediction = _RootIdentityTransactionAntigravity(self.prediction_root)
        elif self.phase == "evaluate":
            self.evaluation_absent.validate_produced(); self.evaluation_absent.close(None); self.evaluation_absent = None
            self.evaluation = _RootIdentityTransactionAntigravity(self.evaluation_root)
        self.identities = (self.prediction.identity if self.prediction is not None else None, self.evaluation.identity if self.evaluation is not None else None)

    def end_validate(self) -> None:
        if self.prediction is not None: self.prediction.end_validate()
        if self.evaluation is not None: self.evaluation.end_validate()
        if self.prediction_absent is not None: self.prediction_absent.end_validate()
        if self.evaluation_absent is not None: self.evaluation_absent.end_validate()

    def close(self, primary: BaseException | None) -> None:
        first: BaseException | None = None
        for item in (self.evaluation, self.prediction, self.evaluation_absent, self.prediction_absent):
            if item is None:
                continue
            try: item.close(primary)
            except BaseException as error:
                if primary is not None: primary.add_note(f"command_root_close_error={type(error).__name__}: {error}")
                elif first is None: first = error
        if primary is None and first is not None:
            raise first


def _fault_antigravity(fault_hook, point: str) -> None:
    if fault_hook is not None:
        fault_hook(point)


def _append_leaf_at_descriptor_antigravity(directory: Path, directory_fd: int, name: str, payload: bytes, fault_hook, created_holder: list[tuple[int, int, int, int]] | None = None) -> tuple[int, int, int, int]:
    """Create and authenticate a leaf; caller owns all final durability barriers."""
    _fault_antigravity(fault_hook, "leaf_create")
    leaf = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
    created: tuple[int, int] | None = None
    primary: BaseException | None = None
    try:
        info = os.fstat(leaf); created = _namespace_identity_antigravity(info)
        if created_holder is not None: created_holder.append(created)
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
        close_error = None
        try: _fault_antigravity(fault_hook, "leaf_close")
        except BaseException as error: close_error = error
        try: os.close(leaf)
        except BaseException as error:
            if close_error is None: close_error = error
        if close_error is not None:
            if primary is not None: primary.add_note(f"leaf_close_error={type(close_error).__name__}: {close_error}")
            else: raise close_error
    _fault_antigravity(fault_hook, "leaf_reread")
    reader = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    primary = None
    try:
        opened = os.fstat(reader)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if created is None or _namespace_identity_antigravity(opened) != created or _namespace_identity_antigravity(current) != created or not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("receipt leaf was rebound")
        reread = b"".join(iter(lambda: os.read(reader, 1 << 20), b""))
        if reread != payload or canonical_json_bytes_antigravity(strict_json_bytes_antigravity(reread)) != reread:
            raise ValueError("receipt leaf reread differs")
    except BaseException as error:
        primary = error
        raise
    finally:
        close_error = None
        try: _fault_antigravity(fault_hook, "reread_close")
        except BaseException as error: close_error = error
        try: os.close(reader)
        except BaseException as error:
            if close_error is None: close_error = error
        if close_error is not None:
            if primary is not None: primary.add_note(f"reread_close_error={type(close_error).__name__}: {close_error}")
            else: raise close_error
    if created is None:
        raise ValueError("receipt leaf creation identity is unavailable")
    return created


def _remove_created_leaf_antigravity(directory_fd: int, name: str, created: tuple[int, int, int, int]) -> None:
    """Remove only the exact precommit leaf inode, then durably publish its removal."""
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if _namespace_identity_antigravity(current) != created:
        raise ValueError("created receipt leaf cleanup is unsafe")
    os.unlink(name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def append_receipt_antigravity(receipt_dir: Path, phase: str, event: str, reviewed_commit: str, *, prediction_root: Path = Path(_PREDICTION_ROOT_ANTIGRAVITY), evaluation_root: Path = Path(_EVALUATION_ROOT_ANTIGRAVITY), root_identities: tuple[str | None, str | None] | None = None, fault_hook=None, _transaction=None, _precommit=None) -> LifecycleReceiptAntigravity:
    """Append one receipt, with parent fsync as its only final durable operation.

    ``_precommit`` is command-private authority validation/close work.  It runs
    after the new leaf is reread but before receipt directory and parent fsync.
    """
    if (not isinstance(receipt_dir, Path) or not isinstance(prediction_root, Path) or not isinstance(evaluation_root, Path)
            or phase not in _PREDECESSOR_ANTIGRAVITY or event not in ("started", "complete", "stopped")
            or not _is_lower_hex_antigravity(reviewed_commit, 40) or (_precommit is not None and not callable(_precommit))):
        raise ValueError("receipt append inputs are invalid")
    own_transaction = _transaction is None
    transaction = None; prediction_transaction = evaluation_transaction = None
    primary: BaseException | None = None; created = None; appended_leaf = None; created_holder: list[tuple[int, int, int, int]] = []
    try:
        if not receipt_dir.exists() and not receipt_dir.is_symlink() and own_transaction:
            _fault_antigravity(fault_hook, "directory_create")
        transaction = _ReceiptTransactionAntigravity(receipt_dir, create_s0=True) if own_transaction else _transaction
        classification = transaction.classification
        if classification is None or classification.stopped:
            raise ValueError("receipt namespace cannot be appended")
        state_number = int(classification.state[1:])
        if event == "stopped":
            name = _STOPPED_LEAF_ANTIGRAVITY
        else:
            if state_number >= len(_SUCCESS_LEAVES_ANTIGRAVITY) or _LEAF_EVENTS_ANTIGRAVITY[state_number] != (phase, event):
                raise ValueError("receipt transition is not the next governed leaf")
            name = _SUCCESS_LEAVES_ANTIGRAVITY[state_number]
        previous = classification.previous_leaf_sha256
        if state_number and (classification.reviewed_commit != reviewed_commit or previous is None):
            raise ValueError("receipt reviewed commit differs")
        if root_identities is None:
            prediction_transaction = _RootIdentityTransactionAntigravity(prediction_root, fault_hook)
            evaluation_transaction = _RootIdentityTransactionAntigravity(evaluation_root, fault_hook)
            root_identities = (prediction_transaction.identity, evaluation_transaction.identity)
        if type(root_identities) is not tuple or len(root_identities) != 2:
            raise ValueError("receipt root identities are invalid")
        prediction_identity, evaluation_identity = _stage_root_fields_antigravity(phase, event, root_identities[0], root_identities[1])
        receipt = LifecycleReceiptAntigravity(_SCHEMA_ANTIGRAVITY, _EXPERIMENT_ID_ANTIGRAVITY, _RUN_IDENTITY_ANTIGRAVITY, reviewed_commit, phase, event, state_number, previous, prediction_identity, evaluation_identity, event == "stopped")
        _fault_antigravity(fault_hook, "before_end_validate")
        transaction.end_validate()
        if prediction_transaction is not None: prediction_transaction.end_validate()
        if evaluation_transaction is not None: evaluation_transaction.end_validate()
        payload = canonical_json_bytes_antigravity(receipt.payload_antigravity())
        created = _append_leaf_at_descriptor_antigravity(receipt_dir, transaction.directory_fd, name, payload, fault_hook, created_holder)
        leaf_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=transaction.directory_fd)
        leaf_info = os.fstat(leaf_fd); leaf_payload = b"".join(iter(lambda: os.read(leaf_fd, 1 << 20), b""))
        appended_leaf = (name, leaf_fd, _namespace_identity_antigravity(leaf_info), leaf_payload, receipt)
        transaction.leaves.append(appended_leaf)
        transaction.directory_identity = _namespace_identity_antigravity(os.fstat(transaction.directory_fd))
        transaction.end_validate()
        _fault_antigravity(fault_hook, "before_precommit")
        if _precommit is not None: _precommit()
        # Direct append owns the two root/absence transactions.  Their close
        # failures remain precommit failures, not post-durable cleanup.
        for root_transaction in (prediction_transaction, evaluation_transaction):
            if root_transaction is not None:
                root_transaction.close(None)
        prediction_transaction = evaluation_transaction = None
        _fault_antigravity(fault_hook, "directory_close")
        _fault_antigravity(fault_hook, "before_directory_fsync"); os.fsync(transaction.directory_fd)
        _fault_antigravity(fault_hook, "before_parent_fsync"); _fault_antigravity(fault_hook, "parent_fsync"); os.fsync(transaction.parent_fd)
        expected_state = "STOPPED" if event == "stopped" else f"S{state_number + 1}"
        transaction.classification = ReceiptClassificationAntigravity(expected_state, event == "stopped", False, None, reviewed_commit, hashlib.sha256(payload).hexdigest())
        if own_transaction:
            try: transaction.close(None)
            except BaseException: pass
        return receipt
    except BaseException as error:
        primary = error
        if created is None and created_holder: created = created_holder[0]
        if appended_leaf is not None and transaction is not None:
            try: os.close(appended_leaf[1])
            except BaseException as close_error: error.add_note(f"created_leaf_close_error={type(close_error).__name__}: {close_error}")
            if transaction.leaves and transaction.leaves[-1] is appended_leaf: transaction.leaves.pop()
        if created is not None and transaction is not None:
            try:
                _remove_created_leaf_antigravity(transaction.directory_fd, name, created)
                transaction.directory_identity = _namespace_identity_antigravity(os.fstat(transaction.directory_fd))
            except BaseException as cleanup_error:
                error.add_note(f"created_leaf_cleanup_error={type(cleanup_error).__name__}: {cleanup_error}")
        raise
    finally:
        for root_transaction in (evaluation_transaction, prediction_transaction):
            if root_transaction is not None:
                try: root_transaction.close(primary)
                except BaseException as close_error:
                    if primary is not None: primary.add_note(f"root_transaction_close_error={type(close_error).__name__}: {close_error}")
                    else: raise
        if own_transaction and transaction is not None:
            try: transaction.close(primary)
            except BaseException as close_error:
                if primary is not None: primary.add_note(f"receipt_transaction_close_error={type(close_error).__name__}: {close_error}")
                else: raise


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


def _stop_and_audit_antigravity(receipt_dir: Path, audit_path: Path, phase: str, reviewed_commit: str, primary: BaseException, prediction_root: Path, evaluation_root: Path, *, transaction: _LifecycleTransactionAntigravity | None = None, root_identities: tuple[str | None, str | None] = (None, None)) -> None:
    notes: list[str] = []
    try:
        append_receipt_antigravity(receipt_dir, phase, "stopped", reviewed_commit, prediction_root=prediction_root, evaluation_root=evaluation_root, root_identities=root_identities, _transaction=transaction)
    except BaseException as secondary:
        primary.add_note(f"stopped_receipt_error={type(secondary).__name__}: {secondary}")
        notes.append(f"stopped_receipt_error={type(secondary).__name__}: {secondary}")
    post = transaction.classification if transaction is not None else validate_receipt_inventory_antigravity(receipt_dir, prediction_root=prediction_root, evaluation_root=evaluation_root)
    recovery_state = "stopped" if post.stopped else ("receipt_failed" if post.invalid else "stop_failed")
    if post.reason is not None:
        notes.append(f"post_stop_receipt={post.reason}")
    try:
        write_failure_audit_antigravity(audit_path, phase, _root_destinations_antigravity(phase), primary, recovery_state=recovery_state, cleanup_notes=tuple(notes))
    except BaseException as secondary:
        primary.add_note(f"failure_audit_error={type(secondary).__name__}: {secondary}")


def _close_complete_authorities_antigravity(roots: _CommandRootAuthorityAntigravity, audit_pin: _AbsentPathPinAntigravity) -> None:
    """Attempt root and audit close-all, raising only after both ownership sets close."""
    first: BaseException | None = None
    for authority in (roots, audit_pin):
        try: authority.close(None)
        except BaseException as error:
            if first is None: first = error
    if first is not None:
        raise first


def run_phase_antigravity(phase: str, reviewed_commit: str | None, operation: Callable[..., _T], *, receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY, audit_path: Path = _AUDIT_PATH_ANTIGRAVITY, prediction_root: Path = Path(_PREDICTION_ROOT_ANTIGRAVITY), evaluation_root: Path = Path(_EVALUATION_ROOT_ANTIGRAVITY), path_precondition: Callable[[], None] | None = None) -> _T:
    """Run one reusable lifecycle phase; synthetic tests inject only operation and paths."""
    bare_inspect = phase == "inspect" and reviewed_commit is None
    if phase not in _PREDECESSOR_ANTIGRAVITY or (not bare_inspect and not _is_lower_hex_antigravity(reviewed_commit, 40)) or not callable(operation) or not isinstance(receipt_dir, Path) or not isinstance(audit_path, Path) or not isinstance(prediction_root, Path) or not isinstance(evaluation_root, Path) or (path_precondition is not None and not callable(path_precondition)):
        raise ValueError("lifecycle phase inputs are invalid")
    try:
        transaction = _LifecycleTransactionAntigravity(receipt_dir, create_s0=True)
        classification = transaction.classification
        transaction.end_validate()
    except BaseException as error:
        if 'transaction' in locals():
            try: transaction.close(error)
            except BaseException as close_error: error.add_note(f"lifecycle_close_error={type(close_error).__name__}: {close_error}")
        raise ValueError("receipt inventory is invalid") from error
    if classification is None or classification.invalid:
        transaction.close(None)
        raise ValueError("receipt inventory is invalid")
    if classification.stopped:
        transaction.close(None)
        raise ValueError("receipt lifecycle is permanently stopped")
    audit_occupied = audit_path.exists() or audit_path.is_symlink()
    primary: BaseException | None = None; audit_pin = None
    # S0 bare inspect has no chain to derive.  Bind its required terminal
    # receipt to this deterministic sentinel so it follows the same direct
    # out-of-order stop/audit rule without inventing a caller-supplied review.
    chain_commit = classification.reviewed_commit or reviewed_commit or _NO_CHAIN_INSPECT_COMMIT_ANTIGRAVITY
    if classification.state != _PREDECESSOR_ANTIGRAVITY[phase]:
        primary = ValueError("command does not have the exact predecessor")
    elif not bare_inspect and classification.reviewed_commit is not None and classification.reviewed_commit != reviewed_commit:
        primary = ValueError("reviewed commit does not match receipt chain")
    elif audit_occupied:
        primary = ValueError("failure audit destination is occupied")
    else:
        try:
            permitted = ((prediction_root.name,) if phase == "predict" else ((evaluation_root.name,) if phase == "evaluate" else ()))
            audit_pin = _AbsentPathPinAntigravity(audit_path, permitted_names=permitted)
            roots = _CommandRootAuthorityAntigravity(phase, prediction_root, evaluation_root)
            roots.verify_receipts(transaction)
            if path_precondition is not None:
                path_precondition()
        except BaseException as error:
            primary = error
    if 'roots' not in locals():
        roots = None
    if primary is not None:
        if audit_pin is not None:
            try: audit_pin.close(primary)
            except BaseException as close_error: primary.add_note(f"audit_pin_close_error={type(close_error).__name__}: {close_error}")
        if roots is not None:
            try: roots.close(primary)
            except BaseException as close_error: primary.add_note(f"command_root_close_error={type(close_error).__name__}: {close_error}")
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, primary, prediction_root, evaluation_root, transaction=transaction)
        transaction.close(primary)
        raise primary
    try:
        audit_pin.end_validate()
        if phase != "readiness":
            append_receipt_antigravity(receipt_dir, phase, "started", chain_commit, prediction_root=prediction_root, evaluation_root=evaluation_root, root_identities=roots.identities, _transaction=transaction)
        audit_pin.end_validate()
        result = operation(chain_commit) if bare_inspect else operation()
        roots.capture_produced()
    except BaseException as error:
        if audit_pin is not None:
            try: audit_pin.close(error)
            except BaseException as close_error: error.add_note(f"audit_pin_close_error={type(close_error).__name__}: {close_error}")
        try: roots.close(error)
        except BaseException as close_error: error.add_note(f"command_root_close_error={type(close_error).__name__}: {close_error}")
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, error, prediction_root, evaluation_root, transaction=transaction)
        transaction.close(error)
        raise
    try:
        def complete_precommit() -> None:
            nonlocal audit_pin
            # The leaf is now written and reread.  Retained root and audit
            # authority remains live through this final validation and every
            # fallible authority close, all before receipt durability fsync.
            roots.end_validate(); audit_pin.end_validate()
            _close_complete_authorities_antigravity(roots, audit_pin); audit_pin = None
        append_receipt_antigravity(receipt_dir, phase, "complete", chain_commit, prediction_root=prediction_root, evaluation_root=evaluation_root, root_identities=roots.identities, _transaction=transaction, _precommit=complete_precommit)
    except BaseException as error:
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, error, prediction_root, evaluation_root, transaction=transaction)
        transaction.close(error)
        raise
    # Receipt close is deliberately post-commit: all descriptors are attempted,
    # but a late close cannot reclassify an already durable complete leaf.
    try: transaction.close(None)
    except BaseException: pass
    return result


def run_inspect_phase_antigravity(operation: Callable[[str], _T], **keywords) -> _T:
    """Derive inspect's reviewed commit from its already-pinned receipt transaction."""
    return run_phase_antigravity("inspect", None, operation, **keywords)


__all__ = ["LifecycleReceiptAntigravity", "ReceiptClassificationAntigravity", "append_receipt_antigravity", "receipt_reviewed_commit_antigravity", "run_inspect_phase_antigravity", "run_phase_antigravity", "validate_receipt_inventory_antigravity", "write_failure_audit_antigravity"]
