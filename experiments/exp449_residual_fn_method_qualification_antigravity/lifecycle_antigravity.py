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


class _ReceiptTransactionAntigravity:
    """One retained parent/receipt descriptor transaction for classify, append, and revalidation."""
    __slots__ = ("path", "parent_fd", "directory_fd", "directory_identity", "leaves", "classification")

    def __init__(self, path: Path, *, create_s0: bool) -> None:
        self.path = path; self.parent_fd = self.directory_fd = -1; self.directory_identity = None
        self.leaves: list[tuple[str, int, tuple[int, int], bytes, LifecycleReceiptAntigravity]] = []
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
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("receipt directory was replaced")
            self.directory_identity = (opened.st_dev, opened.st_ino)
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
            info = os.fstat(fd); identity = (info.st_dev, info.st_ino)
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
            self.leaves.append((name, fd, identity, payload, receipt)); previous = hashlib.sha256(payload).hexdigest()
        if stopped:
            fd = os.open(_STOPPED_LEAF_ANTIGRAVITY, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
            info = os.fstat(fd); identity = (info.st_dev, info.st_ino); payload = b"".join(iter(lambda: os.read(fd, 1 << 20), b"")); receipt = _receipt_from_bytes_antigravity(payload)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or receipt.event != "stopped" or not receipt.permanent_stop or receipt.ordinal != len(success) or receipt.previous_leaf_sha256 != previous or (reviewed is not None and receipt.reviewed_commit != reviewed):
                raise ValueError("terminal receipt differs")
            self.leaves.append((_STOPPED_LEAF_ANTIGRAVITY, fd, identity, payload, receipt)); return ReceiptClassificationAntigravity("STOPPED", True, False, None, receipt.reviewed_commit, hashlib.sha256(payload).hexdigest())
        return ReceiptClassificationAntigravity(f"S{len(success)}", False, False, None, reviewed, previous)

    def end_validate(self) -> None:
        if self.directory_fd < 0: return
        retained = os.fstat(self.directory_fd)
        current = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if (retained.st_dev, retained.st_ino) != self.directory_identity or (current.st_dev, current.st_ino) != self.directory_identity:
            raise ValueError("receipt directory was rebound")
        if set(os.listdir(self.directory_fd)) != {item[0] for item in self.leaves}:
            raise ValueError("receipt inventory changed during transaction")
        for name, fd, identity, payload, _receipt in self.leaves:
            info = os.fstat(fd); current = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != identity or (current.st_dev, current.st_ino) != identity:
                raise ValueError("receipt leaf was replaced")
            os.lseek(fd, 0, os.SEEK_SET)
            if b"".join(iter(lambda: os.read(fd, 1 << 20), b"")) != payload:
                raise ValueError("receipt leaf changed")

    def close(self, primary: BaseException | None) -> None:
        for _name, fd, _identity, _payload, _receipt in reversed(self.leaves):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"receipt_leaf_close_error={type(error).__name__}: {error}")
                else: raise
        self.leaves.clear()
        for descriptor in (self.directory_fd, self.parent_fd):
            if descriptor >= 0:
                try: os.close(descriptor)
                except BaseException as error:
                    if primary is not None: primary.add_note(f"receipt_directory_close_error={type(error).__name__}: {error}")
                    else: raise
        self.directory_fd = self.parent_fd = -1


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
        transaction = _ReceiptTransactionAntigravity(receipt_dir, create_s0=False)
        result = transaction.classification
        transaction.end_validate()
        transaction.close(None)
        if result is None: raise ValueError("receipt transaction did not classify")
        return result
    except BaseException as error:
        return ReceiptClassificationAntigravity("INVALID", False, True, f"{type(error).__name__}: {error}")
    # Historical path-based implementation retained below only for source
    # archaeology; every caller returns through the retained transaction above.
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
    __slots__ = ("root", "root_fd", "descriptors", "records", "identity")

    def __init__(self, root: Path, fault_hook=None) -> None:
        self.root = root; self.root_fd = -1; self.descriptors: list[tuple[int, tuple[int, int], tuple[str, ...]]] = []; self.records: list[tuple[int, tuple[int, int], str, bytes]] = []; self.identity = None
        try:
            path_info = os.lstat(root)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(path_info.st_mode) or not stat.S_ISDIR(path_info.st_mode): raise ValueError("root identity path is unsafe")
        self.root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        root_info = os.fstat(self.root_fd)
        if (root_info.st_dev, root_info.st_ino) != (path_info.st_dev, path_info.st_ino): raise ValueError("root identity path was replaced")
        digest = hashlib.sha256()
        def walk(fd: int, prefix: tuple[str, ...]) -> None:
            info = os.fstat(fd); self.descriptors.append((fd, (info.st_dev, info.st_ino), prefix))
            for name in sorted(os.listdir(fd)):
                item = os.stat(name, dir_fd=fd, follow_symlinks=False); relative = "/".join(prefix + (name,))
                if stat.S_ISDIR(item.st_mode):
                    digest.update(b"D\0" + relative.encode() + b"\0")
                    child = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
                    if (os.fstat(child).st_dev, os.fstat(child).st_ino) != (item.st_dev, item.st_ino): raise ValueError("root directory was replaced")
                    walk(child, prefix + (name,))
                    _fault_antigravity(fault_hook, "after_child_capture")
                elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                    child = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd); child_info = os.fstat(child)
                    if (child_info.st_dev, child_info.st_ino) != (item.st_dev, item.st_ino): raise ValueError("root leaf was replaced")
                    payload = b"".join(iter(lambda: os.read(child, 1 << 20), b"")); digest.update(b"F\0" + relative.encode() + b"\0" + payload)
                    self.records.append((child, (child_info.st_dev, child_info.st_ino), relative, payload))
                    _fault_antigravity(fault_hook, "after_leaf_read")
                else: raise ValueError("root identity inventory is unsafe")
        walk(self.root_fd, ())
        self.identity = digest.hexdigest()

    def end_validate(self) -> None:
        if self.root_fd < 0: return
        path_info = os.stat(self.root, follow_symlinks=False); info = os.fstat(self.root_fd)
        if (path_info.st_dev, path_info.st_ino) != (info.st_dev, info.st_ino): raise ValueError("root namespace was replaced")
        for fd, identity, prefix in self.descriptors:
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) != identity: raise ValueError("root descriptor changed")
            if prefix:
                parent_fd = next(parent for parent, _parent_identity, parent_prefix in self.descriptors if parent_prefix == prefix[:-1])
                namespace = os.stat(prefix[-1], dir_fd=parent_fd, follow_symlinks=False)
                if (namespace.st_dev, namespace.st_ino) != identity: raise ValueError("root directory was replaced")
            expected = {path[-1] for _fd, _id, path in self.descriptors if path and path[:-1] == prefix} | {relative.rsplit("/", 1)[-1] for _fd, _id, relative, _payload in self.records if tuple(relative.split("/")[:-1]) == prefix}
            if set(os.listdir(fd)) != expected: raise ValueError("root inventory changed")
        for fd, identity, relative, payload in self.records:
            info = os.fstat(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            if (info.st_dev, info.st_ino) != identity or b"".join(iter(lambda: os.read(fd, 1 << 20), b"")) != payload: raise ValueError("root leaf changed")

    def close(self, primary: BaseException | None) -> None:
        for fd, *_ in reversed(self.records):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_leaf_close_error={type(error).__name__}: {error}")
        for fd, *_ in reversed(self.descriptors):
            try: os.close(fd)
            except BaseException as error:
                if primary is not None: primary.add_note(f"root_directory_close_error={type(error).__name__}: {error}")
        self.records.clear(); self.descriptors.clear(); self.root_fd = -1


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
    if not receipt_dir.exists() and not receipt_dir.is_symlink():
        _fault_antigravity(fault_hook, "directory_create")
    transaction = _ReceiptTransactionAntigravity(receipt_dir, create_s0=True)
    classification = transaction.classification
    if classification is None or classification.stopped:
        transaction.close(None); raise ValueError("receipt namespace cannot be appended")
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
    prediction_transaction = _RootIdentityTransactionAntigravity(prediction_root, fault_hook)
    evaluation_transaction = _RootIdentityTransactionAntigravity(evaluation_root, fault_hook)
    prediction_identity, evaluation_identity = _stage_root_fields_antigravity(phase, event, prediction_transaction.identity, evaluation_transaction.identity)
    receipt = LifecycleReceiptAntigravity(_SCHEMA_ANTIGRAVITY, _EXPERIMENT_ID_ANTIGRAVITY, _RUN_IDENTITY_ANTIGRAVITY, reviewed_commit, phase, event, state_number, previous, prediction_identity, evaluation_identity, event == "stopped")
    primary: BaseException | None = None
    try:
        _append_leaf_at_descriptor_antigravity(receipt_dir, transaction.directory_fd, name, canonical_json_bytes_antigravity(receipt.payload_antigravity()), fault_hook)
        leaf_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=transaction.directory_fd)
        leaf_info = os.fstat(leaf_fd); leaf_payload = b"".join(iter(lambda: os.read(leaf_fd, 1 << 20), b""))
        transaction.leaves.append((name, leaf_fd, (leaf_info.st_dev, leaf_info.st_ino), leaf_payload, receipt))
        transaction.end_validate(); os.fsync(transaction.directory_fd)
    except BaseException as error:
        primary = error
        raise
    finally:
        close_error: BaseException | None = None
        try:
            _fault_antigravity(fault_hook, "directory_close")
        except BaseException as error:
            if primary is not None: primary.add_note(f"directory_close_error={type(error).__name__}: {error}")
            else: close_error = error
        transaction.close(primary)
        try:
            _fault_antigravity(fault_hook, "before_end_validate")
            prediction_transaction.end_validate(); evaluation_transaction.end_validate()
        except BaseException as error:
            if primary is not None: primary.add_note(f"root_identity_revalidation_error={type(error).__name__}: {error}")
            else: raise
        finally:
            prediction_transaction.close(primary); evaluation_transaction.close(primary)
        if close_error is not None:
            raise close_error
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
    notes: list[str] = []
    try:
        append_receipt_antigravity(receipt_dir, phase, "stopped", reviewed_commit, prediction_root=prediction_root, evaluation_root=evaluation_root)
    except BaseException as secondary:
        primary.add_note(f"stopped_receipt_error={type(secondary).__name__}: {secondary}")
        notes.append(f"stopped_receipt_error={type(secondary).__name__}: {secondary}")
    post = validate_receipt_inventory_antigravity(receipt_dir)
    recovery_state = "stopped" if post.stopped else ("receipt_failed" if post.invalid else "stop_failed")
    if post.reason is not None:
        notes.append(f"post_stop_receipt={post.reason}")
    try:
        write_failure_audit_antigravity(audit_path, phase, _root_destinations_antigravity(phase), primary, recovery_state=recovery_state, cleanup_notes=tuple(notes))
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
    try:
        # A complete receipt is a durable commit point only after the complete
        # append function has returned from its own descriptor revalidation.
        append_receipt_antigravity(receipt_dir, phase, "complete", chain_commit, prediction_root=prediction_root, evaluation_root=evaluation_root)
    except BaseException as error:
        _stop_and_audit_antigravity(receipt_dir, audit_path, phase, chain_commit, error, prediction_root, evaluation_root)
        raise
    return result


__all__ = ["LifecycleReceiptAntigravity", "ReceiptClassificationAntigravity", "append_receipt_antigravity", "receipt_reviewed_commit_antigravity", "run_phase_antigravity", "validate_receipt_inventory_antigravity", "write_failure_audit_antigravity"]
