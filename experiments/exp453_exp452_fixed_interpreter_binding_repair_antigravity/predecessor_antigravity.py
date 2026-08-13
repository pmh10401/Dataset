"""Twice-captured dual predecessor: Exp452 failure evidence and Exp449 predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity
from .config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
    EXP449_PREDICTION_ROOT_ANTIGRAVITY,
    FROZEN_CONFIG_ANTIGRAVITY,
)


_EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp452_index_order_repair_lifecycle_receipts_antigravity_20260813"
)
_EXP452_FAILURE_AUDIT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp452_index_order_repair_failure_audit_antigravity.json"
)
_EXP452_EVALUATION_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp452_exp449_index_order_repair_evaluation_antigravity_20260813"
)
_EXP452_STOPPED_RECEIPT_SHA256_ANTIGRAVITY = (
    "8f7238ea20499a4136ffa1aaa730c9d94e35670b45eb780ccbe9d5594aff97e2"
)
_EXP452_FAILURE_AUDIT_SHA256_ANTIGRAVITY = (
    "e55985a66d18561a403c4927bfcd45a1cd20fe5af74841b96997e9c6e59a5b86"
)
_EXP452_STOPPED_LEAF_ANTIGRAVITY = "stopped_antigravity.json"
_DIRECTORY_FLAGS_ANTIGRAVITY = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_LEAF_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_IdentityAntigravity = tuple[int, int, int, int, int, int, int]


def _identity_antigravity(info: os.stat_result) -> _IdentityAntigravity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_descriptor_antigravity(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks)


def _after_leaf_reads_antigravity(_root: Path) -> None:
    """Private deterministic race seam used only by synthetic descriptor tests."""


@dataclass(frozen=True, slots=True)
class _CapturedDirectoryAntigravity:
    identity: _IdentityAntigravity
    leaves: tuple[tuple[str, bytes], ...]
    digests: tuple[tuple[str, str], ...]

    def payloads_antigravity(self) -> dict[str, bytes]:
        return dict(self.leaves)


def _capture_exact_directory_antigravity(
    root: Path,
    inventory: tuple[str, ...],
    *,
    expected_inode: int | None = None,
) -> _CapturedDirectoryAntigravity:
    """Pin one flat directory and every exact single-link leaf until final revalidation."""
    if not isinstance(root, Path) or type(inventory) is not tuple or len(set(inventory)) != len(inventory):
        raise ValueError("predecessor directory inputs are invalid")
    root_fd = -1
    pinned: list[tuple[str, int, _IdentityAntigravity, bytes]] = []
    try:
        before = os.stat(root, follow_symlinks=False)
        root_fd = os.open(root, _DIRECTORY_FLAGS_ANTIGRAVITY)
        opened = os.fstat(root_fd)
        root_identity = _identity_antigravity(opened)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity_antigravity(before) != root_identity
            or (expected_inode is not None and opened.st_ino != expected_inode)
            or set(os.listdir(root_fd)) != set(inventory)
        ):
            raise ValueError("predecessor directory identity or inventory differs")
        leaves: list[tuple[str, bytes]] = []
        digests: list[tuple[str, str]] = []
        for name in inventory:
            if type(name) is not str or Path(name).parts != (name,):
                raise ValueError("predecessor inventory name is invalid")
            namespace = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            descriptor = os.open(name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=root_fd)
            leaf_info = os.fstat(descriptor)
            identity = _identity_antigravity(leaf_info)
            if (
                not stat.S_ISREG(leaf_info.st_mode)
                or leaf_info.st_nlink != 1
                or _identity_antigravity(namespace) != identity
            ):
                os.close(descriptor)
                raise ValueError("predecessor leaf is not a pinned single-link regular file")
            payload = _read_descriptor_antigravity(descriptor)
            pinned.append((name, descriptor, identity, payload))
            leaves.append((name, payload))
            digests.append((name, hashlib.sha256(payload).hexdigest()))
        _after_leaf_reads_antigravity(root)
        for name, descriptor, identity, payload in pinned:
            if (
                _identity_antigravity(os.fstat(descriptor)) != identity
                or _identity_antigravity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity
                or _read_descriptor_antigravity(descriptor) != payload
            ):
                raise ValueError("predecessor leaf changed during capture")
        if (
            _identity_antigravity(os.fstat(root_fd)) != root_identity
            or _identity_antigravity(os.stat(root, follow_symlinks=False)) != root_identity
            or set(os.listdir(root_fd)) != set(inventory)
        ):
            raise ValueError("predecessor directory changed during capture")
        return _CapturedDirectoryAntigravity(root_identity, tuple(leaves), tuple(digests))
    except OSError as error:
        raise ValueError("predecessor directory is missing or unsafe") from error
    finally:
        for _name, descriptor, _identity, _payload in reversed(pinned):
            os.close(descriptor)
        if root_fd >= 0:
            os.close(root_fd)


@dataclass(frozen=True, slots=True)
class _CapturedLeafAntigravity:
    identity: _IdentityAntigravity
    payload: bytes
    sha256: str


def _capture_leaf_antigravity(path: Path) -> _CapturedLeafAntigravity:
    if not isinstance(path, Path):
        raise ValueError("predecessor leaf path is invalid")
    parent_fd = descriptor = -1
    try:
        parent_fd = os.open(path.parent, _DIRECTORY_FLAGS_ANTIGRAVITY)
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(path.name, _LEAF_FLAGS_ANTIGRAVITY, dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        identity = _identity_antigravity(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _identity_antigravity(before) != identity:
            raise ValueError("predecessor leaf is not a pinned single-link regular file")
        payload = _read_descriptor_antigravity(descriptor)
        if (
            _identity_antigravity(os.fstat(descriptor)) != identity
            or _identity_antigravity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)) != identity
            or _read_descriptor_antigravity(descriptor) != payload
        ):
            raise ValueError("predecessor leaf changed during capture")
        return _CapturedLeafAntigravity(identity, payload, hashlib.sha256(payload).hexdigest())
    except OSError as error:
        raise ValueError("predecessor leaf is missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def _require_absent_antigravity(path: Path) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Exp452 evaluation root absence is unverifiable") from error
    raise ValueError("Exp452 evaluation root must remain absent")


def _scientific_config_sha256_antigravity() -> str:
    keys = (
        "seeds",
        "alphas",
        "seed_aggregation",
        "calibration",
        "crossfit_folds",
        "knn_neighbors",
        "max_train_reference",
        "query_chunk_size",
        "minirocket_requested_kernels",
        "multirocket_requested_kernels",
        "multirocket_max_dilations_per_kernel",
        "multirocket_features_per_kernel",
        "multirocket_normalise",
        "hydra_kernels",
        "hydra_groups",
        "global_fn_must_be_below",
        "global_fp_max",
        "global_f1_min",
        "candidate_matrix",
    )
    projection = {key: FROZEN_CONFIG_ANTIGRAVITY[key] for key in keys}
    return hashlib.sha256(canonical_json_bytes_antigravity(projection)).hexdigest()


def _strict_object_antigravity(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = strict_json_bytes_antigravity(payload)
    except ValueError as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict or canonical_json_bytes_antigravity(value) != payload:
        raise ValueError(f"{name} is not a canonical JSON object")
    return value


@dataclass(frozen=True, slots=True)
class _PredictionSnapshotAntigravity:
    root_identity: _IdentityAntigravity
    aggregate_sha256: str
    canonical_rows: bytes
    dataset_order: tuple[str, ...]


def _capture_prediction_antigravity(
    root: Path,
    expected_aggregate_sha256: str,
) -> _PredictionSnapshotAntigravity:
    from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (
        capture_frozen_prediction_authority_antigravity,
    )

    descriptor = -1
    try:
        before = os.stat(root, follow_symlinks=False)
        descriptor = os.open(root, _DIRECTORY_FLAGS_ANTIGRAVITY)
        opened = os.fstat(descriptor)
        root_identity = _identity_antigravity(opened)
        if not stat.S_ISDIR(opened.st_mode) or _identity_antigravity(before) != root_identity:
            raise ValueError("Exp449 prediction root is unsafe")
        capture = capture_frozen_prediction_authority_antigravity(root=root)
        authority = capture.authority
        if (
            authority.aggregate_sha256 != expected_aggregate_sha256
            or authority.family_method_pairs != 16
            or authority.dataset_units != 96
            or authority.label_access_count != 0
            or type(capture.canonical_rows) is not bytes
        ):
            raise ValueError("Exp449 prediction aggregate differs")
        try:
            rows = tuple(strict_json_bytes_antigravity(line + b"\n") for line in capture.canonical_rows.splitlines())
        except ValueError as error:
            raise ValueError("Exp449 prediction rows are invalid") from error
        expected_pairs = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
        if len(rows) != 96 or any(type(row) is not dict for row in rows):
            raise ValueError("Exp449 prediction row count differs")
        datasets: list[str] = []
        for sequence, row in enumerate(rows):
            required = {"sequence", "candidate_name", "dataset", "family", "method", "indices", "target_length"}
            family = row.get("family") if type(row) is dict else None
            if (
                set(row) != required
                or row.get("sequence") != sequence
                or type(row.get("sequence")) is not int
                or type(row.get("dataset")) is not str
                or type(family) is not str
                or row.get("method") not in expected_pairs.get(family, ())
                or row.get("candidate_name") != f'{row.get("dataset")}__{row.get("method")}'
                or type(row.get("indices")) is not list
                or row["indices"] != sorted(set(row["indices"]))
                or any(type(index) is not int or index < 0 or index >= row.get("target_length", -1) for index in row["indices"])
            ):
                raise ValueError("Exp449 prediction row differs")
            if sequence % 2 == 0:
                datasets.append(row["dataset"])
            elif row["dataset"] != datasets[-1]:
                raise ValueError("Exp449 prediction dataset pairing differs")
        dataset_order = tuple(datasets)
        if len(dataset_order) != 48 or len(set(dataset_order)) != 48:
            raise ValueError("Exp449 prediction dataset order differs")
        if (
            _identity_antigravity(os.fstat(descriptor)) != root_identity
            or _identity_antigravity(os.stat(root, follow_symlinks=False)) != root_identity
        ):
            raise ValueError("Exp449 prediction root changed during capture")
        return _PredictionSnapshotAntigravity(
            root_identity,
            authority.aggregate_sha256,
            capture.canonical_rows,
            dataset_order,
        )
    except OSError as error:
        raise ValueError("Exp449 prediction root is missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_exp452_receipt_antigravity(payload: bytes, expected_sha256: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Exp452 stopped receipt digest differs")
    receipt = _strict_object_antigravity(payload, "Exp452 stopped receipt")
    if (
        receipt.get("schema") != "exp452_lifecycle_receipt"
        or receipt.get("experiment_id") != "Exp452"
        or receipt.get("event") != "stopped"
        or receipt.get("phase") != "readiness"
        or receipt.get("permanent_stop") is not True
        or receipt.get("evaluation_root_identity") is not None
    ):
        raise ValueError("Exp452 stopped receipt facts differ")


def _validate_exp452_audit_antigravity(captured: _CapturedLeafAntigravity, expected_sha256: str) -> None:
    if captured.sha256 != expected_sha256:
        raise ValueError("Exp452 failure audit digest differs")
    audit = _strict_object_antigravity(captured.payload, "Exp452 failure audit")
    if (
        audit.get("schema") != "exp452.lifecycle.failure"
        or audit.get("phase") != "readiness"
        or audit.get("permanent_stop") is not True
        or audit.get("recovery_state") != "stopped"
        or audit.get("primary_error") != "ValueError: Exp452 requires the exact fixed Python and NumPy runtime"
    ):
        raise ValueError("Exp452 failure audit facts differ")


def _no_between_captures_antigravity() -> None:
    return None


@dataclass(frozen=True, slots=True)
class _PredecessorDependenciesAntigravity:
    prediction_root: Path
    exp452_receipt_directory: Path
    exp452_failure_audit: Path
    exp452_evaluation_root: Path
    expected_prediction_aggregate_sha256: str
    expected_exp452_receipt_sha256: str
    expected_exp452_audit_sha256: str
    between_captures: Callable[[], None] = _no_between_captures_antigravity

    def __post_init__(self) -> None:
        paths = (
            self.prediction_root,
            self.exp452_receipt_directory,
            self.exp452_failure_audit,
            self.exp452_evaluation_root,
        )
        hashes = (
            self.expected_prediction_aggregate_sha256,
            self.expected_exp452_receipt_sha256,
            self.expected_exp452_audit_sha256,
        )
        if (
            any(not isinstance(path, Path) for path in paths)
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or not callable(self.between_captures)
        ):
            raise ValueError("predecessor dependencies are invalid")


def _production_dependencies_antigravity() -> _PredecessorDependenciesAntigravity:
    return _PredecessorDependenciesAntigravity(
        prediction_root=Path(EXP449_PREDICTION_ROOT_ANTIGRAVITY),
        exp452_receipt_directory=_EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY,
        exp452_failure_audit=_EXP452_FAILURE_AUDIT_ANTIGRAVITY,
        exp452_evaluation_root=_EXP452_EVALUATION_ROOT_ANTIGRAVITY,
        expected_prediction_aggregate_sha256=EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
        expected_exp452_receipt_sha256=_EXP452_STOPPED_RECEIPT_SHA256_ANTIGRAVITY,
        expected_exp452_audit_sha256=_EXP452_FAILURE_AUDIT_SHA256_ANTIGRAVITY,
    )


@dataclass(frozen=True, slots=True)
class _DualSnapshotAntigravity:
    prediction: _PredictionSnapshotAntigravity
    receipt_identity: _IdentityAntigravity
    receipt_sha256: str
    failure_audit_identity: _IdentityAntigravity
    failure_audit_sha256: str
    scientific_config_sha256: str


def _capture_once_antigravity(dependencies: _PredecessorDependenciesAntigravity) -> _DualSnapshotAntigravity:
    _require_absent_antigravity(dependencies.exp452_evaluation_root)
    receipts = _capture_exact_directory_antigravity(
        dependencies.exp452_receipt_directory,
        (_EXP452_STOPPED_LEAF_ANTIGRAVITY,),
    )
    stopped_payload = receipts.payloads_antigravity()[_EXP452_STOPPED_LEAF_ANTIGRAVITY]
    _validate_exp452_receipt_antigravity(stopped_payload, dependencies.expected_exp452_receipt_sha256)
    audit = _capture_leaf_antigravity(dependencies.exp452_failure_audit)
    _validate_exp452_audit_antigravity(audit, dependencies.expected_exp452_audit_sha256)
    prediction = _capture_prediction_antigravity(
        dependencies.prediction_root,
        dependencies.expected_prediction_aggregate_sha256,
    )
    _require_absent_antigravity(dependencies.exp452_evaluation_root)
    return _DualSnapshotAntigravity(
        prediction,
        receipts.identity,
        hashlib.sha256(stopped_payload).hexdigest(),
        audit.identity,
        audit.sha256,
        _scientific_config_sha256_antigravity(),
    )


@dataclass(frozen=True, slots=True)
class _Exp453PredecessorAuthorityAntigravity:
    predecessor_attestation_sha256: str
    prediction_aggregate_sha256: str
    scientific_config_sha256: str
    exp452_stopped_receipt_sha256: str
    exp452_failure_audit_sha256: str
    exp452_evaluation_root_absent: bool
    dataset_order: tuple[str, ...]
    canonical_prediction_rows: bytes
    dataset_count: int
    prediction_count: int
    label_session_count: int
    label_query_count: int
    label_row_count: int


_AUTHORITY_STATES_ANTIGRAVITY: dict[int, tuple[_Exp453PredecessorAuthorityAntigravity, str]] = {}


def _attestation_sha256_antigravity(snapshot: _DualSnapshotAntigravity) -> str:
    payload = {
        "prediction_aggregate_sha256": snapshot.prediction.aggregate_sha256,
        "prediction_root_identity": list(snapshot.prediction.root_identity),
        "exp452_stopped_receipt_sha256": snapshot.receipt_sha256,
        "exp452_failure_audit_sha256": snapshot.failure_audit_sha256,
        "scientific_config_sha256": snapshot.scientific_config_sha256,
        "dataset_order": list(snapshot.prediction.dataset_order),
        "label_session_count": 0,
        "label_query_count": 0,
        "label_row_count": 0,
    }
    return hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()


def _capture_predecessors_with_dependencies_antigravity(
    dependencies: _PredecessorDependenciesAntigravity,
) -> _Exp453PredecessorAuthorityAntigravity:
    if type(dependencies) is not _PredecessorDependenciesAntigravity:
        raise ValueError("predecessor dependencies are invalid")
    first = _capture_once_antigravity(dependencies)
    dependencies.between_captures()
    try:
        second = _capture_once_antigravity(dependencies)
    except BaseException as error:
        raise ValueError("two independent captures differ") from error
    if first is second or first != second:
        raise ValueError("two distinct byte/root-identical independent captures are required")
    authority = _Exp453PredecessorAuthorityAntigravity(
        predecessor_attestation_sha256=_attestation_sha256_antigravity(first),
        prediction_aggregate_sha256=first.prediction.aggregate_sha256,
        scientific_config_sha256=first.scientific_config_sha256,
        exp452_stopped_receipt_sha256=first.receipt_sha256,
        exp452_failure_audit_sha256=first.failure_audit_sha256,
        exp452_evaluation_root_absent=True,
        dataset_order=first.prediction.dataset_order,
        canonical_prediction_rows=first.prediction.canonical_rows,
        dataset_count=48,
        prediction_count=96,
        label_session_count=0,
        label_query_count=0,
        label_row_count=0,
    )
    _AUTHORITY_STATES_ANTIGRAVITY[id(authority)] = (authority, "issued")
    return authority


def capture_predecessors_antigravity() -> _Exp453PredecessorAuthorityAntigravity:
    """Authenticate Exp452 failure evidence and Exp449 prediction bytes twice."""
    return _capture_predecessors_with_dependencies_antigravity(_production_dependencies_antigravity())


def _authority_entry_antigravity(authority: object) -> tuple[_Exp453PredecessorAuthorityAntigravity, str]:
    entry = _AUTHORITY_STATES_ANTIGRAVITY.get(id(authority))
    if (
        type(authority) is not _Exp453PredecessorAuthorityAntigravity
        or entry is None
        or entry[0] is not authority
        or authority.scientific_config_sha256 != _scientific_config_sha256_antigravity()
        or authority.dataset_count != 48
        or authority.prediction_count != 96
        or authority.exp452_evaluation_root_absent is not True
        or authority.label_session_count != 0
        or authority.label_query_count != 0
        or authority.label_row_count != 0
    ):
        raise ValueError("Exp453 predecessor authority is invalid")
    return entry


def validate_predecessors_antigravity(authority: object) -> tuple[str, ...]:
    """Validate only an exact object issued after two full matching captures."""
    exact, _state = _authority_entry_antigravity(authority)
    return exact.dataset_order


def canonical_prediction_rows_antigravity(authority: object) -> bytes:
    exact, _state = _authority_entry_antigravity(authority)
    return exact.canonical_prediction_rows


def _claim_label_capability_antigravity(authority: object) -> tuple[str, ...]:
    exact, state = _authority_entry_antigravity(authority)
    if state != "issued":
        raise ValueError("Exp453 predecessor authority label capability was already consumed")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, "opening")
    return exact.dataset_order


def _finish_label_capability_antigravity(authority: object, outcome: str) -> None:
    exact, state = _authority_entry_antigravity(authority)
    if state not in {"opening", "open"} or outcome not in {"open", "closed", "terminal", "close_failed", "setup_failed"}:
        raise ValueError("label capability transition is invalid")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, outcome)


__all__ = [
    "canonical_prediction_rows_antigravity",
    "capture_predecessors_antigravity",
    "validate_predecessors_antigravity",
]
