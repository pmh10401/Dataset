"""Exp453 terminal hashes plus the single Exp449 target-authority conversion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable

from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import (
    capture_exp448_authority_antigravity,
)

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity
from .config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
    EXP449_PREDICTION_ROOT_ANTIGRAVITY,
    FROZEN_CONFIG_ANTIGRAVITY,
)
from .contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity


EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY = "1b6b2ea01c0c4dde9128051e57c764d32cbdf8bc1af7fa0115f19c7a10181d97"
EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY = "38607b6c8cad785ca8cd0830dacd42775d43f9175a67ed01d487657a3c44764e"
EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY = "caa705be90ddd56a00c53d5eb94a90604768dac0a80274846adce2061349b3b6"
EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY = "97e7bea79bd1a830b925e7ffa2234e4e34725ac6734e080216f409688cb58276"


@dataclass(frozen=True, slots=True)
class NormalizedTargetAuthorityAntigravity:
    target_dataset_names: tuple[str, ...]
    target_baseline_rows: tuple[DatasetBaselineAntigravity, ...]
    baseline_confusion: ConfusionAntigravity


def _target_snapshot_antigravity(authority: object) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...], tuple[int, int, int, int]]:
    datasets = getattr(authority, "target_dataset_names", None)
    raw_rows = getattr(authority, "target_baseline_rows", None)
    raw_global = getattr(authority, "baseline_confusion", None)
    if type(datasets) is not tuple or type(raw_rows) is not tuple or len(datasets) != 48 or len(raw_rows) != 48:
        raise ValueError("Exp449 target inventory differs")
    rows: list[tuple[object, ...]] = []
    for dataset, raw in zip(datasets, raw_rows, strict=True):
        order = getattr(raw, "dataset_order", None)
        family = getattr(raw, "family", None)
        candidates = getattr(raw, "stable_candidates", None)
        confusion = getattr(raw, "confusion", None)
        if type(dataset) is not str or type(order) is not int or type(family) is not str or type(candidates) is not frozenset:
            raise ValueError("Exp449 target order differs")
        rows.append(
            (
                dataset,
                order,
                family,
                frozenset(candidates),
                (
                    getattr(confusion, "tp", None),
                    getattr(confusion, "fp", None),
                    getattr(confusion, "fn", None),
                    getattr(confusion, "tn", None),
                ),
            )
        )
    global_confusion = (
        getattr(raw_global, "tp", None),
        getattr(raw_global, "fp", None),
        getattr(raw_global, "fn", None),
        getattr(raw_global, "tn", None),
    )
    if any(type(value) is not int for value in global_confusion) or any(
        type(value) is not int for _dataset, _order, _family, _candidates, counts in rows for value in counts
    ):
        raise ValueError("Exp449 target inventory differs")
    return datasets, tuple(rows), global_confusion  # type: ignore[return-value]


def normalize_exp449_target_authority_antigravity(authority: object) -> NormalizedTargetAuthorityAntigravity:
    """Copy exact Exp449 target fields into Exp454 immutable types once."""
    snapshot = _target_snapshot_antigravity(authority)
    expected = _target_snapshot_antigravity(capture_exp448_authority_antigravity())
    if snapshot != expected:
        if snapshot[0] != expected[0] or tuple(row[1] for row in snapshot[1]) != tuple(row[1] for row in expected[1]):
            raise ValueError("Exp449 target order differs")
        raise ValueError("Exp449 target authority differs")
    datasets, rows, global_confusion = snapshot
    copied = []
    for _dataset, order, family, candidates, counts in rows:
        copied.append(
            DatasetBaselineAntigravity(
                order,  # type: ignore[arg-type]
                family,  # type: ignore[arg-type]
                frozenset(candidates),  # type: ignore[arg-type]
                ConfusionAntigravity(*counts),  # type: ignore[arg-type]
            )
        )
    return NormalizedTargetAuthorityAntigravity(datasets, tuple(copied), ConfusionAntigravity(*global_confusion))


def capture_normalized_target_authority_antigravity() -> NormalizedTargetAuthorityAntigravity:
    """Call the real Exp449 capture and apply the single conversion."""
    return normalize_exp449_target_authority_antigravity(capture_exp448_authority_antigravity())


_EXP453_RECEIPT_DIRECTORY_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp453_fixed_interpreter_binding_repair_lifecycle_receipts_antigravity_20260813"
)
_EXP453_FAILURE_AUDIT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp453_fixed_interpreter_binding_repair_failure_audit_antigravity.json"
)
_EXP453_EVALUATION_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp453_exp452_fixed_interpreter_binding_repair_evaluation_antigravity_20260813"
)
_EXP453_RECEIPT_INVENTORY_ANTIGRAVITY = (
    "readiness_complete_antigravity.json",
    "evaluate_started_antigravity.json",
    "stopped_antigravity.json",
)
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
        raise ValueError("Exp453 evaluation root absence is unverifiable") from error
    raise ValueError("Exp453 evaluation root must remain absent")


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


def _validate_exp453_receipts_antigravity(
    payloads: dict[str, bytes],
    expected_readiness_sha256: str,
    expected_evaluate_sha256: str,
    expected_stopped_sha256: str,
) -> tuple[str, str, str]:
    readiness_payload = payloads["readiness_complete_antigravity.json"]
    evaluate_payload = payloads["evaluate_started_antigravity.json"]
    stopped_payload = payloads["stopped_antigravity.json"]
    readiness_sha = hashlib.sha256(readiness_payload).hexdigest()
    evaluate_sha = hashlib.sha256(evaluate_payload).hexdigest()
    stopped_sha = hashlib.sha256(stopped_payload).hexdigest()
    if readiness_sha != expected_readiness_sha256:
        raise ValueError("Exp453 readiness receipt digest differs")
    if evaluate_sha != expected_evaluate_sha256:
        raise ValueError("Exp453 evaluate receipt digest differs")
    if stopped_sha != expected_stopped_sha256:
        raise ValueError("Exp453 stopped receipt digest differs")
    readiness = _strict_object_antigravity(readiness_payload, "Exp453 readiness receipt")
    evaluate = _strict_object_antigravity(evaluate_payload, "Exp453 evaluate receipt")
    stopped = _strict_object_antigravity(stopped_payload, "Exp453 stopped receipt")
    if (
        readiness.get("schema") != "exp453_lifecycle_receipt"
        or readiness.get("experiment_id") != "Exp453"
        or readiness.get("phase") != "readiness"
        or readiness.get("event") != "complete"
        or readiness.get("permanent_stop") is not False
        or readiness.get("evaluation_root_identity") is not None
    ):
        raise ValueError("Exp453 readiness receipt facts differ")
    if (
        evaluate.get("schema") != "exp453_lifecycle_receipt"
        or evaluate.get("experiment_id") != "Exp453"
        or evaluate.get("phase") != "evaluate"
        or evaluate.get("event") != "started"
        or evaluate.get("permanent_stop") is not False
        or evaluate.get("evaluation_root_identity") is not None
        or evaluate.get("previous_leaf_sha256") != readiness_sha
    ):
        raise ValueError("Exp453 evaluate receipt facts differ")
    if (
        stopped.get("schema") != "exp453_lifecycle_receipt"
        or stopped.get("experiment_id") != "Exp453"
        or stopped.get("phase") != "evaluate"
        or stopped.get("event") != "stopped"
        or stopped.get("permanent_stop") is not True
        or stopped.get("evaluation_root_identity") is not None
        or stopped.get("previous_leaf_sha256") != evaluate_sha
    ):
        raise ValueError("Exp453 stopped receipt facts differ")
    return readiness_sha, evaluate_sha, stopped_sha


def _validate_exp453_audit_antigravity(captured: _CapturedLeafAntigravity, expected_sha256: str) -> None:
    if captured.sha256 != expected_sha256:
        raise ValueError("Exp453 failure audit digest differs")
    audit = _strict_object_antigravity(captured.payload, "Exp453 failure audit")
    if (
        audit.get("schema") != "exp453.lifecycle.failure"
        or audit.get("phase") != "evaluate"
        or audit.get("permanent_stop") is not True
        or audit.get("recovery_state") != "stopped"
        or audit.get("primary_error") != "ValueError: Exp448 target identity differs"
    ):
        raise ValueError("Exp453 failure audit facts differ")


def _no_between_captures_antigravity() -> None:
    return None


@dataclass(frozen=True, slots=True)
class _PredecessorDependenciesAntigravity:
    prediction_root: Path
    exp453_receipt_directory: Path
    exp453_failure_audit: Path
    exp453_evaluation_root: Path
    expected_prediction_aggregate_sha256: str
    expected_exp453_stopped_receipt_sha256: str
    expected_exp453_readiness_receipt_sha256: str
    expected_exp453_evaluate_receipt_sha256: str
    expected_exp453_audit_sha256: str
    capture_normalized_target: Callable[[], NormalizedTargetAuthorityAntigravity]
    between_captures: Callable[[], None] = _no_between_captures_antigravity

    def __post_init__(self) -> None:
        paths = (
            self.prediction_root,
            self.exp453_receipt_directory,
            self.exp453_failure_audit,
            self.exp453_evaluation_root,
        )
        hashes = (
            self.expected_prediction_aggregate_sha256,
            self.expected_exp453_stopped_receipt_sha256,
            self.expected_exp453_readiness_receipt_sha256,
            self.expected_exp453_evaluate_receipt_sha256,
            self.expected_exp453_audit_sha256,
        )
        if (
            any(not isinstance(path, Path) for path in paths)
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or not callable(self.capture_normalized_target)
            or not callable(self.between_captures)
        ):
            raise ValueError("predecessor dependencies are invalid")


def _production_dependencies_antigravity() -> _PredecessorDependenciesAntigravity:
    return _PredecessorDependenciesAntigravity(
        prediction_root=Path(EXP449_PREDICTION_ROOT_ANTIGRAVITY),
        exp453_receipt_directory=_EXP453_RECEIPT_DIRECTORY_ANTIGRAVITY,
        exp453_failure_audit=_EXP453_FAILURE_AUDIT_ANTIGRAVITY,
        exp453_evaluation_root=_EXP453_EVALUATION_ROOT_ANTIGRAVITY,
        expected_prediction_aggregate_sha256=EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
        expected_exp453_stopped_receipt_sha256=EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY,
        expected_exp453_readiness_receipt_sha256=EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY,
        expected_exp453_evaluate_receipt_sha256=EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY,
        expected_exp453_audit_sha256=EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY,
        capture_normalized_target=capture_normalized_target_authority_antigravity,
    )


@dataclass(frozen=True, slots=True)
class _DualSnapshotAntigravity:
    prediction: _PredictionSnapshotAntigravity
    receipt_identity: _IdentityAntigravity
    readiness_receipt_sha256: str
    evaluate_receipt_sha256: str
    stopped_receipt_sha256: str
    failure_audit_identity: _IdentityAntigravity
    failure_audit_sha256: str
    scientific_config_sha256: str
    normalized_target: NormalizedTargetAuthorityAntigravity


def _capture_once_antigravity(dependencies: _PredecessorDependenciesAntigravity) -> _DualSnapshotAntigravity:
    _require_absent_antigravity(dependencies.exp453_evaluation_root)
    receipts = _capture_exact_directory_antigravity(
        dependencies.exp453_receipt_directory,
        _EXP453_RECEIPT_INVENTORY_ANTIGRAVITY,
    )
    readiness_sha, evaluate_sha, stopped_sha = _validate_exp453_receipts_antigravity(
        receipts.payloads_antigravity(),
        dependencies.expected_exp453_readiness_receipt_sha256,
        dependencies.expected_exp453_evaluate_receipt_sha256,
        dependencies.expected_exp453_stopped_receipt_sha256,
    )
    audit = _capture_leaf_antigravity(dependencies.exp453_failure_audit)
    _validate_exp453_audit_antigravity(audit, dependencies.expected_exp453_audit_sha256)
    prediction = _capture_prediction_antigravity(
        dependencies.prediction_root,
        dependencies.expected_prediction_aggregate_sha256,
    )
    normalized = dependencies.capture_normalized_target()
    if type(normalized) is not NormalizedTargetAuthorityAntigravity:
        raise ValueError("Exp449 target authority differs")
    if (
        len(normalized.target_dataset_names) != 48
        or len(normalized.target_baseline_rows) != 48
        or type(normalized.target_baseline_rows[0]) is not DatasetBaselineAntigravity
    ):
        raise ValueError("Exp449 target inventory differs")
    _require_absent_antigravity(dependencies.exp453_evaluation_root)
    return _DualSnapshotAntigravity(
        prediction,
        receipts.identity,
        readiness_sha,
        evaluate_sha,
        stopped_sha,
        audit.identity,
        audit.sha256,
        _scientific_config_sha256_antigravity(),
        normalized,
    )


@dataclass(frozen=True, slots=True)
class _Exp454PredecessorAuthorityAntigravity:
    predecessor_attestation_sha256: str
    prediction_aggregate_sha256: str
    scientific_config_sha256: str
    exp453_stopped_receipt_sha256: str
    exp453_readiness_receipt_sha256: str
    exp453_evaluate_receipt_sha256: str
    exp453_failure_audit_sha256: str
    exp453_evaluation_root_absent: bool
    dataset_order: tuple[str, ...]
    canonical_prediction_rows: bytes
    normalized_target: NormalizedTargetAuthorityAntigravity
    dataset_count: int
    prediction_count: int
    label_session_count: int
    label_query_count: int
    label_row_count: int


_AUTHORITY_STATES_ANTIGRAVITY: dict[int, tuple[_Exp454PredecessorAuthorityAntigravity, str]] = {}


def _attestation_sha256_antigravity(snapshot: _DualSnapshotAntigravity) -> str:
    payload = {
        "prediction_aggregate_sha256": snapshot.prediction.aggregate_sha256,
        "prediction_root_identity": list(snapshot.prediction.root_identity),
        "exp453_stopped_receipt_sha256": snapshot.stopped_receipt_sha256,
        "exp453_readiness_receipt_sha256": snapshot.readiness_receipt_sha256,
        "exp453_evaluate_receipt_sha256": snapshot.evaluate_receipt_sha256,
        "exp453_failure_audit_sha256": snapshot.failure_audit_sha256,
        "scientific_config_sha256": snapshot.scientific_config_sha256,
        "dataset_order": list(snapshot.prediction.dataset_order),
        "normalized_target_count": 48,
        "label_session_count": 0,
        "label_query_count": 0,
        "label_row_count": 0,
    }
    return hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()


def _capture_predecessors_with_dependencies_antigravity(
    dependencies: _PredecessorDependenciesAntigravity,
) -> _Exp454PredecessorAuthorityAntigravity:
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
    authority = _Exp454PredecessorAuthorityAntigravity(
        predecessor_attestation_sha256=_attestation_sha256_antigravity(first),
        prediction_aggregate_sha256=first.prediction.aggregate_sha256,
        scientific_config_sha256=first.scientific_config_sha256,
        exp453_stopped_receipt_sha256=first.stopped_receipt_sha256,
        exp453_readiness_receipt_sha256=first.readiness_receipt_sha256,
        exp453_evaluate_receipt_sha256=first.evaluate_receipt_sha256,
        exp453_failure_audit_sha256=first.failure_audit_sha256,
        exp453_evaluation_root_absent=True,
        dataset_order=first.prediction.dataset_order,
        canonical_prediction_rows=first.prediction.canonical_rows,
        normalized_target=first.normalized_target,
        dataset_count=48,
        prediction_count=96,
        label_session_count=0,
        label_query_count=0,
        label_row_count=0,
    )
    _AUTHORITY_STATES_ANTIGRAVITY[id(authority)] = (authority, "issued")
    return authority


def capture_predecessors_antigravity() -> _Exp454PredecessorAuthorityAntigravity:
    """Authenticate Exp453 terminal evidence, Exp449 predictions, and one normalized target twice."""
    return _capture_predecessors_with_dependencies_antigravity(_production_dependencies_antigravity())


def _authority_entry_antigravity(authority: object) -> tuple[_Exp454PredecessorAuthorityAntigravity, str]:
    entry = _AUTHORITY_STATES_ANTIGRAVITY.get(id(authority))
    if (
        type(authority) is not _Exp454PredecessorAuthorityAntigravity
        or entry is None
        or entry[0] is not authority
        or authority.scientific_config_sha256 != _scientific_config_sha256_antigravity()
        or authority.dataset_count != 48
        or authority.prediction_count != 96
        or authority.exp453_evaluation_root_absent is not True
        or type(authority.normalized_target) is not NormalizedTargetAuthorityAntigravity
        or len(authority.normalized_target.target_dataset_names) != 48
        or authority.label_session_count != 0
        or authority.label_query_count != 0
        or authority.label_row_count != 0
    ):
        raise ValueError("Exp454 predecessor authority is invalid")
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
        raise ValueError("Exp454 predecessor authority label capability was already consumed")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, "opening")
    return exact.dataset_order


def _finish_label_capability_antigravity(authority: object, outcome: str) -> None:
    exact, state = _authority_entry_antigravity(authority)
    if state not in {"opening", "open"} or outcome not in {"open", "closed", "terminal", "close_failed", "setup_failed"}:
        raise ValueError("label capability transition is invalid")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, outcome)


__all__ = [
    "EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY",
    "EXP453_EVALUATE_RECEIPT_SHA256_ANTIGRAVITY",
    "EXP453_FAILURE_AUDIT_SHA256_ANTIGRAVITY",
    "EXP453_READINESS_RECEIPT_SHA256_ANTIGRAVITY",
    "EXP453_STOPPED_RECEIPT_SHA256_ANTIGRAVITY",
    "NormalizedTargetAuthorityAntigravity",
    "capture_normalized_target_authority_antigravity",
    "capture_predecessors_antigravity",
    "normalize_exp449_target_authority_antigravity",
    "validate_predecessors_antigravity",
]
