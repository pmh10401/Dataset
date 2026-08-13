"""Read-only, twice-captured authority for Exp449's immutable stopped run."""

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


_EXP449_RECEIPT_DIRECTORY_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp449_lifecycle_receipts_antigravity_20260812"
)
_EXP449_FAILURE_AUDIT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp449_failure_audit_antigravity.json"
)
_EXP449_FAILURE_AUDIT_SHA256_ANTIGRAVITY = (
    "bdbb09216fa1c23c63df6202b97e83867596c4da421a7f00687d70206bd4716e"
)
_EXP449_STAGING_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/.exp449_residual_fn_method_qualification_evaluation_"
    "antigravity_20260812.staging_dfeedd9bf543d1e03a0667c814318413_antigravity"
)
_EXP449_STAGING_INODE_ANTIGRAVITY = 21764083
_EXP449_STAGING_ARTIFACT_MAP_SHA256_ANTIGRAVITY = (
    "b7c19bd9fb74c99d8d6890a77e98a5cea89d74249f7a03d8851fd54cb177de7f"
)
_EXP449_STAGING_EVALUATION_SHA256_ANTIGRAVITY = (
    "fa1674f4a1d7fce88d0176008ac462f416dc6c737e2c1ce7424d9af42cd5fbbc"
)
_EXP449_STAGING_DASHBOARD_SHA256_ANTIGRAVITY = (
    "f115c6c69f76a5e48d0c6723f11b0eef82a40116846f0ede2b8d8090c49cd0d1"
)
_EXP449_PUBLISHED_EVALUATION_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_evaluation_antigravity_20260812"
)
_EXP449_RECEIPT_LEAVES_ANTIGRAVITY = (
    "readiness_complete_antigravity.json",
    "predict_started_antigravity.json",
    "predict_complete_antigravity.json",
    "evaluate_started_antigravity.json",
    "stopped_antigravity.json",
)
_EXP449_STAGING_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "compatibility_antigravity.json",
    "candidate_registry_antigravity.json",
    "frozen_predictions_antigravity.jsonl",
    "prediction_manifest_antigravity.json",
    "evaluation_antigravity.json",
    "candidate_metrics_antigravity.jsonl",
    "family_metrics_antigravity.jsonl",
    "bootstrap_antigravity.json",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_EXP449_FAILURE_TEXT_ANTIGRAVITY = "ValueError: frozen prediction indices differ from evidence"
_EXP449_RUN_IDENTITY_ANTIGRAVITY = "exp449-residual-fn-method-qualification-antigravity-20260812"
_EXP449_RECEIPT_SCHEMA_ANTIGRAVITY = "exp449_lifecycle_receipt"
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
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
        raise ValueError("published Exp449 evaluation root absence is unverifiable") from error
    raise ValueError("published Exp449 evaluation root must remain absent")


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


@dataclass(frozen=True, slots=True)
class _PredictionSnapshotAntigravity:
    root_identity: _IdentityAntigravity
    lifecycle_root_identity: str
    reviewed_commit: str
    aggregate_sha256: str
    canonical_rows: bytes
    canonical_row_sha256: tuple[str, ...]
    canonical_aggregate_sha256: str
    context_sha256: str
    source_identity_sha256: str
    manifest_sha256: str
    dataset_order: tuple[str, ...]


def _capture_prediction_antigravity(
    root: Path,
    expected_aggregate_sha256: str,
) -> _PredictionSnapshotAntigravity:
    """Retain an outer root descriptor around Exp449's own exact descriptor capture."""
    from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (
        capture_frozen_prediction_authority_antigravity,
    )
    from experiments.exp449_residual_fn_method_qualification_antigravity.lifecycle_antigravity import (
        _root_identity_antigravity,
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
        lifecycle_root_identity = _root_identity_antigravity(root)
        if (
            authority.aggregate_sha256 != expected_aggregate_sha256
            or authority.family_method_pairs != 16
            or authority.dataset_units != 96
            or authority.label_access_count != 0
            or type(capture.canonical_rows) is not bytes
            or type(lifecycle_root_identity) is not str
            or len(lifecycle_root_identity) != 64
            or any(character not in "0123456789abcdef" for character in lifecycle_root_identity)
        ):
            raise ValueError("Exp449 prediction authority differs")
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
            lifecycle_root_identity,
            authority.reviewed_commit,
            authority.aggregate_sha256,
            capture.canonical_rows,
            capture.canonical_row_sha256,
            capture.canonical_aggregate_sha256,
            capture.context_sha256,
            capture.source_identity_sha256,
            capture.manifest_sha256,
            dataset_order,
        )
    except OSError as error:
        raise ValueError("Exp449 prediction root is missing or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_object_antigravity(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = strict_json_bytes_antigravity(payload)
    except ValueError as error:
        raise ValueError(f"Exp449 {name} is not strict JSON") from error
    if type(value) is not dict or canonical_json_bytes_antigravity(value) != payload:
        raise ValueError(f"Exp449 {name} is not a canonical JSON object")
    return value


def _validate_receipts_antigravity(
    captured: _CapturedDirectoryAntigravity,
    expected_prediction_identity: str,
) -> str:
    expected_events = (
        ("readiness", "complete", False),
        ("predict", "started", False),
        ("predict", "complete", False),
        ("evaluate", "started", False),
        ("evaluate", "stopped", True),
    )
    previous: str | None = None
    reviewed_commit: str | None = None
    prediction_identity: str | None = None
    for ordinal, (name, payload) in enumerate(captured.leaves):
        receipt = _strict_object_antigravity(payload, name)
        fields = {
            "schema", "experiment_id", "run_identity", "reviewed_commit", "phase", "event",
            "ordinal", "previous_leaf_sha256", "prediction_root_identity",
            "evaluation_root_identity", "permanent_stop",
        }
        phase, event, stopped = expected_events[ordinal]
        commit = receipt.get("reviewed_commit")
        if (
            set(receipt) != fields
            or receipt.get("schema") != _EXP449_RECEIPT_SCHEMA_ANTIGRAVITY
            or receipt.get("experiment_id") != "Exp449"
            or receipt.get("run_identity") != _EXP449_RUN_IDENTITY_ANTIGRAVITY
            or type(commit) is not str
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or receipt.get("phase") != phase
            or receipt.get("event") != event
            or receipt.get("ordinal") != ordinal
            or receipt.get("previous_leaf_sha256") != previous
            or receipt.get("evaluation_root_identity") is not None
            or receipt.get("permanent_stop") is not stopped
        ):
            raise ValueError("Exp449 receipt chain differs")
        if reviewed_commit is None:
            reviewed_commit = commit
        elif commit != reviewed_commit:
            raise ValueError("Exp449 receipt reviewed commit differs")
        bound = receipt.get("prediction_root_identity")
        if ordinal < 2 or ordinal == 4:
            if bound is not None:
                raise ValueError("Exp449 receipt prediction binding differs")
        else:
            if (
                type(bound) is not str
                or bound != expected_prediction_identity
                or len(bound) != 64
                or any(character not in "0123456789abcdef" for character in bound)
            ):
                raise ValueError("Exp449 receipt prediction binding differs")
            if prediction_identity is None:
                prediction_identity = bound
            elif bound != prediction_identity:
                raise ValueError("Exp449 receipt prediction binding changed")
        previous = hashlib.sha256(payload).hexdigest()
    if reviewed_commit is None:
        raise ValueError("Exp449 receipt chain is empty")
    return reviewed_commit


def _validate_failure_audit_antigravity(
    captured: _CapturedLeafAntigravity,
    expected_sha256: str,
    expected_root_destinations: tuple[str, ...],
) -> None:
    if captured.sha256 != expected_sha256:
        raise ValueError("Exp449 failure audit digest differs")
    audit = _strict_object_antigravity(captured.payload, "failure audit")
    expected = {
        "phase": "evaluate",
        "root_destinations": list(expected_root_destinations),
        "recovery_state": "stopped",
        "primary_error": _EXP449_FAILURE_TEXT_ANTIGRAVITY,
        "cleanup_notes": [],
        "permanent_stop": True,
    }
    if audit != expected:
        raise ValueError("Exp449 failure audit facts differ")


def _validate_staging_antigravity(
    captured: _CapturedDirectoryAntigravity,
    prediction: _PredictionSnapshotAntigravity,
    dependencies: "_PredecessorDependenciesAntigravity",
) -> tuple[int, int]:
    payloads = captured.payloads_antigravity()
    digests = dict(captured.digests)
    if (
        digests["artifact_sha256_antigravity.json"] != dependencies.expected_staging_artifact_map_sha256
        or digests["evaluation_antigravity.json"] != dependencies.expected_staging_evaluation_sha256
        or digests["dashboard_snapshot_antigravity.json"] != dependencies.expected_staging_dashboard_snapshot_sha256
    ):
        raise ValueError("Exp449 staging frozen leaf digest differs")
    artifact_map = _strict_object_antigravity(payloads["artifact_sha256_antigravity.json"], "staging artifact map")
    expected_names = set(_EXP449_STAGING_LEAVES_ANTIGRAVITY[:-1])
    if set(artifact_map) != expected_names or any(artifact_map.get(name) != digests[name] for name in expected_names):
        raise ValueError("Exp449 staging artifact map differs")
    frozen = payloads["frozen_predictions_antigravity.jsonl"]
    if frozen != prediction.canonical_rows:
        raise ValueError("Exp449 staging frozen predictions differ")
    frozen_rows = tuple(strict_json_bytes_antigravity(line + b"\n") for line in frozen.splitlines())
    evaluation = _strict_object_antigravity(payloads["evaluation_antigravity.json"], "staging evaluation")
    fields = {
        "schema", "reviewed_commit", "candidate_count", "qualified_count", "label_session_count",
        "label_query_count", "label_row_count", "routing_research_eligible", "dataset_evidence",
    }
    evidence = evaluation.get("dataset_evidence")
    if (
        set(evaluation) != fields
        or evaluation.get("schema") != "exp449"
        or evaluation.get("reviewed_commit") != prediction.reviewed_commit
        or evaluation.get("candidate_count") != 16
        or evaluation.get("label_session_count") != 1
        or evaluation.get("label_query_count") != 48
        or evaluation.get("label_row_count") != 72300
        or type(evidence) is not list
        or len(evidence) != 48
    ):
        raise ValueError("Exp449 staging evaluation facts differ")
    by_dataset: dict[str, dict[str, object]] = {}
    for expected_dataset, row in zip(prediction.dataset_order, evidence, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"dataset", "family", "target_length", "baseline_confusion", "candidates"}
            or row.get("dataset") != expected_dataset
            or type(row.get("family")) is not str
            or type(row.get("target_length")) is not int
            or type(row.get("candidates")) is not dict
            or expected_dataset in by_dataset
        ):
            raise ValueError("Exp449 staging dataset evidence differs")
        by_dataset[expected_dataset] = row
    mismatches = equal_sets = 0
    for frozen_row in frozen_rows:
        if type(frozen_row) is not dict:
            raise ValueError("Exp449 staging frozen prediction row differs")
        dataset = frozen_row.get("dataset")
        method = frozen_row.get("method")
        evidence_row = by_dataset.get(dataset)
        candidate = evidence_row.get("candidates", {}).get(method) if evidence_row is not None else None
        frozen_indices = frozen_row.get("indices")
        evidence_indices = candidate.get("indices") if type(candidate) is dict else None
        if (
            type(frozen_indices) is not list
            or type(evidence_indices) is not list
            or any(type(index) is not int for index in evidence_indices)
            or len(evidence_indices) != len(frozen_indices)
        ):
            raise ValueError("Exp449 staging candidate evidence differs")
        if evidence_indices != frozen_indices:
            mismatches += 1
            if set(evidence_indices) == set(frozen_indices):
                equal_sets += 1
    if (mismatches, equal_sets) != (75, 75):
        raise ValueError("Exp449 order-only mismatch facts differ")
    return mismatches, equal_sets


def _no_between_captures_antigravity() -> None:
    return None


@dataclass(frozen=True, slots=True)
class _PredecessorDependenciesAntigravity:
    prediction_root: Path
    receipt_directory: Path
    failure_audit: Path
    staging_root: Path
    published_evaluation_root: Path
    expected_prediction_aggregate_sha256: str
    expected_failure_audit_sha256: str
    expected_staging_inode: int
    expected_staging_artifact_map_sha256: str
    expected_staging_evaluation_sha256: str
    expected_staging_dashboard_snapshot_sha256: str
    expected_failure_root_destinations: tuple[str, ...]
    between_captures: Callable[[], None] = _no_between_captures_antigravity

    def __post_init__(self) -> None:
        paths = (
            self.prediction_root,
            self.receipt_directory,
            self.failure_audit,
            self.staging_root,
            self.published_evaluation_root,
        )
        hashes = (
            self.expected_prediction_aggregate_sha256,
            self.expected_failure_audit_sha256,
            self.expected_staging_artifact_map_sha256,
            self.expected_staging_evaluation_sha256,
            self.expected_staging_dashboard_snapshot_sha256,
        )
        if (
            any(not isinstance(path, Path) for path in paths)
            or any(type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in hashes)
            or type(self.expected_staging_inode) is not int
            or self.expected_staging_inode < 1
            or type(self.expected_failure_root_destinations) is not tuple
            or any(type(value) is not str for value in self.expected_failure_root_destinations)
            or not callable(self.between_captures)
        ):
            raise ValueError("predecessor dependencies are invalid")


def _production_dependencies_antigravity() -> _PredecessorDependenciesAntigravity:
    prediction = Path(EXP449_PREDICTION_ROOT_ANTIGRAVITY)
    return _PredecessorDependenciesAntigravity(
        prediction_root=prediction,
        receipt_directory=_EXP449_RECEIPT_DIRECTORY_ANTIGRAVITY,
        failure_audit=_EXP449_FAILURE_AUDIT_ANTIGRAVITY,
        staging_root=_EXP449_STAGING_ROOT_ANTIGRAVITY,
        published_evaluation_root=_EXP449_PUBLISHED_EVALUATION_ROOT_ANTIGRAVITY,
        expected_prediction_aggregate_sha256=EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
        expected_failure_audit_sha256=_EXP449_FAILURE_AUDIT_SHA256_ANTIGRAVITY,
        expected_staging_inode=_EXP449_STAGING_INODE_ANTIGRAVITY,
        expected_staging_artifact_map_sha256=_EXP449_STAGING_ARTIFACT_MAP_SHA256_ANTIGRAVITY,
        expected_staging_evaluation_sha256=_EXP449_STAGING_EVALUATION_SHA256_ANTIGRAVITY,
        expected_staging_dashboard_snapshot_sha256=_EXP449_STAGING_DASHBOARD_SHA256_ANTIGRAVITY,
        expected_failure_root_destinations=(
            str(prediction),
            str(_EXP449_PUBLISHED_EVALUATION_ROOT_ANTIGRAVITY),
        ),
    )


@dataclass(frozen=True, slots=True)
class _StoppedSnapshotAntigravity:
    prediction: _PredictionSnapshotAntigravity
    receipt_identity: _IdentityAntigravity
    receipt_sha256: tuple[tuple[str, str], ...]
    failure_audit_identity: _IdentityAntigravity
    failure_audit_sha256: str
    staging_identity: _IdentityAntigravity
    staging_sha256: tuple[tuple[str, str], ...]
    reviewed_commit: str
    order_mismatch_count: int
    order_equal_set_count: int
    scientific_config_sha256: str


def _capture_once_antigravity(dependencies: _PredecessorDependenciesAntigravity) -> _StoppedSnapshotAntigravity:
    _require_absent_antigravity(dependencies.published_evaluation_root)
    prediction = _capture_prediction_antigravity(
        dependencies.prediction_root,
        dependencies.expected_prediction_aggregate_sha256,
    )
    receipts = _capture_exact_directory_antigravity(
        dependencies.receipt_directory,
        _EXP449_RECEIPT_LEAVES_ANTIGRAVITY,
    )
    reviewed_commit = _validate_receipts_antigravity(
        receipts,
        prediction.lifecycle_root_identity,
    )
    if reviewed_commit != prediction.reviewed_commit:
        raise ValueError("Exp449 prediction and receipt commits differ")
    audit = _capture_leaf_antigravity(dependencies.failure_audit)
    _validate_failure_audit_antigravity(
        audit,
        dependencies.expected_failure_audit_sha256,
        dependencies.expected_failure_root_destinations,
    )
    staging = _capture_exact_directory_antigravity(
        dependencies.staging_root,
        _EXP449_STAGING_LEAVES_ANTIGRAVITY,
        expected_inode=dependencies.expected_staging_inode,
    )
    mismatches, equal_sets = _validate_staging_antigravity(staging, prediction, dependencies)
    _require_absent_antigravity(dependencies.published_evaluation_root)
    return _StoppedSnapshotAntigravity(
        prediction,
        receipts.identity,
        receipts.digests,
        audit.identity,
        audit.sha256,
        staging.identity,
        staging.digests,
        reviewed_commit,
        mismatches,
        equal_sets,
        _scientific_config_sha256_antigravity(),
    )


@dataclass(frozen=True, slots=True)
class _Exp449StoppedAuthorityAntigravity:
    predecessor_attestation_sha256: str
    prediction_aggregate_sha256: str
    scientific_config_sha256: str
    reviewed_commit: str
    dataset_order: tuple[str, ...]
    canonical_prediction_rows: bytes
    dataset_count: int
    prediction_count: int
    label_session_count: int
    label_query_count: int
    label_row_count: int
    order_mismatch_count: int
    order_equal_set_count: int


_AUTHORITY_STATES_ANTIGRAVITY: dict[int, tuple[_Exp449StoppedAuthorityAntigravity, str]] = {}


def _attestation_sha256_antigravity(snapshot: _StoppedSnapshotAntigravity) -> str:
    payload = {
        "prediction_aggregate_sha256": snapshot.prediction.aggregate_sha256,
        "prediction_root_identity": list(snapshot.prediction.root_identity),
        "receipt_sha256": dict(snapshot.receipt_sha256),
        "failure_audit_sha256": snapshot.failure_audit_sha256,
        "staging_sha256": dict(snapshot.staging_sha256),
        "reviewed_commit": snapshot.reviewed_commit,
        "scientific_config_sha256": snapshot.scientific_config_sha256,
        "dataset_order": list(snapshot.prediction.dataset_order),
        "order_mismatch_count": snapshot.order_mismatch_count,
        "order_equal_set_count": snapshot.order_equal_set_count,
        "label_session_count": 1,
        "label_query_count": 48,
        "label_row_count": 72300,
    }
    return hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()


def _capture_exp449_stopped_authority_with_dependencies_antigravity(
    dependencies: _PredecessorDependenciesAntigravity,
) -> _Exp449StoppedAuthorityAntigravity:
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
    authority = _Exp449StoppedAuthorityAntigravity(
        predecessor_attestation_sha256=_attestation_sha256_antigravity(first),
        prediction_aggregate_sha256=first.prediction.aggregate_sha256,
        scientific_config_sha256=first.scientific_config_sha256,
        reviewed_commit=first.reviewed_commit,
        dataset_order=first.prediction.dataset_order,
        canonical_prediction_rows=first.prediction.canonical_rows,
        dataset_count=48,
        prediction_count=96,
        label_session_count=1,
        label_query_count=48,
        label_row_count=72300,
        order_mismatch_count=first.order_mismatch_count,
        order_equal_set_count=first.order_equal_set_count,
    )
    _AUTHORITY_STATES_ANTIGRAVITY[id(authority)] = (authority, "issued")
    return authority


def capture_exp449_stopped_authority_antigravity() -> _Exp449StoppedAuthorityAntigravity:
    """Authenticate the fixed stopped Exp449 predecessor twice without label capability."""
    return _capture_exp449_stopped_authority_with_dependencies_antigravity(
        _production_dependencies_antigravity()
    )


def _authority_entry_antigravity(authority: object) -> tuple[_Exp449StoppedAuthorityAntigravity, str]:
    entry = _AUTHORITY_STATES_ANTIGRAVITY.get(id(authority))
    if (
        type(authority) is not _Exp449StoppedAuthorityAntigravity
        or entry is None
        or entry[0] is not authority
        or authority.scientific_config_sha256 != _scientific_config_sha256_antigravity()
        or authority.dataset_count != 48
        or authority.prediction_count != 96
        or authority.label_session_count != 1
        or authority.label_query_count != 48
        or authority.label_row_count != 72300
        or authority.order_mismatch_count != 75
        or authority.order_equal_set_count != 75
    ):
        raise ValueError("Exp449 stopped authority is invalid")
    return entry


def validate_exp449_stopped_authority_antigravity(authority: object) -> tuple[str, ...]:
    """Validate only an exact object issued after two full matching captures."""
    exact, _state = _authority_entry_antigravity(authority)
    return exact.dataset_order


def canonical_prediction_rows_antigravity(authority: object) -> bytes:
    exact, _state = _authority_entry_antigravity(authority)
    return exact.canonical_prediction_rows


def _claim_label_capability_antigravity(authority: object) -> tuple[str, ...]:
    exact, state = _authority_entry_antigravity(authority)
    if state != "issued":
        raise ValueError("Exp449 stopped authority label capability was already consumed")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, "opening")
    return exact.dataset_order


def _finish_label_capability_antigravity(authority: object, outcome: str) -> None:
    exact, state = _authority_entry_antigravity(authority)
    if state not in {"opening", "open"} or outcome not in {"open", "closed", "terminal", "close_failed", "setup_failed"}:
        raise ValueError("label capability transition is invalid")
    _AUTHORITY_STATES_ANTIGRAVITY[id(exact)] = (exact, outcome)


__all__ = [
    "canonical_prediction_rows_antigravity",
    "capture_exp449_stopped_authority_antigravity",
    "validate_exp449_stopped_authority_antigravity",
]
