"""Independent replay and integrity checks for Exp468 Task 5 publications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .artifacts_antigravity import (
    LEAF_NAMES_ANTIGRAVITY,
    _HASHED_LEAF_NAMES_ANTIGRAVITY,
    _context_payload_antigravity,
    _dashboard_snapshot_payload_antigravity,
    _decision_payload_antigravity,
    _failed_folds_payload_antigravity,
    _fold_metrics_payload_antigravity,
    _method_summary_payload_antigravity,
    _parse_json_bytes_antigravity,
    _parse_jsonl_bytes_antigravity,
    _record_from_payload_antigravity,
)
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import qualify_antigravity
from .predecessor_antigravity import capture_exp465_authority_antigravity
from .contracts_antigravity import Exp468QualificationAntigravity
from .scoring_antigravity import FoldMethodScoresAntigravity, score_fold_methods_antigravity


def _arrays_equal_antigravity(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _records_equal_antigravity(
    stored: FoldMethodScoresAntigravity,
    replayed: FoldMethodScoresAntigravity,
) -> None:
    """Require byte-exact equality for every retained Task 4 score field."""

    if (
        stored.seed != replayed.seed
        or stored.fold_index != replayed.fold_index
        or stored.methods != replayed.methods
        or stored.backend != replayed.backend
        or stored.fit_normal_max_raw_js != replayed.fit_normal_max_raw_js
    ):
        raise ValueError("stored record scalar provenance differs from exact scorer replay")
    array_fields = (
        "fit_raw_js",
        "input_train140",
        "input_fit_indices",
        "input_query_indices",
    )
    for field_name in array_fields:
        if not _arrays_equal_antigravity(
            getattr(stored, field_name),
            getattr(replayed, field_name),
        ):
            raise ValueError(f"stored {field_name} differs from exact scorer replay")
    for method in stored.methods:
        if not _arrays_equal_antigravity(stored.original[method], replayed.original[method]):
            raise ValueError(f"stored original[{method}] differs from exact scorer replay")
        for change_type in replayed.primary[method]:
            if not _arrays_equal_antigravity(
                stored.primary[method][change_type],
                replayed.primary[method][change_type],
            ):
                raise ValueError(
                    f"stored primary[{method}][{change_type}] differs from exact scorer replay",
                )
    for scope in replayed.query_raw_js:
        if not _arrays_equal_antigravity(
            stored.query_raw_js[scope],
            replayed.query_raw_js[scope],
        ):
            raise ValueError(f"stored query_raw_js[{scope}] differs from exact scorer replay")
        if not _arrays_equal_antigravity(
            stored.confirmed[scope],
            replayed.confirmed[scope],
        ):
            raise ValueError(f"stored confirmed[{scope}] differs from exact scorer replay")


def _replay_records_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
) -> Mapping[tuple[int, int], FoldMethodScoresAntigravity]:
    """Re-score each stored raw input without loading ECG200 or trusting scalars."""

    authority = capture_exp465_authority_antigravity()
    replayed: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for identity, stored in records.items():
        candidate = score_fold_methods_antigravity(
            stored.input_train140,
            stored.input_fit_indices,
            stored.input_query_indices,
            seed=stored.seed,
            fold_index=stored.fold_index,
            authority=authority,
        )
        _records_equal_antigravity(stored, candidate)
        replayed[identity] = candidate
    return MappingProxyType(replayed)


def _read_leaf_set_antigravity(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise ValueError("publication root must be an existing directory")
    actual = tuple(sorted(path.name for path in root.iterdir()))
    expected = tuple(sorted(LEAF_NAMES_ANTIGRAVITY))
    if actual != expected:
        raise ValueError("publication root must contain exactly seven leaves")


def _read_and_verify_hashes_antigravity(root: Path) -> dict[str, str]:
    hash_path = root / "artifact_sha256_antigravity.json"
    hashes = _parse_json_bytes_antigravity(
        hash_path.read_bytes(),
        name=hash_path.name,
    )
    if type(hashes) is not dict or set(hashes) != set(_HASHED_LEAF_NAMES_ANTIGRAVITY):
        raise ValueError("artifact digest map must contain exactly the six non-self leaves")
    for name in _HASHED_LEAF_NAMES_ANTIGRAVITY:
        digest = hashes[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("artifact digest map contains an invalid SHA-256")
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"stale artifact digest for {name}")
    return hashes


def _records_from_context_antigravity(
    context: Mapping[str, object],
) -> Mapping[tuple[int, int], FoldMethodScoresAntigravity]:
    raw_records = context.get("records")
    if type(raw_records) is not list or len(raw_records) != 10:
        raise ValueError("context must contain exactly ten records")
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for raw_record in raw_records:
        record = _record_from_payload_antigravity(raw_record)
        identity = (record.seed, record.fold_index)
        if identity in records:
            raise ValueError("context contains duplicate seed/fold record")
        records[identity] = record
    return MappingProxyType(records)


@dataclass(frozen=True, slots=True)
class InspectedResultAntigravity:
    root: Path
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity]
    qualification: Exp468QualificationAntigravity
    decision: QualificationDecisionAntigravity
    context: Mapping[str, object]
    dashboard_snapshot: Mapping[str, object]


def inspect_result_antigravity(root: str | Path) -> InspectedResultAntigravity:
    """Read a seven-leaf root and independently recompute every Task 4 scalar."""

    target = Path(root)
    _read_leaf_set_antigravity(target)
    hashes = _read_and_verify_hashes_antigravity(target)

    context_path = target / "context_antigravity.json"
    context = _parse_json_bytes_antigravity(
        context_path.read_bytes(),
        name=context_path.name,
    )
    if type(context) is not dict:
        raise ValueError("context must be a JSON object")
    records = _records_from_context_antigravity(context)

    # First compare every projection with the records that were actually read.
    # This rejects rehashed scalar/dashboard tampering before the more expensive
    # source-free scorer replay.  The scorer replay below remains mandatory for
    # an otherwise internally consistent publication.
    stored_qualification = qualify_antigravity(records)
    stored_decision_value = evaluate_qualification_antigravity(stored_qualification)
    expected_context = _context_payload_antigravity(
        records,
        stored_qualification,
        stored_decision_value,
    )
    if context != expected_context:
        raise ValueError("context differs from the exact retained record evidence")

    decision_path = target / "decision_antigravity.json"
    stored_decision = _parse_json_bytes_antigravity(
        decision_path.read_bytes(),
        name=decision_path.name,
    )
    if stored_decision != _decision_payload_antigravity(stored_decision_value):
        raise ValueError("decision differs from the exact scalar replay")

    dashboard_path = target / "dashboard_snapshot_antigravity.json"
    dashboard = _parse_json_bytes_antigravity(
        dashboard_path.read_bytes(),
        name=dashboard_path.name,
    )
    expected_dashboard = _dashboard_snapshot_payload_antigravity(
        stored_qualification,
        stored_decision_value,
    )
    if dashboard != expected_dashboard:
        raise ValueError("dashboard snapshot differs from the exact scalar replay")

    metrics_path = target / "method_fold_metrics_antigravity.jsonl"
    metrics = _parse_jsonl_bytes_antigravity(
        metrics_path.read_bytes(),
        name=metrics_path.name,
    )
    expected_metrics = _fold_metrics_payload_antigravity(records, stored_qualification)
    if metrics != expected_metrics:
        raise ValueError("method fold metrics differ from the exact raw-input replay")

    summary_path = target / "method_summary_antigravity.jsonl"
    summary = _parse_jsonl_bytes_antigravity(
        summary_path.read_bytes(),
        name=summary_path.name,
    )
    expected_summary = _method_summary_payload_antigravity(expected_metrics)
    if summary != expected_summary:
        raise ValueError("method summary differs from the exact scalar replay")

    failed_path = target / "failed_folds_antigravity.jsonl"
    failed = _parse_jsonl_bytes_antigravity(
        failed_path.read_bytes(),
        name=failed_path.name,
    )
    expected_failed = _failed_folds_payload_antigravity(stored_qualification)
    if failed != expected_failed:
        raise ValueError("failed-fold evidence differs from the exact scalar replay")

    # This is the authoritative source-free replay.  It never calls the ECG200
    # loader and it does not trust any stored scalar, AUC, or decision value.
    replayed_records = _replay_records_antigravity(records)
    qualification = qualify_antigravity(replayed_records)
    decision = evaluate_qualification_antigravity(qualification)
    if qualification != stored_qualification:
        raise ValueError("qualification differs from the exact raw-input scorer replay")
    if decision != stored_decision_value:
        raise ValueError("decision differs from the exact raw-input scorer replay")
    if context != _context_payload_antigravity(replayed_records, qualification, decision):
        raise ValueError("context differs from the exact raw-input scorer replay")

    # Keep the digest map in the return value only as a validation witness; the
    # map is intentionally not a ninth payload field or a scientific result.
    _ = hashes
    return InspectedResultAntigravity(
        root=target,
        records=replayed_records,
        qualification=qualification,
        decision=decision,
        context=MappingProxyType(context),
        dashboard_snapshot=MappingProxyType(dashboard),
    )


__all__ = (
    "InspectedResultAntigravity",
    "inspect_result_antigravity",
)
