"""Canonical seven-leaf publication for Exp466 Task 5."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp466FoldResultAntigravity,
    Exp466QualificationAntigravity,
)
from .dashboard_export_antigravity import (
    _decision_payload_antigravity,
    _expected_decision_payload_antigravity,
    export_dashboard_antigravity,
)
from .evaluation_antigravity import QualificationDecisionAntigravity
from .scoring_antigravity import FoldMethodScoresAntigravity


_METHODS_ANTIGRAVITY = ("C0", "P9", "C9")
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
_LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
_FLOAT_TOLERANCE_ANTIGRAVITY = 1e-12


@dataclass(frozen=True, slots=True)
class PublishedResultAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def canonical_json_bytes_antigravity(value: Any) -> bytes:
    """Encode one canonical UTF-8 JSON value with one terminal newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _jsonl_bytes_antigravity(rows: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_bytes_antigravity(row) for row in rows)


def _float64_blob_antigravity(values: np.ndarray) -> dict[str, object]:
    little_endian = np.asarray(values, dtype=np.dtype("<f8"), order="C")
    payload = little_endian.tobytes(order="C")
    return {
        "encoding": "base64_float64_le_c_v1",
        "shape": list(little_endian.shape),
        "sha256": sha256(payload).hexdigest(),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _auc_antigravity(normal: np.ndarray, anomaly: np.ndarray) -> float:
    wins = int(np.count_nonzero(anomaly[:, None] > normal[None, :]))
    ties = int(np.count_nonzero(anomaly[:, None] == normal[None, :]))
    return float((wins + 0.5 * ties) / (normal.size * anomaly.size))


def _method_metrics_antigravity(
    record: FoldMethodScoresAntigravity,
    method: str,
) -> tuple[tuple[float, float, float, float], float, bool]:
    arrays = (
        record.original[method],
        *(record.primary[method][name] for name in _CHANGE_TYPES_ANTIGRAVITY),
    )
    finite = bool(all(np.isfinite(values).all() for values in arrays))
    if not finite:
        return (0.0, 0.0, 0.0, 0.0), 0.0, False
    per_type = tuple(
        _auc_antigravity(record.original[method], record.primary[method][name])
        for name in _CHANGE_TYPES_ANTIGRAVITY
    )
    return per_type, float(fsum(per_type) / len(per_type)), True


def _failure_reasons_antigravity(
    *,
    per_type_auc: tuple[float, float, float, float],
    mean_auc: float,
    all_scores_finite: bool,
    c0_gradual_auc: float,
    c0_period_auc: float,
    c9_gradual_auc: float,
    c9_period_auc: float,
    period_confirmations: int,
    normal_confirmations: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not all_scores_finite:
        reasons.append("nonfinite_scores")
    for name, auc in zip(_CHANGE_TYPES_ANTIGRAVITY, per_type_auc, strict=True):
        if auc < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"{name}_auc_below_0.85")
    if mean_auc < FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min:
        reasons.append("mean_auc_below_0.90")
    if c9_period_auc <= c0_period_auc:
        reasons.append("period_auc_not_strictly_above_c0")
    if c9_gradual_auc < c0_gradual_auc:
        reasons.append("gradual_auc_below_c0")
    if period_confirmations == 0:
        reasons.append("period_confirmation_missing")
    if normal_confirmations != 0:
        reasons.append("normal_confirmation_present")
    return tuple(reasons)


def _qualification_row_payload_antigravity(
    row: Exp466FoldResultAntigravity,
) -> dict[str, object]:
    return {
        "seed": row.seed,
        "fold_index": row.fold_index,
        "per_type_auc": list(row.per_type_auc),
        "mean_auc": row.mean_auc,
        "c0_gradual_auc": row.c0_gradual_auc,
        "c0_period_auc": row.c0_period_auc,
        "c9_gradual_auc": row.c9_gradual_auc,
        "c9_period_auc": row.c9_period_auc,
        "period_confirmations": row.period_confirmations,
        "all_scores_finite": row.all_scores_finite,
        "passed": row.passed,
        "failure_reasons": list(row.failure_reasons),
    }


def _derived_qualification_payload_antigravity(
    record: FoldMethodScoresAntigravity,
) -> dict[str, object]:
    c0_auc, _c0_mean, _c0_finite = _method_metrics_antigravity(record, "C0")
    c9_auc, c9_mean, c9_finite = _method_metrics_antigravity(record, "C9")
    period_confirmations = int(np.count_nonzero(record.confirmed["period_change"]))
    normal_confirmations = int(np.count_nonzero(record.confirmed["normal"]))
    reasons = _failure_reasons_antigravity(
        per_type_auc=c9_auc,
        mean_auc=c9_mean,
        all_scores_finite=c9_finite,
        c0_gradual_auc=c0_auc[1],
        c0_period_auc=c0_auc[3],
        c9_gradual_auc=c9_auc[1],
        c9_period_auc=c9_auc[3],
        period_confirmations=period_confirmations,
        normal_confirmations=normal_confirmations,
    )
    return {
        "seed": record.seed,
        "fold_index": record.fold_index,
        "per_type_auc": list(c9_auc),
        "mean_auc": c9_mean,
        "c0_gradual_auc": c0_auc[1],
        "c0_period_auc": c0_auc[3],
        "c9_gradual_auc": c9_auc[1],
        "c9_period_auc": c9_auc[3],
        "period_confirmations": period_confirmations,
        "all_scores_finite": c9_finite,
        "passed": not reasons,
        "failure_reasons": list(reasons),
    }


def _payloads_close_antigravity(
    left: object,
    right: object,
) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return abs(left - right) <= _FLOAT_TOLERANCE_ANTIGRAVITY
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _payloads_close_antigravity(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return tuple(left) == tuple(right) and all(
            _payloads_close_antigravity(left[key], right[key]) for key in left
        )
    return left == right


def _validate_record_evidence_antigravity(
    record: FoldMethodScoresAntigravity,
) -> None:
    if record.fit_normal_max_margin != float(np.max(record.fit_margins)):
        raise ValueError("fit maximum differs from retained margins")
    for scope in _SCOPES_ANTIGRAVITY:
        margins = record.query_margins[scope]
        expected_rank = np.asarray(
            (
                1.0
                + np.count_nonzero(
                    record.fit_margins[:, None] <= margins[None, :],
                    axis=0,
                )
            )
            / (len(record.fit_margins) + 1.0),
            dtype=np.float64,
        )
        p9 = (
            record.original["P9"]
            if scope == "normal"
            else record.primary["P9"][scope]
        )
        if not np.array_equal(p9, expected_rank):
            raise ValueError("P9 rank differs from retained margins")
        expected_confirmation = (margins > 0.0) & (
            margins > record.fit_normal_max_margin
        )
        if not np.array_equal(record.confirmed[scope], expected_confirmation):
            raise ValueError("confirmation differs from strict retained margin rule")
        c0 = (
            record.original["C0"]
            if scope == "normal"
            else record.primary["C0"][scope]
        )
        c9 = (
            record.original["C9"]
            if scope == "normal"
            else record.primary["C9"][scope]
        )
        expected_c9 = np.where(expected_confirmation, np.maximum(c0, p9), c0)
        if not np.array_equal(c9, expected_c9):
            raise ValueError("C9 differs from retained C0/P9 confirmation evidence")


def _normalise_inputs_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: Exp466QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> tuple[tuple[tuple[int, int], FoldMethodScoresAntigravity], ...]:
    if not isinstance(records, Mapping):
        raise ValueError("records must be a mapping")
    expected_keys = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if tuple(records) != expected_keys:
        raise ValueError("records must preserve exact frozen seed-major order")
    if type(qualification) is not Exp466QualificationAntigravity:
        raise ValueError("publication requires exact Exp466 qualification")
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("publication requires exact Exp466 decision")

    items = tuple(records.items())
    first_train: bytes | None = None
    for index, (key, record) in enumerate(items):
        if type(record) is not FoldMethodScoresAntigravity:
            raise ValueError("records must contain exact Task 4 fold records")
        if key != (record.seed, record.fold_index):
            raise ValueError("record identity differs from its mapping key")
        if record.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("record methods differ from frozen C0/P9/C9 order")
        train_bytes = record.input_train140.tobytes(order="C")
        if first_train is None:
            first_train = train_bytes
        elif train_bytes != first_train:
            raise ValueError("records retain different train140 inputs")
        fit = set(int(value) for value in record.input_fit_indices)
        query = set(int(value) for value in record.input_query_indices)
        expected_indices = set(range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows))
        if fit & query or fit | query != expected_indices:
            raise ValueError("record fit/query indices must partition 69 normal rows")
        expected_query = np.array_split(
            np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64),
            FROZEN_CONFIG_ANTIGRAVITY.folds,
        )[record.fold_index]
        expected_query_set = set(int(value) for value in expected_query)
        expected_fit = np.asarray(
            tuple(
                value
                for value in range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows)
                if value not in expected_query_set
            ),
            dtype=np.int64,
        )
        if not np.array_equal(
            record.input_query_indices,
            expected_query,
        ) or not np.array_equal(record.input_fit_indices, expected_fit):
            raise ValueError("record indices differ from exact blocked fold")
        _validate_record_evidence_antigravity(record)
        stored = _qualification_row_payload_antigravity(
            qualification.fold_results[index],
        )
        derived = _derived_qualification_payload_antigravity(record)
        if not _payloads_close_antigravity(stored, derived):
            raise ValueError("qualification differs from retained Task 4 evidence")

    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        oof = np.concatenate(
            tuple(record.input_query_indices for key, record in items if key[0] == seed),
        )
        if not np.array_equal(
            np.sort(oof),
            np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed, dtype=np.int64),
        ):
            raise ValueError("records do not retain exact seed OOF coverage")
        confirmations = sum(
            int(np.count_nonzero(record.confirmed["normal"]))
            for key, record in items
            if key[0] == seed
        )
        if qualification.normal_confirmations[seed] != confirmations:
            raise ValueError("qualification normal confirmations differ from records")
    if _decision_payload_antigravity(decision) != _expected_decision_payload_antigravity(
        qualification,
    ):
        raise ValueError("decision differs from retained qualification")
    return items


def _metric_row_antigravity(
    record: FoldMethodScoresAntigravity,
    method: str,
    qualification_row: Exp466FoldResultAntigravity,
) -> dict[str, object]:
    per_type_auc, mean_auc, finite = _method_metrics_antigravity(record, method)
    c0_auc, _c0_mean, _c0_finite = _method_metrics_antigravity(record, "C0")
    return {
        "schema": "exp466_method_fold_metric_antigravity_v1",
        "seed": record.seed,
        "fold_index": record.fold_index,
        "method": method,
        "change_order": list(_CHANGE_TYPES_ANTIGRAVITY),
        "normal_scores": record.original[method].tolist(),
        "primary_scores": {
            name: record.primary[method][name].tolist()
            for name in _CHANGE_TYPES_ANTIGRAVITY
        },
        "fit_margins": record.fit_margins.tolist(),
        "query_margins": {
            scope: record.query_margins[scope].tolist() for scope in _SCOPES_ANTIGRAVITY
        },
        "fit_normal_max_margin": record.fit_normal_max_margin,
        "confirmed": {
            scope: record.confirmed[scope].tolist() for scope in _SCOPES_ANTIGRAVITY
        },
        "confirmation_counts": {
            scope: int(np.count_nonzero(record.confirmed[scope]))
            for scope in _SCOPES_ANTIGRAVITY
        },
        "per_type_auc": list(per_type_auc),
        "mean_auc": mean_auc,
        "all_scores_finite": finite,
        "c0_deltas": {
            "gradual_drift": per_type_auc[1] - c0_auc[1],
            "period_change": per_type_auc[3] - c0_auc[3],
        },
        "qualification": (
            _qualification_row_payload_antigravity(qualification_row)
            if method == "C9"
            else None
        ),
        "input_train140": _float64_blob_antigravity(record.input_train140),
        "input_fit_indices": record.input_fit_indices.tolist(),
        "input_query_indices": record.input_query_indices.tolist(),
    }


def _summary_rows_antigravity(
    metric_rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for method in _METHODS_ANTIGRAVITY:
        selected = tuple(row for row in metric_rows if row["method"] == method)
        rows.append(
            {
                "schema": "exp466_method_summary_antigravity_v1",
                "method": method,
                "fold_count": len(selected),
                "min_per_type_auc": [
                    min(row["per_type_auc"][index] for row in selected)  # type: ignore[index]
                    for index in range(len(_CHANGE_TYPES_ANTIGRAVITY))
                ],
                "min_mean_auc": min(row["mean_auc"] for row in selected),
                "normal_confirmation_count": sum(
                    row["confirmation_counts"]["normal"] for row in selected  # type: ignore[index]
                ),
                "period_confirmation_count": sum(
                    row["confirmation_counts"]["period_change"]  # type: ignore[index]
                    for row in selected
                ),
                "all_scores_finite": all(row["all_scores_finite"] for row in selected),
                "c0_gradual_delta_min": min(
                    row["c0_deltas"]["gradual_drift"] for row in selected  # type: ignore[index]
                ),
                "c0_period_delta_min": min(
                    row["c0_deltas"]["period_change"] for row in selected  # type: ignore[index]
                ),
                "qualification_passed_count": (
                    sum(row["qualification"]["passed"] for row in selected)  # type: ignore[index]
                    if method == "C9"
                    else None
                ),
                "qualification_failed_count": (
                    sum(not row["qualification"]["passed"] for row in selected)  # type: ignore[index]
                    if method == "C9"
                    else None
                ),
            },
        )
    return tuple(rows)


def _context_payload_antigravity(
    qualification: Exp466QualificationAntigravity,
    *,
    metrics_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "exp466_task5_canonical_result_antigravity_v1",
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split,
        "train_md5": FROZEN_CONFIG_ANTIGRAVITY.train_md5,
        "interpolated_width": FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
        "seeds": list(FROZEN_CONFIG_ANTIGRAVITY.seeds),
        "folds": FROZEN_CONFIG_ANTIGRAVITY.folds,
        "method_order": list(_METHODS_ANTIGRAVITY),
        "change_order": list(_CHANGE_TYPES_ANTIGRAVITY),
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "array_encoding": "base64_float64_le_c_v1",
        "method_fold_metrics_sha256": metrics_sha256,
        "normal_trial_counts": {
            str(seed): qualification.normal_trial_counts[seed]
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        },
        "normal_confirmations": {
            str(seed): qualification.normal_confirmations[seed]
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        },
        "gates": {
            "per_type_auc_min": FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min,
            "mean_auc_min": FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min,
            "period_auc_strictly_above_c0": True,
            "gradual_auc_not_below_c0": True,
            "period_confirmation_min": 1,
            "normal_confirmation_must_equal": 0,
            "arithmetic_tolerance": _FLOAT_TOLERANCE_ANTIGRAVITY,
        },
        "label_query_count": qualification.label_query_count,
        "official_evaluate_count": qualification.official_evaluate_count,
        "official_inspect_count": qualification.official_inspect_count,
        "artifact_map_sha256_before_private_run": qualification.artifact_map_sha256,
        "actual_private_official_result": False,
    }


def _payloads_antigravity(
    items: tuple[tuple[tuple[int, int], FoldMethodScoresAntigravity], ...],
    qualification: Exp466QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, bytes]:
    metric_rows = tuple(
        _metric_row_antigravity(record, method, qualification.fold_results[index])
        for index, (_key, record) in enumerate(items)
        for method in _METHODS_ANTIGRAVITY
    )
    metrics = _jsonl_bytes_antigravity(metric_rows)
    failed = tuple(
        {
            "schema": "exp466_failed_fold_antigravity_v1",
            **_qualification_row_payload_antigravity(row),
        }
        for row in qualification.fold_results
        if not row.passed
    )
    return {
        "context_antigravity.json": canonical_json_bytes_antigravity(
            _context_payload_antigravity(
                qualification,
                metrics_sha256=sha256(metrics).hexdigest(),
            ),
        ),
        "dashboard_snapshot_antigravity.json": canonical_json_bytes_antigravity(
            export_dashboard_antigravity(qualification, decision),
        ),
        "decision_antigravity.json": canonical_json_bytes_antigravity(
            _decision_payload_antigravity(decision),
        ),
        "failed_folds_antigravity.jsonl": _jsonl_bytes_antigravity(failed),
        "method_fold_metrics_antigravity.jsonl": metrics,
        "method_summary_antigravity.jsonl": _jsonl_bytes_antigravity(
            _summary_rows_antigravity(metric_rows),
        ),
    }


def _write_exclusive_antigravity(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def publish_result_antigravity(
    root: Path,
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: Exp466QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> PublishedResultAntigravity:
    """Validate Task 4 values and create exactly seven canonical leaves."""

    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ValueError("publication destination must be a new create-only path")
    items = _normalise_inputs_antigravity(records, qualification, decision)
    payloads = _payloads_antigravity(items, qualification, decision)
    artifact_map = {
        name: sha256(payloads[name]).hexdigest() for name in sorted(payloads)
    }
    map_bytes = canonical_json_bytes_antigravity(artifact_map)
    try:
        root.mkdir(mode=0o700)
        for name in sorted(payloads):
            _write_exclusive_antigravity(root / name, payloads[name])
        _write_exclusive_antigravity(
            root / "artifact_sha256_antigravity.json",
            map_bytes,
        )
    except OSError as error:
        raise ValueError("publication destination could not be created") from error
    leaf_sha256 = tuple(
        (name, sha256((root / name).read_bytes()).hexdigest())
        for name in _LEAVES_ANTIGRAVITY
    )
    return PublishedResultAntigravity(
        root=root,
        leaf_sha256=leaf_sha256,
        artifact_map_sha256=sha256(map_bytes).hexdigest(),
    )


__all__ = (
    "PublishedResultAntigravity",
    "canonical_json_bytes_antigravity",
    "publish_result_antigravity",
)
