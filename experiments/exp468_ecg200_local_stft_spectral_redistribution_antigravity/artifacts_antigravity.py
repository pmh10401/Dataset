"""Canonical, source-backed publication for Exp468 Task 5.

The publication deliberately keeps all replay evidence in the context leaf.  The
other six leaves are deterministic projections of that evidence and the Task 4
scalar contracts.  No ECG200 loader, TEST label, private result root, or routing
side effect belongs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import base64
import hashlib
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp468FoldResultAntigravity,
    Exp468QualificationAntigravity,
)
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import qualify_antigravity
from .scoring_antigravity import FoldMethodScoresAntigravity


LEAF_NAMES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
_HASHED_LEAF_NAMES_ANTIGRAVITY = tuple(
    name for name in LEAF_NAMES_ANTIGRAVITY if name != "artifact_sha256_antigravity.json"
)
_SCOPES_ANTIGRAVITY = ("normal",) + FROZEN_CONFIG_ANTIGRAVITY.change_types
_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_APPROVAL_SHA256_ANTIGRAVITY = (
    "6780edb2683c992b3df361c72e15542a6e974cf3e7cbbc0df65df2739c0203a9"
)
_PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY = (
    "938d83b0319346270a6cc32b7b6ba3efb88122dc1bf431863d8030f4539fb63b"
)


def _json_safe_antigravity(value: object) -> None:
    """Reject NaN, infinity, non-string keys, and unsupported JSON values."""

    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("JSON payload cannot contain NaN or infinity")
        return
    if type(value) is list or type(value) is tuple:
        for item in value:
            _json_safe_antigravity(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON payload keys must be strings")
            _json_safe_antigravity(item)
        return
    raise TypeError(f"unsupported JSON payload value: {type(value).__name__}")


def _canonical_json_bytes_antigravity(payload: object) -> bytes:
    _json_safe_antigravity(payload)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _duplicate_key_guard_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes_antigravity(raw: bytes, *, name: str) -> object:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_duplicate_key_guard_antigravity,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid canonical JSON in {name}: {exc}") from exc
    expected = _canonical_json_bytes_antigravity(payload)
    if raw != expected:
        raise ValueError(f"{name} is not canonical UTF-8 JSON")
    return payload


def _write_json_antigravity(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_json_bytes_antigravity(payload))


def _write_jsonl_antigravity(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        path.write_bytes(b"")
        return
    path.write_bytes(b"".join(_canonical_json_bytes_antigravity(row) for row in rows))


def _parse_jsonl_bytes_antigravity(raw: bytes, *, name: str) -> list[dict[str, object]]:
    if raw == b"":
        return []
    if not raw.endswith(b"\n"):
        raise ValueError(f"{name} must end with exactly one newline per row")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line[:-1].endswith(b"\n"):
            raise ValueError(f"{name} row {index} has invalid newline framing")
        payload = _parse_json_bytes_antigravity(line, name=f"{name} row {index}")
        if type(payload) is not dict:
            raise ValueError(f"{name} row {index} must be an object")
        rows.append(payload)
    return rows


def _dtype_code_antigravity(array: np.ndarray) -> str:
    if array.dtype == np.dtype(np.float64):
        return "<f8"
    if array.dtype == np.dtype(np.int64):
        return "<i8"
    if array.dtype == np.dtype(np.bool_):
        return "|b1"
    raise TypeError(f"unsupported evidence dtype: {array.dtype}")


def _array_payload_antigravity(array: np.ndarray, *, name: str) -> dict[str, object]:
    if type(array) is not np.ndarray or not array.flags.c_contiguous:
        raise TypeError(f"{name} must be a C-contiguous ndarray")
    if array.dtype == np.dtype(np.float64) and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    dtype_code = _dtype_code_antigravity(array)
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype_code))
    raw = canonical.tobytes(order="C")
    return {
        "data": base64.b64encode(raw).decode("ascii"),
        "dtype": dtype_code,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _decode_array_antigravity(payload: object, *, name: str) -> np.ndarray:
    if type(payload) is not dict or set(payload) != {"data", "dtype", "shape", "sha256"}:
        raise ValueError(f"{name} has an invalid array envelope")
    dtype_code = payload["dtype"]
    if dtype_code not in {"<f8", "<i8", "|b1"}:
        raise ValueError(f"{name} has an unsupported dtype")
    shape = payload["shape"]
    if type(shape) is not list or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError(f"{name} has an invalid shape")
    data = payload["data"]
    if type(data) is not str:
        raise ValueError(f"{name} has invalid base64 data")
    try:
        raw = base64.b64decode(data.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError(f"{name} has invalid base64 data") from exc
    digest = payload["sha256"]
    if type(digest) is not str or digest != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"{name} raw-byte digest mismatch")
    dtype = np.dtype(dtype_code)
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError(f"{name} byte length does not match shape")
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy(order="C")
    if dtype == np.dtype(np.float64) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _record_payload_antigravity(record: FoldMethodScoresAntigravity) -> dict[str, object]:
    if type(record) is not FoldMethodScoresAntigravity:
        raise ValueError("records must contain exact FoldMethodScoresAntigravity values")
    return {
        "backend": record.backend,
        "confirmed": {
            scope: _array_payload_antigravity(record.confirmed[scope], name=f"confirmed[{scope}]")
            for scope in _SCOPES_ANTIGRAVITY
        },
        "fit_normal_max_raw_js": record.fit_normal_max_raw_js,
        "fit_raw_js": _array_payload_antigravity(record.fit_raw_js, name="fit_raw_js"),
        "fold_index": record.fold_index,
        "input_fit_indices": _array_payload_antigravity(
            record.input_fit_indices,
            name="input_fit_indices",
        ),
        "input_query_indices": _array_payload_antigravity(
            record.input_query_indices,
            name="input_query_indices",
        ),
        "input_train140": _array_payload_antigravity(record.input_train140, name="input_train140"),
        "methods": list(record.methods),
        "original": {
            method: _array_payload_antigravity(record.original[method], name=f"original[{method}]")
            for method in _METHODS_ANTIGRAVITY
        },
        "primary": {
            method: {
                change_type: _array_payload_antigravity(
                    record.primary[method][change_type],
                    name=f"primary[{method}][{change_type}]",
                )
                for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_types
            }
            for method in _METHODS_ANTIGRAVITY
        },
        "query_raw_js": {
            scope: _array_payload_antigravity(record.query_raw_js[scope], name=f"query_raw_js[{scope}]")
            for scope in _SCOPES_ANTIGRAVITY
        },
        "seed": record.seed,
    }


def _record_from_payload_antigravity(payload: object) -> FoldMethodScoresAntigravity:
    expected = (
        "backend",
        "confirmed",
        "fit_normal_max_raw_js",
        "fit_raw_js",
        "fold_index",
        "input_fit_indices",
        "input_query_indices",
        "input_train140",
        "methods",
        "original",
        "primary",
        "query_raw_js",
        "seed",
    )
    if type(payload) is not dict or set(payload) != set(expected):
        raise ValueError("record payload has an invalid schema")
    confirmed = payload["confirmed"]
    query_raw = payload["query_raw_js"]
    original = payload["original"]
    primary = payload["primary"]
    if (
        type(confirmed) is not dict
        or set(confirmed) != set(_SCOPES_ANTIGRAVITY)
        or type(query_raw) is not dict
        or set(query_raw) != set(_SCOPES_ANTIGRAVITY)
        or type(original) is not dict
        or set(original) != set(_METHODS_ANTIGRAVITY)
        or type(primary) is not dict
        or set(primary) != set(_METHODS_ANTIGRAVITY)
    ):
        raise ValueError("record payload mapping order drifted")
    primary_values: dict[str, Mapping[str, np.ndarray]] = {}
    for method in _METHODS_ANTIGRAVITY:
        values = primary[method]
        if (
            type(values) is not dict
            or set(values) != set(FROZEN_CONFIG_ANTIGRAVITY.change_types)
        ):
            raise ValueError("record primary scope order drifted")
        primary_values[method] = MappingProxyType(
            {
                change_type: _decode_array_antigravity(
                    values[change_type],
                    name=f"primary[{method}][{change_type}]",
                )
                for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_types
            },
        )
    methods = payload["methods"]
    if type(methods) is not list or tuple(methods) != _METHODS_ANTIGRAVITY:
        raise ValueError("record methods order drifted")
    fit_max = payload["fit_normal_max_raw_js"]
    if type(fit_max) is not float or not isfinite(fit_max):
        raise ValueError("fit_normal_max_raw_js must be a finite float")
    return FoldMethodScoresAntigravity(
        seed=payload["seed"],
        fold_index=payload["fold_index"],
        methods=_METHODS_ANTIGRAVITY,
        original=MappingProxyType(
            {
                method: _decode_array_antigravity(original[method], name=f"original[{method}]")
                for method in _METHODS_ANTIGRAVITY
            },
        ),
        primary=MappingProxyType(primary_values),
        fit_raw_js=_decode_array_antigravity(payload["fit_raw_js"], name="fit_raw_js"),
        query_raw_js=MappingProxyType(
            {
                scope: _decode_array_antigravity(query_raw[scope], name=f"query_raw_js[{scope}]")
                for scope in _SCOPES_ANTIGRAVITY
            },
        ),
        fit_normal_max_raw_js=fit_max,
        confirmed=MappingProxyType(
            {
                scope: _decode_array_antigravity(confirmed[scope], name=f"confirmed[{scope}]")
                for scope in _SCOPES_ANTIGRAVITY
            },
        ),
        input_train140=_decode_array_antigravity(payload["input_train140"], name="input_train140"),
        input_fit_indices=_decode_array_antigravity(
            payload["input_fit_indices"],
            name="input_fit_indices",
        ),
        input_query_indices=_decode_array_antigravity(
            payload["input_query_indices"],
            name="input_query_indices",
        ),
        backend=payload["backend"],
    )


def _qualification_payload_antigravity(
    qualification: Exp468QualificationAntigravity,
) -> dict[str, object]:
    return {
        "artifact_map_sha256": qualification.artifact_map_sha256,
        "fold_results": [
            {
                "all_scores_finite": row.all_scores_finite,
                "c0_per_type_auc": list(row.c0_per_type_auc),
                "c10_per_type_auc": list(row.c10_per_type_auc),
                "failure_reasons": list(row.failure_reasons),
                "fold_index": row.fold_index,
                "normal_confirmations": row.normal_confirmations,
                "passed": row.passed,
                "period_confirmations": row.period_confirmations,
                "seed": row.seed,
            }
            for row in qualification.fold_results
        ],
        "label_query_count": qualification.label_query_count,
        "normal_confirmations": [
            {"seed": seed, "count": qualification.normal_confirmations[seed]}
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        ],
        "normal_trial_counts": [
            {"seed": seed, "count": qualification.normal_trial_counts[seed]}
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        ],
    }


def _decision_payload_antigravity(
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "champion_modified": decision.champion_modified,
        "eligible_for_private_reproducibility": decision.eligible_for_private_reproducibility,
        "failure_reasons": list(decision.failure_reasons),
        "label_query_count": decision.label_query_count,
        "pending_modified": decision.pending_modified,
        "primary_candidate": decision.primary_candidate,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "promotion_applied": decision.promotion_applied,
        "routing_applied": decision.routing_applied,
        "status": decision.status,
        "test_labels_opened": decision.test_labels_opened,
    }


def _fold_metrics_payload_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: Exp468QualificationAntigravity,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_identity = {
        (row.seed, row.fold_index): row for row in qualification.fold_results
    }
    for identity, record in records.items():
        qualification_row = by_identity[identity]
        for method in _METHODS_ANTIGRAVITY:
            arrays = (
                record.original[method],
                *(record.primary[method][change_type]
                  for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_types),
            )
            per_type = tuple(
                float(
                    mann_whitney_auc_antigravity(
                        record.original[method],
                        record.primary[method][change_type],
                    ),
                )
                for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_types
            )
            rows.append(
                {
                    "all_scores_finite": bool(all(np.isfinite(array).all() for array in arrays)),
                    "approval_document_sha256": _APPROVAL_SHA256_ANTIGRAVITY,
                    "backend": "numpy",
                    "c0_per_type_auc": list(qualification_row.c0_per_type_auc),
                    "c10_per_type_auc": list(qualification_row.c10_per_type_auc),
                    "delta_per_type_auc": [
                        c10 - c0
                        for c0, c10 in zip(
                            qualification_row.c0_per_type_auc,
                            qualification_row.c10_per_type_auc,
                        )
                    ],
                    "fold_index": record.fold_index,
                    "method": method,
                    "normal_confirmations": int(np.count_nonzero(record.confirmed["normal"])),
                    "per_type_auc": list(per_type),
                    "pre_approval_benchmark_sha256": _PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY,
                    "period_confirmations": int(np.count_nonzero(record.confirmed["period_change"])),
                    "seed": record.seed,
                    "source_fold_passed": qualification_row.passed,
                },
            )
    return rows


def _method_summary_payload_antigravity(
    fold_metrics: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in _METHODS_ANTIGRAVITY:
        selected = [row for row in fold_metrics if row["method"] == method]
        means = tuple(
            float(np.mean([row["per_type_auc"][index] for row in selected]))
            for index in range(len(FROZEN_CONFIG_ANTIGRAVITY.change_types))
        )
        rows.append(
            {
                "fold_count": len(selected),
                "gate_use": "display_only_no_gate",
                "mean_per_type_auc": list(means),
                "method": method,
                "normal_confirmation_total": sum(
                    int(row["normal_confirmations"]) for row in selected
                ),
                "period_confirmation_total": sum(
                    int(row["period_confirmations"]) for row in selected
                ),
            },
        )
    return rows


def _failed_folds_payload_antigravity(
    qualification: Exp468QualificationAntigravity,
) -> list[dict[str, object]]:
    return [
        {
            "failure_reasons": list(row.failure_reasons),
            "fold_index": row.fold_index,
            "passed": row.passed,
            "seed": row.seed,
        }
        for row in qualification.fold_results
        if not row.passed
    ]


def _dashboard_snapshot_payload_antigravity(
    qualification: Exp468QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    period_index = FROZEN_CONFIG_ANTIGRAVITY.change_types.index("period_change")
    protected_names = FROZEN_CONFIG_ANTIGRAVITY.change_types[:3]
    fold_rows = [
        {
            "c0_period_auc": row.c0_per_type_auc[period_index],
            "c10_period_auc": row.c10_per_type_auc[period_index],
            "delta": row.c10_per_type_auc[period_index] - row.c0_per_type_auc[period_index],
            "failure_reasons": list(row.failure_reasons),
            "fold_index": row.fold_index,
            "passed": row.passed,
            "period_confirmations": row.period_confirmations,
            "seed": row.seed,
        }
        for row in qualification.fold_results
    ]
    improved = [
        {"fold_index": row["fold_index"], "seed": row["seed"]}
        for row in fold_rows
        if row["delta"] > 0.0
    ]
    protected_min_deltas = {
        change_type: min(
            row.c10_per_type_auc[index] - row.c0_per_type_auc[index]
            for row in qualification.fold_results
        )
        for index, change_type in enumerate(protected_names)
    }
    return {
        "decision_status": decision.status,
        "folds": fold_rows,
        "improved_fold_count": len(improved),
        "improved_folds": improved,
        "normal_confirmations_by_seed": [
            {"count": qualification.normal_confirmations[seed], "seed": seed}
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        ],
        "private_reproducibility": {
            "comparison_status": "not_yet_compared",
            "run_1": "not_run",
            "run_2": "not_run",
            "status": (
                "awaiting_private_reproducibility"
                if decision.eligible_for_private_reproducibility
                else "qualification_failed_blocked_before_private_run"
            ),
        },
        "protected_min_deltas": protected_min_deltas,
        "qualification_passed": decision.eligible_for_private_reproducibility,
        "schema": "exp468_task5_dashboard_snapshot_v1",
    }


def _context_payload_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: Exp468QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "approved_fallback": {
            "approval_document_sha256": _APPROVAL_SHA256_ANTIGRAVITY,
            "backend": "numpy",
            "pre_approval_benchmark_sha256": _PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY,
        },
        "backend": "numpy",
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "external_runs": {"official": 0, "private": 0},
        "qualification": _qualification_payload_antigravity(qualification),
        "records": [
            _record_payload_antigravity(record) for record in records.values()
        ],
        "schema": "exp468_task5_context_v1",
        "task4_decision_status": decision.status,
        "test_labels_opened": False,
    }


def _validate_public_contract_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: object,
    decision: object,
) -> tuple[Exp468QualificationAntigravity, QualificationDecisionAntigravity]:
    if type(qualification) is not Exp468QualificationAntigravity:
        raise ValueError("qualification must be exact Exp468QualificationAntigravity")
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("decision must be exact QualificationDecisionAntigravity")
    recomputed_qualification = qualify_antigravity(records)
    if recomputed_qualification != qualification:
        raise ValueError("qualification does not equal the exact Task 4 replay")
    recomputed_decision = evaluate_qualification_antigravity(recomputed_qualification)
    if recomputed_decision != decision:
        raise ValueError("decision does not equal the exact Task 4 replay")
    return recomputed_qualification, recomputed_decision


@dataclass(frozen=True, slots=True)
class PublishedResultAntigravity:
    root: Path
    qualification: Exp468QualificationAntigravity
    decision: QualificationDecisionAntigravity
    artifact_sha256: Mapping[str, str]


def publish_result_antigravity(
    root: str | Path,
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
    qualification: Exp468QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> PublishedResultAntigravity:
    """Publish one deterministic seven-leaf result root after exact replay."""

    target = Path(root)
    if target.exists():
        raise FileExistsError(f"publication root already exists: {target}")
    actual_qualification, actual_decision = _validate_public_contract_antigravity(
        records,
        qualification,
        decision,
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    context = _context_payload_antigravity(records, actual_qualification, actual_decision)
    dashboard_snapshot = _dashboard_snapshot_payload_antigravity(
        actual_qualification,
        actual_decision,
    )
    decision_payload = _decision_payload_antigravity(actual_decision)
    fold_metrics = _fold_metrics_payload_antigravity(records, actual_qualification)
    summary = _method_summary_payload_antigravity(fold_metrics)
    failed = _failed_folds_payload_antigravity(actual_qualification)

    target.mkdir()
    _write_json_antigravity(target / "context_antigravity.json", context)
    _write_json_antigravity(target / "dashboard_snapshot_antigravity.json", dashboard_snapshot)
    _write_json_antigravity(target / "decision_antigravity.json", decision_payload)
    _write_jsonl_antigravity(target / "method_fold_metrics_antigravity.jsonl", fold_metrics)
    _write_jsonl_antigravity(target / "method_summary_antigravity.jsonl", summary)
    _write_jsonl_antigravity(target / "failed_folds_antigravity.jsonl", failed)
    digests = {
        name: hashlib.sha256((target / name).read_bytes()).hexdigest()
        for name in _HASHED_LEAF_NAMES_ANTIGRAVITY
    }
    _write_json_antigravity(target / "artifact_sha256_antigravity.json", digests)
    return PublishedResultAntigravity(
        root=target,
        qualification=actual_qualification,
        decision=actual_decision,
        artifact_sha256=MappingProxyType(dict(digests)),
    )


__all__ = (
    "LEAF_NAMES_ANTIGRAVITY",
    "PublishedResultAntigravity",
    "publish_result_antigravity",
)
