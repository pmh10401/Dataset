"""Source-free semantic inspection for canonical Exp470 synthetic evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .artifacts_antigravity import (
    LEAF_NAMES_ANTIGRAVITY,
    _canonical_json_bytes_antigravity,
    _dashboard_snapshot_payload_antigravity,
    _decision_payload_antigravity,
    _fold_result_payload_antigravity,
    _metric_row_antigravity,
    _method_summary_rows_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp470QualificationAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_repair_gates_antigravity
from .qualification_antigravity import blocked_fold_indices_antigravity, qualify_records_antigravity
from .scoring_antigravity import FoldMethodScoresAntigravity, _combine_c11_numba_antigravity, _scope_rows_antigravity
from .spectral_entropy_antigravity import score_spectral_entropy_numba_antigravity


_SCOPES_ANTIGRAVITY = ("normal",) + FROZEN_CONFIG_ANTIGRAVITY.change_types


def _read_canonical_json_antigravity(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid canonical JSON: {path.name}") from error
    if _canonical_json_bytes_antigravity(value) != raw:
        raise ValueError(f"noncanonical JSON: {path.name}")
    return value


def _read_canonical_jsonl_antigravity(path: Path) -> tuple[Any, ...]:
    raw = path.read_bytes()
    if not raw:
        return ()
    lines = raw.splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in lines):
        raise ValueError(f"noncanonical JSONL: {path.name}")
    values: list[Any] = []
    for line in lines:
        try:
            value = json.loads(line.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid canonical JSONL: {path.name}") from error
        if _canonical_json_bytes_antigravity(value) != line:
            raise ValueError(f"noncanonical JSONL: {path.name}")
        values.append(value)
    return tuple(values)


def _validate_leaf_set_and_hashes_antigravity(root: Path) -> tuple[dict[str, Any], Mapping[str, str]]:
    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("result root must be an existing directory")
    names = tuple(sorted(path.name for path in root.iterdir()))
    if names != tuple(sorted(LEAF_NAMES_ANTIGRAVITY)):
        raise ValueError("result root must contain the exact publication leaf set")
    hash_payload = _read_canonical_json_antigravity(root / LEAF_NAMES_ANTIGRAVITY[0])
    if set(hash_payload) != {"hashes", "schema"} or hash_payload["schema"] != "exp470_artifact_sha256_v1":
        raise ValueError("invalid artifact hash leaf")
    hashes = hash_payload["hashes"]
    expected_names = LEAF_NAMES_ANTIGRAVITY[1:]
    if type(hashes) is not dict or tuple(hashes) != expected_names:
        raise ValueError("artifact hash names or order drifted")
    for name in expected_names:
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if type(hashes[name]) is not str or hashes[name] != actual:
            raise ValueError(f"artifact hash mismatch: {name}")
    payloads = {
        "context": _read_canonical_json_antigravity(root / "context_antigravity.json"),
        "dashboard": _read_canonical_json_antigravity(root / "dashboard_snapshot_antigravity.json"),
        "decision": _read_canonical_json_antigravity(root / "decision_antigravity.json"),
        "failed": _read_canonical_jsonl_antigravity(root / "failed_folds_antigravity.jsonl"),
        "metrics": _read_canonical_jsonl_antigravity(root / "method_fold_metrics_antigravity.jsonl"),
        "summary": _read_canonical_jsonl_antigravity(root / "method_summary_antigravity.jsonl"),
    }
    return payloads, MappingProxyType(dict(hashes))


def _float_array_antigravity(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not float64-compatible") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} has invalid shape or finite values")
    output = np.array(result, dtype=np.float64, copy=True, order="C")
    output.setflags(write=False)
    return output


def _int_array_antigravity(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must contain exact integer JSON values")
    output = np.asarray(value, dtype=np.int64)
    if output.shape != shape:
        raise ValueError(f"{name} has invalid shape")
    output = np.array(output, dtype=np.int64, copy=True, order="C")
    output.setflags(write=False)
    return output


def _bool_array_antigravity(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise ValueError(f"{name} must contain exact bool JSON values")
    output = np.asarray(value, dtype=np.bool_)
    if output.shape != shape:
        raise ValueError(f"{name} has invalid shape")
    output = np.array(output, dtype=np.bool_, copy=True, order="C")
    output.setflags(write=False)
    return output


def _rebuild_records_antigravity(context: object, metrics: tuple[object, ...]) -> Mapping[tuple[int, int], FoldMethodScoresAntigravity]:
    if type(context) is not dict or set(context) != {"records", "schema", "shared_train140"} or context["schema"] != "exp470_context_v1":
        raise ValueError("context schema drifted")
    config = FROZEN_CONFIG_ANTIGRAVITY
    train = _float_array_antigravity(context["shared_train140"], (config.normal_rows, config.interpolated_width), "shared_train140")
    expected = tuple((seed, fold) for seed in config.seeds for fold in range(config.folds))
    if not isinstance(context["records"], list) or len(context["records"]) != len(expected) or len(metrics) != len(expected):
        raise ValueError("context or metrics record count drifted")
    folds = blocked_fold_indices_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for identity, item, stored_metric in zip(expected, context["records"], metrics, strict=True):
        if type(item) is not dict or type(stored_metric) is not dict or (item.get("seed"), item.get("fold_index")) != identity or (stored_metric.get("seed"), stored_metric.get("fold_index")) != identity:
            raise ValueError("context/metric identity or order drifted")
        fit_expected, query_expected = folds[identity[1]]
        fit = _int_array_antigravity(item.get("fit_indices"), (fit_expected.size,), "fit_indices")
        query = _int_array_antigravity(item.get("query_indices"), (query_expected.size,), "query_indices")
        if not np.array_equal(fit, fit_expected) or not np.array_equal(query, query_expected):
            raise ValueError("stored blocked fold indices drifted")
        width = query.size
        c0_normal = _float_array_antigravity(item.get("c0_normal"), (width,), "c0_normal")
        c0_primary_source = item.get("c0_primary")
        if type(c0_primary_source) is not dict or set(c0_primary_source) != set(config.change_types):
            raise ValueError("context C0 primary order drifted")
        c0_primary = {change: _float_array_antigravity(c0_primary_source[change], (width,), f"c0_primary[{change}]") for change in config.change_types}
        generated = _scope_rows_antigravity(train, query)
        score = score_spectral_entropy_numba_antigravity(train[fit], np.concatenate(tuple(generated[scope] for scope in _SCOPES_ANTIGRAVITY), axis=0))
        if score.backend != "numba":
            raise ValueError("replay E11 backend drifted")
        raw = {scope: score.query_raw_entropy[position * width:(position + 1) * width] for position, scope in enumerate(_SCOPES_ANTIGRAVITY)}
        rank = {scope: score.query_ranks[position * width:(position + 1) * width] for position, scope in enumerate(_SCOPES_ANTIGRAVITY)}
        confirmed = {scope: score.confirmation_mask[position * width:(position + 1) * width] for position, scope in enumerate(_SCOPES_ANTIGRAVITY)}
        c11_normal = _combine_c11_numba_antigravity(c0_normal, rank["normal"], confirmed["normal"])
        c11_primary = {change: _combine_c11_numba_antigravity(c0_primary[change], rank[change], confirmed[change]) for change in config.change_types}
        record = FoldMethodScoresAntigravity(
            seed=identity[0], fold_index=identity[1], methods=config.method_order,
            original={"C0": c0_normal, "E11": rank["normal"], "C11": c11_normal},
            primary={"C0": c0_primary, "E11": {change: rank[change] for change in config.change_types}, "C11": c11_primary},
            fit_raw_entropy=score.fit_raw_entropy, query_raw_entropy=raw,
            fit_normal_max_raw_entropy=score.fit_normal_max_raw_entropy, confirmed=confirmed,
            input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba",
        )
        records[identity] = record
    return MappingProxyType(records)


@dataclass(frozen=True, slots=True)
class InspectedResultAntigravity:
    root: Path
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity]
    qualification: Exp470QualificationAntigravity
    decision: QualificationDecisionAntigravity
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


def inspect_result_antigravity(root: object) -> InspectedResultAntigravity:
    """Verify bytes first, then regenerate every record and aggregate without source files."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    payloads, hashes = _validate_leaf_set_and_hashes_antigravity(root)
    records = _rebuild_records_antigravity(payloads["context"], payloads["metrics"])
    qualification = qualify_records_antigravity(records)
    decision = evaluate_repair_gates_antigravity(qualification)
    expected_metrics = tuple(_metric_row_antigravity(record, result) for record, result in zip(records.values(), qualification.fold_results, strict=True))
    if tuple(payloads["metrics"]) != expected_metrics:
        raise ValueError("stored fold metrics disagree with regenerated evidence")
    expected_summary = _method_summary_rows_antigravity(expected_metrics)
    if tuple(payloads["summary"]) != expected_summary:
        raise ValueError("stored method summary is not derived from fold metrics")
    expected_failed = tuple({"failure_reasons": list(row.failure_reasons), "fold_index": row.fold_index, "seed": row.seed} for row in qualification.fold_results if not row.passed)
    if tuple(payloads["failed"]) != expected_failed:
        raise ValueError("stored failed-fold rows disagree with qualification")
    if payloads["decision"] != _decision_payload_antigravity(qualification, decision):
        raise ValueError("stored decision disagrees with recomputed gates")
    if payloads["dashboard"] != _dashboard_snapshot_payload_antigravity(qualification, decision, expected_summary):
        raise ValueError("stored dashboard snapshot is not derived evidence")
    return InspectedResultAntigravity(root=root, records=records, qualification=qualification, decision=decision, hashes=hashes)


__all__ = ("InspectedResultAntigravity", "inspect_result_antigravity")
