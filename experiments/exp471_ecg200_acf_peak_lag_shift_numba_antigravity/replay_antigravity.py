"""Source-free semantic inspection for canonical Exp471 synthetic evidence."""

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
    _failed_rows_antigravity,
    _fold_result_payload_antigravity,
    _metric_row_antigravity,
    _method_summary_rows_antigravity,
    evaluate_qualification_antigravity,
)
from .acf_peak_lag_antigravity import score_acf_peak_lag_numba_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp471QualificationAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity
from .qualification_antigravity import blocked_fold_indices_antigravity, qualify_records_antigravity
from .scoring_antigravity import (
    FoldMethodScoresAntigravity,
    _combine_c12_numba_antigravity,
    _scope_rows_antigravity,
)


_SCOPES_ANTIGRAVITY = ("normal",) + FROZEN_CONFIG_ANTIGRAVITY.change_types
_CONTEXT_RECORD_KEYS_ANTIGRAVITY = frozenset((
    "c0_normal",
    "c0_primary",
    "confirmed",
    "e12_ranks",
    "fit_early_peak_lags",
    "fit_indices",
    "fit_late_peak_lags",
    "fit_normal_max_raw_shift",
    "fit_peak_presence",
    "fit_raw_shift",
    "fit_shifts",
    "fold_index",
    "query_early_peak_lags",
    "query_indices",
    "query_late_peak_lags",
    "query_peak_presence",
    "query_raw_shift",
    "query_shifts",
    "seed",
))


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _read_canonical_json_antigravity(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
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
            value = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
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
    if set(hash_payload) != {"hashes", "schema"} or hash_payload["schema"] != "exp471_artifact_sha256_v1":
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
    def validate_tree(item: object, depth: int) -> None:
        if depth == len(shape):
            if type(item) is not float:
                raise ValueError(f"{name} must contain exact float JSON values")
            return
        if type(item) is not list or len(item) != shape[depth]:
            raise ValueError(f"{name} has invalid nested list shape")
        for child in item:
            validate_tree(child, depth + 1)

    validate_tree(value, 0)
    output = np.asarray(value, dtype=np.float64)
    if output.shape != shape or not np.isfinite(output).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    result = np.array(output, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _int_array_antigravity(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    def validate_tree(item: object, depth: int) -> None:
        if depth == len(shape):
            if type(item) is not int:
                raise ValueError(f"{name} must contain exact integer JSON values")
            return
        if type(item) is not list or len(item) != shape[depth]:
            raise ValueError(f"{name} has invalid nested list shape")
        for child in item:
            validate_tree(child, depth + 1)

    validate_tree(value, 0)
    output = np.array(np.asarray(value), dtype=np.int64, copy=True, order="C")
    if output.shape != shape:
        raise ValueError(f"{name} has invalid shape")
    output.setflags(write=False)
    return output


def _bool_array_antigravity(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    def validate_tree(item: object, depth: int) -> None:
        if depth == len(shape):
            if type(item) is not bool:
                raise ValueError(f"{name} must contain exact bool JSON values")
            return
        if type(item) is not list or len(item) != shape[depth]:
            raise ValueError(f"{name} has invalid nested list shape")
        for child in item:
            validate_tree(child, depth + 1)

    validate_tree(value, 0)
    output = np.array(np.asarray(value), dtype=np.bool_, copy=True, order="C")
    if output.shape != shape:
        raise ValueError(f"{name} has invalid shape")
    output.setflags(write=False)
    return output


def _scope_mapping_antigravity(
    value: object,
    shape_for_scope,
    builder,
    name: str,
) -> Mapping[str, np.ndarray]:
    if type(value) is not dict or tuple(value) != _SCOPES_ANTIGRAVITY:
        raise ValueError(f"{name} scope order drifted")
    frozen: dict[str, np.ndarray] = {}
    for scope in _SCOPES_ANTIGRAVITY:
        frozen[scope] = builder(value[scope], shape_for_scope(scope), f"{name}[{scope}]")
    return MappingProxyType(frozen)


def _require_canonical_scope_mapping_antigravity(value: object, name: str) -> None:
    if type(value) is not dict or tuple(value) != tuple(sorted(_SCOPES_ANTIGRAVITY)):
        raise ValueError(f"semantic canonical scope order drifted: {name}")


def _rebuild_records_antigravity(
    context: object,
    metrics: tuple[object, ...],
) -> Mapping[tuple[int, int], FoldMethodScoresAntigravity]:
    """Reconstruct records only from retained inputs; no ECG200 loader is imported."""
    if (
        type(context) is not dict
        or set(context) != {"records", "schema", "shared_train140"}
        or context["schema"] != "exp471_context_v1"
    ):
        raise ValueError("context schema drifted semantically")
    config = FROZEN_CONFIG_ANTIGRAVITY
    train = _float_array_antigravity(context["shared_train140"], (config.normal_rows, config.interpolated_width), "shared_train140")
    expected = tuple((seed, fold) for seed in config.seeds for fold in range(config.folds))
    if not isinstance(context["records"], list) or len(context["records"]) != 10 or len(metrics) != 10:
        raise ValueError("context or metrics record count drifted")
    folds = blocked_fold_indices_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for identity, item, stored_metric in zip(expected, context["records"], metrics, strict=True):
        if (
            type(item) is not dict
            or set(item) != _CONTEXT_RECORD_KEYS_ANTIGRAVITY
            or type(stored_metric) is not dict
            or (item.get("seed"), item.get("fold_index")) != identity
            or (stored_metric.get("seed"), stored_metric.get("fold_index")) != identity
        ):
            raise ValueError("context/metric identity or order drifted")
        fit_expected, query_expected = folds[identity[1]]
        fit = _int_array_antigravity(item["fit_indices"], (fit_expected.size,), "fit_indices")
        query = _int_array_antigravity(item["query_indices"], (query_expected.size,), "query_indices")
        if not np.array_equal(fit, fit_expected) or not np.array_equal(query, query_expected):
            raise ValueError("stored blocked fold indices drifted")
        width = query.size
        fit_width = fit.size
        c0_normal = _float_array_antigravity(item["c0_normal"], (width,), "c0_normal")
        c0_primary_source = item["c0_primary"]
        expected_changes = tuple(sorted(FROZEN_CONFIG_ANTIGRAVITY.change_types))
        if type(c0_primary_source) is not dict or tuple(c0_primary_source) != expected_changes:
            raise ValueError("semantic canonical C0 primary order drifted")
        c0_primary = {
            change: _float_array_antigravity(c0_primary_source[change], (width,), f"c0_primary[{change}]")
            for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
        }
        for name in ("confirmed", "query_early_peak_lags", "query_late_peak_lags", "query_peak_presence", "query_shifts", "query_raw_shift"):
            _require_canonical_scope_mapping_antigravity(item[name], name)
        confirmed = {
            scope: _bool_array_antigravity(item["confirmed"][scope], (width,), f"confirmed[{scope}]")
            for scope in _SCOPES_ANTIGRAVITY
        }
        fit_early = _int_array_antigravity(item["fit_early_peak_lags"], (fit_width,), "fit_early_peak_lags")
        fit_late = _int_array_antigravity(item["fit_late_peak_lags"], (fit_width,), "fit_late_peak_lags")
        fit_presence = _bool_array_antigravity(item["fit_peak_presence"], (fit_width, 2), "fit_peak_presence")
        fit_shifts = _int_array_antigravity(item["fit_shifts"], (fit_width,), "fit_shifts")
        fit_raw = _float_array_antigravity(item["fit_raw_shift"], (fit_width,), "fit_raw_shift")
        maximum_value = item["fit_normal_max_raw_shift"]
        if type(maximum_value) is not float:
            raise ValueError("fit normal maximum must be exact JSON float")
        maximum = float(maximum_value)
        early = {scope: _int_array_antigravity(item["query_early_peak_lags"][scope], (width,), f"early[{scope}]") for scope in _SCOPES_ANTIGRAVITY}
        late = {scope: _int_array_antigravity(item["query_late_peak_lags"][scope], (width,), f"late[{scope}]") for scope in _SCOPES_ANTIGRAVITY}
        presence = {scope: _bool_array_antigravity(item["query_peak_presence"][scope], (width, 2), f"presence[{scope}]") for scope in _SCOPES_ANTIGRAVITY}
        shifts = {scope: _int_array_antigravity(item["query_shifts"][scope], (width,), f"shifts[{scope}]") for scope in _SCOPES_ANTIGRAVITY}
        raw = {scope: _float_array_antigravity(item["query_raw_shift"][scope], (width,), f"raw[{scope}]") for scope in _SCOPES_ANTIGRAVITY}
        stored_e12_ranks = _float_array_antigravity(item["e12_ranks"], (width,), "e12_ranks")
        generated_rows = _scope_rows_antigravity(train, query)
        replayed_peaks = score_acf_peak_lag_numba_antigravity(
            train[fit],
            np.concatenate(tuple(generated_rows[scope] for scope in _SCOPES_ANTIGRAVITY), axis=0),
        )
        for scope in _SCOPES_ANTIGRAVITY:
            if not np.array_equal(shifts[scope], late[scope] - early[scope]):
                raise ValueError(f"semantic query shift differs from peak lags: {scope}")
            if not np.array_equal(presence[scope][:, 0], early[scope] != 0) or not np.array_equal(presence[scope][:, 1], late[scope] != 0):
                raise ValueError(f"semantic query peak presence differs from peak lags: {scope}")
            position = _SCOPES_ANTIGRAVITY.index(scope)
            start = position * width
            stop = (position + 1) * width
            peak_checks = (
                ("early", early[scope], replayed_peaks.query_early_peak_lags[start:stop]),
                ("late", late[scope], replayed_peaks.query_late_peak_lags[start:stop]),
                ("presence", presence[scope], replayed_peaks.query_peak_presence[start:stop]),
                ("shift", shifts[scope], replayed_peaks.query_shifts[start:stop]),
                ("raw", raw[scope], replayed_peaks.query_raw_shift[start:stop]),
                ("confirmation", confirmed[scope], replayed_peaks.confirmation_mask[start:stop]),
            )
            for name, stored, replayed in peak_checks:
                if stored.tobytes(order="C") != np.asarray(replayed).tobytes(order="C"):
                    raise ValueError(f"semantic peak/shift evidence drifted: {name} {scope}")
        fit_checks = (
            (fit_early, replayed_peaks.fit_early_peak_lags),
            (fit_late, replayed_peaks.fit_late_peak_lags),
            (fit_presence, replayed_peaks.fit_peak_presence),
            (fit_shifts, replayed_peaks.fit_shifts),
            (fit_raw, replayed_peaks.fit_raw_shift),
        )
        fit_names = ("fit_early", "fit_late", "fit_presence", "fit_shift", "fit_raw")
        for name, (stored, replayed) in zip(fit_names, fit_checks, strict=True):
            if stored.tobytes(order="C") != replayed.tobytes(order="C"):
                raise ValueError(f"semantic {name} evidence drifted")
        ranks: dict[str, np.ndarray] = {}
        recomputed_confirmed: dict[str, np.ndarray] = {}
        for scope in _SCOPES_ANTIGRAVITY:
            replay_fit_raw, replay_raw, rank, replay_confirmed, replay_maximum = _assemble_evidence(
                fit_shifts.astype(np.float64),
                shifts[scope].astype(np.float64),
                fit_raw,
                raw[scope],
            )
            if maximum != float(replayed_peaks.fit_normal_max_raw_shift):
                raise ValueError("semantic retained fit maximum differs")
            ranks[scope] = rank
            recomputed_confirmed[scope] = replay_confirmed
        if stored_e12_ranks.tobytes(order="C") != ranks["normal"].tobytes(order="C"):
            raise ValueError("semantic retained E12 rank evidence drifted")
        original = {
            "C0": c0_normal,
            "E12": ranks["normal"],
            "C12": _combine_c12_numba_antigravity(c0_normal, ranks["normal"], recomputed_confirmed["normal"]),
        }
        primary = {
            "C0": MappingProxyType(c0_primary),
            "E12": MappingProxyType({change: ranks[change] for change in FROZEN_CONFIG_ANTIGRAVITY.change_types}),
            "C12": MappingProxyType({
                change: _combine_c12_numba_antigravity(c0_primary[change], ranks[change], recomputed_confirmed[change])
                for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
            }),
        }
        record = FoldMethodScoresAntigravity(
            seed=identity[0],
            fold_index=identity[1],
            methods=config.method_order,
            original=original,
            primary=primary,
            fit_early_peak_lags=fit_early,
            fit_late_peak_lags=fit_late,
            fit_peak_presence=fit_presence,
            fit_shifts=fit_shifts,
            fit_raw_shift=fit_raw,
            query_early_peak_lags=early,
            query_late_peak_lags=late,
            query_peak_presence=presence,
            query_shifts=shifts,
            query_raw_shift=raw,
            fit_normal_max_raw_shift=maximum,
            confirmed=recomputed_confirmed,
            input_train140=train,
            input_fit_indices=fit,
            input_query_indices=query,
            backend="numba",
        )
        records[identity] = record
    return MappingProxyType(records)


def _assemble_evidence(
    fit_shifts: np.ndarray,
    query_shifts: np.ndarray,
    stored_fit_raw: np.ndarray,
    stored_query_raw: np.ndarray,
):
    from .acf_peak_lag_antigravity import assemble_raw_rank_numba_antigravity

    fit_raw, query_raw, ranks, confirmed, maximum = assemble_raw_rank_numba_antigravity(
        fit_shifts,
        query_shifts,
    )
    if stored_fit_raw.tobytes(order="C") != fit_raw.tobytes(order="C") or stored_query_raw.tobytes(order="C") != query_raw.tobytes(order="C"):
        raise ValueError("semantic shift replay disagrees with stored raw evidence")
    return fit_raw, query_raw, ranks, confirmed, maximum


@dataclass(frozen=True, slots=True)
class InspectedResultAntigravity:
    root: Path
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity]
    qualification: Exp471QualificationAntigravity
    decision: QualificationDecisionAntigravity
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", MappingProxyType(dict(self.records)))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


def inspect_result_antigravity(root: object) -> InspectedResultAntigravity:
    """Verify bytes first, then regenerate peak-derived scores and every aggregate."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    payloads, hashes = _validate_leaf_set_and_hashes_antigravity(root)
    records = _rebuild_records_antigravity(payloads["context"], payloads["metrics"])
    qualification = qualify_records_antigravity(records)
    decision = evaluate_qualification_antigravity(qualification)
    expected_metrics = tuple(
        _metric_row_antigravity(record, result)
        for record, result in zip(records.values(), qualification.fold_results, strict=True)
    )
    semantic_expectations = (
        ("method_fold_metrics_antigravity.jsonl", b"".join(_canonical_json_bytes_antigravity(row) for row in expected_metrics)),
        ("method_summary_antigravity.jsonl", b"".join(_canonical_json_bytes_antigravity(row) for row in _method_summary_rows_antigravity(expected_metrics))),
        ("failed_folds_antigravity.jsonl", b"".join(_canonical_json_bytes_antigravity(row) for row in _failed_rows_antigravity(qualification))),
        ("decision_antigravity.json", _canonical_json_bytes_antigravity(_decision_payload_antigravity(qualification, decision))),
        ("dashboard_snapshot_antigravity.json", _canonical_json_bytes_antigravity(_dashboard_snapshot_payload_antigravity(tuple(records.values()), qualification, decision, _method_summary_rows_antigravity(expected_metrics)))),
    )
    for name, expected in semantic_expectations:
        if (root / name).read_bytes() != expected:
            raise ValueError(f"stored semantic replay evidence drifted: {name}")
    return InspectedResultAntigravity(root=root, records=records, qualification=qualification, decision=decision, hashes=hashes)


__all__ = ("InspectedResultAntigravity", "inspect_result_antigravity")
