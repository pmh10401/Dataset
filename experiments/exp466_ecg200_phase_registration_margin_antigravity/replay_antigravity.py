"""Dataset-free semantic inspection of Exp466 Task 5 artifacts."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp466FoldResultAntigravity,
    Exp466QualificationAntigravity,
)
from .evaluation_antigravity import QualificationDecisionAntigravity


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
_HASHED_LEAVES_ANTIGRAVITY = tuple(
    name for name in _LEAVES_ANTIGRAVITY if name != _LEAVES_ANTIGRAVITY[0]
)
_FLOAT_TOLERANCE_ANTIGRAVITY = 1e-12
_METRIC_KEYS_ANTIGRAVITY = frozenset(
    {
        "schema",
        "seed",
        "fold_index",
        "method",
        "change_order",
        "normal_scores",
        "primary_scores",
        "fit_margins",
        "query_margins",
        "fit_normal_max_margin",
        "confirmed",
        "confirmation_counts",
        "per_type_auc",
        "mean_auc",
        "all_scores_finite",
        "c0_deltas",
        "qualification",
        "input_train140",
        "input_fit_indices",
        "input_query_indices",
    },
)
_QUALIFICATION_KEYS_ANTIGRAVITY = frozenset(
    {
        "seed",
        "fold_index",
        "per_type_auc",
        "mean_auc",
        "c0_gradual_auc",
        "c0_period_auc",
        "c9_gradual_auc",
        "c9_period_auc",
        "period_confirmations",
        "all_scores_finite",
        "passed",
        "failure_reasons",
    },
)


@dataclass(frozen=True, slots=True)
class InspectedResultAntigravity:
    qualification: Exp466QualificationAntigravity
    decision: QualificationDecisionAntigravity
    dashboard: dict[str, object]
    dashboard_html: str
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


@dataclass(slots=True)
class _MetricAntigravity:
    payload: dict[str, object]
    seed: int
    fold_index: int
    method: str
    normal_scores: np.ndarray
    primary_scores: dict[str, np.ndarray]
    fit_margins: np.ndarray
    query_margins: dict[str, np.ndarray]
    fit_normal_max_margin: float
    confirmed: dict[str, np.ndarray]
    confirmation_counts: dict[str, int]
    per_type_auc: tuple[float, float, float, float]
    mean_auc: float
    all_scores_finite: bool
    c0_deltas: dict[str, float]
    qualification_payload: object
    train_raw: bytes
    fit_indices: tuple[int, ...]
    query_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _DerivedMetricAntigravity:
    per_type_auc: tuple[float, float, float, float]
    mean_auc: float
    all_scores_finite: bool
    c0_gradual_delta: float
    c0_period_delta: float
    confirmation_counts: tuple[tuple[str, int], ...]


def _canonical_json_bytes_antigravity(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("artifact JSON value is not canonicalizable") from error


def _strict_json_antigravity(payload: bytes, name: str) -> object:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if payload != _canonical_json_bytes_antigravity(value):
        raise ValueError(f"{name} is not canonical JSON")
    return value


def _strict_jsonl_antigravity(payload: bytes, name: str) -> tuple[object, ...]:
    if not payload:
        return ()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError(f"{name} is not canonical JSONL")
    return tuple(
        _strict_json_antigravity(line + b"\n", name)
        for line in payload.splitlines()
    )


def _load_leaves_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("inspection root must be a real result directory")
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ValueError("inspection root could not be listed") from error
    if tuple(sorted(entry.name for entry in entries)) != _LEAVES_ANTIGRAVITY:
        raise ValueError("artifact leaf inventory differs from seven-leaf schema")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("every artifact leaf must be a regular file")
    try:
        leaves = {name: (root / name).read_bytes() for name in _LEAVES_ANTIGRAVITY}
    except OSError as error:
        raise ValueError("artifact leaf could not be read") from error

    map_value = _strict_json_antigravity(
        leaves["artifact_sha256_antigravity.json"],
        "artifact hash map",
    )
    if type(map_value) is not dict or tuple(map_value) != _HASHED_LEAVES_ANTIGRAVITY:
        raise ValueError("artifact hash map must list exactly the other six leaves")
    for name in _HASHED_LEAVES_ANTIGRAVITY:
        digest = map_value[name]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("artifact hash map contains an invalid digest")
        if digest != sha256(leaves[name]).hexdigest():
            raise ValueError("artifact hash map differs from current leaf bytes")
    return leaves


def _exact_dict_antigravity(
    value: object,
    keys: frozenset[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or frozenset(value) != keys:
        raise ValueError(f"{name} has an invalid object schema")
    return value


def _exact_int_antigravity(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _exact_bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact boolean")
    return value


def _finite_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _float_vector_antigravity(
    value: object,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a non-empty float list")
    result = np.asarray(
        tuple(_finite_float_antigravity(item, name) for item in value),
        dtype=np.float64,
    )
    if length is not None and result.shape != (length,):
        raise ValueError(f"{name} has the wrong length")
    return result


def _bool_vector_antigravity(
    value: object,
    name: str,
    *,
    length: int,
) -> np.ndarray:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{name} must be a matching boolean list")
    return np.asarray(
        tuple(_exact_bool_antigravity(item, name) for item in value),
        dtype=np.bool_,
    )


def _index_vector_antigravity(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    result = tuple(_exact_int_antigravity(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    if min(result) < 0 or max(result) >= FROZEN_CONFIG_ANTIGRAVITY.normal_rows:
        raise ValueError(f"{name} contains an out-of-range value")
    return result


def _train_blob_antigravity(value: object) -> bytes:
    blob = _exact_dict_antigravity(
        value,
        frozenset({"data", "encoding", "shape", "sha256"}),
        "input_train140",
    )
    if blob["encoding"] != "base64_float64_le_c_v1":
        raise ValueError("input_train140 uses an unknown encoding")
    if blob["shape"] != [
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
        FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
    ]:
        raise ValueError("input_train140 has the wrong shape")
    data = blob["data"]
    digest = blob["sha256"]
    if type(data) is not str:
        raise ValueError("input_train140 data must be base64 text")
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("input_train140 digest is invalid")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("input_train140 base64 is invalid") from error
    if base64.b64encode(raw).decode("ascii") != data:
        raise ValueError("input_train140 base64 is not canonical")
    expected_size = (
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows
        * FROZEN_CONFIG_ANTIGRAVITY.interpolated_width
        * np.dtype("<f8").itemsize
    )
    if len(raw) != expected_size or sha256(raw).hexdigest() != digest:
        raise ValueError("input_train140 bytes differ from shape or digest")
    train = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
        FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
    )
    if not np.isfinite(train).all():
        raise ValueError("input_train140 contains a non-finite value")
    return raw


def _metric_antigravity(
    value: object,
    expected_identity: tuple[int, int, str],
) -> _MetricAntigravity:
    payload = _exact_dict_antigravity(value, _METRIC_KEYS_ANTIGRAVITY, "metric row")
    seed = _exact_int_antigravity(payload["seed"], "metric seed")
    fold_index = _exact_int_antigravity(payload["fold_index"], "metric fold")
    method = payload["method"]
    if type(method) is not str or (seed, fold_index, method) != expected_identity:
        raise ValueError("metric rows must preserve exact seed-major method order")
    if payload["schema"] != "exp466_method_fold_metric_antigravity_v1":
        raise ValueError("metric row schema differs")
    if payload["change_order"] != list(_CHANGE_TYPES_ANTIGRAVITY):
        raise ValueError("metric row change order differs")

    fit_indices = _index_vector_antigravity(
        payload["input_fit_indices"],
        "input fit indices",
    )
    query_indices = _index_vector_antigravity(
        payload["input_query_indices"],
        "input query indices",
    )
    if set(fit_indices) & set(query_indices) or set(fit_indices) | set(
        query_indices,
    ) != set(range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows)):
        raise ValueError("fit/query indices must partition all 69 retained rows")
    expected_query = tuple(
        int(item)
        for item in np.array_split(
            np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64),
            FROZEN_CONFIG_ANTIGRAVITY.folds,
        )[fold_index]
    )
    expected_query_set = set(expected_query)
    expected_fit = tuple(
        item
        for item in range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows)
        if item not in expected_query_set
    )
    if query_indices != expected_query or fit_indices != expected_fit:
        raise ValueError("retained indices differ from exact blocked fold")
    query_length = len(query_indices)
    normal_scores = _float_vector_antigravity(
        payload["normal_scores"],
        "normal scores",
        length=query_length,
    )
    primary_value = _exact_dict_antigravity(
        payload["primary_scores"],
        frozenset(_CHANGE_TYPES_ANTIGRAVITY),
        "primary scores",
    )
    primary_scores = {
        name: _float_vector_antigravity(
            primary_value[name],
            f"primary scores {name}",
            length=query_length,
        )
        for name in _CHANGE_TYPES_ANTIGRAVITY
    }
    fit_margins = _float_vector_antigravity(
        payload["fit_margins"],
        "fit margins",
        length=len(fit_indices),
    )
    query_value = _exact_dict_antigravity(
        payload["query_margins"],
        frozenset(_SCOPES_ANTIGRAVITY),
        "query margins",
    )
    query_margins = {
        scope: _float_vector_antigravity(
            query_value[scope],
            f"query margins {scope}",
            length=query_length,
        )
        for scope in _SCOPES_ANTIGRAVITY
    }
    fit_normal_max_margin = _finite_float_antigravity(
        payload["fit_normal_max_margin"],
        "fit normal maximum margin",
    )
    confirmed_value = _exact_dict_antigravity(
        payload["confirmed"],
        frozenset(_SCOPES_ANTIGRAVITY),
        "confirmations",
    )
    confirmed = {
        scope: _bool_vector_antigravity(
            confirmed_value[scope],
            f"confirmations {scope}",
            length=query_length,
        )
        for scope in _SCOPES_ANTIGRAVITY
    }
    counts_value = _exact_dict_antigravity(
        payload["confirmation_counts"],
        frozenset(_SCOPES_ANTIGRAVITY),
        "confirmation counts",
    )
    confirmation_counts = {
        scope: _exact_int_antigravity(
            counts_value[scope],
            f"confirmation count {scope}",
        )
        for scope in _SCOPES_ANTIGRAVITY
    }
    if any(count < 0 for count in confirmation_counts.values()):
        raise ValueError("confirmation counts must be non-negative")
    per_type_value = payload["per_type_auc"]
    if type(per_type_value) is not list or len(per_type_value) != len(
        _CHANGE_TYPES_ANTIGRAVITY,
    ):
        raise ValueError("per-type AUC must contain four values")
    per_type_auc = tuple(
        _finite_float_antigravity(item, "per-type AUC") for item in per_type_value
    )
    mean_auc = _finite_float_antigravity(payload["mean_auc"], "mean AUC")
    all_scores_finite = _exact_bool_antigravity(
        payload["all_scores_finite"],
        "all scores finite",
    )
    delta_value = _exact_dict_antigravity(
        payload["c0_deltas"],
        frozenset({"gradual_drift", "period_change"}),
        "C0 deltas",
    )
    c0_deltas = {
        name: _finite_float_antigravity(delta_value[name], f"C0 delta {name}")
        for name in ("gradual_drift", "period_change")
    }
    return _MetricAntigravity(
        payload=payload,
        seed=seed,
        fold_index=fold_index,
        method=method,
        normal_scores=normal_scores,
        primary_scores=primary_scores,
        fit_margins=fit_margins,
        query_margins=query_margins,
        fit_normal_max_margin=fit_normal_max_margin,
        confirmed=confirmed,
        confirmation_counts=confirmation_counts,
        per_type_auc=per_type_auc,
        mean_auc=mean_auc,
        all_scores_finite=all_scores_finite,
        c0_deltas=c0_deltas,
        qualification_payload=payload["qualification"],
        train_raw=_train_blob_antigravity(payload["input_train140"]),
        fit_indices=fit_indices,
        query_indices=query_indices,
    )


def _close_antigravity(left: float, right: float) -> bool:
    return abs(left - right) <= _FLOAT_TOLERANCE_ANTIGRAVITY


def _array_close_antigravity(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(
        np.all(np.abs(left - right) <= _FLOAT_TOLERANCE_ANTIGRAVITY),
    )


def _payloads_equivalent_antigravity(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return isfinite(left) and isfinite(right) and _close_antigravity(left, right)
    if type(left) is list:
        return len(left) == len(right) and all(
            _payloads_equivalent_antigravity(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if type(left) is dict:
        return frozenset(left) == frozenset(right) and all(
            _payloads_equivalent_antigravity(left[key], right[key]) for key in left
        )
    return left == right


def _auc_antigravity(normal: np.ndarray, anomaly: np.ndarray) -> float:
    wins = int(np.count_nonzero(anomaly[:, None] > normal[None, :]))
    ties = int(np.count_nonzero(anomaly[:, None] == normal[None, :]))
    return float((wins + 0.5 * ties) / (normal.size * anomaly.size))


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


def _qualification_payload_antigravity(
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


def _derived_metrics_antigravity(
    group: tuple[_MetricAntigravity, _MetricAntigravity, _MetricAntigravity],
) -> tuple[
    dict[str, _DerivedMetricAntigravity],
    Exp466FoldResultAntigravity,
]:
    c0, p9, c9 = group
    shared_fields = (
        "fit_margins",
        "query_margins",
        "fit_normal_max_margin",
        "confirmed",
        "confirmation_counts",
        "input_train140",
        "input_fit_indices",
        "input_query_indices",
    )
    for field in shared_fields:
        if p9.payload[field] != c0.payload[field] or c9.payload[field] != c0.payload[field]:
            raise ValueError("fold rows disagree on retained shared evidence")
    if not _close_antigravity(
        c0.fit_normal_max_margin,
        float(np.max(c0.fit_margins)),
    ):
        raise ValueError("fit maximum differs from retained fit margins")

    expected_confirmed: dict[str, np.ndarray] = {}
    expected_p9: dict[str, np.ndarray] = {}
    for scope in _SCOPES_ANTIGRAVITY:
        margins = c0.query_margins[scope]
        expected_p9[scope] = np.asarray(
            (
                1.0
                + np.count_nonzero(
                    c0.fit_margins[:, None] <= margins[None, :],
                    axis=0,
                )
            )
            / (len(c0.fit_margins) + 1.0),
            dtype=np.float64,
        )
        expected_confirmed[scope] = (margins > 0.0) & (
            margins > c0.fit_normal_max_margin
        )
        if not np.array_equal(c0.confirmed[scope], expected_confirmed[scope]):
            raise ValueError("confirmation differs from strict positive fit-max replay")
        stored_p9 = (
            p9.normal_scores if scope == "normal" else p9.primary_scores[scope]
        )
        if not _array_close_antigravity(stored_p9, expected_p9[scope]):
            raise ValueError("P9 rank differs from retained-margin replay")
        stored_c0 = (
            c0.normal_scores if scope == "normal" else c0.primary_scores[scope]
        )
        stored_c9 = (
            c9.normal_scores if scope == "normal" else c9.primary_scores[scope]
        )
        expected_c9 = np.where(
            expected_confirmed[scope],
            np.maximum(stored_c0, stored_p9),
            stored_c0,
        )
        if not _array_close_antigravity(stored_c9, expected_c9):
            raise ValueError("C9 differs from retained C0/P9/confirmation replay")

    expected_counts = {
        scope: int(np.count_nonzero(expected_confirmed[scope]))
        for scope in _SCOPES_ANTIGRAVITY
    }
    derived_auc: dict[str, tuple[tuple[float, float, float, float], float, bool]] = {}
    for metric in group:
        finite = bool(
            np.isfinite(metric.normal_scores).all()
            and all(np.isfinite(values).all() for values in metric.primary_scores.values())
        )
        if finite:
            per_type = tuple(
                _auc_antigravity(metric.normal_scores, metric.primary_scores[name])
                for name in _CHANGE_TYPES_ANTIGRAVITY
            )
            mean_auc = float(fsum(per_type) / len(per_type))
        else:
            per_type = (0.0, 0.0, 0.0, 0.0)
            mean_auc = 0.0
        derived_auc[metric.method] = (per_type, mean_auc, finite)
    c0_auc = derived_auc["C0"][0]
    derived: dict[str, _DerivedMetricAntigravity] = {}
    for metric in group:
        per_type, mean_auc, finite = derived_auc[metric.method]
        expected = _DerivedMetricAntigravity(
            per_type_auc=per_type,
            mean_auc=mean_auc,
            all_scores_finite=finite,
            c0_gradual_delta=per_type[1] - c0_auc[1],
            c0_period_delta=per_type[3] - c0_auc[3],
            confirmation_counts=tuple(
                (scope, expected_counts[scope]) for scope in _SCOPES_ANTIGRAVITY
            ),
        )
        if not all(
            _close_antigravity(stored, actual)
            for stored, actual in zip(metric.per_type_auc, per_type, strict=True)
        ):
            raise ValueError("stored AUC differs from score replay")
        if not _close_antigravity(metric.mean_auc, mean_auc):
            raise ValueError("stored mean AUC differs from four-type replay")
        if metric.all_scores_finite is not finite:
            raise ValueError("stored finite flag differs from score replay")
        if metric.confirmation_counts != expected_counts:
            raise ValueError("stored confirmation counts differ from evidence replay")
        if not _close_antigravity(
            metric.c0_deltas["gradual_drift"],
            expected.c0_gradual_delta,
        ) or not _close_antigravity(
            metric.c0_deltas["period_change"],
            expected.c0_period_delta,
        ):
            raise ValueError("stored C0 deltas differ from AUC replay")
        derived[metric.method] = expected

    c9_auc, c9_mean, c9_finite = derived_auc["C9"]
    reasons = _failure_reasons_antigravity(
        per_type_auc=c9_auc,
        mean_auc=c9_mean,
        all_scores_finite=c9_finite,
        c0_gradual_auc=c0_auc[1],
        c0_period_auc=c0_auc[3],
        c9_gradual_auc=c9_auc[1],
        c9_period_auc=c9_auc[3],
        period_confirmations=expected_counts["period_change"],
        normal_confirmations=expected_counts["normal"],
    )
    result = Exp466FoldResultAntigravity(
        seed=c0.seed,
        fold_index=c0.fold_index,
        per_type_auc=c9_auc,
        mean_auc=c9_mean,
        c0_gradual_auc=c0_auc[1],
        c0_period_auc=c0_auc[3],
        c9_gradual_auc=c9_auc[1],
        c9_period_auc=c9_auc[3],
        period_confirmations=expected_counts["period_change"],
        all_scores_finite=c9_finite,
        passed=not reasons,
        failure_reasons=reasons,
    )
    if c0.qualification_payload is not None or p9.qualification_payload is not None:
        raise ValueError("only the C9 metric row may carry qualification evidence")
    if type(c9.qualification_payload) is not dict or frozenset(
        c9.qualification_payload,
    ) != _QUALIFICATION_KEYS_ANTIGRAVITY:
        raise ValueError("C9 qualification row schema differs")
    if not _payloads_equivalent_antigravity(
        c9.qualification_payload,
        _qualification_payload_antigravity(result),
    ):
        raise ValueError("stored qualification differs from semantic replay")
    return derived, result


def _summary_payloads_antigravity(
    derived_rows: tuple[
        tuple[dict[str, _DerivedMetricAntigravity], Exp466FoldResultAntigravity],
        ...,
    ],
) -> tuple[dict[str, object], ...]:
    summaries: list[dict[str, object]] = []
    for method in _METHODS_ANTIGRAVITY:
        selected = tuple(item[0][method] for item in derived_rows)
        summaries.append(
            {
                "schema": "exp466_method_summary_antigravity_v1",
                "method": method,
                "fold_count": len(selected),
                "min_per_type_auc": [
                    min(item.per_type_auc[index] for item in selected)
                    for index in range(len(_CHANGE_TYPES_ANTIGRAVITY))
                ],
                "min_mean_auc": min(item.mean_auc for item in selected),
                "normal_confirmation_count": sum(
                    dict(item.confirmation_counts)["normal"] for item in selected
                ),
                "period_confirmation_count": sum(
                    dict(item.confirmation_counts)["period_change"] for item in selected
                ),
                "all_scores_finite": all(item.all_scores_finite for item in selected),
                "c0_gradual_delta_min": min(
                    item.c0_gradual_delta for item in selected
                ),
                "c0_period_delta_min": min(item.c0_period_delta for item in selected),
                "qualification_passed_count": (
                    sum(item[1].passed for item in derived_rows)
                    if method == "C9"
                    else None
                ),
                "qualification_failed_count": (
                    sum(not item[1].passed for item in derived_rows)
                    if method == "C9"
                    else None
                ),
            },
        )
    return tuple(summaries)


def _decision_antigravity(
    qualification: Exp466QualificationAntigravity,
) -> QualificationDecisionAntigravity:
    passed_count = sum(row.passed for row in qualification.fold_results)
    reasons: list[str] = []
    if passed_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
        reasons.append("c9_not_10_of_10")
    for row in qualification.fold_results:
        for reason in row.failure_reasons:
            if reason not in reasons:
                reasons.append(reason)
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if qualification.normal_confirmations[seed] != 0:
            reason = f"seed_{seed}_normal_confirmations_not_zero"
            if reason not in reasons:
                reasons.append(reason)
    eligible = not reasons
    return QualificationDecisionAntigravity(
        status=(
            "qualification_passed_awaiting_exp467_design"
            if eligible
            else "qualification_failed_blocked_before_fn_validation"
        ),
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        eligible_for_exp467_design=eligible,
        primary_passed_count=passed_count,
        primary_required_count=FROZEN_CONFIG_ANTIGRAVITY.required_fold_results,
        failure_reasons=tuple(reasons),
        routing_applied=False,
        champion_modified=False,
        pending_modified=False,
        promotion_applied=False,
        test_labels_opened=False,
        label_query_count=0,
    )


def _decision_payload_antigravity(
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "status": decision.status,
        "primary_candidate": decision.primary_candidate,
        "eligible_for_exp467_design": decision.eligible_for_exp467_design,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "failure_reasons": list(decision.failure_reasons),
        "routing_applied": decision.routing_applied,
        "champion_modified": decision.champion_modified,
        "pending_modified": decision.pending_modified,
        "promotion_applied": decision.promotion_applied,
        "test_labels_opened": decision.test_labels_opened,
        "label_query_count": decision.label_query_count,
    }


def _dashboard_antigravity(
    qualification: Exp466QualificationAntigravity,
) -> dict[str, object]:
    improved = tuple(
        f"{row.seed}:{row.fold_index}"
        for row in qualification.fold_results
        if row.c9_period_auc > row.c0_period_auc
    )
    failed = tuple(
        f"{row.seed}:{row.fold_index}"
        for row in qualification.fold_results
        if not row.passed
    )
    normal_confirmations = sum(qualification.normal_confirmations.values())
    failed_sentence = (
        f"실패한 검증 묶음은 {', '.join(failed)}입니다."
        if failed
        else "실패한 검증 묶음은 없습니다."
    )
    html = (
        '<!doctype html>\n<html lang="ko">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Exp466 Task 5 C9 확인 요약</title>\n"
        "</head>\n<body>\n"
        "<main>\n"
        "<h1>Exp466 Task 5 C9 확인 요약</h1>\n"
        f"<p>C9의 주기 AUC가 C0보다 좋아진 검증 묶음은 {len(improved)}개입니다.</p>\n"
        f"<p>{failed_sentence}</p>\n"
        f"<p>정상 오탐 확인은 {normal_confirmations}개입니다.</p>\n"
        "<p>Task 5에는 실제 비공개 또는 공식 결과가 없습니다.</p>\n"
        "<p>이 합성 값은 성능 결과가 아닙니다.</p>\n"
        "</main>\n"
        "</body>\n</html>\n"
    )
    return {
        "후보": "C9",
        "주기_AUC_개선_fold_수": len(improved),
        "주기_AUC_개선_fold": list(improved),
        "실패_fold": list(failed),
        "정상_confirmation_수": normal_confirmations,
        "Task5_실제_private_official_결과": False,
        "html_antigravity": html,
    }


def _context_antigravity(
    qualification: Exp466QualificationAntigravity,
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


def _inspect_antigravity(root: Path) -> InspectedResultAntigravity:
    leaves = _load_leaves_antigravity(root)
    context_value = _strict_json_antigravity(
        leaves["context_antigravity.json"],
        "context artifact",
    )
    stored_dashboard = _strict_json_antigravity(
        leaves["dashboard_snapshot_antigravity.json"],
        "dashboard artifact",
    )
    stored_decision = _strict_json_antigravity(
        leaves["decision_antigravity.json"],
        "decision artifact",
    )
    failed_values = _strict_jsonl_antigravity(
        leaves["failed_folds_antigravity.jsonl"],
        "failed-fold artifact",
    )
    metric_values = _strict_jsonl_antigravity(
        leaves["method_fold_metrics_antigravity.jsonl"],
        "metric artifact",
    )
    summary_values = _strict_jsonl_antigravity(
        leaves["method_summary_antigravity.jsonl"],
        "summary artifact",
    )
    if type(context_value) is not dict:
        raise ValueError("context artifact must be an object")
    metrics_digest = sha256(
        leaves["method_fold_metrics_antigravity.jsonl"],
    ).hexdigest()
    if context_value.get("method_fold_metrics_sha256") != metrics_digest:
        raise ValueError("context metric digest differs from metric bytes")

    expected_identities = tuple(
        (seed, fold_index, method)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        for method in _METHODS_ANTIGRAVITY
    )
    if len(metric_values) != len(expected_identities):
        raise ValueError("metric artifact must contain exactly thirty rows")
    metrics = tuple(
        _metric_antigravity(value, identity)
        for value, identity in zip(metric_values, expected_identities, strict=True)
    )
    first_train = metrics[0].train_raw
    if any(metric.train_raw != first_train for metric in metrics[1:]):
        raise ValueError("metric rows retain different train140 bytes")

    derived_rows: list[
        tuple[dict[str, _DerivedMetricAntigravity], Exp466FoldResultAntigravity]
    ] = []
    oof: dict[int, list[int]] = {
        seed: [] for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
    }
    normal_confirmations = {
        seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
    }
    for offset in range(0, len(metrics), len(_METHODS_ANTIGRAVITY)):
        group = metrics[offset : offset + len(_METHODS_ANTIGRAVITY)]
        if len(group) != 3:
            raise ValueError("metric rows do not form complete method groups")
        typed_group = (group[0], group[1], group[2])
        derived, result = _derived_metrics_antigravity(typed_group)
        derived_rows.append((derived, result))
        oof[result.seed].extend(group[0].query_indices)
        normal_confirmations[result.seed] += group[0].confirmation_counts["normal"]

    expected_trials = list(range(FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed))
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if sorted(oof[seed]) != expected_trials or len(set(oof[seed])) != len(
            expected_trials,
        ):
            raise ValueError("seed OOF indices do not cover 69 unique retained rows")
    qualification = Exp466QualificationAntigravity(
        fold_results=tuple(item[1] for item in derived_rows),
        normal_trial_counts=MappingProxyType(
            {
                seed: len(set(oof[seed]))
                for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            },
        ),
        normal_confirmations=MappingProxyType(normal_confirmations),
        label_query_count=0,
        official_evaluate_count=0,
        official_inspect_count=0,
        artifact_map_sha256=None,
    )

    expected_summaries = _summary_payloads_antigravity(tuple(derived_rows))
    if len(summary_values) != len(expected_summaries) or any(
        not _payloads_equivalent_antigravity(stored, expected)
        for stored, expected in zip(summary_values, expected_summaries, strict=True)
    ):
        raise ValueError("method summaries differ from semantic replay")
    expected_failed = tuple(
        {
            "schema": "exp466_failed_fold_antigravity_v1",
            **_qualification_payload_antigravity(row),
        }
        for row in qualification.fold_results
        if not row.passed
    )
    if len(failed_values) != len(expected_failed) or any(
        not _payloads_equivalent_antigravity(stored, expected)
        for stored, expected in zip(failed_values, expected_failed, strict=True)
    ):
        raise ValueError("failed-fold rows differ from semantic replay")

    expected_context = _context_antigravity(qualification, metrics_digest)
    if not _payloads_equivalent_antigravity(context_value, expected_context):
        raise ValueError("context differs from semantic replay")
    decision = _decision_antigravity(qualification)
    expected_decision = _decision_payload_antigravity(decision)
    if not _payloads_equivalent_antigravity(stored_decision, expected_decision):
        raise ValueError("decision differs from semantic replay")
    dashboard = _dashboard_antigravity(qualification)
    if not _payloads_equivalent_antigravity(stored_dashboard, dashboard):
        raise ValueError("dashboard differs from semantic replay")
    dashboard_html = dashboard["html_antigravity"]
    if type(dashboard_html) is not str:
        raise ValueError("dashboard HTML must be text")
    leaf_sha256 = tuple(
        (name, sha256(leaves[name]).hexdigest()) for name in _LEAVES_ANTIGRAVITY
    )
    return InspectedResultAntigravity(
        qualification=qualification,
        decision=decision,
        dashboard=dashboard,
        dashboard_html=dashboard_html,
        leaf_sha256=leaf_sha256,
        artifact_map_sha256=sha256(
            leaves["artifact_sha256_antigravity.json"],
        ).hexdigest(),
    )


def inspect_result_antigravity(root: Path) -> InspectedResultAntigravity:
    """Inspect only canonical retained leaves and reconstruct every derived value."""

    try:
        return _inspect_antigravity(root)
    except ValueError:
        raise
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("artifact schema is invalid") from error


__all__ = ("InspectedResultAntigravity", "inspect_result_antigravity")
