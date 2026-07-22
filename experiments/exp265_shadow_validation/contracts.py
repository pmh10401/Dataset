from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from .config import COLLECTION_MODE, FORBIDDEN_FIELDS, POLICY_VERSION, SCHEMA_VERSION, SOURCE_NAMES


class ContractError(ValueError):
    """Raised when an upstream run is not safe for prospective shadow use."""


@dataclass(frozen=True)
class SourceEvidence:
    candidate_indices: frozenset[int]
    scores: Mapping[int, float]
    threshold: float


@dataclass(frozen=True)
class SeriesPreview:
    indices: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class ShadowRunInput:
    schema_version: int
    collection_mode: str
    policy_version: str
    run_id: str
    source_id: str
    observed_at: str
    dataset_key: str
    equipment_id: str | None
    recipe_id: str | None
    sensor_id: str | None
    step_id: str | None
    n_train: int
    n_test: int
    baseline_hard: frozenset[int]
    baseline_standard: frozenset[int]
    baseline_priority: frozenset[int]
    sources: Mapping[str, SourceEvidence]
    quality_flags: frozenset[int]
    has_missing: bool
    missing_quality_status: str
    config_sha256: str
    series_preview: SeriesPreview


ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "collection_mode",
        "policy_version",
        "run_id",
        "source_id",
        "observed_at",
        "dataset_key",
        "equipment_id",
        "recipe_id",
        "sensor_id",
        "step_id",
        "n_train",
        "n_test",
        "baseline_hard_indices",
        "baseline_standard_indices",
        "baseline_priority_indices",
        "source_evidence",
        "quality_flag_indices",
        "has_missing",
        "missing_quality_status",
        "config_sha256",
        "series_preview",
    }
)


def _scan_forbidden(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ContractError(f"forbidden field: {path}.{key}")
            _scan_forbidden(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, f"{path}[{index}]")


def _string(value: object, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ContractError(f"{field} must be finite")
    return float(value)


def _indices(value: object, field: str, n_test: int) -> frozenset[int]:
    if type(value) is not list:
        raise ContractError(f"{field} must be a list")
    if any(type(index) is not int for index in value):
        raise ContractError(f"{field} contains non-integral index")
    if len(value) != len(set(value)):
        raise ContractError(f"{field} contains duplicate index")
    if any(index < 0 or index >= n_test for index in value):
        raise ContractError(f"{field} contains candidate outside run range")
    return frozenset(value)


def _utc_timestamp(value: object) -> str:
    text = _string(value, "observed_at")
    assert text is not None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("observed_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError("observed_at must use UTC")
    return parsed.astimezone(timezone.utc).isoformat()


def _source_evidence(value: object, n_test: int) -> Mapping[str, SourceEvidence]:
    if type(value) is not dict or set(value) != set(SOURCE_NAMES):
        raise ContractError("source_evidence must contain exactly the four frozen sources")
    result: dict[str, SourceEvidence] = {}
    for name in SOURCE_NAMES:
        raw = value[name]
        if type(raw) is not dict or set(raw) != {"candidate_indices", "scores", "threshold"}:
            raise ContractError(f"invalid source evidence schema: {name}")
        indices = _indices(raw["candidate_indices"], f"{name}.candidate_indices", n_test)
        threshold = _finite(raw["threshold"], f"{name}.threshold")
        scores_raw = raw["scores"]
        if type(scores_raw) is not dict:
            raise ContractError(f"{name}.scores must be an object")
        try:
            score_keys = {int(key) for key in scores_raw}
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{name}.scores has invalid index key") from exc
        if score_keys != set(indices) or any(str(index) not in scores_raw for index in indices):
            raise ContractError(f"{name}.scores must match candidate indices")
        scores = {index: _finite(scores_raw[str(index)], f"{name}.scores[{index}]") for index in indices}
        if any(score < threshold for score in scores.values()):
            raise ContractError(f"{name}.candidate score is below threshold")
        result[name] = SourceEvidence(indices, MappingProxyType(scores), threshold)
    return MappingProxyType(result)


def _series_preview(value: object, n_test: int, candidates: set[int]) -> SeriesPreview:
    if type(value) is not dict or set(value) != {"indices", "values"}:
        raise ContractError("series_preview must contain indices and values")
    raw_indices = value["indices"]
    raw_values = value["values"]
    if type(raw_indices) is not list or type(raw_values) is not list:
        raise ContractError("series_preview arrays must be lists")
    if len(raw_indices) != len(raw_values) or not raw_indices or len(raw_indices) > 1024:
        raise ContractError("series_preview arrays must have equal length between 1 and 1024")
    if any(type(index) is not int for index in raw_indices):
        raise ContractError("series_preview indices must be integers")
    if raw_indices != sorted(set(raw_indices)) or any(index < 0 or index >= n_test for index in raw_indices):
        raise ContractError("series_preview indices must be unique, sorted, and in range")
    if not candidates <= set(raw_indices):
        raise ContractError("series preview omits candidate")
    return SeriesPreview(tuple(raw_indices), tuple(_finite(item, "series_preview value") for item in raw_values))


def validate_run_input(raw: Mapping[str, object]) -> ShadowRunInput:
    if type(raw) is not dict:
        raise ContractError("run input must be an object")
    _scan_forbidden(raw)
    unknown = set(raw) - ALLOWED_FIELDS
    missing = ALLOWED_FIELDS - set(raw)
    if unknown:
        raise ContractError(f"unknown field: {sorted(unknown)[0]}")
    if missing:
        raise ContractError(f"missing field: {sorted(missing)[0]}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ContractError("unsupported schema_version")
    if raw["collection_mode"] != COLLECTION_MODE:
        raise ContractError("collection_mode must be prospective_shadow")
    if raw["policy_version"] != POLICY_VERSION:
        raise ContractError("unsupported policy_version")
    n_train = _integer(raw["n_train"], "n_train")
    n_test = _integer(raw["n_test"], "n_test", minimum=1)
    hard = _indices(raw["baseline_hard_indices"], "baseline_hard_indices", n_test)
    standard = _indices(raw["baseline_standard_indices"], "baseline_standard_indices", n_test)
    priority = _indices(raw["baseline_priority_indices"], "baseline_priority_indices", n_test)
    lanes = (hard, standard, priority)
    if any(lanes[left] & lanes[right] for left in range(3) for right in range(left + 1, 3)):
        raise ContractError("baseline lane overlap")
    sources = _source_evidence(raw["source_evidence"], n_test)
    quality = _indices(raw["quality_flag_indices"], "quality_flag_indices", n_test)
    config_hash = _string(raw["config_sha256"], "config_sha256")
    assert config_hash is not None
    if len(config_hash) != 64 or any(char not in "0123456789abcdef" for char in config_hash):
        raise ContractError("config_sha256 must be lowercase SHA256")
    candidates = set().union(hard, standard, priority)
    preview = _series_preview(raw["series_preview"], n_test, candidates)
    if type(raw["has_missing"]) is not bool:
        raise ContractError("has_missing must be boolean")
    return ShadowRunInput(
        schema_version=SCHEMA_VERSION,
        collection_mode=COLLECTION_MODE,
        policy_version=POLICY_VERSION,
        run_id=_string(raw["run_id"], "run_id") or "",
        source_id=_string(raw["source_id"], "source_id") or "",
        observed_at=_utc_timestamp(raw["observed_at"]),
        dataset_key=_string(raw["dataset_key"], "dataset_key") or "",
        equipment_id=_string(raw["equipment_id"], "equipment_id", optional=True),
        recipe_id=_string(raw["recipe_id"], "recipe_id", optional=True),
        sensor_id=_string(raw["sensor_id"], "sensor_id", optional=True),
        step_id=_string(raw["step_id"], "step_id", optional=True),
        n_train=n_train,
        n_test=n_test,
        baseline_hard=hard,
        baseline_standard=standard,
        baseline_priority=priority,
        sources=sources,
        quality_flags=quality,
        has_missing=raw["has_missing"],
        missing_quality_status=_string(raw["missing_quality_status"], "missing_quality_status") or "",
        config_sha256=config_hash,
        series_preview=preview,
    )


def prediction_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        _canonical_json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json_value(value: object) -> object:
    """Match JavaScript JSON number semantics for cross-runtime hashing."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(child) for child in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
