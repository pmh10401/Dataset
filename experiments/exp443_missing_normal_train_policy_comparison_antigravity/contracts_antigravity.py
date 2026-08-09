"""Immutable, label-free contracts for Exp443.

Only RawSeriesAntigravity and KernelSeedScoresAntigravity carry NumPy arrays.
Every persisted contract converts arrays to deterministic JSON lists with
explicit dtype and shape. No class here has a TEST-label or evaluator import.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POLICIES = frozenset({
    "linear_edge_nearest_control",
    "exclude_nonfinite_row",
    "leave_one_out_peer_median",
})
_FORBIDDEN_PREDICTION_TERMS = frozenset({
    "label", "labels", "labels_blob", "ground_truth", "truth", "target",
    "anomaly", "anomalies", "anomaly_positions", "actual", "evaluation",
    "confusion", "precision", "recall", "specificity", "false_positive",
    "false_negative", "fpr", "f1", "metric", "metrics", "family_test",
})


def _missing_positions_sha256(value: tuple[int, ...]) -> str:
    encoded = json.dumps(list(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _name(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field_name} must be an exact integer >= {minimum}")
    return value


def _finite(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _digest(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return value


def _tuple_ints(value: object, field_name: str, *, sorted_unique: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    result = tuple(_integer(item, field_name) for item in value)
    if sorted_unique and (result != tuple(sorted(result)) or len(result) != len(set(result))):
        raise ValueError(f"{field_name} must be sorted and unique")
    return result


def _readonly_array(value: object, field_name: str, *, dtype: str, ndim: int, finite: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field_name} must be a NumPy array")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype or value.ndim != ndim or not value.flags.c_contiguous:
        raise ValueError(f"{field_name} must have dtype {expected_dtype} and ndim {ndim}")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{field_name} must contain finite values")
    # A bytes owner prevents a caller from simply toggling ``writeable`` back
    # to True on a mutable copied ndarray.
    immutable = bytes(np.ascontiguousarray(value, dtype=expected_dtype).tobytes(order="C"))
    return np.frombuffer(immutable, dtype=expected_dtype).reshape(value.shape)


def _array_value(value: float) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def _array_payload(array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "values": [_array_value(item) for item in array.reshape(-1)],
    }


def _array_from_payload(payload: object, field_name: str, *, dtype: str, ndim: int, finite: bool) -> np.ndarray:
    if not isinstance(payload, Mapping) or set(payload) != {"dtype", "shape", "values"}:
        raise ValueError(f"{field_name} payload is invalid")
    if payload["dtype"] != np.dtype(dtype).str or not isinstance(payload["shape"], list) or not isinstance(payload["values"], list):
        raise ValueError(f"{field_name} dtype/shape payload is invalid")
    shape = tuple(_integer(item, f"{field_name} shape") for item in payload["shape"])
    if len(shape) != ndim:
        raise ValueError(f"{field_name} shape rank is invalid")
    expected_size = int(np.prod(shape, dtype=np.int64))
    if len(payload["values"]) != expected_size:
        raise ValueError(f"{field_name} value count is invalid")
    decoded: list[float] = []
    for item in payload["values"]:
        if item == "NaN":
            decoded.append(float("nan"))
        elif item == "Infinity":
            decoded.append(float("inf"))
        elif item == "-Infinity":
            decoded.append(float("-inf"))
        elif type(item) in (int, float) and math.isfinite(float(item)):
            decoded.append(float(item))
        else:
            raise ValueError(f"{field_name} contains an invalid scalar")
    array = np.asarray(decoded, dtype=np.dtype(dtype)).reshape(shape)
    return _readonly_array(array, field_name, dtype=dtype, ndim=ndim, finite=finite)


def _safe_key(key: object) -> str:
    if type(key) is not str or not key:
        raise ValueError("JSON key must be a non-empty string")
    lowered = key.casefold().replace("-", "_")
    if any(term in lowered for term in _FORBIDDEN_PREDICTION_TERMS):
        raise ValueError(f"prediction payload contains forbidden key: {key}")
    return key


def _freeze_json(value: object, *, prediction: bool = False) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return _finite(value, "JSON float")
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            copied[_safe_key(key) if prediction else _name(key, "JSON key")] = _freeze_json(item, prediction=prediction)
        return MappingProxyType(copied)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, prediction=prediction) for item in value)
    raise ValueError("value is not deterministic JSON data")


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def validate_prediction_payload_antigravity(payload: object) -> None:
    """Recursively reject every label/evaluation field before publication."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _safe_key(key)
            validate_prediction_payload_antigravity(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            validate_prediction_payload_antigravity(value)
    elif payload is None or type(payload) in (str, int, bool):
        return
    elif type(payload) is float and math.isfinite(payload):
        return
    else:
        raise ValueError("prediction payload is not JSON-safe")


@dataclass(frozen=True, eq=False)
class RawSeriesAntigravity:
    """One raw TRAIN/TEST values blob with its pre-repair missing mask."""

    dataset: str
    split: str
    instance_index: int
    raw_blob_sha256: str
    values: np.ndarray
    missing_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.split not in {"TRAIN", "TEST"}:
            raise ValueError("split must be TRAIN or TEST")
        _integer(self.instance_index, "instance_index")
        _digest(self.raw_blob_sha256, "raw_blob_sha256")
        values = _readonly_array(self.values, "values", dtype="<f4", ndim=1, finite=False)
        missing = _tuple_ints(self.missing_positions, "missing_positions", sorted_unique=True)
        expected = tuple(int(index) for index in np.flatnonzero(~np.isfinite(values)))
        if missing != expected:
            raise ValueError("missing_positions must equal the non-finite mask")
        if any(index >= len(values) for index in missing):
            raise ValueError("missing_positions are out of range")
        actual_digest = sha256(values.tobytes(order="C")).hexdigest()
        if self.raw_blob_sha256 != actual_digest:
            raise ValueError("raw_blob_sha256 does not match exact float32 values")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "missing_positions", missing)

    def to_payload(self) -> dict[str, object]:
        array = _array_payload(self.values)
        return {
            "dataset": self.dataset, "split": self.split,
            "instance_index": self.instance_index, "raw_blob_sha256": self.raw_blob_sha256,
            "dtype": array["dtype"], "shape": array["shape"],
            "values": array["values"], "missing_positions": list(self.missing_positions),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RawSeriesAntigravity":
        expected = {"dataset", "split", "instance_index", "raw_blob_sha256", "dtype", "shape", "values", "missing_positions"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("raw series payload keys are invalid")
        values = _array_from_payload({"dtype": payload["dtype"], "shape": payload["shape"], "values": payload["values"]}, "values", dtype="<f4", ndim=1, finite=False)
        return cls(payload["dataset"], payload["split"], payload["instance_index"], payload["raw_blob_sha256"], values, tuple(payload["missing_positions"]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RawSeriesAntigravity) and (self.dataset, self.split, self.instance_index, self.raw_blob_sha256, self.missing_positions) == (other.dataset, other.split, other.instance_index, other.raw_blob_sha256, other.missing_positions) and np.array_equal(self.values, other.values, equal_nan=True)


@dataclass(frozen=True)
class RunPolicyAuditAntigravity:
    dataset: str
    policy: str
    instance_index: int
    original_length: int
    missing_positions: tuple[int, ...]
    missing_positions_sha256: str
    raw_blob_sha256: str
    action: str
    reference_instance_indices: tuple[int, ...] = ()
    unsupported_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.policy not in _POLICIES:
            raise ValueError("policy is not preregistered")
        _integer(self.instance_index, "instance_index")
        _integer(self.original_length, "original_length", minimum=1)
        missing = _tuple_ints(self.missing_positions, "missing_positions", sorted_unique=True)
        references = _tuple_ints(self.reference_instance_indices, "reference_instance_indices", sorted_unique=True)
        unsupported = _tuple_ints(self.unsupported_positions, "unsupported_positions", sorted_unique=True)
        if any(index >= self.original_length for index in missing + unsupported):
            raise ValueError("audit position is out of range")
        _digest(self.missing_positions_sha256, "missing_positions_sha256")
        if self.missing_positions_sha256 != _missing_positions_sha256(missing):
            raise ValueError("missing_positions_sha256 does not match positions")
        _digest(self.raw_blob_sha256, "raw_blob_sha256")
        _name(self.action, "action")
        if self.action not in {
            "unchanged", "linear_and_edge_nearest", "excluded_nonfinite_row",
            "leave_one_out_peer_median", "excluded_unsupported_peer",
        }:
            raise ValueError("audit action is invalid")
        if self.instance_index in references:
            raise ValueError("audit references must exclude the row itself")
        if not missing:
            if self.action != "unchanged" or references or unsupported:
                raise ValueError("finite audit action is inconsistent")
        elif self.policy == "linear_edge_nearest_control":
            if self.action != "linear_and_edge_nearest" or references or unsupported:
                raise ValueError("Policy A audit action is inconsistent")
        elif self.policy == "exclude_nonfinite_row":
            if self.action != "excluded_nonfinite_row" or references or unsupported:
                raise ValueError("Policy B audit action is inconsistent")
        elif self.action == "leave_one_out_peer_median":
            if unsupported or not references:
                raise ValueError("Policy C peer-median audit action is inconsistent")
        elif self.action == "excluded_unsupported_peer":
            if not unsupported:
                raise ValueError("Policy C unsupported audit action is inconsistent")
        else:
            raise ValueError("Policy C audit action is inconsistent")
        object.__setattr__(self, "missing_positions", missing)
        object.__setattr__(self, "reference_instance_indices", references)
        object.__setattr__(self, "unsupported_positions", unsupported)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset, "policy": self.policy, "instance_index": self.instance_index,
            "original_length": self.original_length, "missing_positions": list(self.missing_positions),
            "missing_positions_sha256": self.missing_positions_sha256,
            "raw_blob_sha256": self.raw_blob_sha256, "action": self.action,
            "reference_instance_indices": list(self.reference_instance_indices),
            "unsupported_positions": list(self.unsupported_positions),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RunPolicyAuditAntigravity":
        expected = {"dataset", "policy", "instance_index", "original_length", "missing_positions", "missing_positions_sha256", "raw_blob_sha256", "action", "reference_instance_indices", "unsupported_positions"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("policy audit payload keys are invalid")
        raw = dict(payload)
        raw["missing_positions"] = tuple(raw["missing_positions"])
        raw["reference_instance_indices"] = tuple(raw["reference_instance_indices"])
        raw["unsupported_positions"] = tuple(raw["unsupported_positions"])
        return cls(**raw)


@dataclass(frozen=True, eq=False)
class KernelSeedScoresAntigravity:
    kernel: int
    seed: int
    oof_scores: np.ndarray
    test_scores: np.ndarray
    candidate_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kernel not in {1024, 4096}:
            raise ValueError("kernel is not preregistered")
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG.seeds:
            raise ValueError("seed is not one of the two preregistered seeds")
        oof = _readonly_array(self.oof_scores, "oof_scores", dtype="<f8", ndim=1, finite=True)
        test = _readonly_array(self.test_scores, "test_scores", dtype="<f8", ndim=1, finite=True)
        candidates = _tuple_ints(self.candidate_indices, "candidate_indices", sorted_unique=True)
        if any(index >= len(test) for index in candidates):
            raise ValueError("candidate index is outside test_scores")
        object.__setattr__(self, "oof_scores", oof)
        object.__setattr__(self, "test_scores", test)
        object.__setattr__(self, "candidate_indices", candidates)

    def to_payload(self) -> dict[str, object]:
        return {
            "kernel": self.kernel, "seed": self.seed,
            "oof_scores": _array_payload(self.oof_scores),
            "test_scores": _array_payload(self.test_scores),
            "candidate_indices": list(self.candidate_indices),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "KernelSeedScoresAntigravity":
        expected = {"kernel", "seed", "oof_scores", "test_scores", "candidate_indices"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("kernel score payload keys are invalid")
        return cls(payload["kernel"], payload["seed"], _array_from_payload(payload["oof_scores"], "oof_scores", dtype="<f8", ndim=1, finite=True), _array_from_payload(payload["test_scores"], "test_scores", dtype="<f8", ndim=1, finite=True), tuple(payload["candidate_indices"]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KernelSeedScoresAntigravity) and (self.kernel, self.seed, self.candidate_indices) == (other.kernel, other.seed, other.candidate_indices) and np.array_equal(self.oof_scores, other.oof_scores) and np.array_equal(self.test_scores, other.test_scores)


@dataclass(frozen=True, eq=False)
class PolicyDatasetResultAntigravity:
    policy: str
    dataset: str
    target_length: int
    train_total: int
    train_missing: int
    train_used: int
    train_excluded: int
    test_rows: int
    kernel_scores: tuple[KernelSeedScoresAntigravity, ...]
    candidates_by_kernel: Mapping[int, tuple[int, ...]]
    audits: tuple[RunPolicyAuditAntigravity, ...]
    status: str

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError("policy is not preregistered")
        _name(self.dataset, "dataset")
        _integer(self.target_length, "target_length", minimum=1)
        counts = tuple(_integer(getattr(self, name), name) for name in ("train_total", "train_missing", "train_used", "train_excluded", "test_rows"))
        if counts[1] > counts[0] or counts[2] > counts[0] or counts[3] > counts[0] or counts[2] + counts[3] != counts[0]:
            raise ValueError("policy result counts are inconsistent")
        scores = tuple(self.kernel_scores)
        if any(type(item) is not KernelSeedScoresAntigravity for item in scores):
            raise ValueError("kernel_scores are invalid")
        if not isinstance(self.candidates_by_kernel, Mapping):
            raise ValueError("candidates_by_kernel must be a mapping")
        copied_candidates: dict[int, tuple[int, ...]] = {}
        for key, indices in self.candidates_by_kernel.items():
            if type(key) is not int or key not in {1024, 4096}:
                raise ValueError("candidate kernel is invalid")
            copied_candidates[key] = _tuple_ints(indices, "candidate indices", sorted_unique=True)
        audits = tuple(self.audits)
        if any(type(item) is not RunPolicyAuditAntigravity for item in audits):
            raise ValueError("audits are invalid")
        if self.status not in {"scored", "abstained", "failed"}:
            raise ValueError("status is invalid")
        object.__setattr__(self, "kernel_scores", scores)
        object.__setattr__(self, "candidates_by_kernel", MappingProxyType(copied_candidates))
        object.__setattr__(self, "audits", audits)

    def to_payload(self) -> dict[str, object]:
        return {
            "policy": self.policy, "dataset": self.dataset, "target_length": self.target_length,
            "train_total": self.train_total, "train_missing": self.train_missing,
            "train_used": self.train_used, "train_excluded": self.train_excluded,
            "test_rows": self.test_rows,
            "kernel_scores": [item.to_payload() for item in self.kernel_scores],
            "candidates_by_kernel": {str(key): list(value) for key, value in sorted(self.candidates_by_kernel.items())},
            "audits": [item.to_payload() for item in self.audits], "status": self.status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PolicyDatasetResultAntigravity":
        expected = {"policy", "dataset", "target_length", "train_total", "train_missing", "train_used", "train_excluded", "test_rows", "kernel_scores", "candidates_by_kernel", "audits", "status"}
        if not isinstance(payload, Mapping) or set(payload) != expected or not isinstance(payload["kernel_scores"], list) or not isinstance(payload["candidates_by_kernel"], Mapping) or not isinstance(payload["audits"], list):
            raise ValueError("policy result payload is invalid")
        candidates = {int(key): tuple(value) for key, value in payload["candidates_by_kernel"].items()}
        return cls(payload["policy"], payload["dataset"], payload["target_length"], payload["train_total"], payload["train_missing"], payload["train_used"], payload["train_excluded"], payload["test_rows"], tuple(KernelSeedScoresAntigravity.from_payload(item) for item in payload["kernel_scores"]), candidates, tuple(RunPolicyAuditAntigravity.from_payload(item) for item in payload["audits"]), payload["status"])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolicyDatasetResultAntigravity) and self.to_payload() == other.to_payload()


@dataclass(frozen=True)
class PredictionRowAntigravity:
    """One label-free prediction row; no labels or evaluation metrics allowed."""

    dataset: str
    policy: str
    hard_indices: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.policy not in _POLICIES:
            raise ValueError("policy is not preregistered")
        indices = _tuple_ints(self.hard_indices, "hard_indices", sorted_unique=True)
        scores = tuple(_finite(item, "score") for item in self.scores)
        evidence = _freeze_json(self.evidence, prediction=True)
        object.__setattr__(self, "hard_indices", indices)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "evidence", evidence)

    def to_payload(self) -> dict[str, object]:
        payload = {"dataset": self.dataset, "policy": self.policy, "hard_indices": list(self.hard_indices), "scores": list(self.scores), "evidence": _json_safe(self.evidence)}
        validate_prediction_payload_antigravity(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PredictionRowAntigravity":
        validate_prediction_payload_antigravity(payload)
        expected = {"dataset", "policy", "hard_indices", "scores", "evidence"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("prediction row payload is invalid")
        return cls(payload["dataset"], payload["policy"], tuple(payload["hard_indices"]), tuple(payload["scores"]), payload["evidence"])


@dataclass(frozen=True)
class CheckpointContextAntigravity:
    experiment_id: str
    run_identity: str
    scientific_config_sha256: str
    environment_lock_sha256: str
    total_units: int
    outer_workers: int
    inner_threads: int

    def __post_init__(self) -> None:
        if self.experiment_id != FROZEN_CONFIG.experiment_id:
            raise ValueError("checkpoint context experiment is invalid")
        if self.run_identity != FROZEN_CONFIG.run_identity:
            raise ValueError("checkpoint context run identity is not the frozen Exp443 run")
        if self.scientific_config_sha256 != scientific_config_sha256_antigravity():
            raise ValueError("checkpoint context scientific config hash is not the frozen Exp443 hash")
        if self.environment_lock_sha256 != FROZEN_CONFIG.environment_lock_sha256:
            raise ValueError("checkpoint context environment lock hash is not the frozen Exp443 lock")
        if self.total_units != FROZEN_CONFIG.recompute_units:
            raise ValueError("checkpoint context total units are not the frozen Exp443 count")
        if self.outer_workers != FROZEN_CONFIG.outer_workers or self.inner_threads != FROZEN_CONFIG.inner_threads:
            raise ValueError("checkpoint topology is invalid")

    def to_payload(self) -> dict[str, object]:
        return self.__dict__.copy()

    def validate_antigravity(self) -> bool:
        """Recheck the exact Exp443 identity before checkpoint use."""
        self.__post_init__()
        return True

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CheckpointContextAntigravity":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("checkpoint context payload is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class CheckpointStateAntigravity:
    status: str
    completed_units: int
    total_units: int
    error_count: int
    current_unit: str | None
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"running", "interrupted", "complete", "failed"}:
            raise ValueError("checkpoint state status is invalid")
        _integer(self.completed_units, "completed_units")
        _integer(self.total_units, "total_units", minimum=1)
        _integer(self.error_count, "error_count")
        if self.completed_units > self.total_units:
            raise ValueError("completed_units exceed total_units")
        if self.current_unit is not None:
            _name(self.current_unit, "current_unit")
        _digest(self.payload_sha256, "payload_sha256")

    def to_payload(self) -> dict[str, object]:
        return self.__dict__.copy()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CheckpointStateAntigravity":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("checkpoint state payload is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class EvaluationEvidenceAntigravity:
    """Post-freeze metrics evidence, kept separate from prediction rows."""

    policy: str
    tp: int
    fp: int
    fn: int
    tn: int
    micro_f1: float
    gate_passed: bool

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError("evaluation policy is invalid")
        for name in ("tp", "fp", "fn", "tn"):
            _integer(getattr(self, name), name)
        expected = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn) if (2 * self.tp + self.fp + self.fn) else 0.0
        if not math.isclose(_finite(self.micro_f1, "micro_f1"), expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("micro_f1 is inconsistent with confusion counts")
        if type(self.gate_passed) is not bool:
            raise ValueError("gate_passed must be a bool")

    def to_payload(self) -> dict[str, object]:
        return self.__dict__.copy()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvaluationEvidenceAntigravity":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("evaluation evidence payload is invalid")
        return cls(**dict(payload))


def canonical_contract_sha256_antigravity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "RawSeriesAntigravity", "RunPolicyAuditAntigravity", "KernelSeedScoresAntigravity",
    "PolicyDatasetResultAntigravity", "PredictionRowAntigravity",
    "CheckpointContextAntigravity", "CheckpointStateAntigravity",
    "EvaluationEvidenceAntigravity", "canonical_contract_sha256_antigravity",
    "validate_prediction_payload_antigravity",
]
