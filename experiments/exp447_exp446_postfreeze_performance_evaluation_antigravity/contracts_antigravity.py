"""Deeply immutable, evaluation-only Exp447 value contracts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from types import MappingProxyType


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_CONFUSION_KEYS_ANTIGRAVITY = ("tp", "fp", "fn", "tn")
_FORBIDDEN_PRELABEL_TERMS_ANTIGRAVITY = frozenset({
    "label", "labels", "ground_truth", "truth", "target", "anomaly", "anomalies",
    "actual", "evaluation", "metric", "metrics", "confusion", "precision", "recall",
    "f1", "fpr", "false_positive", "false_negative", "tp", "fp", "fn", "tn",
})


def _exact_nonnegative_int_antigravity(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be an exact non-negative integer")
    return value


def _nonempty_name_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _finite_float_antigravity(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _sha256_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return value


def _coordinate_antigravity(value: object, field_name: str) -> tuple[str, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must be a (dataset, instance_index) pair")
    return _nonempty_name_antigravity(value[0], f"{field_name} dataset"), _exact_nonnegative_int_antigravity(value[1], f"{field_name} instance_index")


def _coordinates_antigravity(value: object, field_name: str) -> tuple[tuple[str, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a coordinate sequence")
    result = tuple(_coordinate_antigravity(item, field_name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} contains duplicate coordinates")
    return result


def freeze_prelabel_payload_antigravity(value: object) -> object:
    """Copy only deterministic source evidence while rejecting label-like fields."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return _finite_float_antigravity(value, "pre-label float")
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            name = _nonempty_name_antigravity(key, "pre-label key")
            lowered = name.casefold().replace("-", "_")
            if any(term in lowered for term in _FORBIDDEN_PRELABEL_TERMS_ANTIGRAVITY):
                raise ValueError(f"pre-label payload contains forbidden key: {name}")
            copied[name] = freeze_prelabel_payload_antigravity(item)
        return MappingProxyType(copied)
    if isinstance(value, (tuple, list)):
        return tuple(freeze_prelabel_payload_antigravity(item) for item in value)
    raise ValueError("pre-label payload is not deterministic JSON data")


@dataclass(frozen=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for name in _CONFUSION_KEYS_ANTIGRAVITY:
            _exact_nonnegative_int_antigravity(getattr(self, name), name)

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return 0.0 if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float:
        denominator = 2 * self.tp + self.fp + self.fn
        return 0.0 if denominator == 0 else (2 * self.tp) / denominator

    @property
    def false_positive_rate(self) -> float:
        denominator = self.fp + self.tn
        return 0.0 if denominator == 0 else self.fp / denominator

    def to_payload_antigravity(self) -> dict[str, int]:
        return {"fn": self.fn, "fp": self.fp, "tn": self.tn, "tp": self.tp}


@dataclass(frozen=True)
class MetricSummaryAntigravity:
    comparison_name: str
    confusion: ConfusionAntigravity
    macro_f1: float

    def __post_init__(self) -> None:
        _nonempty_name_antigravity(self.comparison_name, "comparison_name")
        if not isinstance(self.confusion, ConfusionAntigravity):
            raise ValueError("confusion has the wrong type")
        object.__setattr__(self, "macro_f1", _finite_float_antigravity(self.macro_f1, "macro_f1"))

    def to_payload_antigravity(self) -> dict[str, object]:
        confusion = self.confusion
        return {
            "comparison_name": self.comparison_name, "tp": confusion.tp, "fp": confusion.fp,
            "fn": confusion.fn, "tn": confusion.tn, "precision": confusion.precision,
            "recall": confusion.recall, "f1": confusion.f1,
            "false_positive_rate": confusion.false_positive_rate, "macro_f1": self.macro_f1,
        }


@dataclass(frozen=True)
class PublicationIdentityAntigravity:
    """Bytes-derived identity for one immutable Exp446 publication inspection."""

    root_device: int
    root_inode: int
    leaf_sha256: Mapping[str, str]
    manifest_sha256: str
    prediction_sha256: str
    artifact_map_sha256: str
    dataset_order: tuple[str, ...]
    source_coordinates: tuple[tuple[str, int], ...]
    stable_coordinates: tuple[tuple[str, int], ...]
    unstable_coordinates: tuple[tuple[str, int], ...]
    label_access_count: int

    def __post_init__(self) -> None:
        _exact_nonnegative_int_antigravity(self.root_device, "root_device")
        _exact_nonnegative_int_antigravity(self.root_inode, "root_inode")
        if not isinstance(self.leaf_sha256, Mapping) or not self.leaf_sha256:
            raise ValueError("leaf_sha256 is invalid")
        leaves: dict[str, str] = {}
        for name, digest in self.leaf_sha256.items():
            if "/" in _nonempty_name_antigravity(name, "leaf name") or name in leaves:
                raise ValueError("leaf name is invalid")
            leaves[name] = _sha256_antigravity(digest, "leaf SHA256")
        object.__setattr__(self, "leaf_sha256", MappingProxyType(leaves))
        for field_name in ("manifest_sha256", "prediction_sha256", "artifact_map_sha256"):
            _sha256_antigravity(getattr(self, field_name), field_name)
        if isinstance(self.dataset_order, (str, bytes)) or not isinstance(self.dataset_order, Sequence):
            raise ValueError("dataset_order is invalid")
        datasets = tuple(_nonempty_name_antigravity(item, "dataset") for item in self.dataset_order)
        if not datasets or len(datasets) != len(set(datasets)):
            raise ValueError("dataset_order must be non-empty and unique")
        source = _coordinates_antigravity(self.source_coordinates, "source_coordinates")
        stable = _coordinates_antigravity(self.stable_coordinates, "stable_coordinates")
        unstable = _coordinates_antigravity(self.unstable_coordinates, "unstable_coordinates")
        if set(source) != set(stable) | set(unstable) or set(stable) & set(unstable):
            raise ValueError("source/stable/unstable lanes do not conserve coordinates")
        if any(dataset not in set(datasets) for dataset, _ in source):
            raise ValueError("source coordinate dataset is absent from dataset order")
        _exact_nonnegative_int_antigravity(self.label_access_count, "label_access_count")
        if self.label_access_count != 0:
            raise ValueError("publication label access must be zero")
        object.__setattr__(self, "dataset_order", datasets)
        object.__setattr__(self, "source_coordinates", source)
        object.__setattr__(self, "stable_coordinates", stable)
        object.__setattr__(self, "unstable_coordinates", unstable)

    @property
    def source_count(self) -> int:
        return len(self.source_coordinates)

    @property
    def stable_count(self) -> int:
        return len(self.stable_coordinates)

    @property
    def unstable_count(self) -> int:
        return len(self.unstable_coordinates)


@dataclass(frozen=True)
class DatasetConfusionRowAntigravity:
    dataset: str
    confusion: ConfusionAntigravity

    def __post_init__(self) -> None:
        _nonempty_name_antigravity(self.dataset, "dataset")
        if not isinstance(self.confusion, ConfusionAntigravity):
            raise ValueError("dataset confusion has the wrong type")


def normalize_dataset_confusion_rows_antigravity(rows: object) -> tuple[DatasetConfusionRowAntigravity, ...]:
    """Make exact, ordered per-dataset confusions safe for paired comparisons."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("dataset rows are invalid")
    normalized: list[DatasetConfusionRowAntigravity] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"dataset", *_CONFUSION_KEYS_ANTIGRAVITY}:
            raise ValueError("dataset row keys are invalid")
        normalized.append(DatasetConfusionRowAntigravity(
            _nonempty_name_antigravity(row["dataset"], "dataset"),
            ConfusionAntigravity(*(_exact_nonnegative_int_antigravity(row[name], name) for name in _CONFUSION_KEYS_ANTIGRAVITY)),
        ))
    if len({row.dataset for row in normalized}) != len(normalized):
        raise ValueError("dataset rows contain duplicates")
    return tuple(normalized)


@dataclass(frozen=True)
class BootstrapComparisonAntigravity:
    comparator_name: str
    population: int
    seed: int
    samples: int
    sample_dtype: str
    endpoint: bool
    percentile_method: str
    baseline_micro_f1: float
    comparator_micro_f1: float
    baseline_macro_f1: float
    comparator_macro_f1: float
    micro_delta: float
    macro_delta: float
    micro_interval: tuple[float, float]
    macro_interval: tuple[float, float]
    sample_indices_sha256: str

    def __post_init__(self) -> None:
        _nonempty_name_antigravity(self.comparator_name, "comparator_name")
        for field_name in ("population", "seed", "samples"):
            _exact_nonnegative_int_antigravity(getattr(self, field_name), field_name)
        if self.sample_dtype != "int64" or self.endpoint is not False or self.percentile_method != "linear":
            raise ValueError("bootstrap draw contract is invalid")
        for field_name in ("baseline_micro_f1", "comparator_micro_f1", "baseline_macro_f1", "comparator_macro_f1", "micro_delta", "macro_delta"):
            object.__setattr__(self, field_name, _finite_float_antigravity(getattr(self, field_name), field_name))
        for field_name in ("micro_interval", "macro_interval"):
            value = getattr(self, field_name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, tuple(_finite_float_antigravity(item, field_name) for item in value))
        _sha256_antigravity(self.sample_indices_sha256, "sample_indices_sha256")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "comparator_name": self.comparator_name, "population": self.population, "seed": self.seed,
            "samples": self.samples, "sample_dtype": self.sample_dtype, "endpoint": self.endpoint,
            "percentile_method": self.percentile_method, "baseline_micro_f1": self.baseline_micro_f1,
            "comparator_micro_f1": self.comparator_micro_f1, "baseline_macro_f1": self.baseline_macro_f1,
            "comparator_macro_f1": self.comparator_macro_f1, "micro_delta": self.micro_delta,
            "macro_delta": self.macro_delta, "micro_interval": list(self.micro_interval),
            "macro_interval": list(self.macro_interval), "sample_indices_sha256": self.sample_indices_sha256,
        }


__all__ = [
    "BootstrapComparisonAntigravity", "ConfusionAntigravity", "DatasetConfusionRowAntigravity",
    "MetricSummaryAntigravity", "PublicationIdentityAntigravity", "freeze_prelabel_payload_antigravity",
    "normalize_dataset_confusion_rows_antigravity",
]
