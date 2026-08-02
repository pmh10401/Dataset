"""Immutable contracts shared by the Exp422-TEST-EVAL stages."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


Coordinate = tuple[str, int]


def _dataset(value: object, field: str = "dataset") -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty exact string")
    return value


def _index(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative exact integer")
    return value


def _digest(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _scope(value: object, field: str = "dataset_scope") -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field} must be a tuple")
    values = tuple(_dataset(item, f"{field} item") for item in value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{field} must be sorted and unique")
    if not values:
        raise ValueError(f"{field} must not be empty")
    return values


def _coordinates(value: object, field: str = "coordinates") -> tuple[Coordinate, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field} must be a tuple")
    result: list[Coordinate] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"{field} entries must be (dataset, index)")
        result.append((_dataset(item[0], f"{field} dataset"), _index(item[1], f"{field} index")))
    frozen = tuple(result)
    if frozen != tuple(sorted(frozen)) or len(frozen) != len(set(frozen)):
        raise ValueError(f"{field} must be sorted and unique")
    return frozen


def _metric(value: object, field: str) -> None:
    if value is None:
        return
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be a finite float in [0, 1] or None")


@dataclass(frozen=True)
class FrozenPrediction:
    coordinates: tuple[Coordinate, ...]
    dataset_scope: tuple[str, ...]
    prediction_dataset_count: int
    identity_sha256: str
    prediction_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        coordinates = _coordinates(self.coordinates)
        scope = _scope(self.dataset_scope)
        if type(self.prediction_dataset_count) is not int or self.prediction_dataset_count < 0:
            raise ValueError("prediction_dataset_count must be a nonnegative exact integer")
        coordinate_scope = {dataset for dataset, _ in coordinates}
        if self.prediction_dataset_count != len(coordinate_scope):
            raise ValueError("prediction_dataset_count must match coordinate dataset count")
        if not coordinate_scope <= set(scope):
            raise ValueError("prediction coordinates must be within dataset_scope")
        _digest(self.identity_sha256, "identity_sha256")
        _digest(self.prediction_sha256, "prediction_sha256")
        _digest(self.manifest_sha256, "manifest_sha256")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "dataset_scope", scope)


@dataclass(frozen=True)
class LabelRow:
    dataset: str
    candidate_index: int
    is_true_anomaly: int

    def __post_init__(self) -> None:
        _dataset(self.dataset)
        _index(self.candidate_index, "candidate_index")
        if type(self.is_true_anomaly) is not int or self.is_true_anomaly not in (0, 1):
            raise ValueError("is_true_anomaly must be an exact binary integer")

    @property
    def coordinate(self) -> Coordinate:
        return (self.dataset, self.candidate_index)


@dataclass(frozen=True)
class FrozenLabels:
    rows: tuple[LabelRow, ...]
    dataset_scope: tuple[str, ...]
    labels_sha256: str

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or not self.rows:
            raise ValueError("rows must be a non-empty tuple")
        if any(type(row) is not LabelRow for row in self.rows):
            raise ValueError("rows must contain LabelRow records")
        coords = tuple(row.coordinate for row in self.rows)
        if coords != tuple(sorted(coords)) or len(coords) != len(set(coords)):
            raise ValueError("label rows must be sorted and unique")
        scope = _scope(self.dataset_scope)
        if {row.dataset for row in self.rows} != set(scope):
            raise ValueError("label rows must cover exactly dataset_scope")
        _digest(self.labels_sha256, "labels_sha256")
        object.__setattr__(self, "dataset_scope", scope)


@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int

    def __post_init__(self) -> None:
        for name in ("tn", "fp", "fn", "tp"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def as_nested(self) -> list[list[int]]:
        """Return the conventional [actual][predicted] matrix [[TN, FP], [FN, TP]]."""
        return [[self.tn, self.fp], [self.fn, self.tp]]


@dataclass(frozen=True)
class ConfusionMetrics:
    precision: float | None
    recall: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    f1: float | None

    def __post_init__(self) -> None:
        for name in (
            "precision", "recall", "specificity", "false_positive_rate",
            "false_negative_rate", "f1",
        ):
            _metric(getattr(self, name), name)


@dataclass(frozen=True)
class MacroMetric:
    value: float | None
    defined_dataset_count: int

    def __post_init__(self) -> None:
        _metric(self.value, "macro value")
        if type(self.defined_dataset_count) is not int or self.defined_dataset_count < 0:
            raise ValueError("defined_dataset_count must be a nonnegative exact integer")
        if (self.value is None) != (self.defined_dataset_count == 0):
            raise ValueError("macro value and defined_dataset_count disagree")


@dataclass(frozen=True)
class DatasetAuditRow:
    dataset: str
    counts: ConfusionCounts
    metrics: ConfusionMetrics

    def __post_init__(self) -> None:
        _dataset(self.dataset)
        if type(self.counts) is not ConfusionCounts or type(self.metrics) is not ConfusionMetrics:
            raise ValueError("dataset row types are invalid")


@dataclass(frozen=True)
class AuditResult:
    micro_counts: ConfusionCounts
    micro_metrics: ConfusionMetrics
    dataset_rows: tuple[DatasetAuditRow, ...]
    macro_metrics: Mapping[str, MacroMetric]
    label_rows: int
    prediction_coordinates: int

    def __post_init__(self) -> None:
        if type(self.micro_counts) is not ConfusionCounts or type(self.micro_metrics) is not ConfusionMetrics:
            raise ValueError("micro result types are invalid")
        if type(self.dataset_rows) is not tuple or not self.dataset_rows:
            raise ValueError("dataset_rows must be a non-empty tuple")
        names = tuple(row.dataset for row in self.dataset_rows)
        if any(type(row) is not DatasetAuditRow for row in self.dataset_rows):
            raise ValueError("dataset_rows must contain DatasetAuditRow records")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("dataset_rows must be sorted and unique")
        if not isinstance(self.macro_metrics, Mapping):
            raise ValueError("macro_metrics must be a mapping")
        if type(self.label_rows) is not int or self.label_rows < 1:
            raise ValueError("label_rows must be a positive exact integer")
        if type(self.prediction_coordinates) is not int or self.prediction_coordinates < 0:
            raise ValueError("prediction_coordinates must be a nonnegative exact integer")
        if self.micro_counts.total != self.label_rows:
            raise ValueError("micro total must equal label_rows")
        if self.micro_counts.tp + self.micro_counts.fp != self.prediction_coordinates:
            raise ValueError("predicted positive count must equal prediction_coordinates")
        aggregate = ConfusionCounts(
            tn=sum(row.counts.tn for row in self.dataset_rows),
            fp=sum(row.counts.fp for row in self.dataset_rows),
            fn=sum(row.counts.fn for row in self.dataset_rows),
            tp=sum(row.counts.tp for row in self.dataset_rows),
        )
        if aggregate != self.micro_counts:
            raise ValueError("dataset counts must sum to micro counts")
        object.__setattr__(self, "macro_metrics", MappingProxyType(dict(self.macro_metrics)))


__all__ = [
    "AuditResult", "ConfusionCounts", "ConfusionMetrics", "Coordinate",
    "DatasetAuditRow", "FrozenLabels", "FrozenPrediction", "LabelRow",
    "MacroMetric",
]
