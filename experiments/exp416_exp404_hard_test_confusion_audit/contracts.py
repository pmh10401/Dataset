"""Immutable, validated values shared by the Exp416 audit stages."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


Coordinate = tuple[str, int]


def _require_dataset(value: object, field_name: str = "dataset") -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative exact integer")


def _require_nonempty_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_scope(scope: object, field_name: str) -> tuple[str, ...]:
    if type(scope) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    for dataset in scope:
        _require_dataset(dataset, field_name)
    if tuple(sorted(scope)) != scope or len(set(scope)) != len(scope):
        raise ValueError(f"{field_name} must be sorted and unique")
    return scope


def _validate_coordinates(coordinates: object, field_name: str) -> tuple[Coordinate, ...]:
    if type(coordinates) is not tuple:
        raise ValueError(f"{field_name} must be a tuple")
    for coordinate in coordinates:
        if type(coordinate) is not tuple or len(coordinate) != 2:
            raise ValueError(f"{field_name} entries must be (dataset, candidate_index)")
        _require_dataset(coordinate[0], f"{field_name} dataset")
        _require_nonnegative_int(coordinate[1], f"{field_name} candidate_index")
    if tuple(sorted(coordinates)) != coordinates or len(set(coordinates)) != len(coordinates):
        raise ValueError(f"{field_name} must be sorted and unique")
    return coordinates


def _validate_metric(value: object, field_name: str) -> None:
    if value is None:
        return
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be a finite float in [0, 1] or None")


@dataclass(frozen=True)
class FrozenPrediction:
    coordinates: tuple[Coordinate, ...]
    dataset_scope: tuple[str, ...]
    prediction_dataset_count: int
    identity_sha256: str
    prediction_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        coordinates = _validate_coordinates(self.coordinates, "coordinates")
        scope = _validate_scope(self.dataset_scope, "dataset_scope")
        _require_nonnegative_int(self.prediction_dataset_count, "prediction_dataset_count")
        coordinate_scope = {dataset for dataset, _ in coordinates}
        if self.prediction_dataset_count != len(coordinate_scope):
            raise ValueError("prediction_dataset_count must equal coordinate dataset size")
        if not coordinate_scope <= set(scope):
            raise ValueError("coordinates must be within dataset_scope")
        _require_nonempty_string(self.identity_sha256, "identity_sha256")
        _require_nonempty_string(self.prediction_sha256, "prediction_sha256")
        _require_nonempty_string(self.manifest_sha256, "manifest_sha256")


@dataclass(frozen=True)
class LabelRow:
    dataset: str
    candidate_index: int
    is_true_anomaly: int

    def __post_init__(self) -> None:
        _require_dataset(self.dataset)
        _require_nonnegative_int(self.candidate_index, "candidate_index")
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
            raise ValueError("rows must contain LabelRow values")
        coordinates = tuple(row.coordinate for row in self.rows)
        if tuple(sorted(coordinates)) != coordinates or len(set(coordinates)) != len(coordinates):
            raise ValueError("rows must have sorted, unique coordinates")
        scope = _validate_scope(self.dataset_scope, "dataset_scope")
        if {row.dataset for row in self.rows} != set(scope):
            raise ValueError("rows must cover exactly dataset_scope")
        _require_nonempty_string(self.labels_sha256, "labels_sha256")


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for field_name in ("tp", "fp", "fn", "tn"):
            _require_nonnegative_int(getattr(self, field_name), field_name)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


@dataclass(frozen=True)
class ConfusionMetrics:
    precision: float | None
    recall: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    f1: float | None

    def __post_init__(self) -> None:
        for field_name in (
            "precision",
            "recall",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "f1",
        ):
            _validate_metric(getattr(self, field_name), field_name)


def metrics_from_counts(counts: ConfusionCounts) -> ConfusionMetrics:
    """Derive all audit metrics, preserving undefined zero-denominator ratios."""

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    return ConfusionMetrics(
        precision=ratio(counts.tp, counts.tp + counts.fp),
        recall=ratio(counts.tp, counts.tp + counts.fn),
        specificity=ratio(counts.tn, counts.tn + counts.fp),
        false_positive_rate=ratio(counts.fp, counts.fp + counts.tn),
        false_negative_rate=ratio(counts.fn, counts.fn + counts.tp),
        f1=ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
    )


def _validate_metrics_match_counts(
    counts: ConfusionCounts,
    metrics: ConfusionMetrics,
    field_name: str,
) -> None:
    if metrics != metrics_from_counts(counts):
        raise ValueError(f"{field_name} must be derived from confusion counts")


@dataclass(frozen=True)
class MacroMetric:
    value: float | None
    defined_dataset_count: int

    def __post_init__(self) -> None:
        _validate_metric(self.value, "value")
        _require_nonnegative_int(self.defined_dataset_count, "defined_dataset_count")
        if (self.value is None) != (self.defined_dataset_count == 0):
            raise ValueError("value must be None exactly when defined_dataset_count is zero")


@dataclass(frozen=True)
class DatasetAuditRow:
    dataset: str
    counts: ConfusionCounts
    metrics: ConfusionMetrics

    def __post_init__(self) -> None:
        _require_dataset(self.dataset)
        if type(self.counts) is not ConfusionCounts:
            raise ValueError("counts must be ConfusionCounts")
        if type(self.metrics) is not ConfusionMetrics:
            raise ValueError("metrics must be ConfusionMetrics")
        _validate_metrics_match_counts(self.counts, self.metrics, "metrics")


@dataclass(frozen=True)
class AuditResult:
    micro_counts: ConfusionCounts
    micro_metrics: ConfusionMetrics
    dataset_rows: tuple[DatasetAuditRow, ...]
    macro_metrics: Mapping[str, MacroMetric]
    label_rows: int
    prediction_coordinates: int

    def __post_init__(self) -> None:
        if type(self.micro_counts) is not ConfusionCounts:
            raise ValueError("micro_counts must be ConfusionCounts")
        if type(self.micro_metrics) is not ConfusionMetrics:
            raise ValueError("micro_metrics must be ConfusionMetrics")
        if type(self.dataset_rows) is not tuple or not self.dataset_rows:
            raise ValueError("dataset_rows must be a non-empty tuple")
        if any(type(row) is not DatasetAuditRow for row in self.dataset_rows):
            raise ValueError("dataset_rows must contain DatasetAuditRow values")
        datasets = tuple(row.dataset for row in self.dataset_rows)
        if tuple(sorted(datasets)) != datasets or len(set(datasets)) != len(datasets):
            raise ValueError("dataset_rows must be sorted and unique")
        if not isinstance(self.macro_metrics, Mapping):
            raise ValueError("macro_metrics must be a mapping")
        macro_metrics = dict(self.macro_metrics)
        for name, metric in macro_metrics.items():
            _require_nonempty_string(name, "macro metric name")
            if type(metric) is not MacroMetric:
                raise ValueError("macro_metrics values must be MacroMetric")
            if metric.defined_dataset_count > len(self.dataset_rows):
                raise ValueError("macro metric defined_dataset_count exceeds dataset rows")
        _require_nonnegative_int(self.label_rows, "label_rows")
        _require_nonnegative_int(self.prediction_coordinates, "prediction_coordinates")
        if self.prediction_coordinates > self.label_rows:
            raise ValueError("prediction_coordinates exceeds label_rows")
        if self.micro_counts.total != self.label_rows:
            raise ValueError("micro count total must equal label_rows")
        if self.prediction_coordinates != self.micro_counts.tp + self.micro_counts.fp:
            raise ValueError("prediction_coordinates must equal TP + FP")
        _validate_metrics_match_counts(self.micro_counts, self.micro_metrics, "micro_metrics")
        aggregate = ConfusionCounts(
            tp=sum(row.counts.tp for row in self.dataset_rows),
            fp=sum(row.counts.fp for row in self.dataset_rows),
            fn=sum(row.counts.fn for row in self.dataset_rows),
            tn=sum(row.counts.tn for row in self.dataset_rows),
        )
        if aggregate != self.micro_counts:
            raise ValueError("dataset count totals must equal micro_counts")
        object.__setattr__(self, "macro_metrics", MappingProxyType(macro_metrics))
