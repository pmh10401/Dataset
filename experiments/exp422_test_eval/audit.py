"""Pure confusion-matrix aggregation for frozen Exp422 inputs."""

import math

from .contracts import (
    AuditResult,
    ConfusionCounts,
    ConfusionMetrics,
    DatasetAuditRow,
    FrozenLabels,
    FrozenPrediction,
    MacroMetric,
)


_METRIC_NAMES = (
    "precision", "recall", "specificity", "false_positive_rate",
    "false_negative_rate", "f1",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_from_counts(counts: ConfusionCounts) -> ConfusionMetrics:
    if type(counts) is not ConfusionCounts:
        raise ValueError("metrics require ConfusionCounts")
    return ConfusionMetrics(
        precision=_ratio(counts.tp, counts.tp + counts.fp),
        recall=_ratio(counts.tp, counts.tp + counts.fn),
        specificity=_ratio(counts.tn, counts.tn + counts.fp),
        false_positive_rate=_ratio(counts.fp, counts.fp + counts.tn),
        false_negative_rate=_ratio(counts.fn, counts.fn + counts.tp),
        f1=_ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
    )


def _dataset_counts(dataset: str, labels: FrozenLabels, predicted: set[tuple[str, int]]) -> ConfusionCounts:
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for row in labels.rows:
        if row.dataset != dataset:
            continue
        is_predicted = row.coordinate in predicted
        is_actual = row.is_true_anomaly == 1
        if is_actual and is_predicted:
            counts["tp"] += 1
        elif is_actual:
            counts["fn"] += 1
        elif is_predicted:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return ConfusionCounts(**counts)


def _macro(rows: tuple[DatasetAuditRow, ...]) -> dict[str, MacroMetric]:
    result: dict[str, MacroMetric] = {}
    for name in _METRIC_NAMES:
        values = [
            value for row in rows
            if (value := getattr(row.metrics, name)) is not None and math.isfinite(value)
        ]
        result[name] = MacroMetric(
            value=(sum(values) / len(values) if values else None),
            defined_dataset_count=len(values),
        )
    return result


def compute_confusion_audit(
    frozen_prediction: FrozenPrediction,
    frozen_labels: FrozenLabels,
) -> AuditResult:
    """Compute micro and dataset-balanced metrics without any selection logic."""
    if type(frozen_prediction) is not FrozenPrediction:
        raise ValueError("prediction must be FrozenPrediction")
    if type(frozen_labels) is not FrozenLabels:
        raise ValueError("labels must be FrozenLabels")
    if frozen_prediction.dataset_scope != frozen_labels.dataset_scope:
        raise ValueError("prediction and label scopes differ")
    predicted = set(frozen_prediction.coordinates)
    label_coordinates = {row.coordinate for row in frozen_labels.rows}
    if not predicted <= label_coordinates:
        raise ValueError("prediction coordinate is outside label universe")
    datasets = tuple(sorted(frozen_labels.dataset_scope))
    rows = tuple(
        DatasetAuditRow(
            dataset=dataset,
            counts=(counts := _dataset_counts(dataset, frozen_labels, predicted)),
            metrics=metrics_from_counts(counts),
        )
        for dataset in datasets
    )
    micro = ConfusionCounts(
        tn=sum(row.counts.tn for row in rows),
        fp=sum(row.counts.fp for row in rows),
        fn=sum(row.counts.fn for row in rows),
        tp=sum(row.counts.tp for row in rows),
    )
    if micro.total != len(frozen_labels.rows):
        raise ValueError("micro total does not equal label rows")
    return AuditResult(
        micro_counts=micro,
        micro_metrics=metrics_from_counts(micro),
        dataset_rows=rows,
        macro_metrics=_macro(rows),
        label_rows=len(frozen_labels.rows),
        prediction_coordinates=len(predicted),
    )


__all__ = ["compute_confusion_audit", "metrics_from_counts"]
