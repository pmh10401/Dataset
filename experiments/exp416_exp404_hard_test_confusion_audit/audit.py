"""Pure exact confusion-matrix calculation for frozen Exp416 inputs."""

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
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "f1",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_from_counts(counts: ConfusionCounts) -> ConfusionMetrics:
    """Derive metrics from counts, retaining undefined ratios as ``None``."""
    return ConfusionMetrics(
        precision=_ratio(counts.tp, counts.tp + counts.fp),
        recall=_ratio(counts.tp, counts.tp + counts.fn),
        specificity=_ratio(counts.tn, counts.tn + counts.fp),
        false_positive_rate=_ratio(counts.fp, counts.fp + counts.tn),
        false_negative_rate=_ratio(counts.fn, counts.fn + counts.tp),
        f1=_ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn),
    )


def _counts_for_dataset(
    dataset: str, frozen_labels: FrozenLabels, predicted_coordinates: set[tuple[str, int]]
) -> ConfusionCounts:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in frozen_labels.rows:
        if row.dataset != dataset:
            continue
        predicted = row.coordinate in predicted_coordinates
        actual = row.is_true_anomaly == 1
        if predicted and actual:
            counts["tp"] += 1
        elif predicted:
            counts["fp"] += 1
        elif actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return ConfusionCounts(**counts)


def _sum_counts(rows: tuple[DatasetAuditRow, ...]) -> ConfusionCounts:
    return ConfusionCounts(
        tp=sum(row.counts.tp for row in rows),
        fp=sum(row.counts.fp for row in rows),
        fn=sum(row.counts.fn for row in rows),
        tn=sum(row.counts.tn for row in rows),
    )


def _macro_metrics(rows: tuple[DatasetAuditRow, ...]) -> dict[str, MacroMetric]:
    macros: dict[str, MacroMetric] = {}
    for metric_name in _METRIC_NAMES:
        defined_values = [
            value
            for row in rows
            if (value := getattr(row.metrics, metric_name)) is not None
            and math.isfinite(value)
        ]
        macros[metric_name] = MacroMetric(
            value=(sum(defined_values) / len(defined_values) if defined_values else None),
            defined_dataset_count=len(defined_values),
        )
    return macros


def compute_confusion_audit(
    frozen_prediction: FrozenPrediction, frozen_labels: FrozenLabels
) -> AuditResult:
    """Calculate exact micro and dataset-balanced confusion results in memory."""
    if not isinstance(frozen_prediction, FrozenPrediction):
        raise ValueError("Exp416 audit requires FrozenPrediction")
    if not isinstance(frozen_labels, FrozenLabels):
        raise ValueError("Exp416 audit requires FrozenLabels")
    if frozen_prediction.dataset_scope != frozen_labels.dataset_scope:
        raise ValueError("Exp416 prediction and label dataset scopes differ")

    label_coordinates = {row.coordinate for row in frozen_labels.rows}
    predicted_coordinates = set(frozen_prediction.coordinates)
    if not predicted_coordinates <= label_coordinates:
        raise ValueError("Exp416 prediction coordinate is outside label universe")

    datasets = tuple(sorted(frozen_labels.dataset_scope))
    if datasets != frozen_prediction.dataset_scope:
        raise ValueError("Exp416 audit dataset scope must be sorted")
    if {row.dataset for row in frozen_labels.rows} != set(datasets):
        raise ValueError("Exp416 audit dataset rows must be 1:1 with scope")

    dataset_rows = tuple(
        DatasetAuditRow(
            dataset=dataset,
            counts=(counts := _counts_for_dataset(dataset, frozen_labels, predicted_coordinates)),
            metrics=metrics_from_counts(counts),
        )
        for dataset in datasets
    )
    if len(dataset_rows) != len(datasets) or {row.dataset for row in dataset_rows} != set(datasets):
        raise ValueError("Exp416 audit dataset rows must be 1:1 with scope")

    micro_counts = _sum_counts(dataset_rows)
    if micro_counts.total != len(frozen_labels.rows):
        raise ValueError("Exp416 micro count total must equal label row count")
    if micro_counts != _sum_counts(dataset_rows):
        raise ValueError("Exp416 dataset count totals differ from micro counts")

    return AuditResult(
        micro_counts=micro_counts,
        micro_metrics=metrics_from_counts(micro_counts),
        dataset_rows=dataset_rows,
        macro_metrics=_macro_metrics(dataset_rows),
        label_rows=len(frozen_labels.rows),
        prediction_coordinates=len(predicted_coordinates),
    )


__all__ = ["compute_confusion_audit", "metrics_from_counts"]
