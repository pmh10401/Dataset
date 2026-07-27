"""Hash-gated, retrospective-only Exp405 Hard-lane demotion evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .contracts import ALPHA_KEYS, SEED, DemotionEvaluation
from .foundation import FrozenLaneUniverse
from .prediction import (
    load_expected_provenance,
    verify_prediction,
)


BOOTSTRAP_ITERATIONS = 2_000
WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class EvaluationLabels:
    """Labels plus precomputed TRAIN-only context for one evaluated dataset."""

    test_labels: Sequence[object]
    train_oof_normal_stability: float
    lineage: str


@dataclass(frozen=True)
class DemotionGateDecision:
    fp_fraction: float
    tp_retention: float
    train_stability: float
    passes_research_gate: bool


LabelLoader = Callable[[set[str]], Mapping[str, EvaluationLabels]]


def demotion_gate(
    fp_fraction: float,
    tp_retention: float,
    train_stability: float,
) -> DemotionGateDecision:
    """Apply the fixed research-only gate without changing any lane."""
    values = (float(fp_fraction), float(tp_retention), float(train_stability))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("demotion gate values must be finite")
    return DemotionGateDecision(
        fp_fraction=values[0],
        tp_retention=values[1],
        train_stability=values[2],
        passes_research_gate=(
            values[0] >= 0.80 and values[1] >= 0.99 and values[2] >= 0.99
        ),
    )


def evaluate_exp405(
    prediction_output: Path,
    label_loader: LabelLoader,
    *,
    frozen_lane_universe: FrozenLaneUniverse,
    provenance_contract_path: Path,
    expected_provenance_sha256: str,
) -> DemotionEvaluation:
    """Verify frozen prediction and external trust anchor before loading labels."""
    expected_provenance = load_expected_provenance(
        Path(provenance_contract_path),
        expected_sha256=expected_provenance_sha256,
    )
    rows, _manifest = verify_prediction(
        Path(prediction_output),
        frozen_lane_universe=frozen_lane_universe,
        expected_provenance=expected_provenance,
    )
    expected_hard_identities = _frozen_hard_identities(frozen_lane_universe)
    verified_rows = _validated_prediction_rows(rows)
    _validate_lane_parity(verified_rows, expected_hard_identities)
    labels = label_loader({str(row["dataset"]) for row in rows})
    return evaluate_hard_confidence(
        verified_rows,
        labels,
        expected_hard_identities=expected_hard_identities,
    )


def evaluate_hard_confidence(
    prediction: Sequence[Mapping[str, object]],
    labels: Mapping[str, EvaluationLabels],
    *,
    frozen_lane_universe: FrozenLaneUniverse | None = None,
    expected_hard_identities: frozenset[tuple[str, int]] | None = None,
) -> DemotionEvaluation:
    """Measure the all-alpha recheck demotion counterfactual without mutation.

    A candidate is demotable only when every preregistered alpha is ``recheck``.
    This preserves the frozen alpha contract and prevents TEST outcomes from
    selecting a permissive alpha.
    """
    rows = _validated_prediction_rows(prediction)
    expected_identities = _resolve_expected_hard_identities(
        frozen_lane_universe=frozen_lane_universe,
        expected_hard_identities=expected_hard_identities,
    )
    _validate_lane_parity(rows, expected_identities)
    label_data = _validated_labels(rows, labels)
    by_dataset = _dataset_metrics(rows, label_data)
    recheck_count = sum(metrics.recheck_count for metrics in by_dataset.values())
    false_positive_count = sum(
        metrics.recheck_false_positives for metrics in by_dataset.values()
    )
    true_positive_count = sum(
        metrics.recheck_true_positives for metrics in by_dataset.values()
    )
    baseline_true_positives = sum(
        metrics.baseline_true_positives for metrics in by_dataset.values()
    )
    false_positive_fraction = (
        false_positive_count / recheck_count if recheck_count else 0.0
    )
    tp_retention = (
        (baseline_true_positives - true_positive_count) / baseline_true_positives
        if baseline_true_positives
        else 1.0
    )
    train_stability = (
        sum(
            metrics.train_stability * metrics.recheck_count
            for metrics in by_dataset.values()
        )
        / recheck_count
        if recheck_count
        else 0.0
    )
    concentration_counts = {
        dataset: metrics.recheck_count
        for dataset, metrics in by_dataset.items()
        if metrics.recheck_count
    }
    lineage_counts: dict[str, int] = {}
    for dataset, count in concentration_counts.items():
        lineage = label_data[dataset].lineage
        lineage_counts[lineage] = lineage_counts.get(lineage, 0) + count
    fp_bootstrap, retention_bootstrap = _paired_dataset_bootstrap(
        tuple(by_dataset.values()),
        point_fp_fraction=false_positive_fraction,
        point_tp_retention=tp_retention,
    )
    gate = demotion_gate(
        false_positive_fraction,
        tp_retention,
        train_stability,
    )
    return DemotionEvaluation(
        recheck_count=recheck_count,
        false_positive_count=false_positive_count,
        true_positive_count=true_positive_count,
        false_positive_fraction=false_positive_fraction,
        baseline_true_positive_retention=tp_retention,
        train_stability=train_stability,
        false_positive_fraction_wilson_interval=_wilson_interval(
            false_positive_count, recheck_count
        ),
        false_positive_fraction_bootstrap_interval=fp_bootstrap,
        baseline_true_positive_retention_bootstrap_interval=retention_bootstrap,
        dataset_concentration=_concentration(concentration_counts, recheck_count),
        lineage_concentration=_concentration(lineage_counts, recheck_count),
        lane_parity=True,
        passes_research_gate=gate.passes_research_gate,
        operational_policy_changed=False,
    )


@dataclass(frozen=True)
class _DatasetMetrics:
    recheck_count: int
    recheck_false_positives: int
    recheck_true_positives: int
    baseline_true_positives: int
    train_stability: float


def _validated_prediction_rows(
    prediction: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    if not prediction:
        raise ValueError("Exp405 requires one verified prediction row")
    identities: set[tuple[str, int]] = set()
    rows: list[Mapping[str, object]] = []
    for row in prediction:
        if not isinstance(row, Mapping):
            raise ValueError("prediction row must be a mapping")
        dataset = row.get("dataset")
        candidate_index = row.get("candidate_index")
        confidence = row.get("confidence_by_alpha")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("prediction dataset is invalid")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
        ):
            raise ValueError("prediction candidate index is invalid")
        if not isinstance(confidence, Mapping) or set(confidence) != set(ALPHA_KEYS):
            raise ValueError("prediction confidence alpha coverage is invalid")
        if not all(
            grade in {"high", "medium", "recheck", "abstain"}
            for grade in confidence.values()
        ):
            raise ValueError("prediction confidence grade is invalid")
        identity = (dataset, candidate_index)
        if identity in identities:
            raise ValueError("prediction contains duplicate Hard candidate")
        identities.add(identity)
        rows.append(row)
    return tuple(rows)


def _resolve_expected_hard_identities(
    *,
    frozen_lane_universe: FrozenLaneUniverse | None,
    expected_hard_identities: frozenset[tuple[str, int]] | None,
) -> frozenset[tuple[str, int]]:
    if frozen_lane_universe is None and expected_hard_identities is None:
        raise TypeError(
            "Exp405 requires a verified FrozenLaneUniverse or immutable Hard identities"
        )
    if frozen_lane_universe is not None:
        universe_identities = _frozen_hard_identities(frozen_lane_universe)
        if expected_hard_identities is not None and expected_hard_identities != universe_identities:
            raise ValueError("expected Hard identities do not match frozen lane universe")
        return universe_identities
    if not isinstance(expected_hard_identities, frozenset) or not expected_hard_identities:
        raise TypeError("expected Hard identities must be a non-empty frozenset")
    normalized: set[tuple[str, int]] = set()
    for identity in expected_hard_identities:
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise ValueError("expected Hard identity is invalid")
        dataset, candidate_index = identity
        if (
            not isinstance(dataset, str)
            or not dataset
            or isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
        ):
            raise ValueError("expected Hard identity is invalid")
        normalized.add((dataset, candidate_index))
    if len(normalized) != len(expected_hard_identities):
        raise ValueError("expected Hard identities contain duplicates")
    return frozenset(normalized)


def _frozen_hard_identities(
    frozen_lane_universe: FrozenLaneUniverse,
) -> frozenset[tuple[str, int]]:
    if not isinstance(frozen_lane_universe, FrozenLaneUniverse):
        raise TypeError("frozen_lane_universe must be a verified FrozenLaneUniverse")
    return frozenset(
        (str(dataset), int(candidate_index))
        for dataset, candidate_indices in frozen_lane_universe.hard_lanes().items()
        for candidate_index in candidate_indices
    )


def _validate_lane_parity(
    rows: Sequence[Mapping[str, object]],
    expected_hard_identities: frozenset[tuple[str, int]],
) -> None:
    actual = {(str(row["dataset"]), int(row["candidate_index"])) for row in rows}
    if actual != expected_hard_identities:
        raise ValueError("frozen Hard lane coverage mismatch")


def _validated_labels(
    rows: Sequence[Mapping[str, object]], labels: Mapping[str, EvaluationLabels]
) -> dict[str, EvaluationLabels]:
    datasets = {str(row["dataset"]) for row in rows}
    if set(labels) != datasets:
        raise ValueError("prediction and label dataset coverage differs")
    normalized: dict[str, EvaluationLabels] = {}
    for dataset in sorted(datasets):
        value = labels[dataset]
        if not isinstance(value, EvaluationLabels):
            raise TypeError("labels must be EvaluationLabels")
        if not value.lineage:
            raise ValueError("label lineage is required")
        stability = float(value.train_oof_normal_stability)
        if not math.isfinite(stability) or not 0.0 <= stability <= 1.0:
            raise ValueError("TRAIN OOF normal stability is invalid")
        test_labels = tuple(value.test_labels)
        if not test_labels:
            raise ValueError("TEST labels are required")
        for label in test_labels:
            _binary_label(label)
        normalized[dataset] = EvaluationLabels(test_labels, stability, value.lineage)
    for row in rows:
        dataset = str(row["dataset"])
        if int(row["candidate_index"]) >= len(normalized[dataset].test_labels):
            raise ValueError(f"label index is outside TEST labels: {dataset}")
    return normalized


def _dataset_metrics(
    rows: Sequence[Mapping[str, object]], labels: Mapping[str, EvaluationLabels]
) -> dict[str, _DatasetMetrics]:
    metrics: dict[str, _DatasetMetrics] = {}
    for dataset in sorted(labels):
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        label_data = labels[dataset]
        recheck_rows = [
            row
            for row in dataset_rows
            if all(grade == "recheck" for grade in row["confidence_by_alpha"].values())
        ]
        baseline_tp = sum(
            _binary_label(label_data.test_labels[int(row["candidate_index"])])
            for row in dataset_rows
        )
        recheck_tp = sum(
            _binary_label(label_data.test_labels[int(row["candidate_index"])])
            for row in recheck_rows
        )
        metrics[dataset] = _DatasetMetrics(
            recheck_count=len(recheck_rows),
            recheck_false_positives=len(recheck_rows) - recheck_tp,
            recheck_true_positives=recheck_tp,
            baseline_true_positives=baseline_tp,
            train_stability=label_data.train_oof_normal_stability,
        )
    return metrics


def _binary_label(value: object) -> bool:
    """Decode the explicit binary TEST label domain; reject every other value."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        if int(value) in {0, 1}:
            return bool(int(value))
    elif isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return bool(int(numeric))
    elif isinstance(value, str):
        labels = {
            "0": False,
            "0.0": False,
            "normal": False,
            "false": False,
            "1": True,
            "1.0": True,
            "anomaly": True,
            "true": True,
        }
        parsed = labels.get(value.strip().lower())
        if parsed is not None:
            return parsed
    raise ValueError("TEST label must be an explicit binary normal/anomaly value")


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    z_squared = WILSON_Z_95**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z_squared / (4.0 * total**2)
        )
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _paired_dataset_bootstrap(
    metrics: tuple[_DatasetMetrics, ...],
    *,
    point_fp_fraction: float,
    point_tp_retention: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not metrics:
        return (
            (point_fp_fraction, point_fp_fraction),
            (point_tp_retention, point_tp_retention),
        )
    generator = np.random.default_rng(SEED)
    fp_fractions: list[float] = []
    tp_retentions: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = [
            metrics[index]
            for index in generator.integers(0, len(metrics), size=len(metrics))
        ]
        recheck_count = sum(row.recheck_count for row in sampled)
        if recheck_count:
            fp_fractions.append(
                sum(row.recheck_false_positives for row in sampled) / recheck_count
            )
        baseline_tp = sum(row.baseline_true_positives for row in sampled)
        if baseline_tp:
            tp_retentions.append(
                (baseline_tp - sum(row.recheck_true_positives for row in sampled))
                / baseline_tp
            )
    return (
        _percentile_interval(fp_fractions, point_fp_fraction),
        _percentile_interval(tp_retentions, point_tp_retention),
    )


def _percentile_interval(values: Sequence[float], point: float) -> tuple[float, float]:
    if not values:
        return (point, point)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _concentration(counts: Mapping[str, int], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum((count / total) ** 2 for count in counts.values())
