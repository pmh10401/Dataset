"""Pure family-balanced aggregation for the frozen Exp410 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
import math

import numpy as np

from .config import (
    FROZEN_CONFIG,
    Exp410FrozenConfig,
    require_frozen_research_parameters,
)
from .contracts import (
    AuditResult,
    DatasetMetric,
    DatasetProfile,
    FamilyMetric,
    family_from_dataset,
)


_FAMILY_SIZE_BUCKETS = ("1", "2-5", "6-20", "21+")
_TRAIN_BUCKETS = ("<20", "20-49", "50+")
_OVERLAP_BUCKETS = ("<50%", "50-<90%", ">=90%")
_ANOMALY_BUCKETS = ("1", "2-10", "11+")
_DIFFICULTY_NOT_EVALUATED = {
    "status": "not_evaluated",
    "reason": "source_artifact_unavailable",
}


def aggregate_family_robustness(
    metrics: Sequence[DatasetMetric],
    profiles: Sequence[DatasetProfile],
    config: Exp410FrozenConfig = FROZEN_CONFIG,
) -> AuditResult:
    """Compute every preregistered family-balanced diagnostic."""
    require_frozen_research_parameters(config)
    metric_rows, profile_by_dataset = _validated_inputs(metrics, profiles, config)
    grouped: dict[str, list[DatasetMetric]] = defaultdict(list)
    for row in metric_rows:
        grouped[row.family].append(row)
    family_metrics = tuple(
        _family_metric(family, tuple(grouped[family]))
        for family in sorted(grouped)
    )
    _verify_family_size_buckets(family_metrics, config)

    dataset_macro = math.fsum(row.f1 for row in metric_rows) / len(metric_rows)
    family_macro = math.fsum(
        row.mean_dataset_f1 for row in family_metrics
    ) / len(family_metrics)
    sensitivity = tuple(
        {
            "cap": "unlimited" if cap is None else str(cap),
            "mean_f1": capped_mean(family_metrics, cap),
            "total_weight": sum(
                row.dataset_count if cap is None else min(row.dataset_count, cap)
                for row in family_metrics
            ),
        }
        for cap in config.family_weight_caps
    )
    unlimited = sensitivity[-1]["mean_f1"]
    if (
        not isinstance(unlimited, float)
        or abs(unlimited - dataset_macro) > config.float_tolerance
    ):
        raise ValueError("Exp410 unlimited family cap differs from dataset-macro F1")

    influence = _family_influence(metric_rows, grouped, dataset_macro)
    stratified = _stratified_metrics(
        metric_rows,
        profile_by_dataset,
        grouped,
    )
    bootstrap = _clustered_bootstrap(
        grouped,
        dataset_macro=dataset_macro,
        family_macro=family_macro,
        config=config,
    )
    total_tp = sum(row.tp for row in metric_rows)
    total_fp = sum(row.fp for row in metric_rows)
    total_fn = sum(row.fn for row in metric_rows)
    pooled_precision = _precision(total_tp, total_fp)
    pooled_recall = total_tp / (total_tp + total_fn)
    pooled_f1 = _f1(total_tp, total_fp, total_fn)
    summary = {
        "evaluation_mode": "retrospective_family_robustness_audit",
        "dataset_count": len(metric_rows),
        "family_count": len(family_metrics),
        "hard_count": sum(row.hard_count for row in metric_rows),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "dataset_macro_f1": dataset_macro,
        "family_macro_f1": family_macro,
        "delta_f1": family_macro - dataset_macro,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "bootstrap": bootstrap,
        "robustness_classification": bootstrap["classification"],
        "difficulty_stratification": dict(_DIFFICULTY_NOT_EVALUATED),
        "operational_policy_changed": config.operational_policy_changed,
        "deployment_eligible": config.deployment_eligible,
        "unseen_family_validated": config.unseen_family_validated,
        "prospective_validated": config.prospective_validated,
    }
    return AuditResult(
        dataset_metrics=metric_rows,
        family_metrics=family_metrics,
        weight_sensitivity=sensitivity,
        family_influence=influence,
        stratified_metrics=stratified,
        summary=summary,
    )


def capped_mean(
    families: Sequence[FamilyMetric],
    cap: int | None,
) -> float:
    if not families or (cap is not None and cap <= 0):
        raise ValueError("Exp410 family cap requires rows and a positive cap")
    weights = [
        row.dataset_count if cap is None else min(row.dataset_count, cap)
        for row in families
    ]
    return math.fsum(
        weight * row.mean_dataset_f1
        for weight, row in zip(weights, families, strict=True)
    ) / sum(weights)


def _validated_inputs(
    metrics: Sequence[DatasetMetric],
    profiles: Sequence[DatasetProfile],
    config: Exp410FrozenConfig,
) -> tuple[tuple[DatasetMetric, ...], dict[str, DatasetProfile]]:
    metric_rows = tuple(sorted(metrics, key=lambda row: row.dataset))
    profile_rows = tuple(sorted(profiles, key=lambda row: row.dataset))
    if (
        len(metric_rows) != config.expected_datasets
        or len(profile_rows) != config.expected_datasets
    ):
        raise ValueError("Exp410 aggregation dataset count mismatch")
    if len({row.dataset for row in metric_rows}) != len(metric_rows):
        raise ValueError("Exp410 aggregation contains duplicate dataset metrics")
    if len({row.dataset for row in profile_rows}) != len(profile_rows):
        raise ValueError("Exp410 aggregation contains duplicate dataset profiles")
    profile_by_dataset = {row.dataset: row for row in profile_rows}
    if set(profile_by_dataset) != {row.dataset for row in metric_rows}:
        raise ValueError("Exp410 metric/profile dataset coverage mismatch")
    if config.family_weight_caps != (1, 3, 5, 10, None):
        raise ValueError("Exp410 preregistered family caps changed")

    for metric in metric_rows:
        profile = profile_by_dataset[metric.dataset]
        canonical_family = family_from_dataset(metric.dataset)
        if metric.family != canonical_family or profile.family != canonical_family:
            raise ValueError("Exp410 canonical family mismatch")
        if (
            min(metric.hard_count, metric.tp, metric.fp, metric.fn) < 0
            or metric.hard_count != metric.tp + metric.fp
            or metric.tp + metric.fn != profile.test_anomaly_count
        ):
            raise ValueError("Exp410 metric/profile count invariant mismatch")
        expected_f1 = _f1(metric.tp, metric.fp, metric.fn)
        if abs(metric.f1 - expected_f1) > config.float_tolerance:
            raise ValueError("Exp410 dataset F1 invariant mismatch")
        if (
            profile.train_normal_count <= 0
            or profile.test_anomaly_count <= 0
            or profile.test_total_count
            != profile.test_anomaly_count + profile.test_normal_count
            or not 0 <= profile.exact_overlap_count <= profile.test_normal_count
        ):
            raise ValueError("Exp410 dataset profile invariant mismatch")
        expected_overlap = (
            profile.exact_overlap_count / profile.test_normal_count
            if profile.test_normal_count
            else 0.0
        )
        if abs(profile.exact_overlap_fraction - expected_overlap) > config.float_tolerance:
            raise ValueError("Exp410 exact overlap fraction mismatch")

    families = {row.family for row in metric_rows}
    if len(families) != config.expected_families:
        raise ValueError("Exp410 aggregation family count mismatch")
    return metric_rows, profile_by_dataset


def _family_metric(
    family: str,
    rows: Sequence[DatasetMetric],
) -> FamilyMetric:
    count = len(rows)
    tp = sum(row.tp for row in rows)
    fp = sum(row.fp for row in rows)
    fn = sum(row.fn for row in rows)
    return FamilyMetric(
        family=family,
        dataset_count=count,
        mean_dataset_f1=math.fsum(row.f1 for row in rows) / count,
        mean_dataset_fp=math.fsum(float(row.fp) for row in rows) / count,
        pooled_tp=tp,
        pooled_fp=fp,
        pooled_fn=fn,
        pooled_precision=_precision(tp, fp),
        pooled_recall=tp / (tp + fn),
        pooled_f1=_f1(tp, fp, fn),
    )


def _verify_family_size_buckets(
    families: Sequence[FamilyMetric],
    config: Exp410FrozenConfig,
) -> None:
    counts = Counter(_family_size_bucket(row.dataset_count) for row in families)
    observed = tuple(counts[bucket] for bucket in _FAMILY_SIZE_BUCKETS)
    if observed != config.expected_family_size_buckets:
        raise ValueError(
            f"Exp410 family-size bucket count mismatch: {observed!r}"
        )


def _family_influence(
    metrics: Sequence[DatasetMetric],
    grouped: dict[str, list[DatasetMetric]],
    baseline_dataset_macro: float,
) -> tuple[dict[str, object], ...]:
    total_count = len(metrics)
    total_tp = sum(row.tp for row in metrics)
    total_fp = sum(row.fp for row in metrics)
    total_fn = sum(row.fn for row in metrics)
    baseline_precision = _precision(total_tp, total_fp)
    rows: list[dict[str, object]] = []
    for family in sorted(grouped):
        removed = grouped[family]
        removed_count = len(removed)
        remaining_count = total_count - removed_count
        if remaining_count <= 0:
            raise ValueError("Exp410 influence requires at least two families")
        removed_tp = sum(row.tp for row in removed)
        removed_fp = sum(row.fp for row in removed)
        removed_fn = sum(row.fn for row in removed)
        remaining_f1 = (
            math.fsum(row.f1 for row in metrics)
            - math.fsum(row.f1 for row in removed)
        ) / remaining_count
        remaining_precision = _precision(
            total_tp - removed_tp,
            total_fp - removed_fp,
        )
        rows.append(
            {
                "family": family,
                "dataset_count": removed_count,
                "dataset_share": removed_count / total_count,
                "fp_share": removed_fp / total_fp if total_fp else 0.0,
                "removed_alert_count": removed_tp + removed_fp,
                "removed_tp": removed_tp,
                "removed_fp": removed_fp,
                "removed_fn": removed_fn,
                "dataset_macro_f1_without_family": remaining_f1,
                "dataset_macro_f1_delta": (
                    remaining_f1 - baseline_dataset_macro
                ),
                "pooled_precision_without_family": remaining_precision,
                "pooled_precision_delta": (
                    remaining_precision - baseline_precision
                ),
            }
        )
    return tuple(rows)


def _stratified_metrics(
    metrics: Sequence[DatasetMetric],
    profiles: dict[str, DatasetProfile],
    grouped: dict[str, list[DatasetMetric]],
) -> tuple[dict[str, object], ...]:
    family_sizes = {
        family: len(rows)
        for family, rows in grouped.items()
    }
    definitions: tuple[
        tuple[str, tuple[str, ...], Callable[[DatasetMetric], str]],
        ...,
    ] = (
        (
            "family_size",
            _FAMILY_SIZE_BUCKETS,
            lambda row: _family_size_bucket(family_sizes[row.family]),
        ),
        (
            "normal_train_count",
            _TRAIN_BUCKETS,
            lambda row: train_bucket(profiles[row.dataset].train_normal_count),
        ),
        (
            "exact_normal_overlap",
            _OVERLAP_BUCKETS,
            lambda row: overlap_bucket(
                profiles[row.dataset].exact_overlap_fraction
            ),
        ),
        (
            "test_anomaly_count",
            _ANOMALY_BUCKETS,
            lambda row: anomaly_bucket(
                profiles[row.dataset].test_anomaly_count
            ),
        ),
    )
    result: list[dict[str, object]] = []
    for dimension, buckets, assign in definitions:
        assigned = {row.dataset: assign(row) for row in metrics}
        for bucket in buckets:
            eligible = [
                row for row in metrics
                if assigned[row.dataset] == bucket
            ]
            by_family: dict[str, list[float]] = defaultdict(list)
            for row in eligible:
                by_family[row.family].append(row.f1)
            family_values = [
                math.fsum(values) / len(values)
                for _, values in sorted(by_family.items())
            ]
            result.append(
                {
                    "dimension": dimension,
                    "bucket": bucket,
                    "dataset_count": len(eligible),
                    "family_count": len(family_values),
                    "family_balanced_mean_f1": (
                        math.fsum(family_values) / len(family_values)
                        if family_values
                        else None
                    ),
                }
            )
    return tuple(result)


def _clustered_bootstrap(
    grouped: dict[str, list[DatasetMetric]],
    *,
    dataset_macro: float,
    family_macro: float,
    config: Exp410FrozenConfig,
) -> dict[str, object]:
    if config.bootstrap_iterations <= 0:
        raise ValueError("Exp410 bootstrap iterations must be positive")
    families = tuple(sorted(grouped))
    family_sums = np.asarray(
        [math.fsum(row.f1 for row in grouped[family]) for family in families],
        dtype=np.float64,
    )
    family_counts = np.asarray(
        [len(grouped[family]) for family in families],
        dtype=np.int64,
    )
    family_means = family_sums / family_counts
    rng = np.random.default_rng(config.bootstrap_seed)
    draws = rng.integers(
        0,
        len(families),
        size=(config.bootstrap_iterations, len(families)),
    )
    sampled_family_macro = family_means[draws].mean(axis=1)
    sampled_dataset_macro = (
        family_sums[draws].sum(axis=1)
        / family_counts[draws].sum(axis=1)
    )
    deltas = sampled_family_macro - sampled_dataset_macro
    low, high = np.quantile(deltas, [0.025, 0.975])
    point_delta = family_macro - dataset_macro
    if point_delta < config.severe_margin:
        classification = "severe_family_imbalance"
    elif float(low) < config.robustness_margin:
        classification = "sensitive"
    else:
        classification = "robust"
    return {
        "unit": "family",
        "seed": config.bootstrap_seed,
        "iterations": config.bootstrap_iterations,
        "point_family_macro_f1": family_macro,
        "point_dataset_macro_f1": dataset_macro,
        "point_delta_f1": point_delta,
        "delta_f1_interval_95": {
            "low": float(low),
            "high": float(high),
        },
        "robustness_margin": config.robustness_margin,
        "severe_margin": config.severe_margin,
        "classification": classification,
    }


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _family_size_bucket(n: int) -> str:
    if n <= 0:
        raise ValueError("Exp410 family size must be positive")
    return "1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else "21+"


def train_bucket(n: int) -> str:
    if n <= 0:
        raise ValueError("Exp410 normal TRAIN count must be positive")
    return "<20" if n < 20 else "20-49" if n < 50 else "50+"


def overlap_bucket(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("Exp410 overlap fraction must be in [0, 1]")
    return "<50%" if value < 0.5 else "50-<90%" if value < 0.9 else ">=90%"


def anomaly_bucket(n: int) -> str:
    if n <= 0:
        raise ValueError("Exp410 TEST anomaly count must be positive")
    return "1" if n == 1 else "2-10" if n <= 10 else "11+"
