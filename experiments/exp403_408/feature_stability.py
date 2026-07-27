"""Deterministic TRAIN OOF feature stability audit for Exp403.

The audit never reads ``FeatureBlock.test_values``.  It is intentionally a
TRAIN-OOF-only contract that later stages can apply without changing which
features were retained.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import hashlib
import math

import numpy as np

from .contracts import ALPHAS, SEED, FeatureBlock, FeatureGroupSummary, FeatureStability, Fold, StabilityAudit


CORRELATION_THRESHOLD = 0.90
NEAR_ZERO_ROBUST_VARIANCE = 1e-12
BOOTSTRAP_REPLICATES = 128


def audit_feature_stability(
    blocks: tuple[FeatureBlock, ...], folds: Sequence[Fold], *, seed: int = SEED
) -> StabilityAudit:
    """Audit immutable feature blocks using only their TRAIN OOF vectors."""
    ordered_blocks = tuple(sorted(blocks, key=lambda block: block.name))
    names = [block.name for block in ordered_blocks]
    if len(names) != len(set(names)):
        raise ValueError("feature block names must be unique")
    if not ordered_blocks:
        return StabilityAudit()

    excluded: dict[str, str] = {}
    stability: dict[str, FeatureStability] = {}
    values_by_name: dict[str, np.ndarray] = {}
    expected_rows: int | None = None

    for block in ordered_blocks:
        if not block.available:
            excluded[block.name] = block.reason or "source_unavailable"
            continue
        values = _train_oof_values(block)
        if values is None:
            excluded[block.name] = "missing_train_oof"
            continue
        if expected_rows is None:
            expected_rows = len(values)
        if expected_rows is not None and len(values) != expected_rows:
            excluded[block.name] = "train_oof_length_mismatch"
            continue
        finite = np.isfinite(values)
        finite_rate = float(np.mean(finite))
        if not finite.all():
            excluded[block.name] = "non_finite_train_oof"
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_variance = float((1.4826 * mad) ** 2)
        if robust_variance <= NEAR_ZERO_ROBUST_VARIANCE:
            excluded[block.name] = "near_zero_train_variance"
            continue
        if expected_rows is None:
            raise AssertionError("available feature did not establish TRAIN OOF rows")

        _validate_folds(folds, expected_rows)
        stability[block.name] = FeatureStability(
            feature_name=block.name,
            group=block.group,
            finite_rate=finite_rate,
            non_finite_rate=1.0 - finite_rate,
            median=median,
            mad=mad,
            robust_variance=robust_variance,
            fold_center_drift=_fold_center_drift(values, folds, median, mad),
            bootstrap_rank_stability=_bootstrap_rank_stability(values, block.name, seed),
            coverage=finite_rate,
            empirical_p_values=_empirical_p_values(values),
        )
        values_by_name[block.name] = values

    components = _correlation_components(values_by_name)
    representatives = _representatives(stability, components)
    audit = StabilityAudit(
        excluded=dict(sorted(excluded.items())),
        correlation_components=components,
        correlation_threshold=CORRELATION_THRESHOLD,
        representatives=representatives,
        feature_stability=dict(sorted(stability.items())),
    )
    summaries = {
        group: build_group_summary(audit, group=group)
        for group in sorted({metric.group for metric in stability.values()})
    }
    return replace(audit, group_summaries=summaries)


def build_group_summary(
    audit: StabilityAudit, *, group: str | None = None
) -> FeatureGroupSummary:
    """Build one fixed, TRAIN-OOF empirical-p summary for a feature group.

    With more than one retained group, omitting ``group`` produces the fixed
    all-group view.  Per-group summaries are also stored in ``audit``.
    """
    metrics = [
        audit.feature_stability[name]
        for name in audit.representatives
        if name in audit.feature_stability
        and (group is None or audit.feature_stability[name].group == group)
    ]
    summary_group = group or _summary_group(metrics)
    feature_names = tuple(sorted(metric.feature_name for metric in metrics))
    if not metrics:
        return FeatureGroupSummary(group=summary_group, feature_names=feature_names)

    p_values = np.asarray(
        [value for metric in metrics for value in metric.empirical_p_values], dtype=float
    )
    evidence = -np.log10(np.clip(p_values, np.finfo(float).tiny, 1.0))
    ordered_evidence = np.sort(evidence)[::-1]
    top_two_gap = (
        float(ordered_evidence[0] - ordered_evidence[1])
        if len(ordered_evidence) > 1
        else 0.0
    )
    return FeatureGroupSummary(
        group=summary_group,
        feature_names=feature_names,
        minimum=float(np.min(p_values)),
        median=float(np.median(p_values)),
        robust_mean=float(np.median(evidence)),
        maximum=float(np.max(evidence)),
        exceedance_count=int(np.count_nonzero(p_values <= ALPHAS[-1])),
        top_two_gap=top_two_gap,
        stability=float(np.median([metric.bootstrap_rank_stability for metric in metrics])),
    )


def _train_oof_values(block: FeatureBlock) -> np.ndarray | None:
    if block.train_oof is None:
        return None
    values = np.asarray(block.train_oof, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"TRAIN OOF feature must be one-dimensional: {block.name}")
    return values


def _validate_folds(folds: Sequence[Fold], n_rows: int) -> None:
    if not folds:
        raise ValueError("at least one TRAIN fold is required")
    for fold in folds:
        validation = tuple(int(index) for index in fold.validation_indices)
        if not validation:
            raise ValueError("TRAIN fold validation rows must not be empty")
        if set(fold.train_indices) & set(validation):
            raise ValueError("TRAIN fold fit and validation rows overlap")
        if any(index < 0 or index >= n_rows for index in (*fold.train_indices, *validation)):
            raise ValueError("TRAIN fold index outside OOF vector")


def _fold_center_drift(values: np.ndarray, folds: Sequence[Fold], median: float, mad: float) -> float:
    scale = max(1.4826 * mad, np.finfo(float).eps)
    centers = np.asarray(
        [np.median(values[np.asarray(fold.validation_indices, dtype=int)]) for fold in folds],
        dtype=float,
    )
    return float(np.median(np.abs(centers - median)) / scale)


def _bootstrap_rank_stability(values: np.ndarray, feature_name: str, seed: int) -> float:
    ordered = np.sort(values, kind="mergesort")
    n_rows = len(ordered)
    feature_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{feature_name}".encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(feature_seed)
    baseline_ranks = _average_ranks(ordered)
    stability: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, n_rows, size=n_rows)
        stability.append(_spearman(baseline_ranks[sample], _average_ranks(ordered[sample])))
    return float(np.median(stability))


def _empirical_p_values(values: np.ndarray) -> tuple[float, ...]:
    ordered = np.sort(values, kind="mergesort")
    n_rows = len(ordered)
    left = np.searchsorted(ordered, values, side="left")
    return tuple(sorted(float((n_rows - index) / n_rows) for index in left))


def _correlation_components(values_by_name: Mapping[str, np.ndarray]) -> tuple[tuple[str, ...], ...]:
    names = tuple(sorted(values_by_name))
    parents = {name: name for name in names}

    def find(name: str) -> str:
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if abs(_spearman(values_by_name[left], values_by_name[right])) >= CORRELATION_THRESHOLD:
                union(left, right)

    members: dict[str, list[str]] = {}
    for name in names:
        members.setdefault(find(name), []).append(name)
    return tuple(sorted(tuple(names) for names in members.values() if len(names) > 1))


def _representatives(
    stability: Mapping[str, FeatureStability], components: Iterable[tuple[str, ...]]
) -> tuple[str, ...]:
    component_members = {name for component in components for name in component}
    selected = [
        min(component, key=lambda name: _representative_key(stability[name]))
        for component in components
    ]
    selected.extend(sorted(set(stability) - component_members))
    return tuple(sorted(selected))


def _representative_key(metric: FeatureStability) -> tuple[float, float, float, str]:
    return (
        metric.non_finite_rate,
        -metric.bootstrap_rank_stability,
        -metric.coverage,
        metric.feature_name,
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(np.asarray(left, dtype=float))
    right_rank = _average_ranks(np.asarray(right, dtype=float))
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = math.sqrt(float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    return 0.0 if denominator == 0.0 else float(np.dot(left_centered, right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _summary_group(metrics: Sequence[FeatureStability]) -> str:
    groups = sorted({metric.group for metric in metrics})
    if not groups:
        return "none"
    return groups[0] if len(groups) == 1 else "all_groups"
