"""Label-free, row-local TEST scoring for Exp436 strict consensus."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Callable, Protocol

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    Exp436Candidate,
    Exp436DatasetResult,
    InstanceScoreEvidence,
    MissingSafeRawDataset,
    RawValueSeries,
    TrainGroup,
    UnavailableTrainGroup,
)
from .features_antigravity import acf_vector, score_channel_cpu


class LaneMembership(Protocol):
    """The frozen-lane operation needed by label-free prediction."""

    def contains(self, dataset: str, instance_index: int) -> bool:
        ...


class CandidateBudgetExceeded(ValueError):
    """Raised when the frozen global review budget is exceeded."""

    def __init__(self, *, observed: int, budget: int) -> None:
        self.observed = observed
        self.budget = budget
        super().__init__(f"Exp436 candidate budget exceeded: observed {observed}, budget {budget}")


def _normalise_groups(groups: Sequence[TrainGroup] | TrainGroup) -> tuple[TrainGroup, ...]:
    if isinstance(groups, (CalibratedTrainGroup, UnavailableTrainGroup)):
        return (groups,)
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("calibrated_groups must be a sequence")
    return tuple(groups)


def _dataset_and_groups(groups: Sequence[TrainGroup]) -> tuple[str, dict[int, TrainGroup]]:
    if not groups:
        raise ValueError("calibrated_groups must contain a dataset group")
    dataset: str | None = None
    by_length: dict[int, TrainGroup] = {}
    for group in groups:
        if not isinstance(group, (CalibratedTrainGroup, UnavailableTrainGroup)):
            raise ValueError("calibrated_groups contain an invalid group")
        if dataset is None:
            dataset = group.dataset
        elif group.dataset != dataset:
            raise ValueError("calibrated group dataset does not match TEST dataset")
        if group.length in by_length:
            raise ValueError("calibrated_groups contain duplicate lengths")
        by_length[group.length] = group
    assert dataset is not None
    return dataset, by_length


def _raw_float32_vector(series: RawValueSeries) -> np.ndarray | None:
    """Convert and classify a raw row before any routing or transformation."""

    if not isinstance(series, RawValueSeries):
        raise ValueError("test_series must be a RawValueSeries")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(series.values, dtype=np.float32)
    if values.ndim != 1 or values.shape[0] != series.length:
        raise ValueError("TEST row changed during float32 conversion")
    if not np.isfinite(values).all():
        return None
    return np.ascontiguousarray(values, dtype=np.float32)


def _routing_abstention(dataset: str, series: RawValueSeries, reason: str) -> InstanceScoreEvidence:
    return InstanceScoreEvidence(
        dataset=dataset,
        instance_index=series.instance_index,
        length=series.length,
        status="abstained",
        exclusion_reason=reason,
    )


def _standardize_query(values: np.ndarray, calibration: ChannelCalibration) -> np.ndarray:
    center, scale = calibration.center, calibration.scale
    if center is None or scale is None:
        raise ValueError("channel calibration transform is unavailable")
    center_value, scale_value = np.float32(center), np.float32(scale)
    if not np.isfinite(center_value) or not np.isfinite(scale_value) or float(scale_value) <= FROZEN_CONFIG.numeric_floor:
        raise ValueError("channel calibration transform is nonfinite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = np.asarray((values - center_value) / scale_value, dtype=np.float32)
    if transformed.ndim != 1 or not np.isfinite(transformed).all():
        raise ValueError("nonfinite channel query")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _dense_score(
    query: np.ndarray,
    references: np.ndarray,
    calibration: ChannelCalibration,
    channel_scorer: Callable[[object, object, int], float],
) -> tuple[float, float, bool]:
    try:
        score = channel_scorer(query, references, FROZEN_CONFIG.neighbor_count)
    except Exception as error:
        raise ValueError("dense channel scorer failed") from error
    if isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(float(score)):
        raise ValueError("nonfinite channel score")
    threshold = calibration.threshold
    if threshold is None or not np.isfinite(threshold):
        raise ValueError("nonfinite channel threshold")
    score_value, threshold_value = float(score), float(threshold)
    return score_value, threshold_value, bool(score_value > threshold_value)


def _score_standardized_channel(
    values: np.ndarray,
    references: np.ndarray | None,
    calibration: ChannelCalibration,
    channel_scorer: Callable[[object, object, int], float],
) -> tuple[float | None, float | None, bool, str | None]:
    if not calibration.available:
        return None, None, False, calibration.reason
    if references is None:
        raise ValueError("available channel lacks reference matrix")
    score, threshold, abnormal = _dense_score(
        _standardize_query(values, calibration), references, calibration, channel_scorer
    )
    return score, threshold, abnormal, None


def _score_acf_channel(
    values: np.ndarray,
    references: np.ndarray | None,
    calibration: ChannelCalibration,
    channel_scorer: Callable[[object, object, int], float],
) -> tuple[float | None, float | None, bool, str | None]:
    if not calibration.available:
        return None, None, False, calibration.reason
    if references is None:
        raise ValueError("available ACF channel lacks reference matrix")
    vector = acf_vector(values, FROZEN_CONFIG.max_acf_lag, FROZEN_CONFIG.variance_floor)
    if vector is None:
        return None, None, False, "zero_acf_energy"
    score, threshold, abnormal = _dense_score(vector, references, calibration, channel_scorer)
    return score, threshold, abnormal, None


def score_test_instance_cpu(
    test_series: RawValueSeries,
    calibrated_groups: Sequence[TrainGroup] | TrainGroup,
    lanes: LaneMembership,
    *,
    channel_scorer: Callable[[object, object, int], float] = score_channel_cpu,
) -> InstanceScoreEvidence:
    """Score one raw Exp436 TEST row with exact-length TRAIN evidence."""

    values = _raw_float32_vector(test_series)
    groups = _normalise_groups(calibrated_groups)
    dataset, by_length = _dataset_and_groups(groups)
    if not hasattr(lanes, "contains"):
        raise ValueError("lanes must provide contains(dataset, instance_index)")
    if values is None:
        return _routing_abstention(dataset, test_series, "nonfinite_test")
    if test_series.length < FROZEN_CONFIG.min_scored_length:
        return _routing_abstention(dataset, test_series, "insufficient_series_length")
    group = by_length.get(test_series.length)
    if group is None:
        return _routing_abstention(dataset, test_series, "missing_length")
    if isinstance(group, UnavailableTrainGroup):
        return _routing_abstention(dataset, test_series, "insufficient_finite_train_rows")

    prototype_score, prototype_threshold, prototype_abnormal, prototype_reason = _score_standardized_channel(
        values, group.values, group.prototype, channel_scorer
    )
    if group.gradient.available:
        with np.errstate(over="ignore", invalid="ignore"):
            gradient_values = np.asarray(np.diff(values, prepend=values[:1]), dtype=np.float32)
        if gradient_values.ndim != 1 or not np.isfinite(gradient_values).all():
            raise ValueError("nonfinite gradient query")
        gradient_score, gradient_threshold, gradient_abnormal, gradient_reason = _score_standardized_channel(
            gradient_values, group.gradients, group.gradient, channel_scorer
        )
    else:
        gradient_score, gradient_threshold, gradient_abnormal, gradient_reason = _score_standardized_channel(
            values, group.gradients, group.gradient, channel_scorer
        )
    acf_score, acf_threshold, acf_abnormal, acf_reason = _score_acf_channel(
        values, group.acf_values, group.acf, channel_scorer
    )

    if prototype_reason or gradient_reason or acf_reason:
        return InstanceScoreEvidence(
            dataset=dataset,
            instance_index=test_series.instance_index,
            length=test_series.length,
            prototype_score=prototype_score,
            prototype_threshold=prototype_threshold,
            prototype_abnormal=prototype_abnormal,
            prototype_unavailability_reason=prototype_reason,
            gradient_score=gradient_score,
            gradient_threshold=gradient_threshold,
            gradient_abnormal=gradient_abnormal,
            gradient_unavailability_reason=gradient_reason,
            acf_score=acf_score,
            acf_threshold=acf_threshold,
            acf_abnormal=acf_abnormal,
            acf_unavailability_reason=acf_reason,
            status="abstained",
            exclusion_reason="three_channel_consensus_unavailable",
        )

    consensus = bool(prototype_abnormal and gradient_abnormal and acf_abnormal)
    exclusion_reason: str | None = None
    if consensus:
        try:
            if lanes.contains(dataset, test_series.instance_index):
                exclusion_reason = "frozen_lane"
        except Exception:
            exclusion_reason = "lane_check_error"
    return InstanceScoreEvidence(
        dataset=dataset,
        instance_index=test_series.instance_index,
        length=test_series.length,
        prototype_score=prototype_score,
        prototype_threshold=prototype_threshold,
        prototype_abnormal=prototype_abnormal,
        gradient_score=gradient_score,
        gradient_threshold=gradient_threshold,
        gradient_abnormal=gradient_abnormal,
        acf_score=acf_score,
        acf_threshold=acf_threshold,
        acf_abnormal=acf_abnormal,
        consensus_candidate=consensus,
        status="scored",
        exclusion_reason=exclusion_reason,
    )


def score_dataset_cpu(
    calibrated_groups: Sequence[TrainGroup] | TrainGroup,
    test: MissingSafeRawDataset,
    lanes: LaneMembership,
    *,
    channel_scorer: Callable[[object, object, int], float] = score_channel_cpu,
) -> Exp436DatasetResult:
    """Score every exact TEST identity and publish only eligible consensus rows."""

    if not isinstance(test, MissingSafeRawDataset) or test.split != "TEST":
        raise ValueError("test must be a MissingSafeRawDataset with split TEST")
    groups = _normalise_groups(calibrated_groups)
    dataset, _ = _dataset_and_groups(groups)
    if dataset != test.dataset:
        raise ValueError("calibrated group dataset does not match TEST dataset")
    evidence = tuple(sorted((
        score_test_instance_cpu(row, groups, lanes, channel_scorer=channel_scorer)
        for row in test.series
    ), key=lambda item: item.instance_index))
    candidates = tuple(
        Exp436Candidate(item.dataset, item.instance_index, item.length, item.evidence_digest)
        for item in evidence
        if item.status == "scored" and item.consensus_candidate and item.exclusion_reason is None
    )
    if not any(item.status == "scored" for item in evidence):
        return Exp436DatasetResult(
            dataset=test.dataset,
            status="abstained",
            test_instance_indices=test.instance_indices,
            evidence=evidence,
            abstention_reason="all_instances_abstained",
        )
    return Exp436DatasetResult(
        dataset=test.dataset,
        status="scored",
        test_instance_indices=test.instance_indices,
        candidates=candidates,
        evidence=evidence,
    )


def score_dataset_backend(
    calibrated_groups: Sequence[TrainGroup] | TrainGroup,
    test: MissingSafeRawDataset,
    lanes: LaneMembership,
    channel_scorer: Callable[[object, object, int], float],
) -> Exp436DatasetResult:
    """Run shared scoring with only the dense scorer supplied by a backend."""

    if not callable(channel_scorer):
        raise ValueError("channel_scorer must be callable")
    return score_dataset_cpu(calibrated_groups, test, lanes, channel_scorer=channel_scorer)


def validate_global_candidates(results: Sequence[Exp436DatasetResult]) -> tuple[Exp436Candidate, ...]:
    """Canonical-sort and reject duplicate or over-budget Exp436 candidates."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("results must be a sequence")
    if any(not isinstance(result, Exp436DatasetResult) for result in results):
        raise ValueError("results must contain Exp436DatasetResult records")
    candidates = tuple(sorted(
        (candidate for result in results for candidate in result.candidates),
        key=lambda item: item.coordinate,
    ))
    if len({candidate.coordinate for candidate in candidates}) != len(candidates):
        raise ValueError("Exp436 candidates contain a duplicate coordinate")
    if len(candidates) > FROZEN_CONFIG.global_candidate_budget:
        raise CandidateBudgetExceeded(observed=len(candidates), budget=FROZEN_CONFIG.global_candidate_budget)
    return candidates


__all__ = [
    "LaneMembership",
    "CandidateBudgetExceeded",
    "score_test_instance_cpu",
    "score_dataset_cpu",
    "score_dataset_backend",
    "validate_global_candidates",
]
