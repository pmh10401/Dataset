"""CPU TEST scoring and fail-closed candidate publication for Exp435.

The scorer consumes only calibrated TRAIN groups and values-only TEST records.
Every TEST row is routed to one exact raw length group; rows are never joined or
resized.  A candidate is published only after the fixed prototype AND
secondary-channel consensus and frozen-lane exclusion have both passed.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Callable, Protocol

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTestDataset

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CalibratedTrainGroup,
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
    TrainGroup,
    UnavailableTrainGroup,
)
from .features_antigravity import acf_vector, score_channel_cpu


class LaneMembership(Protocol):
    """The one lane operation needed by prediction (also permits test doubles)."""

    def contains(self, dataset: str, instance_index: int) -> bool:
        ...


class CandidateBudgetExceeded(ValueError):
    """Raised before publication when the fixed global candidate budget is exceeded."""

    def __init__(self, *, observed: int, budget: int) -> None:
        self.observed = observed
        self.budget = budget
        super().__init__(
            f"Exp435 candidate budget exceeded: observed {observed}, budget {budget}"
        )


def _normalise_groups(
    groups: Sequence[TrainGroup] | TrainGroup,
) -> tuple[TrainGroup, ...]:
    """Accept one exact-length group or the ordered collection used by runners."""

    if isinstance(groups, (CalibratedTrainGroup, UnavailableTrainGroup)):
        return (groups,)
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("calibrated_groups must be a sequence")
    return tuple(groups)


def _as_test_vector(series: RawSeries) -> np.ndarray:
    if not isinstance(series, RawSeries):
        raise ValueError("test_series must be a RawSeries")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(series.values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("TEST series must be finite float32 values")
    return np.ascontiguousarray(values, dtype=np.float32)


def _groups_by_length(
    groups: Sequence[TrainGroup], dataset: str
) -> dict[int, TrainGroup]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("calibrated_groups must be a sequence")
    by_length: dict[int, TrainGroup] = {}
    for group in groups:
        if not isinstance(group, (CalibratedTrainGroup, UnavailableTrainGroup)):
            raise ValueError("calibrated_groups contain an invalid group")
        if group.dataset != dataset:
            raise ValueError("calibrated group dataset does not match TEST dataset")
        if group.length in by_length:
            raise ValueError("calibrated_groups contain duplicate lengths")
        by_length[group.length] = group
    return by_length


def _abstention(
    *, dataset: str, instance_index: int, length: int, reason: str
) -> InstanceScoreEvidence:
    return InstanceScoreEvidence(
        dataset=dataset,
        instance_index=instance_index,
        length=length,
        status="abstained",
        exclusion_reason=reason,
    )


def _standardize_query(
    values: np.ndarray, *, center: float | None, scale: float | None
) -> np.ndarray:
    if center is None or scale is None:
        raise ValueError("channel calibration transform is unavailable")
    center_value = np.float32(center)
    scale_value = np.float32(scale)
    if not np.isfinite(center_value) or not np.isfinite(scale_value):
        raise ValueError("channel calibration transform is nonfinite")
    if float(scale_value) <= FROZEN_CONFIG.numeric_floor:
        raise ValueError("channel calibration scale is zero")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = np.asarray((values - center_value) / scale_value, dtype=np.float32)
    if transformed.ndim != 1 or not np.isfinite(transformed).all():
        raise ValueError("channel query is nonfinite")
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _score_channel(
    values: np.ndarray,
    group_values: np.ndarray | None,
    calibration: object,
    *,
    transform: bool,
    channel_scorer: Callable[[object, object, int], float] = score_channel_cpu,
) -> tuple[float | None, float | None, bool, str | None]:
    """Score one available channel, returning scalar evidence or a reason."""

    # ``ChannelCalibration`` is intentionally accessed through its public
    # properties here so a malformed optional channel fails closed locally.
    if not calibration.available or group_values is None:
        return None, None, False, getattr(calibration, "reason", None) or "unavailable"
    try:
        query = (
            _standardize_query(values, center=calibration.center, scale=calibration.scale)
            if transform
            else values
        )
        score = channel_scorer(query, group_values, FROZEN_CONFIG.neighbor_count)
        threshold = calibration.threshold
        if threshold is None or not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError("nonfinite score")
        abnormal = score > threshold
        return float(score), float(threshold), bool(abnormal), None
    except (TypeError, ValueError, FloatingPointError):
        return None, None, False, "nonfinite_test"


def score_test_instance_cpu(
    test_series: RawSeries,
    calibrated_groups: Sequence[TrainGroup] | TrainGroup,
    lanes: LaneMembership,
    *,
    channel_scorer: Callable[[object, object, int], float] = score_channel_cpu,
) -> InstanceScoreEvidence:
    """Score one values-only TEST instance against its exact TRAIN length group.

    ``test_series`` is deliberately a single ``RawSeries`` rather than an array
    of concatenated rows.  The returned evidence retains every channel decision,
    including abstention and lane exclusion, before any candidate is published.
    """

    if not isinstance(test_series, RawSeries):
        raise ValueError("test_series must be a RawSeries")
    if not hasattr(lanes, "contains"):
        raise ValueError("lanes must provide contains(dataset, instance_index)")
    calibrated_groups = _normalise_groups(calibrated_groups)
    length = len(test_series.values)
    values = _as_test_vector(test_series)
    dataset = None
    for group in calibrated_groups:
        if isinstance(group, (CalibratedTrainGroup, UnavailableTrainGroup)):
            dataset = group.dataset
            break
    if dataset is None:
        raise ValueError("calibrated_groups must contain a dataset group")
    by_length = _groups_by_length(calibrated_groups, dataset)
    group = by_length.get(length)
    if group is None:
        return _abstention(
            dataset=dataset,
            instance_index=test_series.instance_index,
            length=length,
            reason="missing_length",
        )
    if isinstance(group, UnavailableTrainGroup):
        return _abstention(
            dataset=dataset,
            instance_index=test_series.instance_index,
            length=length,
            reason=group.reason,
        )

    prototype_score, prototype_threshold, prototype_abnormal, prototype_reason = _score_channel(
        values,
        group.values,
        group.prototype,
        transform=True,
        channel_scorer=channel_scorer,
    )
    if prototype_reason is not None:
        return _abstention(
            dataset=dataset,
            instance_index=test_series.instance_index,
            length=length,
            reason="prototype_unavailable",
        )

    gradient_values: np.ndarray | None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            gradient_values = np.asarray(
                np.diff(values, prepend=values[:1]), dtype=np.float32
            )
        if gradient_values.ndim != 1 or not np.isfinite(gradient_values).all():
            raise ValueError("nonfinite gradient")
    except (TypeError, ValueError, FloatingPointError):
        gradient_values = None

    if gradient_values is None:
        # A failed TEST transform is an unavailable gradient channel.  Never
        # compare the raw waveform with the TRAIN gradient references.
        gradient_score, gradient_threshold, gradient_abnormal, gradient_reason = (
            None,
            None,
            False,
            "gradient_unavailable",
        )
    else:
        gradient_score, gradient_threshold, gradient_abnormal, gradient_reason = _score_channel(
            gradient_values,
            group.gradients,
            group.gradient,
            transform=True,
            channel_scorer=channel_scorer,
        )

    acf_values = acf_vector(
        values,
        FROZEN_CONFIG.max_acf_lag,
        FROZEN_CONFIG.numeric_floor,
    )
    acf_score, acf_threshold, acf_abnormal, acf_reason = _score_channel(
        acf_values if acf_values is not None else values,
        group.acf_values,
        group.acf,
        transform=False,
        channel_scorer=channel_scorer,
    )

    if gradient_score is None and acf_score is None:
        # Keep every finite channel decision even when the secondary quorum
        # cannot be formed.  This is explicit per-instance abstention, not a
        # silent drop of the available prototype evidence.
        return InstanceScoreEvidence(
            dataset=dataset,
            instance_index=test_series.instance_index,
            length=length,
            prototype_score=prototype_score,
            prototype_threshold=prototype_threshold,
            prototype_abnormal=prototype_abnormal,
            status="abstained",
            exclusion_reason="secondary_channels_unavailable",
        )

    consensus = bool(prototype_abnormal and (gradient_abnormal or acf_abnormal))
    exclusion_reason: str | None = None
    if consensus:
        try:
            if lanes.contains(dataset, test_series.instance_index):
                exclusion_reason = "frozen_lane"
        except Exception:
            # Lane membership is a scientific boundary; an uncertain answer is
            # never allowed to publish a candidate.
            exclusion_reason = "lane_check_error"

    return InstanceScoreEvidence(
        dataset=dataset,
        instance_index=test_series.instance_index,
        length=length,
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
    test: RawTestDataset,
    lanes: LaneMembership,
    *,
    channel_scorer: Callable[[object, object, int], float] = score_channel_cpu,
) -> Exp435DatasetResult:
    """Score all TEST rows independently and retain ordered evidence."""

    if not isinstance(test, RawTestDataset):
        raise ValueError("test must be a RawTestDataset")
    if not hasattr(lanes, "contains"):
        raise ValueError("lanes must provide contains(dataset, instance_index)")
    calibrated_groups = _normalise_groups(calibrated_groups)
    if any(
        not isinstance(group, (CalibratedTrainGroup, UnavailableTrainGroup))
        for group in calibrated_groups
    ):
        raise ValueError("calibrated_groups contain an invalid group")
    if any(group.dataset != test.dataset for group in calibrated_groups):
        raise ValueError("calibrated group dataset does not match TEST dataset")

    evidence: list[InstanceScoreEvidence] = []
    for series in test.series:
        evidence.append(
            score_test_instance_cpu(
                series,
                calibrated_groups,
                lanes,
                channel_scorer=channel_scorer,
            )
        )
    ordered_evidence = tuple(sorted(evidence, key=lambda item: item.instance_index))
    candidates = tuple(
        Exp435Candidate(
            item.dataset,
            item.instance_index,
            item.length,
            evidence_digest=item.evidence_digest,
        )
        for item in ordered_evidence
        if item.consensus_candidate and item.exclusion_reason is None
    )
    if not any(item.status == "scored" for item in ordered_evidence):
        return Exp435DatasetResult(
            dataset=test.dataset,
            status="abstained",
            test_size=len(test.series),
            candidates=(),
            evidence=ordered_evidence,
            abstention_reason="all_instances_abstained",
        )
    return Exp435DatasetResult(
        dataset=test.dataset,
        status="scored",
        test_size=len(test.series),
        candidates=candidates,
        evidence=ordered_evidence,
    )


def score_dataset_backend(
    calibrated_groups: Sequence[TrainGroup] | TrainGroup,
    test: RawTestDataset,
    lanes: LaneMembership,
    channel_scorer: Callable[[object, object, int], float],
) -> Exp435DatasetResult:
    """Score one dataset with an already-selected channel backend.

    Feature construction, thresholds, routing and consensus remain identical
    to ``score_dataset_cpu``; only the dense channel distance operation is
    injected.  CPU callers should continue using the public CPU wrapper.
    """

    if not callable(channel_scorer):
        raise ValueError("channel_scorer must be callable")
    return score_dataset_cpu(
        calibrated_groups,
        test,
        lanes,
        channel_scorer=channel_scorer,
    )


def validate_global_candidates(
    results: Sequence[Exp435DatasetResult], budget: int = FROZEN_CONFIG.global_candidate_budget
) -> tuple[Exp435Candidate, ...]:
    """Validate canonical coordinate order and reject, never trim, over-budget output."""

    if isinstance(budget, bool) or not isinstance(budget, Integral) or int(budget) < 0:
        raise ValueError("budget must be a non-negative integer")
    candidates = tuple(
        sorted(
            (candidate for result in results for candidate in result.candidates),
            key=lambda item: (item.dataset, item.instance_index),
        )
    )
    if len(candidates) > int(budget):
        raise CandidateBudgetExceeded(observed=len(candidates), budget=int(budget))
    if len({candidate.coordinate for candidate in candidates}) != len(candidates):
        raise ValueError("Exp435 candidates contain a duplicate coordinate")
    return candidates


__all__ = [
    "CandidateBudgetExceeded",
    "score_dataset_cpu",
    "score_dataset_backend",
    "score_test_instance_cpu",
    "validate_global_candidates",
]
