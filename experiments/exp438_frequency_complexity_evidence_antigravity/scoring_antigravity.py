"""TRAIN-only calibration and separate-channel CPU scoring for Exp438."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CHANNELS_ANTIGRAVITY,
    ChannelCalibration,
    Exp438DatasetResult,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
    PreparedTrainGroup,
    RawValueSeries,
)
from .features_antigravity import (
    decompose_at_depth_antigravity,
    select_train_depth_antigravity,
)


class CandidateBudgetExceededAntigravity(ValueError):
    pass


class _UnavailableChannelAntigravity(ValueError):
    pass


@dataclass(frozen=True)
class _FeatureMatricesAntigravity:
    high_frequency: NDArray[np.float32]
    low_frequency: NDArray[np.float32]
    amplitude_complexity: NDArray[np.float32]

    def channel(self, name: str) -> NDArray[np.float32]:
        if name not in CHANNELS_ANTIGRAVITY:
            raise ValueError("feature channel is invalid")
        return getattr(self, name)


@dataclass(frozen=True)
class CalibratedTrainGroupAntigravity:
    dataset: str
    raw_length: int
    depth: int
    calibrations: Mapping[str, ChannelCalibration]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("calibrated dataset is invalid")
        if type(self.raw_length) is not int or self.raw_length < FROZEN_CONFIG.minimum_raw_length:
            raise ValueError("calibrated raw length is invalid")
        if type(self.depth) is not int or self.depth < 1 or self.depth > FROZEN_CONFIG.maximum_depth:
            raise ValueError("calibrated depth is invalid")
        values = dict(self.calibrations)
        if tuple(values) != CHANNELS_ANTIGRAVITY or any(
            type(item) is not ChannelCalibration or item.channel != name
            for name, item in values.items()
        ):
            raise ValueError("calibrations must preserve the three separate channels")
        object.__setattr__(self, "calibrations", MappingProxyType(values))


def _finite_matrix_antigravity(value: object, label: str) -> NDArray[np.float32]:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError(f"{label} is invalid")
    return np.ascontiguousarray(matrix)


def robust_transform_antigravity(
    matrix: object,
    *,
    channel: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    values = _finite_matrix_antigravity(matrix, "TRAIN channel matrix")
    if channel in {"high_frequency", "low_frequency"}:
        center64 = np.asarray([np.median(values.astype(np.float64))], dtype=np.float64)
        deviations = np.abs(values.astype(np.float64) - center64[0])
        scale64 = np.asarray([1.4826 * np.median(deviations)], dtype=np.float64)
        if not math.isfinite(float(scale64[0])) or float(scale64[0]) <= 1e-12:
            raise _UnavailableChannelAntigravity("zero_scale_channel")
    elif channel == "amplitude_complexity":
        matrix64 = values.astype(np.float64)
        center64 = np.median(matrix64, axis=0)
        mad = np.median(np.abs(matrix64 - center64), axis=0)
        scale64 = np.maximum(1.4826 * mad, 1e-6)
    else:
        raise ValueError("channel is invalid")
    standardized = np.ascontiguousarray(
        (values.astype(np.float64) - center64) / scale64,
        dtype=np.float32,
    )
    center = np.ascontiguousarray(center64, dtype=np.float32)
    scale = np.ascontiguousarray(scale64, dtype=np.float32)
    if not np.isfinite(standardized).all() or not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise _UnavailableChannelAntigravity("nonfinite_transform")
    standardized.flags.writeable = False
    center.flags.writeable = False
    scale.flags.writeable = False
    return standardized, center, scale


def _standardize_query_antigravity(
    query: object,
    center: NDArray[np.float32],
    scale: NDArray[np.float32],
) -> NDArray[np.float32]:
    try:
        vector = np.asarray(query, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("query feature is invalid") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("query feature is invalid")
    if center.ndim != 1 or scale.shape != center.shape or center.size not in {1, vector.size}:
        raise ValueError("query transform shape is invalid")
    result = np.ascontiguousarray(
        (vector.astype(np.float64) - center.astype(np.float64))
        / scale.astype(np.float64),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("standardized query is nonfinite")
    result.flags.writeable = False
    return result


def score_channel_cpu_antigravity(
    query: object,
    references: object,
    k: int,
) -> float:
    try:
        vector = np.asarray(query, dtype=np.float32)
        matrix = np.asarray(references, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel score input is invalid") from exc
    if (
        vector.ndim != 1
        or matrix.ndim != 2
        or matrix.shape[1] != vector.size
        or type(k) is not int
        or k < 1
        or matrix.shape[0] < k
        or not np.isfinite(vector).all()
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("channel score input is invalid")
    delta = matrix.astype(np.float64) - vector.astype(np.float64)
    distances = np.sqrt(np.mean(delta * delta, axis=1))
    nearest = np.partition(distances, k - 1)[:k]
    result = float(np.median(nearest))
    if not math.isfinite(result):
        raise ValueError("channel score is nonfinite")
    return result


def higher_quantile_antigravity(scores: object, quantile: float) -> float:
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantile scores are invalid") from exc
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile scores are invalid")
    if isinstance(quantile, bool) or not math.isfinite(float(quantile)) or not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile is invalid")
    result = float(np.quantile(values, float(quantile), method="higher"))
    if not math.isfinite(result):
        raise ValueError("quantile result is nonfinite")
    return result


def _feature_matrices_antigravity(matrix: NDArray[np.float32], depth: int) -> _FeatureMatricesAntigravity:
    decomposed = tuple(decompose_at_depth_antigravity(row, depth) for row in matrix)
    high = np.ascontiguousarray(np.stack([item.high for item in decomposed]), dtype=np.float32)
    low = np.ascontiguousarray(np.stack([item.low for item in decomposed]), dtype=np.float32)
    complexity = np.ascontiguousarray(
        np.stack([item.amplitude_complexity for item in decomposed]),
        dtype=np.float32,
    )
    for value in (high, low, complexity):
        value.flags.writeable = False
    return _FeatureMatricesAntigravity(high, low, complexity)


def _query_features_antigravity(values: NDArray[np.float32], depth: int) -> Mapping[str, NDArray[np.float32]]:
    decomposed = decompose_at_depth_antigravity(values, depth)
    return MappingProxyType(
        {
            "high_frequency": decomposed.high,
            "low_frequency": decomposed.low,
            "amplitude_complexity": decomposed.amplitude_complexity,
        }
    )


def fit_calibrated_group_antigravity(
    group: PreparedTrainGroup,
) -> CalibratedTrainGroupAntigravity:
    if type(group) is not PreparedTrainGroup:
        raise ValueError("calibration requires a prepared TRAIN group")
    if group.matrix.shape[0] < FROZEN_CONFIG.minimum_train_rows:
        raise ValueError("calibration has insufficient finite TRAIN rows")
    final_depth = select_train_depth_antigravity(group.matrix)
    final_features = _feature_matrices_antigravity(group.matrix, final_depth)
    final_states: dict[
        str,
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | str,
    ] = {}
    for channel in CHANNELS_ANTIGRAVITY:
        try:
            final_states[channel] = robust_transform_antigravity(
                final_features.channel(channel),
                channel=channel,
            )
        except _UnavailableChannelAntigravity as exc:
            final_states[channel] = str(exc)

    oof_scores: dict[str, list[float]] = {channel: [] for channel in CHANNELS_ANTIGRAVITY}
    oof_failures: dict[str, str] = {}
    for held_out in range(group.matrix.shape[0]):
        references_raw = np.delete(group.matrix, held_out, axis=0)
        depth = select_train_depth_antigravity(references_raw)
        reference_features = _feature_matrices_antigravity(references_raw, depth)
        query_features = _query_features_antigravity(group.matrix[held_out], depth)
        for channel in CHANNELS_ANTIGRAVITY:
            if channel in oof_failures:
                continue
            try:
                references, center, scale = robust_transform_antigravity(
                    reference_features.channel(channel),
                    channel=channel,
                )
                query = _standardize_query_antigravity(query_features[channel], center, scale)
                oof_scores[channel].append(
                    score_channel_cpu_antigravity(query, references, FROZEN_CONFIG.nearest_k)
                )
            except (_UnavailableChannelAntigravity, ValueError) as exc:
                oof_failures[channel] = str(exc) or "oof_channel_unavailable"

    calibrations: dict[str, ChannelCalibration] = {}
    for channel in CHANNELS_ANTIGRAVITY:
        state = final_states[channel]
        if isinstance(state, str):
            calibrations[channel] = ChannelCalibration(
                channel, None, None, None, None, None, False, state
            )
            continue
        if channel in oof_failures or len(oof_scores[channel]) != group.matrix.shape[0]:
            calibrations[channel] = ChannelCalibration(
                channel,
                None,
                None,
                None,
                None,
                None,
                False,
                oof_failures.get(channel, "incomplete_oof_scores"),
            )
            continue
        references, center, scale = state
        oof = np.ascontiguousarray(oof_scores[channel], dtype=np.float32)
        oof.flags.writeable = False
        threshold = higher_quantile_antigravity(oof, FROZEN_CONFIG.quantile)
        calibrations[channel] = ChannelCalibration(
            channel=channel,
            references=references,
            center=center,
            scale=scale,
            oof_scores=oof,
            threshold=threshold,
            available=True,
            reason=None,
        )
    return CalibratedTrainGroupAntigravity(
        dataset=group.dataset,
        raw_length=group.raw_length,
        depth=final_depth,
        calibrations=calibrations,
    )


def score_test_row_cpu_antigravity(
    row: RawValueSeries,
    calibrated: CalibratedTrainGroupAntigravity,
    *,
    outside_all_frozen_lanes: bool,
) -> InstanceScoreEvidence:
    if type(row) is not RawValueSeries or row.split != "TEST":
        raise ValueError("CPU TEST scoring requires an Exp438 TEST row")
    if type(calibrated) is not CalibratedTrainGroupAntigravity or row.dataset != calibrated.dataset:
        raise ValueError("TEST row does not match the calibrated dataset")
    if row.raw_length != calibrated.raw_length:
        raise ValueError("TEST row does not have the exact TRAIN length")
    if type(outside_all_frozen_lanes) is not bool:
        raise ValueError("lane membership is invalid")
    query_features = _query_features_antigravity(row.values, calibrated.depth)
    evidence: list[InstanceChannelEvidence] = []
    for channel in CHANNELS_ANTIGRAVITY:
        calibration = calibrated.calibrations[channel]
        if not calibration.available:
            evidence.append(
                InstanceChannelEvidence(
                    channel, None, None, False, False, calibration.reason
                )
            )
            continue
        assert calibration.center is not None
        assert calibration.scale is not None
        assert calibration.references is not None
        assert calibration.threshold is not None
        query = _standardize_query_antigravity(
            query_features[channel],
            calibration.center,
            calibration.scale,
        )
        score = score_channel_cpu_antigravity(
            query,
            calibration.references,
            FROZEN_CONFIG.nearest_k,
        )
        evidence.append(
            InstanceChannelEvidence(
                channel=channel,
                score=score,
                threshold=calibration.threshold,
                abnormal=score > calibration.threshold,
                available=True,
                reason=None,
            )
        )
    return InstanceScoreEvidence(
        dataset=row.dataset,
        instance_index=row.instance_index,
        raw_length=row.raw_length,
        channels=tuple(evidence),
        outside_all_frozen_lanes=outside_all_frozen_lanes,
    )


def validate_global_candidates_antigravity(
    results: Iterable[Exp438DatasetResult],
) -> Mapping[str, int]:
    counts = {channel: 0 for channel in CHANNELS_ANTIGRAVITY}
    for result in results:
        if type(result) is not Exp438DatasetResult:
            raise ValueError("global candidate validation requires Exp438 results")
        for channel, coordinates in result.candidate_coordinates.items():
            counts[channel] += len(coordinates)
            if counts[channel] > FROZEN_CONFIG.channel_candidate_budget:
                raise CandidateBudgetExceededAntigravity(
                    f"{channel} candidate budget exceeded"
                )
    total = sum(counts.values())
    if total > FROZEN_CONFIG.total_candidate_budget:
        raise CandidateBudgetExceededAntigravity("total candidate budget exceeded")
    return MappingProxyType(counts)


__all__ = [
    "CalibratedTrainGroupAntigravity",
    "CandidateBudgetExceededAntigravity",
    "fit_calibrated_group_antigravity",
    "higher_quantile_antigravity",
    "robust_transform_antigravity",
    "score_channel_cpu_antigravity",
    "score_test_row_cpu_antigravity",
    "validate_global_candidates_antigravity",
]
