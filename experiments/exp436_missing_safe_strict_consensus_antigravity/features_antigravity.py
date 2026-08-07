"""Missing-safe, TRAIN-only float32 feature preparation for Exp436.

Only source rows containing nonfinite float32 values are removed.  Once a
finite row reaches calibration, numerical failures are scientific failures,
not a second filtering mechanism.
"""

from __future__ import annotations

from hashlib import sha256
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    ExcludedTrainRowAudit,
    MissingSafeRawDataset,
    PreparedTrainGroup,
    TrainGroup,
    TrainPreparation,
    UnavailableTrainGroup,
)


def _as_float32_matrix(value: object, label: str) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains nonfinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_float32_vector(value: object, label: str) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains nonfinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _canonical_matrix_digest(matrix: NDArray[np.float32]) -> str:
    array = np.asarray(matrix, dtype="<f4", order="C")
    if array.ndim != 2:
        raise ValueError("reference matrix must be two-dimensional")
    rows = sorted(row.tobytes(order="C") for row in array)
    payload = (
        int(array.shape[0]).to_bytes(8, "little", signed=False)
        + int(array.shape[1]).to_bytes(8, "little", signed=False)
        + b"".join(rows)
    )
    return sha256(payload).hexdigest()


def _center_scale(matrix: NDArray[np.float32], channel: str) -> tuple[np.float32, np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        center = np.float32(np.median(matrix))
        scale = np.float32(
            np.float32(FROZEN_CONFIG.robust_scale_multiplier)
            * np.median(np.abs(matrix - center))
        )
    if not np.isfinite(center) or not np.isfinite(scale):
        raise ValueError(f"nonfinite {channel} center or scale")
    return center, scale


def _standardize(
    matrix: NDArray[np.float32], center: np.float32, scale: np.float32, channel: str
) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.asarray((matrix - center) / scale, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"nonfinite {channel} standardized matrix")
    return np.ascontiguousarray(result, dtype=np.float32)


def score_channel_cpu(query: object, references: object, k: int) -> float:
    """Approved Exp435 float32 RMS median-nearest-neighbor score."""

    if not isinstance(k, Integral) or isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    query_array = _as_float32_vector(query, "query")
    reference_array = _as_float32_matrix(references, "references")
    if reference_array.shape[1] != query_array.shape[0]:
        raise ValueError("query and references have incompatible lengths")
    if reference_array.shape[0] < int(k):
        raise ValueError("k cannot exceed reference count")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(reference_array - query_array[None, :], dtype=np.float32)
        distances = np.sqrt(
            np.mean(delta * delta, axis=1, dtype=np.float32), dtype=np.float32
        )
    if not np.isfinite(distances).all():
        raise ValueError("score is nonfinite")
    score = np.float32(np.median(np.partition(distances, int(k) - 1)[: int(k)]))
    if not np.isfinite(score):
        raise ValueError("score is nonfinite")
    return float(score)


def acf_vector(values: object, max_lag: int, floor: float) -> NDArray[np.float32] | None:
    """Approved Exp435 centered float32 ACF, with numerical failure explicit."""

    if not isinstance(max_lag, Integral) or isinstance(max_lag, bool) or int(max_lag) < 1:
        return None
    if isinstance(floor, bool) or not isinstance(floor, Real) or not np.isfinite(float(floor)):
        return None
    array = _as_float32_vector(values, "values")
    if array.shape[0] < FROZEN_CONFIG.min_acf_length:
        return None
    lag_count = min(int(max_lag), FROZEN_CONFIG.max_acf_lag, int(array.shape[0]) // 4)
    if lag_count < 1:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.float32(np.mean(array, dtype=np.float32))
        centered = np.asarray(array - mean, dtype=np.float32)
        denominator = np.float32(np.sum(centered * centered, dtype=np.float32))
    if not np.isfinite(mean) or not np.isfinite(centered).all() or not np.isfinite(denominator):
        raise ValueError("nonfinite acf arithmetic")
    if float(denominator) <= float(floor):
        return None
    result = np.empty(lag_count, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for offset in range(1, lag_count + 1):
            numerator = np.float32(np.sum(centered[:-offset] * centered[offset:], dtype=np.float32))
            result[offset - 1] = np.float32(numerator / denominator)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite acf vector")
    return np.ascontiguousarray(result, dtype=np.float32)


def prepare_train_groups(train: MissingSafeRawDataset) -> TrainPreparation:
    """Partition every original TRAIN coordinate by exact float32 finiteness."""

    if not isinstance(train, MissingSafeRawDataset):
        raise ValueError("train must be a MissingSafeRawDataset")
    if train.split != "TRAIN":
        raise ValueError("train split must be TRAIN")
    FROZEN_CONFIG.validate()
    manifest = tuple((series.instance_index, series.length) for series in train.series)
    finite_by_length: dict[int, list[tuple[int, NDArray[np.float32]]]] = {}
    all_lengths: set[int] = set()
    excluded: list[ExcludedTrainRowAudit] = []
    for series in train.series:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(series.values, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] != series.length:
            raise ValueError("TRAIN row length changed during float32 conversion")
        length = int(values.shape[0])
        all_lengths.add(length)
        nonfinite_count = int(values.size - np.count_nonzero(np.isfinite(values)))
        if nonfinite_count:
            excluded.append(ExcludedTrainRowAudit(
                train.dataset, series.instance_index, length, nonfinite_count, series.raw_blob_sha256
            ))
        else:
            finite_by_length.setdefault(length, []).append((series.instance_index, np.ascontiguousarray(values, dtype=np.float32)))
    groups: list[TrainGroup] = []
    for length in sorted(all_lengths):
        rows = sorted(finite_by_length.get(length, ()), key=lambda item: item[0])
        indices = tuple(index for index, _ in rows)
        if len(rows) < FROZEN_CONFIG.min_finite_train_rows:
            groups.append(UnavailableTrainGroup(train.dataset, length, indices))
        else:
            matrix = np.asarray([values for _, values in rows], dtype=np.float32)
            groups.append(PreparedTrainGroup(train.dataset, length, indices, matrix))
    return TrainPreparation(
        train.dataset,
        manifest,
        tuple(groups),
        tuple(sorted(excluded, key=lambda item: (item.length, item.instance_index))),
    )


def _loo_scores(matrix: NDArray[np.float32]) -> tuple[float, ...]:
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        references = np.concatenate((matrix[:index], matrix[index + 1 :]), axis=0)
        scores.append(score_channel_cpu(matrix[index], references, FROZEN_CONFIG.neighbor_count))
    if not np.isfinite(np.asarray(scores, dtype=np.float32)).all():
        raise ValueError("nonfinite LOO score")
    return tuple(scores)


def _available_calibration(channel: str, matrix: NDArray[np.float32], center: np.float32, scale: np.float32) -> ChannelCalibration:
    scores = _loo_scores(matrix)
    threshold = float(np.quantile(
        np.asarray(scores, dtype=np.float32), FROZEN_CONFIG.train_quantile,
        method=FROZEN_CONFIG.quantile_method,
    ))
    if not np.isfinite(threshold):
        raise ValueError(f"nonfinite {channel} threshold")
    return ChannelCalibration(
        channel=channel, threshold=threshold, loo_scores=scores,
        center=float(center), scale=float(scale), reference_count=matrix.shape[0],
        reference_digest=_canonical_matrix_digest(matrix),
    )


def _fit_standardized_channel(
    channel: str, matrix: NDArray[np.float32]
) -> tuple[NDArray[np.float32] | None, ChannelCalibration]:
    center, scale = _center_scale(matrix, channel)
    if float(scale) <= FROZEN_CONFIG.numeric_floor:
        return None, ChannelCalibration.unavailable(channel, "zero_scale")
    standardized = _standardize(matrix, center, scale, channel)
    return standardized, _available_calibration(channel, standardized, center, scale)


def _fit_acf_channel(matrix: NDArray[np.float32]) -> tuple[NDArray[np.float32] | None, ChannelCalibration]:
    if matrix.shape[1] < FROZEN_CONFIG.min_acf_length:
        return None, ChannelCalibration.unavailable("acf", "short_series")
    vectors: list[NDArray[np.float32]] = []
    for row in matrix:
        vector = acf_vector(row, FROZEN_CONFIG.max_acf_lag, FROZEN_CONFIG.variance_floor)
        if vector is None:
            return None, ChannelCalibration.unavailable("acf", "zero_acf_energy")
        vectors.append(vector)
    acf_matrix = np.asarray(vectors, dtype=np.float32)
    if acf_matrix.ndim != 2 or not np.isfinite(acf_matrix).all():
        raise ValueError("nonfinite acf matrix")
    acf_matrix = np.ascontiguousarray(acf_matrix, dtype=np.float32)
    return acf_matrix, _available_calibration("acf", acf_matrix, np.float32(0.0), np.float32(1.0))


def fit_calibrated_group(group: TrainGroup) -> CalibratedTrainGroup | UnavailableTrainGroup:
    """Independently calibrate prototype, gradient, and ACF from finite TRAIN rows."""

    if isinstance(group, UnavailableTrainGroup):
        return group
    if not isinstance(group, PreparedTrainGroup):
        raise ValueError("group must be a PreparedTrainGroup or UnavailableTrainGroup")
    FROZEN_CONFIG.validate()
    matrix = _as_float32_matrix(group.values, "group values")
    if matrix.shape != (group.reference_count, group.length):
        raise ValueError("group row count does not match instance indices")
    prototype_values, prototype = _fit_standardized_channel("prototype", matrix)
    with np.errstate(over="ignore", invalid="ignore"):
        gradient_matrix = np.asarray(np.diff(matrix, axis=1, prepend=matrix[:, :1]), dtype=np.float32)
    if gradient_matrix.ndim != 2 or not np.isfinite(gradient_matrix).all():
        raise ValueError("nonfinite gradient matrix")
    gradients, gradient = _fit_standardized_channel("gradient", np.ascontiguousarray(gradient_matrix, dtype=np.float32))
    acf_values, acf = _fit_acf_channel(matrix)
    return CalibratedTrainGroup(
        group.dataset, group.length, group.instance_indices, prototype_values,
        gradients, acf_values, prototype, gradient, acf,
    )


__all__ = ["acf_vector", "fit_calibrated_group", "prepare_train_groups", "score_channel_cpu"]
