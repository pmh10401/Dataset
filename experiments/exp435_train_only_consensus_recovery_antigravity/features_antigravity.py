"""Deterministic TRAIN-only feature preparation and CPU calibration for Exp435.

This module has no TEST-label boundary and owns only numerical preparation.  A
length group is formed from the values of normal TRAIN rows, then every
transform and threshold is fitted from that group alone.  The LOO reference
set removes exactly one row; duplicate-valued rows remain in the reference
set and therefore remain valid normal evidence.
"""

from __future__ import annotations

from hashlib import sha256
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    RawTrainDataset,
)

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CalibratedTrainGroup,
    ChannelCalibration,
    PreparedTrainGroup,
    TrainGroup,
    UnavailableTrainGroup,
)


_MAD_TO_NORMAL_SCALE = np.float32(1.4826)


def _is_integral(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _as_float32_vector(values: object, label: str) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains nonfinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_float32_matrix(value: object, label: str) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains nonfinite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _robust_center_scale(matrix: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
    """Return the preregistered shared float32 median/MAD transform."""

    with np.errstate(over="ignore", invalid="ignore"):
        center = np.float32(np.median(matrix))
        deviations = np.abs(matrix - center)
        scale = np.float32(_MAD_TO_NORMAL_SCALE * np.median(deviations))
    return center, scale


def _scale_reason(scale: object) -> str | None:
    try:
        number = float(scale)
    except (TypeError, ValueError):
        return "nonfinite_train"
    if not np.isfinite(number):
        return "nonfinite_train"
    if number <= FROZEN_CONFIG.numeric_floor:
        return "zero_scale"
    return None


def _standardize(
    matrix: NDArray[np.float32], center: np.float32, scale: np.float32
) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = np.asarray((matrix - center) / scale, dtype=np.float32)
    if standardized.ndim != 2 or not np.isfinite(standardized).all():
        raise ValueError("nonfinite_train")
    return np.ascontiguousarray(standardized, dtype=np.float32)


def _canonical_matrix_digest(matrix: NDArray[np.float32]) -> str:
    """Hash a row-order-independent little-endian float32 matrix."""

    array = np.asarray(matrix, dtype="<f4", order="C")
    if array.ndim != 2:
        raise ValueError("reference matrix must be two-dimensional")
    rows = [row.tobytes(order="C") for row in array]
    rows.sort()
    payload = (
        int(array.shape[0]).to_bytes(8, "little", signed=False)
        + int(array.shape[1]).to_bytes(8, "little", signed=False)
        + b"".join(rows)
    )
    return sha256(payload).hexdigest()


def _unavailable_group(
    dataset: str,
    length: int,
    instance_indices: tuple[int, ...],
    reason: str,
) -> UnavailableTrainGroup:
    return UnavailableTrainGroup(
        dataset=dataset,
        length=max(1, int(length)),
        instance_indices=tuple(instance_indices),
        reason=reason,
    )


def _validate_group_matrix(
    group: PreparedTrainGroup,
) -> tuple[NDArray[np.float32], int, int] | UnavailableTrainGroup:
    """Validate an in-memory group while preserving a structured failure."""

    try:
        matrix = _as_float32_matrix(group.values, "group values")
    except ValueError:
        return _unavailable_group(
            group.dataset,
            group.length,
            group.instance_indices,
            "nonfinite_train",
        )
    rows, length = matrix.shape
    if length < FROZEN_CONFIG.min_series_length:
        return _unavailable_group(
            group.dataset,
            length,
            group.instance_indices,
            "insufficient_series_length",
        )
    if rows != group.reference_count or rows != len(group.instance_indices):
        raise ValueError("group row count does not match instance indices")
    if rows < FROZEN_CONFIG.min_train_rows:
        return _unavailable_group(
            group.dataset,
            length,
            group.instance_indices,
            "insufficient_train_rows",
        )
    return matrix, rows, length


def prepare_train_groups(train: RawTrainDataset) -> tuple[TrainGroup, ...]:
    """Group TRAIN rows by exact raw length without suppressing siblings.

    Invalid groups are returned as immutable ``UnavailableTrainGroup`` records
    so downstream scoring can abstain only their exact coordinates.  Valid
    groups remain ordinary ``PreparedTrainGroup`` records.
    """

    if not isinstance(train, RawTrainDataset):
        raise ValueError("train must be a RawTrainDataset")
    FROZEN_CONFIG.validate()
    by_length: dict[int, list[tuple[int, tuple[float, ...], bool]]] = {}
    for series in train.series:
        length = len(series.values)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(series.values, dtype=np.float32)
        finite = values.ndim == 1 and len(values) == length and bool(np.isfinite(values).all())
        by_length.setdefault(length, []).append(
            (
                series.instance_index,
                tuple(float(value) for value in values),
                finite,
            )
        )

    groups: list[TrainGroup] = []
    for length in sorted(by_length):
        rows = sorted(by_length[length], key=lambda item: item[0])
        indices = tuple(index for index, _, _ in rows)
        if length < FROZEN_CONFIG.min_series_length:
            groups.append(
                _unavailable_group(
                    train.dataset,
                    length,
                    indices,
                    "insufficient_series_length",
                )
            )
            continue
        if not all(finite for _, _, finite in rows):
            groups.append(_unavailable_group(train.dataset, length, indices, "nonfinite_train"))
            continue
        if len(rows) < FROZEN_CONFIG.min_train_rows:
            groups.append(_unavailable_group(train.dataset, length, indices, "insufficient_train_rows"))
            continue
        matrix = np.asarray([values for _, values, _ in rows], dtype=np.float32)
        if matrix.shape != (len(rows), length) or not np.isfinite(matrix).all():
            groups.append(_unavailable_group(train.dataset, length, indices, "nonfinite_train"))
            continue
        groups.append(
            PreparedTrainGroup(dataset=train.dataset, length=length, instance_indices=indices, values=matrix)
        )
    return tuple(groups)


def acf_vector(
    values: object, max_lag: int, floor: float
) -> NDArray[np.float32] | None:
    """Compute the centered, energy-normalized ACF lags for one TRAIN row."""

    if not _is_integral(max_lag) or int(max_lag) < 1:
        return None
    try:
        floor_value = _finite_float(floor, "floor")
    except ValueError:
        return None
    array = _as_float32_vector(values, "values")
    length = int(array.shape[0])
    if length < FROZEN_CONFIG.min_acf_length:
        return None
    lag_count = min(int(max_lag), FROZEN_CONFIG.max_acf_lag, length // 4)
    if lag_count < 1:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = np.float32(np.mean(array, dtype=np.float32))
        centered = np.asarray(array - mean, dtype=np.float32)
        denominator = np.float32(np.sum(centered * centered, dtype=np.float32))
    if not np.isfinite(denominator) or float(denominator) <= floor_value:
        return None
    output = np.empty(lag_count, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for offset in range(1, lag_count + 1):
            numerator = np.float32(
                np.sum(centered[:-offset] * centered[offset:], dtype=np.float32)
            )
            output[offset - 1] = np.float32(numerator / denominator)
    if not np.isfinite(output).all():
        return None
    return np.ascontiguousarray(output, dtype=np.float32)


def score_channel_cpu(query: object, references: object, k: int) -> float:
    """Return the median of the nearest ``k`` float32 RMS distances."""

    if not _is_integral(k) or int(k) < 1:
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
            np.mean(delta * delta, axis=1, dtype=np.float32),
            dtype=np.float32,
        )
    if not np.isfinite(distances).all():
        raise ValueError("score is nonfinite")
    nearest = np.partition(distances, int(k) - 1)[: int(k)]
    score = np.float32(np.median(nearest))
    if not np.isfinite(score):
        raise ValueError("score is nonfinite")
    return float(score)


def _loo_scores(
    matrix: NDArray[np.float32], *, k: int
) -> tuple[float, ...]:
    rows = matrix.shape[0]
    if rows < FROZEN_CONFIG.min_train_rows:
        raise ValueError("insufficient_train_rows")
    scores: list[float] = []
    for index in range(rows):
        # Delete exactly the current row.  Equal-valued rows are not deduped.
        references = np.concatenate((matrix[:index], matrix[index + 1 :]), axis=0)
        score = score_channel_cpu(matrix[index], references, k)
        if not np.isfinite(score):
            raise ValueError("nonfinite_train")
        scores.append(float(score))
    if len(scores) < FROZEN_CONFIG.min_train_rows or not np.isfinite(scores).all():
        raise ValueError("insufficient_train_rows")
    return tuple(scores)


def _available_calibration(
    channel: str,
    matrix: NDArray[np.float32],
    *,
    center: float,
    scale: float,
    k: int,
) -> ChannelCalibration:
    scores = _loo_scores(matrix, k=k)
    threshold = float(
        np.quantile(
            np.asarray(scores, dtype=np.float32),
            FROZEN_CONFIG.train_quantile,
            method=FROZEN_CONFIG.quantile_method,
        )
    )
    if not np.isfinite(threshold):
        raise ValueError("nonfinite_train")
    return ChannelCalibration(
        channel=channel,
        available=True,
        threshold=threshold,
        loo_scores=scores,
        center=float(center),
        scale=float(scale),
        reference_count=matrix.shape[0],
        reference_digest=_canonical_matrix_digest(matrix),
    )


def _unavailable(channel: str, reason: str) -> ChannelCalibration:
    return ChannelCalibration.unavailable(channel, reason)


def _gradient_matrix(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        gradients = np.diff(matrix, axis=1, prepend=matrix[:, :1])
    gradients = np.asarray(gradients, dtype=np.float32)
    if gradients.ndim != 2 or not np.isfinite(gradients).all():
        raise ValueError("nonfinite_train")
    return np.ascontiguousarray(gradients, dtype=np.float32)


def _acf_failure_reason(row: NDArray[np.float32]) -> str:
    """Distinguish zero ACF energy from float32 overflow/non-finiteness."""

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.float32(np.mean(row, dtype=np.float32))
        centered = np.asarray(row - mean, dtype=np.float32)
        denominator = np.float32(np.sum(centered * centered, dtype=np.float32))
    if (
        not np.isfinite(mean)
        or not np.isfinite(centered).all()
        or not np.isfinite(denominator)
    ):
        return "nonfinite_train"
    return "zero_acf_energy"


def fit_calibrated_group(group: TrainGroup) -> CalibratedTrainGroup | UnavailableTrainGroup:
    """Fit all available TRAIN channels and deterministic LOO thresholds."""

    if isinstance(group, UnavailableTrainGroup):
        return group
    if not isinstance(group, PreparedTrainGroup):
        raise ValueError("group must be a PreparedTrainGroup or UnavailableTrainGroup")
    validated = _validate_group_matrix(group)
    if isinstance(validated, UnavailableTrainGroup):
        return validated
    matrix, rows, length = validated
    raw_center, raw_scale = _robust_center_scale(matrix)
    prototype_reason = _scale_reason(raw_scale)
    if prototype_reason is not None:
        return _unavailable_group(
            group.dataset,
            length,
            group.instance_indices,
            prototype_reason,
        )
    try:
        standardized = _standardize(matrix, raw_center, raw_scale)
        prototype = _available_calibration(
            "prototype",
            standardized,
            center=float(raw_center),
            scale=float(raw_scale),
            k=FROZEN_CONFIG.neighbor_count,
        )
    except ValueError:
        return _unavailable_group(
            group.dataset,
            length,
            group.instance_indices,
            "nonfinite_train",
        )

    gradients: NDArray[np.float32] | None
    gradient: ChannelCalibration
    try:
        gradient_matrix = _gradient_matrix(matrix)
    except ValueError:
        gradient_matrix = None
        gradient_center = np.float32(0.0)
        gradient_scale = np.float32(0.0)
        gradient_reason = "nonfinite_train"
    else:
        gradient_center, gradient_scale = _robust_center_scale(gradient_matrix)
        gradient_reason = _scale_reason(gradient_scale)
    if gradient_reason is not None or gradient_matrix is None:
        gradients = None
        gradient = _unavailable("gradient", gradient_reason or "nonfinite_train")
    else:
        try:
            gradients = _standardize(gradient_matrix, gradient_center, gradient_scale)
            gradient = _available_calibration(
                "gradient",
                gradients,
                center=float(gradient_center),
                scale=float(gradient_scale),
                k=FROZEN_CONFIG.neighbor_count,
            )
        except ValueError as exc:
            # A secondary channel is optional.  Numerical overflow or a
            # non-finite score therefore disables only this channel instead
            # of turning a usable prototype/ACF group into a hard failure.
            gradients = None
            reason = "nonfinite_train" if "nonfinite" in str(exc) else "zero_scale"
            gradient = _unavailable("gradient", reason)

    acf_vectors: NDArray[np.float32] | None
    acf: ChannelCalibration
    if length < FROZEN_CONFIG.min_acf_length:
        acf_vectors = None
        acf = _unavailable("acf", "short_series")
    else:
        vectors: list[NDArray[np.float32]] = []
        acf_failure_reason: str | None = None
        for row in matrix:
            vector = acf_vector(
                row,
                FROZEN_CONFIG.max_acf_lag,
                FROZEN_CONFIG.numeric_floor,
            )
            if vector is None:
                vectors = []
                acf_failure_reason = _acf_failure_reason(row)
                break
            vectors.append(vector)
        if len(vectors) != rows:
            acf_vectors = None
            acf = _unavailable("acf", acf_failure_reason or "zero_acf_energy")
        else:
            acf_vectors = np.asarray(vectors, dtype=np.float32)
            if not np.isfinite(acf_vectors).all():
                acf_vectors = None
                acf = _unavailable("acf", "nonfinite_train")
            else:
                acf_vectors = np.ascontiguousarray(acf_vectors, dtype=np.float32)
                acf = _available_calibration(
                    "acf",
                    acf_vectors,
                    center=0.0,
                    scale=1.0,
                    k=FROZEN_CONFIG.neighbor_count,
                )

    return CalibratedTrainGroup(
        dataset=group.dataset,
        length=length,
        instance_indices=group.instance_indices,
        values=standardized,
        gradients=gradients,
        acf_values=acf_vectors,
        prototype=prototype,
        gradient=gradient,
        acf=acf,
    )


__all__ = [
    "acf_vector",
    "fit_calibrated_group",
    "prepare_train_groups",
    "score_channel_cpu",
]
