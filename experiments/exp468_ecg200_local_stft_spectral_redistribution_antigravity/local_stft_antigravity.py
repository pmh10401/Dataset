"""Frozen Exp468 local-STFT spectral redistribution score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


FloatArrayAntigravity = NDArray[np.float64]
BoolArrayAntigravity = NDArray[np.bool_]


def _immutable_float64_antigravity(
    value: object,
    *,
    name: str,
    ndim: int,
) -> FloatArrayAntigravity:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have exact float64 dtype")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    owned = np.array(array, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _immutable_bool_antigravity(
    value: object,
    *,
    name: str,
) -> BoolArrayAntigravity:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.bool_) or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional bool array")
    owned = np.array(array, dtype=np.bool_, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class SpectralScoresAntigravity:
    fit_raw_js: FloatArrayAntigravity
    query_raw_js: FloatArrayAntigravity
    query_ranks: FloatArrayAntigravity
    confirmation_mask: BoolArrayAntigravity
    fit_normal_max_raw_js: float
    backend: str

    def __post_init__(self) -> None:
        fit_raw = _immutable_float64_antigravity(
            self.fit_raw_js, name="fit_raw_js", ndim=1,
        )
        query_raw = _immutable_float64_antigravity(
            self.query_raw_js, name="query_raw_js", ndim=1,
        )
        ranks = _immutable_float64_antigravity(
            self.query_ranks, name="query_ranks", ndim=1,
        )
        confirmation = _immutable_bool_antigravity(
            self.confirmation_mask, name="confirmation_mask",
        )
        if fit_raw.size < 2:
            raise ValueError("fit_raw_js must contain at least two rows")
        if query_raw.shape != ranks.shape or query_raw.shape != confirmation.shape:
            raise ValueError("query score arrays must have identical shape")
        if np.any((ranks <= 0.0) | (ranks > 1.0)):
            raise ValueError("query_ranks must be in (0, 1]")
        maximum = float(self.fit_normal_max_raw_js)
        if not np.isfinite(maximum) or maximum != float(np.max(fit_raw)):
            raise ValueError("fit_normal_max_raw_js must equal max(fit_raw_js)")
        if self.backend not in {"numpy", "numba", "mlx"}:
            raise ValueError("backend is not a supported Exp468 backend")
        object.__setattr__(self, "fit_raw_js", fit_raw)
        object.__setattr__(self, "query_raw_js", query_raw)
        object.__setattr__(self, "query_ranks", ranks)
        object.__setattr__(self, "confirmation_mask", confirmation)
        object.__setattr__(self, "fit_normal_max_raw_js", maximum)

    @property
    def ranks(self) -> FloatArrayAntigravity:
        """Compatibility name used by the approved implementation plan."""

        return self.query_ranks


def _validated_rows_antigravity(value: object, *, name: str) -> FloatArrayAntigravity:
    rows = np.asarray(value)
    if rows.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have exact float64 dtype")
    if rows.ndim != 2 or rows.shape[1:] != (FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,):
        raise ValueError(f"{name} must have shape (n, 140)")
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(rows)


def _power_distributions_numpy_antigravity(rows: FloatArrayAntigravity) -> FloatArrayAntigravity:
    config = FROZEN_CONFIG_ANTIGRAVITY
    suffix = rows[:, config.onset : config.interpolated_width]
    frames = np.stack(
        tuple(
            suffix[:, start : start + config.window_length]
            for start in config.window_starts
        ),
        axis=1,
    )
    centered = frames - frames.mean(axis=2, keepdims=True)
    weighted = centered * np.hanning(config.window_length)[None, None, :]
    spectrum = np.fft.rfft(weighted, n=config.rfft_length, axis=2)
    power = np.abs(spectrum[:, :, 1:]) ** 2
    power += config.power_epsilon
    power /= power.sum(axis=2, keepdims=True)
    return np.asarray(power, dtype=np.float64)


def window_power_distributions_antigravity(row: object) -> FloatArrayAntigravity:
    array = np.asarray(row)
    if array.dtype != np.dtype(np.float64):
        raise TypeError("row must have exact float64 dtype")
    if array.shape != (FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,):
        raise ValueError("row must have exact shape (140,)")
    if not np.isfinite(array).all():
        raise ValueError("row must contain only finite values")
    return _power_distributions_numpy_antigravity(array[None, :])[0]


def jensen_shannon_antigravity(p: object, q: object) -> float:
    left = np.asarray(p)
    right = np.asarray(q)
    if left.dtype != np.dtype(np.float64) or right.dtype != np.dtype(np.float64):
        raise TypeError("p and q must have exact float64 dtype")
    if left.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError("p and q must be nonempty one-dimensional arrays of equal shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("p and q must contain only finite values")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("p and q must be nonnegative")
    if not np.isclose(float(np.sum(left)), 1.0, rtol=0.0, atol=1e-15):
        raise ValueError("p must sum to one")
    if not np.isclose(float(np.sum(right)), 1.0, rtol=0.0, atol=1e-15):
        raise ValueError("q must sum to one")
    if np.array_equal(left, right):
        return 0.0
    midpoint = (left + right) / 2.0
    left_positive = left > 0.0
    right_positive = right > 0.0
    divergence = 0.5 * float(
        np.sum(left[left_positive] * np.log(left[left_positive] / midpoint[left_positive])),
    )
    divergence += 0.5 * float(
        np.sum(right[right_positive] * np.log(right[right_positive] / midpoint[right_positive])),
    )
    return divergence


def _reference_antigravity(distributions: FloatArrayAntigravity) -> FloatArrayAntigravity:
    reference = np.median(distributions, axis=0)
    reference /= reference.sum(axis=1, keepdims=True)
    return reference


def _raw_rows_antigravity(
    distributions: FloatArrayAntigravity,
    references: FloatArrayAntigravity,
) -> FloatArrayAntigravity:
    midpoint = (distributions + references) / 2.0
    divergence = 0.5 * np.sum(
        distributions * np.log(distributions / midpoint), axis=2,
    )
    divergence += 0.5 * np.sum(
        references * np.log(references / midpoint), axis=2,
    )
    raw = np.median(divergence, axis=1).astype(np.float64, copy=False)
    return raw


def strict_confirmation_antigravity(
    query_raw_js: object,
    fit_normal_max_raw_js: float,
) -> BoolArrayAntigravity:
    query = np.asarray(query_raw_js)
    if query.dtype != np.dtype(np.float64) or query.ndim != 1:
        raise TypeError("query_raw_js must be a one-dimensional float64 array")
    maximum = float(fit_normal_max_raw_js)
    if not np.isfinite(query).all() or not np.isfinite(maximum):
        raise ValueError("confirmation inputs must be finite")
    return np.asarray((query > 0.0) & (query > maximum), dtype=np.bool_)


def _assemble_scores_antigravity(
    fit_raw: FloatArrayAntigravity,
    query_raw: FloatArrayAntigravity,
    *,
    backend: str,
) -> SpectralScoresAntigravity:
    maximum = float(np.max(fit_raw))
    ranks = (
        1.0 + np.count_nonzero(fit_raw[:, None] <= query_raw[None, :], axis=0)
    ) / float(fit_raw.size + 1)
    confirmation = strict_confirmation_antigravity(query_raw, maximum)
    return SpectralScoresAntigravity(
        fit_raw_js=fit_raw,
        query_raw_js=query_raw,
        query_ranks=np.asarray(ranks, dtype=np.float64),
        confirmation_mask=confirmation,
        fit_normal_max_raw_js=maximum,
        backend=backend,
    )


def _score_local_stft_numpy_antigravity(
    fit_rows: FloatArrayAntigravity,
    query_rows: FloatArrayAntigravity,
) -> SpectralScoresAntigravity:
    fit_power = _power_distributions_numpy_antigravity(fit_rows)
    query_power = _power_distributions_numpy_antigravity(query_rows)
    return _score_from_power_numpy_antigravity(
        fit_power, query_power, backend="numpy",
    )


def _score_from_power_numpy_antigravity(
    fit_power: FloatArrayAntigravity,
    query_power: FloatArrayAntigravity,
    *,
    backend: str,
) -> SpectralScoresAntigravity:
    """Apply the frozen reference/JS/rank steps to precomputed distributions."""

    fit_references = np.empty_like(fit_power)
    for index in range(fit_power.shape[0]):
        fit_references[index] = _reference_antigravity(
            np.concatenate((fit_power[:index], fit_power[index + 1 :]), axis=0),
        )
    fit_raw = _raw_rows_antigravity(fit_power, fit_references)
    query_reference = _reference_antigravity(fit_power)
    query_references = np.broadcast_to(query_reference, query_power.shape)
    query_raw = _raw_rows_antigravity(query_power, query_references)
    return _assemble_scores_antigravity(fit_raw, query_raw, backend=backend)


def score_local_stft_antigravity(
    fit_rows: object,
    query_rows: object,
    *,
    backend: str = "numpy",
) -> SpectralScoresAntigravity:
    fit = _validated_rows_antigravity(fit_rows, name="fit_rows")
    query = _validated_rows_antigravity(query_rows, name="query_rows")
    if fit.shape[0] < 2:
        raise ValueError("fit_rows must contain at least two rows for LOO scoring")
    if query.shape[0] < 1:
        raise ValueError("query_rows must contain at least one row")
    if backend == "numpy":
        return _score_local_stft_numpy_antigravity(fit, query)
    if backend == "numba":
        from .acceleration_antigravity import _score_local_stft_numba_antigravity

        return _score_local_stft_numba_antigravity(fit, query)
    if backend == "mlx":
        from .acceleration_antigravity import _score_local_stft_mlx_antigravity

        return _score_local_stft_mlx_antigravity(fit, query)
    raise ValueError("backend must be one of: numpy, numba, mlx")


__all__ = (
    "SpectralScoresAntigravity",
    "jensen_shannon_antigravity",
    "score_local_stft_antigravity",
    "strict_confirmation_antigravity",
    "window_power_distributions_antigravity",
)
