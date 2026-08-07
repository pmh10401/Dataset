"""Immutable TRAIN-OOF calibration cache with frozen Exp417 score semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import HorizonScore
from experiments.exp417_causal_multihorizon_shadow.scoring import (
    _aligned_residual_series,
    _residual_vector,
    _series_score,
    empirical_p_value,
)


_FORMULA_VERSION = "exp417_score_horizon_v1"
_MAD_TO_NORMAL_SCALE = 1.4826
_MIN_ROBUST_SCALE = 1e-12


def _positive_horizon(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("horizon must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    """Return an owning immutable copy with a platform-independent float dtype."""
    copied = np.ascontiguousarray(values, dtype="<f8")
    return np.frombuffer(copied.tobytes(), dtype="<f8").reshape(copied.shape)


def _aligned_oof(values: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    oof = _aligned_residual_series(values, "oof_residuals")
    length = oof[0].size
    if any(item.size != length for item in oof):
        raise ValueError("residuals must preserve the exact aligned length")
    return oof


def _cache_arrays(values: object) -> tuple[np.ndarray, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("normalized_oof_residuals must be a non-empty sequence")
    arrays = tuple(values)
    if not arrays:
        raise ValueError("normalized_oof_residuals must be a non-empty sequence")
    length: int | None = None
    frozen: list[np.ndarray] = []
    for item in arrays:
        if (
            not isinstance(item, np.ndarray)
            or item.dtype != np.dtype("float64")
            or item.ndim != 1
            or item.size == 0
        ):
            raise ValueError("normalized_oof_residuals must contain one-dimensional float64 arrays")
        if item.flags.writeable:
            raise ValueError("normalized_oof_residuals must be read-only")
        if not np.isfinite(item).all():
            raise ValueError("normalized_oof_residuals must be finite")
        if length is None:
            length = int(item.size)
        elif item.size != length:
            raise ValueError("residuals must preserve the exact aligned length")
        frozen.append(_readonly_float64(item))
    return tuple(frozen)


def _cache_scores(values: object, expected_count: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("oof_scores must be a sequence")
    scores = tuple(_finite_number(item, "oof_scores") for item in values)
    if len(scores) != expected_count:
        raise ValueError("oof_scores must match normalized OOF residuals")
    return scores


def _validated_cache_evidence(
    *,
    horizon: object,
    center: object,
    scale: object,
    exceedance_threshold: object,
    normalized_oof_residuals: object,
    oof_scores: object,
    formula_version: object,
) -> tuple[int, float, float, float, tuple[np.ndarray, ...], tuple[float, ...]]:
    frozen_horizon = _positive_horizon(horizon)
    frozen_center = _finite_number(center, "center")
    frozen_scale = _finite_number(scale, "scale")
    if frozen_scale <= _MIN_ROBUST_SCALE:
        raise ValueError("OOF residual robust scale must be finite and greater than 1e-12")
    threshold = _finite_number(exceedance_threshold, "exceedance_threshold")
    normalized = _cache_arrays(normalized_oof_residuals)
    scores = _cache_scores(oof_scores, len(normalized))
    if formula_version != _FORMULA_VERSION:
        raise ValueError("formula_version must match frozen Exp417 scoring")
    expected_threshold = float(
        np.percentile(np.concatenate(normalized), 99, method="linear")
    )
    if threshold != expected_threshold:
        raise ValueError("exceedance_threshold must match normalized OOF residuals")
    expected_scores = tuple(
        _series_score(values, threshold)[0] for values in normalized
    )
    if scores != expected_scores:
        raise ValueError("oof_scores must match frozen series scoring")
    return frozen_horizon, frozen_center, frozen_scale, threshold, normalized, scores


def _cache_evidence_digest(
    *,
    horizon: int,
    center: float,
    scale: float,
    exceedance_threshold: float,
    normalized_oof_residuals: tuple[np.ndarray, ...],
    oof_scores: tuple[float, ...],
    formula_version: str,
) -> str:
    """Return the canonical digest for exactly the immutable cache evidence."""
    digest = sha256()
    digest.update(formula_version.encode("ascii"))
    digest.update(b"\x00")
    digest.update(horizon.to_bytes(8, byteorder="little", signed=False))
    for value in (center, scale, exceedance_threshold):
        digest.update(np.asarray([value], dtype="<f8").tobytes())
    digest.update(len(normalized_oof_residuals).to_bytes(8, byteorder="little", signed=False))
    for values in normalized_oof_residuals:
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(canonical.ndim.to_bytes(8, byteorder="little", signed=False))
        for dimension in canonical.shape:
            digest.update(dimension.to_bytes(8, byteorder="little", signed=False))
        digest.update(canonical.nbytes.to_bytes(8, byteorder="little", signed=False))
        digest.update(canonical.tobytes())
    digest.update(len(oof_scores).to_bytes(8, byteorder="little", signed=False))
    digest.update(np.asarray(oof_scores, dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationCache:
    """Precomputed, immutable normal-TRAIN evidence for one exact horizon."""

    horizon: int
    center: float
    scale: float
    exceedance_threshold: float
    normalized_oof_residuals: tuple[np.ndarray, ...]
    oof_scores: tuple[float, ...]
    digest: str
    formula_version: str = _FORMULA_VERSION

    def __post_init__(self) -> None:
        horizon, center, scale, threshold, normalized, scores = _validated_cache_evidence(
            horizon=self.horizon,
            center=self.center,
            scale=self.scale,
            exceedance_threshold=self.exceedance_threshold,
            normalized_oof_residuals=self.normalized_oof_residuals,
            oof_scores=self.oof_scores,
            formula_version=self.formula_version,
        )
        if not (
            isinstance(self.digest, str)
            and len(self.digest) == 64
            and all(character in "0123456789abcdef" for character in self.digest)
        ):
            raise ValueError("digest must be a lowercase SHA-256 digest")
        expected_digest = _cache_evidence_digest(
            horizon=horizon,
            center=center,
            scale=scale,
            exceedance_threshold=threshold,
            normalized_oof_residuals=normalized,
            oof_scores=scores,
            formula_version=self.formula_version,
        )
        if self.digest != expected_digest:
            raise ValueError("digest must bind the complete calibration cache evidence")

        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "exceedance_threshold", threshold)
        object.__setattr__(self, "normalized_oof_residuals", normalized)
        object.__setattr__(self, "oof_scores", scores)


def calibration_digest(
    *,
    horizon: int,
    center: float,
    scale: float,
    exceedance_threshold: float,
    normalized_oof_residuals: Sequence[np.ndarray],
    oof_scores: Sequence[float],
    formula_version: str = _FORMULA_VERSION,
) -> str:
    """Hash complete validated cache evidence in canonical endian-stable form."""
    evidence = _validated_cache_evidence(
        horizon=horizon,
        center=center,
        scale=scale,
        exceedance_threshold=exceedance_threshold,
        normalized_oof_residuals=normalized_oof_residuals,
        oof_scores=oof_scores,
        formula_version=formula_version,
    )
    return _cache_evidence_digest(
        horizon=evidence[0],
        center=evidence[1],
        scale=evidence[2],
        exceedance_threshold=evidence[3],
        normalized_oof_residuals=evidence[4],
        oof_scores=evidence[5],
        formula_version=formula_version,
    )


def build_calibration_cache(
    oof_residuals: Sequence[np.ndarray], *, horizon: int
) -> CalibrationCache:
    """Precompute exact Exp417 scoring evidence from normal TRAIN OOF residuals."""
    _positive_horizon(horizon)
    # Copy first so later source mutation cannot alter the evidence or its digest.
    source = tuple(_readonly_float64(values) for values in _aligned_oof(oof_residuals))
    pooled = np.concatenate(source)
    center = float(np.median(pooled))
    scale = float(_MAD_TO_NORMAL_SCALE * np.median(np.abs(pooled - center)))
    if not isfinite(center) or not isfinite(scale) or scale <= _MIN_ROBUST_SCALE:
        raise ValueError("OOF residual robust scale must be finite and greater than 1e-12")
    normalized = tuple(_readonly_float64((values - center) / scale) for values in source)
    if not all(np.isfinite(values).all() for values in normalized):
        raise ValueError("standardized residuals must be finite")
    threshold = float(np.percentile(np.concatenate(normalized), 99, method="linear"))
    scores = tuple(_series_score(values, threshold)[0] for values in normalized)
    return CalibrationCache(
        horizon=horizon,
        center=center,
        scale=scale,
        exceedance_threshold=threshold,
        normalized_oof_residuals=normalized,
        oof_scores=scores,
        digest=calibration_digest(
            horizon=horizon,
            center=center,
            scale=scale,
            exceedance_threshold=threshold,
            normalized_oof_residuals=normalized,
            oof_scores=scores,
        ),
    )


def score_query_with_cache(cache: CalibrationCache, query_residual: np.ndarray) -> HorizonScore:
    """Score one aligned residual series using only immutable TRAIN OOF evidence."""
    if not isinstance(cache, CalibrationCache):
        raise ValueError("cache must be a CalibrationCache")
    query = _residual_vector(query_residual, "query_residual")
    length = cache.normalized_oof_residuals[0].size
    if query.size != length:
        raise ValueError("residuals must preserve the exact aligned length")
    normalized_query = (query - cache.center) / cache.scale
    if not np.isfinite(normalized_query).all():
        raise ValueError("standardized residuals must be finite")
    score, persistence = _series_score(normalized_query, cache.exceedance_threshold)
    return HorizonScore(
        horizon=cache.horizon,
        score=score,
        p_value=empirical_p_value(cache.oof_scores, score),
        persistence=persistence,
    )


__all__ = [
    "CalibrationCache",
    "build_calibration_cache",
    "calibration_digest",
    "score_query_with_cache",
]
