"""Immutable TRAIN-only cache for the frozen Exp417 perturbations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.qualification import (
    build_fixed_perturbations,
)
from .config import FROZEN_CONFIG


_PRIMARY_PERTURBATIONS = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_PERTURBATIONS = _PRIMARY_PERTURBATIONS + ("secondary_spike",)
_FORBIDDEN_IDENTIFIER_TOKENS = ("test", "label", "location", "metric")
_CACHE: dict[tuple[str, int, int, int], CachedPerturbations] = {}
_PRIMARY_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, ...]] = {}


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    copied = np.ascontiguousarray(values, dtype="<f8")
    return np.frombuffer(copied.tobytes(), dtype="<f8").reshape(copied.shape)


def _dataset_identifier(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("dataset must be a non-empty TRAIN identifier")
    lowered = value.lower()
    if not lowered.startswith("train") or any(token in lowered for token in _FORBIDDEN_IDENTIFIER_TOKENS):
        raise ValueError("dataset must be a TRAIN identifier without evaluation fields")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _series(values: object) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size < 2:
        raise ValueError("series must be a one-dimensional array with at least two points")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("series must be finite")
    return values


def _scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0.0:
        raise ValueError("robust_scale must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class PerturbationCacheKey:
    """Exact TRAIN identity of a single frozen perturbation array."""

    dataset: str
    instance_index: int
    length: int
    perturbation: str
    seed: int

    def __post_init__(self) -> None:
        _dataset_identifier(self.dataset)
        _positive_integer(self.instance_index, "instance_index")
        if type(self.length) is not int or self.length < 1:
            raise ValueError("length must be a positive integer")
        if self.perturbation not in _PERTURBATIONS:
            raise ValueError("perturbation must be one of the frozen perturbations")
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG.frozen_perturbation_seeds:
            raise ValueError("seed must be one of the exact frozen perturbation seeds")


@dataclass(frozen=True)
class CachedPerturbations(Mapping[str, np.ndarray]):
    """Read-only frozen perturbation arrays and their canonical provenance."""

    arrays: tuple[np.ndarray, ...]
    cache_keys: tuple[PerturbationCacheKey, ...]
    digest: str

    def __post_init__(self) -> None:
        if len(self.arrays) != len(_PERTURBATIONS) or len(self.cache_keys) != len(_PERTURBATIONS):
            raise ValueError("cached perturbations must contain every frozen perturbation")
        if tuple(key.perturbation for key in self.cache_keys) != _PERTURBATIONS:
            raise ValueError("cache keys must use the frozen perturbation order")
        if any(array.dtype != np.dtype("float64") or array.ndim != 1 or array.flags.writeable for array in self.arrays):
            raise ValueError("cached perturbation arrays must be read-only one-dimensional float64")
        if any(key.length != array.size for key, array in zip(self.cache_keys, self.arrays, strict=True)):
            raise ValueError("cache key length must match its immutable array")
        if not isinstance(self.digest, str) or len(self.digest) != 64 or any(character not in "0123456789abcdef" for character in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 digest")
        if self.digest != _digest(self.cache_keys, self.arrays):
            raise ValueError("digest must bind cache keys and immutable arrays")

    def __getitem__(self, perturbation: str) -> np.ndarray:
        try:
            return self.arrays[_PERTURBATIONS.index(perturbation)]
        except ValueError as exc:
            raise KeyError(perturbation) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(_PERTURBATIONS)

    def __len__(self) -> int:
        return len(_PERTURBATIONS)


def _digest(keys: tuple[PerturbationCacheKey, ...], arrays: tuple[np.ndarray, ...]) -> str:
    hashed = sha256()
    for key, array in zip(keys, arrays, strict=True):
        hashed.update(key.dataset.encode("utf-8"))
        hashed.update(b"\x00")
        hashed.update(key.instance_index.to_bytes(8, "little", signed=False))
        hashed.update(key.length.to_bytes(8, "little", signed=False))
        hashed.update(key.perturbation.encode("ascii"))
        hashed.update(b"\x00")
        hashed.update(key.seed.to_bytes(8, "little", signed=False))
        hashed.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return hashed.hexdigest()


def clear_perturbation_cache() -> None:
    """Clear in-memory TRAIN cache state for isolated process-local use."""
    _CACHE.clear()
    _PRIMARY_CACHE.clear()


def build_cached_perturbations(
    dataset: str,
    instance_index: int,
    series: np.ndarray,
    robust_scale: float,
    seed: int,
) -> CachedPerturbations:
    """Return immutable frozen perturbations for one exact TRAIN instance and seed."""
    frozen_dataset = _dataset_identifier(dataset)
    frozen_index = _positive_integer(instance_index, "instance_index")
    values = _series(series)
    scale = _scale(robust_scale)
    if type(seed) is not int or seed not in FROZEN_CONFIG.frozen_perturbation_seeds:
        raise ValueError("seed must be one of the exact frozen perturbation seeds")
    identity = (frozen_dataset, frozen_index, int(values.size), seed)
    cached = _CACHE.get(identity)
    if cached is not None:
        return cached

    built = build_fixed_perturbations(values, scale, seed=seed)
    primary_identity = identity[:3]
    primary = _PRIMARY_CACHE.get(primary_identity)
    if primary is None:
        primary = tuple(_readonly_float64(built[name]) for name in _PRIMARY_PERTURBATIONS)
        _PRIMARY_CACHE[primary_identity] = primary
    arrays = primary + (_readonly_float64(built["secondary_spike"]),)
    keys = tuple(
        PerturbationCacheKey(frozen_dataset, frozen_index, int(values.size), name, seed)
        for name in _PERTURBATIONS
    )
    result = CachedPerturbations(arrays=arrays, cache_keys=keys, digest=_digest(keys, arrays))
    _CACHE[identity] = result
    return result


__all__ = [
    "CachedPerturbations",
    "PerturbationCacheKey",
    "build_cached_perturbations",
    "clear_perturbation_cache",
]
