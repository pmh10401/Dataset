"""Standalone Exp249 official MiniROCKET evidence closure.

Copied scientific closure: ``score_configuration`` (Exp249 scoring), aeon's
``MiniRocket`` configuration, local-gap scoring, deterministic crossfit,
reference sampling, and chunked query scoring.  This module intentionally has
no runtime dependency on ``experiments.exp*`` or prediction artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from ..calibration_paper import upper_tail_pvalues
from ..contracts_paper import RawDataset
from .preprocessing_paper import DependencyUnavailablePaper, align_series_lengths_paper, z_normalize_paper


SEEDS_PAPER = (20260721, 20261721)
CROSSFIT_FOLDS_PAPER = 5
KNN_NEIGHBORS_PAPER = 3
MAX_TRAIN_REFERENCE_PAPER = 1000
QUERY_CHUNK_SIZE_PAPER = 128
MINIMUM_TRAIN_PAPER = 5
PROVENANCE_PAPER = {
    "experiments/exp249_250_official_mini_multirocket/scoring.py": ("ConfigurationScores", "_reference_indices", "_score_query_chunks", "score_configuration"),
    "experiments/exp249_250_official_mini_multirocket/transforms.py": ("actual_kernel_count", "expected_feature_dimension", "as_aeon_collection", "make_transformer"),
    "experiments/exp249_250_official_mini_multirocket/local_gap.py": ("LocalGapModel",),
    "experiments/exp239_248_diverse_score_sources/foundation.py": ("deterministic_folds", "crossfit_plan"),
}


@dataclass(frozen=True, slots=True)
class MiniRocketConfig:
    transformer: str = "minirocket"
    max_dilations_per_kernel: int = 32
    n_features_per_kernel: int = 1
    normalise: bool = False

    def __post_init__(self) -> None:
        if self.transformer != "minirocket" or int(self.max_dilations_per_kernel) != 32 or int(self.n_features_per_kernel) != 1 or self.normalise is not False:
            raise ValueError("MiniROCKET configuration differs from the frozen origin")


@dataclass(frozen=True, slots=True)
class ConfigurationScoresPaper:
    train_oof: np.ndarray
    test_score: np.ndarray
    feature_dimension: int

    def __post_init__(self) -> None:
        train, test = np.asarray(self.train_oof), np.asarray(self.test_score)
        if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test) or train.dtype.kind not in "iuf" or test.dtype.kind not in "iuf":
            raise ValueError("MiniROCKET scores must be nonempty numeric vectors")
        train_copy, test_copy = np.array(train, dtype=np.float64, copy=True), np.array(test, dtype=np.float64, copy=True)
        if not np.isfinite(train_copy).all() or not np.isfinite(test_copy).all() or int(self.feature_dimension) < 1:
            raise ValueError("MiniROCKET scores and dimensions must be finite")
        train_copy.setflags(write=False)
        test_copy.setflags(write=False)
        object.__setattr__(self, "train_oof", train_copy)
        object.__setattr__(self, "test_score", test_copy)
        object.__setattr__(self, "feature_dimension", int(self.feature_dimension))


@dataclass(frozen=True, slots=True)
class MiniRocketKernelResult:
    requested_kernels: int
    expected_dimension: int
    by_seed: Mapping[int, ConfigurationScoresPaper]
    candidate: frozenset[int]

    def __post_init__(self) -> None:
        requested, expected = int(self.requested_kernels), int(self.expected_dimension)
        if (requested, expected) not in ((1024, 1008), (4096, 4032)):
            raise ValueError("MiniROCKET kernel configuration differs from the frozen origin")
        scores = dict(self.by_seed)
        if scores:
            if set(scores) != set(SEEDS_PAPER) or any(not isinstance(value, ConfigurationScoresPaper) for value in scores.values()):
                raise ValueError("MiniROCKET seed coverage differs from the frozen origin")
            if any(value.feature_dimension != expected for value in scores.values()):
                raise ValueError("MiniROCKET feature dimensions differ from the frozen origin")
            n_test = next(iter(scores.values())).test_score.size
            if any(value.train_oof.shape != next(iter(scores.values())).train_oof.shape or value.test_score.size != n_test for value in scores.values()):
                raise ValueError("MiniROCKET seed score shapes differ")
            expected_candidate = frozenset.intersection(*(frozenset(np.flatnonzero(upper_tail_pvalues(value.train_oof, value.test_score) <= .005).astype(int).tolist()) for value in scores.values()))
            if frozenset(self.candidate) != expected_candidate:
                raise ValueError("MiniROCKET candidate differs from frozen p-value intersection")
            if any(index >= n_test for index in self.candidate):
                raise ValueError("MiniROCKET candidate index exceeds TEST length")
        elif self.candidate:
            raise ValueError("empty MiniROCKET seed coverage cannot have candidates")
        if any(type(index) is not int or index < 0 for index in self.candidate):
            raise ValueError("MiniROCKET candidate indices are invalid")
        object.__setattr__(self, "requested_kernels", requested)
        object.__setattr__(self, "expected_dimension", expected)
        object.__setattr__(self, "by_seed", MappingProxyType(scores))
        object.__setattr__(self, "candidate", frozenset(self.candidate))


@dataclass(frozen=True, slots=True)
class MiniRocketResult:
    k1024: frozenset[int]
    k4096: frozenset[int]
    by_kernel: Mapping[int, MiniRocketKernelResult]

    def __post_init__(self) -> None:
        kernels = dict(self.by_kernel)
        if kernels:
            if set(kernels) != {1024, 4096} or any(not isinstance(value, MiniRocketKernelResult) for value in kernels.values()):
                raise ValueError("MiniROCKET kernel coverage differs from the frozen origin")
            if kernels[1024].candidate != frozenset(self.k1024) or kernels[4096].candidate != frozenset(self.k4096):
                raise ValueError("MiniROCKET result candidates differ from kernel evidence")
        elif self.k1024 or self.k4096:
            raise ValueError("empty MiniROCKET result cannot have candidates")
        for candidate in (self.k1024, self.k4096):
            if not isinstance(candidate, frozenset) or any(type(index) is not int or index < 0 for index in candidate):
                raise ValueError("MiniROCKET result indices are invalid")
        object.__setattr__(self, "by_kernel", MappingProxyType(kernels))


class LocalGapModelPaper:
    """Exact sklearn local-gap closure used by Exp249."""

    def __init__(self, reference: np.ndarray, *, neighbors: int = KNN_NEIGHBORS_PAPER) -> None:
        values = np.asarray(reference, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("local-gap requires a two-dimensional reference")
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(values)
        scaled = self.scaler.transform(values)
        self.neighbors = max(1, min(int(neighbors), len(scaled)))
        self.reference_model = NearestNeighbors(n_neighbors=max(1, min(self.neighbors + 1, len(scaled))), metric="euclidean").fit(scaled)
        distances, indices = self.reference_model.kneighbors(scaled)
        self.local_radius = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        self.query_model = NearestNeighbors(n_neighbors=self.neighbors, metric="euclidean").fit(scaled)
        self.scale = float(np.median(self.local_radius) + 1e-9)

    def score(self, query: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(query, dtype=float))
        distances, indices = self.query_model.kneighbors(scaled)
        return np.asarray((distances.mean(axis=1) - (self.local_radius[indices].mean(axis=1) + 1e-9)) / self.scale, dtype=float)


def actual_kernel_count_paper(requested_kernels: int) -> int:
    requested = int(requested_kernels)
    if requested < 84:
        raise ValueError("official ROCKET transforms require at least 84 kernels")
    return requested - requested % 84


def expected_feature_dimension_paper(config: MiniRocketConfig, requested_kernels: int) -> int:
    kernels = actual_kernel_count_paper(requested_kernels)
    if config.transformer != "minirocket":
        raise ValueError(f"unknown transformer: {config.transformer}")
    return kernels


def as_aeon_collection_paper(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("aligned series must be a two-dimensional matrix")
    if values.shape[1] < 9:
        values = np.pad(values, ((0, 0), (0, 9 - values.shape[1])))
    return values[:, np.newaxis, :]


def make_transformer_paper(config: MiniRocketConfig, *, requested_kernels: int, seed: int):
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
    except ImportError as error:
        raise DependencyUnavailablePaper("dependency_unavailable:aeon") from error
    return MiniRocket(n_kernels=int(requested_kernels), max_dilations_per_kernel=int(config.max_dilations_per_kernel), n_jobs=1, random_state=int(seed))


def deterministic_folds_paper(n_train: int, *, seed: int, folds: int = CROSSFIT_FOLDS_PAPER) -> tuple[np.ndarray, ...]:
    if n_train < 2:
        raise ValueError("at least two TRAIN rows are required")
    shuffled = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    return tuple(np.sort(part) for part in np.array_split(shuffled, min(int(folds), int(n_train))) if len(part))


def crossfit_plan_paper(n_train: int, *, seed: int, folds: int = CROSSFIT_FOLDS_PAPER) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    queries = deterministic_folds_paper(n_train, seed=seed, folds=folds)
    return tuple((np.sort(np.concatenate([part for other, part in enumerate(queries) if other != index])), query) for index, query in enumerate(queries))


def _reference_indices_paper(indices: np.ndarray, *, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(maximum):
        return values
    return np.sort(np.random.default_rng(int(seed)).choice(values, size=int(maximum), replace=False))


def _score_query_chunks_paper(transformer, model: LocalGapModelPaper, rows: np.ndarray, *, chunk_size: int) -> np.ndarray:
    parts = [model.score(np.asarray(transformer.transform(as_aeon_collection_paper(rows[start:start + int(chunk_size)])), dtype=float)) for start in range(0, len(rows), int(chunk_size))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def score_configuration_paper(train_rows: np.ndarray, test_rows: np.ndarray, *, config: MiniRocketConfig, requested_kernels: int, seed: int, expected_dimension: int | None, max_train_reference: int = MAX_TRAIN_REFERENCE_PAPER, query_chunk_size: int = QUERY_CHUNK_SIZE_PAPER, transformer_factory=None) -> ConfigurationScoresPaper:
    train, test = np.asarray(train_rows, dtype=np.float32), np.asarray(test_rows, dtype=np.float32)
    train_oof, dimensions = np.empty(len(train), dtype=float), set()
    factory = transformer_factory or (lambda **kwargs: make_transformer_paper(**kwargs))
    for fold_index, (fit_indices, query_indices) in enumerate(crossfit_plan_paper(len(train), seed=int(seed))):
        reference = _reference_indices_paper(fit_indices, maximum=max_train_reference, seed=int(seed) + 1009 * (fold_index + 1) + int(requested_kernels))
        transformer = factory(config=config, requested_kernels=requested_kernels, seed=seed)
        fit = np.asarray(transformer.fit_transform(as_aeon_collection_paper(train[reference])), dtype=float)
        dimensions.add(int(fit.shape[1]))
        train_oof[query_indices] = _score_query_chunks_paper(transformer, LocalGapModelPaper(fit), train[query_indices], chunk_size=query_chunk_size)
    reference = _reference_indices_paper(np.arange(len(train), dtype=int), maximum=max_train_reference, seed=int(seed) + 7919 + int(requested_kernels))
    transformer = factory(config=config, requested_kernels=requested_kernels, seed=seed)
    fit = np.asarray(transformer.fit_transform(as_aeon_collection_paper(train[reference])), dtype=float)
    dimensions.add(int(fit.shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    feature_dimension = dimensions.pop()
    if expected_dimension is not None and feature_dimension != int(expected_dimension):
        raise ValueError(f"feature dimension mismatch: expected={expected_dimension} actual={feature_dimension}")
    test_score = _score_query_chunks_paper(transformer, LocalGapModelPaper(fit), test, chunk_size=query_chunk_size)
    if not np.isfinite(train_oof).all() or not np.isfinite(test_score).all():
        raise ValueError("official ROCKET produced non-finite local-gap scores")
    return ConfigurationScoresPaper(train_oof, test_score, feature_dimension)


def score_minirocket_confirmations(values: RawDataset) -> MiniRocketResult:
    if len(values.train_rows) < MINIMUM_TRAIN_PAPER:
        empty = {requested: MiniRocketKernelResult(requested, dimension, {}, frozenset()) for requested, dimension in ((1024, 1008), (4096, 4032))}
        return MiniRocketResult(frozenset(), frozenset(), empty)
    target = int(np.clip(np.median([len(row) for row in values.train_rows]), 8, 2048))
    train = z_normalize_paper(align_series_lengths_paper(values.train_rows, target))
    test = z_normalize_paper(align_series_lengths_paper(values.test_rows, target))
    by_kernel: dict[int, MiniRocketKernelResult] = {}
    for requested, expected in ((1024, 1008), (4096, 4032)):
        scores = {seed: score_configuration_paper(train, test, config=MiniRocketConfig(), requested_kernels=requested, seed=seed, expected_dimension=expected) for seed in SEEDS_PAPER}
        candidates = [frozenset(np.flatnonzero(upper_tail_pvalues(score.train_oof, score.test_score) <= .005).astype(int).tolist()) for score in scores.values()]
        by_kernel[requested] = MiniRocketKernelResult(requested, expected, scores, frozenset.intersection(*candidates))
    return MiniRocketResult(by_kernel[1024].candidate, by_kernel[4096].candidate, by_kernel)


__all__ = ["ConfigurationScoresPaper", "DependencyUnavailablePaper", "MiniRocketConfig", "MiniRocketKernelResult", "MiniRocketResult", "actual_kernel_count_paper", "as_aeon_collection_paper", "expected_feature_dimension_paper", "score_configuration_paper", "score_minirocket_confirmations"]
