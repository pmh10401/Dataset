"""Exact immutable records shared by the later Exp454 phases."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np


def _exact_str_antigravity(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be an exact non-empty string")


def _exact_int_antigravity(value: object, field: str, *, nonnegative: bool = False) -> None:
    if type(value) is not int:
        raise ValueError(f"{field} must be an exact integer")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be non-negative")


def _exact_float_antigravity(value: object, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be an exact finite float")


def _exact_tuple_antigravity(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{field} must be an exact tuple")
    return value


@dataclass(frozen=True, slots=True)
class MethodCandidateAntigravity:
    name: str
    representation: str
    seeds: tuple[int, ...]
    alphas: tuple[float, ...]

    def __post_init__(self) -> None:
        _exact_str_antigravity(self.name, "name")
        _exact_str_antigravity(self.representation, "representation")
        seeds = _exact_tuple_antigravity(self.seeds, "seeds")
        alphas = _exact_tuple_antigravity(self.alphas, "alphas")
        if not seeds or not alphas:
            raise ValueError("seeds and alphas must be non-empty")
        for seed in seeds:
            _exact_int_antigravity(seed, "seed", nonnegative=True)
        for alpha in alphas:
            _exact_float_antigravity(alpha, "alpha")
            if not 0.0 < alpha <= 1.0:
                raise ValueError("alpha must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class MethodAuditAntigravity:
    name: str
    status: Literal["score_compatible", "classifier_only", "unavailable"]
    module: str
    symbol: str
    package_version: str | None
    source_sha256: str | None
    methods: tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_str_antigravity(self.name, "name")
        if type(self.status) is not str or self.status not in ("score_compatible", "classifier_only", "unavailable"):
            raise ValueError("status is invalid")
        _exact_str_antigravity(self.module, "module")
        _exact_str_antigravity(self.symbol, "symbol")
        if self.package_version is not None:
            _exact_str_antigravity(self.package_version, "package_version")
        if self.source_sha256 is not None and (type(self.source_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)):
            raise ValueError("source_sha256 must be a lowercase SHA256")
        methods = _exact_tuple_antigravity(self.methods, "methods")
        for method in methods:
            _exact_str_antigravity(method, "method")
        if self.status == "unavailable":
            if self.source_sha256 is not None or methods:
                raise ValueError("unavailable method audit must not bind source or methods")
        elif self.source_sha256 is None or not methods:
            raise ValueError("available method audit requires source and methods")


@dataclass(frozen=True, slots=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for field in ("tp", "fp", "fn", "tn"):
            _exact_int_antigravity(getattr(self, field), field, nonnegative=True)


@dataclass(frozen=True, slots=True)
class DatasetBundleAntigravity:
    dataset: str
    train_normal: np.ndarray
    test_values: np.ndarray
    target_length: int

    def __post_init__(self) -> None:
        _exact_str_antigravity(self.dataset, "dataset")
        _exact_int_antigravity(self.target_length, "target_length", nonnegative=True)
        for field in ("train_normal", "test_values"):
            value = getattr(self, field)
            if type(value) is not np.ndarray or value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"{field} must be an exact two-dimensional numeric ndarray")
            if not np.isfinite(value).all():
                raise ValueError(f"{field} must be finite")
            # A copied ndarray can later re-enable WRITEABLE.  A bytes-backed
            # array cannot: bytes is immutable and does not alias caller data.
            protected = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, field, protected)


@dataclass(frozen=True, slots=True)
class DatasetBaselineAntigravity:
    dataset_order: int
    family: str
    stable_candidates: frozenset[int]
    confusion: ConfusionAntigravity

    def __post_init__(self) -> None:
        _exact_int_antigravity(self.dataset_order, "dataset_order", nonnegative=True)
        _exact_str_antigravity(self.family, "family")
        if type(self.stable_candidates) is not frozenset:
            raise ValueError("stable_candidates must be an exact frozenset")
        for candidate in self.stable_candidates:
            _exact_int_antigravity(candidate, "stable candidate", nonnegative=True)
        if type(self.confusion) is not ConfusionAntigravity:
            raise ValueError("confusion must be an exact ConfusionAntigravity")


@dataclass(frozen=True, slots=True)
class CandidateMetricAntigravity:
    family: str
    method: str
    target_confusion: ConfusionAntigravity
    global_confusion: ConfusionAntigravity
    target_fn_delta: int
    global_f1: float
    qualified: bool

    def __post_init__(self) -> None:
        _exact_str_antigravity(self.family, "family")
        _exact_str_antigravity(self.method, "method")
        if type(self.target_confusion) is not ConfusionAntigravity or type(self.global_confusion) is not ConfusionAntigravity:
            raise ValueError("confusions must be exact ConfusionAntigravity")
        _exact_int_antigravity(self.target_fn_delta, "target_fn_delta")
        _exact_float_antigravity(self.global_f1, "global_f1")
        if type(self.qualified) is not bool:
            raise ValueError("qualified must be an exact bool")
