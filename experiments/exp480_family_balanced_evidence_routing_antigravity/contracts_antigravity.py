"""Strict immutable evidence contracts for Exp480."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config_antigravity import FEATURE_KEYS_ANTIGRAVITY, METHODS_ANTIGRAVITY


@dataclass(frozen=True)
class CandidateMetricAntigravity:
    method: str
    auc: float
    zf: int

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in METHODS_ANTIGRAVITY:
            raise ValueError("method must be one of the four frozen methods")
        if type(self.auc) is not float or not math.isfinite(self.auc):
            raise ValueError("AUC must be an exact finite float")
        if not 0.0 <= self.auc <= 1.0:
            raise ValueError("AUC must be in [0, 1]")
        if type(self.zf) is not int or self.zf < 0:
            raise ValueError("ZF-TP must be an exact nonnegative integer")


@dataclass(frozen=True)
class EvidenceRowAntigravity:
    dataset: str
    source_family: str
    proposal: str
    features: tuple[float, ...]
    scores: tuple[CandidateMetricAntigravity, ...]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or "_normal_" not in self.dataset:
            raise ValueError("dataset must contain _normal_")
        expected_family = self.dataset.split("_normal_", 1)[0]
        if type(self.source_family) is not str or self.source_family != expected_family:
            raise ValueError("source_family must be the prefix before the first _normal_")
        if type(self.proposal) is not str or self.proposal not in METHODS_ANTIGRAVITY:
            raise ValueError("proposal must be one of the four frozen methods")
        if type(self.features) is not tuple or len(self.features) != len(
            FEATURE_KEYS_ANTIGRAVITY
        ):
            raise ValueError("features must be the exact seven-value tuple")
        if any(type(value) is not float or not math.isfinite(value) for value in self.features):
            raise ValueError("features must contain exact finite floats")
        if type(self.scores) is not tuple or tuple(
            metric.method for metric in self.scores
        ) != METHODS_ANTIGRAVITY:
            raise ValueError("scores must contain the four frozen methods in order")


__all__ = ["CandidateMetricAntigravity", "EvidenceRowAntigravity"]
