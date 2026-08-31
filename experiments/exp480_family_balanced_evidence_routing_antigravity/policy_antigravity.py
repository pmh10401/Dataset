"""Family-balanced candidate-wise KNN policy for Exp480."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config_antigravity import FEATURE_KEYS_ANTIGRAVITY, METHODS_ANTIGRAVITY
from .contracts_antigravity import EvidenceRowAntigravity


BASELINE_METHOD_ANTIGRAVITY = METHODS_ANTIGRAVITY[0]
ALTERNATIVE_METHODS_ANTIGRAVITY = METHODS_ANTIGRAVITY[1:]


@dataclass(frozen=True)
class CandidateAssessmentAntigravity:
    method: str
    passed: bool
    support: int
    positive_auc_rate: float
    positive_zf_rate: float
    lower_auc_gain: float
    mean_zf_gain: float
    neighbour_families: tuple[str, ...]


@dataclass(frozen=True)
class FamilyBalancedDecisionAntigravity:
    method: str
    accepted: bool
    reason: str
    candidate: str
    support: int
    positive_auc_rate: float
    positive_zf_rate: float
    lower_auc_gain: float
    mean_zf_gain: float
    neighbour_families: tuple[str, ...]
    candidates: tuple[CandidateAssessmentAntigravity, ...]


def nearest_family_indices_antigravity(
    distances: np.ndarray, families: Sequence[str], k: int
) -> np.ndarray:
    """Keep the nearest row from each family under a stable distance order."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(families):
        raise ValueError("distances and families must be aligned one-dimensional values")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    chosen: list[int] = []
    seen: set[str] = set()
    for raw_index in np.argsort(values, kind="stable"):
        index = int(raw_index)
        family = str(families[index])
        if family not in seen:
            seen.add(family)
            chosen.append(index)
            if len(chosen) == k:
                break
    return np.asarray(chosen, dtype=np.int64)


def _transformed_tuple_antigravity(features: Sequence[float]) -> np.ndarray:
    raw = np.asarray(features, dtype=np.float64)
    if raw.shape != (len(FEATURE_KEYS_ANTIGRAVITY),):
        raise ValueError("features must contain the seven frozen values")
    raw = raw.copy()
    raw[0] = np.log1p(max(0.0, raw[0]))
    raw[1] = np.log1p(max(0.0, raw[1]))
    return raw


def _query_tuple_antigravity(features: Mapping[str, float]) -> np.ndarray:
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    try:
        values = tuple(float(features[key]) for key in FEATURE_KEYS_ANTIGRAVITY)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("query is missing a finite frozen feature") from error
    return _transformed_tuple_antigravity(values)


class FamilyBalancedCandidatePolicyAntigravity:
    """Evaluate all alternatives with at most one nearest vote per source family."""

    def __init__(
        self,
        *,
        k: int = 31,
        min_support: int = 20,
        positive_rate_threshold: float = 0.90,
        lower_quantile: float = 0.20,
        min_auc_gain: float = 0.0,
        min_zf_gain: float = 0.0,
    ) -> None:
        if type(k) is not int or type(min_support) is not int or not 1 <= min_support <= k:
            raise ValueError("require exact integers with 1 <= min_support <= k")
        if not 0.0 <= positive_rate_threshold <= 1.0:
            raise ValueError("positive_rate_threshold must be in [0, 1]")
        if not 0.0 <= lower_quantile <= 1.0:
            raise ValueError("lower_quantile must be in [0, 1]")
        self.k = k
        self.min_support = min_support
        self.positive_rate_threshold = float(positive_rate_threshold)
        self.lower_quantile = float(lower_quantile)
        self.min_auc_gain = float(min_auc_gain)
        self.min_zf_gain = float(min_zf_gain)
        self._fitted = False

    def fit(
        self, rows: Iterable[EvidenceRowAntigravity]
    ) -> "FamilyBalancedCandidatePolicyAntigravity":
        fitted = tuple(rows)
        if not fitted or any(type(row) is not EvidenceRowAntigravity for row in fitted):
            raise ValueError("fit requires strict EvidenceRowAntigravity rows")
        matrix = np.stack(
            [_transformed_tuple_antigravity(row.features) for row in fitted]
        )
        finite = np.isfinite(matrix)
        medians = np.asarray(
            [
                np.median(matrix[finite[:, index], index])
                if finite[:, index].any()
                else 0.0
                for index in range(matrix.shape[1])
            ],
            dtype=np.float64,
        )
        matrix = np.where(finite, matrix, medians[None, :])
        scales = matrix.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        self._medians = medians
        self._mean = matrix.mean(axis=0)
        self._scale = scales
        self._matrix = (matrix - self._mean[None, :]) / self._scale[None, :]
        self._families = tuple(row.source_family for row in fitted)
        baseline_index = 0
        alternatives = []
        for method_index, method in enumerate(ALTERNATIVE_METHODS_ANTIGRAVITY, start=1):
            auc_gains = np.asarray(
                [row.scores[method_index].auc - row.scores[baseline_index].auc for row in fitted],
                dtype=np.float64,
            )
            zf_gains = np.asarray(
                [row.scores[method_index].zf - row.scores[baseline_index].zf for row in fitted],
                dtype=np.float64,
            )
            alternatives.append((method, auc_gains, zf_gains))
        self._alternatives = tuple(alternatives)
        self._fitted = True
        return self

    def _assess_antigravity(
        self, method: str, auc_gains: np.ndarray, zf_gains: np.ndarray, distances: np.ndarray
    ) -> CandidateAssessmentAntigravity:
        neighbours = nearest_family_indices_antigravity(distances, self._families, self.k)
        selected_auc = auc_gains[neighbours]
        selected_zf = zf_gains[neighbours]
        support = int(len(neighbours))
        positive_auc_rate = float(np.mean(selected_auc >= 0.0))
        positive_zf_rate = float(np.mean(selected_zf >= 0.0))
        lower_auc_gain = float(np.quantile(selected_auc, self.lower_quantile))
        mean_zf_gain = float(np.mean(selected_zf))
        passed = (
            support >= self.min_support
            and positive_auc_rate >= self.positive_rate_threshold
            and positive_zf_rate >= self.positive_rate_threshold
            and lower_auc_gain >= self.min_auc_gain
            and mean_zf_gain > self.min_zf_gain
        )
        return CandidateAssessmentAntigravity(
            method=method,
            passed=passed,
            support=support,
            positive_auc_rate=positive_auc_rate,
            positive_zf_rate=positive_zf_rate,
            lower_auc_gain=lower_auc_gain,
            mean_zf_gain=mean_zf_gain,
            neighbour_families=tuple(self._families[int(index)] for index in neighbours),
        )

    def route(self, features: Mapping[str, float]) -> FamilyBalancedDecisionAntigravity:
        if not self._fitted:
            raise RuntimeError("fit must be called before route")
        raw = _query_tuple_antigravity(features)
        raw = np.where(np.isfinite(raw), raw, self._medians)
        query = (raw - self._mean) / self._scale
        distances = np.linalg.norm(self._matrix - query[None, :], axis=1)
        candidates = tuple(
            self._assess_antigravity(method, auc_gains, zf_gains, distances)
            for method, auc_gains, zf_gains in self._alternatives
        )
        method_rank = {method: index for index, method in enumerate(ALTERNATIVE_METHODS_ANTIGRAVITY)}
        passed = tuple(candidate for candidate in candidates if candidate.passed)
        considered = passed or candidates
        best = max(
            considered,
            key=lambda candidate: (
                candidate.mean_zf_gain,
                candidate.lower_auc_gain,
                -method_rank[candidate.method],
            ),
        )
        accepted = bool(passed)
        if accepted:
            method = best.method
            reason = "family_evidence_accept"
        else:
            method = BASELINE_METHOD_ANTIGRAVITY
            reason = (
                "insufficient_family_support"
                if all(candidate.support < self.min_support for candidate in candidates)
                else "candidate_gates_failed"
            )
        decision = FamilyBalancedDecisionAntigravity(
            method=method,
            accepted=accepted,
            reason=reason,
            candidate=best.method,
            support=best.support,
            positive_auc_rate=best.positive_auc_rate,
            positive_zf_rate=best.positive_zf_rate,
            lower_auc_gain=best.lower_auc_gain,
            mean_zf_gain=best.mean_zf_gain,
            neighbour_families=best.neighbour_families,
            candidates=candidates,
        )
        if any(not math.isfinite(value) for value in (
            decision.positive_auc_rate,
            decision.positive_zf_rate,
            decision.lower_auc_gain,
            decision.mean_zf_gain,
        )):
            raise RuntimeError("candidate evidence must remain finite")
        return decision


__all__ = [
    "CandidateAssessmentAntigravity",
    "FamilyBalancedCandidatePolicyAntigravity",
    "FamilyBalancedDecisionAntigravity",
    "nearest_family_indices_antigravity",
]
