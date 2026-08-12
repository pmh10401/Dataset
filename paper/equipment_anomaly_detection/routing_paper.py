"""Exact four-policy proposal routes with continuous threshold-equivalent scores."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .contracts_paper import LowFrequencyResult, RawDataset
from .score_adapter_paper import route_preserving_scores
from .sources_paper.stability_paper import StabilityPartition


POLICY_ORDER = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
EXPECTED_COUNTS = (2218, 2248, 2244, 2222)  # Full-population Task 9 assertions, never fixture claims.


def _readonly_float(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite evidence")
    result = np.array(array, dtype=np.float64, copy=True)
    if np.any((result < 0.0) | (result >= 1.0)):
        raise ValueError(f"{name} must be in [0, 1)")
    result.setflags(write=False)
    return result


def _readonly_bool(values: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(bool) or array.size != size:
        raise ValueError(f"{name} must be an aligned bool vector")
    result = np.array(array, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _alerts(values: object, size: int, name: str) -> frozenset[int]:
    if not isinstance(values, frozenset) or any(type(index) is not int or not 0 <= index < size for index in values):
        raise ValueError(f"{name} must be bounded frozenset coordinates")
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class EvidenceSurface:
    """Deeply immutable all-row source evidence and frozen alert coordinates."""

    alerts: Mapping[str, frozenset[int]]
    strength: Mapping[str, np.ndarray]
    evidence_missing: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        alerts, strength, missing = dict(self.alerts), dict(self.strength), dict(self.evidence_missing)
        if not alerts or set(alerts) != set(strength) or set(alerts) != set(missing) or any(type(dataset) is not str or not dataset for dataset in alerts):
            raise ValueError("evidence surface dataset identities are invalid")
        copied_strength = {dataset: _readonly_float(values, "strength") for dataset, values in strength.items()}
        copied_missing = {dataset: _readonly_bool(missing[dataset], copied_strength[dataset].size, "evidence_missing") for dataset in alerts}
        copied_alerts = {dataset: _alerts(alerts[dataset], copied_strength[dataset].size, "alerts") for dataset in alerts}
        object.__setattr__(self, "alerts", MappingProxyType(copied_alerts))
        object.__setattr__(self, "strength", MappingProxyType(copied_strength))
        object.__setattr__(self, "evidence_missing", MappingProxyType(copied_missing))

    @property
    def coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset((dataset, index) for dataset, indices in self.alerts.items() for index in indices)


@dataclass(frozen=True, slots=True)
class PolicyRoute:
    name: str
    alerts: Mapping[str, frozenset[int]]
    strength: Mapping[str, np.ndarray]
    scores: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.name not in POLICY_ORDER:
            raise ValueError("policy name is invalid")
        alerts, strength, scores = dict(self.alerts), dict(self.strength), dict(self.scores)
        if not alerts or set(alerts) != set(strength) or set(alerts) != set(scores):
            raise ValueError("policy route datasets are invalid")
        copied_strength = {dataset: _readonly_float(values, "strength") for dataset, values in strength.items()}
        copied_alerts = {dataset: _alerts(alerts[dataset], copied_strength[dataset].size, "alerts") for dataset in alerts}
        copied_scores: dict[str, np.ndarray] = {}
        for dataset, values in scores.items():
            score = _readonly_float(values, "scores")
            if score.size != copied_strength[dataset].size:
                raise ValueError("policy scores are not aligned")
            expected = np.asarray([index in copied_alerts[dataset] for index in range(score.size)], dtype=bool)
            if not np.array_equal(score >= .5, expected):
                raise ValueError("policy alert and score route mismatch")
            copied_scores[dataset] = score
        object.__setattr__(self, "alerts", MappingProxyType(copied_alerts))
        object.__setattr__(self, "strength", MappingProxyType(copied_strength))
        object.__setattr__(self, "scores", MappingProxyType(copied_scores))

    @property
    def coordinates(self) -> frozenset[tuple[str, int]]:
        return frozenset((dataset, index) for dataset, indices in self.alerts.items() for index in indices)


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    policies: tuple[PolicyRoute, ...]
    order: tuple[str, ...] = POLICY_ORDER

    def __post_init__(self) -> None:
        policies, order = tuple(self.policies), tuple(self.order)
        if order != POLICY_ORDER or tuple(policy.name for policy in policies) != POLICY_ORDER:
            raise ValueError("policy order is invalid")
        if any(not isinstance(policy, PolicyRoute) for policy in policies):
            raise ValueError("policy routes are invalid")
        datasets = set(policies[0].alerts)
        if any(set(policy.alerts) != datasets for policy in policies[1:]):
            raise ValueError("policy route population identities differ")
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "order", order)

    @property
    def by_name(self) -> Mapping[str, PolicyRoute]:
        return MappingProxyType({policy.name: policy for policy in self.policies})


def _source_surface(low_frequency: LowFrequencyResult, raw_datasets: Mapping[str, RawDataset]) -> EvidenceSurface:
    if not isinstance(low_frequency, LowFrequencyResult) or not isinstance(raw_datasets, Mapping):
        raise ValueError("source policy inputs are invalid")
    raw = dict(raw_datasets)
    if set(raw) != set(low_frequency.datasets) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
        raise ValueError("source policy raw population identities are invalid")
    selected = tuple(low_frequency.selected_candidates)
    if len(selected) > 30 or len({candidate.coordinate for candidate in selected}) != len(selected) or len({candidate.dataset for candidate in selected}) != len(selected):
        raise ValueError("source policy candidates violate the frozen budget")
    alerts = {dataset: frozenset() for dataset in low_frequency.datasets}
    strength = {dataset: np.zeros(len(raw[dataset].test_rows), dtype=np.float64) for dataset in low_frequency.datasets}
    missing = {dataset: np.ones(len(raw[dataset].test_rows), dtype=bool) for dataset in low_frequency.datasets}
    for candidate in selected:
        if candidate.instance_index >= strength[candidate.dataset].size:
            raise ValueError("selected source coordinate is outside the raw TEST population")
        alerts[candidate.dataset] = frozenset({candidate.instance_index})
        # Exp440's empirical rank is finite-sample TRAIN-only evidence; its
        # complement is the bounded continuous source strength.
        strength[candidate.dataset][candidate.instance_index] = 1.0 - candidate.second_empirical_rank
        missing[candidate.dataset][candidate.instance_index] = False
    return EvidenceSurface(alerts, strength, missing)


def _filtered_alerts(source: EvidenceSurface, coordinates: frozenset[tuple[str, int]]) -> Mapping[str, frozenset[int]]:
    return MappingProxyType({
        dataset: frozenset(index for index in source.alerts[dataset] if (dataset, index) in coordinates)
        for dataset in source.alerts
    })


def _route(name: str, hard: EvidenceSurface, source: EvidenceSurface, source_coordinates: frozenset[tuple[str, int]]) -> PolicyRoute:
    source_alerts = _filtered_alerts(source, source_coordinates)
    alerts = {dataset: hard.alerts[dataset] | source_alerts[dataset] for dataset in hard.alerts}
    strength: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for dataset in hard.alerts:
        # Every ablation retains the same continuous Hard/source evidence;
        # only its alert mask changes.  Missing leaves are neutral before the
        # max, and only a doubly-missing row remains missing in the adapter.
        hard_values = np.where(hard.evidence_missing[dataset], 0.0, hard.strength[dataset])
        source_values = np.where(source.evidence_missing[dataset], 0.0, source.strength[dataset])
        combined = np.maximum(hard_values, source_values)
        combined_missing = hard.evidence_missing[dataset] & source.evidence_missing[dataset]
        mask = np.asarray([index in alerts[dataset] for index in range(combined.size)], dtype=bool)
        strength[dataset] = combined
        scores[dataset] = route_preserving_scores(combined, mask, combined_missing)
    return PolicyRoute(name, alerts, strength, scores)


def build_policies(
    hard: EvidenceSurface,
    low_frequency: LowFrequencyResult,
    stability: StabilityPartition,
    raw_datasets: Mapping[str, RawDataset],
) -> PolicyBundle:
    """Construct hard-only/source/stable/unstable routes in the frozen order."""
    if not isinstance(hard, EvidenceSurface) or not isinstance(stability, StabilityPartition):
        raise ValueError("policy inputs are invalid")
    source = _source_surface(low_frequency, raw_datasets)
    if set(hard.alerts) != set(source.alerts) or any(hard.strength[dataset].size != source.strength[dataset].size for dataset in hard.alerts):
        raise ValueError("Hard and source evidence populations are mismatched")
    selected = source.coordinates
    if selected != frozenset(stability.selected_coordinates):
        raise ValueError("stability partition identities do not match selected source coordinates")
    if selected & hard.coordinates:
        raise ValueError("Hard and selected source coordinates overlap")
    stable = frozenset(stability.stable_coordinates)
    unstable = frozenset(stability.unstable_coordinates)
    if stable & unstable or stable | unstable != selected:
        raise ValueError("stable/unstable source partition is invalid")
    routes = (
        _route("hard_only", hard, source, frozenset()),
        _route("hard_plus_source", hard, source, selected),
        _route("hard_plus_stable", hard, source, stable),
        _route("hard_plus_unstable", hard, source, unstable),
    )
    return PolicyBundle(routes)


__all__ = [
    "EXPECTED_COUNTS", "POLICY_ORDER", "EvidenceSurface", "PolicyBundle", "PolicyRoute", "build_policies",
]
