"""Immutable data contracts for the planned Exp455 routing boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True, slots=True)
class DatasetRoutingFeaturesAntigravity:
    dataset: str
    family: str
    target_length: int
    baseline_count: int
    candidate_count: int
    overlap_count: int
    union_count: int
    added_count: int
    removed_count: int
    overlap_ppm: int


@dataclass(frozen=True, slots=True)
class DatasetRoutingAuthorityAntigravity:
    dataset: str
    family: str
    target_length: int
    method: str | None
    baseline_indices: frozenset[int]
    candidate_indices: frozenset[int] | None
    baseline_confusion: ConfusionAntigravity
    candidate_confusion: ConfusionAntigravity | None


@dataclass(frozen=True, slots=True)
class FrozenRoutingAuthorityAntigravity:
    rows: tuple[DatasetRoutingAuthorityAntigravity, ...]
    exp454_leaf_sha256: tuple[tuple[str, str], ...]
    label_query_count: int = 0


@dataclass(frozen=True, slots=True)
class RouteRuleAntigravity:
    feature_name: str | None
    threshold_twice: int | None
    candidate_when: str


@dataclass(frozen=True, slots=True)
class FoldDecisionAntigravity:
    dataset: str
    family: str
    method: str
    selected_route: str
    rule: RouteRuleAntigravity


@dataclass(frozen=True, slots=True)
class ComparisonMetricAntigravity:
    name: str
    confusion: ConfusionAntigravity
    f1: float


@dataclass(frozen=True, slots=True)
class RoutingEvaluationAntigravity:
    comparisons: tuple[ComparisonMetricAntigravity, ...]
    family_fn_delta: tuple[tuple[str, int], ...]
    decisions: tuple[FoldDecisionAntigravity, ...]
    passed: bool
    status: str


@dataclass(frozen=True, slots=True)
class FrozenResultAntigravity:
    root: str
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str
