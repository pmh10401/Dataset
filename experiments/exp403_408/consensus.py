"""TRAIN-only source independence and review-only Exp408 consensus.

This module intentionally does not evaluate TEST outcomes or modify frozen
lanes.  It treats same-lineage sources as one component before using aligned
TRAIN OOF Spearman correlation to collapse other redundant sources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Hashable

import numpy as np

from .contracts import ALPHA_KEYS, ShadowSourcePrediction
from .feature_registry import FEATURE_SOURCES
from .foundation import FrozenLaneUniverse


VARIANTS = (
    "tri_view_only",
    "synthetic_only",
    "tri_view_and_synthetic",
    "one_new_plus_existing_independent",
    "two_new_plus_existing_independent",
)

TRI_VIEW_KIND = "tri_view"
SYNTHETIC_KIND = "synthetic"
EXISTING_KIND = "existing"
_VALID_KINDS = frozenset((TRI_VIEW_KIND, SYNTHETIC_KIND, EXISTING_KIND))
_TRI_VIEW_SOURCE_NAMES = frozenset(("tri_view",))
_SYNTHETIC_SOURCE_NAMES = frozenset(
    (
        "synthetic",
        "synthetic_logistic_regression",
        "synthetic_hist_gradient_boosting",
        "synthetic_small_mlp",
    )
)
_REGISTERED_EXISTING_SOURCE_MARKERS = frozenset(
    source.name for source in FEATURE_SOURCES
)


@dataclass(frozen=True)
class ConsensusSource:
    """One frozen Shadow or existing-evidence source for one dataset.

    ``train_oof_scores`` accepts either a sequence in its frozen shared row
    order or a mapping from OOF row identity to score.  Mappings are preferred
    when a producer can expose explicit fold-row identities.
    """

    name: str
    lineage: str
    kind: str
    train_oof_scores: Mapping[Hashable, float] | Sequence[float]
    shadow_predictions: Mapping[str, Sequence[ShadowSourcePrediction]] = field(
        default_factory=dict
    )
    dataset: str = ""


@dataclass(frozen=True)
class IndependenceComponent:
    """One evidence vote after lineage and OOF-correlation collapse."""

    source_names: tuple[str, ...]
    source_markers: tuple[str, ...]
    lineages: tuple[str, ...]
    kinds: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ConsensusCandidate:
    """A review-only candidate; it never contains an operational lane action."""

    dataset: str
    candidate_index: int
    variant: str
    alpha: str
    supporting_components: tuple[tuple[str, ...], ...]
    supporting_source_markers: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ConsensusResult:
    """Exact frozen lane copies plus separately packaged review candidates."""

    dataset: str
    hard_indices: frozenset[int]
    standard_indices: frozenset[int]
    priority_indices: frozenset[int]
    components: tuple[IndependenceComponent, ...]
    candidates_by_variant: Mapping[
        str, Mapping[str, tuple[ConsensusCandidate, ...]]
    ]

    @property
    def variants(self) -> Mapping[str, Mapping[str, tuple[ConsensusCandidate, ...]]]:
        """Compatibility name for the five review-only variant packages."""
        return self.candidates_by_variant


def build_independence_components(
    sources: Sequence[ConsensusSource], threshold: float = 0.90, minimum_rows: int = 20
) -> tuple[IndependenceComponent, ...]:
    """Collapse identical lineage first, then redundant TRAIN OOF components.

    A component with any insufficient or undefined distinct-lineage comparison
    is reported as ``independence_unverified``.  It remains visible for audit,
    but cannot become an independent consensus vote.
    """
    normalized = _normalize_sources(sources)
    limit = _validate_independence_parameters(threshold, minimum_rows)
    lineage_groups = _lineage_groups(normalized)
    group_count = len(lineage_groups)
    parent = list(range(group_count))
    unverified_groups: set[int] = set()

    for left in range(group_count):
        for right in range(left + 1, group_count):
            correlation = _aligned_spearman(
                _representative_oof(lineage_groups[left]),
                _representative_oof(lineage_groups[right]),
                minimum_rows=limit,
            )
            if correlation is None:
                unverified_groups.update((left, right))
            elif abs(correlation) >= float(threshold):
                _union(parent, left, right)

    grouped: dict[int, list[int]] = {}
    for group_index in range(group_count):
        grouped.setdefault(_find(parent, group_index), []).append(group_index)

    components: list[IndependenceComponent] = []
    for group_indices in grouped.values():
        grouped_sources = tuple(sorted(
            (
                source
                for group_index in group_indices
                for source in lineage_groups[group_index]
            ),
            key=lambda source: source.name,
        ))
        components.append(
            IndependenceComponent(
                source_names=tuple(source.name for source in grouped_sources),
                source_markers=tuple(
                    expected_source_marker(source.kind, source.name)
                    for source in grouped_sources
                ),
                lineages=tuple(sorted({source.lineage for source in grouped_sources})),
                kinds=tuple(sorted({source.kind for source in grouped_sources})),
                status=(
                    "independence_unverified"
                    if any(group_index in unverified_groups for group_index in group_indices)
                    else "independent"
                ),
            )
        )
    return tuple(components)


def build_consensus(
    frozen_lane_universe: FrozenLaneUniverse,
    source_predictions: Sequence[ConsensusSource],
    *,
    threshold: float = 0.90,
    minimum_rows: int = 20,
) -> ConsensusResult:
    """Build all five preregistered review-only variants for one dataset."""
    if not isinstance(frozen_lane_universe, FrozenLaneUniverse):
        raise TypeError("frozen_lane_universe must be a verified FrozenLaneUniverse")
    sources = _normalize_sources(source_predictions)
    if not sources:
        raise ValueError("Exp408 consensus requires at least one source")
    components = build_independence_components(
        sources, threshold=threshold, minimum_rows=minimum_rows
    )
    dataset = _resolve_dataset(sources)
    lane = frozen_lane_universe.lane_for(dataset)
    if set(lane) != {"hard", "standard", "priority"}:
        raise ValueError(f"dataset is outside the verified frozen lanes: {dataset}")
    hard_indices = frozenset(int(index) for index in lane["hard"])
    standard_indices = frozenset(int(index) for index in lane["standard"])
    priority_indices = frozenset(int(index) for index in lane["priority"])
    source_lookup = {source.name: source for source in sources}
    component_candidates = {
        component.source_names: _component_candidates(
            component, source_lookup, hard_indices
        )
        for component in components
    }
    component_markers = {
        component.source_names: component.source_markers for component in components
    }
    packages = {
        variant: _variant_candidates(
            variant,
            dataset=dataset,
            components=components,
            component_candidates=component_candidates,
            component_markers=component_markers,
        )
        for variant in VARIANTS
    }
    return ConsensusResult(
        dataset=dataset,
        hard_indices=hard_indices,
        standard_indices=standard_indices,
        priority_indices=priority_indices,
        components=components,
        candidates_by_variant=MappingProxyType(packages),
    )


def _normalize_sources(sources: Sequence[ConsensusSource]) -> tuple[ConsensusSource, ...]:
    normalized = tuple(sources)
    names: set[str] = set()
    for source in normalized:
        if not isinstance(source, ConsensusSource):
            raise TypeError("sources must contain ConsensusSource values")
        if not source.name or not source.lineage:
            raise ValueError("source name and lineage must be non-empty")
        if source.kind not in _VALID_KINDS:
            raise ValueError(f"unknown Exp408 source kind: {source.kind}")
        expected_source_marker(source.kind, source.name)
        if source.name in names:
            raise ValueError(f"duplicate Exp408 source name: {source.name}")
        names.add(source.name)
        _oof_mapping(source.train_oof_scores)
        _validate_shadow_predictions(source)
    return normalized


def expected_source_marker(kind: str, name: str) -> str:
    """Return the one frozen structured-record marker for a source identity."""
    if kind == TRI_VIEW_KIND and name in _TRI_VIEW_SOURCE_NAMES:
        return "self_supervised_shadow"
    if kind == SYNTHETIC_KIND and name in _SYNTHETIC_SOURCE_NAMES:
        return "synthetic_anomaly_shadow"
    if kind == EXISTING_KIND and name in _REGISTERED_EXISTING_SOURCE_MARKERS:
        return name
    raise ValueError(f"unregistered Exp408 source kind/name: {kind}/{name}")


def _validate_independence_parameters(threshold: float, minimum_rows: int) -> int:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be a finite value from zero through one")
    if isinstance(minimum_rows, bool) or int(minimum_rows) != minimum_rows:
        raise ValueError("minimum_rows must be an integer")
    if int(minimum_rows) < 2:
        raise ValueError("minimum_rows must be at least two")
    return int(minimum_rows)


def _lineage_groups(
    sources: Sequence[ConsensusSource],
) -> tuple[tuple[ConsensusSource, ...], ...]:
    grouped: dict[str, list[ConsensusSource]] = {}
    for source in sources:
        grouped.setdefault(source.lineage, []).append(source)
    return tuple(tuple(group) for group in grouped.values())


def _representative_oof(group: Sequence[ConsensusSource]) -> dict[Hashable, float]:
    representative = min(group, key=lambda source: source.name)
    return _oof_mapping(representative.train_oof_scores)


def _oof_mapping(
    scores: Mapping[Hashable, float] | Sequence[float],
) -> dict[Hashable, float]:
    values = dict(scores) if isinstance(scores, Mapping) else dict(enumerate(scores))
    if not values:
        return {}
    for row_id, score in values.items():
        if isinstance(score, bool) or not np.isfinite(float(score)):
            raise ValueError(f"TRAIN OOF score is not finite: {row_id!r}")
    return {row_id: float(score) for row_id, score in values.items()}


def _aligned_spearman(
    left: Mapping[Hashable, float],
    right: Mapping[Hashable, float],
    *,
    minimum_rows: int,
) -> float | None:
    aligned_ids = tuple(row_id for row_id in left if row_id in right)
    if len(aligned_ids) < minimum_rows:
        return None
    left_values = np.asarray([left[row_id] for row_id in aligned_ids], dtype=float)
    right_values = np.asarray([right[row_id] for row_id in aligned_ids], dtype=float)
    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    value = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    return value if np.isfinite(value) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _validate_shadow_predictions(source: ConsensusSource) -> None:
    predictions = source.shadow_predictions
    expected_marker = expected_source_marker(source.kind, source.name)
    if set(predictions) != set(ALPHA_KEYS):
        raise ValueError("Shadow predictions must contain exactly the frozen alpha keys")
    for alpha, rows in predictions.items():
        if not isinstance(rows, Sequence):
            raise ValueError(f"Shadow predictions for alpha {alpha} must be a sequence")
        for row in rows:
            if not isinstance(row, ShadowSourcePrediction):
                raise TypeError("Exp408 requires structured ShadowSourcePrediction rows")
            if set(row.scores_by_alpha) != set(ALPHA_KEYS):
                raise ValueError("Shadow scores must contain exactly the frozen alpha keys")
            if row.source != expected_marker:
                raise ValueError(
                    "Shadow source marker does not match the frozen "
                    f"{source.kind}/{source.name} identity"
                )
            if not row.dataset or int(row.candidate_index) < 0:
                raise ValueError("Shadow prediction identity is invalid")
            if any(not np.isfinite(float(value)) for value in row.scores_by_alpha.values()):
                raise ValueError("Shadow scores must be finite")


def _resolve_dataset(sources: Sequence[ConsensusSource]) -> str:
    datasets = {source.dataset for source in sources if source.dataset}
    datasets.update(
        row.dataset
        for source in sources
        for rows in source.shadow_predictions.values()
        for row in rows
    )
    if len(datasets) != 1:
        raise ValueError("Exp408 sources must describe exactly one dataset")
    return next(iter(datasets))


def _component_candidates(
    component: IndependenceComponent,
    source_lookup: Mapping[str, ConsensusSource],
    hard_indices: frozenset[int],
) -> Mapping[str, frozenset[int]]:
    result: dict[str, frozenset[int]] = {}
    for alpha in ALPHA_KEYS:
        indices = frozenset(
            int(row.candidate_index)
            for source_name in component.source_names
            for row in source_lookup[source_name].shadow_predictions[alpha]
            if int(row.candidate_index) not in hard_indices
        )
        result[alpha] = indices
    return MappingProxyType(result)


def _variant_candidates(
    variant: str,
    *,
    dataset: str,
    components: Sequence[IndependenceComponent],
    component_candidates: Mapping[tuple[str, ...], Mapping[str, frozenset[int]]],
    component_markers: Mapping[tuple[str, ...], tuple[str, ...]],
) -> Mapping[str, tuple[ConsensusCandidate, ...]]:
    output: dict[str, tuple[ConsensusCandidate, ...]] = {}
    for alpha in ALPHA_KEYS:
        candidate_support = _candidate_support_for_variant(
            variant, alpha, components, component_candidates
        )
        output[alpha] = tuple(
            ConsensusCandidate(
                dataset=dataset,
                candidate_index=index,
                variant=variant,
                alpha=alpha,
                supporting_components=support,
                supporting_source_markers=tuple(
                    component_markers[component] for component in support
                ),
            )
            for index, support in sorted(candidate_support.items())
        )
    return MappingProxyType(output)


def _candidate_support_for_variant(
    variant: str,
    alpha: str,
    components: Sequence[IndependenceComponent],
    component_candidates: Mapping[tuple[str, ...], Mapping[str, frozenset[int]]],
) -> dict[int, tuple[tuple[str, ...], ...]]:
    tri = _components_with_kind(components, TRI_VIEW_KIND)
    synthetic = _components_with_kind(components, SYNTHETIC_KIND)
    existing = _components_with_kind(components, EXISTING_KIND)
    if variant == "tri_view_only":
        return _union_support(alpha, tri, component_candidates)
    if variant == "synthetic_only":
        return _union_support(alpha, synthetic, component_candidates)
    if variant == "tri_view_and_synthetic":
        return _intersection_support(alpha, (tri, synthetic), component_candidates)
    if variant == "one_new_plus_existing_independent":
        return _union_of_intersections(
            alpha, ((tri, existing), (synthetic, existing)), component_candidates
        )
    if variant == "two_new_plus_existing_independent":
        return _intersection_support(alpha, (tri, synthetic, existing), component_candidates)
    raise ValueError(f"unknown frozen Exp408 variant: {variant}")


def _components_with_kind(
    components: Sequence[IndependenceComponent], kind: str
) -> tuple[IndependenceComponent, ...]:
    return tuple(component for component in components if kind in component.kinds)


def _union_support(
    alpha: str,
    components: Sequence[IndependenceComponent],
    component_candidates: Mapping[tuple[str, ...], Mapping[str, frozenset[int]]],
) -> dict[int, tuple[tuple[str, ...], ...]]:
    result: dict[int, tuple[tuple[str, ...], ...]] = {}
    for component in components:
        for index in component_candidates[component.source_names][alpha]:
            support = result.get(index, ())
            result[index] = tuple(sorted((*support, component.source_names)))
    return result


def _union_of_intersections(
    alpha: str,
    component_groups: Sequence[Sequence[Sequence[IndependenceComponent]]],
    component_candidates: Mapping[tuple[str, ...], Mapping[str, frozenset[int]]],
) -> dict[int, tuple[tuple[str, ...], ...]]:
    result: dict[int, tuple[tuple[str, ...], ...]] = {}
    for groups in component_groups:
        for index, support in _intersection_support(
            alpha, groups, component_candidates
        ).items():
            existing = result.get(index, ())
            result[index] = tuple(sorted(set(existing).union(support)))
    return result


def _intersection_support(
    alpha: str,
    component_groups: Sequence[Sequence[IndependenceComponent]],
    component_candidates: Mapping[tuple[str, ...], Mapping[str, frozenset[int]]],
) -> dict[int, tuple[tuple[str, ...], ...]]:
    choices = tuple(
        tuple(component for component in group if component.status == "independent")
        for group in component_groups
    )
    if any(not group for group in choices):
        return {}
    result: dict[int, tuple[tuple[str, ...], ...]] = {}
    for selected in _cartesian_components(choices):
        if len({component.source_names for component in selected}) != len(selected):
            continue
        shared = set(component_candidates[selected[0].source_names][alpha])
        for component in selected[1:]:
            shared.intersection_update(component_candidates[component.source_names][alpha])
        support = tuple(sorted(component.source_names for component in selected))
        for index in shared:
            result[index] = tuple(sorted(set(result.get(index, ())).union(support)))
    return result


def _cartesian_components(
    groups: Sequence[Sequence[IndependenceComponent]],
) -> tuple[tuple[IndependenceComponent, ...], ...]:
    result: tuple[tuple[IndependenceComponent, ...], ...] = ((),)
    for group in groups:
        result = tuple(
            (*prefix, item) for prefix in result for item in group
        )
    return result
