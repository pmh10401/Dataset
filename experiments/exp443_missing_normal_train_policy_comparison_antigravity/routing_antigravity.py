"""Pure, label-free Exp443 MiniROCKET routing.

Routing consumes only frozen TEST identity/lane metadata, inherited Exp236
candidate maps, and the four-call result contract.  It never opens the
database, reads labels, or computes an evaluation metric.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import PolicyDatasetResultAntigravity


def _indices(value: object, *, dataset: str, n_test: int, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} for {dataset} is not a sequence")
    result = tuple(value)
    if any(type(index) is not int or index < 0 or index >= n_test for index in result):
        raise ValueError(f"{field} for {dataset} is outside TEST identity")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{field} for {dataset} is not sorted and unique")
    return result


def _coordinates(value: object, *, field: str) -> frozenset[tuple[str, int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{field} is not an iterable of coordinates")
    result: set[tuple[str, int]] = set()
    for coordinate in value:
        if not isinstance(coordinate, Sequence) or len(coordinate) != 2:
            raise ValueError(f"{field} has malformed coordinate")
        dataset, index = coordinate
        if type(dataset) is not str or not dataset or type(index) is not int or index < 0:
            raise ValueError(f"{field} has malformed coordinate")
        if (dataset, index) in result:
            raise ValueError(f"{field} contains duplicate coordinate")
        result.add((dataset, index))
    return frozenset(result)


def _snapshot_datasets(snapshot: object) -> tuple[str, ...]:
    datasets = getattr(snapshot, "datasets", None)
    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        raise ValueError("snapshot datasets are invalid")
    result = tuple(datasets)
    if any(type(dataset) is not str or not dataset for dataset in result) or result != tuple(sorted(set(result))):
        raise ValueError("snapshot datasets are not sorted and unique")
    return result


def _exact_normal_coordinates(
    snapshot: object,
    datasets: tuple[str, ...],
    n_test_by_dataset: Mapping[str, int],
    explicit: object | None,
) -> frozenset[tuple[str, int]]:
    if explicit is not None:
        coordinates = _coordinates(explicit, field="exact normal coordinates")
    elif hasattr(snapshot, "exact_normal_coordinates"):
        coordinates = _coordinates(getattr(snapshot, "exact_normal_coordinates"), field="exact normal coordinates")
    else:
        # FrozenSourceSnapshot intentionally stores Exp236 lanes and the
        # verified Exp291 Hard set, not a duplicate Exp291 normal map.  The
        # exact-normal set is nevertheless determined by the frozen algebra:
        # (Exp236 Hard ∪ Exp236 promotion) − Exp291 Hard.
        lanes = getattr(snapshot, "exp236_lanes", None)
        hard291 = _coordinates(getattr(snapshot, "hard_coordinates_exp291", ()), field="Exp291 Hard coordinates")
        if not isinstance(lanes, Mapping):
            raise ValueError("snapshot has no exact-normal authority")
        base: set[tuple[str, int]] = set()
        for dataset in datasets:
            entry = lanes.get(dataset)
            if not isinstance(entry, Mapping):
                raise ValueError("snapshot Exp236 lane is missing")
            base.update((dataset, index) for index in _indices(entry.get("hard"), dataset=dataset, n_test=n_test_by_dataset[dataset], field="Exp236 hard"))
            base.update((dataset, index) for index in _indices(entry.get("promotion"), dataset=dataset, n_test=n_test_by_dataset[dataset], field="Exp236 promotion"))
        coordinates = frozenset(base.difference(hard291))
    for dataset, index in coordinates:
        if dataset not in n_test_by_dataset or index >= n_test_by_dataset[dataset]:
            raise ValueError("exact-normal coordinate is outside TEST identity")
    return coordinates


@dataclass(frozen=True)
class PolicyPredictionAntigravity:
    """Deterministic label-free routing rows and aggregate coordinates."""

    policy: str
    rows: tuple[Mapping[str, object], ...]
    promotions: frozenset[tuple[str, int]]
    pre_exact_coordinates: frozenset[tuple[str, int]]
    exact_normal_removals: frozenset[tuple[str, int]]
    final_hard_coordinates: frozenset[tuple[str, int]]
    contains_test_labels: bool = False
    contains_evaluation_metrics: bool = False

    def __post_init__(self) -> None:
        if self.policy not in FROZEN_CONFIG.policies:
            raise ValueError("prediction policy is not preregistered")
        if type(self.contains_test_labels) is not bool or type(self.contains_evaluation_metrics) is not bool:
            raise ValueError("prediction safety flags must be bool")
        rows = tuple(self.rows)
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("prediction rows are not mappings")
        datasets = tuple(row.get("dataset") for row in rows)
        if any(type(dataset) is not str or not dataset for dataset in datasets) or datasets != tuple(sorted(set(datasets))):
            raise ValueError("prediction row identity is not sorted and unique")
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in rows))
        for name in ("promotions", "pre_exact_coordinates", "exact_normal_removals", "final_hard_coordinates"):
            object.__setattr__(self, name, _coordinates(getattr(self, name), field=name))

    @property
    def dataset_rows(self) -> tuple[Mapping[str, object], ...]:
        return self.rows

    @property
    def promotion_coordinates(self) -> frozenset[tuple[str, int]]:
        return self.promotions

    def to_payload(self) -> dict[str, object]:
        """Return canonical-safe payload without label/evaluation fields."""

        return {
            "policy": self.policy,
            "rows": [dict(row) for row in self.rows],
            "promotions": [[dataset, index] for dataset, index in sorted(self.promotions)],
            "pre_exact": [[dataset, index] for dataset, index in sorted(self.pre_exact_coordinates)],
            "exact_normal_removed": [[dataset, index] for dataset, index in sorted(self.exact_normal_removals)],
            "final_hard": [[dataset, index] for dataset, index in sorted(self.final_hard_coordinates)],
        }


def canonical_prediction_bytes_antigravity(
    prediction: PolicyPredictionAntigravity,
) -> bytes:
    if type(prediction) is not PolicyPredictionAntigravity:
        raise ValueError("prediction has the wrong contract type")
    return json.dumps(
        prediction.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def route_policy_antigravity(
    snapshot: object,
    result_by_dataset: Mapping[str, PolicyDatasetResultAntigravity],
    policy: str,
    *,
    exact_normal_coordinates: Iterable[tuple[str, int]] | None = None,
) -> PolicyPredictionAntigravity:
    """Apply the frozen promotion/exact-normal routing algebra.

    For affected datasets, ``result_by_dataset`` carries the one recomputed
    policy result.  Every other dataset imports the verified Exp249 seed
    intersection candidate map.  Candidates outside Standard are retained in
    each row's audit evidence but cannot promote into Hard.
    """

    if policy not in FROZEN_CONFIG.policies:
        raise ValueError("routing policy is not preregistered")
    if not isinstance(result_by_dataset, Mapping):
        raise ValueError("result_by_dataset must be a mapping")
    datasets = _snapshot_datasets(snapshot)
    n_test_values = getattr(snapshot, "test_row_count_by_dataset", None)
    if not isinstance(n_test_values, Mapping) or set(n_test_values) != set(datasets):
        raise ValueError("snapshot TEST identity authority is incomplete")
    n_test_by_dataset: dict[str, int] = {}
    for dataset in datasets:
        value = n_test_values[dataset]
        if type(value) is not int or value < 1:
            raise ValueError("snapshot TEST row count is invalid")
        n_test_by_dataset[dataset] = value

    affected = tuple(getattr(snapshot, "affected_datasets", ()))
    if affected != tuple(sorted(set(affected))) or not set(affected).issubset(datasets):
        raise ValueError("snapshot affected identity is invalid")
    if set(result_by_dataset) != set(affected):
        raise ValueError("each affected dataset requires exactly one recomputed result")

    exact_normal = _exact_normal_coordinates(snapshot, datasets, n_test_by_dataset, exact_normal_coordinates)
    source_candidates = getattr(snapshot, "exp249_candidate_maps", None)
    lanes_source = getattr(snapshot, "exp236_lanes", None)
    if not isinstance(source_candidates, Mapping) or not isinstance(lanes_source, Mapping):
        raise ValueError("snapshot routing authorities are incomplete")
    rows: list[Mapping[str, object]] = []
    promotions: set[tuple[str, int]] = set()
    pre_exact: set[tuple[str, int]] = set()
    removals: set[tuple[str, int]] = set()
    final_hard: set[tuple[str, int]] = set()

    for dataset in datasets:
        n_test = n_test_by_dataset[dataset]
        lanes = lanes_source.get(dataset)
        if not isinstance(lanes, Mapping) or set(lanes) != {"hard", "standard", "priority", "promotion"}:
            raise ValueError(f"snapshot lanes for {dataset} are invalid")
        inherited_hard = _indices(lanes["hard"], dataset=dataset, n_test=n_test, field="inherited Hard")
        standard = _indices(lanes["standard"], dataset=dataset, n_test=n_test, field="Standard")
        priority = _indices(lanes["priority"], dataset=dataset, n_test=n_test, field="Priority")

        if dataset in result_by_dataset:
            result = result_by_dataset[dataset]
            if type(result) is not PolicyDatasetResultAntigravity or result.dataset != dataset or result.policy != policy:
                raise ValueError(f"recomputed result identity for {dataset} is invalid")
            if result.test_rows != n_test or set(result.candidates_by_kernel) != {1024, 4096}:
                raise ValueError(f"recomputed result dimensions for {dataset} are invalid")
            mini1024 = _indices(result.candidates_by_kernel[1024], dataset=dataset, n_test=n_test, field="MiniROCKET 1024")
            mini4096 = _indices(result.candidates_by_kernel[4096], dataset=dataset, n_test=n_test, field="MiniROCKET 4096")
            candidate_source = "recomputed_policy"
        else:
            candidate_map = source_candidates.get(dataset)
            if not isinstance(candidate_map, Mapping) or set(candidate_map) != {1024, 4096}:
                raise ValueError(f"frozen Exp249 candidates for {dataset} are invalid")
            mini1024 = _indices(candidate_map[1024], dataset=dataset, n_test=n_test, field="MiniROCKET 1024")
            mini4096 = _indices(candidate_map[4096], dataset=dataset, n_test=n_test, field="MiniROCKET 4096")
            candidate_source = "exp249_frozen_seed_intersection"

        union = set(mini1024) | set(mini4096)
        promotion = tuple(sorted(union.intersection(standard)))
        outside_standard = tuple(sorted(union.difference(standard)))
        pre = tuple(sorted(set(inherited_hard) | set(promotion)))
        removed = tuple(sorted(index for index in pre if (dataset, index) in exact_normal))
        final = tuple(sorted(index for index in pre if (dataset, index) not in exact_normal))
        promotion_coordinates = {(dataset, index) for index in promotion}
        pre_coordinates = {(dataset, index) for index in pre}
        removal_coordinates = {(dataset, index) for index in removed}
        final_coordinates = {(dataset, index) for index in final}
        promotions.update(promotion_coordinates)
        pre_exact.update(pre_coordinates)
        removals.update(removal_coordinates)
        final_hard.update(final_coordinates)
        rows.append(
            MappingProxyType(
                {
                    "dataset": dataset,
                    "policy": policy,
                    "n_test": n_test,
                    "mini1024": mini1024,
                    "mini4096": mini4096,
                    "inherited_hard": inherited_hard,
                    "inherited_standard": standard,
                    "inherited_priority": priority,
                    "outside_standard": outside_standard,
                    "promotion": promotion,
                    "pre_exact": pre,
                    "exact_normal_removed": removed,
                    "final_hard": final,
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "provenance": {
                        "candidate_source": candidate_source,
                        "routing_rule": "standard_intersection_union_then_exact_normal_subtraction",
                    },
                }
            )
        )

    return PolicyPredictionAntigravity(
        policy,
        tuple(rows),
        frozenset(promotions),
        frozenset(pre_exact),
        frozenset(removals),
        frozenset(final_hard),
        False,
        False,
    )


__all__ = [
    "PolicyPredictionAntigravity",
    "canonical_prediction_bytes_antigravity",
    "route_policy_antigravity",
]
