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

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
    validate_prediction_payload_antigravity,
)


_PREDICTION_ROW_KEYS_ANTIGRAVITY = frozenset({
    "dataset", "policy", "n_test", "mini1024", "mini4096",
    "inherited_hard", "inherited_standard", "inherited_priority",
    "outside_standard", "promotion", "pre_exact", "exact_normal_removed",
    "final_hard", "contains_test_labels", "contains_evaluation_metrics",
    "provenance",
})
_PREDICTION_INDEX_FIELDS_ANTIGRAVITY = (
    "mini1024", "mini4096", "inherited_hard", "inherited_standard",
    "inherited_priority", "outside_standard", "promotion", "pre_exact",
    "exact_normal_removed", "final_hard",
)
_PREDICTION_FLAG_FIELDS_ANTIGRAVITY = frozenset({
    "contains_test_labels", "contains_evaluation_metrics",
})
_PROVENANCE_KEYS_ANTIGRAVITY = frozenset({"candidate_source", "routing_rule"})


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


def _freeze_prediction_value_antigravity(value: object) -> object:
    """Detach and recursively freeze deterministic prediction values."""

    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("prediction payload contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError("prediction mapping keys must be non-empty strings")
            copied[key] = _freeze_prediction_value_antigravity(item)
        return MappingProxyType(copied)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_prediction_value_antigravity(item) for item in value)
    raise ValueError("prediction payload is not deterministic JSON data")


def _normalize_prediction_row_antigravity(row: object) -> Mapping[str, object]:
    """Validate one public row and return a detached immutable copy."""

    if not isinstance(row, Mapping):
        raise ValueError("prediction rows are not mappings")
    raw = dict(row)
    # Keep the established recursive firewall authoritative for all ordinary
    # fields.  The two explicit safety flags are part of this public schema,
    # so they are checked separately below instead of weakening the helper.
    validate_prediction_payload_antigravity(
        {key: value for key, value in raw.items() if key not in _PREDICTION_FLAG_FIELDS_ANTIGRAVITY}
    )
    if set(raw) != set(_PREDICTION_ROW_KEYS_ANTIGRAVITY):
        raise ValueError("prediction row schema is not exact")
    dataset = raw["dataset"]
    if type(dataset) is not str or not dataset:
        raise ValueError("prediction row dataset is invalid")
    policy = raw["policy"]
    if type(policy) is not str or policy not in FROZEN_CONFIG.policies:
        raise ValueError("prediction row policy is invalid")
    if type(raw["n_test"]) is not int or raw["n_test"] < 1:
        raise ValueError("prediction row TEST count is invalid")
    n_test = raw["n_test"]
    normalized: dict[str, object] = {
        "dataset": dataset,
        "policy": policy,
        "n_test": n_test,
    }
    for field in _PREDICTION_INDEX_FIELDS_ANTIGRAVITY:
        normalized[field] = _indices(raw[field], dataset=dataset, n_test=n_test, field=field)
    for field in _PREDICTION_FLAG_FIELDS_ANTIGRAVITY:
        if type(raw[field]) is not bool or raw[field] is not False:
            raise ValueError(f"{field} must be the exact boolean false")
        normalized[field] = False
    provenance = raw["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != set(_PROVENANCE_KEYS_ANTIGRAVITY):
        raise ValueError("prediction provenance schema is not exact")
    if any(type(provenance[key]) is not str or not provenance[key] for key in _PROVENANCE_KEYS_ANTIGRAVITY):
        raise ValueError("prediction provenance values are invalid")
    normalized["provenance"] = dict(provenance)
    frozen = _freeze_prediction_value_antigravity(normalized)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by helper
        raise ValueError("prediction row could not be frozen")
    return frozen


def _validated_prediction_payload_antigravity(
    prediction: "PolicyPredictionAntigravity",
) -> dict[str, object]:
    """Revalidate every public field immediately before persistence."""

    if type(prediction) is not PolicyPredictionAntigravity:
        raise ValueError("prediction has the wrong contract type")
    try:
        policy = prediction.policy
        contains_test_labels = prediction.contains_test_labels
        contains_evaluation_metrics = prediction.contains_evaluation_metrics
        raw_rows = prediction.rows
    except AttributeError as exc:
        raise ValueError("prediction fields are incomplete") from exc
    if policy not in FROZEN_CONFIG.policies:
        raise ValueError("prediction policy is not preregistered")
    if type(contains_test_labels) is not bool or contains_test_labels is not False:
        raise ValueError("prediction contains_test_labels must be the exact boolean false")
    if type(contains_evaluation_metrics) is not bool or contains_evaluation_metrics is not False:
        raise ValueError("prediction contains_evaluation_metrics must be the exact boolean false")
    try:
        rows = tuple(_normalize_prediction_row_antigravity(row) for row in raw_rows)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("prediction rows are not iterable") from exc
    datasets = tuple(row["dataset"] for row in rows)
    if datasets != tuple(sorted(set(datasets))):
        raise ValueError("prediction row identity is not sorted and unique")
    coordinates: dict[str, frozenset[tuple[str, int]]] = {}
    for name in ("promotions", "pre_exact_coordinates", "exact_normal_removals", "final_hard_coordinates"):
        try:
            value = getattr(prediction, name)
        except AttributeError as exc:
            raise ValueError("prediction coordinates are incomplete") from exc
        coordinates[name] = _coordinates(value, field=name)
    return {
        "policy": policy,
        "rows": [
            {
                str(key): _json_safe_prediction_value_antigravity(value)
                for key, value in row.items()
            }
            for row in rows
        ],
        "promotions": [[dataset, index] for dataset, index in sorted(coordinates["promotions"])],
        "pre_exact": [[dataset, index] for dataset, index in sorted(coordinates["pre_exact_coordinates"])],
        "exact_normal_removed": [[dataset, index] for dataset, index in sorted(coordinates["exact_normal_removals"])],
        "final_hard": [[dataset, index] for dataset, index in sorted(coordinates["final_hard_coordinates"])],
    }


def _json_safe_prediction_value_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_safe_prediction_value_antigravity(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_prediction_value_antigravity(item) for item in value]
    return value


def _validate_recomputed_result_antigravity(
    result: object,
    *,
    dataset: str,
    policy: str,
    n_test: int,
) -> PolicyDatasetResultAntigravity:
    """Enforce the exact four-call scored-result boundary used by routing."""

    if type(result) is not PolicyDatasetResultAntigravity:
        raise ValueError(f"recomputed result identity for {dataset} is invalid")
    if result.dataset != dataset or result.policy != policy or result.status != "scored":
        raise ValueError(f"recomputed result status or identity for {dataset} is invalid")
    if type(result.test_rows) is not int or type(result.train_used) is not int or result.test_rows != n_test or result.train_used < 5:
        raise ValueError(f"recomputed result dimensions for {dataset} are invalid")
    if type(result.kernel_scores) is not tuple or len(result.kernel_scores) != 4:
        raise ValueError(f"recomputed result must contain exactly four kernel-seed scores for {dataset}")
    expected_pairs = {
        (kernel, seed)
        for kernel in FROZEN_CONFIG.kernel_requests
        for seed in FROZEN_CONFIG.seeds
    }
    observed_pairs: list[tuple[int, int]] = []
    for score in result.kernel_scores:
        if type(score) is not KernelSeedScoresAntigravity:
            raise ValueError(f"recomputed kernel-seed score for {dataset} is invalid")
        observed_pairs.append((score.kernel, score.seed))
        if len(score.oof_scores) != result.train_used or len(score.test_scores) != n_test:
            raise ValueError(f"recomputed score lengths for {dataset} are invalid")
        if score.oof_scores.dtype != np.dtype("<f8") or score.test_scores.dtype != np.dtype("<f8"):
            raise ValueError(f"recomputed score dtypes for {dataset} are invalid")
        if score.oof_scores.ndim != 1 or score.test_scores.ndim != 1:
            raise ValueError(f"recomputed score dimensions for {dataset} are invalid")
        if score.oof_scores.flags.writeable or score.test_scores.flags.writeable:
            raise ValueError(f"recomputed score arrays for {dataset} are mutable")
        if not np.isfinite(score.oof_scores).all() or not np.isfinite(score.test_scores).all():
            raise ValueError(f"recomputed score arrays for {dataset} are non-finite")
    if set(observed_pairs) != expected_pairs or len(set(observed_pairs)) != len(expected_pairs):
        raise ValueError(f"recomputed result kernel-seed identity for {dataset} is invalid")
    if not isinstance(result.candidates_by_kernel, Mapping) or set(result.candidates_by_kernel) != set(FROZEN_CONFIG.kernel_requests):
        raise ValueError(f"recomputed result candidate kernels for {dataset} are invalid")
    candidates_by_kernel: dict[int, tuple[int, ...]] = {}
    for kernel in FROZEN_CONFIG.kernel_requests:
        candidates_by_kernel[kernel] = _indices(
            result.candidates_by_kernel[kernel],
            dataset=dataset,
            n_test=n_test,
            field=f"MiniROCKET {kernel}",
        )
        scores = tuple(score for score in result.kernel_scores if score.kernel == kernel)
        if len(scores) != len(FROZEN_CONFIG.seeds):
            raise ValueError(f"recomputed result seed pair for {dataset} is incomplete")
        from .minirocket_antigravity import seed_intersection_candidates_antigravity

        expected_candidates = seed_intersection_candidates_antigravity(
            scores, alpha=FROZEN_CONFIG.alpha
        )
        if candidates_by_kernel[kernel] != expected_candidates:
            raise ValueError(f"recomputed candidates for {dataset} disagree with seed intersections")
    return result


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
        if type(self.contains_test_labels) is not bool or self.contains_test_labels is not False:
            raise ValueError("prediction contains_test_labels must be the exact boolean false")
        if type(self.contains_evaluation_metrics) is not bool or self.contains_evaluation_metrics is not False:
            raise ValueError("prediction contains_evaluation_metrics must be the exact boolean false")
        rows = tuple(self.rows)
        normalized_rows = tuple(_normalize_prediction_row_antigravity(row) for row in rows)
        datasets = tuple(row["dataset"] for row in normalized_rows)
        if any(type(dataset) is not str or not dataset for dataset in datasets) or datasets != tuple(sorted(set(datasets))):
            raise ValueError("prediction row identity is not sorted and unique")
        object.__setattr__(self, "rows", normalized_rows)
        for name in ("promotions", "pre_exact_coordinates", "exact_normal_removals", "final_hard_coordinates"):
            object.__setattr__(self, name, _coordinates(getattr(self, name), field=name))

    @property
    def dataset_rows(self) -> tuple[Mapping[str, object], ...]:
        return self.rows

    @property
    def promotion_coordinates(self) -> frozenset[tuple[str, int]]:
        return self.promotions

    def to_payload(self) -> dict[str, object]:
        """Return a freshly validated, canonical-safe payload."""

        return _validated_prediction_payload_antigravity(self)


def canonical_prediction_bytes_antigravity(
    prediction: PolicyPredictionAntigravity,
) -> bytes:
    # Do not rely solely on construction-time checks: this is the publication
    # boundary, so schema/firewall validation runs immediately before bytes.
    payload = _validated_prediction_payload_antigravity(prediction)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
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
        hard_set = set(inherited_hard)
        standard_set = set(standard)
        priority_set = set(priority)
        if hard_set & standard_set or hard_set & priority_set or standard_set & priority_set:
            raise ValueError(f"Hard, Standard, and Priority lanes overlap for {dataset}")

        if dataset in result_by_dataset:
            result = result_by_dataset[dataset]
            _validate_recomputed_result_antigravity(
                result, dataset=dataset, policy=policy, n_test=n_test
            )
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
