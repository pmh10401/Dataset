from __future__ import annotations

from collections.abc import Mapping


def _indices(row: Mapping[str, object], key: str) -> set[int]:
    values = row.get(key, ())
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError(f"invalid index field: {key}")
    return set(map(int, values))


def build_componentized_row(
    *, dataset: str, exp236: Mapping[str, object], exp253: Mapping[str, object],
    exp290: Mapping[str, object], exp291: Mapping[str, object],
) -> dict[str, object]:
    exp236_added = _indices(exp236, "new_hard_indices")
    exp236_hard = _indices(exp236, "hard_indices")
    exp137_core = exp236_hard - exp236_added
    exp253_added = _indices(exp253, "promoted_indices")
    exact_normal = _indices(exp290, "train_raw_exact_normal_indices")
    core = exp137_core - exact_normal
    shadow236 = exp236_added - exact_normal
    shadow253 = exp253_added - exact_normal
    benchmark = core | shadow236 | shadow253
    expected = _indices(exp291, "automatic_hard_indices")
    if benchmark != expected or benchmark != _indices(exp290, "guarded_hard_indices"):
        raise ValueError("componentized benchmark union does not match Exp291")
    if core & shadow236 or core & shadow253 or shadow236 & shadow253:
        raise ValueError("componentized lanes overlap")
    n_test = int(exp291["n_test"])
    all_indices = benchmark | exact_normal
    if any(index < 0 or index >= n_test for index in all_indices):
        raise ValueError("component index outside TEST range")
    return {
        "experiment_id": "experiment_306_componentized_operational_routing",
        "dataset": str(dataset),
        "n_test": n_test,
        "core_hard_indices": sorted(core),
        "exp236_research_shadow_indices": sorted(shadow236),
        "exp253_research_shadow_indices": sorted(shadow253),
        "benchmark_union_hard_indices": sorted(benchmark),
        "automatic_normal_exact_indices": sorted(exact_normal),
        "standard_review_indices": sorted(_indices(exp291, "standard_review_indices")),
        "priority_review_indices": sorted(_indices(exp291, "priority_review_indices")),
        "shadow_review_indices": sorted(_indices(exp291, "shadow_review_indices")),
        "recovery_shadow_review_indices": sorted(_indices(exp291, "recovery_shadow_review_indices")),
        "research_only_recovery_priority_indices": sorted(_indices(exp291, "research_only_recovery_priority_indices")),
        "candidate_operational_hard_indices": sorted(core),
        "operational_policy_updated": False,
        "research_extensions_autonomous": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
