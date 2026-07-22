from __future__ import annotations

from itertools import combinations


def _indices(row: dict[str, object], field: str) -> set[int]:
    values = {int(index) for index in row[field]}  # type: ignore[arg-type]
    n_test = int(row["n_test"])
    if any(index < 0 or index >= n_test for index in values):
        raise ValueError(f"index out of range: field={field}")
    return values


def route_dataset(
    exp290: dict[str, object], exp272: dict[str, object]
) -> dict[str, object]:
    dataset = str(exp290["dataset"])
    if dataset != str(exp272["dataset"]) or int(exp290["n_test"]) != int(exp272["n_test"]):
        raise ValueError("dataset identity mismatch")

    baseline_hard = _indices(exp290, "baseline_hard_indices")
    if baseline_hard != _indices(exp272, "hard_indices"):
        raise ValueError(f"Exp253 Hard parity mismatch: dataset={dataset}")
    automatic_hard = _indices(exp290, "guarded_hard_indices")
    automatic_normal = _indices(exp290, "train_raw_exact_normal_indices")
    if automatic_hard & automatic_normal or automatic_hard | automatic_normal != baseline_hard:
        raise ValueError(f"exact normal partition mismatch: dataset={dataset}")

    lanes = {
        "automatic_hard_indices": automatic_hard,
        "automatic_normal_exact_indices": automatic_normal,
        "standard_review_indices": _indices(exp272, "standard_indices"),
        "priority_review_indices": _indices(exp272, "priority_indices"),
        "shadow_review_indices": _indices(exp272, "shadow_indices"),
        "recovery_shadow_review_indices": _indices(exp272, "recovery_shadow_indices"),
        "research_only_recovery_priority_indices": _indices(exp272, "recovery_priority_indices"),
    }
    for (left_name, left), (right_name, right) in combinations(lanes.items(), 2):
        overlap = left & right
        if overlap:
            raise ValueError(
                f"final lane overlap: dataset={dataset} lanes={left_name},{right_name} indices={sorted(overlap)}"
            )

    default_review = (
        lanes["standard_review_indices"]
        | lanes["priority_review_indices"]
        | lanes["shadow_review_indices"]
        | lanes["recovery_shadow_review_indices"]
    )
    output: dict[str, object] = {
        "experiment_id": "experiment_291_final_operational_routing_freeze",
        "dataset": dataset,
        "n_test": int(exp290["n_test"]),
        "source_baseline_hard_indices": sorted(baseline_hard),
        **{name: sorted(values) for name, values in lanes.items()},
        "default_review_indices": sorted(default_review),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_normal_rule": "sqlite_values_blob_byte_exact",
        "recovery_priority_scope": "research_only",
        "retrospective_integration": True,
        "prospective_validated": False,
    }
    return output


__all__ = ["route_dataset"]
