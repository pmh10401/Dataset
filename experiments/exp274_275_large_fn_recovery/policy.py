from __future__ import annotations

from collections.abc import Mapping

from .config import FORECAST_GROUP, MIN_TRAIN_FOR_EXPANSION


def _indices(row: Mapping[str, object], key: str) -> set[int]:
    values = row.get(key)
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError(f"invalid index list: {key}")
    return set(values)


def build_prediction_row(
    exp272_row: Mapping[str, object],
    *,
    wavelet: set[int],
    matrix_profile: set[int],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Build four pre-registered review-only views without TEST labels."""
    dataset = str(exp272_row["dataset"])
    n_train, n_test = int(exp272_row["n_train"]), int(exp272_row["n_test"])
    hard = _indices(exp272_row, "hard_indices")
    existing_review = set().union(*(
        _indices(exp272_row, key) for key in (
            "shadow_indices", "priority_indices", "standard_indices",
            "recovery_shadow_indices", "recovery_priority_indices",
            "data_quality_review_indices",
        )
    ))
    occupied = hard | existing_review
    one_source = _indices(exp272_row, "one_source_sensitivity_indices") - occupied
    recovery_shadow = _indices(exp272_row, "recovery_shadow_indices")
    support = exp272_row.get("support_by_index")
    if not isinstance(support, Mapping):
        raise ValueError("support_by_index must be a mapping")
    if any(index < 0 or index >= n_test for index in wavelet | matrix_profile | occupied | one_source):
        raise ValueError(f"candidate index outside TEST range: dataset={dataset}")

    eligible = n_train >= MIN_TRAIN_FOR_EXPANSION
    both_structure = wavelet & matrix_profile
    any_source = one_source & both_structure if eligible else set()
    forecast = {
        index for index in any_source
        if FORECAST_GROUP in set(str(value) for value in support.get(str(index), []))
    }
    shadow_either = recovery_shadow & (wavelet | matrix_profile)
    shadow_both = recovery_shadow & both_structure

    return {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "hard_indices": sorted(hard),
        "current_recovery_shadow_indices": sorted(recovery_shadow),
        "current_recovery_priority_indices": sorted(_indices(exp272_row, "recovery_priority_indices")),
        "forecast_wavelet_matrix_indices": sorted(forecast),
        "any_source_wavelet_matrix_indices": sorted(any_source),
        "shadow_either_structure_indices": sorted(shadow_either),
        "shadow_both_structures_indices": sorted(shadow_both),
        "large_train_eligible": eligible,
        "large_train_abstained": not eligible,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "hard_policy_changed": False,
        "review_only": True,
        "retrospective": True,
        "input_sha256": dict(input_sha256),
    }


VARIANT_FIELDS = {
    "forecast_wavelet_matrix": "forecast_wavelet_matrix_indices",
    "any_source_wavelet_matrix": "any_source_wavelet_matrix_indices",
    "shadow_either_structure": "shadow_either_structure_indices",
    "shadow_both_structures": "shadow_both_structures_indices",
}

