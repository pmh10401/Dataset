from __future__ import annotations

from typing import Any

from .config import MIN_TRAIN_FOR_SHADOW, PRIORITY_AGREEMENT, SHADOW_AGREEMENT, SOURCE_NAMES
from .contracts import ShadowRunInput, prediction_sha256


def _support(inputs: ShadowRunInput, index: int) -> tuple[str, ...]:
    return tuple(name for name in SOURCE_NAMES if index in inputs.sources[name].candidate_indices)


def route_shadow_run(inputs: ShadowRunInput) -> dict[str, Any]:
    all_baseline = inputs.baseline_hard | inputs.baseline_standard | inputs.baseline_priority
    support = {index: _support(inputs, index) for index in all_baseline}
    shadow = {
        index
        for index in inputs.baseline_standard
        if inputs.n_train >= MIN_TRAIN_FOR_SHADOW
        and len(support[index]) >= SHADOW_AGREEMENT
        and index not in inputs.quality_flags
    }
    priority_added = {
        index
        for index in inputs.baseline_standard
        if len(support[index]) >= PRIORITY_AGREEMENT and index not in shadow
    }
    priority = inputs.baseline_priority | priority_added
    standard = inputs.baseline_standard - shadow - priority_added
    hard = set(inputs.baseline_hard)
    confidence = {
        str(index): "A" if len(support[index]) >= 3 else "B" if support[index] else "C"
        for index in sorted(hard)
    }
    lane_by_index = {
        **{index: "Hard" for index in hard},
        **{index: "Shadow" for index in shadow},
        **{index: "Priority" for index in priority},
        **{index: "Standard" for index in standard},
    }
    candidates: list[dict[str, Any]] = []
    for index in sorted(lane_by_index):
        evidence = {
            name: {
                "agrees": index in inputs.sources[name].candidate_indices,
                "score": inputs.sources[name].scores.get(index),
                "threshold": inputs.sources[name].threshold,
            }
            for name in SOURCE_NAMES
        }
        candidates.append(
            {
                "candidate_id": f"{inputs.run_id}:{index}",
                "candidate_index": index,
                "final_lane": lane_by_index[index],
                "confidence_grade": confidence.get(str(index)),
                "support_count": len(support[index]),
                "support_sources": list(support[index]),
                "quality_flagged": index in inputs.quality_flags,
                "source_evidence": evidence,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": inputs.schema_version,
        "collection_mode": inputs.collection_mode,
        "policy_version": inputs.policy_version,
        "run_id": inputs.run_id,
        "source_id": inputs.source_id,
        "observed_at": inputs.observed_at,
        "dataset_key": inputs.dataset_key,
        "equipment_id": inputs.equipment_id,
        "recipe_id": inputs.recipe_id,
        "sensor_id": inputs.sensor_id,
        "step_id": inputs.step_id,
        "n_train": inputs.n_train,
        "n_test": inputs.n_test,
        "has_missing": inputs.has_missing,
        "missing_quality_status": inputs.missing_quality_status,
        "config_sha256": inputs.config_sha256,
        "series_preview": {"indices": list(inputs.series_preview.indices), "values": list(inputs.series_preview.values)},
        "baseline_hard_indices": sorted(inputs.baseline_hard),
        "baseline_standard_indices": sorted(inputs.baseline_standard),
        "baseline_priority_indices": sorted(inputs.baseline_priority),
        "hard_indices": sorted(hard),
        "shadow_indices": sorted(shadow),
        "priority_indices": sorted(priority),
        "standard_indices": sorted(standard),
        "confidence": confidence,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "hard_count": len(hard),
        "shadow_count": len(shadow),
        "priority_count": len(priority),
        "standard_count": len(standard),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
    }
    payload["prediction_sha256"] = prediction_sha256(payload)
    return payload
