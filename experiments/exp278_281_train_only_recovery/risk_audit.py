from __future__ import annotations

import json
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import verify_recovery_prediction_hash
from experiments.exp276_277_shadow_confidence.freeze import _cache_index, _risk_index, _source_evidence

from .contracts import (
    CONTINUOUS_GROUPS, EXP272_ROOT, RISK_TARGETS, candidate_rows, sha256_file, write_csv,
)


def run_risk_resolution_audit(output_root: Path) -> tuple[Path, Path]:
    source_rows, datasets = candidate_rows()
    verify_recovery_prediction_hash(
        EXP272_ROOT / "exp272_frozen_predictions.jsonl",
        EXP272_ROOT / "exp272_prediction_manifest.json",
        expected_datasets=datasets,
    )
    caches, risks = _cache_index(), _risk_index()
    detail: list[dict[str, object]] = []
    for row in source_rows:
        support_by_index = row["support_by_index"]
        for lane in ("recovery_shadow_indices", "recovery_priority_indices"):
            for index in row[lane]:
                support = set(support_by_index[str(index)])
                evidence = _source_evidence(str(row["dataset"]), index, support, caches, risks)
                max_risk = max((float(item["train_risk_upper_95"]) for item in evidence), default=1.0)
                references = [int(item["reference_count"]) for item in evidence]
                detail.append({
                    "dataset": row["dataset"], "candidate_index": index,
                    "source_lane": lane.removesuffix("_indices"),
                    "continuous_group_count": len(evidence),
                    "continuous_groups": json.dumps([item["group"] for item in evidence], separators=(",", ":")),
                    "reference_counts": json.dumps(references, separators=(",", ":")),
                    "minimum_attainable_p": min((1.0 / (value + 1.0) for value in references), default=1.0),
                    "maximum_train_risk_upper_95": max_risk,
                    **{f"risk_target_{target:g}_expressible": bool(evidence) and max_risk <= target for target in RISK_TARGETS},
                    "eligible_two_continuous_groups": len(support & CONTINUOUS_GROUPS) >= 2,
                    "contains_test_labels": False, "contains_evaluation_metrics": False,
                })
    if len(detail) != 179:
        raise ValueError(f"Exp278 candidate coverage mismatch: {len(detail)}")
    summary = [{
        "experiment_id": "experiment_278_train_risk_resolution_audit",
        "datasets": len(datasets), "candidate_rows": len(detail),
        "two_continuous_group_candidates": sum(bool(row["eligible_two_continuous_groups"]) for row in detail),
        **{f"risk_target_{target:g}_candidate_count": sum(bool(row[f"risk_target_{target:g}_expressible"]) for row in detail) for target in RISK_TARGETS},
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "retrospective": True,
    }]
    root = Path(output_root); detail_path = root / "exp278_risk_resolution_dataset.csv"; summary_path = root / "exp278_summary.csv"
    write_csv(detail_path, detail); write_csv(summary_path, summary)
    (root / "exp278_manifest.json").write_text(json.dumps({
        "datasets": len(datasets), "rows": len(detail),
        "input_sha256": {
            "exp272_prediction": sha256_file(EXP272_ROOT / "exp272_frozen_predictions.jsonl"),
            "train_risk": sha256_file(EXP272_ROOT / "exp270_train_risk_dataset.csv"),
        },
        "contains_test_labels": False, "contains_evaluation_metrics": False,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path


__all__ = ["run_risk_resolution_audit"]
