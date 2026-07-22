from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp226_235_hard_alert_quality.preprocessing import preprocessing_contract
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows

from .artifacts import load_existing_audits, load_new_source
from .config import (
    ALPHA, EXPECTED_DATASETS, EXPECTED_STANDARD, EXP292_PATH, EXP292_SHA256,
    EXP306_ROOT, EXPERIMENT_ID, OUTPUT_ROOT, WORKLOADS,
)
from .routing import (
    GATE_VARIANTS, build_candidate_records, order_eligible_experts,
    select_distinct_experts,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _load_baseline() -> list[dict[str, object]]:
    if sha256_file(EXP292_PATH) != EXP292_SHA256:
        raise ValueError("Exp292 prediction SHA256 mismatch")
    rows = _read_jsonl(EXP292_PATH)
    if len(rows) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in rows}) != EXPECTED_DATASETS:
        raise ValueError("Exp292 coverage mismatch")
    return rows


def _load_component_rows() -> tuple[dict[str, dict[str, object]], str]:
    path = EXP306_ROOT / "exp306_frozen_predictions.jsonl"
    manifest = EXP306_ROOT / "exp306_prediction_manifest.json"
    rows = verify_rows(path, manifest)
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError("Exp306 coverage mismatch")
    return {str(row["dataset"]): row for row in rows}, sha256_file(path)


def run_prediction(root: Path) -> dict[str, object]:
    baseline = _load_baseline()
    components, exp306_sha = _load_component_rows()
    audits = load_existing_audits()
    ordinal_audit, ordinal_candidates, ordinal_provenance = load_new_source("ordinal_pattern")
    masked_audit, masked_candidates, masked_provenance = load_new_source("masked_context")
    audit_rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for base in baseline:
        dataset = str(base["dataset"])
        dataset_audit = dict(audits[dataset])
        dataset_audit["ordinal_pattern"] = ordinal_audit[dataset]
        dataset_audit["masked_context"] = masked_audit[dataset]
        for expert, audit in dataset_audit.items():
            row = dict(audit); row["dataset"] = dataset; row["expert"] = expert
            row["experiment_id"] = EXPERIMENT_ID; audit_rows.append(row)
        standard = list(map(int, base["standard_review_indices"]))
        support: dict[int, set[str]] = {
            int(index): set(map(str, values))
            for index, values in base["review_support_by_index"].items()
        }
        for index in standard:
            support.setdefault(index, set())
            if index in ordinal_candidates[dataset]: support[index].add("ordinal_pattern")
            if index in masked_candidates[dataset]: support[index].add("masked_context")
        component = components[dataset]
        for variant in GATE_VARIANTS:
            ordered = order_eligible_experts(dataset_audit, variant=variant)
            selected = select_distinct_experts(dataset_audit, variant=variant)
            records = build_candidate_records(
                dataset=dataset, standard_indices=standard, support_by_index=support,
                ordered_experts=ordered, selected_experts=selected,
            )
            predictions.append({
                "experiment_id": EXPERIMENT_ID, "dataset": dataset,
                "n_test": int(base["n_test"]), "gate_variant": variant,
                "ordered_experts": ordered, "selected_experts": selected,
                "candidate_rank_records": records,
                "core_hard_indices": component["core_hard_indices"],
                "exp236_research_shadow_indices": component["exp236_research_shadow_indices"],
                "exp253_research_shadow_indices": component["exp253_research_shadow_indices"],
                "benchmark_union_hard_indices": component["benchmark_union_hard_indices"],
                "standard_review_indices": standard,
                "moe_shadow_2groups_indices": [int(row["candidate_index"]) for row in records if row["shadow_2groups"]],
                "moe_shadow_3groups_indices": [int(row["candidate_index"]) for row in records if row["shadow_3groups"]],
                "automatic_policy_changed": False, "review_candidate_set_changed": False,
                "moe_shadow_autonomous": False, "contains_test_values": False,
                "contains_test_labels": False, "contains_evaluation_metrics": False,
            })
    if len(audit_rows) != EXPECTED_DATASETS * 6 or len(predictions) != EXPECTED_DATASETS * len(GATE_VARIANTS):
        raise ValueError("Exp307 row coverage mismatch")
    for variant in GATE_VARIANTS:
        if sum(len(row["standard_review_indices"]) for row in predictions if row["gate_variant"] == variant) != EXPECTED_STANDARD:
            raise ValueError(f"Exp307 Standard candidate parity mismatch: {variant}")
    _write_csv(root / "exp307_expert_audit.csv", audit_rows)
    summary_rows = []
    for expert in sorted({str(row["expert"]) for row in audit_rows}):
        values = [row for row in audit_rows if row["expert"] == expert]
        available = [row for row in values if bool(row["available"])]
        summary_rows.append({
            "experiment_id": EXPERIMENT_ID, "expert": expert, "datasets": len(values),
            "available_datasets": len(available),
            "provenance_verified": all(bool(row["provenance_verified"]) for row in values),
            "stability_observable_datasets": sum(bool(row.get("stability_observable")) for row in values),
            "median_risk_upper_95": float(np.median([float(row["risk_upper_95"]) for row in available])) if available else 1.0,
            "median_stability": float(np.median([float(row["stability"]) for row in available if bool(row.get("stability_observable"))])) if any(bool(row.get("stability_observable")) for row in available) else 0.0,
            "contains_test_values": False, "contains_test_labels": False,
        })
    _write_csv(root / "exp307_expert_summary.csv", summary_rows)
    return freeze_rows(
        predictions, root / "exp307_frozen_predictions.jsonl", root / "exp307_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "alpha": ALPHA, "datasets": EXPECTED_DATASETS, "standard_candidates": EXPECTED_STANDARD,
            "exp292_sha256": EXP292_SHA256, "exp306_sha256": exp306_sha,
            "new_source_provenance": [ordinal_provenance, masked_provenance],
            "preprocessing_contract": preprocessing_contract(),
            "gate_variants": list(GATE_VARIANTS), "workloads": list(WORKLOADS),
            "primary_policy_selected": False, "automatic_policy_changed": False,
            "review_only": True, "retrospective": True, "prospective_validated": False,
        },
    )


def _metric(indices: list[tuple[str, int]], labels: dict[str, np.ndarray]) -> tuple[int, int]:
    tp = sum(int(labels[dataset][index] == 1) for dataset, index in indices)
    return tp, len(indices) - tp


def run_evaluation(root: Path) -> list[dict[str, object]]:
    path = root / "exp307_frozen_predictions.jsonl"
    predictions = verify_rows(path, root / "exp307_prediction_manifest.json")
    datasets = sorted({str(row["dataset"]) for row in predictions})
    labels = {dataset: np.asarray(load_test_labels_for_evaluation(dataset), dtype=np.int8) for dataset in datasets}
    truth = sum(int(np.count_nonzero(values == 1)) for values in labels.values())
    output: list[dict[str, object]] = []
    for variant in GATE_VARIANTS:
        rows = [row for row in predictions if row["gate_variant"] == variant]
        records = [dict(record) for row in rows for record in row["candidate_rank_records"]]
        records.sort(key=lambda row: (
            -int(row["selected_expert_support_count"]), -int(row["independent_group_support_count"]),
            -int(row["verified_expert_support_count"]), int(row["expert_rank_sum"]),
            str(row["dataset"]), int(row["candidate_index"]),
        ))
        for workload in WORKLOADS:
            selected = records[:workload]
            indices = [(str(row["dataset"]), int(row["candidate_index"])) for row in selected]
            tp, fp = _metric(indices, labels)
            output.append({
                "experiment_id": EXPERIMENT_ID, "gate_variant": variant,
                "evaluation_kind": "fixed_review_workload", "policy": f"top_{workload}",
                "datasets": EXPECTED_DATASETS, "count": len(indices), "tp": tp, "fp": fp,
                "precision": tp / len(indices) if indices else 0.0,
                "recall": tp / truth if truth else 0.0,
                "human_assisted_only": True, "autonomous": False,
                "automatic_policy_changed": False, "retrospective": True,
            })
        for policy, key in (("shadow_2groups", "moe_shadow_2groups_indices"), ("shadow_3groups", "moe_shadow_3groups_indices")):
            indices = [(str(row["dataset"]), int(index)) for row in rows for index in row[key]]
            tp, fp = _metric(indices, labels)
            output.append({
                "experiment_id": EXPERIMENT_ID, "gate_variant": variant,
                "evaluation_kind": "research_shadow", "policy": policy,
                "datasets": EXPECTED_DATASETS, "count": len(indices), "tp": tp, "fp": fp,
                "precision": tp / len(indices) if indices else 0.0,
                "recall": tp / truth if truth else 0.0,
                "human_assisted_only": True, "autonomous": False,
                "automatic_policy_changed": False, "retrospective": True,
            })
    _write_csv(root / "exp307_results.csv", output)
    _write_csv(root / "exp307_summary.csv", output)
    report = [
        "# Exp307 Expanded Six-expert MoE", "",
        "The 806 Standard candidates are unchanged. All metrics are retrospective review/Shadow diagnostics.", "",
        "| Gate | Policy | Count | TP | FP | Precision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in output:
        report.append(f"| {row['gate_variant']} | {row['policy']} | {row['count']} | {row['tp']} | {row['fp']} | {100*float(row['precision']):.3f}% |")
    report.extend(["", "- Ordinal pattern and masked context are added as methodologically distinct experts.", "- Single-variant stability is explicitly marked unobservable.", "- No Gate or Shadow policy is selected or promoted to autonomous Hard.", ""])
    (root / "RESULT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (root / "exp307_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "prediction_sha256": sha256_file(path),
        "labels_loaded_after_prediction_freeze": True, "primary_policy_selected": False,
        "automatic_policy_changed": False, "retrospective": True, "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors=None) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None, errors=errors or [], timestamps=[], stdout_log=log, started_unix=started)
    payload.update({"autonomous_metric_scope": "Exp306 core and benchmark Hard unchanged", "review_metric_scope": "Six-expert MoE review ranking and research Shadow", "retrospective": True, "prospective": False})
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all"); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args()
    started = time.time(); _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"): run_prediction(args.output_root)
        if args.phase in ("evaluate", "all"): run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"]); raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__": main()
