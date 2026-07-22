from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .artifacts import load_primary_train_scores, load_reconstruction_train_scores
from .config import (
    ALPHA, EXPECTED_DATASETS, EXP292_PATH, EXP292_SHA256, EXP293_ID, EXP293_ROOT,
    EXP294_ID, EXP294_ROOT, WORKLOADS,
)
from .contracts import freeze_rows, sha256_file, verify_rows
from .core import audit_expert
from .routing import GATE_VARIANTS, candidate_rank_records, gate_experts


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _exp292_rows() -> list[dict[str, object]]:
    if sha256_file(EXP292_PATH) != EXP292_SHA256:
        raise ValueError("Exp292 frozen prediction hash mismatch")
    rows = _load_jsonl(EXP292_PATH)
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError("Exp292 coverage mismatch")
    return rows


def run_exp293(root: Path) -> dict[str, object]:
    baseline = _exp292_rows()
    datasets = sorted(str(row["dataset"]) for row in baseline)
    primary, source_manifests = load_primary_train_scores()
    reconstruction, reconstruction_manifest = load_reconstruction_train_scores(datasets)
    audit_rows: list[dict[str, object]] = []
    choices: list[dict[str, object]] = []
    by_dataset: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    expert_names = [*sorted(primary), "masked_reconstruction"]
    for dataset in datasets:
        for expert in expert_names:
            values = reconstruction.get(dataset, {}) if expert == "masked_reconstruction" else primary[expert].get(dataset, {})
            row = audit_expert(
                dataset=dataset, expert=expert, train_variants=values,
                provenance_verified=expert != "masked_reconstruction", alpha=ALPHA,
            )
            row["experiment_id"] = EXP293_ID
            row["policy_scope"] = "research_sensitivity_only" if expert == "masked_reconstruction" else "primary_gate_eligible"
            audit_rows.append(row)
            by_dataset[dataset][expert] = row
        for variant in GATE_VARIANTS:
            ordered = gate_experts(by_dataset[dataset], variant=variant)
            choices.append({
                "experiment_id": EXP293_ID, "dataset": dataset, "gate_variant": variant,
                "ordered_experts": ordered, "selected_experts": ordered[:2],
                "contains_test_values": False, "contains_test_labels": False,
                "contains_evaluation_metrics": False, "family_policy_used": False,
                "test_length_policy_used": False,
            })
    if len(audit_rows) != EXPECTED_DATASETS * 5 or len(choices) != EXPECTED_DATASETS * len(GATE_VARIANTS):
        raise ValueError("Exp293 row coverage mismatch")
    _write_csv(root / "exp293_expert_audit.csv", audit_rows)
    summary_rows = []
    for expert in expert_names:
        values = [row for row in audit_rows if row["expert"] == expert]
        available = [row for row in values if row["available"]]
        summary_rows.append({
            "experiment_id": EXP293_ID, "expert": expert, "datasets": len(values),
            "available_datasets": len(available), "provenance_verified": expert != "masked_reconstruction",
            "median_n_train": float(np.median([int(row["n_train"]) for row in available])) if available else 0.0,
            "median_risk_upper_95": float(np.median([float(row["risk_upper_95"]) for row in available])) if available else 1.0,
            "median_stability": float(np.median([float(row["stability"]) for row in available])) if available else 0.0,
            "normal_false_count": sum(int(row["normal_false_count"]) for row in available),
            "contains_test_values": False, "contains_test_labels": False,
        })
    _write_csv(root / "exp293_summary.csv", summary_rows)
    manifest = freeze_rows(
        choices, root / "exp293_frozen_gate_choices.jsonl", root / "exp293_gate_manifest.json",
        experiment_id=EXP293_ID,
        metadata={"alpha": ALPHA, "source_manifests": source_manifests + [reconstruction_manifest], "exp292_sha256": EXP292_SHA256, "primary_policy_selected": False},
    )
    return {"audit_rows": len(audit_rows), "gate_rows": len(choices), "prediction_sha256": manifest["prediction_sha256"]}


def run_exp294_prediction(root: Path, exp293_root: Path = EXP293_ROOT) -> dict[str, object]:
    gate_rows = verify_rows(exp293_root / "exp293_frozen_gate_choices.jsonl", exp293_root / "exp293_gate_manifest.json")
    gate_map = {(str(row["dataset"]), str(row["gate_variant"])): row for row in gate_rows}
    predictions: list[dict[str, object]] = []
    for baseline in _exp292_rows():
        dataset = str(baseline["dataset"])
        support = baseline["review_support_by_index"]
        for variant in GATE_VARIANTS:
            gate = gate_map[(dataset, variant)]
            ordered = list(map(str, gate["ordered_experts"]))
            records = candidate_rank_records(
                dataset=dataset,
                standard_indices=baseline["standard_review_indices"],
                support_by_index=support,
                ordered_experts=ordered,
            )
            predictions.append({
                "experiment_id": EXP294_ID, "dataset": dataset, "n_test": int(baseline["n_test"]),
                "gate_variant": variant, "ordered_experts": ordered, "selected_experts": ordered[:2],
                "automatic_hard_indices": baseline["automatic_hard_indices"],
                "standard_review_indices": baseline["standard_review_indices"],
                "candidate_rank_records": records,
                "automatic_policy_changed": False, "review_candidate_set_changed": False,
                "contains_test_values": False, "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            })
    if len(predictions) != EXPECTED_DATASETS * len(GATE_VARIANTS):
        raise ValueError("Exp294 prediction coverage mismatch")
    for variant in GATE_VARIANTS:
        rows = [row for row in predictions if row["gate_variant"] == variant]
        if sum(len(row["standard_review_indices"]) for row in rows) != 806:
            raise ValueError(f"Exp294 candidate parity mismatch: {variant}")
    return freeze_rows(
        predictions, root / "exp294_frozen_predictions.jsonl", root / "exp294_prediction_manifest.json",
        experiment_id=EXP294_ID,
        metadata={"exp292_sha256": EXP292_SHA256, "exp293_gate_sha256": sha256_file(exp293_root / "exp293_frozen_gate_choices.jsonl"), "workloads": list(WORKLOADS), "automatic_policy_changed": False, "review_only": True},
    )


def _binary_metrics(tp: int, fp: int, truth: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": truth - tp, "precision": precision, "recall": recall, "micro_f1": f1}


def run_exp294_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp294_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp294_prediction_manifest.json")
    datasets = sorted({str(row["dataset"]) for row in predictions})
    labels = {dataset: np.asarray(load_test_labels_for_evaluation(dataset), dtype=np.int8) for dataset in datasets}
    output: list[dict[str, object]] = []
    for variant in GATE_VARIANTS:
        variant_rows = [row for row in predictions if row["gate_variant"] == variant]
        for scope in ("all_1117", "n_test_ge_100_diagnostic"):
            scoped = [row for row in variant_rows if scope == "all_1117" or int(row["n_test"]) >= 100]
            records = [dict(record) for row in scoped for record in row["candidate_rank_records"]]
            records.sort(key=lambda row: (
                -int(row["selected_expert_support_count"]), -int(row["verified_expert_support_count"]),
                int(row["expert_rank_sum"]), str(row["dataset"]), int(row["candidate_index"]),
            ))
            truth = sum(int(np.count_nonzero(labels[str(row["dataset"])] == 1)) for row in scoped)
            hard_tp = hard_fp = 0
            for row in scoped:
                y = labels[str(row["dataset"])]
                for index in row["automatic_hard_indices"]:
                    hard_tp += int(y[int(index)] == 1)
                    hard_fp += int(y[int(index)] != 1)
            for workload in WORKLOADS:
                selected = records[: min(workload, len(records))]
                review_tp = sum(int(labels[str(item["dataset"])][int(item["candidate_index"])] == 1) for item in selected)
                review_fp = len(selected) - review_tp
                review = _binary_metrics(review_tp, review_fp, truth)
                combined = _binary_metrics(hard_tp + review_tp, hard_fp + review_fp, truth)
                output.append({
                    "experiment_id": EXP294_ID, "gate_variant": variant, "scope": scope,
                    "datasets": len(scoped), "workload": workload, "available_candidates": len(records),
                    "review_count": len(selected),
                    **{f"review_{key}": value for key, value in review.items()},
                    "hard_count": hard_tp + hard_fp, "hard_tp": hard_tp, "hard_fp": hard_fp,
                    **{f"combined_{key}": value for key, value in combined.items()},
                    "review_metrics_human_assisted_only": True, "automatic_policy_changed": False,
                    "retrospective": True, "prospective_validated": False,
                })
    _write_csv(root / "exp294_results.csv", output)
    _write_csv(root / "exp294_summary.csv", output)
    (root / "exp294_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXP294_ID, "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True, "primary_policy_selected": False,
        "retrospective": True, "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _publish(root: Path, experiment_id: str, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(experiment_id, status, phase, done, EXPECTED_DATASETS, None, errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{experiment_id}.log", started_unix=started)
    payload.update({"autonomous_metric_scope": "Exp291 Hard unchanged", "review_metric_scope": "TRAIN-only sparse MoE review ranking", "retrospective": True, "prospective": False})
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("293", "294-predict", "294-evaluate", "294-all"), required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    is_293 = args.stage == "293"
    root = args.output_root or (EXP293_ROOT if is_293 else EXP294_ROOT)
    experiment_id = EXP293_ID if is_293 else EXP294_ID
    started = time.time()
    _publish(root, experiment_id, "running", args.stage, 0, started)
    try:
        if is_293:
            run_exp293(root)
        else:
            if args.stage in ("294-predict", "294-all"):
                run_exp294_prediction(root)
            if args.stage in ("294-evaluate", "294-all"):
                run_exp294_evaluation(root)
    except Exception as exc:
        _publish(root, experiment_id, "failed", args.stage, 0, started, [f"{type(exc).__name__}: {exc}"])
        raise
    _publish(root, experiment_id, "completed", args.stage, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()

