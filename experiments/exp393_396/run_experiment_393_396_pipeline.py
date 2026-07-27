from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp393_396.artifact_io import (
    DATA_ROOT,
    PREDICTION_MANIFEST_NAME,
    PREDICTION_NAME,
    load_baseline_lanes,
    load_test_labels,
    load_verified_prediction,
    load_prediction_inputs,
    write_prediction_artifact,
)
from experiments.exp393_396.contracts import GrammarConfig
from experiments.exp393_396.grammar_core import (
    candidate_indices,
    conformal_p_value,
    crossfit_train_scores,
    fit_grammar_model,
    score_trace,
)
from experiments.exp393_396.preprocessing import (
    prepare_dataset,
    segment_descriptors,
)
from experiments.exp393_396.reporting import (
    aggregate_calibration_rows,
    audit_prediction_row,
    render_calibration_report,
    write_csv,
)


DEFAULT_OUTPUT_ROOT = DATA_ROOT / "exp393_396_normal_process_grammar"
SCORE_NAMES = ("transition", "dwell", "recovery", "composite")


def _anomaly_percentile(reference: np.ndarray, value: float) -> float:
    return float((1 + np.count_nonzero(reference <= value)) / (reference.size + 1))


def _empty_prediction(
    n_train: int,
    n_test: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "n_train_normal": n_train,
        "n_test": n_test,
        "target_length": 0,
        "effective_state_count": 0,
        "scores": {},
        "p_values": {},
        "candidate_indices_by_source_alpha": {},
        "candidate_indices_by_alpha": {},
        "train_oof_scores": {},
        "abstained": True,
        "abstention_reason": reason,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_budget": False,
        "test_metrics_select_policy": False,
        "eligible_for_deployment": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }


def predict_dataset(
    x_train_normal: Sequence[np.ndarray],
    x_test: Sequence[np.ndarray],
    frozen_config: GrammarConfig,
) -> dict[str, Any]:
    if len(x_train_normal) < 5:
        return _empty_prediction(len(x_train_normal), len(x_test), "n_train_lt_5")
    started = time.monotonic()
    try:
        prepared = prepare_dataset(x_train_normal, x_test)
        train_descriptors = [
            segment_descriptors(trace, prepared.target_length)
            for trace in prepared.train
        ]
        test_descriptors = [
            segment_descriptors(trace, prepared.target_length)
            for trace in prepared.test
        ]
        train_oof = crossfit_train_scores(train_descriptors, frozen_config)
        model = fit_grammar_model(train_descriptors, frozen_config)
        test_components = [score_trace(model, descriptors) for descriptors in test_descriptors]
        raw_scores = {
            "transition": np.asarray(
                [score.transition for score in test_components], dtype=float
            ),
            "dwell": np.asarray(
                [score.dwell for score in test_components], dtype=float
            ),
            "recovery": np.asarray(
                [score.recovery for score in test_components], dtype=float
            ),
        }
        component_percentiles = np.column_stack(
            [
                np.asarray(
                    [
                        _anomaly_percentile(train_oof[name], value)
                        for value in raw_scores[name]
                    ],
                    dtype=float,
                )
                for name in ("transition", "dwell", "recovery")
            ]
        ) if test_descriptors else np.empty((0, 3), dtype=float)
        raw_scores["composite"] = (
            np.mean(component_percentiles, axis=1)
            if component_percentiles.size
            else np.empty(0, dtype=float)
        )
        p_values = {
            name: np.asarray(
                [conformal_p_value(train_oof[name], value) for value in raw_scores[name]],
                dtype=float,
            )
            for name in SCORE_NAMES
        }
        source_candidates: dict[str, dict[str, list[int]]] = {}
        composite_candidates: dict[str, list[int]] = {}
        for alpha in frozen_config.alphas:
            alpha_key = f"{alpha:.3f}"
            source_candidates[alpha_key] = {
                name: candidate_indices(p_values[name], alpha) for name in SCORE_NAMES
            }
            composite_candidates[alpha_key] = source_candidates[alpha_key]["composite"]
        return {
            "n_train_normal": len(x_train_normal),
            "n_test": len(x_test),
            "target_length": prepared.target_length,
            "effective_state_count": int(model.clusterer.n_clusters),
            "scores": {name: raw_scores[name].tolist() for name in SCORE_NAMES},
            "p_values": {name: p_values[name].tolist() for name in SCORE_NAMES},
            "candidate_indices_by_source_alpha": source_candidates,
            "candidate_indices_by_alpha": composite_candidates,
            "train_oof_scores": {
                name: train_oof[name].tolist() for name in SCORE_NAMES
            },
            "abstained": False,
            "abstention_reason": "",
            "runtime_seconds": time.monotonic() - started,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "uses_family_name": False,
            "uses_dataset_name_for_policy": False,
            "uses_test_length_for_budget": False,
            "test_metrics_select_policy": False,
            "eligible_for_deployment": False,
            "retrospective_counterfactual": True,
            "prospective_validated": False,
        }
    except ValueError as error:
        row = _empty_prediction(len(x_train_normal), len(x_test), str(error))
        row["runtime_seconds"] = time.monotonic() - started
        return row


def _write_status(
    output_root: Path,
    *,
    phase: str,
    completed: int,
    total: int,
    current_dataset: str,
    status: str,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": "experiment_393_396_normal_process_grammar",
        "phase": phase,
        "completed": completed,
        "total": total,
        "current_dataset": current_dataset,
        "status": status,
        "progress": (completed / total) if total else 0.0,
        "updated_at_epoch": time.time(),
    }
    temporary = output_root / "status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_root / "status.json")


def run_predict(
    output_root: Path,
    dataset_limit: int | None = None,
) -> dict[str, Any]:
    config = GrammarConfig()
    inputs = load_prediction_inputs(dataset_limit)
    rows: list[dict[str, Any]] = []
    _write_status(
        output_root,
        phase="predict",
        completed=0,
        total=len(inputs),
        current_dataset="",
        status="running",
    )
    for index, item in enumerate(inputs, start=1):
        row = predict_dataset(item["train"], item["test"], config)
        row["dataset"] = item["dataset"]
        rows.append(row)
        _write_status(
            output_root,
            phase="predict",
            completed=index,
            total=len(inputs),
            current_dataset=item["dataset"],
            status="running",
        )
    prediction_path, manifest_path = write_prediction_artifact(
        output_root,
        rows,
        config,
    )
    _write_status(
        output_root,
        phase="predict",
        completed=len(inputs),
        total=len(inputs),
        current_dataset="",
        status="completed",
    )
    return {
        "prediction_path": str(prediction_path),
        "manifest_path": str(manifest_path),
        "datasets": len(rows),
        "abstained": sum(bool(row["abstained"]) for row in rows),
        "config": asdict(config),
    }


def run_audit(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    prediction_path = output_root / PREDICTION_NAME
    manifest_path = output_root / PREDICTION_MANIFEST_NAME
    rows, manifest = load_verified_prediction(prediction_path, manifest_path)
    config = GrammarConfig()
    if manifest.get("config") != asdict(config):
        raise ValueError("prediction config does not match frozen GrammarConfig")
    audit_rows = [audit_prediction_row(row, config) for row in rows]
    summary_rows = aggregate_calibration_rows(audit_rows, config)
    write_csv(output_root / "exp394_calibration_dataset.csv", audit_rows)
    write_csv(output_root / "exp394_alpha_summary.csv", summary_rows)
    correlation_rows = [
        {
            "existing_source": "all",
            "status": "unavailable",
            "reason": "frozen existing evidence artifacts contain indices, not raw TRAIN OOF vectors",
            "contains_test_labels": False,
        }
    ]
    write_csv(output_root / "exp394_existing_source_correlation.csv", correlation_rows)
    (output_root / "exp394_calibration_report.md").write_text(
        render_calibration_report(audit_rows, summary_rows),
        encoding="utf-8",
    )
    _write_status(
        output_root,
        phase="audit",
        completed=len(rows),
        total=len(rows),
        current_dataset="",
        status="completed",
    )
    return {
        "datasets": len(rows),
        "abstained": sum(bool(row.get("abstained")) for row in audit_rows),
        "summary_rows": len(summary_rows),
        "contains_test_labels": False,
    }


def evaluate_candidate_set(
    *,
    candidates: set[int],
    hard: set[int],
    standard: set[int],
    priority: set[int],
    labels: Sequence[str],
    n_test: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(labels) != n_test:
        raise ValueError("TEST label count does not match n_test")
    if any(index < 0 or index >= n_test for index in candidates):
        raise ValueError("candidate index is out of range")
    true_indices = {
        index for index, label in enumerate(labels) if str(label) not in {"0", "normal"}
    }
    hard_fn = true_indices - hard
    review = standard | priority
    details: list[dict[str, Any]] = []
    for index in sorted(candidates):
        details.append(
            {
                "candidate_index": index,
                "is_true_anomaly": index in true_indices,
                "is_baseline_hard": index in hard,
                "is_baseline_standard": index in standard,
                "is_baseline_priority": index in priority,
                "recovers_hard_fn": index in hard_fn,
                "outside_existing_review": index not in review,
            }
        )
    candidate_tp = candidates & true_indices
    candidate_fp = candidates - true_indices
    review_overlap = candidates & review
    return (
        {
            "candidate_count": len(candidates),
            "candidate_tp": len(candidate_tp),
            "added_fp": len(candidate_fp),
            "candidate_precision": (
                len(candidate_tp) / len(candidates) if candidates else 0.0
            ),
            "baseline_hard_fn": len(hard_fn),
            "recovered_hard_fn": len(candidates & hard_fn),
            "review_overlap_count": len(review_overlap),
            "review_recovered_fn": len(review_overlap & hard_fn),
            "review_overlap_fp": len(review_overlap - true_indices),
            "novel_recovered_fn": len((candidates - review) & hard_fn),
            "outside_review_fp": len((candidates - review) - true_indices),
        },
        details,
    )


def _candidate_signature(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows,
        key=lambda item: (str(item["dataset"]), int(item["candidate_index"])),
    ):
        digest.update(
            f"{row['dataset']}\0{int(row['candidate_index'])}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _lineage(dataset: str) -> str:
    for marker in ("_normal_", "_anomaly_"):
        if marker in dataset:
            return dataset.split(marker, 1)[0]
    return dataset


def run_evaluate(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    prediction_rows, prediction_manifest = load_verified_prediction(
        output_root / PREDICTION_NAME,
        output_root / PREDICTION_MANIFEST_NAME,
    )
    if not (output_root / "exp394_alpha_summary.csv").exists():
        raise ValueError("Exp394 calibration audit must complete before evaluation")
    baseline, baseline_metadata = load_baseline_lanes()
    dataset_names = {str(row["dataset"]) for row in prediction_rows}
    if not dataset_names.issubset(baseline):
        raise ValueError("baseline lane coverage is incomplete")
    labels_map = load_test_labels(dataset_names)

    config = GrammarConfig()
    dataset_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        dataset = str(prediction["dataset"])
        n_test = int(prediction["n_test"])
        lanes = baseline[dataset]
        labels = labels_map[dataset]
        for alpha in config.alphas:
            alpha_key = f"{alpha:.3f}"
            source_map = prediction.get(
                "candidate_indices_by_source_alpha",
                {},
            ).get(alpha_key, {})
            for score_name in SCORE_NAMES:
                candidates = {int(value) for value in source_map.get(score_name, [])}
                metrics, details = evaluate_candidate_set(
                    candidates=candidates,
                    hard=lanes["hard"],
                    standard=lanes["standard"],
                    priority=lanes["priority"],
                    labels=labels,
                    n_test=n_test,
                )
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "lineage": _lineage(dataset),
                        "score": score_name,
                        "alpha": alpha,
                        "n_test": n_test,
                        "abstained": bool(prediction.get("abstained", False)),
                        **metrics,
                    }
                )
                candidate_rows.extend(
                    {
                        "dataset": dataset,
                        "lineage": _lineage(dataset),
                        "score": score_name,
                        "alpha": alpha,
                        **detail,
                    }
                    for detail in details
                )

    summary_rows: list[dict[str, Any]] = []
    for alpha in config.alphas:
        for score_name in SCORE_NAMES:
            selected = [
                row
                for row in dataset_rows
                if row["alpha"] == alpha and row["score"] == score_name
            ]
            selected_candidates = [
                row
                for row in candidate_rows
                if row["alpha"] == alpha and row["score"] == score_name
            ]
            candidate_count = sum(int(row["candidate_count"]) for row in selected)
            candidate_tp = sum(int(row["candidate_tp"]) for row in selected)
            added_fp = sum(int(row["added_fp"]) for row in selected)
            contributing_lineages = {
                str(row["lineage"])
                for row in selected_candidates
                if bool(row["recovers_hard_fn"])
            }
            summary_rows.append(
                {
                    "score": score_name,
                    "alpha": alpha,
                    "datasets": len(selected),
                    "abstained_datasets": sum(
                        int(bool(row["abstained"])) for row in selected
                    ),
                    "candidate_count": candidate_count,
                    "recovered_hard_fn": sum(
                        int(row["recovered_hard_fn"]) for row in selected
                    ),
                    "added_fp": added_fp,
                    "candidate_precision": (
                        candidate_tp / candidate_count if candidate_count else 0.0
                    ),
                    "review_overlap_count": sum(
                        int(row["review_overlap_count"]) for row in selected
                    ),
                    "review_recovered_fn": sum(
                        int(row["review_recovered_fn"]) for row in selected
                    ),
                    "novel_recovered_fn": sum(
                        int(row["novel_recovered_fn"]) for row in selected
                    ),
                    "outside_review_fp": sum(
                        int(row["outside_review_fp"]) for row in selected
                    ),
                    "tp_contributing_lineages": len(contributing_lineages),
                    "candidate_signature_sha256": _candidate_signature(
                        selected_candidates
                    ),
                    "eligible_for_deployment": False,
                    "test_metrics_select_policy": False,
                    "retrospective_counterfactual": True,
                    "prospective_validated": False,
                }
            )

    write_csv(output_root / "exp395_dataset_results.csv", dataset_rows)
    write_csv(output_root / "exp395_candidate_diff.csv", candidate_rows)
    write_csv(output_root / "exp395_summary.csv", summary_rows)
    report_lines = [
        "# Exp395 locked complementarity evaluation",
        "",
        "Prediction SHA256를 검증한 뒤 TEST 라벨을 사후 평가에만 사용했다.",
        "어떤 score 또는 alpha도 운영 정책으로 선택하지 않는다.",
        "",
        "| score | alpha | candidates | recovered Hard FN | FP | precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['score']} | {row['alpha']:.3f} | "
            f"{row['candidate_count']:,} | {row['recovered_hard_fn']:,} | "
            f"{row['added_fp']:,} | {row['candidate_precision']:.4f} |"
        )
    (output_root / "exp395_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    evaluation_manifest = {
        "experiment_id": "experiment_395_grammar_complementarity_evaluation",
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "baseline": baseline_metadata,
        "datasets": len(prediction_rows),
        "summary_rows": len(summary_rows),
        "labels_loaded_after_prediction_verification": True,
        "eligible_for_deployment": False,
        "test_metrics_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp395_evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_status(
        output_root,
        phase="evaluate",
        completed=len(prediction_rows),
        total=len(prediction_rows),
        current_dataset="",
        status="completed",
    )
    return evaluation_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("predict", "audit", "evaluate"),
        default="predict",
    )
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.stage == "predict":
        result = run_predict(args.output_dir, args.dataset_limit)
    elif args.stage == "audit":
        result = run_audit(args.output_dir)
    else:
        result = run_evaluate(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
