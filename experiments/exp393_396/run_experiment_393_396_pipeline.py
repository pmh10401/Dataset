from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from experiments.exp393_396.artifact_io import (
    DATA_ROOT,
    PREDICTION_MANIFEST_NAME,
    PREDICTION_NAME,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("predict", "audit"), default="predict")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    if args.stage == "predict":
        result = run_predict(args.output_dir, args.dataset_limit)
    else:
        result = run_audit(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
