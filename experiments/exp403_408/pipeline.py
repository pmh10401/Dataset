"""Checkpointed orchestration for the Exp403-408 TRAIN-only validation run.

The orchestration boundary owns progress and resume state.  Individual Exp403-
408 modules remain responsible for their frozen, label-free calculations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

import numpy as np

from .artifact_io import canonical_json_bytes, sha256_file, write_csv_atomic, write_json_atomic, write_jsonl_atomic
from .consensus import ConsensusSource, build_consensus
from .contracts import ALPHAS, FrozenConfig, ShadowSourcePrediction
from .evaluation import EvaluationLabels, evaluate_exp405
from .feature_registry import load_feature_blocks
from .feature_stability import audit_feature_stability, build_group_summary
from .foundation import DB_PATH, EXP291_MANIFEST_PATH, EXP291_PREDICTION_PATH, EXP292_MANIFEST_PATH, EXP292_PREDICTION_PATH, crossfit_plan, list_datasets_small_to_large, load_dataset_input, load_frozen_lanes
from .prediction import freeze_provenance_contract, load_expected_provenance, predict_dataset, verify_prediction, write_prediction
from .independent_reaggregate import independent_reaggregate
from .reporting import validate_final_reports, write_reports
from .synthetic_anomaly import SyntheticAnomalyConfig, _crossfit_normal_scores, score_synthetic_models, synthetic_shadow_predictions
from .tri_view_ssl import _calibrate_raw_scores, _produce_raw_scores, score_tri_view, shadow_predictions
from experiments.exp393_396.artifact_io import load_test_labels


EXPERIMENT_ID = "experiment_403_408_feature_ml_validation"
DEFAULT_OUTPUT_DIR = Path("/Users/minho/Documents/Dataset/exp403_408_feature_ml_validation")
STAGES = (
    "audit-features",
    "classical-predict",
    "classical-evaluate",
    "ssl-predict",
    "synthetic-predict",
    "consensus-evaluate",
    "report",
)
CPU_CLASSICAL_STAGES = frozenset(("audit-features", "classical-predict"))
LEARNED_STAGES = frozenset(("ssl-predict", "synthetic-predict"))
AGGREGATE_STAGES = frozenset(("classical-evaluate", "report"))
STATUS_NAME = "status.json"
AUDIT_ARTIFACT = "exp403_feature_audit.jsonl"
EVALUATION_ARTIFACT = "exp405_demotion_evaluation.json"
CONSENSUS_ARTIFACT = "exp408_consensus_results.jsonl"
CONSENSUS_SUMMARY = "exp408_consensus_summary.csv"
EXP405_DATASET_METRICS = "exp405_demotion_dataset_metrics.csv"
EXP405_CANDIDATE_METRICS = "exp405_demotion_candidate_evaluation.csv"
EXP408_VARIANT_METRICS = "exp408_variant_metrics.csv"
EXP408_CANDIDATE_METRICS = "exp408_variant_candidate_evaluation.csv"
FROZEN_LABELS_ARTIFACT = "exp403_408_frozen_test_labels.csv"
EVALUATION_INPUT_MANIFEST = "exp403_408_evaluation_input_manifest.json"
EVALUATION_PUBLICATION_MANIFEST = "exp403_408_evaluation_publication_manifest.json"
RESULT_ARTIFACT = "exp403_408_results.csv"
SUMMARY_ARTIFACT = "exp403_408_summary.csv"
STATUS_HEARTBEAT_SECONDS = 30.0
REPORT_ROW_STAGES = (
    "audit-features",
    "classical-predict",
    "ssl-predict",
    "synthetic-predict",
    "consensus-evaluate",
)


@dataclass(frozen=True)
class StageResult:
    stage: str
    completed: int
    total: int
    recomputed_datasets: int
    errors: int
    output_dir: Path


Processor = Callable[[str], Mapping[str, object]]
InputHash = Callable[[str], str]


def read_status(output_dir: Path) -> dict[str, object]:
    """Read the small atomic status payload without touching stage JSONL output."""
    try:
        value = json.loads((Path(output_dir) / STATUS_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read pipeline status.json") from error
    if not isinstance(value, dict):
        raise ValueError("pipeline status.json must be an object")
    return value


def run_smoke(
    output_dir: Path,
    *,
    limit: int = 3,
    stop_after: int | None = None,
    resume: bool = False,
) -> StageResult:
    """Run a deterministic no-data smoke used to prove checkpoint semantics."""
    if limit < 1:
        raise ValueError("smoke limit must be positive")
    datasets = [f"smoke-{index:04d}" for index in range(limit)]
    return run_stage(
        "audit-features",
        datasets,
        Path(output_dir),
        processor=lambda dataset: {"dataset": dataset, "kind": "smoke"},
        stop_after=stop_after,
        resume=resume,
        workers=1,
    )


def run_pipeline(
    *,
    stage: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    resume: bool = True,
    stop_after: int | None = None,
    workers: int = 7,
) -> tuple[StageResult, ...]:
    """Run one requested stage or all stages in frozen order.

    CPU classical work uses no more than seven outer processes.  Learned
    stages deliberately use one outer process because their modules already
    train each model serially and must never acquire nested model parallelism.
    """
    selected = STAGES if stage == "all" else (stage,)
    if any(name not in STAGES for name in selected):
        raise ValueError(f"unknown stage: {stage}")
    datasets = list_datasets_small_to_large()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        datasets = datasets[:limit]
    if not datasets:
        raise ValueError("dataset source returned no datasets")

    run_id = os.environ.get("EXP403_408_RUN_ID") or uuid.uuid4().hex
    run_started_unix = float(os.environ.get("EXP403_408_RUN_STARTED_UNIX", time.time()))
    results: list[StageResult] = []
    for stage_name in selected:
        stage_datasets = ("__aggregate__",) if stage_name in AGGREGATE_STAGES else datasets
        stage_workers = 1 if stage_name in AGGREGATE_STAGES else stage_worker_count(stage_name, workers)
        stage_resume = resume
        if stage_name == "report" and resume:
            try:
                validate_final_reports(Path(output_dir))
            except (OSError, ValueError, json.JSONDecodeError):
                stage_resume = False
        try:
            result = run_stage(
                stage_name,
                stage_datasets,
                Path(output_dir),
                processor=_processor_for(stage_name, Path(output_dir)),
                stop_after=stop_after,
                resume=stage_resume,
                workers=stage_workers,
                device="cpu",
                run_id=run_id,
                run_started_unix=run_started_unix,
                input_hash_for_dataset=lambda dataset, name=stage_name: _stage_input_hash(
                    name, dataset, Path(output_dir), "cpu"
                ),
            )
            if result.errors:
                raise RuntimeError(f"{stage_name} recorded {result.errors} dataset errors")
            _finalize_stage(stage_name, Path(output_dir), stage_datasets)
        except Exception as error:
            _write_failed_status(Path(output_dir), stage_name, error)
            raise
        results.append(result)
    if stage == "all" or stage == "report":
        try:
            _require_final_outputs(Path(output_dir))
            validate_final_reports(Path(output_dir))
        except Exception as error:
            _write_failed_status(Path(output_dir), "report", error)
            raise
    return tuple(results)


def run_stage(
    stage: str,
    datasets: Sequence[str] | Iterable[str],
    output_dir: Path,
    *,
    processor: Processor,
    stop_after: int | None = None,
    resume: bool = True,
    workers: int = 1,
    device: str = "cpu",
    input_hash_for_dataset: InputHash | None = None,
    run_id: str | None = None,
    run_started_unix: float | None = None,
    heartbeat_interval_seconds: float = STATUS_HEARTBEAT_SECONDS,
) -> StageResult:
    """Execute one stage with atomic per-dataset result or abstention records."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    ordered = tuple(dict.fromkeys(str(dataset) for dataset in datasets))
    if not ordered:
        raise ValueError("stage requires at least one dataset")
    if stop_after is not None and stop_after < 0:
        raise ValueError("stop_after must be non-negative")
    if heartbeat_interval_seconds <= 0.0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    output = Path(output_dir)
    checkpoint_dir = output / "checkpoints" / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(ordered)
    started = time.monotonic()
    active_run_id = str(run_id or uuid.uuid4().hex)
    active_run_started_unix = float(run_started_unix if run_started_unix is not None else time.time())
    completed = 0
    errors = 0
    recomputed = 0
    last_hash = ""
    last_processed_dataset: str | None = None
    pending: list[str] = []
    input_hashes: dict[str, str] = {}

    _write_status(
        output,
        stage=stage,
        current_dataset=ordered[0],
        completed=completed,
        total=total,
        errors=errors,
        device=device,
        checkpoint_sha256=last_hash,
        newly_processed=recomputed,
        started=started,
        status="running",
        run_id=active_run_id,
        run_started_unix=active_run_started_unix,
    )
    for dataset in ordered:
        checkpoint = checkpoint_dir / f"{_safe_name(dataset)}.json"
        try:
            input_hash = (
                input_hash_for_dataset(dataset)
                if input_hash_for_dataset is not None
                else _checkpoint_input_hash(stage, dataset, device)
            )
            input_hashes[dataset] = input_hash
            if resume and _checkpoint_is_valid(checkpoint, input_hash):
                completed += 1
                existing = _read_checkpoint(checkpoint)
                errors += _is_execution_error(existing)
                last_hash = str(existing["checkpoint_sha256"])
                continue
        except Exception as error:
            checkpoint_data = _error_checkpoint(
                stage,
                dataset,
                device,
                _hash_failure_input_hash(stage, dataset, device, error),
                error,
            )
            last_hash = _write_checkpoint(checkpoint, checkpoint_data)
            completed += 1
            recomputed += 1
            errors += 1
            last_processed_dataset = dataset
            _write_status(
                output,
                stage=stage,
                current_dataset=dataset,
                completed=completed,
                total=total,
                errors=errors,
                device=device,
                checkpoint_sha256=last_hash,
                newly_processed=recomputed,
                started=started,
                status="running",
                run_id=active_run_id,
                run_started_unix=active_run_started_unix,
            )
            continue
        pending.append(dataset)

    _write_status(
        output,
        stage=stage,
        current_dataset=pending[0] if pending else ordered[-1],
        completed=completed,
        total=total,
        errors=errors,
        device=device,
        checkpoint_sha256=last_hash,
        newly_processed=recomputed,
        started=started,
        status="running",
        run_id=active_run_id,
        run_started_unix=active_run_started_unix,
    )

    if workers > 1 and len(pending) > 1 and stop_after is None:
        # Parent process serializes checkpoints and status.  Workers only run
        # CPU-only dataset processors, which keeps writes and progress atomic.
        with ProcessPoolExecutor(max_workers=min(7, int(workers))) as executor:
            futures = {executor.submit(_run_processor, processor, dataset): dataset for dataset in pending}
            for future in as_completed(futures):
                dataset = futures[future]
                checkpoint = _checkpoint_from_future(
                    stage, dataset, device, input_hashes[dataset], future
                )
                last_hash = _write_checkpoint(checkpoint_dir / f"{_safe_name(dataset)}.json", checkpoint)
                completed += 1
                recomputed += 1
                last_processed_dataset = dataset
                errors += _is_execution_error(checkpoint)
                _write_status(output, stage=stage, current_dataset=dataset, completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, newly_processed=recomputed, started=started, status="running", run_id=active_run_id, run_started_unix=active_run_started_unix)
    else:
        for dataset in pending:
            if stop_after is not None and recomputed >= stop_after:
                break
            checkpoint = _checkpoint_from_call_with_heartbeat(
                stage,
                dataset,
                device,
                input_hashes[dataset],
                processor,
                output=output,
                completed=completed,
                total=total,
                errors=errors,
                checkpoint_sha256=last_hash,
                newly_processed=recomputed,
                started=started,
                run_id=active_run_id,
                run_started_unix=active_run_started_unix,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            )
            last_hash = _write_checkpoint(checkpoint_dir / f"{_safe_name(dataset)}.json", checkpoint)
            completed += 1
            recomputed += 1
            last_processed_dataset = dataset
            errors += _is_execution_error(checkpoint)
            _write_status(output, stage=stage, current_dataset=dataset, completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, newly_processed=recomputed, started=started, status="running", run_id=active_run_id, run_started_unix=active_run_started_unix)

    final_status = "completed" if completed == total else "running"
    _write_status(output, stage=stage, current_dataset=last_processed_dataset or ordered[-1], completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, newly_processed=recomputed, started=started, status=final_status, run_id=active_run_id, run_started_unix=active_run_started_unix)
    return StageResult(stage, completed, total, recomputed, errors, output)


def _processor_for(stage: str, output: Path) -> Processor:
    processors: dict[str, Processor] = {
        "audit-features": _audit_features,
        "classical-predict": _classical_predict,
        "classical-evaluate": lambda _dataset: _classical_evaluate(output),
        "ssl-predict": _ssl_predict,
        "synthetic-predict": _synthetic_predict,
        "consensus-evaluate": lambda dataset: _consensus_evaluate(dataset, output),
        "report": lambda _dataset: _write_report(output),
    }
    return processors[stage]


def stage_worker_count(stage: str, requested_workers: int) -> int:
    """Return the frozen outer-worker policy for one pipeline stage."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if stage in LEARNED_STAGES:
        return 1
    if stage in CPU_CLASSICAL_STAGES:
        return 7
    return 1


def _audit_features(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    blocks = load_feature_blocks(dataset, bundle)
    audit = audit_feature_stability(blocks, crossfit_plan(len(bundle.train_normal)))
    return {
        "dataset": dataset,
        "outcome": "completed",
        "train_size": int(len(bundle.train_normal)),
        "audit": _json_value(audit),
        "group_summary": _json_value(build_group_summary(audit)),
    }


def _classical_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    if not bundle.hard_indices:
        return {
            "dataset": dataset,
            "outcome": "completed",
            "result_kind": "no_frozen_hard_candidates",
            "prediction_rows": [],
        }
    rows = predict_dataset(bundle, FrozenConfig(target_length=bundle.target_length))
    return {"dataset": dataset, "outcome": "completed", "prediction_rows": _json_value(rows)}


def _ssl_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    config = FrozenConfig(target_length=bundle.target_length)
    scores = score_tri_view(bundle.train_normal, bundle.test_values, config, device="cpu")
    raw_oof: tuple[float, ...] = ()
    if scores.available:
        raw = _produce_raw_scores(
            np.asarray(bundle.train_normal), np.asarray(bundle.test_values), config, "cpu"
        )
        scores = _calibrate_raw_scores(raw)
        raw_oof = raw.train_oof_scores
    return {
        "dataset": dataset,
        "outcome": "completed" if scores.available else "abstained",
        "abstention_reason": scores.abstention_reason,
        "scores": _json_value(scores),
        "train_oof_scores": _json_value(raw_oof),
        "shadow_predictions": _json_value(shadow_predictions(dataset, scores, bundle.hard_indices)),
    }


def _synthetic_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    config = SyntheticAnomalyConfig(frozen=FrozenConfig(target_length=bundle.target_length))
    scores = score_synthetic_models(bundle.train_normal, bundle.test_values, config)
    oof_by_model = {
        score.model_name: _crossfit_normal_scores(
            np.asarray(bundle.train_normal),
            config=config,
            model_name=score.model_name,
            seed=int(config.frozen.seed) + model_index * 10_000,
        )
        for model_index, score in enumerate(scores)
        if score.available
    }
    return {
        "dataset": dataset,
        "outcome": "completed" if all(score.available for score in scores) else "abstained",
        "abstention_reason": next((score.abstention_reason for score in scores if not score.available), ""),
        "scores": _json_value(scores),
        "train_oof_by_model": _json_value(oof_by_model),
        "shadow_predictions": _json_value({score.model_name: synthetic_shadow_predictions(dataset, score, bundle.hard_indices) for score in scores}),
    }


def _consensus_evaluate(dataset: str, output: Path) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    sources = _consensus_sources(dataset, bundle, output)
    result = build_consensus(load_frozen_lanes(), sources)
    return {"dataset": dataset, "outcome": "completed", "consensus": _json_value(result)}


def _classical_evaluate(output: Path) -> Mapping[str, object]:
    manifest = _read_json(output / "exp404_prediction_manifest.json")
    contract_path = output / "exp403_404_provenance_contract.json"
    evaluation = evaluate_exp405(
        output,
        _evaluation_labels(output),
        frozen_lane_universe=load_frozen_lanes(),
        provenance_contract_path=contract_path,
        expected_provenance_sha256=str(manifest["provenance_contract_sha256"]),
    )
    write_json_atomic(output / EVALUATION_ARTIFACT, _json_value(evaluation))
    return {"dataset": "__aggregate__", "outcome": "completed", "evaluation": _json_value(evaluation)}


def _write_report(output: Path) -> Mapping[str, object]:
    _write_published_evaluation_tables(output)
    _require_final_outputs(output, require_report=False)
    result_rows = _report_rows(output)
    write_csv_atomic(
        output / RESULT_ARTIFACT,
        fieldnames=("dataset", "stage", "outcome", "checkpoint_sha256"),
        rows=result_rows,
    )
    summary_rows = [
        {"artifact": AUDIT_ARTIFACT, "sha256": sha256_file(output / AUDIT_ARTIFACT)},
        {"artifact": "exp404_prediction_manifest.json", "sha256": sha256_file(output / "exp404_prediction_manifest.json")},
        {"artifact": EVALUATION_ARTIFACT, "sha256": sha256_file(output / EVALUATION_ARTIFACT)},
        {"artifact": EXP405_DATASET_METRICS, "sha256": sha256_file(output / EXP405_DATASET_METRICS)},
        {"artifact": EXP405_CANDIDATE_METRICS, "sha256": sha256_file(output / EXP405_CANDIDATE_METRICS)},
        {"artifact": CONSENSUS_ARTIFACT, "sha256": sha256_file(output / CONSENSUS_ARTIFACT)},
        {"artifact": CONSENSUS_SUMMARY, "sha256": sha256_file(output / CONSENSUS_SUMMARY)},
        {"artifact": EXP408_VARIANT_METRICS, "sha256": sha256_file(output / EXP408_VARIANT_METRICS)},
        {"artifact": EXP408_CANDIDATE_METRICS, "sha256": sha256_file(output / EXP408_CANDIDATE_METRICS)},
        {"artifact": FROZEN_LABELS_ARTIFACT, "sha256": sha256_file(output / FROZEN_LABELS_ARTIFACT)},
        {"artifact": EVALUATION_INPUT_MANIFEST, "sha256": sha256_file(output / EVALUATION_INPUT_MANIFEST)},
        {"artifact": EVALUATION_PUBLICATION_MANIFEST, "sha256": sha256_file(output / EVALUATION_PUBLICATION_MANIFEST)},
    ]
    write_csv_atomic(output / SUMMARY_ARTIFACT, fieldnames=("artifact", "sha256"), rows=summary_rows)
    write_reports(
        output,
        command=f"python3 scratch/run_exp403_408_feature_ml_validation.py --stage report --output-dir {output}",
    )
    return {"dataset": "__aggregate__", "outcome": "completed", "result_rows": len(result_rows)}


def _write_published_evaluation_tables(output: Path) -> None:
    """Freeze official label-based detail rows before the read-only audit runs."""
    _verify_evaluation_input_manifest(output)
    prediction = _read_jsonl_artifact(output / "exp404_hard_confidence.jsonl")
    consensus_rows = _read_jsonl_artifact(output / CONSENSUS_ARTIFACT)
    audit_rows = _read_jsonl_artifact(output / AUDIT_ARTIFACT)
    audit_by_dataset = {str(row["dataset"]): row for row in audit_rows}
    prediction_datasets = {str(row["dataset"]) for row in prediction}
    datasets = {str(row["dataset"]) for row in consensus_rows}
    if not prediction_datasets <= datasets or datasets != set(audit_by_dataset):
        raise ValueError("published Exp403-408 evaluation dataset coverage differs")
    labels = load_test_labels(datasets)
    if set(labels) != datasets:
        raise ValueError("published Exp403-408 labels are incomplete")
    frozen_labels = [
        {"dataset": dataset, "candidate_index": index, "is_true_anomaly": int(_label_is_anomaly(value))}
        for dataset in sorted(labels)
        for index, value in enumerate(labels[dataset])
    ]
    write_csv_atomic(output / FROZEN_LABELS_ARTIFACT, fieldnames=("dataset", "candidate_index", "is_true_anomaly"), rows=frozen_labels)
    by_dataset: dict[str, list[Mapping[str, object]]] = {}
    for row in prediction:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    exp405_rows = []
    exp405_candidate_rows = []
    for dataset in sorted(prediction_datasets):
        rows = by_dataset[dataset]
        values = labels[dataset]
        _validate_label_indices(rows, values, dataset)
        recheck = [
            row for row in rows
            if all(value == "recheck" for value in dict(row["confidence_by_alpha"]).values())
        ]
        exp405_candidate_rows.extend(
            {
                "dataset": dataset,
                "candidate_index": int(row["candidate_index"]),
                "is_recheck": int(all(value == "recheck" for value in dict(row["confidence_by_alpha"]).values())),
                "is_true_anomaly": int(_label_is_anomaly(values[int(row["candidate_index"])])),
            }
            for row in rows
        )
        exp405_rows.append(
            {
                "dataset": dataset,
                "lineage": _lineage(dataset),
                "train_stability": _dataset_train_stability(output, dataset),
                "baseline_hard_tp": sum(_label_is_anomaly(values[int(row["candidate_index"])]) for row in rows),
                "recheck_count": len(recheck),
                "recheck_false_positive_count": sum(not _label_is_anomaly(values[int(row["candidate_index"])]) for row in recheck),
                "recheck_true_positive_count": sum(_label_is_anomaly(values[int(row["candidate_index"])]) for row in recheck),
            }
        )
    write_csv_atomic(
        output / EXP405_DATASET_METRICS,
        fieldnames=(
            "dataset", "lineage", "train_stability", "baseline_hard_tp", "recheck_count",
            "recheck_false_positive_count", "recheck_true_positive_count",
        ),
        rows=exp405_rows,
    )
    write_csv_atomic(
        output / EXP405_CANDIDATE_METRICS,
        fieldnames=("dataset", "candidate_index", "is_recheck", "is_true_anomaly"),
        rows=exp405_candidate_rows,
    )
    consensus_by_dataset = {str(row["dataset"]): _mapping_object(row.get("consensus"), "published consensus") for row in consensus_rows}
    exp408_rows = []
    exp408_candidate_rows = []
    for dataset in sorted(datasets):
        payload = consensus_by_dataset[dataset]
        values = labels[dataset]
        hard = _published_indices(payload.get("hard_indices"), "published Hard indices")
        standard = _published_indices(payload.get("standard_indices"), "published Standard indices")
        priority = _published_indices(payload.get("priority_indices"), "published Priority indices")
        if hard != {int(row["candidate_index"]) for row in by_dataset.get(dataset, [])}:
            raise ValueError(f"published Exp408 Hard parity mismatch: {dataset}")
        _validate_index_set(hard | standard | priority, values, dataset)
        baseline_tp = sum(_label_is_anomaly(values[index]) for index in hard)
        baseline_fp = len(hard) - baseline_tp
        baseline_fn = sum(_label_is_anomaly(value) for value in values) - baseline_tp
        variants = _mapping_object(payload.get("candidates_by_variant"), "published consensus variants")
        for variant, alpha_rows in variants.items():
            for alpha, candidates in _mapping_object(alpha_rows, "published consensus alpha rows").items():
                indices = {
                    int(_mapping_object(candidate, "published consensus candidate")["candidate_index"])
                    for candidate in _sequence_object(candidates, "published consensus candidate list")
                }
                if len(indices) != len(_sequence_object(candidates, "published consensus candidate list")) or indices & hard:
                    raise ValueError(f"published Exp408 Shadow candidates are invalid: {dataset}")
                _validate_index_set(indices, values, dataset)
                tp = sum(_label_is_anomaly(values[index]) for index in indices)
                fp = len(indices) - tp
                exp408_candidate_rows.extend(
                    {
                        "dataset": dataset,
                        "variant": str(variant),
                        "alpha": str(alpha),
                        "candidate_index": index,
                        "is_true_anomaly": int(_label_is_anomaly(values[index])),
                    }
                    for index in sorted(indices)
                )
                exp408_rows.append(
                    {
                        "dataset": dataset,
                        "lineage": _lineage(dataset),
                        "train_size": int(audit_by_dataset[dataset]["train_size"]),
                        "variant": str(variant),
                        "alpha": str(alpha),
                        "candidate_count": len(indices),
                        "tp": tp,
                        "fp": fp,
                        "precision": tp / len(indices) if indices else 0.0,
                        "fn_recovered": tp,
                        "zero_f1_before": int(baseline_tp == 0),
                        "zero_f1_after": int(baseline_tp + tp == 0),
                        "standard_overlap": len(indices & standard),
                        "priority_overlap": len(indices & priority),
                        "review_outside_existing": len(indices - standard - priority),
                        "baseline_hard_tp": baseline_tp,
                        "baseline_hard_fp": baseline_fp,
                        "baseline_hard_fn": baseline_fn,
                    }
                )
    write_csv_atomic(
        output / EXP408_VARIANT_METRICS,
        fieldnames=(
            "dataset", "lineage", "train_size", "variant", "alpha", "candidate_count", "tp", "fp",
            "precision", "fn_recovered", "zero_f1_before", "zero_f1_after", "standard_overlap",
            "priority_overlap", "review_outside_existing", "baseline_hard_tp", "baseline_hard_fp", "baseline_hard_fn",
        ),
        rows=exp408_rows,
    )
    write_csv_atomic(
        output / EXP408_CANDIDATE_METRICS,
        fieldnames=("dataset", "variant", "alpha", "candidate_index", "is_true_anomaly"),
        rows=exp408_candidate_rows,
    )
    _write_evaluation_publication_manifest(output)


def _finalize_stage(stage: str, output: Path, datasets: Sequence[str]) -> None:
    if stage == "audit-features":
        rows = [_read_stage_checkpoint(output, stage, dataset) for dataset in datasets]
        _require_completed_rows(stage, rows)
        write_jsonl_atomic(output / AUDIT_ARTIFACT, rows)
    elif stage == "classical-predict":
        _publish_classical_prediction(output, datasets)
    elif stage == "consensus-evaluate":
        rows = [_read_stage_checkpoint(output, stage, dataset) for dataset in datasets]
        _require_completed_rows(stage, rows)
        write_jsonl_atomic(output / CONSENSUS_ARTIFACT, rows)
        summary_rows = []
        for row in rows:
            consensus = row["consensus"]
            for variant, values in consensus["candidates_by_variant"].items():
                summary_rows.append(
                    {
                        "dataset": row["dataset"],
                        "variant": variant,
                        "candidate_count": sum(len(items) for items in values.values()),
                    }
                )
        write_csv_atomic(
            output / CONSENSUS_SUMMARY,
            fieldnames=("dataset", "variant", "candidate_count"),
            rows=summary_rows,
        )
        _write_evaluation_input_manifest(output)


def _publish_classical_prediction(output: Path, datasets: Sequence[str]) -> None:
    rows = [_read_stage_checkpoint(output, "classical-predict", dataset) for dataset in datasets]
    _require_completed_rows("classical-predict", rows)
    prediction_rows = [
        prediction
        for row in rows
        for prediction in row.get("prediction_rows", [])
    ]
    # Target length is derived from each dataset's TRAIN runs and is captured
    # in every checkpoint input hash.  The published contract freezes the
    # shared scoring parameters consumed by Task 6.
    config = FrozenConfig()
    config_path = output / "exp403_404_frozen_config.json"
    write_json_atomic(config_path, _json_value(config))
    contract_path = output / "exp403_404_provenance_contract.json"
    contract_sha256 = freeze_provenance_contract(
        contract_path,
        feature_artifacts={"exp403_audit": output / AUDIT_ARTIFACT},
        input_artifacts={"dataset_db": DB_PATH},
        baseline_artifacts={"exp291_frozen_lanes": EXP291_PREDICTION_PATH, "exp292_frozen_lanes": EXP292_PREDICTION_PATH},
        config_artifacts={"frozen_config": config_path},
        code_artifacts={
            "pipeline": Path(__file__),
            "foundation": Path(load_dataset_input.__code__.co_filename),
            "feature_registry": Path(load_feature_blocks.__code__.co_filename),
            "classical_prediction": Path(predict_dataset.__code__.co_filename),
        },
    )
    expected_provenance = load_expected_provenance(contract_path, expected_sha256=contract_sha256)
    write_prediction(
        prediction_rows,
        output,
        config=config,
        frozen_lane_universe=load_frozen_lanes(),
        expected_provenance=expected_provenance,
    )


def _consensus_sources(dataset: str, bundle: object, output: Path) -> tuple[ConsensusSource, ...]:
    ssl = _read_stage_checkpoint(output, "ssl-predict", dataset)
    synthetic = _read_stage_checkpoint(output, "synthetic-predict", dataset)
    sources: list[ConsensusSource] = []
    if ssl.get("outcome") == "completed" and ssl.get("train_oof_scores"):
        sources.append(
            ConsensusSource(
                name="tri_view",
                lineage="tri_view_encoder",
                kind="tri_view",
                train_oof_scores=tuple(float(value) for value in ssl["train_oof_scores"]),
                shadow_predictions=_shadow_mapping(ssl["shadow_predictions"]),
                dataset=dataset,
            )
        )
    for score in synthetic.get("scores", []):
        name = str(score.get("model_name", ""))
        oof = (synthetic.get("train_oof_by_model") or {}).get(name)
        if name and oof:
            sources.append(
                ConsensusSource(
                    name=f"synthetic_{name}",
                    lineage="synthetic_anomaly",
                    kind="synthetic",
                    train_oof_scores=tuple(float(value) for value in oof),
                    shadow_predictions=_shadow_mapping((synthetic.get("shadow_predictions") or {}).get(name, {})),
                    dataset=dataset,
                )
            )
    for block in load_feature_blocks(dataset, bundle):
        if not block.available:
            continue
        train = np.asarray(block.train_oof, dtype=float)
        test = np.asarray(block.test_values, dtype=float)
        sources.append(
            ConsensusSource(
                name=block.name,
                lineage=block.group,
                kind="existing",
                train_oof_scores=tuple(float(value) for value in train),
                shadow_predictions=_existing_shadow_predictions(dataset, train, test, bundle.hard_indices, block.name),
                dataset=dataset,
            )
        )
    if not sources:
        raise ValueError(f"Exp408 has no real source for dataset={dataset}")
    return tuple(sources)


def _existing_shadow_predictions(dataset: str, train: np.ndarray, test: np.ndarray, hard: object, source: str) -> dict[str, tuple[ShadowSourcePrediction, ...]]:
    if train.ndim != 1 or test.ndim != 1 or not len(train) or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("existing source scores are invalid")
    p_values = np.asarray([(np.count_nonzero(train >= value) + 1.0) / (len(train) + 1.0) for value in test], dtype=float)
    hard_indices = frozenset(int(index) for index in hard)
    return {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=dataset,
                candidate_index=int(index),
                scores_by_alpha={str(other): float(p_values[index]) for other in ALPHAS},
                source=source,
            )
            for index, value in enumerate(p_values)
            if index not in hard_indices and float(value) <= float(alpha)
        )
        for alpha in ALPHAS
    }


def _shadow_mapping(value: Mapping[str, object]) -> dict[str, tuple[ShadowSourcePrediction, ...]]:
    return {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=str(row["dataset"]),
                candidate_index=int(row["candidate_index"]),
                scores_by_alpha={str(key): float(score) for key, score in row["scores_by_alpha"].items()},
                source=str(row["source"]),
            )
            for row in value.get(str(alpha), [])
        )
        for alpha in ALPHAS
    }


def _evaluation_labels(output: Path) -> Callable[[set[str]], Mapping[str, EvaluationLabels]]:
    def loader(datasets: set[str]) -> Mapping[str, EvaluationLabels]:
        labels = load_test_labels(datasets)
        return {
            dataset: EvaluationLabels(
                test_labels=tuple(labels[dataset]),
                train_oof_normal_stability=_dataset_train_stability(output, dataset),
                lineage=_lineage(dataset),
            )
            for dataset in datasets
        }
    return loader


def _dataset_train_stability(output: Path, dataset: str) -> float:
    row = _read_stage_checkpoint(output, "audit-features", dataset)
    values = [
        float(value.get("bootstrap_rank_stability", 0.0))
        for value in (row.get("audit", {}).get("feature_stability", {}) or {}).values()
    ]
    return float(np.median(values)) if values else 0.0


def _lineage(dataset: str) -> str:
    for marker in ("_normal_", "_anomaly_"):
        if marker in dataset:
            return dataset.split(marker, 1)[0]
    return dataset


def _report_rows(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage in REPORT_ROW_STAGES:
        for path in sorted((output / "checkpoints" / stage).glob("*.json")):
            row = _read_checkpoint(path)
            rows.append({
                "dataset": row["dataset"],
                "stage": stage,
                "outcome": row["outcome"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            })
    return rows


def _require_completed_rows(stage: str, rows: Sequence[Mapping[str, object]]) -> None:
    failures = [row for row in rows if row.get("outcome") != "completed"]
    if failures:
        raise ValueError(f"{stage} has {len(failures)} non-completed checkpoints")


def _require_final_outputs(output: Path, *, require_report: bool = True) -> None:
    paths = [
        output / AUDIT_ARTIFACT,
        output / "exp404_prediction_manifest.json",
        output / "exp404_hard_confidence.jsonl",
        output / EVALUATION_ARTIFACT,
        output / CONSENSUS_ARTIFACT,
        output / CONSENSUS_SUMMARY,
    ]
    if require_report:
        paths.extend((
            output / RESULT_ARTIFACT,
            output / SUMMARY_ARTIFACT,
            output / EXP405_DATASET_METRICS,
            output / EXP405_CANDIDATE_METRICS,
            output / EXP408_VARIANT_METRICS,
            output / EXP408_CANDIDATE_METRICS,
            output / FROZEN_LABELS_ARTIFACT,
            output / EVALUATION_INPUT_MANIFEST,
            output / EVALUATION_PUBLICATION_MANIFEST,
            output / "EXP403_408_RESULT_REPORT_KO.md",
            output / "MANIFEST.json",
            output / "MANIFEST.md",
        ))
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"required Exp403-408 outputs are missing: {', '.join(missing)}")


def _read_stage_checkpoint(output: Path, stage: str, dataset: str) -> dict[str, object]:
    return _read_checkpoint(output / "checkpoints" / stage / f"{_safe_name(dataset)}.json")


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _evaluation_input_artifacts(output: Path) -> tuple[Path, ...]:
    return (
        output / AUDIT_ARTIFACT,
        output / "exp404_prediction_manifest.json",
        output / "exp404_hard_confidence.jsonl",
        output / "exp403_404_provenance_contract.json",
        output / EVALUATION_ARTIFACT,
        output / CONSENSUS_ARTIFACT,
        output / CONSENSUS_SUMMARY,
    )


def _write_evaluation_input_manifest(output: Path) -> None:
    paths = _evaluation_input_artifacts(output)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("cannot freeze evaluation inputs: " + ", ".join(missing))
    prediction_manifest = _read_json(output / "exp404_prediction_manifest.json")
    write_json_atomic(
        output / EVALUATION_INPUT_MANIFEST,
        {
            "schema": "exp403_408_evaluation_input/v1",
            "provenance_contract_sha256": prediction_manifest.get("provenance_contract_sha256"),
            "artifact_sha256": {path.name: sha256_file(path) for path in paths},
            "external_sha256": {str(path): sha256_file(path) for path in (*_database_snapshot_paths(DB_PATH), *_frozen_lane_input_paths())},
        },
    )


def _verify_evaluation_input_manifest(output: Path) -> None:
    manifest = _read_json(output / EVALUATION_INPUT_MANIFEST)
    hashes = _mapping_object(manifest.get("artifact_sha256"), "evaluation input artifact_sha256")
    for path in _evaluation_input_artifacts(output):
        if hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"evaluation input SHA256 mismatch: {path.name}")
    external = _mapping_object(manifest.get("external_sha256"), "evaluation input external_sha256")
    for path in (*_database_snapshot_paths(DB_PATH), *_frozen_lane_input_paths()):
        if external.get(str(path)) != sha256_file(path):
            raise ValueError(f"evaluation input SHA256 mismatch: {path}")
    prediction_manifest = _read_json(output / "exp404_prediction_manifest.json")
    expected_sha = str(manifest.get("provenance_contract_sha256", ""))
    if expected_sha != prediction_manifest.get("provenance_contract_sha256"):
        raise ValueError("evaluation input externally pinned provenance mismatch")
    expected = load_expected_provenance(output / "exp403_404_provenance_contract.json", expected_sha256=expected_sha)
    verify_prediction(output, frozen_lane_universe=load_frozen_lanes(), expected_provenance=expected)


def _write_evaluation_publication_manifest(output: Path) -> None:
    names = (
        EVALUATION_INPUT_MANIFEST, EXP405_DATASET_METRICS, EXP405_CANDIDATE_METRICS,
        EXP408_VARIANT_METRICS, EXP408_CANDIDATE_METRICS, FROZEN_LABELS_ARTIFACT,
        EVALUATION_ARTIFACT,
    )
    write_json_atomic(
        output / EVALUATION_PUBLICATION_MANIFEST,
        {"schema": "exp403_408_evaluation_publication/v1", "artifact_sha256": {name: sha256_file(output / name) for name in names}},
    )


def _read_jsonl_artifact(path: Path) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read required JSONL artifact: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL artifact is invalid: {path}")
    return rows


def _mapping_object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): child for key, child in value.items()}


def _sequence_object(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _published_indices(value: object, name: str) -> set[int]:
    indices: set[int] = set()
    for raw in _sequence_object(value, name):
        if isinstance(raw, bool):
            raise ValueError(f"{name} must contain integer indices")
        try:
            index = int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain integer indices") from error
        if index < 0 or str(raw).strip() not in {str(index), f"+{index}"}:
            raise ValueError(f"{name} must contain non-negative integer indices")
        indices.add(index)
    return indices


def _validate_label_indices(rows: Sequence[Mapping[str, object]], labels: Sequence[object], dataset: str) -> None:
    _validate_index_set({int(row["candidate_index"]) for row in rows}, labels, dataset)


def _validate_index_set(indices: set[int], labels: Sequence[object], dataset: str) -> None:
    if not labels or any(index < 0 or index >= len(labels) for index in indices):
        raise ValueError(f"published evaluation index is outside TEST labels: {dataset}")
    for value in labels:
        _label_is_anomaly(value)


def _label_is_anomaly(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) in {0.0, 1.0}:
        return bool(int(value))
    if isinstance(value, str):
        parsed = {"normal": False, "false": False, "0": False, "0.0": False, "anomaly": True, "true": True, "1": True, "1.0": True}.get(value.strip().lower())
        if parsed is not None:
            return parsed
    raise ValueError("published evaluation TEST label is invalid")


def _stage_input_hash(stage: str, dataset: str, output: Path, device: str) -> str:
    """Bind resume to real values, code, frozen config, and stage prerequisites."""
    code_paths = {
        "pipeline": Path(__file__),
        "foundation": Path(load_dataset_input.__code__.co_filename),
        "feature_registry": Path(load_feature_blocks.__code__.co_filename),
        "prediction": Path(predict_dataset.__code__.co_filename),
        "ssl": Path(score_tri_view.__code__.co_filename),
        "synthetic": Path(score_synthetic_models.__code__.co_filename),
        "consensus": Path(build_consensus.__code__.co_filename),
        "evaluation": Path(evaluate_exp405.__code__.co_filename),
        "reporting": Path(write_reports.__code__.co_filename),
        "independent_reaggregation": Path(independent_reaggregate.__code__.co_filename),
        "frozen_lanes": Path(load_frozen_lanes.__code__.co_filename),
        "labels": Path(load_test_labels.__code__.co_filename),
    }
    payload: dict[str, object] = {
        "stage": stage,
        "dataset": dataset,
        "device": device,
        "code_sha256": {name: sha256_file(path) for name, path in sorted(code_paths.items())},
    }
    if stage in AGGREGATE_STAGES:
        payload["aggregate_artifacts"] = _content_hashes(
            _aggregate_dependencies(stage, output)
        )
    else:
        bundle = load_dataset_input(dataset)
        payload["frozen_config"] = _json_value(
            FrozenConfig(target_length=int(bundle.target_length))
        )
        payload["dataset_values"] = _dataset_values_sha256(bundle)
        payload["frozen_lanes"] = {
            "hard": sorted(int(value) for value in bundle.hard_indices),
            "standard": sorted(int(value) for value in bundle.standard_indices),
            "priority": sorted(int(value) for value in bundle.priority_indices),
            "target_length": int(bundle.target_length),
        }
        if stage in {"audit-features", "consensus-evaluate"}:
            payload["feature_sources"] = {
                block.name: block.source_hash
                for block in load_feature_blocks(dataset, bundle)
            }
        if stage == "consensus-evaluate":
            payload["frozen_lane_artifacts"] = _content_hashes(
                _frozen_lane_input_paths()
            )
        dependencies = _dataset_stage_dependencies(stage)
        payload["upstream_checkpoints"] = {
            dependency: _read_stage_checkpoint(output, dependency, dataset)["checkpoint_sha256"]
            for dependency in dependencies
        }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _dataset_values_sha256(bundle: object) -> str:
    digest = hashlib.sha256()
    for name in ("train_normal", "test_values"):
        values = np.ascontiguousarray(np.asarray(getattr(bundle, name), dtype=np.float64))
        digest.update(name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _dataset_stage_dependencies(stage: str) -> tuple[str, ...]:
    dependencies = {
        "audit-features": (),
        "classical-predict": ("audit-features",),
        "ssl-predict": ("audit-features",),
        "synthetic-predict": ("audit-features",),
        "consensus-evaluate": ("audit-features", "ssl-predict", "synthetic-predict"),
    }
    return dependencies.get(stage, ())


def _aggregate_dependencies(stage: str, output: Path) -> tuple[Path, ...]:
    if stage == "classical-evaluate":
        manifest = _read_json(output / "exp404_prediction_manifest.json")
        contract = _read_json(output / "exp403_404_provenance_contract.json")
        paths = (
            output / "exp404_prediction_manifest.json",
            output / str(manifest.get("hard_confidence_file", "")),
            output / str(manifest.get("feature_manifest_file", "")),
            output / "exp403_404_provenance_contract.json",
            *_database_snapshot_paths(DB_PATH),
            *_frozen_lane_input_paths(),
            *_provenance_references(contract),
            *_checkpoint_files(output / "checkpoints" / "audit-features"),
        )
    elif stage == "report":
        paths = (
            output / AUDIT_ARTIFACT,
            output / "exp404_prediction_manifest.json",
            output / "exp404_hard_confidence.jsonl",
            output / EVALUATION_ARTIFACT,
            output / CONSENSUS_ARTIFACT,
            output / CONSENSUS_SUMMARY,
            *_database_snapshot_paths(DB_PATH),
            *(
                checkpoint
                for report_stage in REPORT_ROW_STAGES
                for checkpoint in _checkpoint_files(output / "checkpoints" / report_stage)
            ),
        )
    else:
        raise ValueError(f"unknown aggregate stage: {stage}")
    paths = _unique_paths(paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"{stage} requires missing artifacts: {', '.join(missing)}")
    return paths


def _frozen_lane_input_paths() -> tuple[Path, ...]:
    return (
        EXP291_PREDICTION_PATH,
        EXP291_MANIFEST_PATH,
        EXP292_PREDICTION_PATH,
        EXP292_MANIFEST_PATH,
    )


def _database_snapshot_paths(database: Path) -> tuple[Path, ...]:
    path = Path(database)
    return (
        path,
        *(candidate for candidate in (path.with_name(f"{path.name}-wal"), path.with_name(f"{path.name}-shm")) if candidate.is_file()),
    )


def _provenance_references(contract: Mapping[str, object]) -> tuple[Path, ...]:
    provenance = contract.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance contract is missing provenance entries")
    paths: list[Path] = []
    for category in sorted(provenance):
        entries = provenance[category]
        if not isinstance(entries, Mapping):
            raise ValueError("provenance contract category is invalid")
        for name in sorted(entries):
            details = entries[name]
            if not isinstance(details, Mapping) or not isinstance(details.get("path"), str):
                raise ValueError("provenance contract artifact path is invalid")
            paths.append(Path(str(details["path"])))
    return tuple(paths)


def _checkpoint_files(directory: Path) -> tuple[Path, ...]:
    root = Path(directory)
    if not root.is_dir():
        return ()
    return tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=str))


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = Path(path).resolve()
        unique[str(resolved)] = resolved
    return tuple(unique[key] for key in sorted(unique))


def _content_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in _unique_paths(paths)
    }


def _write_failed_status(output: Path, stage: str, error: Exception) -> None:
    try:
        status = read_status(output)
    except ValueError:
        status = {}
    write_json_atomic(
        output / STATUS_NAME,
        {
            "id": EXPERIMENT_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": status.get("run_id", uuid.uuid4().hex),
            "run_started_unix": float(status.get("run_started_unix", time.time())),
            "status": "failed",
            "stage": stage,
            "current_dataset": status.get("current_dataset", ""),
            "completed": int(status.get("completed", 0)),
            "total": int(status.get("total", 0)),
            "newly_processed": int(status.get("newly_processed", 0)),
            "done": int(status.get("completed", 0)),
            "expected": int(status.get("total", 0)),
            "progress_percent": float(status.get("progress_percent", 0.0)),
            "eta_seconds": status.get("eta_seconds"),
            "throughput_per_minute": float(status.get("throughput_per_minute", 0.0)),
            "errors": int(status.get("errors", 0)) + 1,
            "device": status.get("device", "cpu"),
            "checkpoint_sha256": status.get("checkpoint_sha256", ""),
            "status_checkpoint_sha256": status.get("checkpoint_sha256", ""),
            "elapsed_seconds": float(status.get("elapsed_seconds", 0.0)),
            "last_error": f"{type(error).__name__}: {error}",
            "updated_unix": time.time(),
        },
    )


def _run_processor(processor: Processor, dataset: str) -> Mapping[str, object]:
    return processor(dataset)


def _checkpoint_from_call_with_heartbeat(
    stage: str,
    dataset: str,
    device: str,
    input_hash: str,
    processor: Processor,
    *,
    output: Path,
    completed: int,
    total: int,
    errors: int,
    checkpoint_sha256: str,
    newly_processed: int,
    started: float,
    run_id: str,
    run_started_unix: float,
    heartbeat_interval_seconds: float,
) -> dict[str, object]:
    """Keep atomic live status fresh while one learned dataset is executing."""
    if stage not in LEARNED_STAGES:
        return _checkpoint_from_call(stage, dataset, device, input_hash, processor)

    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(heartbeat_interval_seconds):
            _write_status(
                output,
                stage=stage,
                current_dataset=dataset,
                completed=completed,
                total=total,
                errors=errors,
                device=device,
                checkpoint_sha256=checkpoint_sha256,
                newly_processed=newly_processed,
                started=started,
                status="running",
                run_id=run_id,
                run_started_unix=run_started_unix,
            )

    thread = threading.Thread(target=heartbeat, name=f"exp403-heartbeat-{dataset}", daemon=True)
    thread.start()
    try:
        return _checkpoint_from_call(stage, dataset, device, input_hash, processor)
    finally:
        stopped.set()
        thread.join(timeout=heartbeat_interval_seconds + 1.0)


def _checkpoint_from_call(
    stage: str, dataset: str, device: str, input_hash: str, processor: Processor
) -> dict[str, object]:
    try:
        payload = processor(dataset)
    except Exception as error:  # An error is a durable, explicit abstention.
        return _error_checkpoint(stage, dataset, device, input_hash, error)
    return _success_checkpoint(stage, dataset, device, input_hash, payload)


def _checkpoint_from_future(
    stage: str, dataset: str, device: str, input_hash: str, future: object
) -> dict[str, object]:
    try:
        payload = future.result()
    except Exception as error:
        return _error_checkpoint(stage, dataset, device, input_hash, error)
    return _success_checkpoint(stage, dataset, device, input_hash, payload)


def _success_checkpoint(
    stage: str, dataset: str, device: str, input_hash: str, payload: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return _error_checkpoint(stage, dataset, device, input_hash, ValueError("stage processor must return a mapping"))
    result = dict(payload)
    result["dataset"] = dataset
    result.setdefault("outcome", "completed")
    if result["outcome"] not in {"completed", "abstained"}:
        return _error_checkpoint(stage, dataset, device, input_hash, ValueError("stage outcome is invalid"))
    if result["outcome"] == "abstained" and not result.get("abstention_reason"):
        return _error_checkpoint(stage, dataset, device, input_hash, ValueError("abstention requires a reason"))
    return {"stage": stage, "dataset": dataset, "device": device, "input_sha256": input_hash, **_json_value(result)}


def _error_checkpoint(
    stage: str, dataset: str, device: str, input_hash: str, error: Exception
) -> dict[str, object]:
    return {
        "stage": stage,
        "dataset": dataset,
        "device": device,
        "input_sha256": input_hash,
        "outcome": "abstained",
        "abstention_reason": "stage_error",
        "error": f"{type(error).__name__}: {error}",
    }


def _hash_failure_input_hash(
    stage: str, dataset: str, device: str, error: Exception
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "stage": stage,
                "dataset": dataset,
                "device": device,
                "input_hash_failure": f"{type(error).__name__}: {error}",
            }
        )
    ).hexdigest()


def _is_execution_error(checkpoint: Mapping[str, object]) -> int:
    return int(checkpoint.get("abstention_reason") == "stage_error")


def _write_checkpoint(path: Path, checkpoint: Mapping[str, object]) -> str:
    payload = dict(checkpoint)
    payload.pop("checkpoint_sha256", None)
    checkpoint_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload["checkpoint_sha256"] = checkpoint_hash
    write_json_atomic(path, payload)
    return checkpoint_hash


def _checkpoint_is_valid(path: Path, input_hash: str) -> bool:
    try:
        value = _read_checkpoint(path)
    except ValueError:
        return False
    stored = value.get("checkpoint_sha256")
    if not isinstance(stored, str) or len(stored) != 64 or value.get("input_sha256") != input_hash:
        return False
    canonical = dict(value)
    canonical.pop("checkpoint_sha256", None)
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest() == stored


def _read_checkpoint(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint cannot be read") from error
    if not isinstance(value, dict):
        raise ValueError("checkpoint must be an object")
    return value


def _write_status(output: Path, *, stage: str, current_dataset: str, completed: int, total: int, errors: int, device: str, checkpoint_sha256: str, newly_processed: int, started: float, status: str, run_id: str, run_started_unix: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    throughput = newly_processed / elapsed * 60.0 if elapsed > 0.0 else 0.0
    remaining = max(0, total - completed)
    eta = remaining / (newly_processed / elapsed) if newly_processed and elapsed > 0.0 else None
    write_json_atomic(
        output / STATUS_NAME,
        {
            "id": EXPERIMENT_ID,
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "run_started_unix": run_started_unix,
            "status": status,
            "stage": stage,
            "current_dataset": current_dataset,
            "completed": completed,
            "total": total,
            "newly_processed": newly_processed,
            "done": completed,
            "expected": total,
            "progress_percent": completed / total * 100.0,
            "eta_seconds": eta,
            "throughput_per_minute": throughput,
            "errors": errors,
            "device": device,
            "checkpoint_sha256": checkpoint_sha256,
            "status_checkpoint_sha256": checkpoint_sha256,
            "elapsed_seconds": elapsed,
            "updated_unix": time.time(),
        },
    )


def _checkpoint_input_hash(stage: str, dataset: str, device: str) -> str:
    return hashlib.sha256(canonical_json_bytes({"stage": stage, "dataset": dataset, "device": device, "pipeline": 1})).hexdigest()


def _safe_name(dataset: str) -> str:
    digest = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:12]
    return f"{''.join(char if char.isalnum() or char in '-_' else '_' for char in dataset)}-{digest}"


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Exp403-408 feature/ML validation")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")))
    args = parser.parse_args(argv)
    run_pipeline(stage=args.stage, output_dir=args.output_dir, limit=args.limit, resume=not args.no_resume, stop_after=args.stop_after, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
