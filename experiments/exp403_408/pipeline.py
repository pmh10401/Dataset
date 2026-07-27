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
import time
from typing import Any

from .artifact_io import canonical_json_bytes, write_json_atomic
from .contracts import FrozenConfig
from .feature_registry import load_feature_blocks
from .feature_stability import audit_feature_stability, build_group_summary
from .foundation import crossfit_plan, list_datasets_small_to_large, load_dataset_input
from .prediction import predict_dataset
from .synthetic_anomaly import SyntheticAnomalyConfig, score_synthetic_models, synthetic_shadow_predictions
from .tri_view_ssl import score_tri_view, shadow_predictions


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
STATUS_NAME = "status.json"


@dataclass(frozen=True)
class StageResult:
    stage: str
    completed: int
    total: int
    recomputed_datasets: int
    errors: int
    output_dir: Path


Processor = Callable[[str], Mapping[str, object]]


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

    results: list[StageResult] = []
    for stage_name in selected:
        stage_workers = stage_worker_count(stage_name, workers)
        results.append(
            run_stage(
                stage_name,
                datasets,
                Path(output_dir),
                processor=_processor_for(stage_name),
                stop_after=stop_after,
                resume=resume,
                workers=stage_workers,
                device="cpu",
            )
        )
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
) -> StageResult:
    """Execute one stage with atomic per-dataset result or abstention records."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    ordered = tuple(dict.fromkeys(str(dataset) for dataset in datasets))
    if not ordered:
        raise ValueError("stage requires at least one dataset")
    if stop_after is not None and stop_after < 0:
        raise ValueError("stop_after must be non-negative")
    output = Path(output_dir)
    checkpoint_dir = output / "checkpoints" / stage
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(ordered)
    started = time.monotonic()
    completed = 0
    errors = 0
    recomputed = 0
    last_hash = ""
    pending: list[str] = []
    for dataset in ordered:
        checkpoint = checkpoint_dir / f"{_safe_name(dataset)}.json"
        input_hash = _checkpoint_input_hash(stage, dataset, device)
        if resume and _checkpoint_is_valid(checkpoint, input_hash):
            completed += 1
            existing = _read_checkpoint(checkpoint)
            errors += _is_execution_error(existing)
            last_hash = str(existing["checkpoint_sha256"])
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
        started=started,
        status="running",
    )

    if workers > 1 and len(pending) > 1:
        # Parent process serializes checkpoints and status.  Workers only run
        # CPU-only dataset processors, which keeps writes and progress atomic.
        with ProcessPoolExecutor(max_workers=min(7, int(workers))) as executor:
            futures = {executor.submit(_run_processor, processor, dataset): dataset for dataset in pending}
            for future in as_completed(futures):
                dataset = futures[future]
                checkpoint = _checkpoint_from_future(stage, dataset, device, future)
                last_hash = _write_checkpoint(checkpoint_dir / f"{_safe_name(dataset)}.json", checkpoint)
                completed += 1
                recomputed += 1
                errors += _is_execution_error(checkpoint)
                _write_status(output, stage=stage, current_dataset=dataset, completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, started=started, status="running")
    else:
        for dataset in pending:
            if stop_after is not None and recomputed >= stop_after:
                break
            checkpoint = _checkpoint_from_call(stage, dataset, device, processor)
            last_hash = _write_checkpoint(checkpoint_dir / f"{_safe_name(dataset)}.json", checkpoint)
            completed += 1
            recomputed += 1
            errors += _is_execution_error(checkpoint)
            _write_status(output, stage=stage, current_dataset=dataset, completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, started=started, status="running")

    final_status = "completed" if completed == total else "running"
    _write_status(output, stage=stage, current_dataset=ordered[min(completed, total) - 1] if completed else ordered[0], completed=completed, total=total, errors=errors, device=device, checkpoint_sha256=last_hash, started=started, status=final_status)
    return StageResult(stage, completed, total, recomputed, errors, output)


def _processor_for(stage: str) -> Processor:
    processors: dict[str, Processor] = {
        "audit-features": _audit_features,
        "classical-predict": _classical_predict,
        "classical-evaluate": _evaluation_abstention,
        "ssl-predict": _ssl_predict,
        "synthetic-predict": _synthetic_predict,
        "consensus-evaluate": _consensus_abstention,
        "report": _report_ready,
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
        "audit": _json_value(audit),
        "group_summary": _json_value(build_group_summary(audit)),
    }


def _classical_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    rows = predict_dataset(bundle, FrozenConfig(target_length=bundle.target_length))
    return {"dataset": dataset, "outcome": "completed", "prediction_rows": _json_value(rows)}


def _evaluation_abstention(dataset: str) -> Mapping[str, object]:
    return _abstention(dataset, "requires_hash_verified_full_prediction_and_separate_label_loader")


def _ssl_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    scores = score_tri_view(bundle.train_normal, bundle.test_values, FrozenConfig(target_length=bundle.target_length), device="cpu")
    return {
        "dataset": dataset,
        "outcome": "completed" if scores.available else "abstained",
        "abstention_reason": scores.abstention_reason,
        "scores": _json_value(scores),
        "shadow_predictions": _json_value(shadow_predictions(dataset, scores, bundle.hard_indices)),
    }


def _synthetic_predict(dataset: str) -> Mapping[str, object]:
    bundle = load_dataset_input(dataset)
    config = SyntheticAnomalyConfig(frozen=FrozenConfig(target_length=bundle.target_length))
    scores = score_synthetic_models(bundle.train_normal, bundle.test_values, config)
    return {
        "dataset": dataset,
        "outcome": "completed" if all(score.available for score in scores) else "abstained",
        "abstention_reason": next((score.abstention_reason for score in scores if not score.available), ""),
        "scores": _json_value(scores),
        "shadow_predictions": _json_value({score.model_name: synthetic_shadow_predictions(dataset, score, bundle.hard_indices) for score in scores}),
    }


def _consensus_abstention(dataset: str) -> Mapping[str, object]:
    return _abstention(dataset, "requires_aligned_train_oof_source_artifacts")


def _report_ready(dataset: str) -> Mapping[str, object]:
    return {"dataset": dataset, "outcome": "completed", "report_scope": "checkpoint_ledger_only"}


def _abstention(dataset: str, reason: str) -> Mapping[str, object]:
    return {"dataset": dataset, "outcome": "abstained", "abstention_reason": reason}


def _run_processor(processor: Processor, dataset: str) -> Mapping[str, object]:
    return processor(dataset)


def _checkpoint_from_call(stage: str, dataset: str, device: str, processor: Processor) -> dict[str, object]:
    try:
        payload = processor(dataset)
    except Exception as error:  # An error is a durable, explicit abstention.
        return _error_checkpoint(stage, dataset, device, error)
    return _success_checkpoint(stage, dataset, device, payload)


def _checkpoint_from_future(stage: str, dataset: str, device: str, future: object) -> dict[str, object]:
    try:
        payload = future.result()
    except Exception as error:
        return _error_checkpoint(stage, dataset, device, error)
    return _success_checkpoint(stage, dataset, device, payload)


def _success_checkpoint(stage: str, dataset: str, device: str, payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return _error_checkpoint(stage, dataset, device, ValueError("stage processor must return a mapping"))
    result = dict(payload)
    result["dataset"] = dataset
    result.setdefault("outcome", "completed")
    if result["outcome"] not in {"completed", "abstained"}:
        return _error_checkpoint(stage, dataset, device, ValueError("stage outcome is invalid"))
    if result["outcome"] == "abstained" and not result.get("abstention_reason"):
        return _error_checkpoint(stage, dataset, device, ValueError("abstention requires a reason"))
    return {"stage": stage, "dataset": dataset, "device": device, "input_sha256": _checkpoint_input_hash(stage, dataset, device), **_json_value(result)}


def _error_checkpoint(stage: str, dataset: str, device: str, error: Exception) -> dict[str, object]:
    return {
        "stage": stage,
        "dataset": dataset,
        "device": device,
        "input_sha256": _checkpoint_input_hash(stage, dataset, device),
        "outcome": "abstained",
        "abstention_reason": "stage_error",
        "error": f"{type(error).__name__}: {error}",
    }


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


def _write_status(output: Path, *, stage: str, current_dataset: str, completed: int, total: int, errors: int, device: str, checkpoint_sha256: str, started: float, status: str) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    throughput = completed / elapsed * 60.0 if elapsed > 0.0 else 0.0
    remaining = max(0, total - completed)
    eta = remaining / (completed / elapsed) if completed and elapsed > 0.0 else None
    write_json_atomic(
        output / STATUS_NAME,
        {
            "id": EXPERIMENT_ID,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "stage": stage,
            "current_dataset": current_dataset,
            "completed": completed,
            "total": total,
            "done": completed,
            "expected": total,
            "progress_percent": completed / total * 100.0,
            "eta_seconds": eta,
            "throughput_per_minute": throughput,
            "errors": errors,
            "device": device,
            "checkpoint_sha256": checkpoint_sha256,
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
