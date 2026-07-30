"""Exact, label-free reporting for verified Exp419 parallel evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.backend import (
    FROZEN_TASK7_BACKEND_AUDIT_SHA256,
)
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import RunSummary
from experiments.exp417_causal_multihorizon_shadow.reporting import (
    BACKEND_FILE as EXP417_BACKEND_FILE,
    CONFIG_FILE as EXP417_CONFIG_FILE,
    DATASET_FILE as EXP417_DATASET_FILE,
    OOF_FILE as EXP417_OOF_FILE,
    PREDICTION_FILE as EXP417_PREDICTION_FILE,
    REPORT_FILE as EXP417_REPORT_FILE,
    SYNTHETIC_FILE as EXP417_SYNTHETIC_FILE,
    TrustedCheckpointProvenance,
    render_exp417_payloads,
    verify_exp417_payloads,
)

from .config import FROZEN_CONFIG
from .coordinator import select_runtime
from .contracts import (
    ParallelRun,
    RuntimeBenchmark,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerCompletion,
)
from .workers import _THREAD_ENVIRONMENT


PREDICTION_FILE = "exp419_shadow_predictions.jsonl"
DATASET_FILE = "exp419_dataset_qualification.csv"
OOF_FILE = "exp419_oof_normal_metrics.csv"
SYNTHETIC_FILE = "exp419_synthetic_safety.csv"
BACKEND_FILE = "exp419_backend_benchmark.json"
RUNTIME_BENCHMARK_FILE = "exp419_parallel_runtime_benchmark.json"
CONFIG_FILE = "exp419_frozen_config.json"
SHARD_PLAN_FILE = "exp419_shard_plan.json"
REPORT_FILE = "EXP419_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"
OFFICIAL_FILES = (
    PREDICTION_FILE,
    DATASET_FILE,
    OOF_FILE,
    SYNTHETIC_FILE,
    BACKEND_FILE,
    RUNTIME_BENCHMARK_FILE,
    CONFIG_FILE,
    SHARD_PLAN_FILE,
    REPORT_FILE,
    MANIFEST_FILE,
)
NON_MANIFEST_FILES = OFFICIAL_FILES[:-1]

_EXP417_TO_EXP419 = {
    EXP417_PREDICTION_FILE: PREDICTION_FILE,
    EXP417_DATASET_FILE: DATASET_FILE,
    EXP417_OOF_FILE: OOF_FILE,
    EXP417_SYNTHETIC_FILE: SYNTHETIC_FILE,
    EXP417_BACKEND_FILE: BACKEND_FILE,
}
_FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "label",
        "labels",
        "labels_blob",
        "confusion_matrix",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "mean_combined_f1",
        "actual_anomaly_location",
        "human_review",
        "human_reviewed_result",
    }
)
_SOURCE_GROUPS = {
    "preprocessing_contract_sha256": (
        "experiments/exp417_causal_multihorizon_shadow/preprocessing.py",
    ),
    "fold_contract_sha256": (
        "experiments/exp417_causal_multihorizon_shadow/preprocessing.py",
        "experiments/exp417_causal_multihorizon_shadow/model.py",
    ),
    "model_contract_sha256": (
        "experiments/exp417_causal_multihorizon_shadow/model.py",
        "experiments/exp417_causal_multihorizon_shadow/scoring.py",
        "experiments/exp417_causal_multihorizon_shadow/qualification.py",
        "experiments/exp417_causal_multihorizon_shadow/runner.py",
    ),
}
_EXP419_SOURCE_PATHS = (
    "experiments/exp419_deterministic_cpu_parallel/planner.py",
    "experiments/exp419_deterministic_cpu_parallel/workers.py",
    "experiments/exp419_deterministic_cpu_parallel/coordinator.py",
    "experiments/exp419_deterministic_cpu_parallel/reporting.py",
    "experiments/exp419_deterministic_cpu_parallel/publication.py",
)
_EXP419_SOURCE_ROLES = {
    "scheduler": _EXP419_SOURCE_PATHS[0],
    "worker": _EXP419_SOURCE_PATHS[1],
    "merge": _EXP419_SOURCE_PATHS[2],
    "reporting": _EXP419_SOURCE_PATHS[3],
    "publication": _EXP419_SOURCE_PATHS[4],
}


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _source_digest(relative_paths: tuple[str, ...]) -> str:
    root = Path(__file__).resolve().parents[2]
    digest = sha256()
    for relative_path in relative_paths:
        payload = (root / relative_path).read_bytes()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _exp419_source_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative_path: sha256((root / relative_path).read_bytes()).hexdigest()
        for relative_path in _EXP419_SOURCE_PATHS
    }


def _derive_exp417_provenance(summary: RunSummary) -> Exp417CheckpointProvenance:
    """Recreate the official Exp417 source-bound provenance from frozen inputs."""
    if not isinstance(summary, RunSummary):
        raise ValueError("Exp419 scientific evidence requires a RunSummary")
    context = summary.context
    lanes = context.lanes
    provenance = Exp417CheckpointProvenance(
        git_commit=context.code_commit,
        database_sha256=context.database_sha256,
        exp291_prediction_sha256=lanes.exp291_prediction_sha256,
        exp291_manifest_sha256=lanes.exp291_manifest_sha256,
        exp292_prediction_sha256=lanes.exp292_prediction_sha256,
        exp292_manifest_sha256=lanes.exp292_manifest_sha256,
        frozen_config_sha256=sha256(
            canonical_json_bytes(asdict(FROZEN_CONFIG.exp417_config))
        ).hexdigest(),
        preprocessing_contract_sha256=_source_digest(
            _SOURCE_GROUPS["preprocessing_contract_sha256"]
        ),
        fold_contract_sha256=_source_digest(
            _SOURCE_GROUPS["fold_contract_sha256"]
        ),
        model_contract_sha256=_source_digest(
            _SOURCE_GROUPS["model_contract_sha256"]
        ),
        model_seeds=FROZEN_CONFIG.exp417_config.model_seeds,
        perturbation_seeds=FROZEN_CONFIG.exp417_config.perturbation_seeds,
        runtime_versions=dict(context.runtime_versions),
        backend_contract_sha256=FROZEN_TASK7_BACKEND_AUDIT_SHA256,
    )
    return provenance


def _plan_payload(plan: ShardPlan) -> dict[str, object]:
    return {
        "assignments": [
            {
                "datasets": list(assignment.datasets),
                "shard_index": assignment.shard_index,
                "total_cost": assignment.total_cost,
            }
            for assignment in plan.assignments
        ],
        "population": list(plan.population),
        "sha256": plan.sha256,
        "worker_count": plan.worker_count,
    }


def _trial_payload(trial: RuntimeTrial) -> dict[str, object]:
    return {
        "completed_datasets": list(trial.completed_datasets),
        "elapsed_seconds": trial.elapsed_seconds,
        "evidence_sha256": trial.evidence_sha256,
        "fallback": trial.fallback,
        "finite": trial.finite,
        "parity": trial.parity,
        "peak_rss_bytes": trial.peak_rss_bytes,
        "worker_count": trial.worker_count,
    }


def _benchmark_payload(benchmark: RuntimeBenchmark) -> dict[str, object]:
    return {
        "benchmark_datasets": list(benchmark.benchmark_datasets),
        "selected_worker_count": benchmark.selected_worker_count,
        "sha256": benchmark.sha256,
        "status": benchmark.status,
        "trials": [_trial_payload(trial) for trial in benchmark.trials],
    }


def _completion_payload(completion: WorkerCompletion) -> dict[str, object]:
    return {
        "completed_datasets": list(completion.completed_datasets),
        "elapsed_seconds": completion.elapsed_seconds,
        "error": completion.error,
        "peak_rss_bytes": completion.peak_rss_bytes,
        "pid": completion.pid,
        "record_digests": dict(completion.record_digests),
        "shard_index": completion.shard_index,
        "status": completion.status,
        "thread_environment": dict(completion.thread_environment),
    }


def _runtime_config_payload() -> dict[str, object]:
    return json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))


def _render_report(
    scientific: Mapping[str, object],
    benchmark: RuntimeBenchmark,
    plan: ShardPlan,
    aggregate_peak_rss_bytes: int,
) -> bytes:
    run = scientific["run"]
    return (
        "# Exp419 Deterministic CPU Parallel Runtime 결과 보고서\n\n"
        "## 거버넌스\n\n"
        "- 이 결과는 Shadow-only 검토 근거다.\n"
        "- 전체 후보 생성과 검증은 label-free이며 TEST label을 사용하지 않았다.\n"
        "- no lane promotion (기존 Hard/Standard/Priority/automatic-normal lane 승격 없음).\n"
        "- 자동 경보 승격, tail replacement, top-N trimming을 수행하지 않았다.\n\n"
        "## 병렬 실행 근거\n\n"
        f"- 데이터셋 수: {run['dataset_count']}\n"
        f"- 공식 Shadow 후보 수: {run['published_candidate_count']}\n"
        f"- 선택 worker 수: {benchmark.selected_worker_count}\n"
        f"- shard plan SHA256: {plan.sha256}\n"
        f"- benchmark SHA256: {benchmark.sha256}\n"
        f"- aggregate peak RSS bytes: {aggregate_peak_rss_bytes}\n"
    ).encode("utf-8")


def _trusted_map_digest(values: Mapping[str, str]) -> str:
    return sha256(canonical_json_bytes(dict(values))).hexdigest()


def _manifest_payload(
    payloads: Mapping[str, bytes],
    *,
    scientific: Mapping[str, object],
    run: ParallelRun,
    benchmark: RuntimeBenchmark,
    plan: ShardPlan,
    trusted_checkpoint: TrustedCheckpointProvenance,
    aggregate_peak_rss_bytes: int,
) -> dict[str, object]:
    return {
        "aggregate_peak_rss_bytes": aggregate_peak_rss_bytes,
        "artifact_sha256": {
            name: sha256(payloads[name]).hexdigest()
            for name in NON_MANIFEST_FILES
        },
        "benchmark_sha256": benchmark.sha256,
        "canonical_checkpoint_record_sha256": dict(
            trusted_checkpoint.checkpoint_payload_sha256
        ),
        "canonical_checkpoint_record_map_sha256": _trusted_map_digest(
            trusted_checkpoint.checkpoint_payload_sha256
        ),
        "experiment_id": "Exp419",
        "exp419_source_roles": dict(_EXP419_SOURCE_ROLES),
        "exp419_source_sha256": _exp419_source_digests(),
        "git_commit": scientific["context"]["code_commit"],
        "runtime_versions": scientific["context"]["runtime_versions"],
        "selected_worker_count": benchmark.selected_worker_count,
        "shard_plan_sha256": plan.sha256,
        "shard_records": [
            {
                "datasets": list(completion.completed_datasets),
                "record_digests": dict(completion.record_digests),
                "shard_index": completion.shard_index,
            }
            for completion in run.completions
        ],
        "start_method": FROZEN_CONFIG.start_method,
        "thread_environment": [
            {
                "shard_index": completion.shard_index,
                "values": dict(completion.thread_environment),
            }
            for completion in run.completions
        ],
        "trusted_checkpoint_provenance_sha256":
            trusted_checkpoint.checkpoint_provenance_sha256,
        "trusted_result_evidence_sha256": dict(
            trusted_checkpoint.result_evidence_sha256
        ),
        "trusted_result_map_sha256": _trusted_map_digest(
            trusted_checkpoint.result_evidence_sha256
        ),
    }


def _verify_runtime_qualification(benchmark: RuntimeBenchmark) -> None:
    recomputed = select_runtime(benchmark.trials)
    if recomputed != benchmark:
        raise ValueError(
            "Exp419 runtime qualification differs from recomputed selection"
        )


def _verify_completion_runtime_evidence(
    completions: tuple[WorkerCompletion, ...],
    aggregate_peak_rss_bytes: object,
) -> int:
    if any(
        dict(completion.thread_environment) != dict(_THREAD_ENVIRONMENT)
        for completion in completions
    ):
        raise ValueError("Exp419 worker thread environment is not frozen")
    worker_peak_sum = sum(
        completion.peak_rss_bytes for completion in completions
    )
    if (
        type(aggregate_peak_rss_bytes) is not int
        or aggregate_peak_rss_bytes <= 0
        or aggregate_peak_rss_bytes < worker_peak_sum
        or aggregate_peak_rss_bytes > FROZEN_CONFIG.memory_limit_bytes
    ):
        raise ValueError(
            "Exp419 aggregate peak RSS is inconsistent with worker evidence"
        )
    return aggregate_peak_rss_bytes


def render_exp419_payloads(
    run: ParallelRun,
    benchmark: RuntimeBenchmark,
    plan: ShardPlan,
    trusted_checkpoint: TrustedCheckpointProvenance,
) -> Mapping[str, bytes]:
    """Compose Exp417 science and Exp419 runtime into exactly ten payloads."""
    FROZEN_CONFIG.validate()
    if not isinstance(run, ParallelRun) or run.status != "completed":
        raise ValueError("Exp419 rendering requires a completed ParallelRun")
    if not run.publication_allowed:
        raise ValueError("Exp419 rendering requires publication approval")
    if not isinstance(plan, ShardPlan) or run.plan != plan:
        raise ValueError("Exp419 run and shard plan mismatch")
    if (
        not isinstance(benchmark, RuntimeBenchmark)
        or benchmark.status != "qualified_parallel_runtime"
        or benchmark.selected_worker_count != plan.worker_count
    ):
        raise ValueError("Exp419 runtime benchmark does not authorize this plan")
    _verify_runtime_qualification(benchmark)
    if not isinstance(trusted_checkpoint, TrustedCheckpointProvenance):
        raise ValueError("Exp419 requires trusted canonical checkpoint provenance")
    if dict(run.record_digests) != dict(
        trusted_checkpoint.checkpoint_payload_sha256
    ):
        raise ValueError("Exp419 run differs from the trusted canonical checkpoint")
    summary = run.summary.get("run_summary")
    if not isinstance(summary, RunSummary):
        raise ValueError("Exp419 run lacks the verified Exp417 RunSummary")
    if summary.context.datasets != plan.population:
        raise ValueError("Exp419 population differs from the scientific context")
    if not summary.publication_allowed or summary.status != "complete":
        raise ValueError("Exp417 scientific publication gate is not complete")
    aggregate_rss = _verify_completion_runtime_evidence(
        run.completions,
        run.summary.get("aggregate_peak_rss_bytes"),
    )

    provenance = _derive_exp417_provenance(summary)
    exp417 = dict(
        render_exp417_payloads(
            summary,
            provenance,
            trusted_checkpoint,
            config=FROZEN_CONFIG.exp417_config,
        )
    )
    scientific = json.loads(exp417[EXP417_CONFIG_FILE])
    payloads: dict[str, bytes] = {
        target: exp417[source]
        for source, target in _EXP417_TO_EXP419.items()
    }
    payloads[RUNTIME_BENCHMARK_FILE] = _json_bytes(
        _benchmark_payload(benchmark)
    )
    payloads[CONFIG_FILE] = _json_bytes(
        {
            "exp417_scientific_evidence": scientific,
            "exp417_scientific_report":
                exp417[EXP417_REPORT_FILE].decode("utf-8"),
            "experiment_id": "Exp419",
            "parallel_run": {
                "aggregate_peak_rss_bytes": aggregate_rss,
                "completions": [
                    _completion_payload(completion)
                    for completion in run.completions
                ],
                "plan_sha256": plan.sha256,
                "publication_allowed": run.publication_allowed,
                "record_digests": dict(run.record_digests),
                "status": run.status,
            },
            "runtime_config": _runtime_config_payload(),
            "runtime_config_sha256": sha256(
                canonical_json_bytes(_runtime_config_payload())
            ).hexdigest(),
        }
    )
    payloads[SHARD_PLAN_FILE] = _json_bytes(_plan_payload(plan))
    payloads[REPORT_FILE] = _render_report(
        scientific, benchmark, plan, aggregate_rss
    )
    ordered = {name: payloads[name] for name in NON_MANIFEST_FILES}
    ordered[MANIFEST_FILE] = _json_bytes(
        _manifest_payload(
            ordered,
            scientific=scientific,
            run=run,
            benchmark=benchmark,
            plan=plan,
            trusted_checkpoint=trusted_checkpoint,
            aggregate_peak_rss_bytes=aggregate_rss,
        )
    )
    verify_exp419_payloads(
        ordered, trusted_checkpoint=trusted_checkpoint
    )
    return MappingProxyType(ordered)


def _canonical_object(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise ValueError(f"{name} must be bytes")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict) or payload != _json_bytes(value):
        raise ValueError(f"{name} is not a canonical JSON object")
    return value


def _reconstruct_plan(value: Mapping[str, object]) -> ShardPlan:
    try:
        assignments = tuple(
            ShardAssignment(
                row["shard_index"],
                tuple(row["datasets"]),
                row["total_cost"],
            )
            for row in value["assignments"]
        )
        plan = ShardPlan(
            value["worker_count"],
            assignments,
            tuple(value["population"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp419 shard plan is invalid") from error
    if value != _plan_payload(plan):
        raise ValueError("Exp419 shard plan is not exact")
    return plan


def _reconstruct_benchmark(value: Mapping[str, object]) -> RuntimeBenchmark:
    try:
        trials = tuple(
            RuntimeTrial(
                row["worker_count"],
                row["elapsed_seconds"],
                row["peak_rss_bytes"],
                row["parity"],
                row["finite"],
                row["fallback"],
                tuple(row["completed_datasets"]),
                row["evidence_sha256"],
            )
            for row in value["trials"]
        )
        benchmark = RuntimeBenchmark(
            trials,
            value["selected_worker_count"],
            value["status"],
            tuple(value["benchmark_datasets"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp419 runtime benchmark is invalid") from error
    if value != _benchmark_payload(benchmark):
        raise ValueError("Exp419 runtime benchmark is not exact")
    return benchmark


def _reconstruct_completions(
    values: object,
) -> tuple[WorkerCompletion, ...]:
    if not isinstance(values, list):
        raise ValueError("Exp419 completions must be a list")
    try:
        completions = tuple(
            WorkerCompletion(
                row["shard_index"],
                row["pid"],
                tuple(row["completed_datasets"]),
                row["record_digests"],
                row["elapsed_seconds"],
                row["peak_rss_bytes"],
                row["thread_environment"],
                row["status"],
                row["error"],
            )
            for row in values
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp419 completion evidence is invalid") from error
    if values != [_completion_payload(item) for item in completions]:
        raise ValueError("Exp419 completion evidence is not exact")
    return completions


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_KEYS:
                raise ValueError(f"Exp419 forbidden evaluation field: {key}")
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def verify_exp419_payloads(
    payloads: Mapping[str, bytes],
    *,
    trusted_checkpoint: TrustedCheckpointProvenance,
) -> None:
    """Reconstruct and cross-bind every scientific and runtime artifact."""
    FROZEN_CONFIG.validate()
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != set(OFFICIAL_FILES)
        or len(payloads) != len(OFFICIAL_FILES)
        or any(type(name) is not str or type(value) is not bytes
               for name, value in payloads.items())
    ):
        raise ValueError("Exp419 reporting requires the exact ordered ten-file set")
    if not isinstance(trusted_checkpoint, TrustedCheckpointProvenance):
        raise ValueError("Exp419 verification requires trusted checkpoint provenance")

    config = _canonical_object(payloads[CONFIG_FILE], CONFIG_FILE)
    if set(config) != {
        "exp417_scientific_evidence",
        "exp417_scientific_report",
        "experiment_id",
        "parallel_run",
        "runtime_config",
        "runtime_config_sha256",
    } or config["experiment_id"] != "Exp419":
        raise ValueError("Exp419 frozen config schema mismatch")
    runtime_config = config["runtime_config"]
    if runtime_config != _runtime_config_payload():
        raise ValueError("Exp419 runtime config mismatch")
    if config["runtime_config_sha256"] != sha256(
        canonical_json_bytes(runtime_config)
    ).hexdigest():
        raise ValueError("Exp419 runtime config digest mismatch")
    scientific = config["exp417_scientific_evidence"]
    if not isinstance(scientific, dict):
        raise ValueError("Exp419 scientific evidence is invalid")
    scientific_report = config["exp417_scientific_report"]
    if not isinstance(scientific_report, str):
        raise ValueError("Exp419 scientific report binding is invalid")
    exp417_payloads = {
        source: payloads[target]
        for source, target in _EXP417_TO_EXP419.items()
    }
    exp417_payloads[EXP417_CONFIG_FILE] = _json_bytes(scientific)
    exp417_payloads[EXP417_REPORT_FILE] = scientific_report.encode("utf-8")
    verify_exp417_payloads(
        exp417_payloads,
        trusted_checkpoint=trusted_checkpoint,
        config=FROZEN_CONFIG.exp417_config,
    )
    for line in payloads[PREDICTION_FILE].splitlines():
        try:
            prediction = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp419 prediction JSONL is invalid") from error
        _reject_forbidden_keys(prediction)

    plan = _reconstruct_plan(
        _canonical_object(payloads[SHARD_PLAN_FILE], SHARD_PLAN_FILE)
    )
    benchmark = _reconstruct_benchmark(
        _canonical_object(
            payloads[RUNTIME_BENCHMARK_FILE], RUNTIME_BENCHMARK_FILE
        )
    )
    _verify_runtime_qualification(benchmark)
    if (
        benchmark.status != "qualified_parallel_runtime"
        or benchmark.selected_worker_count != plan.worker_count
    ):
        raise ValueError("Exp419 benchmark does not authorize the shard plan")
    context = scientific.get("context")
    if (
        not isinstance(context, dict)
        or tuple(context.get("datasets", ())) != plan.population
    ):
        raise ValueError("Exp419 shard population differs from scientific evidence")

    parallel_run = config["parallel_run"]
    if not isinstance(parallel_run, dict) or set(parallel_run) != {
        "aggregate_peak_rss_bytes",
        "completions",
        "plan_sha256",
        "publication_allowed",
        "record_digests",
        "status",
    }:
        raise ValueError("Exp419 parallel run schema mismatch")
    if (
        parallel_run["status"] != "completed"
        or parallel_run["publication_allowed"] is not True
        or parallel_run["plan_sha256"] != plan.sha256
        or parallel_run["record_digests"]
        != dict(trusted_checkpoint.checkpoint_payload_sha256)
    ):
        raise ValueError("Exp419 parallel run binding mismatch")
    completions = _reconstruct_completions(parallel_run["completions"])
    if len(completions) != plan.worker_count:
        raise ValueError("Exp419 completions do not cover the shard plan")
    for completion, assignment in zip(
        completions, plan.assignments, strict=True
    ):
        if (
            completion.shard_index != assignment.shard_index
            or completion.completed_datasets != assignment.datasets
            or dict(completion.record_digests)
            != {
                name: trusted_checkpoint.checkpoint_payload_sha256[name]
                for name in assignment.datasets
            }
        ):
            raise ValueError("Exp419 shard record binding mismatch")
    aggregate_peak_rss_bytes = _verify_completion_runtime_evidence(
        completions,
        parallel_run["aggregate_peak_rss_bytes"],
    )

    expected_report = _render_report(
        scientific,
        benchmark,
        plan,
        aggregate_peak_rss_bytes,
    )
    if payloads[REPORT_FILE] != expected_report:
        raise ValueError("Exp419 report differs from verified evidence")
    expected_manifest = _manifest_payload(
        payloads,
        scientific=scientific,
        run=ParallelRun(
            plan,
            completions,
            parallel_run["record_digests"],
            {"verified": True},
            True,
            "completed",
        ),
        benchmark=benchmark,
        plan=plan,
        trusted_checkpoint=trusted_checkpoint,
        aggregate_peak_rss_bytes=aggregate_peak_rss_bytes,
    )
    manifest = _canonical_object(payloads[MANIFEST_FILE], MANIFEST_FILE)
    if manifest != expected_manifest:
        raise ValueError("Exp419 manifest binding mismatch")
