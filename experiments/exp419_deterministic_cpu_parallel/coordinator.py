"""Fail-closed Exp419 checkpoint coordination and runtime qualification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from types import MappingProxyType
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetTerminalEvidence,
    FrozenRunContext,
    RunSummary,
)
from experiments.exp417_causal_multihorizon_shadow.runner import (
    apply_global_workload,
)

from .config import FROZEN_CONFIG
from .contracts import (
    ParallelRoots,
    ParallelRun,
    RuntimeBenchmark,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerBootstrap,
    WorkerCompletion,
    WorkerSpec,
)
from .workers import _context_sha256, spawn_workers


def _provenance_payload(provenance: object) -> object:
    to_payload = getattr(provenance, "to_payload", None)
    return to_payload() if callable(to_payload) else provenance


def merge_worker_checkpoints(
    context: FrozenRunContext,
    plan: ShardPlan,
    worker_stores: Mapping[int, Exp417CheckpointStore],
    canonical: Exp417CheckpointStore,
) -> Mapping[str, str]:
    """Import exact preregistered shard records in canonical dataset order."""
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    if not isinstance(plan, ShardPlan):
        raise ValueError("plan must be a ShardPlan")
    if plan.population != context.datasets:
        raise ValueError("shard plan does not match the frozen context")
    expected_indices = tuple(range(plan.worker_count))
    if (
        not isinstance(worker_stores, Mapping)
        or tuple(sorted(worker_stores)) != expected_indices
    ):
        raise ValueError("worker stores do not cover the exact shard plan")
    if getattr(canonical, "expected_datasets", None) != context.datasets:
        raise ValueError("canonical store does not match the frozen context")

    canonical_provenance = _provenance_payload(
        getattr(canonical, "provenance", None)
    )
    owner: dict[str, int] = {}
    source_digests: dict[int, Mapping[str, str]] = {}
    for assignment in plan.assignments:
        source = worker_stores[assignment.shard_index]
        if getattr(source, "expected_datasets", None) != assignment.datasets:
            raise ValueError("worker store inventory does not match its shard")
        if (
            _provenance_payload(getattr(source, "provenance", None))
            != canonical_provenance
        ):
            raise ValueError("worker store provenance mismatch")
        digests, completed = _validated_partial_store_state(source)
        if set(digests) != set(assignment.datasets):
            raise ValueError("worker store coverage does not match its shard")
        if any(
            isinstance(result.decision_evidence, DatasetTerminalEvidence)
            for result in completed.values()
        ):
            raise ValueError("worker store contains terminal dataset evidence")
        for dataset in digests:
            if dataset in owner:
                raise ValueError("duplicate dataset across worker stores")
            owner[dataset] = assignment.shard_index
        source_digests[assignment.shard_index] = digests
    if set(owner) != set(context.datasets):
        raise ValueError("worker stores do not cover the frozen context")

    existing, canonical_completed = _validated_partial_store_state(canonical)
    if set(existing) - set(context.datasets):
        raise ValueError("canonical store contains an extra dataset")
    if any(
        isinstance(result.decision_evidence, DatasetTerminalEvidence)
        for result in canonical_completed.values()
    ):
        raise ValueError("canonical store contains terminal dataset evidence")
    for dataset, digest in existing.items():
        source_digest = source_digests[owner[dataset]][dataset]
        if digest != source_digest:
            raise ValueError("canonical record digest differs from its shard")

    for dataset in context.datasets:
        if dataset in existing:
            continue
        source = worker_stores[owner[dataset]]
        imported = canonical.import_validated_dataset(source, dataset)
        if imported != source_digests[owner[dataset]][dataset]:
            raise ValueError("imported canonical digest differs from its shard")

    merged = canonical.completed_record_digests()
    if set(merged) != set(context.datasets):
        raise ValueError("canonical checkpoint is incomplete after merge")
    return MappingProxyType({name: merged[name] for name in context.datasets})


def _open_checkpoint_store(
    root: Path,
    provenance: object,
    datasets: tuple[str, ...],
) -> Exp417CheckpointStore:
    return Exp417CheckpointStore(root, provenance, datasets)


def _freeze_plan_binding(
    roots: ParallelRoots,
    plan: ShardPlan,
    context_sha256: str,
) -> None:
    if not isinstance(roots, ParallelRoots):
        raise ValueError("roots must be ParallelRoots")
    if tuple(sorted(roots.worker_roots)) != tuple(range(plan.worker_count)):
        raise ValueError("worker roots do not cover the exact shard plan")
    parent = roots.root.resolve(strict=False)
    children = {
        "canonical": roots.canonical_root.resolve(strict=False),
        **{
            f"worker-{index}": path.resolve(strict=False)
            for index, path in roots.worker_roots.items()
        },
    }
    if any(path == parent or not path.is_relative_to(parent) for path in children.values()):
        raise ValueError("checkpoint roots must be distinct children of the run root")
    child_paths = tuple(children.values())
    if len(child_paths) != len(set(child_paths)):
        raise ValueError("checkpoint roots must be distinct")
    for index, path in enumerate(child_paths):
        for other in child_paths[index + 1 :]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("checkpoint roots must not be nested")

    roots.root.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(
        {
            "canonical_root": str(children["canonical"]),
            "context_sha256": context_sha256,
            "plan_sha256": plan.sha256,
            "worker_roots": {
                str(index): str(
                    roots.worker_roots[index].resolve(strict=False)
                )
                for index in range(plan.worker_count)
            },
        }
    ) + b"\n"
    marker = roots.root / "exp419_shard_plan.json"
    try:
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        if marker.read_bytes() != payload:
            raise ValueError("frozen shard-plan binding mismatch")
        return
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_partial_store_state(
    store: object,
) -> tuple[Mapping[str, str], Mapping[str, object]]:
    first = store.completed_record_digests()
    completed = store.load_completed()
    second = store.completed_record_digests()
    if dict(first) != dict(second) or set(completed) != set(first):
        raise ValueError("checkpoint changed during coordinator validation")
    return first, completed


def _store_has_terminal_results(store: object) -> bool:
    _, completed = _validated_partial_store_state(store)
    return any(
        isinstance(result.decision_evidence, DatasetTerminalEvidence)
        for result in completed.values()
    )


def _reconstruct_global_summary(
    context: FrozenRunContext,
    canonical: object,
) -> RunSummary:
    completed, _ = canonical.require_complete_snapshot()
    results = tuple(completed[name] for name in context.datasets)
    total_test = sum(context.test_sizes.values())
    if any(
        isinstance(result.decision_evidence, DatasetTerminalEvidence)
        for result in results
    ):
        return RunSummary(results, total_test, False, "failed", context)
    summary = apply_global_workload(results, total_test, context=context)
    if summary.status == "global_workload_gate_failed":
        return summary
    if any(result.status == "failed" for result in results):
        return RunSummary(results, total_test, False, "failed", context)
    return summary


def _completion_records(
    plan: ShardPlan,
    digests: Mapping[str, str],
    metrics: Mapping[int, WorkerCompletion],
    *,
    error: str = "",
) -> tuple[WorkerCompletion, ...]:
    completions = []
    for assignment in plan.assignments:
        metric = metrics.get(assignment.shard_index)
        completions.append(
            WorkerCompletion(
                assignment.shard_index,
                os.getpid() if metric is None else metric.pid,
                assignment.datasets,
                {
                    name: digests[name]
                    for name in assignment.datasets
                },
                0.0 if metric is None else metric.elapsed_seconds,
                0 if metric is None else metric.peak_rss_bytes,
                {} if metric is None else metric.thread_environment,
                "failed" if error else "completed",
                error,
            )
        )
    return tuple(completions)


def _failed_parallel_run(plan: ShardPlan, error: BaseException | str) -> ParallelRun:
    message = str(error)
    completions = tuple(
        WorkerCompletion(
            assignment.shard_index,
            os.getpid(),
            (),
            {},
            0.0,
            0,
            {},
            "failed",
            message or type(error).__name__,
        )
        for assignment in plan.assignments
    )
    return ParallelRun(
        plan,
        completions,
        {},
        {"error": message or type(error).__name__, "plan_sha256": plan.sha256},
        False,
        "failed",
    )


def _validate_partial_worker_stores(
    plan: ShardPlan,
    stores: Mapping[int, object],
) -> dict[int, Mapping[str, str]]:
    seen: set[str] = set()
    observed: dict[int, Mapping[str, str]] = {}
    for assignment in plan.assignments:
        store = stores[assignment.shard_index]
        if getattr(store, "expected_datasets", None) != assignment.datasets:
            raise ValueError("worker store inventory does not match its shard")
        digests, completed = _validated_partial_store_state(store)
        if set(digests) - set(assignment.datasets):
            raise ValueError("worker store contains a wrong-shard record")
        if seen & set(digests):
            raise ValueError("duplicate dataset across worker stores")
        seen.update(digests)
        if any(
            isinstance(result.decision_evidence, DatasetTerminalEvidence)
            for result in completed.values()
        ):
            raise ValueError("worker store contains terminal dataset evidence")
        observed[assignment.shard_index] = digests
    return observed


def resume_or_run_parallel(
    context: FrozenRunContext,
    plan: ShardPlan,
    roots: ParallelRoots,
    bootstrap: WorkerBootstrap,
) -> ParallelRun:
    """Resume only original-shard gaps, validate, merge, and summarize."""
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    if not isinstance(plan, ShardPlan):
        raise ValueError("plan must be a ShardPlan")
    if not isinstance(bootstrap, WorkerBootstrap):
        raise ValueError("bootstrap must be a WorkerBootstrap")
    try:
        if plan.population != context.datasets:
            raise ValueError("shard plan does not match the frozen context")
        context_digest = _context_sha256(context)
        if bootstrap.expected_context_sha256 != context_digest:
            raise ValueError("bootstrap context digest mismatch")
        _freeze_plan_binding(roots, plan, context_digest)
        canonical = _open_checkpoint_store(
            roots.canonical_root,
            bootstrap.provenance,
            context.datasets,
        )
        canonical_digests, canonical_completed = (
            _validated_partial_store_state(canonical)
        )
        if any(
            isinstance(result.decision_evidence, DatasetTerminalEvidence)
            for result in canonical_completed.values()
        ):
            raise ValueError(
                "canonical checkpoint contains terminal dataset evidence"
            )
        if set(canonical_digests) == set(context.datasets):
            summary = _reconstruct_global_summary(context, canonical)
            error = "" if summary.publication_allowed else summary.status
            completions = _completion_records(
                plan, canonical_digests, {}, error=error
            )
            return ParallelRun(
                plan,
                completions,
                {
                    name: canonical_digests[name]
                    for assignment in plan.assignments
                    for name in assignment.datasets
                },
                {
                    "run_summary": summary,
                    "plan_sha256": plan.sha256,
                    "aggregate_peak_rss_bytes": 0,
                },
                summary.publication_allowed,
                "completed" if summary.publication_allowed else "failed",
            )
        if set(canonical_digests) - set(context.datasets):
            raise ValueError("canonical checkpoint contains an extra dataset")

        worker_stores = {
            assignment.shard_index: _open_checkpoint_store(
                roots.worker_roots[assignment.shard_index],
                bootstrap.provenance,
                assignment.datasets,
            )
            for assignment in plan.assignments
        }
        observed = _validate_partial_worker_stores(plan, worker_stores)
        specs = []
        attempt_token = uuid.uuid4().hex
        for assignment in plan.assignments:
            missing = tuple(
                name
                for name in assignment.datasets
                if name not in observed[assignment.shard_index]
            )
            if missing:
                specs.append(
                    WorkerSpec(
                        ShardAssignment(
                            assignment.shard_index,
                            missing,
                            assignment.total_cost,
                        ),
                        roots.root
                        / ".attempts"
                        / plan.sha256
                        / attempt_token
                        / f"worker-{assignment.shard_index}",
                    )
                )

        raw_completions = spawn_workers(tuple(specs), bootstrap) if specs else ()
        metrics = {
            completion.shard_index: completion
            for completion in raw_completions
        }
        if tuple(sorted(metrics)) != tuple(
            spec.shard.shard_index for spec in specs
        ):
            raise ValueError("worker completions do not cover scheduled shards")
        aggregate_rss = sum(
            completion.peak_rss_bytes for completion in raw_completions
        )
        if aggregate_rss > FROZEN_CONFIG.memory_limit_bytes:
            raise ValueError("aggregate worker RSS exceeds the frozen limit")

        for spec in specs:
            completion = metrics[spec.shard.shard_index]
            if completion.status != "completed":
                raise ValueError(
                    f"worker shard {spec.shard.shard_index} failed: "
                    f"{completion.error}"
                )
            source = _open_checkpoint_store(
                spec.checkpoint_root,
                bootstrap.provenance,
                spec.shard.datasets,
            )
            on_disk = source.completed_record_digests()
            if (
                completion.completed_datasets != spec.shard.datasets
                or dict(completion.record_digests) != dict(on_disk)
                or set(on_disk) != set(spec.shard.datasets)
            ):
                raise ValueError(
                    "worker completion does not match on-disk record digests"
                )
            if _store_has_terminal_results(source):
                raise ValueError("worker produced terminal dataset evidence")
            target = worker_stores[spec.shard.shard_index]
            for dataset in spec.shard.datasets:
                imported = target.import_validated_dataset(source, dataset)
                if imported != on_disk[dataset]:
                    raise ValueError(
                        "worker shard import digest mismatch"
                    )

        complete = _validate_partial_worker_stores(plan, worker_stores)
        if any(
            set(complete[assignment.shard_index])
            != set(assignment.datasets)
            for assignment in plan.assignments
        ):
            raise ValueError("worker stores remain incomplete after resume")
        merged = merge_worker_checkpoints(
            context, plan, worker_stores, canonical
        )
        summary = _reconstruct_global_summary(context, canonical)
        error = "" if summary.publication_allowed else summary.status
        completions = _completion_records(
            plan, merged, metrics, error=error
        )
        record_digests = {
            name: merged[name]
            for assignment in plan.assignments
            for name in assignment.datasets
        }
        return ParallelRun(
            plan,
            completions,
            record_digests,
            {
                "run_summary": summary,
                "plan_sha256": plan.sha256,
                "aggregate_peak_rss_bytes": aggregate_rss,
            },
            summary.publication_allowed,
            "completed" if summary.publication_allowed else "failed",
        )
    except Exception as error:
        return _failed_parallel_run(plan, error)


def select_runtime(trials: Sequence[RuntimeTrial]) -> RuntimeBenchmark:
    """Select the fastest exact parallel trial passing every frozen gate."""
    frozen = tuple(trials)
    if not frozen:
        raise ValueError("trials must not be empty")
    benchmark_datasets = frozen[0].completed_datasets
    baseline = frozen[0]
    baseline_valid = (
        baseline.worker_count == 1
        and baseline.elapsed_seconds > 0.0
        and baseline.parity
        and baseline.finite
        and not baseline.fallback
        and baseline.peak_rss_bytes <= FROZEN_CONFIG.memory_limit_bytes
    )
    qualified = []
    if baseline_valid:
        for trial in frozen[1:]:
            if (
                trial.worker_count > 1
                and trial.elapsed_seconds > 0.0
                and trial.parity
                and trial.finite
                and not trial.fallback
                and trial.peak_rss_bytes
                <= FROZEN_CONFIG.memory_limit_bytes
                and baseline.elapsed_seconds / trial.elapsed_seconds
                >= FROZEN_CONFIG.speedup_gate
            ):
                qualified.append(trial)
    selected = (
        min(
            qualified,
            key=lambda trial: (
                trial.elapsed_seconds,
                trial.worker_count,
            ),
        ).worker_count
        if qualified
        else None
    )
    return RuntimeBenchmark(
        frozen,
        selected,
        (
            "qualified_parallel_runtime"
            if selected is not None
            else "no_qualified_parallel_runtime"
        ),
        benchmark_datasets,
    )


__all__ = [
    "merge_worker_checkpoints",
    "resume_or_run_parallel",
    "select_runtime",
]
