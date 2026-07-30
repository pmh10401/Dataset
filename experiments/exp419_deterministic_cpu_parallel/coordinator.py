"""Fail-closed Exp419 checkpoint coordination and runtime qualification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from time import monotonic
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
from .workers import (
    _THREAD_ENVIRONMENT,
    _casefolded_path_parts,
    _context_sha256,
    _filesystem_identity,
    _parts_are_nested,
    _peak_rss_bytes,
    spawn_workers,
)


_BINDING_FILE = "exp419_shard_plan.json"
_EVIDENCE_FILE = "exp419_runtime_evidence.json"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


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
    context: FrozenRunContext,
) -> str:
    if not isinstance(roots, ParallelRoots):
        raise ValueError("roots must be ParallelRoots")
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be FrozenRunContext")
    if tuple(sorted(roots.worker_roots)) != tuple(range(plan.worker_count)):
        raise ValueError("worker roots do not cover the exact shard plan")
    context_sha256 = _context_sha256(context)
    parent = roots.root.resolve(strict=False)
    children = {
        "canonical": roots.canonical_root.resolve(strict=False),
        **{
            f"worker-{index}": path.resolve(strict=False)
            for index, path in roots.worker_roots.items()
        },
    }
    if any(
        path == parent or not path.is_relative_to(parent)
        for path in children.values()
    ):
        raise ValueError("checkpoint roots must be distinct children of the run root")
    child_paths = tuple(children.values())
    identities = tuple(_filesystem_identity(path) for path in child_paths)
    folded = tuple(_casefolded_path_parts(path) for path in child_paths)
    for index, path in enumerate(child_paths):
        for other_index, other in enumerate(
            child_paths[index + 1 :], start=index + 1
        ):
            if (
                path == other
                or _parts_are_nested(folded[index], folded[other_index])
                or (
                    identities[index] is not None
                    and identities[index] == identities[other_index]
                )
            ):
                raise ValueError(
                    "checkpoint roots contain a Unicode/casefold or inode alias"
                )

    roots.root.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(
        {
            "canonical_root": str(children["canonical"]),
            "context_sha256": context_sha256,
            "frozen_config": asdict(FROZEN_CONFIG),
            "plan_sha256": plan.sha256,
            "runtime_versions": dict(context.runtime_versions),
            "start_method": FROZEN_CONFIG.start_method,
            "thread_environment": dict(_THREAD_ENVIRONMENT),
            "worker_roots": {
                str(index): str(
                    roots.worker_roots[index].resolve(strict=False)
                )
                for index in range(plan.worker_count)
            },
        }
    ) + b"\n"
    binding_sha256 = sha256(payload).hexdigest()
    root_descriptor = os.open(roots.root, _DIRECTORY_FLAGS)
    try:
        try:
            descriptor = os.open(
                _BINDING_FILE,
                _REGULAR_READ_FLAGS,
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            for path in child_paths:
                if _nonempty_checkpoint_root(path):
                    raise ValueError(
                        "unbound prior checkpoint evidence cannot be adopted"
                    )
            descriptor = os.open(
                _BINDING_FILE,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_descriptor,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(root_descriptor)
            return binding_sha256
        except OSError as error:
            raise ValueError(
                "frozen shard-plan binding must be a regular no-follow file"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "frozen shard-plan binding must be a regular file"
                )
            observed = _read_all(descriptor)
        finally:
            os.close(descriptor)
        if observed != payload:
            raise ValueError("frozen shard-plan binding mismatch")
        return binding_sha256
    finally:
        os.close(root_descriptor)


def _nonempty_checkpoint_root(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("checkpoint root must be a no-follow directory")
    descriptor = os.open(path, _DIRECTORY_FLAGS)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ValueError("checkpoint root identity changed")
        return bool(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise OSError("short write")
        written += count


def _read_all(descriptor: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


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


def _completion_from_payload(payload: object) -> WorkerCompletion:
    if not isinstance(payload, dict) or set(payload) != {
        "completed_datasets",
        "elapsed_seconds",
        "error",
        "peak_rss_bytes",
        "pid",
        "record_digests",
        "shard_index",
        "status",
        "thread_environment",
    }:
        raise ValueError("runtime completion evidence fields are invalid")
    return WorkerCompletion(
        payload["shard_index"],
        payload["pid"],
        payload["completed_datasets"],
        payload["record_digests"],
        payload["elapsed_seconds"],
        payload["peak_rss_bytes"],
        payload["thread_environment"],
        payload["status"],
        payload["error"],
    )


def _evidence_bytes(
    binding_sha256: str,
    attempts: Sequence[Mapping[str, object]],
) -> bytes:
    return canonical_json_bytes(
        {
            "attempts": [
                {
                    "attempt_id": attempt["attempt_id"],
                    "completions": [
                        _completion_payload(completion)
                        for completion in attempt["completions"]
                    ],
                    "parent_peak_rss_bytes": attempt[
                        "parent_peak_rss_bytes"
                    ],
                    "peak_total_rss_bytes": attempt[
                        "peak_total_rss_bytes"
                    ],
                }
                for attempt in attempts
            ],
            "binding_sha256": binding_sha256,
            "schema_version": 1,
        }
    ) + b"\n"


def _load_runtime_evidence(
    roots: ParallelRoots,
    binding_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    root_descriptor = os.open(roots.root, _DIRECTORY_FLAGS)
    try:
        try:
            descriptor = os.open(
                _EVIDENCE_FILE,
                _REGULAR_READ_FLAGS,
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            return ()
        except OSError as error:
            raise ValueError(
                "runtime evidence must be a regular no-follow file"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("runtime evidence must be a regular file")
            raw = _read_all(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(root_descriptor)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("runtime evidence is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "attempts",
            "binding_sha256",
            "schema_version",
        }
        or payload["schema_version"] != 1
        or payload["binding_sha256"] != binding_sha256
        or not isinstance(payload["attempts"], list)
    ):
        raise ValueError("runtime evidence binding is invalid")
    attempts = []
    seen_ids = set()
    for raw_attempt in payload["attempts"]:
        if not isinstance(raw_attempt, dict) or set(raw_attempt) != {
            "attempt_id",
            "completions",
            "parent_peak_rss_bytes",
            "peak_total_rss_bytes",
        }:
            raise ValueError("runtime attempt evidence fields are invalid")
        attempt_id = raw_attempt["attempt_id"]
        parent_rss = raw_attempt["parent_peak_rss_bytes"]
        peak_total = raw_attempt["peak_total_rss_bytes"]
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in seen_ids
            or type(parent_rss) is not int
            or parent_rss < 0
            or type(peak_total) is not int
            or peak_total < 0
            or not isinstance(raw_attempt["completions"], list)
            or not raw_attempt["completions"]
        ):
            raise ValueError("runtime attempt evidence is invalid")
        completions = tuple(
            _completion_from_payload(item)
            for item in raw_attempt["completions"]
        )
        if peak_total != parent_rss + sum(
            item.peak_rss_bytes for item in completions
        ):
            raise ValueError("runtime attempt memory evidence is invalid")
        seen_ids.add(attempt_id)
        attempts.append(
            MappingProxyType(
                {
                    "attempt_id": attempt_id,
                    "completions": completions,
                    "parent_peak_rss_bytes": parent_rss,
                    "peak_total_rss_bytes": peak_total,
                }
            )
        )
    frozen = tuple(attempts)
    if raw != _evidence_bytes(binding_sha256, frozen):
        raise ValueError("runtime evidence is noncanonical")
    return frozen


def _replace_runtime_evidence(
    roots: ParallelRoots,
    binding_sha256: str,
    attempts: Sequence[Mapping[str, object]],
) -> None:
    payload = _evidence_bytes(binding_sha256, attempts)
    root_descriptor = os.open(roots.root, _DIRECTORY_FLAGS)
    temporary = f".{_EVIDENCE_FILE}.{uuid.uuid4().hex}"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.rename(
            temporary,
            _EVIDENCE_FILE,
            src_dir_fd=root_descriptor,
            dst_dir_fd=root_descriptor,
        )
        os.fsync(root_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=root_descriptor)
        except FileNotFoundError:
            pass
        os.close(root_descriptor)


def _append_runtime_evidence(
    roots: ParallelRoots,
    binding_sha256: str,
    attempts: Sequence[Mapping[str, object]],
    attempt_id: str,
    completions: Sequence[WorkerCompletion],
    parent_peak_rss_bytes: int,
) -> tuple[Mapping[str, object], ...]:
    frozen_completions = tuple(completions)
    attempt = MappingProxyType(
        {
            "attempt_id": attempt_id,
            "completions": frozen_completions,
            "parent_peak_rss_bytes": parent_peak_rss_bytes,
            "peak_total_rss_bytes": parent_peak_rss_bytes
            + sum(item.peak_rss_bytes for item in frozen_completions),
        }
    )
    updated = (*attempts, attempt)
    _replace_runtime_evidence(roots, binding_sha256, updated)
    return tuple(updated)


def _validated_evidence_digests(
    plan: ShardPlan,
    attempts: Sequence[Mapping[str, object]],
) -> Mapping[str, str]:
    assignments = {
        assignment.shard_index: assignment
        for assignment in plan.assignments
    }
    digests = {}
    for attempt in attempts:
        for completion in attempt["completions"]:
            assignment = assignments.get(completion.shard_index)
            if assignment is None:
                raise ValueError("runtime evidence contains an unknown shard")
            if dict(completion.thread_environment) != dict(
                _THREAD_ENVIRONMENT
            ):
                raise ValueError(
                    "runtime evidence thread environment mismatch"
                )
            if any(
                name not in assignment.datasets
                for name in completion.completed_datasets
            ):
                raise ValueError("runtime evidence contains a wrong-shard record")
            for name, digest in completion.record_digests.items():
                if name in digests:
                    raise ValueError(
                        "runtime evidence contains duplicate dataset evidence"
                    )
                digests[name] = digest
    return MappingProxyType(digests)


def _aggregate_evidence_completions(
    plan: ShardPlan,
    attempts: Sequence[Mapping[str, object]],
    digests: Mapping[str, str],
    *,
    error: str = "",
) -> tuple[WorkerCompletion, ...]:
    result = []
    for assignment in plan.assignments:
        items = tuple(
            completion
            for attempt in attempts
            for completion in attempt["completions"]
            if completion.shard_index == assignment.shard_index
        )
        covered = {
            name
            for item in items
            for name in item.completed_datasets
        }
        if covered != set(assignment.datasets) or not items:
            raise ValueError("runtime evidence is incomplete for a shard")
        last = items[-1]
        result.append(
            WorkerCompletion(
                assignment.shard_index,
                last.pid,
                assignment.datasets,
                {
                    name: digests[name]
                    for name in assignment.datasets
                },
                sum(item.elapsed_seconds for item in items),
                max(item.peak_rss_bytes for item in items),
                last.thread_environment,
                "failed" if error else "completed",
                error,
            )
        )
    return tuple(result)


def _parent_peak_rss_bytes() -> int:
    return _peak_rss_bytes()


def _peak_total_rss_bytes(
    attempts: Sequence[Mapping[str, object]],
) -> int:
    persisted = max(
        (
            attempt["peak_total_rss_bytes"]
            for attempt in attempts
        ),
        default=0,
    )
    return max(persisted, _parent_peak_rss_bytes())


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


def _failed_parallel_run(
    plan: ShardPlan,
    error: BaseException | str,
    started: float,
) -> ParallelRun:
    message = str(error)
    elapsed = monotonic() - started
    parent_rss = _parent_peak_rss_bytes()
    environment = {
        name: os.environ[name]
        for name in _THREAD_ENVIRONMENT
        if name in os.environ
    }
    completions = tuple(
        WorkerCompletion(
            assignment.shard_index,
            os.getpid(),
            (),
            {},
            elapsed,
            parent_rss,
            environment,
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
    started = monotonic()
    try:
        if plan.population != context.datasets:
            raise ValueError("shard plan does not match the frozen context")
        context_digest = _context_sha256(context)
        if bootstrap.expected_context_sha256 != context_digest:
            raise ValueError("bootstrap context digest mismatch")
        binding_sha256 = _freeze_plan_binding(roots, plan, context)
        attempts = _load_runtime_evidence(roots, binding_sha256)
        evidence_digests = _validated_evidence_digests(plan, attempts)
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
        for name, digest in canonical_digests.items():
            if evidence_digests.get(name) != digest:
                raise ValueError(
                    "canonical checkpoint lacks matching durable evidence"
                )
        if set(canonical_digests) == set(context.datasets):
            if set(evidence_digests) != set(context.datasets):
                raise ValueError(
                    "complete canonical checkpoint lacks durable worker evidence"
                )
            summary = _reconstruct_global_summary(context, canonical)
            error = "" if summary.publication_allowed else summary.status
            completions = _aggregate_evidence_completions(
                plan,
                attempts,
                evidence_digests,
                error=error,
            )
            aggregate_rss = _peak_total_rss_bytes(attempts)
            if aggregate_rss > FROZEN_CONFIG.memory_limit_bytes:
                raise ValueError(
                    "aggregate worker and parent RSS exceeds the frozen limit"
                )
            return ParallelRun(
                plan,
                completions,
                {
                    name: canonical_digests[name]
                    for name in context.datasets
                },
                {
                    "run_summary": summary,
                    "plan_sha256": plan.sha256,
                    "aggregate_peak_rss_bytes": aggregate_rss,
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
        observed_digests = {
            name: digest
            for assignment in plan.assignments
            for name, digest in observed[
                assignment.shard_index
            ].items()
        }
        if observed_digests != dict(evidence_digests):
            raise ValueError(
                "reused worker checkpoints lack matching durable evidence"
            )
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
        if raw_completions:
            attempts = _append_runtime_evidence(
                roots,
                binding_sha256,
                attempts,
                attempt_token,
                raw_completions,
                _parent_peak_rss_bytes(),
            )
            evidence_digests = _validated_evidence_digests(
                plan, attempts
            )
        metrics = {
            completion.shard_index: completion
            for completion in raw_completions
        }
        if tuple(sorted(metrics)) != tuple(
            spec.shard.shard_index for spec in specs
        ):
            raise ValueError("worker completions do not cover scheduled shards")
        aggregate_rss = _peak_total_rss_bytes(attempts)
        if aggregate_rss > FROZEN_CONFIG.memory_limit_bytes:
            raise ValueError(
                "aggregate worker and parent RSS exceeds the frozen limit"
            )

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
        complete_digests = {
            name: digest
            for assignment in plan.assignments
            for name, digest in complete[
                assignment.shard_index
            ].items()
        }
        if complete_digests != dict(evidence_digests):
            raise ValueError(
                "complete worker stores differ from durable evidence"
            )
        merged = merge_worker_checkpoints(
            context, plan, worker_stores, canonical
        )
        summary = _reconstruct_global_summary(context, canonical)
        error = "" if summary.publication_allowed else summary.status
        completions = _aggregate_evidence_completions(
            plan,
            attempts,
            evidence_digests,
            error=error,
        )
        record_digests = {
            name: merged[name]
            for name in context.datasets
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
        return _failed_parallel_run(plan, error, started)


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
