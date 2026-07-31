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
    BenchmarkSelectionProof,
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


class _RunRootTransaction:
    """Pin one run-root inode and final pathname for one coordinator call."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ValueError("run root must be a Path")
        self.path = Path(os.path.abspath(root))
        try:
            initial = os.lstat(self.path)
        except FileNotFoundError:
            self.path.mkdir(parents=True)
            initial = os.lstat(self.path)
        if not stat.S_ISDIR(initial.st_mode):
            raise ValueError("run root must be a no-follow directory")
        self.descriptor = os.open(self.path, _DIRECTORY_FLAGS)
        opened = os.fstat(self.descriptor)
        self.identity = (opened.st_dev, opened.st_ino)
        if self.identity != (initial.st_dev, initial.st_ino):
            os.close(self.descriptor)
            raise ValueError("run root identity changed while opening")
        self.resolved_path = self.path.resolve(strict=True)
        try:
            self.verify()
        except Exception:
            os.close(self.descriptor)
            raise

    def verify(self) -> None:
        try:
            observed = os.lstat(self.path)
            opened = os.fstat(self.descriptor)
            resolved = self.path.resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise ValueError(
                "run root identity or final pathname changed"
            ) from error
        if (
            not stat.S_ISDIR(observed.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (observed.st_dev, observed.st_ino) != self.identity
            or (opened.st_dev, opened.st_ino) != self.identity
            or resolved != self.resolved_path
        ):
            raise ValueError("run root identity or final pathname changed")

    def verify_checkpoint_path(self, path: Path) -> Path:
        self.verify()
        resolved = path.resolve(strict=False)
        if (
            resolved == self.resolved_path
            or not resolved.is_relative_to(self.resolved_path)
        ):
            raise ValueError("checkpoint path escaped the pinned run root")
        return resolved

    def close(self) -> None:
        os.close(self.descriptor)


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
        if getattr(source, "expected_datasets", None) != tuple(
            sorted(assignment.datasets)
        ):
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
    return Exp417CheckpointStore(root, provenance, tuple(sorted(datasets)))


def _freeze_plan_binding(
    roots: ParallelRoots,
    plan: ShardPlan,
    context: FrozenRunContext,
    *,
    transaction: _RunRootTransaction | None = None,
) -> str:
    if not isinstance(roots, ParallelRoots):
        raise ValueError("roots must be ParallelRoots")
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be FrozenRunContext")
    if tuple(sorted(roots.worker_roots)) != tuple(range(plan.worker_count)):
        raise ValueError("worker roots do not cover the exact shard plan")
    owned_transaction = transaction is None
    active = transaction or _RunRootTransaction(roots.root)
    try:
        active.verify()
        if Path(os.path.abspath(roots.root)) != active.path:
            raise ValueError("run root transaction path mismatch")
        context_sha256 = _context_sha256(context)
        parent = active.resolved_path
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
            raise ValueError(
                "checkpoint roots must be distinct children of the run root"
            )
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

        payload = canonical_json_bytes(
            {
                "canonical_root": str(children["canonical"]),
                "context_sha256": context_sha256,
                "frozen_config": asdict(FROZEN_CONFIG),
                "plan_sha256": plan.sha256,
                "run_root": str(active.resolved_path),
                "run_root_identity": list(active.identity),
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
        observed = _read_optional_root_file(active, _BINDING_FILE)
        if observed is None:
            for path in child_paths:
                if _nonempty_checkpoint_root(path, active):
                    raise ValueError(
                        "unbound prior checkpoint evidence cannot be adopted"
                    )
            _publish_binding_no_replace(active, payload)
            observed = _read_optional_root_file(active, _BINDING_FILE)
            if observed is None:
                raise ValueError("frozen shard-plan binding was not published")
        if observed != payload:
            raise ValueError("frozen shard-plan binding mismatch")
        active.verify()
        return binding_sha256
    finally:
        if owned_transaction:
            active.close()


def _read_optional_root_file(
    transaction: _RunRootTransaction,
    name: str,
) -> bytes | None:
    transaction.verify()
    try:
        descriptor = os.open(
            name,
            _REGULAR_READ_FLAGS,
            dir_fd=transaction.descriptor,
        )
    except FileNotFoundError:
        transaction.verify()
        return None
    except OSError as error:
        raise ValueError(
            f"{name} must be a regular no-follow file"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be a regular file")
        observed = _read_all(descriptor)
    finally:
        os.close(descriptor)
    transaction.verify()
    return observed


def _publish_binding_no_replace(
    transaction: _RunRootTransaction,
    payload: bytes,
) -> None:
    temporary = f".{_BINDING_FILE}.{uuid.uuid4().hex}"
    descriptor = None
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=transaction.descriptor,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        transaction.verify()
        try:
            os.link(
                temporary,
                _BINDING_FILE,
                src_dir_fd=transaction.descriptor,
                dst_dir_fd=transaction.descriptor,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            pass
        if linked:
            os.fsync(transaction.descriptor)
        transaction.verify()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=transaction.descriptor)
        except FileNotFoundError:
            pass
        os.fsync(transaction.descriptor)
        transaction.verify()


def _nonempty_checkpoint_root(
    path: Path,
    transaction: _RunRootTransaction,
) -> bool:
    transaction.verify_checkpoint_path(path)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        transaction.verify()
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
        nonempty = bool(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    transaction.verify_checkpoint_path(path)
    return nonempty


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
    serialized_attempts = []
    for attempt in attempts:
        state = attempt["state"]
        workers = []
        for worker in attempt["workers"]:
            serialized = {
                "checkpoint_root": worker["checkpoint_root"],
                "inventory": list(worker["inventory"]),
                "resolved_checkpoint_root": worker[
                    "resolved_checkpoint_root"
                ],
                "shard_index": worker["shard_index"],
            }
            if state == "terminal":
                serialized["completion"] = _completion_payload(
                    worker["completion"]
                )
            elif state == "interrupted":
                serialized["record_digests"] = dict(
                    worker["record_digests"]
                )
                serialized["recovered_datasets"] = list(
                    worker["recovered_datasets"]
                )
            workers.append(serialized)
        serialized_attempt = {
            "attempt_id": attempt["attempt_id"],
            "binding_sha256": attempt["binding_sha256"],
            "context_sha256": attempt["context_sha256"],
            "plan_sha256": attempt["plan_sha256"],
            "state": state,
            "workers": workers,
        }
        if state == "terminal":
            serialized_attempt["parent_peak_rss_bytes"] = attempt[
                "parent_peak_rss_bytes"
            ]
            serialized_attempt["peak_total_rss_bytes"] = attempt[
                "peak_total_rss_bytes"
            ]
        serialized_attempts.append(serialized_attempt)
    return canonical_json_bytes(
        {
            "attempts": serialized_attempts,
            "binding_sha256": binding_sha256,
            "schema_version": 3,
        }
    ) + b"\n"


def _load_runtime_evidence(
    roots: ParallelRoots,
    binding_sha256: str,
    *,
    transaction: _RunRootTransaction | None = None,
) -> tuple[Mapping[str, object], ...]:
    owned_transaction = transaction is None
    active = transaction or _RunRootTransaction(roots.root)
    try:
        raw = _read_optional_root_file(active, _EVIDENCE_FILE)
        if raw is None:
            return ()
    finally:
        if owned_transaction:
            active.close()
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
        or payload["schema_version"] != 3
        or payload["binding_sha256"] != binding_sha256
        or not isinstance(payload["attempts"], list)
    ):
        raise ValueError("runtime evidence binding is invalid")
    attempts = []
    seen_ids = set()
    for raw_attempt in payload["attempts"]:
        if not isinstance(raw_attempt, dict):
            raise ValueError("runtime attempt evidence fields are invalid")
        state = raw_attempt.get("state")
        common_fields = {
            "attempt_id",
            "binding_sha256",
            "context_sha256",
            "plan_sha256",
            "state",
            "workers",
        }
        expected_fields = (
            common_fields
            | {
                "parent_peak_rss_bytes",
                "peak_total_rss_bytes",
            }
            if state == "terminal"
            else common_fields
        )
        if (
            state not in {"running", "interrupted", "terminal"}
            or set(raw_attempt) != expected_fields
        ):
            raise ValueError("runtime attempt evidence fields are invalid")
        attempt_id = raw_attempt["attempt_id"]
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or attempt_id in seen_ids
            or raw_attempt["binding_sha256"] != binding_sha256
            or not _is_sha256(raw_attempt["context_sha256"])
            or not _is_sha256(raw_attempt["plan_sha256"])
            or not isinstance(raw_attempt["workers"], list)
            or not raw_attempt["workers"]
        ):
            raise ValueError("runtime attempt evidence is invalid")
        workers = []
        seen_shards = set()
        for raw_worker in raw_attempt["workers"]:
            common_worker_fields = {
                "checkpoint_root",
                "inventory",
                "resolved_checkpoint_root",
                "shard_index",
            }
            expected_worker_fields = (
                common_worker_fields | {"completion"}
                if state == "terminal"
                else (
                    common_worker_fields
                    | {"record_digests", "recovered_datasets"}
                    if state == "interrupted"
                    else common_worker_fields
                )
            )
            if (
                not isinstance(raw_worker, dict)
                or set(raw_worker) != expected_worker_fields
            ):
                raise ValueError(
                    "runtime attempt worker evidence fields are invalid"
                )
            checkpoint_root = raw_worker["checkpoint_root"]
            resolved_root = raw_worker["resolved_checkpoint_root"]
            shard_index = raw_worker["shard_index"]
            inventory = raw_worker["inventory"]
            if (
                not isinstance(checkpoint_root, str)
                or not checkpoint_root
                or not isinstance(resolved_root, str)
                or not resolved_root
                or type(shard_index) is not int
                or shard_index < 0
                or shard_index in seen_shards
                or not isinstance(inventory, list)
                or not inventory
                or any(
                    not isinstance(name, str) or not name
                    for name in inventory
                )
                or len(inventory) != len(set(inventory))
            ):
                raise ValueError(
                    "runtime attempt worker evidence is invalid"
                )
            seen_shards.add(shard_index)
            worker = {
                "checkpoint_root": checkpoint_root,
                "inventory": tuple(inventory),
                "resolved_checkpoint_root": resolved_root,
                "shard_index": shard_index,
            }
            if state == "terminal":
                worker["completion"] = _completion_from_payload(
                    raw_worker["completion"]
                )
            elif state == "interrupted":
                recovered = raw_worker["recovered_datasets"]
                digests = raw_worker["record_digests"]
                if (
                    not isinstance(recovered, list)
                    or any(
                        not isinstance(name, str) or not name
                        for name in recovered
                    )
                    or len(recovered) != len(set(recovered))
                    or not isinstance(digests, dict)
                    or set(digests) != set(recovered)
                    or any(
                        not _is_sha256(digest)
                        for digest in digests.values()
                    )
                ):
                    raise ValueError(
                        "interrupted attempt record evidence is invalid"
                    )
                worker["recovered_datasets"] = tuple(recovered)
                worker["record_digests"] = MappingProxyType(
                    dict(digests)
                )
            workers.append(
                MappingProxyType(worker)
            )
        frozen_workers = tuple(workers)
        attempt = {
            "attempt_id": attempt_id,
            "binding_sha256": raw_attempt["binding_sha256"],
            "context_sha256": raw_attempt["context_sha256"],
            "plan_sha256": raw_attempt["plan_sha256"],
            "state": state,
            "workers": frozen_workers,
        }
        if state == "terminal":
            parent_rss = raw_attempt["parent_peak_rss_bytes"]
            peak_total = raw_attempt["peak_total_rss_bytes"]
            if (
                type(parent_rss) is not int
                or parent_rss < 0
                or type(peak_total) is not int
                or peak_total < 0
                or peak_total
                != parent_rss
                + sum(
                    worker["completion"].peak_rss_bytes
                    for worker in frozen_workers
                )
            ):
                raise ValueError(
                    "runtime attempt memory evidence is invalid"
                )
            attempt["parent_peak_rss_bytes"] = parent_rss
            attempt["peak_total_rss_bytes"] = peak_total
        seen_ids.add(attempt_id)
        attempts.append(
            MappingProxyType(attempt)
        )
    frozen = tuple(attempts)
    if raw != _evidence_bytes(binding_sha256, frozen):
        raise ValueError("runtime evidence is noncanonical")
    return frozen


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _replace_runtime_evidence(
    roots: ParallelRoots,
    binding_sha256: str,
    attempts: Sequence[Mapping[str, object]],
    *,
    transaction: _RunRootTransaction | None = None,
) -> None:
    payload = _evidence_bytes(binding_sha256, attempts)
    owned_transaction = transaction is None
    active = transaction or _RunRootTransaction(roots.root)
    temporary = f".{_EVIDENCE_FILE}.{uuid.uuid4().hex}"
    descriptor = None
    try:
        active.verify()
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=active.descriptor,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        active.verify()
        os.rename(
            temporary,
            _EVIDENCE_FILE,
            src_dir_fd=active.descriptor,
            dst_dir_fd=active.descriptor,
        )
        os.fsync(active.descriptor)
        active.verify()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=active.descriptor)
        except FileNotFoundError:
            pass
        if owned_transaction:
            active.close()


def _running_attempt_record(
    roots: ParallelRoots,
    plan: ShardPlan,
    binding_sha256: str,
    context_sha256: str,
    attempt_id: str,
    specs: Sequence[WorkerSpec],
) -> Mapping[str, object]:
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("runtime attempt id is invalid")
    frozen_specs = tuple(specs)
    if not frozen_specs:
        raise ValueError("running attempt must contain worker specs")
    workers = []
    for spec in frozen_specs:
        lexical_root = Path(
            os.path.abspath(spec.checkpoint_root)
        )
        expected_lexical = Path(
            os.path.abspath(
                roots.root
                / ".attempts"
                / plan.sha256
                / attempt_id
                / f"worker-{spec.shard.shard_index}"
            )
        )
        expected_resolved = (
            roots.root
            / ".attempts"
            / plan.sha256
            / attempt_id
            / f"worker-{spec.shard.shard_index}"
        ).resolve(strict=False)
        observed_resolved = spec.checkpoint_root.resolve(strict=False)
        if (
            lexical_root != expected_lexical
            or observed_resolved != expected_resolved
        ):
            raise ValueError("runtime attempt root is not plan-bound")
        workers.append(
            MappingProxyType(
                {
                    "checkpoint_root": str(lexical_root),
                    "inventory": spec.shard.datasets,
                    "resolved_checkpoint_root": str(observed_resolved),
                    "shard_index": spec.shard.shard_index,
                }
            )
        )
    return MappingProxyType(
        {
            "attempt_id": attempt_id,
            "binding_sha256": binding_sha256,
            "context_sha256": context_sha256,
            "plan_sha256": plan.sha256,
            "state": "running",
            "workers": tuple(workers),
        }
    )


def _terminal_attempt_record(
    running: Mapping[str, object],
    completions: Sequence[WorkerCompletion],
    parent_peak_rss_bytes: int,
) -> Mapping[str, object]:
    if running["state"] != "running":
        raise ValueError("only a running attempt can become terminal")
    if type(parent_peak_rss_bytes) is not int or parent_peak_rss_bytes < 0:
        raise ValueError("parent peak RSS evidence is invalid")
    frozen_completions = tuple(completions)
    completion_by_shard = {
        completion.shard_index: completion
        for completion in frozen_completions
    }
    expected_shards = tuple(
        worker["shard_index"] for worker in running["workers"]
    )
    if (
        len(completion_by_shard) != len(frozen_completions)
        or tuple(sorted(completion_by_shard)) != expected_shards
    ):
        raise ValueError("worker completions do not cover scheduled shards")
    workers = tuple(
        MappingProxyType(
            {
                **dict(worker),
                "completion": completion_by_shard[
                    worker["shard_index"]
                ],
            }
        )
        for worker in running["workers"]
    )
    return MappingProxyType(
        {
            "attempt_id": running["attempt_id"],
            "binding_sha256": running["binding_sha256"],
            "context_sha256": running["context_sha256"],
            "parent_peak_rss_bytes": parent_peak_rss_bytes,
            "peak_total_rss_bytes": parent_peak_rss_bytes
            + sum(
                worker["completion"].peak_rss_bytes
                for worker in workers
            ),
            "plan_sha256": running["plan_sha256"],
            "state": "terminal",
            "workers": workers,
        }
    )


def _register_running_attempt(
    roots: ParallelRoots,
    plan: ShardPlan,
    binding_sha256: str,
    context_sha256: str,
    attempts: Sequence[Mapping[str, object]],
    attempt_id: str,
    specs: Sequence[WorkerSpec],
    *,
    transaction: _RunRootTransaction | None = None,
) -> tuple[Mapping[str, object], ...]:
    if any(attempt["state"] == "running" for attempt in attempts):
        raise ValueError("a running attempt is already registered")
    attempt = _running_attempt_record(
        roots,
        plan,
        binding_sha256,
        context_sha256,
        attempt_id,
        specs,
    )
    updated = (*attempts, attempt)
    _validated_evidence_digests(
        plan,
        roots,
        updated,
        binding_sha256=binding_sha256,
        context_sha256=context_sha256,
    )
    _replace_runtime_evidence(
        roots,
        binding_sha256,
        updated,
        transaction=transaction,
    )
    return tuple(updated)


def _advance_running_attempt(
    roots: ParallelRoots,
    plan: ShardPlan,
    binding_sha256: str,
    context_sha256: str,
    attempts: Sequence[Mapping[str, object]],
    replacement: Mapping[str, object],
    *,
    transaction: _RunRootTransaction | None = None,
) -> tuple[Mapping[str, object], ...]:
    if replacement["state"] not in {"interrupted", "terminal"}:
        raise ValueError("running attempt requires a terminal transition")
    replaced = False
    updated = []
    for attempt in attempts:
        if attempt["attempt_id"] == replacement["attempt_id"]:
            if replaced or attempt["state"] != "running":
                raise ValueError("running attempt transition is invalid")
            immutable_fields = (
                "attempt_id",
                "binding_sha256",
                "context_sha256",
                "plan_sha256",
            )
            if any(
                replacement[field] != attempt[field]
                for field in immutable_fields
            ) or tuple(
                (
                    worker["checkpoint_root"],
                    worker["inventory"],
                    worker["resolved_checkpoint_root"],
                    worker["shard_index"],
                )
                for worker in replacement["workers"]
            ) != tuple(
                (
                    worker["checkpoint_root"],
                    worker["inventory"],
                    worker["resolved_checkpoint_root"],
                    worker["shard_index"],
                )
                for worker in attempt["workers"]
            ):
                raise ValueError(
                    "immutable running attempt descriptor changed"
                )
            updated.append(replacement)
            replaced = True
        else:
            updated.append(attempt)
    if not replaced:
        raise ValueError("registered running attempt is missing")
    frozen = tuple(updated)
    _validated_evidence_digests(
        plan,
        roots,
        frozen,
        binding_sha256=binding_sha256,
        context_sha256=context_sha256,
    )
    _replace_runtime_evidence(
        roots,
        binding_sha256,
        frozen,
        transaction=transaction,
    )
    return frozen


def _validated_evidence_digests(
    plan: ShardPlan,
    roots: ParallelRoots,
    attempts: Sequence[Mapping[str, object]],
    *,
    binding_sha256: str | None = None,
    context_sha256: str | None = None,
) -> Mapping[str, str]:
    assignments = {
        assignment.shard_index: assignment
        for assignment in plan.assignments
    }
    digests = {}
    for attempt in attempts:
        if (
            attempt["plan_sha256"] != plan.sha256
            or (
                binding_sha256 is not None
                and attempt["binding_sha256"] != binding_sha256
            )
            or (
                context_sha256 is not None
                and attempt["context_sha256"] != context_sha256
            )
        ):
            raise ValueError("runtime attempt binding is invalid")
        seen_shards = set()
        for worker in attempt["workers"]:
            shard_index = worker["shard_index"]
            assignment = assignments.get(shard_index)
            if assignment is None:
                raise ValueError("runtime evidence contains an unknown shard")
            if shard_index in seen_shards:
                raise ValueError(
                    "runtime attempt contains a duplicate shard"
                )
            seen_shards.add(shard_index)
            inventory = worker["inventory"]
            inventory_set = set(inventory)
            if inventory != tuple(
                name
                for name in assignment.datasets
                if name in inventory_set
            ):
                raise ValueError(
                    "runtime attempt inventory is not original-shard ordered"
                )
            expected_lexical = Path(
                os.path.abspath(
                    roots.root
                    / ".attempts"
                    / plan.sha256
                    / attempt["attempt_id"]
                    / f"worker-{shard_index}"
                )
            )
            expected_resolved = (
                roots.root
                / ".attempts"
                / plan.sha256
                / attempt["attempt_id"]
                / f"worker-{shard_index}"
            ).resolve(strict=False)
            if (
                Path(worker["checkpoint_root"]) != expected_lexical
                or Path(worker["resolved_checkpoint_root"])
                != expected_resolved
            ):
                raise ValueError(
                    "runtime attempt root is not plan-bound"
                )
            if attempt["state"] == "running":
                continue
            if attempt["state"] == "terminal":
                completion = worker["completion"]
                if completion.shard_index != shard_index:
                    raise ValueError(
                        "runtime completion shard binding is invalid"
                    )
                completed = completion.completed_datasets
                record_digests = completion.record_digests
            else:
                completed = worker["recovered_datasets"]
                record_digests = worker["record_digests"]
            if completed != inventory[: len(completed)]:
                raise ValueError(
                    "runtime completion is not an atomic dataset prefix"
                )
            if (
                attempt["state"] == "terminal"
                and completion.status == "completed"
                and completed != inventory
            ):
                raise ValueError(
                    "completed runtime evidence has incomplete inventory"
                )
            if (
                attempt["state"] == "terminal"
                and completed
                and dict(completion.thread_environment)
                != dict(_THREAD_ENVIRONMENT)
            ):
                raise ValueError(
                    "runtime evidence thread environment mismatch"
                )
            for name, digest in record_digests.items():
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
            worker["completion"]
            for attempt in attempts
            if attempt["state"] == "terminal"
            for worker in attempt["workers"]
            if worker["completion"].shard_index
            == assignment.shard_index
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
            if attempt["state"] == "terminal"
        ),
        default=0,
    )
    return max(persisted, _parent_peak_rss_bytes())


def _has_missing_terminal_evidence(
    attempts: Sequence[Mapping[str, object]],
) -> bool:
    return any(
        attempt["state"] == "interrupted"
        for attempt in attempts
    )


def _validated_partial_store_state(
    store: object,
    *,
    transaction: _RunRootTransaction | None = None,
    root: Path | None = None,
) -> tuple[Mapping[str, str], Mapping[str, object]]:
    if (transaction is None) != (root is None):
        raise ValueError(
            "checkpoint transaction and root must be provided together"
        )
    if transaction is not None:
        transaction.verify_checkpoint_path(root)
    first = store.completed_record_digests()
    if transaction is not None:
        transaction.verify_checkpoint_path(root)
    completed = store.load_completed()
    if transaction is not None:
        transaction.verify_checkpoint_path(root)
    second = store.completed_record_digests()
    if transaction is not None:
        transaction.verify_checkpoint_path(root)
    if dict(first) != dict(second) or set(completed) != set(first):
        raise ValueError("checkpoint changed during coordinator validation")
    return first, completed


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
    *,
    transaction: _RunRootTransaction | None = None,
    roots: Mapping[int, Path] | None = None,
) -> dict[int, Mapping[str, str]]:
    if (transaction is None) != (roots is None):
        raise ValueError(
            "worker transaction and roots must be provided together"
        )
    seen: set[str] = set()
    observed: dict[int, Mapping[str, str]] = {}
    for assignment in plan.assignments:
        store = stores[assignment.shard_index]
        if getattr(store, "expected_datasets", None) != tuple(
            sorted(assignment.datasets)
        ):
            raise ValueError("worker store inventory does not match its shard")
        digests, completed = _validated_partial_store_state(
            store,
            transaction=transaction,
            root=(
                None
                if roots is None
                else roots[assignment.shard_index]
            ),
        )
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


def _open_checkpoint_store_checked(
    transaction: _RunRootTransaction,
    root: Path,
    provenance: object,
    datasets: tuple[str, ...],
) -> Exp417CheckpointStore:
    transaction.verify_checkpoint_path(root)
    store = _open_checkpoint_store(root, provenance, datasets)
    transaction.verify_checkpoint_path(root)
    return store


def _interrupted_attempt_record(
    running: Mapping[str, object],
    bootstrap: WorkerBootstrap,
    transaction: _RunRootTransaction,
) -> Mapping[str, object]:
    if running["state"] != "running":
        raise ValueError("only a running attempt can be interrupted")
    workers = []
    for worker in running["workers"]:
        source_root = Path(worker["checkpoint_root"])
        inventory = worker["inventory"]
        source = _open_checkpoint_store_checked(
            transaction,
            source_root,
            bootstrap.provenance,
            inventory,
        )
        on_disk, completed = _validated_partial_store_state(
            source,
            transaction=transaction,
            root=source_root,
        )
        recovered = tuple(
            name for name in inventory if name in on_disk
        )
        if (
            recovered != inventory[: len(recovered)]
            or set(on_disk) != set(recovered)
            or set(completed) != set(recovered)
        ):
            raise ValueError(
                "registered running attempt is not an atomic dataset prefix"
            )
        if any(
            isinstance(
                result.decision_evidence,
                DatasetTerminalEvidence,
            )
            for result in completed.values()
        ):
            raise ValueError(
                "registered running attempt contains terminal evidence"
            )
        workers.append(
            MappingProxyType(
                {
                    **dict(worker),
                    "record_digests": MappingProxyType(
                        {
                            name: on_disk[name]
                            for name in recovered
                        }
                    ),
                    "recovered_datasets": recovered,
                }
            )
        )
    return MappingProxyType(
        {
            "attempt_id": running["attempt_id"],
            "binding_sha256": running["binding_sha256"],
            "context_sha256": running["context_sha256"],
            "plan_sha256": running["plan_sha256"],
            "state": "interrupted",
            "workers": tuple(workers),
        }
    )


def _recover_registered_running_attempt(
    roots: ParallelRoots,
    plan: ShardPlan,
    binding_sha256: str,
    context_sha256: str,
    bootstrap: WorkerBootstrap,
    attempts: Sequence[Mapping[str, object]],
    transaction: _RunRootTransaction,
) -> tuple[Mapping[str, object], ...]:
    _validated_evidence_digests(
        plan,
        roots,
        attempts,
        binding_sha256=binding_sha256,
        context_sha256=context_sha256,
    )
    running = tuple(
        attempt
        for attempt in attempts
        if attempt["state"] == "running"
    )
    if len(running) > 1:
        raise ValueError("multiple running attempts are registered")
    if not running:
        return tuple(attempts)
    interrupted = _interrupted_attempt_record(
        running[0],
        bootstrap,
        transaction,
    )
    return _advance_running_attempt(
        roots,
        plan,
        binding_sha256,
        context_sha256,
        attempts,
        interrupted,
        transaction=transaction,
    )


def _validate_and_recover_attempts(
    plan: ShardPlan,
    roots: ParallelRoots,
    bootstrap: WorkerBootstrap,
    attempts: Sequence[Mapping[str, object]],
    transaction: _RunRootTransaction,
    *,
    worker_stores: Mapping[int, object] | None,
) -> None:
    _validated_evidence_digests(plan, roots, attempts)
    for attempt in attempts:
        if attempt["state"] == "running":
            raise ValueError(
                "running attempt must be terminalized before recovery"
            )
        for worker in attempt["workers"]:
            shard_index = worker["shard_index"]
            source_root = Path(worker["checkpoint_root"])
            inventory = worker["inventory"]
            if attempt["state"] == "terminal":
                completion = worker["completion"]
                evidence_datasets = completion.completed_datasets
                evidence_digests = completion.record_digests
            else:
                evidence_datasets = worker["recovered_datasets"]
                evidence_digests = worker["record_digests"]
            source = _open_checkpoint_store_checked(
                transaction,
                source_root,
                bootstrap.provenance,
                inventory,
            )
            on_disk, completed = _validated_partial_store_state(
                source,
                transaction=transaction,
                root=source_root,
            )
            transaction.verify_checkpoint_path(source_root)
            if tuple(evidence_datasets) != tuple(
                name for name in inventory if name in on_disk
            ):
                raise ValueError(
                    "runtime attempt dataset evidence differs from its checkpoint"
                )
            if dict(evidence_digests) != dict(on_disk):
                raise ValueError(
                    "runtime attempt digest evidence differs from its checkpoint"
                )
            if set(completed) != set(evidence_datasets):
                raise ValueError(
                    "runtime attempt loaded evidence differs from its checkpoint"
                )
            if any(
                isinstance(
                    result.decision_evidence,
                    DatasetTerminalEvidence,
                )
                for result in completed.values()
            ):
                raise ValueError(
                    "runtime attempt contains terminal dataset evidence"
                )
            if worker_stores is None:
                continue
            target_root = roots.worker_roots[shard_index]
            target = worker_stores[shard_index]
            target_digests, _ = _validated_partial_store_state(
                target,
                transaction=transaction,
                root=target_root,
            )
            transaction.verify_checkpoint_path(target_root)
            for dataset in evidence_datasets:
                if dataset in target_digests:
                    if target_digests[dataset] != on_disk[dataset]:
                        raise ValueError(
                            "recovered shard digest differs from attempt evidence"
                        )
                    continue
                imported = target.import_validated_dataset(source, dataset)
                transaction.verify_checkpoint_path(target_root)
                transaction.verify_checkpoint_path(source_root)
                if imported != on_disk[dataset]:
                    raise ValueError(
                        "recovered shard import digest mismatch"
                    )
                target_digests = {
                    **target_digests,
                    dataset: imported,
                }


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
    transaction = None
    try:
        if plan.population != context.datasets:
            raise ValueError("shard plan does not match the frozen context")
        context_digest = _context_sha256(context)
        if bootstrap.expected_context_sha256 != context_digest:
            raise ValueError("bootstrap context digest mismatch")
        transaction = _RunRootTransaction(roots.root)
        binding_sha256 = _freeze_plan_binding(
            roots,
            plan,
            context,
            transaction=transaction,
        )
        attempts = _load_runtime_evidence(
            roots,
            binding_sha256,
            transaction=transaction,
        )
        attempts = _recover_registered_running_attempt(
            roots,
            plan,
            binding_sha256,
            context_digest,
            bootstrap,
            attempts,
            transaction,
        )
        evidence_digests = _validated_evidence_digests(
            plan,
            roots,
            attempts,
            binding_sha256=binding_sha256,
            context_sha256=context_digest,
        )
        canonical = _open_checkpoint_store_checked(
            transaction,
            roots.canonical_root,
            bootstrap.provenance,
            context.datasets,
        )
        canonical_digests, canonical_completed = (
            _validated_partial_store_state(
                canonical,
                transaction=transaction,
                root=roots.canonical_root,
            )
        )
        transaction.verify_checkpoint_path(roots.canonical_root)
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
        if set(canonical_digests) - set(context.datasets):
            raise ValueError("canonical checkpoint contains an extra dataset")

        worker_stores = {
            assignment.shard_index: _open_checkpoint_store_checked(
                transaction,
                roots.worker_roots[assignment.shard_index],
                bootstrap.provenance,
                assignment.datasets,
            )
            for assignment in plan.assignments
        }
        _validate_and_recover_attempts(
            plan,
            roots,
            bootstrap,
            attempts,
            transaction,
            worker_stores=worker_stores,
        )
        observed = _validate_partial_worker_stores(
            plan,
            worker_stores,
            transaction=transaction,
            roots=roots.worker_roots,
        )
        transaction.verify()
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

        if set(canonical_digests) == set(context.datasets):
            if _has_missing_terminal_evidence(attempts):
                raise ValueError(
                    "missing terminal worker evidence for interrupted attempt"
                )
            summary = _reconstruct_global_summary(context, canonical)
            transaction.verify_checkpoint_path(roots.canonical_root)
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
            transaction.verify()
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
                        outer_worker_count=plan.worker_count,
                    )
                )

        if specs:
            attempts = _register_running_attempt(
                roots,
                plan,
                binding_sha256,
                context_digest,
                attempts,
                attempt_token,
                specs,
                transaction=transaction,
            )
        transaction.verify()
        raw_completions = (
            spawn_workers(tuple(specs), bootstrap) if specs else ()
        )
        transaction.verify()
        if raw_completions:
            running_attempt = next(
                attempt
                for attempt in attempts
                if attempt["attempt_id"] == attempt_token
            )
            candidate = _terminal_attempt_record(
                running_attempt,
                raw_completions,
                _parent_peak_rss_bytes(),
            )
            candidate_attempts = tuple(
                candidate
                if attempt["attempt_id"] == attempt_token
                else attempt
                for attempt in attempts
            )
            _validated_evidence_digests(
                plan,
                roots,
                candidate_attempts,
                binding_sha256=binding_sha256,
                context_sha256=context_digest,
            )
            _validate_and_recover_attempts(
                plan,
                roots,
                bootstrap,
                (candidate,),
                transaction,
                worker_stores=None,
            )
            attempts = _advance_running_attempt(
                roots,
                plan,
                binding_sha256,
                context_digest,
                attempts,
                candidate,
                transaction=transaction,
            )
            evidence_digests = _validated_evidence_digests(
                plan,
                roots,
                attempts,
                binding_sha256=binding_sha256,
                context_sha256=context_digest,
            )
            _validate_and_recover_attempts(
                plan,
                roots,
                bootstrap,
                (attempts[-1],),
                transaction,
                worker_stores=worker_stores,
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

        complete = _validate_partial_worker_stores(
            plan,
            worker_stores,
            transaction=transaction,
            roots=roots.worker_roots,
        )
        transaction.verify()
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
        if _has_missing_terminal_evidence(attempts):
            raise ValueError(
                "missing terminal worker evidence for interrupted attempt"
            )
        transaction.verify()
        merged = merge_worker_checkpoints(
            context, plan, worker_stores, canonical
        )
        transaction.verify()
        summary = _reconstruct_global_summary(context, canonical)
        transaction.verify()
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
    finally:
        if transaction is not None:
            transaction.close()


def select_runtime(
    trials: Sequence[RuntimeTrial],
    selection_proof: BenchmarkSelectionProof,
) -> RuntimeBenchmark:
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
                trial.worker_count
                in FROZEN_CONFIG.benchmark_proxy_headroom_workers
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
        selection_proof,
    )


__all__ = [
    "merge_worker_checkpoints",
    "resume_or_run_parallel",
    "select_runtime",
]
