"""Selective, single-MPS-process runtime for the frozen Exp425 contract.

This module deliberately stores compact effective results only.  Complete Exp417
``DatasetResult`` objects are live execution material and never become part of a
progress or checkpoint payload, where they could accidentally carry held-out
evidence beyond the frozen review boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import copy
import fcntl
import json
import os
from pathlib import Path
import pickle
from statistics import median
from time import perf_counter, time
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import canonical_result_sha256
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit, DatasetResult, FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.qualification import apply_global_workload
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerRuntime,
    _production_qualify_train,
    _production_score_test,
    _production_train_models,
    run_dataset,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import preprocess_length_groups

from . import capabilities
from .checkpoints import Exp425CheckpointStore, _native_record_body
from .contracts import NativeRecordEnvelope, thaw_payload
from .database import Exp425DatabaseSnapshot
from .eligibility import apply_group_eligibility


ExecutionCommand = Literal["new", "resume", "inspect"]
_STATES = frozenset({"initializing", "importing", "running", "finalizing", "complete_review_only", "failed"})
_FORBIDDEN_KEY_TOKENS = ("label", "target", "groundtruth", "anomalylocation", "evaluation", "confusion", "precision", "recall", "familytestperformance")


def _now() -> float:
    return time()


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _safe_message(error: BaseException) -> str:
    """Persist a stable exception class, never arbitrary source values."""
    message = str(error).replace("\n", " ").strip()
    return message[:240] if message else type(error).__name__


def _assert_label_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("runtime payload keys must be strings")
            normalized = "".join(character for character in key.lower() if character.isalnum())
            if any(token in normalized for token in _FORBIDDEN_KEY_TOKENS):
                raise ValueError("runtime payload contains forbidden evaluation evidence")
            _assert_label_free(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_label_free(nested)
    elif isinstance(value, float) and value != value:
        raise ValueError("runtime payload contains nonfinite value")


@dataclass(frozen=True)
class RuntimePaths:
    output_root: Path
    checkpoint_root: Path
    progress: Path
    events: Path
    parent_log: Path
    pid: Path

    def __post_init__(self) -> None:
        for name in ("output_root", "checkpoint_root", "progress", "events", "parent_log", "pid"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())

    def payload(self) -> dict[str, str]:
        return {
            "output_root": str(self.output_root), "checkpoint_root": str(self.checkpoint_root),
            "progress_path": str(self.progress), "events_path": str(self.events),
            "parent_log_path": str(self.parent_log), "pid_path": str(self.pid),
        }


@dataclass(frozen=True)
class Exp425RunCounts:
    total: int
    completed: int = 0
    imported: int = 0
    native: int = 0
    recomputed: int = 0
    abstained: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0 for field in self.__dataclass_fields__):
            raise ValueError("runtime counters must be non-negative integers")
        if self.completed > self.total or self.imported + self.native != self.completed:
            raise ValueError("runtime counters do not conserve completed records")

    @property
    def remaining(self) -> int:
        return self.total - self.completed


@dataclass(frozen=True)
class RuntimeProgress:
    paths: RuntimePaths
    run_identity: str
    command: ExecutionCommand
    total_datasets: int
    backend: str
    state: str = "initializing"
    stage: str = "initializing"
    counts: Exp425RunCounts | None = None
    process_start_time: float = field(default_factory=_now)
    timestamp: float = field(default_factory=_now)
    current_dataset: str | None = None
    ordinal: int = 0
    origin: str | None = None
    current_phase: str | None = None
    dataset_start_time: float | None = None
    last_terminal_time: float | None = None
    elapsed_seconds: float = 0.0
    cumulative_seconds: float = 0.0
    terminal_samples: tuple[float, ...] = ()
    bytes_processed: int = 0
    values_processed: int = 0
    last_completed_dataset: str | None = None
    last_status: str | None = None
    last_reason: str | None = None
    last_elapsed_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    heartbeat_sequence: int = 0
    train_instance_count: int | None = None
    test_instance_count: int | None = None
    size_summary: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.state not in _STATES or not self.run_identity or self.total_datasets < 1:
            raise ValueError("runtime progress identity or state is invalid")
        if self.counts is None:
            object.__setattr__(self, "counts", Exp425RunCounts(self.total_datasets))
        if not isinstance(self.counts, Exp425RunCounts) or self.counts.total != self.total_datasets:
            raise ValueError("runtime progress counters are invalid")
        if self.current_dataset is None and self.state == "running" and self.current_phase == "dataset":
            raise ValueError("running dataset phase requires a current dataset")
        if self.current_dataset is not None and self.state not in {"running", "failed"}:
            raise ValueError("current dataset is only valid during running or failure")
        object.__setattr__(self, "terminal_samples", tuple(float(value) for value in self.terminal_samples if value >= 0.0))
        object.__setattr__(self, "size_summary", MappingProxyType(dict(self.size_summary or {})))

    @classmethod
    def initializing(cls, paths: RuntimePaths, run_identity: str, command: ExecutionCommand, total: int, backend: str) -> "RuntimeProgress":
        return cls(paths, run_identity, command, total, backend)

    def payload(self) -> dict[str, object]:
        samples = self.terminal_samples
        eta = _robust_eta(samples, self.counts.remaining)
        recent = samples[-min(5, len(samples)):] if samples else ()
        elapsed = max(0.0, _now() - self.process_start_time)
        payload: dict[str, object] = {
            "schema": "exp425.runtime-progress.v1", "timestamp": self.timestamp,
            "state": self.state, "stage": self.stage, "run_identity": self.run_identity,
            "pid": os.getpid(), "process_start_time": self.process_start_time,
            "command": self.command, **self.paths.payload(), "total_datasets": self.total_datasets,
            "completed_datasets": self.counts.completed, "remaining_datasets": self.counts.remaining,
            "imported_datasets": self.counts.imported, "native_datasets": self.counts.native,
            "recomputed_datasets": self.counts.recomputed, "abstained_datasets": self.counts.abstained,
            "failed_datasets": self.counts.failed, "current_dataset": self.current_dataset,
            "successful_datasets": self.counts.completed - self.counts.failed,
            "ordinal": self.ordinal, "origin": self.origin, "current_phase": self.current_phase,
            "train_instance_count": self.train_instance_count, "test_instance_count": self.test_instance_count,
            "size_summary": dict(self.size_summary), "dataset_start_time": self.dataset_start_time,
            "last_terminal_time": self.last_terminal_time, "elapsed_seconds": elapsed,
            "cumulative_seconds": self.cumulative_seconds, "recent_datasets_per_second": _rate(recent),
            "cumulative_datasets_per_second": _rate(samples), "values_per_second": _quantity_rate(self.values_processed, elapsed),
            "bytes_per_second": _quantity_rate(self.bytes_processed, elapsed), "eta_seconds": eta,
            "eta_timestamp": None if eta is None else _now() + eta,
            "eta_sample_count": len(samples), "eta_method": None if eta is None else "median_completed_dataset_wall_seconds",
            "last_completed_dataset": self.last_completed_dataset, "last_status": self.last_status,
            "last_reason": self.last_reason, "last_elapsed_seconds": self.last_elapsed_seconds,
            "error_type": self.error_type, "error_message": self.error_message,
            "backend": self.backend, "device": "mps", "mps_available": _mps_available(),
            "memory_workers": {"dataset_workers": 1, "mps_heavy_processes": 1},
            "mlx_assessment": _mlx_assessment(),
            "heartbeat_sequence": self.heartbeat_sequence,
        }
        _assert_label_free(payload)
        return payload


def _rate(samples: Sequence[float]) -> float | None:
    total = sum(samples)
    return None if not samples or total <= 0 else len(samples) / total


def _quantity_rate(quantity: int, elapsed: float) -> float | None:
    return None if quantity <= 0 or elapsed <= 0 else quantity / elapsed


def _robust_eta(samples: Sequence[float], remaining: int) -> float | None:
    return None if len(samples) < 2 else median(samples) * remaining


class AtomicProgressWriter:
    """Durable runtime state plus append-only event and parent-log streams."""

    def __init__(self, paths: RuntimePaths, run_identity: str, command: ExecutionCommand) -> None:
        self.paths = paths
        self.run_identity = run_identity
        self.command = command

    @classmethod
    def create(cls, *, output_root: Path, run_identity: str, command: ExecutionCommand, checkpoint_root: Path, total_datasets: int, backend: str) -> "AtomicProgressWriter":
        root = Path(output_root).absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        paths = RuntimePaths(root, checkpoint_root, root / "progress.json", root / "events.jsonl", root / "parent.log", root / "run.pid")
        writer = cls(paths, run_identity, command)
        writer._write_pid()
        return writer

    @classmethod
    def resume(cls, *, output_root: Path, run_identity: str, checkpoint_root: Path) -> "AtomicProgressWriter":
        root = Path(output_root).absolute()
        paths = RuntimePaths(root, checkpoint_root, root / "progress.json", root / "events.jsonl", root / "parent.log", root / "run.pid")
        if not root.is_dir() or not all(path.is_file() for path in (paths.progress, paths.events, paths.parent_log, paths.pid)):
            raise ValueError("runtime output is absent or incomplete for resume")
        payload = _read_canonical(paths.progress)
        if payload.get("run_identity") != run_identity or payload.get("checkpoint_root") != str(paths.checkpoint_root):
            raise ValueError("runtime output does not match checkpoint identity")
        writer = cls(paths, run_identity, "resume")
        # Validate the complete append-only evidence before changing PID or
        # progress.  A truncated/tampered event stream is never repaired by
        # resume and therefore cannot be double-counted.
        writer.read_terminal_samples()
        writer._write_pid()
        return writer

    def _write_pid(self) -> None:
        _atomic_write(self.paths.pid, (str(os.getpid()) + "\n").encode("ascii"))

    def write(self, progress: RuntimeProgress, *, event: str, detail: Mapping[str, object] | None = None) -> None:
        if not isinstance(progress, RuntimeProgress) or progress.paths != self.paths or progress.run_identity != self.run_identity:
            raise ValueError("runtime progress writer binding is invalid")
        state = progress.payload()
        _atomic_write(self.paths.progress, canonical_json_bytes(state))
        event_payload = {"schema": "exp425.runtime-event.v1", "event": event, "timestamp": _now(), "run_identity": self.run_identity, "pid": os.getpid(), "state": progress.state, "stage": progress.stage, "current_dataset": progress.current_dataset, **(dict(detail) if detail else {})}
        _assert_label_free(event_payload)
        line = canonical_json_bytes(event_payload) + b"\n"
        _append_fsync(self.paths.events, line)
        human = (
            f"{event_payload['timestamp']:.6f} pid={os.getpid()} run={self.run_identity} "
            f"state={progress.state} stage={progress.stage} event={event} "
            f"dataset={event_payload.get('dataset', progress.current_dataset) or '-'} "
            f"ordinal={event_payload.get('ordinal', progress.ordinal)} "
            f"completed={event_payload.get('completed_datasets', progress.counts.completed)} "
            f"remaining={event_payload.get('remaining_datasets', progress.counts.remaining)}\n"
        )
        _append_fsync(self.paths.parent_log, human.encode("utf-8"))

    def read_terminal_samples(self) -> tuple[float, ...]:
        samples: list[float] = []
        seen: set[str] = set()
        raw = self.paths.events.read_bytes()
        for line in raw.splitlines():
            payload = _canonical_json_line(line)
            if payload.get("event") != "dataset_terminal":
                continue
            dataset, elapsed = payload.get("dataset"), payload.get("elapsed_seconds")
            if not isinstance(dataset, str) or dataset in seen or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
                raise ValueError("runtime terminal event log is invalid")
            seen.add(dataset); samples.append(float(elapsed))
        return tuple(samples)

    def close(self) -> None:
        try:
            self.paths.pid.unlink()
            _fsync_directory(self.paths.pid.parent)
        except FileNotFoundError:
            pass


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_fsync(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_line(raw: bytes) -> dict[str, object]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError("runtime JSONL line is not canonical")
    _assert_label_free(payload)
    return payload


def _read_canonical(path: Path) -> dict[str, object]:
    return _canonical_json_line(path.read_bytes())


@dataclass(frozen=True)
class DatasetRuntimeTrace:
    dataset: str
    origin: str
    elapsed_seconds: float
    result_evidence_sha256: str
    eligibility_sha256: str
    processed_groups_sha256: str
    model_evidence_sha256: str
    frozen_train_contract_sha256: str
    scoring_evidence_sha256: str
    backend_audit_sha256: str
    trace_sha256: str


@dataclass(frozen=True)
class Exp425RunResult:
    state: str
    counts: Exp425RunCounts
    traces: tuple[DatasetRuntimeTrace, ...]
    run_identity: str
    global_workload_status: str


_SESSION_SENTINEL = object()


class Exp425RuntimeSession:
    """Live registered native-write authority; construction is package-private."""

    __slots__ = ("snapshot", "audit", "writer", "process_nonce", "runner", "_closed", "_lock_fd", "_eligibility", "_fold_contract_sha256", "_sentinel")

    def __init__(self, sentinel: object, snapshot: Exp425DatabaseSnapshot, audit: BackendAudit, progress: AtomicProgressWriter) -> None:
        if sentinel is not _SESSION_SENTINEL:
            raise TypeError("Exp425 runtime session construction is private")
        if not isinstance(snapshot, Exp425DatabaseSnapshot) or not isinstance(audit, BackendAudit) or not isinstance(progress, AtomicProgressWriter):
            raise ValueError("Exp425 runtime inputs are invalid")
        if audit.backend != "mps" or audit.fallback or not audit.candidate_identity:
            raise ValueError("Exp425 native execution requires a passing MPS audit")
        _require_mps()
        self.snapshot, self.audit, self.writer = snapshot, audit, progress
        self.process_nonce = uuid4().hex
        self._closed = False
        self._eligibility: dict[str, Mapping[int, object]] = {}
        self._fold_contract_sha256: str | None = None
        self._lock_fd = _acquire_heavy_process_guard()
        self.runner = RunnerRuntime(
            verify=lambda context: self.snapshot.verify_boundary("runtime:dataset"),
            load_train=self.snapshot.load_train,
            preprocess=self._preprocess,
            train_models=lambda groups: _production_train_models(groups, self.audit),
            qualify=_production_qualify_train,
            freeze=lambda dataset, qualification, models: __import__("experiments.exp417_causal_multihorizon_shadow.qualification", fromlist=["freeze_train_contract"]).freeze_train_contract(dataset, qualification.groups, qualification.backend, train_identity_sha256=models.train_identity_sha256),
            load_test=self.snapshot.load_test,
            score=_production_score_test,
        )
        capabilities._register_authority(self, "exp425_native")

    def __copy__(self) -> object:
        raise TypeError("Exp425 runtime sessions cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Exp425 runtime sessions cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Exp425 runtime sessions cannot be pickled")

    def _require_live(self) -> None:
        if self._closed:
            raise ValueError("Exp425 runtime session is closed")

    def _preprocess(self, train: object) -> Sequence[object]:
        if self._fold_contract_sha256 is None:
            raise ValueError("runtime checkpoint fold contract is not bound")
        groups = preprocess_length_groups(train)  # type: ignore[arg-type]
        assessment = apply_group_eligibility(train.dataset, groups, fold_contract_sha256=self._fold_contract_sha256)  # type: ignore[union-attr]
        self._eligibility[train.dataset] = assessment.evidence  # type: ignore[union-attr]
        return assessment.groups

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        capabilities._close_authority(self)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_fd)

    def __enter__(self) -> "Exp425RuntimeSession":
        self._require_live()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _mlx_assessment() -> dict[str, object]:
    """Record the AGENTS.md gate result without changing Exp424 computation."""
    try:
        import mlx  # type: ignore[import-not-found]
        version = getattr(mlx, "__version__", "unknown")
    except Exception:
        version = None
    return {
        "adopted": False,
        "mlx_version": version,
        "reason": "frozen_exp424_mps_path_has_no_verified_exact_mlx_parity_and_1_1x_speed_gate",
    }


def _require_mps() -> None:
    try:
        import torch
        if not torch.backends.mps.is_available() or torch.device("mps").type != "mps":
            raise ValueError("MPS is unavailable; CPU fallback is forbidden")
    except ImportError as error:
        raise ValueError("MPS is unavailable; CPU fallback is forbidden") from error


def _acquire_heavy_process_guard() -> int:
    path = Path("/tmp/exp425-selective-recovery-mps.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise ValueError("another Exp425 MPS-heavy process is already active") from error
    return descriptor


def build_exp425_runtime(snapshot: Exp425DatabaseSnapshot, audit: BackendAudit, progress: AtomicProgressWriter) -> Exp425RuntimeSession:
    return Exp425RuntimeSession(_SESSION_SENTINEL, snapshot, audit, progress)


def execute_runtime_command(
    command: ExecutionCommand,
    *,
    output_root: Path,
    checkpoint_root: Path,
    provenance: object,
    context: FrozenRunContext,
    snapshot: Exp425DatabaseSnapshot,
    audit: BackendAudit,
    importer: object | None = None,
) -> Exp425RunResult:
    """Execute the explicit ``new``/``resume`` runtime state machine once.

    The launch assembler is deliberately outside this module: it is the only
    component allowed to create the Task 4 snapshot and, for a new run, call
    the already-validated Task 3 importer.  That separation keeps ``inspect``
    unable to accidentally open a database or source snapshot.
    """
    if command not in {"new", "resume"}:
        raise ValueError("runtime command must be new or resume")
    output_root, checkpoint_root = Path(output_root).absolute(), Path(checkpoint_root).absolute()
    if command == "new" and (output_root.exists() or checkpoint_root.exists()):
        raise ValueError("new runtime output or checkpoint root already exists")
    if command == "resume" and (not output_root.is_dir() or not checkpoint_root.is_dir()):
        raise ValueError("resume runtime output or checkpoint root is absent")
    if not isinstance(context, FrozenRunContext) or not isinstance(snapshot, Exp425DatabaseSnapshot) or not isinstance(audit, BackendAudit):
        raise ValueError("runtime launch inputs are invalid")
    from .contracts import Exp425RunProvenance
    if not isinstance(provenance, Exp425RunProvenance):
        raise ValueError("runtime provenance is invalid")
    if command == "new":
        writer = AtomicProgressWriter.create(output_root=output_root, run_identity=provenance.run_identity, command="new", checkpoint_root=checkpoint_root, total_datasets=len(context.datasets), backend=audit.backend)
        store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
        if importer is not None:
            if not callable(importer):
                writer.close()
                raise ValueError("runtime importer must be callable")
            importer(store, context)
    else:
        writer = AtomicProgressWriter.resume(output_root=output_root, run_identity=provenance.run_identity, checkpoint_root=checkpoint_root)
        store = Exp425CheckpointStore.open(checkpoint_root, provenance, context)
    session: Exp425RuntimeSession | None = None
    try:
        session = build_exp425_runtime(snapshot, audit, writer)
        return run_selective_population(context, store, session)
    finally:
        if session is not None:
            session.close()
        writer.close()


def _effective_payload(result: DatasetResult) -> dict[str, object]:
    status = "scored" if result.status == "qualified" else result.status
    return {"status": status, "candidate_coordinates": [[candidate.dataset, candidate.instance_index] for candidate in result.candidates], "result_evidence_sha256": canonical_result_sha256(result)}


def _audit_digest(audit: BackendAudit) -> str:
    return _digest(asdict(audit))


def _eligibility_payload(evidence: Mapping[int, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for length, item in sorted(evidence.items()):
        item_payload = asdict(item)
        output[str(length)] = item_payload
    _assert_label_free(output)
    return output


def _native_record(store: Exp425CheckpointStore, runtime: Exp425RuntimeSession, result: DatasetResult, elapsed: float) -> tuple[NativeRecordEnvelope, DatasetRuntimeTrace]:
    evidence = runtime._eligibility.get(result.dataset)
    if evidence is None:
        raise ValueError("native result has no TRAIN eligibility evidence")
    group_evidence = _eligibility_payload(evidence)
    eligibility_sha = _digest(group_evidence)
    contract = result.decision_evidence.frozen_train_contract
    if contract is None:
        raise ValueError("native result has unexpected terminal evidence")
    processed_sha = _digest({"groups": [(group.group.length, group.status) for group in contract.groups]})
    model_sha = _digest({"train_identity_sha256": contract.train_identity_sha256, "backend": runtime.audit.backend})
    frozen_contract_sha = _digest({"dataset": result.dataset, "groups": [(group.length, group.status) for group in contract.groups], "train_identity_sha256": contract.train_identity_sha256})
    result_payload = _effective_payload(result)
    scoring_sha = _digest({"result_evidence_sha256": result_payload["result_evidence_sha256"]})
    audit_sha = _audit_digest(runtime.audit)
    values = {
        "train_identity_sha256": contract.train_identity_sha256, "eligibility_sha256": eligibility_sha,
        "processed_groups_sha256": processed_sha, "model_evidence_sha256": model_sha,
        "frozen_train_contract_sha256": frozen_contract_sha, "scoring_evidence_sha256": scoring_sha,
        "result_evidence_sha256": result_payload["result_evidence_sha256"], "backend_audit_sha256": audit_sha,
    }
    trace_body = {"schema": "exp425.native-trace.v1", "run_identity": store.provenance.run_identity, "dataset": result.dataset, "run_context_sha256": store.run_context_sha256, **values, "device": "mps"}
    trace_sha = _digest(trace_body)
    payload = {"effective_result": result_payload, "native_trace_body": trace_body}
    models = {"train_identity_sha256": values["train_identity_sha256"], "model_evidence_sha256": model_sha, "frozen_train_contract_sha256": frozen_contract_sha}
    calibration = {"scoring_evidence_sha256": scoring_sha}
    statuses = {getattr(item, "status", None) for item in evidence.values()}
    reason = "oof_identity_insufficient" if statuses == {"oof_identity_insufficient"} else "native_mps"
    provisional = NativeRecordEnvelope(result.dataset, payload, {"eligibility_sha256": eligibility_sha, "processed_groups_sha256": processed_sha, "evidence": group_evidence}, trace_sha256=trace_sha, record_payload_sha256="0" * 64, model_hashes=models, calibration_hashes=calibration, elapsed_seconds=elapsed, reason=reason, **values)
    record_sha = _digest(_native_record_body(store.run_context_sha256, provisional))
    record = replace(provisional, record_payload_sha256=record_sha)
    # Full decoded validation happens in the store before the capability is consumed.
    trace = DatasetRuntimeTrace(result.dataset, "exp425_native", elapsed, str(values["result_evidence_sha256"]), eligibility_sha, processed_sha, model_sha, frozen_contract_sha, scoring_sha, audit_sha, trace_sha)
    return record, trace


def _compute_native(context: FrozenRunContext, store: Exp425CheckpointStore, runtime: Exp425RuntimeSession, dataset: str, train: object) -> tuple[NativeRecordEnvelope, DatasetRuntimeTrace]:
    """Private exact-signature seam; tests may replace it, public callers cannot."""
    started = perf_counter()
    original_loader = runtime.runner.load_train
    runtime.runner = replace(runtime.runner, load_train=lambda name, frozen: train if name == dataset and frozen is context else original_loader(name, frozen))
    try:
        result = run_dataset(dataset, context, runtime=runtime.runner)
    finally:
        runtime.runner = replace(runtime.runner, load_train=original_loader)
    elapsed = perf_counter() - started
    if result.status == "failed" or result.decision_evidence.__class__.__name__ == "DatasetTerminalEvidence":
        raise ValueError("native frozen runtime returned unexpected terminal or failed result")
    return _native_record(store, runtime, result, elapsed)


def _terminal_event(progress: RuntimeProgress, record: object, elapsed: float, *, ordinal: int, status: str, reason: str | None, origin: str) -> RuntimeProgress:
    counts = progress.counts
    assert counts is not None
    imported = counts.imported + (origin == "exp424_import")
    native = counts.native + (origin == "exp425_native")
    abstained = counts.abstained + (status == "abstained")
    failed = counts.failed + (status == "failed")
    next_counts = Exp425RunCounts(counts.total, counts.completed + 1, imported, native, counts.recomputed, abstained, failed)
    return replace(progress, timestamp=_now(), counts=next_counts, current_dataset=None, origin=None, current_phase=None, ordinal=ordinal, last_terminal_time=_now(), last_completed_dataset=getattr(record, "dataset", None), last_status=status, last_reason=reason, last_elapsed_seconds=elapsed, cumulative_seconds=progress.cumulative_seconds + elapsed, terminal_samples=progress.terminal_samples + (elapsed,), train_instance_count=None, test_instance_count=None, size_summary={})


def _record_status(record: object) -> tuple[str, str | None]:
    payload = thaw_payload(record.result_payload)  # type: ignore[union-attr]
    if not isinstance(payload, Mapping) or not isinstance(payload.get("status"), str):
        raise ValueError("checkpoint effective result is invalid")
    return str(payload["status"]), getattr(record, "reason", None)


def run_selective_population(context: FrozenRunContext, store: Exp425CheckpointStore, runtime: Exp425RuntimeSession) -> Exp425RunResult:
    """Run only absent records sequentially, preserving compact checkpoint semantics."""
    if not isinstance(context, FrozenRunContext) or not isinstance(store, Exp425CheckpointStore) or not isinstance(runtime, Exp425RuntimeSession):
        raise ValueError("Exp425 selective runtime inputs are invalid")
    runtime._require_live()
    if store.context != context or store.expected_datasets != context.datasets:
        raise ValueError("runtime checkpoint universe does not match frozen context")
    runtime._fold_contract_sha256 = store.provenance.fold_contract_sha256
    progress = RuntimeProgress.initializing(runtime.writer.paths, store.provenance.run_identity, runtime.writer.command, len(context.datasets), runtime.audit.backend)
    traces: list[DatasetRuntimeTrace] = []
    try:
        runtime.writer.write(progress, event="phase_change")
        progress = replace(progress, state="importing", stage="importing", timestamp=_now())
        runtime.writer.write(progress, event="phase_change")
        progress = replace(progress, state="running", stage="running", timestamp=_now(), terminal_samples=runtime.writer.read_terminal_samples() if runtime.writer.command == "resume" else ())
        runtime.writer.write(progress, event="phase_change")
        for ordinal, dataset in enumerate(context.datasets, start=1):
            completed = store.completed_record_digests()
            if dataset in completed:
                record = store.load_record(dataset)
                status, reason = _record_status(record)
                progress = _terminal_event(progress, record, record.elapsed_seconds, ordinal=ordinal, status=status, reason=reason, origin=record.origin)
                runtime.writer.write(progress, event="dataset_terminal", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": record.origin, "status": status, "reason": reason, "elapsed_seconds": record.elapsed_seconds, "completed_datasets": progress.counts.completed, "remaining_datasets": progress.counts.remaining})
                progress = replace(progress, heartbeat_sequence=progress.heartbeat_sequence + 1, timestamp=_now())
                runtime.writer.write(progress, event="heartbeat")
                continue
            runtime.snapshot.verify_boundary("runtime:before_dataset")
            train = runtime.snapshot.load_train(dataset, context)
            sizes = _size_summary(train)
            progress = replace(progress, timestamp=_now(), current_dataset=dataset, ordinal=ordinal, origin="exp425_native", current_phase="dataset", dataset_start_time=_now(), train_instance_count=len(train.series), test_instance_count=context.test_sizes[dataset], size_summary=sizes)
            runtime.writer.write(progress, event="dataset_start", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": "exp425_native", "phase": "train"})
            record, trace = _compute_native(context, store, runtime, dataset, train)
            elapsed = record.elapsed_seconds
            capability = capabilities._mint(runtime, kind="native", run_identity=store.provenance.run_identity, dataset=dataset, context_sha256=store.run_context_sha256, trace_sha256=record.trace_sha256, record_sha256=record.record_payload_sha256, backend_sha256=record.backend_audit_sha256, process_nonce=runtime.process_nonce)
            store._commit_native(capability, record)
            traces.append(trace)
            status, reason = _record_status(record)
            progress = _terminal_event(progress, record, elapsed, ordinal=ordinal, status=status, reason=reason, origin=record.origin)
            runtime.writer.write(progress, event="dataset_terminal", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": record.origin, "status": status, "reason": reason, "elapsed_seconds": elapsed, "completed_datasets": progress.counts.completed, "remaining_datasets": progress.counts.remaining})
            progress = replace(progress, heartbeat_sequence=progress.heartbeat_sequence + 1, timestamp=_now())
            runtime.writer.write(progress, event="heartbeat")
        if progress.counts.completed != len(context.datasets) or progress.counts.failed:
            raise ValueError("runtime population accounting is incomplete or failed")
        # Applying Exp417's exact workload rule requires the live DatasetResult
        # objects.  Imported records have intentionally been compacted, so their
        # persisted coordinate count is checked against the same frozen threshold.
        candidate_count = sum(len(thaw_payload(store.load_record(dataset).result_payload).get("candidate_coordinates", [])) for dataset in context.datasets)
        if candidate_count / sum(context.test_sizes.values()) > context.config.global_workload_rate:
            raise ValueError("frozen global workload gate failed")
        progress = replace(progress, state="finalizing", stage="finalizing", timestamp=_now())
        runtime.writer.write(progress, event="finalization")
        progress = replace(progress, state="complete_review_only", stage="complete_review_only", timestamp=_now(), current_dataset=None, current_phase=None)
        runtime.writer.write(progress, event="finalization")
        return Exp425RunResult(progress.state, progress.counts, tuple(traces), store.provenance.run_identity, "complete")
    except Exception as error:
        progress = replace(progress, state="failed", stage="failed", timestamp=_now(), current_dataset=None, current_phase=None, error_type=type(error).__name__, error_message=_safe_message(error))
        runtime.writer.write(progress, event="failure", detail={"error_type": type(error).__name__, "error_message": _safe_message(error)})
        return Exp425RunResult("failed", progress.counts, tuple(traces), store.provenance.run_identity, "failed")


def _size_summary(train: object) -> dict[str, int]:
    series = tuple(getattr(train, "series", ()))
    values = sum(len(item.values) for item in series)
    bytes_count = values * 4
    lengths: dict[int, int] = {}
    for item in series:
        lengths[len(item.values)] = lengths.get(len(item.values), 0) + 1
    return {"train_values": values, "train_bytes": bytes_count, "length_groups": len(lengths), "minimum_length": min(lengths, default=0), "maximum_length": max(lengths, default=0)}


def inspect_runtime_output(output_root: Path, *, expected_run_identity: str | None = None) -> Mapping[str, object]:
    """Read a published runtime state only; this is intentionally write-free."""
    root = Path(output_root).absolute()
    payload = _read_canonical(root / "progress.json")
    if expected_run_identity is not None and payload.get("run_identity") != expected_run_identity:
        raise ValueError("runtime progress identity does not match")
    return MappingProxyType(payload)


__all__ = [
    "AtomicProgressWriter", "DatasetRuntimeTrace", "ExecutionCommand", "Exp425RunCounts",
    "Exp425RunResult", "Exp425RuntimeSession", "RuntimePaths", "RuntimeProgress",
    "build_exp425_runtime", "inspect_runtime_output", "run_selective_population",
    "execute_runtime_command",
]
