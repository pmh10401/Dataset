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
import subprocess
from pathlib import Path
import pickle
from statistics import median
from threading import Event, RLock, Thread
from time import perf_counter, time
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import canonical_result_sha256, _context_payload
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenRunContext,
    GroupQualification,
)
from experiments.exp417_causal_multihorizon_shadow import qualification as exp417_qualification
from experiments.exp417_causal_multihorizon_shadow.runner import (
    RunnerRuntime,
    TrainModelEvidence,
    TestScoringEvidence,
    _production_qualify_train,
    _production_score_test,
    _production_train_models,
    run_dataset,
)
from experiments.exp417_causal_multihorizon_shadow.model import fit_final_seed_models, train_oof_models
from experiments.exp417_causal_multihorizon_shadow.preprocessing import preprocess_length_groups

from . import capabilities
from .checkpoints import Exp425CheckpointStore, _native_record_body
from .contracts import MPSAdoptionAssessment, MPSExecutionEvidence, NativeRecordEnvelope, thaw_payload
from .database import Exp425DatabaseSnapshot
from .eligibility import apply_group_eligibility


ExecutionCommand = Literal["new", "resume", "inspect"]
_STATES = frozenset({"initializing", "importing", "running", "finalizing", "complete_review_only", "failed"})
_FORBIDDEN_KEY_TOKENS = ("label", "target", "groundtruth", "anomalylocation", "evaluation", "confusion", "precision", "recall", "familytestperformance")
_HEARTBEAT_INTERVAL_SECONDS = 30.0


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
    oof_identity_insufficient: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not int or getattr(self, field) < 0 for field in self.__dataclass_fields__):
            raise ValueError("runtime counters must be non-negative integers")
        if (
            self.completed > self.total
            or self.imported + self.native != self.completed
            or self.recomputed > self.native
            or self.abstained > self.completed
            or self.oof_identity_insufficient > self.abstained
            or self.failed > self.completed
        ):
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
            "oof_identity_insufficient_datasets": self.counts.oof_identity_insufficient,
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
        # This nonce makes a PID lease owner-bound.  A later process must not
        # remove a lease it did not create merely because it happens to know
        # the run directory.
        self._pid_nonce = uuid4().hex
        self._pid_claim: bytes | None = None
        self._write_lock = RLock()
        self._heartbeat_sequence = 0

    @classmethod
    def create(cls, *, output_root: Path, run_identity: str, command: ExecutionCommand, checkpoint_root: Path, total_datasets: int, backend: str) -> "AtomicProgressWriter":
        root = Path(output_root).absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        paths = RuntimePaths(root, checkpoint_root, root / "progress.json", root / "events.jsonl", root / "parent.log", root / "run.pid")
        writer = cls(paths, run_identity, command)
        writer.claim_pid()
        return writer

    @classmethod
    def resume(
        cls, *, output_root: Path, run_identity: str, checkpoint_root: Path,
        claim_pid: bool = True,
    ) -> "AtomicProgressWriter":
        root = Path(output_root).absolute()
        paths = RuntimePaths(root, checkpoint_root, root / "progress.json", root / "events.jsonl", root / "parent.log", root / "run.pid")
        # A completed or failed inactive run intentionally has no PID lease.
        # It is therefore not part of the durable resume inventory.
        if not root.is_dir() or not all(path.is_file() for path in (paths.progress, paths.events, paths.parent_log)):
            raise ValueError("runtime output is absent or incomplete for resume")
        payload = _read_canonical(paths.progress)
        if payload.get("run_identity") != run_identity or payload.get("checkpoint_root") != str(paths.checkpoint_root):
            raise ValueError("runtime output does not match checkpoint identity")
        writer = cls(paths, run_identity, "resume")
        # Validate the complete append-only evidence before changing PID or
        # progress.  A truncated/tampered event stream is never repaired by
        # resume and therefore cannot be double-counted.
        writer.read_terminal_samples()
        if claim_pid:
            writer._write_pid()
        return writer

    def _pid_payload(self) -> bytes:
        return canonical_json_bytes({
            "schema": "exp425.runtime-pid.v1", "run_identity": self.run_identity,
            "pid": os.getpid(), "nonce": self._pid_nonce,
        })

    @staticmethod
    def _pid_is_live(payload: Mapping[str, object]) -> bool:
        pid = payload.get("pid")
        if type(pid) is not int or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_pid(self) -> tuple[dict[str, object], bytes]:
        raw = self.paths.pid.read_bytes()
        payload = _canonical_json_line(raw)
        if set(payload) != {"schema", "run_identity", "pid", "nonce"} or payload.get("schema") != "exp425.runtime-pid.v1":
            raise ValueError("runtime PID lease is invalid")
        if payload.get("run_identity") != self.run_identity or not isinstance(payload.get("nonce"), str):
            raise ValueError("runtime PID lease does not match this run")
        return payload, raw

    # Kept private for older focused tests; it has the same owner-bound claim
    # semantics and never overwrites a live lease.
    def _write_pid(self) -> None:
        self.claim_pid()

    def write(self, progress: RuntimeProgress, *, event: str, detail: Mapping[str, object] | None = None) -> None:
        if not isinstance(progress, RuntimeProgress) or progress.paths != self.paths or progress.run_identity != self.run_identity:
            raise ValueError("runtime progress writer binding is invalid")
        with self._write_lock:
            sequence = max(self._heartbeat_sequence, progress.heartbeat_sequence)
            if event == "heartbeat":
                sequence += 1
            self._heartbeat_sequence = sequence
            progress = replace(progress, heartbeat_sequence=sequence, timestamp=_now())
            state = progress.payload()
            _atomic_write(self.paths.progress, canonical_json_bytes(state))
            event_payload = {"schema": "exp425.runtime-event.v1", "event": event, "timestamp": _now(), "run_identity": self.run_identity, "pid": os.getpid(), "state": progress.state, "stage": progress.stage, "current_dataset": progress.current_dataset, "heartbeat_sequence": sequence, "elapsed_seconds": state["elapsed_seconds"], "cumulative_seconds": progress.cumulative_seconds, "backend": progress.backend, "checkpoint_root": str(progress.paths.checkpoint_root), **(dict(detail) if detail else {})}
            _assert_label_free(event_payload)
            line = canonical_json_bytes(event_payload) + b"\n"
            _append_fsync(self.paths.events, line)
            human = (
                f"{event_payload['timestamp']:.6f} pid={os.getpid()} run={self.run_identity} "
                f"state={progress.state} stage={progress.stage} event={event} "
                f"dataset={event_payload.get('dataset', progress.current_dataset) or '-'} "
                f"ordinal={event_payload.get('ordinal', progress.ordinal)} "
                f"completed={event_payload.get('completed_datasets', progress.counts.completed)} "
                f"remaining={event_payload.get('remaining_datasets', progress.counts.remaining)} "
                f"elapsed={state['elapsed_seconds']:.3f} backend={progress.backend} "
                f"checkpoint={progress.paths.checkpoint_root}\n"
            )
            _assert_label_free({"log": human})
            _append_fsync(self.paths.parent_log, human.encode("utf-8"))

    def read_terminal_samples(self) -> tuple[float, ...]:
        return tuple(event[1] for event in self.read_terminal_events())

    def read_terminal_events(self) -> tuple[tuple[str, float], ...]:
        samples: list[float] = []
        events: list[tuple[str, float]] = []
        seen: set[str] = set()
        raw = self.paths.events.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("runtime event log is truncated")
        for line in raw.splitlines():
            payload = _canonical_json_line(line)
            if payload.get("schema") != "exp425.runtime-event.v1" or payload.get("run_identity") != self.run_identity:
                raise ValueError("runtime event log does not match this run")
            if payload.get("event") != "dataset_terminal":
                continue
            dataset, elapsed = payload.get("dataset"), payload.get("elapsed_seconds")
            if not isinstance(dataset, str) or dataset in seen or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
                raise ValueError("runtime terminal event log is invalid")
            seen.add(dataset); samples.append(float(elapsed)); events.append((dataset, float(elapsed)))
        return tuple(events)

    def close(self) -> None:
        try:
            if self._pid_claim is None:
                return
            # Compare exact bytes before unlinking.  This prevents an orderly
            # old owner from deleting a newer owner's lease.
            try:
                if self.paths.pid.read_bytes() != self._pid_claim:
                    return
            except FileNotFoundError:
                return
            self.paths.pid.unlink()
            _fsync_directory(self.paths.pid.parent)
        except FileNotFoundError:
            pass

    def claim_pid(self) -> None:
        """Claim the live PID only after every resume validation has succeeded."""
        if self._pid_claim is not None:
            return
        payload = self._pid_payload()
        while True:
            try:
                descriptor = os.open(
                    self.paths.pid,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
            except FileExistsError:
                existing, _ = self._read_pid()
                if self._pid_is_live(existing):
                    raise ValueError("runtime PID lease is owned by a live process")
                # Only a stale, fully validated lease for this exact run can
                # be replaced.  Do not unlink malformed or foreign content.
                try:
                    self.paths.pid.unlink()
                    _fsync_directory(self.paths.pid.parent)
                except FileNotFoundError:
                    pass
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.paths.pid.parent)
            self._pid_claim = payload
            return


class _HeartbeatPump:
    """One harmless periodic writer for a deliberately long dataset stage."""

    def __init__(self, writer: AtomicProgressWriter, progress: RuntimeProgress, *, interval: float | None = None) -> None:
        self._writer = writer
        self._progress = progress
        self._interval = _HEARTBEAT_INTERVAL_SECONDS if interval is None else interval
        if not isinstance(self._interval, (int, float)) or self._interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        self._stop = Event()
        self._thread = Thread(target=self._run, name="exp425-heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(float(self._interval)):
            self._writer.write(self._progress, event="heartbeat", detail={"phase": self._progress.current_phase or self._progress.stage})

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, float(self._interval) * 2.0))
        if self._thread.is_alive():
            raise RuntimeError("heartbeat thread did not stop")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
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
_TASK7_AUDIT_PATH = "/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow/task7_backend_audit.json"
_TASK7_AUDIT_SHA256 = "f77efa941f6b33fe6687b8ac0c2b01c152dac9c15e8b0966e3d9bb41b119b592"
_TASK7_METRICS = (4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965)
_TASK7_SOURCE_COMMIT = "4a398374d880f859f285a271ad63de4a5e55b028"
_SOURCE_HASHES = {"runner": "f784fb77805e353b7652582626672686f52a2fdc06d27f9c1ca58730d27b772c", "model": "422e096613be84ba77350b068e6bd837eebbb46169bb6e3e275f3f03ab2fcedd", "scoring": "06a92422ea7909efbb7354f1f0499329bd19c04642bbbd5a136fa3f2f5a39941", "entrypoint": "98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1"}
_SOURCE_ARTIFACT_PATHS = {
    "runner": "experiments/exp417_causal_multihorizon_shadow/runner.py",
    "model": "experiments/exp417_causal_multihorizon_shadow/model.py",
    "scoring": "experiments/exp417_causal_multihorizon_shadow/scoring.py",
    "entrypoint": "scratch/run_exp424_mps_score_cache.py",
}


def _validate_task7_evidence(audit: BackendAudit, adoption: MPSAdoptionAssessment) -> None:
    """Authenticate the frozen Task7 artifact before a native authority exists."""
    try:
        raw = Path(_TASK7_AUDIT_PATH).read_bytes()
        task7 = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Task7 MPS audit artifact is unavailable") from error
    expected_audit = {
        "backend": "cpu", "candidate_identity": True, "fallback": False,
        "fallback_reason": None, "max_normalized_error": _TASK7_METRICS[0],
        "spearman": _TASK7_METRICS[1], "speedup": _TASK7_METRICS[2],
    }
    if (
        sha256(raw).hexdigest() != _TASK7_AUDIT_SHA256
        or not isinstance(task7, dict)
        or task7.get("schema") != "exp417.task7-backend-audit.v1"
        or task7.get("selected_backend") != "cpu"
        or task7.get("execution_audit") != expected_audit
        or audit.backend != "cpu" or audit.fallback or not audit.candidate_identity
        or (audit.max_normalized_error, audit.spearman, audit.speedup) != _TASK7_METRICS
        or adoption.payload() != MPSAdoptionAssessment(
            _TASK7_AUDIT_PATH, _TASK7_AUDIT_SHA256, True, *_TASK7_METRICS,
            1.1, False, "speedup_below_threshold",
        ).payload()
    ):
        raise ValueError("Exp425 native execution requires authentic Task7 MPS audit evidence")


def _validate_source_artifacts() -> None:
    root = Path(__file__).resolve().parents[2]
    observed = {
        name: sha256((root / relative).read_bytes()).hexdigest()
        for name, relative in _SOURCE_ARTIFACT_PATHS.items()
    }
    if observed != _SOURCE_HASHES:
        raise ValueError("Exp425 native source artifacts do not match the frozen Task7 provenance")


class Exp425RuntimeSession:
    """Live registered native-write authority; construction is package-private."""

    __slots__ = ("_snapshot", "_audit", "_adoption", "_writer", "_executed_head", "process_nonce", "_runner", "_closed", "_lock_fd", "_eligibility", "_processed", "_models", "_scoring", "_execution", "_active_dataset", "_fold_contract_sha256", "_binding", "_sealed")

    def __init__(self, sentinel: object, snapshot: Exp425DatabaseSnapshot, audit: BackendAudit, adoption: MPSAdoptionAssessment, progress: AtomicProgressWriter, executed_head: str) -> None:
        if sentinel is not _SESSION_SENTINEL:
            raise TypeError("Exp425 runtime session construction is private")
        if not isinstance(snapshot, Exp425DatabaseSnapshot) or not isinstance(audit, BackendAudit) or not isinstance(adoption, MPSAdoptionAssessment) or not isinstance(progress, AtomicProgressWriter):
            raise ValueError("Exp425 runtime inputs are invalid")
        if not isinstance(executed_head, str) or len(executed_head) != 40 or any(char not in "0123456789abcdef" for char in executed_head):
            raise ValueError("Exp425 native execution head is invalid")
        _validate_task7_evidence(audit, adoption)
        _validate_source_artifacts()
        _require_mps()
        object.__setattr__(self, "_sealed", False)
        self._snapshot, self._audit, self._adoption, self._writer, self._executed_head = snapshot, audit, adoption, progress, executed_head
        self.process_nonce = uuid4().hex
        self._closed = False
        self._eligibility: dict[str, Mapping[int, object]] = {}
        self._processed: dict[str, tuple[object, ...]] = {}
        self._models: dict[str, TrainModelEvidence] = {}
        self._scoring: dict[str, TestScoringEvidence] = {}
        self._execution: dict[str, MPSExecutionEvidence] = {}
        self._active_dataset: str | None = None
        self._fold_contract_sha256: str | None = None
        self._lock_fd = _acquire_heavy_process_guard()
        self._runner = RunnerRuntime(
            verify=lambda context: self.snapshot.verify_boundary("runtime:dataset"),
            load_train=self.snapshot.load_train,
            preprocess=self._preprocess,
            train_models=self._train_models,
            qualify=_production_qualify_train,
            freeze=lambda dataset, qualification, models: __import__("experiments.exp417_causal_multihorizon_shadow.qualification", fromlist=["freeze_train_contract"]).freeze_train_contract(dataset, qualification.groups, qualification.backend, train_identity_sha256=models.train_identity_sha256),
            load_test=self.snapshot.load_test,
            score=self._score,
        )
        self._binding = (id(snapshot), id(audit), id(adoption), id(progress), id(self._runner), progress.run_identity)
        capabilities._register_authority(self, "exp425_native")
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name in {"snapshot", "audit", "adoption", "writer", "runner", "_snapshot", "_audit", "_adoption", "_writer", "_runner", "_binding"}:
            raise AttributeError("Exp425 authority-critical bindings are immutable")
        object.__setattr__(self, name, value)

    @property
    def snapshot(self) -> Exp425DatabaseSnapshot:
        return self._snapshot

    @property
    def audit(self) -> BackendAudit:
        return self._audit

    @property
    def adoption(self) -> MPSAdoptionAssessment:
        return self._adoption

    @property
    def writer(self) -> AtomicProgressWriter:
        return self._writer

    @property
    def runner(self) -> RunnerRuntime:
        return self._runner

    def __copy__(self) -> object:
        raise TypeError("Exp425 runtime sessions cannot be copied")

    def __deepcopy__(self, memo: object) -> object:
        raise TypeError("Exp425 runtime sessions cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("Exp425 runtime sessions cannot be pickled")

    def _require_live(self) -> None:
        if self._closed:
            raise ValueError("Exp425 runtime session is closed")

    def _verify_authority_binding(self) -> None:
        self._require_live()
        if self._binding != (id(self._snapshot), id(self._audit), id(self._adoption), id(self._writer), id(self._runner), self._writer.run_identity):
            raise ValueError("Exp425 authority-critical binding was mutated")
        _validate_task7_evidence(self._audit, self._adoption)
        _validate_source_artifacts()

    def _preprocess(self, train: object) -> Sequence[object]:
        if self._fold_contract_sha256 is None:
            raise ValueError("runtime checkpoint fold contract is not bound")
        groups = preprocess_length_groups(train)  # type: ignore[arg-type]
        assessment = apply_group_eligibility(train.dataset, groups, fold_contract_sha256=self._fold_contract_sha256)  # type: ignore[union-attr]
        self._eligibility[train.dataset] = assessment.evidence  # type: ignore[union-attr]
        self._processed[train.dataset] = tuple(assessment.groups)  # type: ignore[union-attr]
        return assessment.groups

    def _train_models(self, groups: Sequence[object]) -> TrainModelEvidence:
        """Capture the exact frozen model/calibration evidence used downstream."""
        import torch
        device = torch.device("mps")
        groups = tuple(groups)  # type: ignore[arg-type]
        usable = tuple(group for group in groups if getattr(group, "status", None) in {"candidate_eligible", "diagnostic_only"})
        finals = tuple(group for group in groups if getattr(group, "status", None) == "candidate_eligible")
        bundles = {group.group.length: train_oof_models(group, device=device) for group in usable}
        final_models = {group.group.length: fit_final_seed_models(group, device=device) for group in finals}
        models = TrainModelEvidence(groups, bundles, final_models, self.audit)  # type: ignore[arg-type]
        devices = {f"oof:{length}:{item.seed}": item.device for length, bundle in bundles.items() for item in bundle.models}
        devices.update({f"final:{length}:{item.seed}": item.device for length, items in final_models.items() for item in items})
        dataset = self._active_dataset
        if dataset is None:
            raise ValueError("native model evidence is not bound to preprocessed TRAIN")
        self._execution[dataset] = MPSExecutionEvidence(True, "mps", "mps", bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available()), False, devices, torch.__version__, _TASK7_SOURCE_COMMIT, _SOURCE_HASHES, self._executed_head, "_compute_native return after existing CPU/NumPy/serialization/calibration implicit synchronization", 0)
        self._models[dataset] = models
        return models

    def _score(self, test: object, contract: object, models: TrainModelEvidence) -> TestScoringEvidence:
        scoring = _production_score_test(test, contract, models)  # type: ignore[arg-type]
        dataset = self._active_dataset
        if dataset is None or getattr(contract, "dataset", None) != dataset:
            raise ValueError("native TEST scoring evidence is not bound to the active dataset")
        self._scoring[dataset] = scoring
        return scoring

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary: BaseException | None = None
        try:
            capabilities._close_authority(self)
        except BaseException as error:
            primary = error
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except BaseException as error:
            if primary is None:
                primary = error
        try:
            os.close(self._lock_fd)
        except BaseException as error:
            if primary is None:
                primary = error
        close_snapshot = getattr(self.snapshot, "close", None)
        if callable(close_snapshot):
            try:
                close_snapshot()
            except BaseException as error:
                if primary is None:
                    primary = error
        if primary is not None:
            raise primary

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


def build_exp425_runtime(snapshot: Exp425DatabaseSnapshot, audit: BackendAudit, adoption: MPSAdoptionAssessment, progress: AtomicProgressWriter, executed_head: str) -> Exp425RuntimeSession:
    return Exp425RuntimeSession(_SESSION_SENTINEL, snapshot, audit, adoption, progress, executed_head)


def _terminal_event_details(paths: RuntimePaths, run_identity: str) -> tuple[Mapping[str, object], ...]:
    raw = paths.events.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("runtime event log is truncated")
    terminal: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        event = _canonical_json_line(line)
        if event.get("schema") != "exp425.runtime-event.v1" or event.get("run_identity") != run_identity:
            raise ValueError("runtime event log does not match this run")
        if event.get("event") != "dataset_terminal":
            continue
        dataset = event.get("dataset")
        if not isinstance(dataset, str) or dataset in seen:
            raise ValueError("runtime terminal event log is invalid")
        seen.add(dataset)
        terminal.append(MappingProxyType(event))
    return tuple(terminal)


def _reconcile_runtime_events(runtime: Mapping[str, object], inspection: object, paths: RuntimePaths) -> None:
    """Accept only a checkpoint-committed terminal suffix missing from JSONL."""
    summaries = getattr(inspection, "record_summaries")
    expected = getattr(inspection, "expected_datasets")
    terminal = _terminal_event_details(paths, str(runtime["run_identity"]))
    completed = tuple(summary[0] for summary in summaries)
    logged = tuple(str(event["dataset"]) for event in terminal)
    if logged != completed[:len(logged)]:
        raise ValueError("runtime events do not match checkpoint completion order")
    for event, summary in zip(terminal, summaries):
        dataset, origin, status, reason, elapsed = summary
        if (
            event.get("ordinal") != expected.index(dataset) + 1
            or event.get("origin") != origin or event.get("status") != status
            or event.get("reason") != reason or event.get("elapsed_seconds") != elapsed
        ):
            raise ValueError("runtime terminal event does not match checkpoint record")
    # The predecessor may legitimately import a sparse subset.  Within that
    # frozen-order subset, missing events can only be the exact suffix caused
    # by a checkpoint-commit→JSONL crash.
    state = runtime.get("state")
    if state == "complete_review_only" and (len(logged) != len(expected) or getattr(inspection, "import_revalidation_required")):
        raise ValueError("complete runtime has unresolved events or checkpoint intents")


def preflight_runtime_paths(command: ExecutionCommand, *, output_root: Path, checkpoint_root: Path, run_identity: str | None = None, provenance: object | None = None, context: FrozenRunContext | None = None) -> None:
    """Validate command lifecycle before a CLI factory opens any live resource."""
    output_root, checkpoint_root = Path(output_root).absolute(), Path(checkpoint_root).absolute()
    if command == "new":
        if output_root.exists() or checkpoint_root.exists():
            raise ValueError("new runtime output or checkpoint root already exists")
        return
    if command != "resume":
        raise ValueError("runtime command must be new or resume")
    if not output_root.is_dir() or not checkpoint_root.is_dir():
        raise ValueError("resume runtime output or checkpoint root is absent")
    runtime = inspect_runtime_output(output_root, expected_run_identity=run_identity)
    inspected = Exp425CheckpointStore.inspect_read_only(checkpoint_root, expected_run_identity=run_identity)
    if runtime.get("checkpoint_root") != str(checkpoint_root):
        raise ValueError("runtime output does not bind the requested checkpoint root")
    if runtime.get("total_datasets") != len(inspected.expected_datasets):
        raise ValueError("runtime output total does not match checkpoint universe")
    if provenance is not None and context is not None:
        expected_context_sha = sha256(canonical_json_bytes(_context_payload(context))).hexdigest()
        expected_provenance = provenance.with_run_context_sha256(expected_context_sha).to_payload()
        if inspected.provenance_payload != expected_provenance or inspected.run_context_sha256 != expected_context_sha:
            raise ValueError("checkpoint provenance or frozen context does not match launch")
    _reconcile_runtime_events(runtime, inspected, RuntimePaths(output_root, checkpoint_root, output_root / "progress.json", output_root / "events.jsonl", output_root / "parent.log", output_root / "run.pid"))


def execute_runtime_command(
    command: ExecutionCommand,
    *,
    output_root: Path,
    checkpoint_root: Path,
    provenance: object,
    context: FrozenRunContext,
    snapshot: Exp425DatabaseSnapshot,
    audit: BackendAudit,
    adoption: MPSAdoptionAssessment | None = None,
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
    preflight_runtime_paths(command, output_root=output_root, checkpoint_root=checkpoint_root)
    if not isinstance(context, FrozenRunContext) or not isinstance(snapshot, Exp425DatabaseSnapshot) or not isinstance(audit, BackendAudit):
        raise ValueError("runtime launch inputs are invalid")
    from .contracts import Exp425RunProvenance
    if not isinstance(provenance, Exp425RunProvenance):
        raise ValueError("runtime provenance is invalid")
    if provenance.git_commit != context.code_commit:
        raise ValueError("runtime provenance git commit does not match frozen context")
    preflight_runtime_paths(
        command, output_root=output_root, checkpoint_root=checkpoint_root,
        run_identity=provenance.run_identity, provenance=provenance, context=context,
    )
    writer: AtomicProgressWriter | None = None
    output_claimed = False
    try:
        if command == "new":
            if not isinstance(adoption, MPSAdoptionAssessment):
                raise ValueError("runtime MPS adoption evidence is invalid")
            _validate_task7_evidence(audit, adoption)
            _validate_source_artifacts()
            writer = AtomicProgressWriter.create(output_root=output_root, run_identity=provenance.run_identity, command="new", checkpoint_root=checkpoint_root, total_datasets=len(context.datasets), backend=audit.backend)
            output_claimed = True
            writer.write(RuntimeProgress.initializing(writer.paths, provenance.run_identity, "new", len(context.datasets), audit.backend), event="phase_change")
            store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
            if importer is not None:
                if not callable(importer):
                    raise ValueError("runtime importer must be callable")
                writer.write(replace(RuntimeProgress.initializing(writer.paths, provenance.run_identity, "new", len(context.datasets), audit.backend), state="importing", stage="importing"), event="phase_change", detail={"import_stage": "begin"})
                importer(store, context)
                if Exp425CheckpointStore.inspect_read_only(checkpoint_root, expected_run_identity=provenance.run_identity).import_revalidation_required:
                    raise ValueError("pending import intents remain after source import")
        else:
            # This validation is intentionally complete before PID/progress
            # mutation.  Invalid resume consequently leaves the full output
            # tree byte/inode/metadata untouched.
            writer = AtomicProgressWriter.resume(output_root=output_root, run_identity=provenance.run_identity, checkpoint_root=checkpoint_root, claim_pid=False)
            # Preserve the established precedence for an existing live or
            # foreign lease.  This is a read-only check; valid launches still
            # authenticate Task7 before taking any PID lease.
            if writer.paths.pid.exists():
                lease, _ = writer._read_pid()
                if writer._pid_is_live(lease):
                    raise ValueError("runtime PID lease is owned by a live process")
            if not isinstance(adoption, MPSAdoptionAssessment):
                raise ValueError("runtime MPS adoption evidence is invalid")
            _validate_task7_evidence(audit, adoption)
            _validate_source_artifacts()
            existing = inspect_runtime_output(output_root, expected_run_identity=provenance.run_identity)
            inspection = Exp425CheckpointStore.inspect_read_only(checkpoint_root, expected_run_identity=provenance.run_identity)
            if existing.get("checkpoint_root") != str(checkpoint_root) or existing.get("total_datasets") != len(context.datasets):
                raise ValueError("runtime output does not match frozen checkpoint context")
            if existing.get("state") == "complete_review_only" and inspection.completed_count == len(context.datasets):
                if inspection.import_revalidation_required:
                    raise ValueError("complete runtime has pending import revalidation")
                counts = _counts_from_inspection(len(context.datasets), inspection.record_summaries)
                return Exp425RunResult("complete_review_only", counts, (), provenance.run_identity, "complete")
            writer.claim_pid()
            output_claimed = True
            store = Exp425CheckpointStore.open(checkpoint_root, provenance, context)
            if inspection.import_revalidation_required or store.import_revalidation_required:
                if not callable(importer):
                    raise ValueError("resume requires importer revalidation for pending import intents")
                writer.write(replace(RuntimeProgress.initializing(writer.paths, provenance.run_identity, "resume", len(context.datasets), audit.backend), state="importing", stage="importing"), event="phase_change", detail={"import_stage": "revalidation", "pending_imports": len(inspection.import_revalidation_required)})
                importer(store, context)
                if store.import_revalidation_required:
                    raise ValueError("pending import intents remain after source revalidation")
    except Exception as error:
        if writer is not None and output_claimed:
            try:
                failed = RuntimeProgress.initializing(writer.paths, provenance.run_identity, command, len(context.datasets), audit.backend)
                writer.write(replace(failed, state="failed", stage="failed", error_type=type(error).__name__, error_message=_safe_message(error)), event="failure", detail={"error_type": type(error).__name__, "error_message": _safe_message(error)})
            finally:
                writer.close()
        close_snapshot = getattr(snapshot, "close", None)
        if callable(close_snapshot):
            close_snapshot()
        raise
    session: Exp425RuntimeSession | None = None
    try:
        session = build_exp425_runtime(snapshot, audit, adoption, writer, provenance.git_commit)
        return run_selective_population(context, store, session)
    finally:
        if session is not None:
            session.close()
        else:
            close_snapshot = getattr(snapshot, "close", None)
            if callable(close_snapshot):
                close_snapshot()
        assert writer is not None
        writer.close()


def _effective_payload(result: DatasetResult) -> dict[str, object]:
    status = "scored" if result.status == "qualified" else result.status
    coordinates = [[candidate.dataset, candidate.instance_index] for candidate in result.candidates]
    return {"status": status, "candidate_coordinates": coordinates,
            "result_evidence_sha256": canonical_result_sha256(result),
            "result_source_sha256": _digest({"status": status, "candidate_coordinates": coordinates})}


def _audit_digest(audit: BackendAudit) -> str:
    return _digest(asdict(audit))


def _eligibility_payload(evidence: Mapping[int, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for length, item in sorted(evidence.items()):
        item_payload = asdict(item)
        output[str(length)] = item_payload
    _assert_label_free(output)
    return output


def _model_and_calibration_evidence(
    models: TrainModelEvidence,
    scoring: TestScoringEvidence | None,
    *,
    models_trained: bool,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    """Canonicalize actual frozen identities; an abstention has explicit N/A evidence.

    The checkpoint must never pretend that a no-model path trained a model.  For
    eligible work, retain every OOF/final model hash and every cached OOF
    calibration identity used by the unmodified scorer.
    """
    def final_entries() -> list[dict[str, object]]:
        return [
            {"length": length, "seed": model.seed, "model_sha256": model.model_sha256}
            for length, seed_models in sorted(models.final_models.items())
            for model in sorted(seed_models, key=lambda item: item.seed)
        ]

    def calibration_entries() -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for length, by_seed in sorted(models.score_calibrations.items()):
            for seed, by_horizon in sorted(by_seed.items()):
                for horizon, calibration in sorted(by_horizon.items()):
                    statistic = {
                        "horizon": calibration.horizon, "oof_count": calibration.oof_count,
                        "length": calibration.length, "center": calibration.center,
                        "scale": calibration.scale, "threshold": calibration.threshold,
                        "oof_scores_dtype": "<f8", "oof_scores_shape": list(calibration.oof_scores.shape),
                        "oof_scores_sha256": sha256(np.ascontiguousarray(
                            calibration.oof_scores, dtype=np.dtype("<f8")
                        ).tobytes(order="C")).hexdigest(),
                    }
                    entries.append({"length": length, "seed": seed, "horizon": horizon,
                                    "digest": _digest(statistic), "statistic": statistic})
        return entries

    if not models_trained:
        model_payload = {
            "schema": "exp425.runtime-model-evidence.v1",
            "status": "not_applicable_oof_identity_insufficient",
            "processed_identity_sha256": models.processed_identity_sha256,
            "oof_models": [], "final_models": [],
        }
        calibration = []
    else:
        oof_models: list[dict[str, object]] = []
        for length, bundle in sorted(models.oof_bundles.items()):
            fold_by_validation = {fold.validation_indices: index for index, fold in enumerate(bundle.folds)}
            for model in bundle.models:
                oof_models.append({"length": length, "seed": model.seed,
                                   "fold": fold_by_validation[model.validation_indices],
                                   "model_sha256": model.model_sha256})
        model_payload = {
            "schema": "exp425.runtime-model-evidence.v1",
            "status": "captured",
            "processed_identity_sha256": models.processed_identity_sha256,
            "oof_models": sorted(oof_models, key=lambda item: (item["length"], item["seed"], item["fold"])),
            "final_models": final_entries(),
        }
        calibration = calibration_entries()
    scoring_payload = (
        {
            "schema": "exp425.runtime-scoring-evidence.v1",
            "status": "not_applicable_no_qualified_group" if models_trained else "not_applicable_oof_identity_insufficient",
            "calibrations": calibration,
            "final_models": [],
        }
        if scoring is None
        else {
            "schema": "exp425.runtime-scoring-evidence.v1",
            "status": "captured",
            "calibrations": calibration,
            "final_models": [
                {"length": length, "seed": seed, "model_sha256": model_sha}
                for length, by_seed in sorted(scoring.final_model_hashes.items())
                for seed, model_sha in sorted(by_seed.items())
            ],
        }
    )
    return model_payload, scoring_payload, _digest(model_payload), _digest(scoring_payload)


def _frozen_contract_payload(contract: object) -> dict[str, object]:
    """Persist every frozen TRAIN decision field, not a lossy status summary."""
    if not hasattr(contract, "dataset") or not hasattr(contract, "groups"):
        raise ValueError("native result has no frozen TRAIN contract")
    groups = getattr(contract, "groups")
    backend = getattr(contract, "backend")
    payload = {
        "dataset": getattr(contract, "dataset"),
        "groups": [
            {
                "length": group.length, "series_count": group.series_count,
                "status": group.status, "oof_candidate_rate": group.oof_candidate_rate,
                "wilson_upper": group.wilson_upper,
                "pooled_synthetic_sensitivity": group.pooled_synthetic_sensitivity,
                "per_type_synthetic_sensitivity": dict(group.per_type_synthetic_sensitivity),
                "seed_agreement": group.seed_agreement, "finite": group.finite,
                "backend_fallback": group.backend_fallback,
            }
            for group in groups
        ],
        "backend": asdict(backend),
        "train_identity_sha256": getattr(contract, "train_identity_sha256"),
        "near_repeat_status": getattr(contract, "near_repeat_status"),
    }
    _assert_label_free(payload)
    return payload


def _processed_manifest(models: TrainModelEvidence) -> dict[str, object]:
    """Compact identifiers only: model inputs remain outside the checkpoint."""
    return {
        "schema": "exp425.processed-manifest.v1",
        "groups": [
            {"length": group.group.length, "status": group.status,
             "series_count": len(group.group.series),
             "instance_indices": [series.instance_index for series in group.group.series]}
            for group in models.processed
        ],
    }


def _native_record(
    store: Exp425CheckpointStore,
    runtime: Exp425RuntimeSession,
    result: DatasetResult,
    elapsed: float,
    *,
    models: TrainModelEvidence,
    scoring: TestScoringEvidence | None,
    models_trained: bool,
) -> tuple[NativeRecordEnvelope, DatasetRuntimeTrace]:
    evidence = runtime._eligibility.get(result.dataset)
    if evidence is None:
        raise ValueError("native result has no TRAIN eligibility evidence")
    group_evidence = _eligibility_payload(evidence)
    eligibility_sha = _digest(group_evidence)
    contract = result.decision_evidence.frozen_train_contract
    if contract is None:
        raise ValueError("native result has unexpected terminal evidence")
    processed_sha = models.processed_identity_sha256
    models_payload, calibration_payload, model_sha, scoring_sha = _model_and_calibration_evidence(models, scoring, models_trained=models_trained)
    frozen_contract = _frozen_contract_payload(contract)
    frozen_contract_sha = _digest(frozen_contract)
    result_payload = _effective_payload(result)
    audit_sha = _audit_digest(runtime.audit)
    values = {
        "train_identity_sha256": contract.train_identity_sha256, "eligibility_sha256": eligibility_sha,
        "processed_groups_sha256": processed_sha, "model_evidence_sha256": model_sha,
        "frozen_train_contract_sha256": frozen_contract_sha, "scoring_evidence_sha256": scoring_sha,
        "result_evidence_sha256": result_payload["result_evidence_sha256"], "backend_audit_sha256": audit_sha,
    }
    execution = getattr(runtime, "_execution", {}).pop(result.dataset, None)
    if execution is None:
        import torch
        execution = MPSExecutionEvidence(False, "mps", None, bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available()), False, {}, torch.__version__, _TASK7_SOURCE_COMMIT, _SOURCE_HASHES, getattr(store.provenance, "git_commit", _TASK7_SOURCE_COMMIT), "not applicable: OOF identity insufficiency before model execution", 0)
    adoption = runtime.adoption
    trace_body = {"schema": "exp425.native-trace.v2", "run_identity": store.provenance.run_identity, "dataset": result.dataset, "run_context_sha256": store.run_context_sha256, **values, "device": "mps", "mps_adoption_sha256": adoption.sha256, "mps_execution_sha256": execution.sha256}
    trace_sha = _digest(trace_body)
    payload = {"effective_result": result_payload, "native_trace_body": trace_body}
    model_hashes = {
        "train_identity_sha256": values["train_identity_sha256"],
        "model_evidence_sha256": model_sha,
        "frozen_train_contract_sha256": frozen_contract_sha,
        "frozen_train_contract": frozen_contract,
        "model_evidence": models_payload,
    }
    calibration = {"scoring_evidence_sha256": scoring_sha, "scoring_evidence": calibration_payload}
    statuses = {getattr(item, "status", None) for item in evidence.values()}
    reason = (
        "oof_identity_insufficient" if not models_trained and result.status == "abstained"
        and "oof_identity_insufficient" in statuses else
        "native_no_qualified_group" if models_trained and scoring is None else "native_mps"
    )
    processed_manifest = _processed_manifest(models)
    native_group_evidence = {
        "schema": "exp425.runtime-group-evidence.v1",
        "eligibility_sha256": eligibility_sha, "processed_groups_sha256": processed_sha,
        "eligibility": {"payload": group_evidence, "sha256": eligibility_sha},
        "processed": {"manifest": processed_manifest, "manifest_sha256": _digest(processed_manifest),
                      "identity_sha256": processed_sha},
    }
    provisional = NativeRecordEnvelope(result.dataset, payload, native_group_evidence, trace_sha256=trace_sha, record_payload_sha256="0" * 64, model_hashes=model_hashes, calibration_hashes=calibration, elapsed_seconds=elapsed, reason=reason, mps_adoption=adoption, mps_execution=execution, **values)
    record_sha = _digest(_native_record_body(store.run_context_sha256, provisional))
    record = replace(provisional, record_payload_sha256=record_sha)
    # Full decoded validation happens in the store before the capability is consumed.
    trace = DatasetRuntimeTrace(result.dataset, "exp425_native", elapsed, str(values["result_evidence_sha256"]), eligibility_sha, processed_sha, model_sha, frozen_contract_sha, scoring_sha, audit_sha, trace_sha)
    return record, trace


def _no_model_abstention(
    context: FrozenRunContext,
    store: Exp425CheckpointStore,
    runtime: Exp425RuntimeSession,
    dataset: str,
    processed: Sequence[object],
    elapsed: float,
) -> tuple[NativeRecordEnvelope, DatasetRuntimeTrace]:
    """Make the canonical TRAIN-only OOF-insufficiency result without a TEST read."""
    models = TrainModelEvidence(processed, {}, {}, runtime.audit)  # type: ignore[arg-type]
    qualification = _production_qualify_train(dataset, processed, models)  # type: ignore[arg-type]
    contract = runtime.runner.freeze(dataset, qualification, models)
    evidence = DatasetDecisionEvidence(
        dataset=dataset,
        context=context,
        frozen_train_contract=contract,
        database_sha256=context.database_sha256,
        train_identity_sha256=contract.train_identity_sha256,
        decisions=(),
    )
    result = DatasetResult(
        dataset=dataset,
        test_size=context.test_sizes[dataset],
        group_count=len(contract.groups),
        groups=contract.groups,
        candidates=(),
        decision_evidence=evidence,
        lanes=context.lanes,
        context=context,
        status="abstained",
    )
    return _native_record(store, runtime, result, elapsed, models=models, scoring=None, models_trained=False)


def _compute_native(context: FrozenRunContext, store: Exp425CheckpointStore, runtime: Exp425RuntimeSession, dataset: str, train: object) -> tuple[NativeRecordEnvelope, DatasetRuntimeTrace]:
    """Private exact-signature seam; tests may replace it, public callers cannot."""
    started = perf_counter()
    try:
        runtime._models.pop(dataset, None); runtime._scoring.pop(dataset, None)
        runtime._active_dataset = dataset
        processed = tuple(runtime._preprocess(train))
        evidence = runtime._eligibility.get(dataset)
        if evidence is None:
            raise ValueError("native TRAIN preprocessing did not retain eligibility evidence")
        statuses = {getattr(item, "status", None) for item in evidence.values()}
        if "eligible" not in statuses and "oof_identity_insufficient" in statuses:
            return _no_model_abstention(
                context, store, runtime, dataset, processed, perf_counter() - started
            )
        original_loader = runtime.runner.load_train
        original_preprocess = runtime.runner.preprocess
        dataset_runner = replace(
            runtime.runner,
            load_train=lambda name, frozen: train if name == dataset and frozen is context else original_loader(name, frozen),
            preprocess=lambda supplied: processed if supplied is train else original_preprocess(supplied),
        )
        result = run_dataset(dataset, context, runtime=dataset_runner)
        elapsed = perf_counter() - started
        if result.status == "failed" or result.decision_evidence.__class__.__name__ == "DatasetTerminalEvidence":
            raise ValueError("native frozen runtime returned unexpected terminal or failed result")
        models = runtime._models.pop(dataset, None)
        if models is None:
            raise ValueError("eligible native result lacks frozen model/calibration evidence")
        scoring = runtime._scoring.pop(dataset, None)
        contract = result.decision_evidence.frozen_train_contract
        qualified = contract is not None and any(group.status == "qualified" for group in contract.groups)
        if scoring is None and qualified:
            raise ValueError("eligible native result lacks frozen TEST scoring evidence")
        if scoring is None and result.status != "abstained":
            raise ValueError("native result omitted TEST scoring without a valid abstention")
        return _native_record(store, runtime, result, elapsed, models=models, scoring=scoring, models_trained=True)
    finally:
        runtime._active_dataset = None


def _terminal_event(progress: RuntimeProgress, record: object, elapsed: float, *, ordinal: int, status: str, reason: str | None, origin: str) -> RuntimeProgress:
    counts = progress.counts
    assert counts is not None
    imported = counts.imported + (origin == "exp424_import")
    native = counts.native + (origin == "exp425_native")
    # Native records are exactly the predecessor-missing/incompatible work
    # recomputed by Exp425; ``recomputed`` is a subset of ``native``, never a
    # fictional third origin that would break population conservation.
    recomputed = counts.recomputed + (origin == "exp425_native")
    abstained = counts.abstained + (status == "abstained")
    failed = counts.failed + (status == "failed")
    insufficient = counts.oof_identity_insufficient + (reason == "oof_identity_insufficient")
    next_counts = Exp425RunCounts(counts.total, counts.completed + 1, imported, native, recomputed, abstained, insufficient, failed)
    return replace(progress, timestamp=_now(), counts=next_counts, current_dataset=None, origin=None, current_phase=None, ordinal=ordinal, last_terminal_time=_now(), last_completed_dataset=getattr(record, "dataset", None), last_status=status, last_reason=reason, last_elapsed_seconds=elapsed, cumulative_seconds=progress.cumulative_seconds + elapsed, terminal_samples=progress.terminal_samples + (elapsed,), train_instance_count=None, test_instance_count=None, size_summary={})


def _record_status(record: object) -> tuple[str, str | None]:
    payload = _effective_record_payload(record)
    if not isinstance(payload.get("status"), str):
        raise ValueError("checkpoint effective result is invalid")
    return str(payload["status"]), getattr(record, "reason", None)


def _effective_record_payload(record: object) -> Mapping[str, object]:
    """Decode the two deliberately distinct compact record envelopes exactly once."""
    payload = thaw_payload(record.result_payload)  # type: ignore[union-attr]
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint effective result is invalid")
    effective = payload.get("effective_result") if getattr(record, "origin", None) == "exp425_native" else payload
    if not isinstance(effective, Mapping) or not isinstance(effective.get("status"), str):
        raise ValueError("checkpoint effective result is invalid")
    coordinates = effective.get("candidate_coordinates", ())
    if not isinstance(coordinates, list):
        raise ValueError("checkpoint candidate coordinates are invalid")
    return effective


def _restored_progress(
    progress: RuntimeProgress, store: Exp425CheckpointStore, terminal_samples: tuple[float, ...]
) -> RuntimeProgress:
    """Restore accounting from validated compact records without replaying events."""
    records = tuple(store.iter_records())
    imported = sum(record.origin == "exp424_import" for record in records)
    native = sum(record.origin == "exp425_native" for record in records)
    statuses = tuple(_record_status(record)[0] for record in records)
    counts = Exp425RunCounts(
        total=progress.total_datasets,
        completed=len(records),
        imported=imported,
        native=native,
        recomputed=native,
        abstained=sum(status == "abstained" for status in statuses),
        oof_identity_insufficient=sum(_record_status(record)[1] == "oof_identity_insufficient" for record in records),
        failed=sum(status == "failed" for status in statuses),
    )
    return replace(
        progress,
        counts=counts,
        terminal_samples=terminal_samples,
        cumulative_seconds=sum(record.elapsed_seconds for record in records),
    )


def _counts_from_inspection(total: int, summaries: Sequence[tuple[str, str, str, str | None, float]]) -> Exp425RunCounts:
    imported = sum(origin == "exp424_import" for _, origin, _, _, _ in summaries)
    native = sum(origin == "exp425_native" for _, origin, _, _, _ in summaries)
    return Exp425RunCounts(
        total=total, completed=len(summaries), imported=imported, native=native,
        recomputed=native, abstained=sum(status == "abstained" for _, _, status, _, _ in summaries),
        oof_identity_insufficient=sum(reason == "oof_identity_insufficient" for _, _, _, reason, _ in summaries),
        failed=sum(status == "failed" for _, _, status, _, _ in summaries),
    )


def _compact_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validated_compact_effective(record: object, dataset: str) -> Mapping[str, object]:
    """Decode the complete origin-specific compact-result schema for one dataset."""
    if getattr(record, "dataset", None) != dataset:
        raise ValueError("checkpoint record dataset does not match frozen coverage")
    origin = getattr(record, "origin", None)
    payload = thaw_payload(getattr(record, "result_payload", None))
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint effective result is invalid")
    if origin == "exp425_native":
        if set(payload) != {"effective_result", "native_trace_body"}:
            raise ValueError("native checkpoint result payload is invalid")
        effective = payload.get("effective_result")
        required = {"status", "candidate_coordinates", "result_evidence_sha256", "result_source_sha256"}
        if not isinstance(effective, Mapping) or set(effective) != required:
            raise ValueError("native checkpoint effective result schema is invalid")
        if (
            effective.get("result_evidence_sha256") != getattr(record, "result_evidence_sha256", None)
            or not _compact_sha256(effective.get("result_evidence_sha256"))
        ):
            raise ValueError("native checkpoint result evidence is invalid")
    elif origin == "exp424_import":
        effective = payload
        required = {"status", "candidate_coordinates", "source_result_sha256"}
        if not isinstance(effective, Mapping) or set(effective) != required:
            raise ValueError("imported checkpoint effective result schema is invalid")
        if (
            effective.get("source_result_sha256") != getattr(record, "source_result_sha256", None)
            or not _compact_sha256(effective.get("source_result_sha256"))
        ):
            raise ValueError("imported checkpoint result source is invalid")
    else:
        raise ValueError("checkpoint record origin is invalid")
    coordinates = effective.get("candidate_coordinates")
    if not isinstance(coordinates, list):
        raise ValueError("checkpoint candidate coordinates are invalid")
    source_preimage = {"status": effective.get("status"), "candidate_coordinates": coordinates}
    if origin == "exp425_native" and effective.get("result_source_sha256") != _digest(source_preimage):
        raise ValueError("native checkpoint result source is invalid")
    return effective


def _apply_compact_global_workload(context: FrozenRunContext, store: Exp425CheckpointStore) -> str:
    """Prove Exp417 global-gate parity from complete, validated compact records.

    The compact records retain only the exact candidate coordinates, so rebuilding
    full decision evidence here would add no gate information.  This validation
    establishes a one-to-one mapping from each stored coordinate to Exp417's
    countable candidate domain before applying the unchanged strict ``>`` cap.
    """
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    frozen_cap = exp417_qualification.FROZEN_CONFIG.global_workload_rate
    if context.config.global_workload_rate != frozen_cap:
        raise ValueError("context global workload cap differs from frozen Exp417 cap")
    if getattr(store, "context", None) != context or tuple(getattr(store, "expected_datasets", ())) != context.datasets:
        raise ValueError("checkpoint universe does not match frozen context")
    records = tuple(store.iter_records())
    if tuple(getattr(record, "dataset", None) for record in records) != context.datasets:
        raise ValueError("checkpoint records do not provide exact sorted frozen coverage")
    total = sum(context.test_sizes.values())
    manifest_total = sum(len(context.test_identity_manifest[dataset]) for dataset in context.datasets)
    if type(total) is not int or total <= 0 or total != manifest_total:
        raise ValueError("frozen TEST population is invalid")
    candidates = 0
    for dataset, record in zip(context.datasets, records, strict=True):
        effective = _validated_compact_effective(record, dataset)
        status, coordinates = effective.get("status"), effective["candidate_coordinates"]
        if status == "failed":
            raise ValueError("failed checkpoint result cannot pass the global workload gate")
        if status not in {"scored", "abstained"}:
            raise ValueError("checkpoint effective result status is invalid")
        assert isinstance(coordinates, list)
        manifest = {index for index, _ in context.test_identity_manifest[dataset]}
        normalized: list[tuple[str, int]] = []
        for coordinate in coordinates:
            if not isinstance(coordinate, list) or len(coordinate) != 2 or coordinate[0] != dataset or type(coordinate[1]) is not int:
                raise ValueError("checkpoint candidate coordinate is invalid")
            if coordinate[1] not in manifest or context.lanes.contains(dataset, coordinate[1]):
                raise ValueError("checkpoint candidate is outside frozen TEST identities")
            normalized.append((dataset, coordinate[1]))
        if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("checkpoint candidates must be sorted and unique")
        if status == "abstained" and normalized:
            raise ValueError("abstained checkpoint result has candidates")
        candidates += len(normalized)
    return "global_workload_gate_failed" if candidates / total > frozen_cap else "complete"


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
        # ``execute_runtime_command(new)`` has already durably published the
        # one initializing transition before it can create/import a store.
        # Direct test/integration callers still receive it when no state exists.
        if not runtime.writer.paths.progress.exists():
            runtime.writer.write(progress, event="phase_change")
        progress = replace(progress, state="importing", stage="importing", timestamp=_now())
        runtime.writer.write(progress, event="phase_change")
        terminal_events = runtime.writer.read_terminal_events() if runtime.writer.command == "resume" else ()
        terminal_datasets = {dataset for dataset, _ in terminal_events}
        if runtime.writer.command == "resume":
            progress = _restored_progress(progress, store, tuple(elapsed for _, elapsed in terminal_events))
        progress = replace(progress, state="running", stage="running", timestamp=_now())
        runtime.writer.write(progress, event="phase_change")
        for ordinal, dataset in enumerate(context.datasets, start=1):
            completed = store.completed_record_digests()
            if dataset in completed:
                record = store.load_record(dataset)
                status, reason = _record_status(record)
                if runtime.writer.command == "resume" and dataset in terminal_datasets:
                    continue
                if runtime.writer.command == "resume":
                    # A native commit can survive a crash between durable
                    # checkpoint publication and the terminal runtime event.
                    # It is recorded once here without incrementing restored
                    # counters or duplicating estimator samples.
                    if dataset not in terminal_datasets:
                        progress = replace(
                            progress,
                            terminal_samples=progress.terminal_samples + (record.elapsed_seconds,),
                        )
                    runtime.writer.write(progress, event="dataset_terminal", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": record.origin, "status": status, "reason": reason, "elapsed_seconds": record.elapsed_seconds, "completed_datasets": progress.counts.completed, "remaining_datasets": progress.counts.remaining})
                    continue
                progress = _terminal_event(progress, record, record.elapsed_seconds, ordinal=ordinal, status=status, reason=reason, origin=record.origin)
                runtime.writer.write(progress, event="dataset_terminal", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": record.origin, "status": status, "reason": reason, "elapsed_seconds": record.elapsed_seconds, "completed_datasets": progress.counts.completed, "remaining_datasets": progress.counts.remaining})
                progress = replace(progress, heartbeat_sequence=progress.heartbeat_sequence + 1, timestamp=_now())
                runtime.writer.write(progress, event="heartbeat")
                continue
            runtime.snapshot.verify_boundary("runtime:before_dataset")
            train = runtime.snapshot.load_train(dataset, context)
            sizes = _size_summary(train)
            progress = replace(progress, timestamp=_now(), current_dataset=dataset, ordinal=ordinal, origin="exp425_native", current_phase="preprocess", dataset_start_time=_now(), train_instance_count=len(train.series), test_instance_count=context.test_sizes[dataset], size_summary=sizes, values_processed=progress.values_processed + sizes["train_values"], bytes_processed=progress.bytes_processed + sizes["train_bytes"])
            runtime.writer.write(progress, event="dataset_start", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": "exp425_native", "phase": "train_loaded", "train_values": sizes["train_values"], "train_bytes": sizes["train_bytes"]})
            heartbeat = _HeartbeatPump(runtime.writer, progress, interval=getattr(runtime.writer, "heartbeat_interval_seconds", None))
            heartbeat.start()
            try:
                runtime.writer.write(replace(progress, current_phase="preprocess"), event="stage", detail={"dataset": dataset, "phase": "preprocess_before"})
                record, trace = _compute_native(context, store, runtime, dataset, train)
                runtime.writer.write(replace(progress, current_phase="commit"), event="stage", detail={"dataset": dataset, "phase": "commit_before"})
            finally:
                heartbeat.close()
            elapsed = record.elapsed_seconds
            runtime._verify_authority_binding()
            capability = capabilities._mint(runtime, kind="native", run_identity=store.provenance.run_identity, dataset=dataset, context_sha256=store.run_context_sha256, trace_sha256=record.trace_sha256, record_sha256=record.record_payload_sha256, backend_sha256=record.backend_audit_sha256, mps_adoption_sha256=record.mps_adoption.sha256, mps_execution_sha256=record.mps_execution.sha256, process_nonce=runtime.process_nonce)
            store._commit_native(capability, record)
            traces.append(trace)
            status, reason = _record_status(record)
            progress = _terminal_event(progress, record, elapsed, ordinal=ordinal, status=status, reason=reason, origin=record.origin)
            runtime.writer.write(progress, event="dataset_terminal", detail={"dataset": dataset, "ordinal": ordinal, "total": len(context.datasets), "origin": record.origin, "status": status, "reason": reason, "elapsed_seconds": elapsed, "completed_datasets": progress.counts.completed, "remaining_datasets": progress.counts.remaining})
            progress = replace(progress, heartbeat_sequence=progress.heartbeat_sequence + 1, timestamp=_now())
            runtime.writer.write(progress, event="heartbeat")
        if progress.counts.completed != len(context.datasets) or progress.counts.failed:
            raise ValueError("runtime population accounting is incomplete or failed")
        global_gate_status = _apply_compact_global_workload(context, store)
        if global_gate_status != "complete":
            raise ValueError("frozen global workload gate failed")
        progress = replace(progress, state="finalizing", stage="finalizing", timestamp=_now())
        runtime.writer.write(progress, event="finalization")
        inspection = Exp425CheckpointStore.inspect_read_only(store.root, expected_run_identity=store.provenance.run_identity)
        if inspection.completed_count != len(context.datasets) or inspection.import_revalidation_required:
            raise ValueError("final checkpoint integrity is incomplete")
        runtime.close()
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
    run_identity = payload.get("run_identity")
    checkpoint_root = payload.get("checkpoint_root")
    if not isinstance(run_identity, str) or not isinstance(checkpoint_root, str):
        raise ValueError("runtime progress identity is invalid")
    paths = RuntimePaths(root, Path(checkpoint_root), root / "progress.json", root / "events.jsonl", root / "parent.log", root / "run.pid")
    # Validate canonical append-only evidence without creating a PID, lock,
    # temporary file, database connection, checkpoint writer, or hash object.
    AtomicProgressWriter(paths, run_identity, str(payload.get("command", "inspect"))).read_terminal_samples()
    return MappingProxyType(payload)


__all__ = [
    "AtomicProgressWriter", "DatasetRuntimeTrace", "ExecutionCommand", "Exp425RunCounts",
    "Exp425RunResult", "Exp425RuntimeSession", "RuntimePaths", "RuntimeProgress",
    "build_exp425_runtime", "inspect_runtime_output", "run_selective_population",
    "execute_runtime_command", "preflight_runtime_paths",
]
