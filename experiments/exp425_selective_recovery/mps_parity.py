"""Fail-closed evidence primitives for Exp425 Task 5's one-fixture MPS smoke.

This module intentionally has no population runner.  The CLI owns the single
actual native invocation; these helpers validate the immutable predecessor and
make its representation-neutral scientific comparison reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Event, RLock, Thread
from time import time
from typing import Any

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import _record_from_bytes
from experiments.exp417_causal_multihorizon_shadow.contracts import DatasetDecisionEvidence, DatasetResult
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from .contracts import MPSAdoptionAssessment


FIXTURE = "ECG200_normal_1"
DATABASE_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
GOLDEN_RECORD_SHA256 = "dc4babf53ab20653322fe0d4f931d20ef68c8c28979244fb40b1367f5c5f0bb7"
GOLDEN_PAYLOAD_SHA256 = "61140e63d63886ac59968c83959dbbed0afbdf42bf2b1a5abe434712480987c4"
GOLDEN_ROOT_SHA256 = "f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757"
GOLDEN_PROGRESS_SHA256 = "5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84"
GOLDEN_SOURCE_COMMIT = "4a398374d880f859f285a271ad63de4a5e55b028"
EXPECTED_SOURCES = {
    "experiments/exp417_causal_multihorizon_shadow/runner.py": "f784fb77805e353b7652582626672686f52a2fdc06d27f9c1ca58730d27b772c",
    "experiments/exp417_causal_multihorizon_shadow/model.py": "422e096613be84ba77350b068e6bd837eebbb46169bb6e3e275f3f03ab2fcedd",
    "experiments/exp417_causal_multihorizon_shadow/scoring.py": "06a92422ea7909efbb7354f1f0499329bd19c04642bbbd5a136fa3f2f5a39941",
}
EXPECTED_MODEL_HASHES = {
    "96": {
        "41701": "1db68f931ef0c6fc98cccc2ea46b95ba94684641eb6a1d6846ad9a0162f61f37",
        "41702": "36b9bdd6a4ffd854837642889e6e8ea30d2e3daa20b8ecd2b6b58f74cb83c019",
        "41703": "e6bb3e8a51722c4719035cc5043dafe8080c5a53e5683375ac5dc457ee66d9b2",
    }
}
EXPECTED_LEGACY_OOF = {"96": "5eaa66a03e1fb60ea4163b2bb89fb57bcb8ab8ac9d23e9b3a768abe033310357"}

# This is the exact, documented common semantic envelope.  It deliberately
# excludes schema/path/timing/provenance/run wrappers and backend-specific audit
# representations while retaining the fields that can change the science.
NORMALIZED_SHARED_KEYS = frozenset({
    "dataset", "database_sha256", "raw_train_sha256", "repaired_train_sha256",
    "length_identity_sha256", "config", "series_count", "lengths",
    "final_model_hashes", "legacy_oof_digest", "qualification", "status",
    "candidate_coordinates", "finiteness", "fallback",
})


def _digest_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def verify_source_hashes(root: Path, expected: Mapping[str, str] = EXPECTED_SOURCES) -> dict[str, str]:
    """Hash exact bytes, not imports, so a source mutation fails before MPS."""
    result: dict[str, str] = {}
    for relative, wanted in sorted(expected.items()):
        raw = (Path(root) / relative).read_bytes()
        actual = _digest_bytes(raw)
        if actual != wanted:
            raise ValueError(f"source digest mismatch: {relative}")
        result[relative] = actual
    return result


@dataclass(frozen=True)
class GoldenReference:
    record: Mapping[str, object]
    record_sha256: str
    source_hashes: Mapping[str, str]
    root_sha256: str
    progress_sha256: str


def validate_golden_reference(repository_root: Path, source_root: Path, record_path: Path) -> GoldenReference:
    """Authenticate the Exp424 record and its exact decoded Exp417 payload."""
    source_hashes = verify_source_hashes(repository_root)
    root_raw = (Path(source_root) / "checkpoint_provenance.json").read_bytes()
    progress_raw = (Path(source_root) / "progress.json").read_bytes()
    if _digest_bytes(root_raw) != GOLDEN_ROOT_SHA256 or _digest_bytes(progress_raw) != GOLDEN_PROGRESS_SHA256:
        raise ValueError("golden source provenance digest mismatch")
    raw = Path(record_path).read_bytes()
    if _digest_bytes(raw) != GOLDEN_RECORD_SHA256:
        raise ValueError("golden record digest mismatch")
    record = _record_from_bytes(raw)
    if record.get("dataset") != FIXTURE or record.get("payload_sha256") != GOLDEN_PAYLOAD_SHA256:
        raise ValueError("golden record fixture or payload mismatch")
    if record.get("canonical_model_hashes") != EXPECTED_MODEL_HASHES:
        raise ValueError("golden final model hashes mismatch")
    if record.get("score_calibration_hashes") != EXPECTED_LEGACY_OOF:
        raise ValueError("golden legacy OOF digest mismatch")
    qualification = record.get("qualification")
    if not isinstance(qualification, DatasetResult) or qualification.dataset != FIXTURE or qualification.status != "abstained" or qualification.candidates or not isinstance(qualification.decision_evidence, DatasetDecisionEvidence) or qualification.decision_evidence.frozen_train_contract is None or any(group.status != "failed" or group.finite is not True for group in qualification.groups):
        raise ValueError("golden qualification is not a no-candidate abstention")
    return GoldenReference(record, GOLDEN_RECORD_SHA256, source_hashes, GOLDEN_ROOT_SHA256, GOLDEN_PROGRESS_SHA256)


def normalized_digest(payload: Mapping[str, object]) -> str:
    """Digest precisely the allowlisted shared scientific semantic payload."""
    if set(payload) != NORMALIZED_SHARED_KEYS:
        raise ValueError("normalized digest allowlist mismatch")
    return _digest_bytes(canonical_json_bytes(dict(payload)))


def _compare(left: object, right: object, tolerance: float, path: str, failures: list[str]) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            failures.append(path + ".keys")
            return
        for key in sorted(left):
            _compare(left[key], right[key], tolerance, f"{path}.{key}", failures)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(path + ".length")
            return
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, tolerance, f"{path}[{index}]", failures)
        return
    if isinstance(left, float) and isinstance(right, float):
        if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
            failures.append(path)
        return
    if left != right:
        failures.append(path)


def compare_shared_science(golden: Mapping[str, object], actual: Mapping[str, object], *, tolerance: float) -> dict[str, object]:
    """Compare exactly shared scientific fields; raw audits stay outside this digest."""
    if not isinstance(tolerance, float) or tolerance < 0:
        raise ValueError("comparison tolerance is invalid")
    golden_digest, actual_digest = normalized_digest(golden), normalized_digest(actual)
    failures: list[str] = []
    _compare(golden, actual, tolerance, "shared", failures)
    return {"schema": "exp425.task5.shared-parity.v1", "passed": not failures,
            "tolerance": tolerance, "golden_normalized_digest": golden_digest,
            "actual_normalized_digest": actual_digest, "mismatches": failures}


def authenticated_cpu_audit() -> BackendAudit:
    """The frozen Task7 scientific audit remains truthfully CPU."""
    return BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)


def review_only_mps_assessment() -> MPSAdoptionAssessment:
    return MPSAdoptionAssessment("/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow/task7_backend_audit.json", "f77efa941f6b33fe6687b8ac0c2b01c152dac9c15e8b0966e3d9bb41b119b592", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, 1.1, False, "speedup_below_threshold")


class SmokeEvidenceWriter:
    """Small self-contained atomic/fsynced operational evidence writer."""

    def __init__(self, root: Path, run_identity: str, fixture: str) -> None:
        self.root, self.run_identity, self.fixture = Path(root), run_identity, fixture
        self.events = self.root / "events.jsonl"
        self.parent_log = self.root / "parent.log"
        self._lock = RLock()
        self._heartbeat_sequence = 0
        self._last_progress: dict[str, object] = {}

    @classmethod
    def create(cls, root: Path, run_identity: str, fixture: str) -> "SmokeEvidenceWriter":
        root = Path(root).absolute()
        if root.exists():
            raise ValueError("no-clobber output root already exists")
        root.mkdir(mode=0o700, parents=False)
        writer = cls(root, run_identity, fixture)
        writer._write_new("run.pid", (str(os.getpid()) + "\n").encode())
        writer._append("parent.log", f"pid={os.getpid()} fixture={fixture}\n".encode())
        writer.progress(stage="initializing", current_dataset=None, success=None)
        return writer

    def _write_new(self, name: str, data: bytes) -> None:
        descriptor = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, data); os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sync_directory()

    def _replace(self, name: str, payload: Mapping[str, object]) -> None:
        with self._lock:
            temporary = self.root / ("." + name + ".tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, canonical_json_bytes(dict(payload)) + b"\n"); os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self.root / name); self._sync_directory()

    def _append(self, name: str, data: bytes) -> None:
        with self._lock:
            descriptor = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, data); os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _sync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)

    def event(self, event: str, **detail: object) -> None:
        with self._lock:
            payload = {"schema": "exp425.task5.event.v1", "run_identity": self.run_identity,
                       "timestamp": time(), "event": event, **detail}
            self._append("events.jsonl", canonical_json_bytes(payload) + b"\n")

    def _publish(self, event: str, *, stage: str, current_dataset: str | None, success: bool | None, **detail: object) -> None:
        with self._lock:
            payload = {"schema": "exp425.task5.progress.v1", "run_identity": self.run_identity,
                       "pid": os.getpid(), "fixture": self.fixture, "stage": stage,
                       "current_dataset": current_dataset, "success": success, "timestamp": time(),
                       "heartbeat_sequence": self._heartbeat_sequence, **detail}
            self._replace("progress.json", payload)
            self._last_progress = dict(payload)
            self.event(event, **{key: value for key, value in payload.items() if key not in {"schema", "run_identity", "pid", "fixture", "timestamp"}})
            self._append(
                "parent.log",
                (f"event={event} stage={stage} dataset={current_dataset or '-'} "
                 f"heartbeat={self._heartbeat_sequence} completed={payload.get('completed', 0)} "
                 f"total={payload.get('total', 1)} elapsed={payload.get('elapsed_seconds', 0)} "
                 f"backend={payload.get('backend', '-')} device={payload.get('device', '-')} "
                 f"checkpoint={payload.get('checkpoint_path', '-')}\n").encode(),
            )

    def progress(self, *, stage: str, current_dataset: str | None, success: bool | None, **detail: object) -> None:
        self._publish("progress", stage=stage, current_dataset=current_dataset, success=success, **detail)

    def heartbeat(self, *, stage: str, current_dataset: str, **detail: object) -> None:
        with self._lock:
            self._heartbeat_sequence += 1
            self._publish("heartbeat", stage=stage, current_dataset=current_dataset, success=None, **detail)

    def terminal_success(self, **detail: object) -> None:
        detail = dict(detail) | {"throughput_per_second": 0, "eta_seconds": 0, "latest_error": None}
        self._publish("complete", stage="complete", current_dataset=None, success=True, **detail)

    def failure(self, error: BaseException) -> None:
        message = str(error).replace("\n", " ")[:240]
        carried = {
            key: value for key, value in self._last_progress.items()
            if key not in {"schema", "run_identity", "pid", "fixture", "stage", "current_dataset", "success", "timestamp", "latest_error", "heartbeat_sequence"}
        }
        terminal = carried | {
            "error_type": type(error).__name__, "latest_error": message,
            "throughput_per_second": 0, "eta_seconds": 0,
        }
        terminal.setdefault("elapsed_seconds", 0)
        self._publish("failed", stage="failed", current_dataset=None, success=False, **terminal)

    def result(self, payload: Mapping[str, object]) -> None:
        self._replace("result.json", payload)


class SmokeHeartbeat:
    """One serialized, fsynced heartbeat stream during the long MPS stage."""

    def __init__(self, writer: SmokeEvidenceWriter, detail: callable, *, interval_seconds: float = 30.0) -> None:
        if interval_seconds <= 0 or interval_seconds > 30:
            raise ValueError("smoke heartbeat interval must be in (0, 30] seconds")
        self._writer, self._detail, self._interval = writer, detail, interval_seconds
        self._stop = Event()
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, name="exp425-smoke-heartbeat", daemon=True)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._interval):
                detail = self._detail()
                if not isinstance(detail, Mapping):
                    raise ValueError("smoke heartbeat detail must be a mapping")
                self._writer.heartbeat(**dict(detail))
        except BaseException as error:
            self._error = error
            self._stop.set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set(); self._thread.join()
        if self._error is not None:
            raise self._error

    def is_alive(self) -> bool:
        return self._thread.is_alive()
