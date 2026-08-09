"""Ordered, create-only checkpoint records for the Exp443 policy run."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import PolicyDatasetResultAntigravity


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_RECORD = re.compile(r"^(\d{4})_record_antigravity\.json$")
_STATUSES = frozenset({"running", "interrupted_resumable", "error", "complete"})


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _nonempty_string_antigravity(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} is invalid")
    return value


def _sha256_antigravity(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")
    return value


def _timestamp_antigravity(value: object, field: str) -> str:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{field} is invalid")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} is invalid") from error
    return value


@dataclass(frozen=True)
class WorkUnitAntigravity:
    """One ordered policy/dataset recomputation unit."""

    sequence: int
    policy: str
    dataset: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("work-unit sequence is invalid")
        if self.policy not in FROZEN_CONFIG.policies:
            raise ValueError("work-unit policy is not preregistered")
        _nonempty_string_antigravity(self.dataset, "work-unit dataset")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {"sequence": self.sequence, "policy": self.policy, "dataset": self.dataset}

    @classmethod
    def from_payload_antigravity(cls, payload: Mapping[str, object]) -> "WorkUnitAntigravity":
        if not isinstance(payload, Mapping) or set(payload) != {"sequence", "policy", "dataset"}:
            raise ValueError("work-unit payload is invalid")
        return cls(payload["sequence"], payload["policy"], payload["dataset"])


def ordered_work_units_antigravity(
    datasets: Sequence[str],
    policies: Sequence[str] = FROZEN_CONFIG.policies,
) -> tuple[WorkUnitAntigravity, ...]:
    """Build the reviewed policy-major order without normalizing case."""

    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        raise ValueError("datasets are invalid")
    names = tuple(datasets)
    if not names or any(type(name) is not str or not name for name in names) or names != tuple(sorted(set(names))):
        raise ValueError("datasets must be sorted, unique, and case-sensitive")
    policy_order = tuple(policies)
    if policy_order != FROZEN_CONFIG.policies:
        raise ValueError("policy order is not the frozen Exp443 order")
    return tuple(
        WorkUnitAntigravity(sequence, policy, dataset)
        for sequence, (policy, dataset) in enumerate(
            ( (policy, dataset) for policy in policy_order for dataset in names )
        )
    )


@dataclass(frozen=True)
class CheckpointBindingAntigravity:
    """Immutable source/config identity shared by context and resume."""

    run_identity: str
    scientific_config_sha256: str
    environment_lock_sha256: str
    source_snapshot_sha256: str
    database_sha256: str
    reviewed_commit: str
    experiment_id: str = FROZEN_CONFIG.experiment_id

    def __post_init__(self) -> None:
        _nonempty_string_antigravity(self.run_identity, "run identity")
        _sha256_antigravity(self.scientific_config_sha256, "scientific config SHA256")
        _sha256_antigravity(self.environment_lock_sha256, "environment lock SHA256")
        _sha256_antigravity(self.source_snapshot_sha256, "source snapshot SHA256")
        _sha256_antigravity(self.database_sha256, "database SHA256")
        if type(self.reviewed_commit) is not str or _COMMIT.fullmatch(self.reviewed_commit) is None:
            raise ValueError("reviewed commit is invalid")
        if self.experiment_id != FROZEN_CONFIG.experiment_id:
            raise ValueError("experiment identity is invalid")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "run_identity": self.run_identity,
            "scientific_config_sha256": self.scientific_config_sha256,
            "environment_lock_sha256": self.environment_lock_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "database_sha256": self.database_sha256,
            "reviewed_commit": self.reviewed_commit,
        }

    @classmethod
    def from_payload_antigravity(cls, payload: Mapping[str, object]) -> "CheckpointBindingAntigravity":
        expected = {
            "experiment_id", "run_identity", "scientific_config_sha256",
            "environment_lock_sha256", "source_snapshot_sha256", "database_sha256",
            "reviewed_commit",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("checkpoint binding payload is invalid")
        return cls(
            payload["run_identity"], payload["scientific_config_sha256"],
            payload["environment_lock_sha256"], payload["source_snapshot_sha256"],
            payload["database_sha256"], payload["reviewed_commit"], payload["experiment_id"],
        )


@dataclass(frozen=True)
class ProcessIdentityAntigravity:
    """PID plus a caller-provided start identity used by resume gating."""

    pid: int
    start_identity: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid < 1:
            raise ValueError("process PID is invalid")
        _nonempty_string_antigravity(self.start_identity, "process start identity")

    def to_payload_antigravity(self) -> dict[str, object]:
        return {"pid": self.pid, "start_identity": self.start_identity}

    @classmethod
    def from_payload_antigravity(cls, payload: Mapping[str, object]) -> "ProcessIdentityAntigravity":
        if not isinstance(payload, Mapping) or set(payload) != {"pid", "start_identity"}:
            raise ValueError("process identity payload is invalid")
        return cls(payload["pid"], payload["start_identity"])


def _validate_units_antigravity(units: Sequence[WorkUnitAntigravity]) -> tuple[WorkUnitAntigravity, ...]:
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise ValueError("work units are invalid")
    normalized = tuple(units)
    if not normalized:
        raise ValueError("work units are empty")
    if any(type(unit) is not WorkUnitAntigravity for unit in normalized):
        raise ValueError("work units have the wrong identity")
    if tuple(unit.sequence for unit in normalized) != tuple(range(len(normalized))):
        raise ValueError("work-unit sequence order is invalid")
    seen_policies: set[str] = set()
    last_policy: str | None = None
    names_by_policy: dict[str, list[str]] = {}
    for unit in normalized:
        if unit.policy != last_policy:
            if unit.policy in seen_policies:
                raise ValueError("work-unit policy-major order is invalid")
            if last_policy is not None and FROZEN_CONFIG.policies.index(unit.policy) < FROZEN_CONFIG.policies.index(last_policy):
                raise ValueError("work-unit policy-major order is invalid")
            seen_policies.add(unit.policy)
            last_policy = unit.policy
        names_by_policy.setdefault(unit.policy, []).append(unit.dataset)
    if any(names != sorted(set(names)) for names in names_by_policy.values()):
        raise ValueError("work-unit duplicate or dataset order is invalid")
    return normalized


def _root_paths_antigravity(root: Path) -> tuple[Path, Path, Path]:
    root = Path(root)
    if "antigravity" not in root.name:
        raise ValueError("checkpoint root must contain antigravity")
    return root / "context_antigravity.json", root / "state_antigravity.json", root / "records_antigravity"


def _read_canonical_antigravity(path: Path) -> tuple[object, bytes]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint file is not regular")
        payload = path.read_bytes()
        return parse_canonical_json_bytes_antigravity(payload), payload
    except (OSError, ValueError) as error:
        raise ValueError(f"checkpoint record read failed: {path.name}") from error


def _write_state_antigravity(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name("state_antigravity.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("checkpoint state temporary already exists")
    encoded = canonical_json_bytes_antigravity(payload)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("checkpoint state write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        fsync_directory_antigravity(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _validate_state_temporary_antigravity(
    path: Path,
    current_state: Mapping[str, object],
    context_sha256: str,
    expected_prefix: tuple[str, ...],
    expected_error_sequence: int | None,
    units: tuple[WorkUnitAntigravity, ...],
    stored_process: ProcessIdentityAntigravity,
) -> None:
    """Validate the one state temp produced by a record/state crash window."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint state temporary is invalid")
    temporary_payload, _ = _read_canonical_antigravity(path)
    if not isinstance(temporary_payload, Mapping):
        raise ValueError("checkpoint state temporary is invalid")
    expected_keys = {
        "context_sha256", "status", "completed_count", "record_sha256_prefix",
        "error_sequence", "process", "updated_at",
    }
    if set(temporary_payload) != expected_keys:
        raise ValueError("checkpoint state temporary schema is invalid")
    if current_state.get("status") != "running":
        raise ValueError("checkpoint state temporary is unexpected")
    current_prefix = current_state.get("record_sha256_prefix")
    if (
        not isinstance(current_prefix, list)
        or len(expected_prefix) != len(current_prefix) + 1
        or tuple(current_prefix) != expected_prefix[:-1]
    ):
        raise ValueError("checkpoint state temporary prefix is invalid")
    if temporary_payload["context_sha256"] != context_sha256:
        raise ValueError("checkpoint state temporary context changed")
    expected_status = "error" if expected_error_sequence is not None else (
        "complete" if len(expected_prefix) == len(units) else "running"
    )
    if temporary_payload["status"] != expected_status:
        raise ValueError("checkpoint state temporary status is invalid")
    if temporary_payload["completed_count"] != len(expected_prefix):
        raise ValueError("checkpoint state temporary prefix is invalid")
    temporary_prefix = temporary_payload["record_sha256_prefix"]
    if (
        not isinstance(temporary_prefix, list)
        or tuple(temporary_prefix) != expected_prefix
        or any(_SHA256.fullmatch(value or "") is None for value in temporary_prefix)
    ):
        raise ValueError("checkpoint state temporary prefix is invalid")
    if temporary_payload["error_sequence"] != expected_error_sequence:
        raise ValueError("checkpoint state temporary error is invalid")
    temporary_process = ProcessIdentityAntigravity.from_payload_antigravity(temporary_payload["process"])
    if temporary_process != stored_process:
        raise ValueError("checkpoint state temporary process changed")
    _timestamp_antigravity(temporary_payload["updated_at"], "checkpoint state temporary heartbeat")


def _record_path_antigravity(records: Path, sequence: int) -> Path:
    return records / f"{sequence:04d}_record_antigravity.json"


class Exp443CheckpointStoreAntigravity:
    """Parent-owned ordered prefix with create-only result/error records."""

    def __init__(
        self,
        root: Path,
        units: tuple[WorkUnitAntigravity, ...],
        binding: CheckpointBindingAntigravity,
        process: ProcessIdentityAntigravity,
        prefix: tuple[str, ...],
        status: str,
        error_sequence: int | None,
    ) -> None:
        self.root = Path(root)
        self.context_path, self.state_path, self.records_path = _root_paths_antigravity(self.root)
        self.units = units
        self.binding = binding
        self.process = process
        self._prefix = list(prefix)
        self._status = status
        self._error_sequence = error_sequence

    @classmethod
    def create_antigravity(
        cls,
        root: Path,
        units: tuple[WorkUnitAntigravity, ...],
        binding: CheckpointBindingAntigravity,
        process: ProcessIdentityAntigravity,
    ) -> "Exp443CheckpointStoreAntigravity":
        normalized = _validate_units_antigravity(units)
        if type(binding) is not CheckpointBindingAntigravity or type(process) is not ProcessIdentityAntigravity:
            raise ValueError("checkpoint identity is invalid")
        root = Path(root)
        context, state, records = _root_paths_antigravity(root)
        if root.exists() or root.is_symlink():
            raise ValueError("checkpoint root already exists")
        root.mkdir(mode=0o700)
        records.mkdir(mode=0o700)
        context_payload = {
            "binding": binding.to_payload_antigravity(),
            "units": [unit.to_payload_antigravity() for unit in normalized],
        }
        context_bytes = canonical_json_bytes_antigravity(context_payload)
        write_create_only_antigravity(context, context_bytes)
        state_payload = {
            "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "status": "running",
            "completed_count": 0,
            "record_sha256_prefix": [],
            "error_sequence": None,
            "process": process.to_payload_antigravity(),
            "updated_at": _utc_now_antigravity(),
        }
        _write_state_antigravity(state, state_payload)
        return cls(root, normalized, binding, process, (), "running", None)

    @classmethod
    def open_for_resume_antigravity(
        cls,
        root: Path,
        binding: CheckpointBindingAntigravity,
        process: ProcessIdentityAntigravity,
        pid_probe: Callable[[ProcessIdentityAntigravity], bool],
    ) -> "Exp443CheckpointStoreAntigravity":
        if (
            type(binding) is not CheckpointBindingAntigravity
            or type(process) is not ProcessIdentityAntigravity
            or not callable(pid_probe)
        ):
            raise ValueError("checkpoint resume identity is invalid")
        root = Path(root)
        context_path, state_path, records_path = _root_paths_antigravity(root)
        if not root.is_dir() or root.is_symlink() or not records_path.is_dir() or records_path.is_symlink():
            raise ValueError("checkpoint root is invalid")
        context_payload, context_bytes = _read_canonical_antigravity(context_path)
        if not isinstance(context_payload, Mapping) or set(context_payload) != {"binding", "units"}:
            raise ValueError("checkpoint context is invalid")
        stored_binding = CheckpointBindingAntigravity.from_payload_antigravity(context_payload["binding"])
        if stored_binding != binding:
            raise ValueError("checkpoint binding changed")
        if not isinstance(context_payload["units"], list):
            raise ValueError("checkpoint units are invalid")
        units = _validate_units_antigravity(tuple(WorkUnitAntigravity.from_payload_antigravity(item) for item in context_payload["units"]))
        state_payload, _ = _read_canonical_antigravity(state_path)
        if not isinstance(state_payload, Mapping):
            raise ValueError("checkpoint state is invalid")
        expected = {"context_sha256", "status", "completed_count", "record_sha256_prefix", "error_sequence", "process", "updated_at"}
        if set(state_payload) != expected:
            raise ValueError("checkpoint state schema is invalid")
        _timestamp_antigravity(state_payload["updated_at"], "checkpoint heartbeat")
        if state_payload["context_sha256"] != hashlib.sha256(context_bytes).hexdigest():
            raise ValueError("checkpoint context changed")
        status = state_payload["status"]
        if status not in _STATUSES:
            raise ValueError("checkpoint terminal status is invalid")
        stored_process = ProcessIdentityAntigravity.from_payload_antigravity(state_payload["process"])
        if process == stored_process:
            raise ValueError("checkpoint replacement process identity is unchanged")
        try:
            active = bool(pid_probe(stored_process))
        except BaseException as error:
            raise ValueError("checkpoint PID probe failed") from error
        if active:
            raise ValueError("checkpoint process is active")
        completed = state_payload["completed_count"]
        prefix_value = state_payload["record_sha256_prefix"]
        if type(completed) is not int or completed < 0 or completed > len(units) or not isinstance(prefix_value, list):
            raise ValueError("checkpoint prefix is invalid")
        if len(prefix_value) != completed or any(_SHA256.fullmatch(value or "") is None for value in prefix_value):
            raise ValueError("checkpoint prefix is invalid")
        error_sequence = state_payload["error_sequence"]
        if error_sequence is not None and (type(error_sequence) is not int or error_sequence != completed - 1):
            raise ValueError("checkpoint error sequence is invalid")
        if status == "complete":
            raise ValueError("checkpoint attempt is complete")
        if status == "error":
            raise ValueError("checkpoint attempt is terminal error")
        if completed >= len(units):
            raise ValueError("checkpoint attempt is complete")
        prefix, reconciled_error_sequence = _validate_record_inventory_antigravity(
            records_path, units, tuple(prefix_value), error_sequence
        )
        effective_status = "running"
        if reconciled_error_sequence is not None:
            effective_status = "error"
        elif len(prefix) == len(units):
            effective_status = "complete"
        state_temporary = state_path.with_name("state_antigravity.tmp")
        state_temporary_present = state_temporary.exists() or state_temporary.is_symlink()
        if state_temporary_present:
            if len(prefix) == len(prefix_value):
                raise ValueError("checkpoint state temporary is unexpected")
            _validate_state_temporary_antigravity(
                state_temporary,
                state_payload,
                hashlib.sha256(context_bytes).hexdigest(),
                prefix,
                reconciled_error_sequence,
                units,
                stored_process,
            )
        resumed = cls(
            root,
            units,
            binding,
            process,
            prefix,
            effective_status,
            reconciled_error_sequence,
        )
        # An orphan record has already been durably created, so advance the
        # state prefix before returning.  A supplied process always becomes
        # the active owner of the resumed attempt.
        if state_temporary_present:
            try:
                state_temporary.unlink()
                fsync_directory_antigravity(state_temporary.parent)
            except OSError as error:
                raise ValueError("checkpoint state temporary cleanup failed") from error
        if prefix != tuple(prefix_value) or reconciled_error_sequence != error_sequence or process != stored_process:
            resumed._write_state_antigravity()
        return resumed

    def _write_state_antigravity(self) -> None:
        context_bytes = self.context_path.read_bytes()
        _write_state_antigravity(
            self.state_path,
            {
                "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
                "status": self._status,
                "completed_count": len(self._prefix),
                "record_sha256_prefix": list(self._prefix),
                "error_sequence": self._error_sequence,
                "process": self.process.to_payload_antigravity(),
                "updated_at": _utc_now_antigravity(),
            },
        )

    def _require_next_antigravity(self, sequence: int, unit: WorkUnitAntigravity) -> None:
        if self._status in {"error", "complete"}:
            raise ValueError("checkpoint is terminal")
        if type(sequence) is not int or sequence != len(self._prefix):
            raise ValueError("checkpoint commit is not the next sequence")
        expected = self.units[sequence]
        if unit != expected:
            raise ValueError("checkpoint unit identity mismatch")

    def commit_result_antigravity(self, sequence: int, result: PolicyDatasetResultAntigravity) -> str:
        if type(result) is not PolicyDatasetResultAntigravity:
            raise ValueError("checkpoint result identity is invalid")
        if sequence < 0 or sequence >= len(self.units):
            raise ValueError("checkpoint sequence is invalid")
        unit = self.units[sequence]
        self._require_next_antigravity(sequence, unit)
        if result.policy != unit.policy or result.dataset != unit.dataset or result.status not in {"scored", "abstained"}:
            raise ValueError("checkpoint result identity or status mismatch")
        result_payload = result.to_payload()
        trace_digest = hashlib.sha256(canonical_json_bytes_antigravity(result_payload)).hexdigest()
        record = {
            "sequence": sequence,
            "policy": unit.policy,
            "dataset": unit.dataset,
            "result": result_payload,
            "timing": {"elapsed_seconds": 0.0, "throughput": 0.0},
            "sql_trace_sha256": trace_digest,
            "status": result.status,
            "error": None,
        }
        encoded = canonical_json_bytes_antigravity(record)
        write_create_only_antigravity(_record_path_antigravity(self.records_path, sequence), encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        self._prefix.append(digest)
        if len(self._prefix) == len(self.units):
            self._status = "complete"
        self._write_state_antigravity()
        return digest

    def commit_error_antigravity(self, sequence: int, unit: WorkUnitAntigravity, error: BaseException) -> str:
        self._require_next_antigravity(sequence, unit)
        if not isinstance(error, BaseException):
            raise ValueError("checkpoint error identity is invalid")
        message = str(error) or type(error).__name__
        result = {
            "sequence": sequence,
            "policy": unit.policy,
            "dataset": unit.dataset,
            "result": None,
            "timing": {"elapsed_seconds": 0.0, "throughput": 0.0},
            "sql_trace_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "status": "error",
            "error": {"type": type(error).__name__, "message": message},
        }
        encoded = canonical_json_bytes_antigravity(result)
        write_create_only_antigravity(_record_path_antigravity(self.records_path, sequence), encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        self._prefix.append(digest)
        self._error_sequence = sequence
        self._status = "error"
        self._write_state_antigravity()
        return digest

    def mark_interrupted_resumable_antigravity(self) -> None:
        if self._status in {"error", "complete"} or self._error_sequence is not None:
            raise ValueError("checkpoint interruption is terminal")
        self._status = "interrupted_resumable"
        self._write_state_antigravity()

    @property
    def status_antigravity(self) -> str:
        return self._status

    @property
    def completed_count_antigravity(self) -> int:
        return len(self._prefix)

    @property
    def record_sha256_prefix_antigravity(self) -> tuple[str, ...]:
        return tuple(self._prefix)


def _validate_record_inventory_antigravity(
    records: Path,
    units: tuple[WorkUnitAntigravity, ...],
    prefix: tuple[str, ...],
    error_sequence: int | None,
) -> tuple[tuple[str, ...], int | None]:
    names = []
    for path in records.iterdir():
        match = _RECORD.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint record inventory is invalid")
        names.append(int(match.group(1)))
    completed = len(prefix)
    sorted_names = sorted(names)
    orphan_sequence: int | None = None
    if sorted_names == list(range(completed)):
        pass
    elif completed < len(units) and sorted_names == list(range(completed + 1)):
        orphan_sequence = completed
    else:
        raise ValueError("checkpoint record inventory is invalid")
    for sequence, digest in enumerate(prefix):
        payload, encoded = _read_canonical_antigravity(_record_path_antigravity(records, sequence))
        if hashlib.sha256(encoded).hexdigest() != digest or not isinstance(payload, Mapping):
            raise ValueError("checkpoint record bytes changed")
        _validate_record_payload_antigravity(payload, encoded, sequence, units[sequence])
        if payload.get("status") == "error" and sequence != error_sequence:
            raise ValueError("checkpoint error record is misplaced")
    if error_sequence is not None:
        payload, _ = _read_canonical_antigravity(_record_path_antigravity(records, error_sequence))
        if not isinstance(payload, Mapping) or payload.get("status") != "error":
            raise ValueError("checkpoint error record is invalid")
    if orphan_sequence is None:
        return prefix, error_sequence
    if error_sequence is not None:
        raise ValueError("checkpoint orphan follows terminal error")
    orphan_payload, orphan_encoded = _read_canonical_antigravity(
        _record_path_antigravity(records, orphan_sequence)
    )
    _validate_record_payload_antigravity(
        orphan_payload, orphan_encoded, orphan_sequence, units[orphan_sequence]
    )
    orphan_status = orphan_payload["status"]
    return (
        (*prefix, hashlib.sha256(orphan_encoded).hexdigest()),
        orphan_sequence if orphan_status == "error" else None,
    )


def _validate_record_payload_antigravity(
    payload: object,
    encoded: bytes,
    sequence: int,
    unit: WorkUnitAntigravity,
) -> None:
    """Validate the canonical bytes before they can extend the prefix."""
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint record payload is invalid")
    expected_keys = {
        "sequence", "policy", "dataset", "result", "timing",
        "sql_trace_sha256", "status", "error",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint record content is invalid")
    if payload["sequence"] != sequence or payload["policy"] != unit.policy or payload["dataset"] != unit.dataset:
        raise ValueError("checkpoint record identity mismatch")
    status = payload["status"]
    if status not in {"scored", "abstained", "error"}:
        raise ValueError("checkpoint record status is invalid")
    trace_digest = payload["sql_trace_sha256"]
    if _SHA256.fullmatch(trace_digest or "") is None:
        raise ValueError("checkpoint record trace is invalid")
    timing = payload["timing"]
    if not isinstance(timing, Mapping) or set(timing) != {"elapsed_seconds", "throughput"}:
        raise ValueError("checkpoint record timing is invalid")
    if any(type(timing[key]) not in (int, float) or not math.isfinite(float(timing[key])) for key in timing):
        raise ValueError("checkpoint record timing is invalid")
    if status == "error":
        error = payload["error"]
        if payload["result"] is not None or not isinstance(error, Mapping) or set(error) != {"type", "message"}:
            raise ValueError("checkpoint error record is invalid")
        if type(error["type"]) is not str or not error["type"] or type(error["message"]) is not str or not error["message"]:
            raise ValueError("checkpoint error record is invalid")
        return
    if payload["error"] is not None or not isinstance(payload["result"], Mapping):
        raise ValueError("checkpoint record result is invalid")
    try:
        result = PolicyDatasetResultAntigravity.from_payload(payload["result"])
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint record result is invalid") from error
    if result.policy != unit.policy or result.dataset != unit.dataset or result.status != status:
        raise ValueError("checkpoint record result identity mismatch")
    expected_trace = hashlib.sha256(canonical_json_bytes_antigravity(payload["result"])).hexdigest()
    if trace_digest != expected_trace:
        raise ValueError("checkpoint record trace changed")


__all__ = [
    "CheckpointBindingAntigravity",
    "Exp443CheckpointStoreAntigravity",
    "ProcessIdentityAntigravity",
    "WorkUnitAntigravity",
    "ordered_work_units_antigravity",
]
