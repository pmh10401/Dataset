"""Durable Exp451 resume scheduling, native checkpoints, and live JSONL progress."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from time import monotonic
from uuid import uuid4

from paper.equipment_anomaly_detection.contracts_paper import RawDataset

from .artifacts_antigravity import _checked_target_path_antigravity, _publish_create_only_antigravity
from .config_antigravity import EXP451_TOTAL_COORDINATES
from .contracts_antigravity import (
    Exp451Condition,
    Exp451DatasetConditionResult,
    Exp451MigrationManifest,
    Exp451RunContext,
    canonical_json_bytes,
)
from .scoring_antigravity import score_dataset_condition_exp451_antigravity


_CONDITIONS_ANTIGRAVITY = ("A", "B", "C", "D")
_NATIVE_ORIGIN_ANTIGRAVITY = "exp451_native_cpu_pool_fallback"
_NATIVE_SCHEMA_ANTIGRAVITY = "exp451-dataset-condition-native-success-antigravity.v1"
_NATIVE_NAME_ANTIGRAVITY = "dataset_condition_native_success_antigravity.json"
_FAILURE_SCHEMA_ANTIGRAVITY = "exp451-dataset-condition-failure-antigravity.v1"
_PROGRESS_NAME_ANTIGRAVITY = "progress_antigravity.jsonl"


class Exp451IncompleteAntigravity(RuntimeError):
    """Raised after the permitted two waves leave incomplete coordinates."""

    def __init__(self, pending: tuple[tuple[str, str], ...]) -> None:
        self.pending = pending
        super().__init__(f"exp451_incomplete pending={len(pending)}")


@dataclass(frozen=True, slots=True)
class FailureRecordAntigravity:
    dataset: str
    condition_id: str
    wave: int
    exception_type: str
    exception_message: str

    def __post_init__(self) -> None:
        if (
            type(self.dataset) is not str or not self.dataset
            or self.condition_id not in _CONDITIONS_ANTIGRAVITY
            or self.wave not in {1, 2}
            or type(self.exception_type) is not str or not self.exception_type
            or type(self.exception_message) is not str
        ):
            raise ValueError("Exp451 failure record is invalid")

    def artifact_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "condition_id": self.condition_id,
            "wave": self.wave,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True, slots=True)
class Exp451ResumeInventoryAntigravity:
    """The verified imported/native success state and deterministic work queues."""

    completed: tuple[tuple[str, str], ...]
    imported: tuple[tuple[str, str], ...]
    native: tuple[tuple[str, str], ...]
    failures: tuple[FailureRecordAntigravity, ...]
    pending: tuple[tuple[str, str], ...]

    @property
    def completed_count(self) -> int:
        return len(self.completed)

    @property
    def remaining_count(self) -> int:
        return EXP451_TOTAL_COORDINATES - self.completed_count

    @property
    def by_condition(self) -> dict[str, int]:
        return {
            condition: sum(item[1] == condition for item in self.completed)
            for condition in _CONDITIONS_ANTIGRAVITY
        }

    @property
    def wave_two(self) -> tuple[tuple[str, str], ...]:
        complete = set(self.completed)
        failed_once = {
            (record.dataset, record.condition_id)
            for record in self.failures if record.wave == 1
        }
        exhausted = {
            (record.dataset, record.condition_id)
            for record in self.failures if record.wave == 2
        }
        return tuple(
            coordinate for coordinate in self.pending
            if coordinate in failed_once and coordinate not in complete and coordinate not in exhausted
        )


@dataclass(frozen=True, slots=True)
class Exp451ExecutionScheduleAntigravity:
    cpu_workers: int
    inner_numerical_threads: int
    maximum_concurrent_mps_jobs: int
    mps_conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Exp451ProgressStateAntigravity:
    completed: int
    failed: int
    native_attempts: int
    native_elapsed_seconds: float
    eta_seconds: float | None
    reconciled: bool = False


def _catalog_datasets_antigravity(catalog: Mapping[str, object]) -> tuple[str, ...]:
    entries = catalog.get("datasets") if isinstance(catalog, Mapping) else None
    datasets = tuple(
        entry.get("dataset") if isinstance(entry, Mapping) else None
        for entry in entries
    ) if isinstance(entries, list) else ()
    if (
        len(datasets) != 1117 or any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(set(datasets)) != len(datasets) or tuple(sorted(datasets)) != datasets
    ):
        raise ValueError("Exp451 catalog dataset inventory differs")
    return datasets


def _all_coordinates_antigravity(catalog: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (dataset, condition)
        for condition in _CONDITIONS_ANTIGRAVITY
        for dataset in _catalog_datasets_antigravity(catalog)
    )


def _validate_imported_manifest_antigravity(
    manifest: Exp451MigrationManifest, catalog: Mapping[str, object]
) -> tuple[tuple[str, str], ...]:
    if not isinstance(manifest, Exp451MigrationManifest):
        raise ValueError("Exp451 migration manifest is invalid")
    datasets = set(_catalog_datasets_antigravity(catalog))
    imported = tuple((entry.dataset, entry.condition_id) for entry in manifest.entries)
    if (
        len(imported) != 4234 or len(set(imported)) != 4234
        or any(dataset not in datasets for dataset, _ in imported)
        or manifest.counts_by_condition != {"A": 1117, "B": 1117, "C": 1000, "D": 1000}
    ):
        raise ValueError("Exp451 imported migration inventory differs")
    return imported


def validate_failure_records_antigravity(records: Sequence[FailureRecordAntigravity]) -> None:
    """Require unique coordinates per wave and a globally completed first wave."""
    seen: set[tuple[str, str, int]] = set()
    second_wave_started = False
    first_wave: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, FailureRecordAntigravity):
            raise ValueError("Exp451 failure ledger differs")
        key = (record.dataset, record.condition_id, record.wave)
        if key in seen:
            raise ValueError("Exp451 failure ledger has duplicate coordinate wave")
        coordinate = key[:2]
        if record.wave == 1:
            if second_wave_started:
                raise ValueError("Exp451 failure ledger has wave one after wave two")
            first_wave.add(coordinate)
        else:
            second_wave_started = True
            if coordinate not in first_wave:
                raise ValueError("Exp451 failure ledger has wave two without wave one")
        seen.add(key)


def build_resume_inventory_antigravity(
    manifest: Exp451MigrationManifest,
    catalog: Mapping[str, object],
    *,
    native: Sequence[tuple[str, str]] = (),
    failures: Sequence[FailureRecordAntigravity] = (),
) -> Exp451ResumeInventoryAntigravity:
    """Build the stable full-population schedule from the verified migration inventory."""
    imported = _validate_imported_manifest_antigravity(manifest, catalog)
    expected = _all_coordinates_antigravity(catalog)
    expected_set = set(expected)
    native_coordinates = tuple(native)
    if len(native_coordinates) != len(set(native_coordinates)) or any(
        coordinate not in expected_set or coordinate in set(imported)
        for coordinate in native_coordinates
    ):
        raise ValueError("Exp451 native checkpoint inventory differs")
    records = tuple(failures)
    validate_failure_records_antigravity(records)
    if any((record.dataset, record.condition_id) not in expected_set for record in records):
        raise ValueError("Exp451 failure coordinate differs from catalog")
    completed_set = set(imported) | set(native_coordinates)
    completed = tuple(coordinate for coordinate in expected if coordinate in completed_set)
    pending = tuple(coordinate for coordinate in expected if coordinate not in completed_set)
    return Exp451ResumeInventoryAntigravity(
        completed, tuple(imported), native_coordinates, records, pending
    )


def wave_one_coordinates_antigravity(
    inventory: Exp451ResumeInventoryAntigravity,
) -> tuple[tuple[str, str], ...]:
    """Return unattempted gaps only; all first-wave work precedes every retry."""
    if not isinstance(inventory, Exp451ResumeInventoryAntigravity):
        raise ValueError("Exp451 resume inventory is invalid")
    failed_once = {
        (record.dataset, record.condition_id)
        for record in inventory.failures if record.wave == 1
    }
    return tuple(coordinate for coordinate in inventory.pending if coordinate not in failed_once)


def build_execution_schedule_antigravity(context: Exp451RunContext) -> Exp451ExecutionScheduleAntigravity:
    """Record CPU policy while explicitly serialising MPS-heavy C/D coordinates."""
    if not isinstance(context, Exp451RunContext):
        raise ValueError("Exp451 run context is invalid")
    return Exp451ExecutionScheduleAntigravity(
        cpu_workers=context.backend.cpu_workers,
        inner_numerical_threads=context.backend.inner_numerical_threads,
        maximum_concurrent_mps_jobs=1 if context.backend.tensor_backend == "mps" else 0,
        mps_conditions=("C", "D") if context.backend.tensor_backend == "mps" else (),
    )


def progress_event_antigravity(
    *, dataset: str, condition: str, status: str, completed: int, failed: int,
    total: int, elapsed_seconds: float, native_attempts: int,
    native_elapsed_seconds: float,
) -> dict[str, object]:
    """Return one complete event; ETA is native elapsed / native attempts × remaining."""
    if (
        type(dataset) is not str or not dataset or condition not in _CONDITIONS_ANTIGRAVITY
        or status not in {"success", "failure", "reconciled"} or type(completed) is not int
        or type(failed) is not int or type(total) is not int or completed < 0
        or failed < 0 or total != EXP451_TOTAL_COORDINATES or completed > total
        or type(elapsed_seconds) not in {int, float} or elapsed_seconds < 0
        or type(native_attempts) is not int or native_attempts < 0
        or type(native_elapsed_seconds) not in {int, float} or native_elapsed_seconds < 0
    ):
        raise ValueError("Exp451 progress event is invalid")
    remaining = total - completed
    # One timing observation is too noisy to responsibly predict a 234-coordinate tail.
    eta = None if native_attempts < 2 else max(
        0.0, float(native_elapsed_seconds) / native_attempts * remaining
    )
    return {
        "experiment": "Exp451", "dataset": dataset, "condition": condition,
        "status": status, "completed": completed, "failed": failed,
        "remaining": remaining, "total": total,
        "elapsed_seconds": float(elapsed_seconds), "eta_seconds": None if eta is None else float(eta),
        "native_attempts": native_attempts,
        "native_elapsed_seconds": float(native_elapsed_seconds),
    }


def _native_relative_path_antigravity(dataset: str, condition: str) -> str:
    if type(dataset) is not str or not dataset or condition not in _CONDITIONS_ANTIGRAVITY:
        raise ValueError("Exp451 native checkpoint coordinate is invalid")
    return (
        f"native_checkpoints_antigravity/{condition}_antigravity/"
        f"{sha256(dataset.encode('utf-8')).hexdigest()}_antigravity/{_NATIVE_NAME_ANTIGRAVITY}"
    )


def _catalog_entry_antigravity(catalog: Mapping[str, object], dataset: str) -> dict[str, object]:
    entries = catalog.get("datasets") if isinstance(catalog, Mapping) else None
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("dataset") == dataset] if isinstance(entries, list) else []
    if len(matches) != 1:
        raise ValueError("Exp451 native checkpoint catalog identity differs")
    return matches[0]


def _native_payload_antigravity(
    raw: RawDataset, result: Exp451DatasetConditionResult, entry: dict[str, object]
) -> dict[str, object]:
    """Use the frozen typed source/route serializer without changing Exp450 artifacts."""
    from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import _payload
    from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import Exp450DatasetConditionResult

    if not isinstance(result, Exp451DatasetConditionResult) or result.dataset != raw.dataset:
        raise ValueError("Exp451 native scorer result differs")
    return _payload(
        raw,
        Exp450DatasetConditionResult(result.dataset, result.condition_id, result.sources, result.route),
        entry,
    )


def _native_envelope_antigravity(
    context: Exp451RunContext, catalog: Mapping[str, object], raw: RawDataset,
    result: Exp451DatasetConditionResult,
) -> dict[str, object]:
    entry = _catalog_entry_antigravity(catalog, raw.dataset)
    payload = _native_payload_antigravity(raw, result, entry)
    return {
        "schema": _NATIVE_SCHEMA_ANTIGRAVITY,
        "context_sha256": context.sha256,
        "catalog_sha256": context.catalog_sha256,
        "condition_id": result.condition_id,
        "dataset": result.dataset,
        "raw_identity": entry,
        "backend": context.backend.artifact_record(),
        "execution_origin": _NATIVE_ORIGIN_ANTIGRAVITY,
        "payload": payload,
        "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def write_native_success_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object], raw: RawDataset,
    result: Exp451DatasetConditionResult,
) -> Path:
    """Create-only persist a native C/D result before the progress event is emitted."""
    if Path(root) != context.run_root or result.condition_id not in {"C", "D"}:
        raise ValueError("Exp451 native checkpoint context differs")
    envelope = _native_envelope_antigravity(context, catalog, raw, result)
    target = _checked_target_path_antigravity(
        Path(root), _native_relative_path_antigravity(result.dataset, result.condition_id), create=True
    )
    _publish_create_only_antigravity(target, canonical_json_bytes(envelope))
    return target


def _canonical_read_antigravity(path: Path) -> dict[str, object]:
    detail = path.lstat() if path.exists() or path.is_symlink() else None
    if detail is None or stat.S_ISLNK(detail.st_mode) or not stat.S_ISREG(detail.st_mode):
        raise ValueError("Exp451 artifact is missing or invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp451 artifact JSON is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("Exp451 artifact is noncanonical")
    return value


def _write_all_antigravity(descriptor: int, data: bytes) -> None:
    """Write a complete durable-record buffer; short writes are never accepted."""
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Exp451 durable write made no progress")
        view = view[written:]


def _read_native_coordinates_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object]
) -> tuple[tuple[str, str], ...]:
    results: list[tuple[str, str]] = []
    for dataset in _catalog_datasets_antigravity(catalog):
        entry = _catalog_entry_antigravity(catalog, dataset)
        for condition in ("C", "D"):
            relative = _native_relative_path_antigravity(dataset, condition)
            candidate = Path(root) / relative
            if not candidate.exists() and not candidate.is_symlink():
                continue
            path = _checked_target_path_antigravity(
                Path(root), relative, create=False
            )
            value = _canonical_read_antigravity(path)
            required = {
                "schema", "context_sha256", "catalog_sha256", "condition_id", "dataset",
                "raw_identity", "backend", "execution_origin", "payload", "payload_sha256",
            }
            if (
                set(value) != required or value["schema"] != _NATIVE_SCHEMA_ANTIGRAVITY
                or value["context_sha256"] != context.sha256 or value["catalog_sha256"] != context.catalog_sha256
                or value["dataset"] != dataset or value["condition_id"] != condition
                or value["raw_identity"] != entry or value["backend"] != context.backend.artifact_record()
                or value["execution_origin"] != _NATIVE_ORIGIN_ANTIGRAVITY
                or value["payload_sha256"] != sha256(canonical_json_bytes(value["payload"])).hexdigest()
            ):
                raise ValueError("Exp451 native checkpoint differs")
            results.append((dataset, condition))
    return tuple(results)


def _failure_relative_path_antigravity(record: FailureRecordAntigravity) -> str:
    token = sha256(f"{record.dataset}:{record.condition_id}".encode("utf-8")).hexdigest()
    return f"failures_antigravity/wave_{record.wave}_antigravity/{token}_antigravity.json"


def _failure_envelope_antigravity(
    context: Exp451RunContext, catalog: Mapping[str, object], record: FailureRecordAntigravity
) -> dict[str, object]:
    bare = {
        "schema": _FAILURE_SCHEMA_ANTIGRAVITY, "context_sha256": context.sha256,
        "catalog_sha256": context.catalog_sha256, "dataset": record.dataset,
        "condition_id": record.condition_id,
        "raw_identity": _catalog_entry_antigravity(catalog, record.dataset),
        "backend": context.backend.artifact_record(), **record.artifact_record(),
    }
    return {**bare, "record_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}


def append_failure_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object],
    record: FailureRecordAntigravity,
) -> Path:
    """Create one unique coordinate-wave failure record without overwriting history."""
    if Path(root) != context.run_root:
        raise ValueError("Exp451 failure root differs")
    history = _read_failure_records_antigravity(root, context, catalog)
    validate_failure_records_antigravity((*history, record))
    envelope = _failure_envelope_antigravity(context, catalog, record)
    target = _checked_target_path_antigravity(
        Path(root), _failure_relative_path_antigravity(record), create=True
    )
    _publish_create_only_antigravity(target, canonical_json_bytes(envelope))
    return target


def _read_failure_records_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object]
) -> tuple[FailureRecordAntigravity, ...]:
    directory = Path(root) / "failures_antigravity"
    if not directory.exists() and not directory.is_symlink():
        return ()
    detail = directory.lstat()
    if stat.S_ISLNK(detail.st_mode) or not stat.S_ISDIR(detail.st_mode):
        raise ValueError("Exp451 failure ledger differs")
    expected_directories = {"wave_1_antigravity", "wave_2_antigravity"}
    if {path.name for path in directory.iterdir()} - expected_directories:
        raise ValueError("Exp451 failure ledger differs")
    records: list[FailureRecordAntigravity] = []
    for wave in (1, 2):
        wave_directory = directory / f"wave_{wave}_antigravity"
        if not wave_directory.exists() and not wave_directory.is_symlink():
            continue
        wave_detail = wave_directory.lstat()
        if stat.S_ISLNK(wave_detail.st_mode) or not stat.S_ISDIR(wave_detail.st_mode):
            raise ValueError("Exp451 failure ledger differs")
        paths = sorted(wave_directory.iterdir())
        if any(
            path.is_symlink() or not path.is_file()
            or not path.name.endswith("_antigravity.json")
            for path in paths
        ):
            raise ValueError("Exp451 failure ledger differs")
        for path in paths:
            value = _canonical_read_antigravity(path)
            required = {
                "schema", "context_sha256", "catalog_sha256", "dataset", "condition_id",
                "raw_identity", "backend", "wave", "exception_type", "exception_message", "record_sha256",
            }
            bare = {key: value[key] for key in value if key != "record_sha256"} if isinstance(value, dict) else {}
            if (
                set(value) != required or value["schema"] != _FAILURE_SCHEMA_ANTIGRAVITY
                or value["context_sha256"] != context.sha256 or value["catalog_sha256"] != context.catalog_sha256
                or value["backend"] != context.backend.artifact_record()
                or value["raw_identity"] != _catalog_entry_antigravity(catalog, value["dataset"])
                or value["record_sha256"] != sha256(canonical_json_bytes(bare)).hexdigest()
            ):
                raise ValueError("Exp451 failure record differs")
            record = FailureRecordAntigravity(
                value["dataset"], value["condition_id"], value["wave"],
                value["exception_type"], value["exception_message"],
            )
            if record.wave != wave or path.name != Path(_failure_relative_path_antigravity(record)).name:
                raise ValueError("Exp451 failure ledger differs")
            records.append(record)
    validate_failure_records_antigravity(records)
    return tuple(records)


def read_resume_inventory_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object],
    manifest: Exp451MigrationManifest,
) -> Exp451ResumeInventoryAntigravity:
    """Read all durable Exp451 success/failure evidence after manifest authority is verified."""
    if Path(root) != context.run_root or manifest.target_context_sha256 != context.sha256:
        raise ValueError("Exp451 resume inventory context differs")
    return build_resume_inventory_antigravity(
        manifest, catalog,
        native=_read_native_coordinates_antigravity(Path(root), context, catalog),
        failures=_read_failure_records_antigravity(Path(root), context, catalog),
    )


def append_progress_event_antigravity(root: Path, event: Mapping[str, object]) -> Path:
    """Append one fsynced canonical event only after its result/failure is durable."""
    if not Path(root).is_absolute():
        raise ValueError("Exp451 progress root must be absolute")
    required = {
        "experiment", "dataset", "condition", "status", "completed", "failed", "remaining",
        "total", "elapsed_seconds", "eta_seconds", "native_attempts", "native_elapsed_seconds",
    }
    if set(event) != required:
        raise ValueError("Exp451 progress event schema differs")
    path = _checked_target_path_antigravity(Path(root), _PROGRESS_NAME_ANTIGRAVITY, create=True)
    expected = progress_event_antigravity(
        dataset=event["dataset"], condition=event["condition"], status=event["status"],
        completed=event["completed"], failed=event["failed"], total=event["total"],
        elapsed_seconds=event["elapsed_seconds"], native_attempts=event["native_attempts"],
        native_elapsed_seconds=event["native_elapsed_seconds"],
    )
    if dict(event) != expected:
        raise ValueError("Exp451 progress event differs")
    details = path.lstat() if path.exists() or path.is_symlink() else None
    if details is not None and (stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode)):
        raise ValueError("Exp451 progress path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Exp451 progress path is unsafe")
        _write_all_antigravity(descriptor, canonical_json_bytes(dict(event)))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _read_progress_events_antigravity(root: Path) -> tuple[dict[str, object], ...]:
    """Read the append-only JSONL log without creating paths or following links."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Exp451 progress root must be absolute")
    if not root.exists() and not root.is_symlink():
        return ()
    root_details = root.lstat()
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise ValueError("Exp451 progress root is unsafe")
    path = root / _PROGRESS_NAME_ANTIGRAVITY
    if not path.exists() and not path.is_symlink():
        return ()
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp451 progress path is unsafe")
    records: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("Exp451 progress log is noncanonical")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp451 progress log is invalid") from error
        if not isinstance(value, dict) or canonical_json_bytes(value) != line:
            raise ValueError("Exp451 progress log is noncanonical")
        expected = progress_event_antigravity(
            dataset=value.get("dataset"), condition=value.get("condition"), status=value.get("status"),
            completed=value.get("completed"), failed=value.get("failed"), total=value.get("total"),
            elapsed_seconds=value.get("elapsed_seconds"), native_attempts=value.get("native_attempts"),
            native_elapsed_seconds=value.get("native_elapsed_seconds"),
        )
        if value != expected:
            raise ValueError("Exp451 progress log differs")
        records.append(value)
    return tuple(records)


def status_snapshot_antigravity(
    root: Path, *, imported_completed: int, durable_completed: int | None = None,
    durable_failed: int | None = None,
) -> dict[str, object]:
    """Return durable live counters; unlogged checkpoints deliberately suppress ETA."""
    if type(imported_completed) is not int or not 0 <= imported_completed <= EXP451_TOTAL_COORDINATES:
        raise ValueError("Exp451 imported progress count is invalid")
    if durable_completed is not None and (
        type(durable_completed) is not int or not imported_completed <= durable_completed <= EXP451_TOTAL_COORDINATES
    ):
        raise ValueError("Exp451 durable progress count is invalid")
    if durable_failed is not None and (type(durable_failed) is not int or durable_failed < 0):
        raise ValueError("Exp451 durable failure count is invalid")
    events = _read_progress_events_antigravity(Path(root))
    if not events:
        return {
            "completed": imported_completed if durable_completed is None else durable_completed,
            "failed": 0 if durable_failed is None else durable_failed,
            "remaining": EXP451_TOTAL_COORDINATES - (imported_completed if durable_completed is None else durable_completed),
            "total": EXP451_TOTAL_COORDINATES, "elapsed_seconds": 0.0,
            "eta_seconds": None, "latest_coordinate": None, "latest_event": None,
            "native_attempts": 0, "native_elapsed_seconds": 0.0,
        }
    latest = events[-1]
    differs = (
        durable_completed is not None and latest["completed"] != durable_completed
    ) or (
        durable_failed is not None and latest["failed"] != durable_failed
    )
    completed = latest["completed"] if durable_completed is None else durable_completed
    failed = latest["failed"] if durable_failed is None else durable_failed
    return {
        "completed": completed, "failed": failed,
        "remaining": EXP451_TOTAL_COORDINATES - completed, "total": latest["total"],
        "elapsed_seconds": latest["elapsed_seconds"], "eta_seconds": None if differs else latest["eta_seconds"],
        "latest_coordinate": [latest["dataset"], latest["condition"]],
        "latest_event": latest, "native_attempts": latest["native_attempts"],
        "native_elapsed_seconds": latest["native_elapsed_seconds"],
    }


def reconcile_resume_progress_antigravity(
    root: Path, inventory: Exp451ResumeInventoryAntigravity,
    *, progress: Callable[[dict[str, object]], None] | None = None,
) -> Exp451ProgressStateAntigravity:
    """Record a checkpoint/log crash boundary without inventing missing run duration."""
    if not isinstance(inventory, Exp451ResumeInventoryAntigravity):
        raise ValueError("Exp451 reconciliation inventory is invalid")
    events = _read_progress_events_antigravity(Path(root))
    latest = events[-1] if events else None
    logged_successes = {
        (event["dataset"], event["condition"])
        for event in events if event["status"] == "success"
    }
    unlogged_native = tuple(coordinate for coordinate in inventory.native if coordinate not in logged_successes)
    completed = inventory.completed_count
    failed = len(inventory.failures)
    latest_completed = 0 if latest is None else latest["completed"]
    latest_failed = 0 if latest is None else latest["failed"]
    attempts = 0 if latest is None else latest["native_attempts"]
    elapsed = 0.0 if latest is None else latest["native_elapsed_seconds"]
    if unlogged_native or completed != latest_completed or failed != latest_failed:
        dataset, condition = unlogged_native[-1] if unlogged_native else (
            inventory.pending[0] if inventory.pending else ("reconciliation_antigravity", "C")
        )
        event = progress_event_antigravity(
            dataset=dataset, condition=condition, status="reconciled", completed=completed,
            failed=failed, total=EXP451_TOTAL_COORDINATES, elapsed_seconds=elapsed,
            native_attempts=attempts, native_elapsed_seconds=elapsed,
        )
        append_progress_event_antigravity(Path(root), event)
        if progress is not None:
            progress(event)
        return Exp451ProgressStateAntigravity(completed, failed, attempts, elapsed, None, True)
    return Exp451ProgressStateAntigravity(
        completed, failed, attempts, elapsed,
        None if latest is None else latest["eta_seconds"], False,
    )


@contextmanager
def exclusive_run_lock_antigravity(root: Path):
    """Fail closed on an existing lock; stale locks require explicit operator inspection."""
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Exp451 run lock root must be absolute")
    path = _checked_target_path_antigravity(root, "run_lock_antigravity.json", create=True)
    existing = path.lstat() if path.exists() or path.is_symlink() else None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise ValueError("Exp451 run lock is unsafe")
        raise ValueError("Exp451 run lock exists; stale locks require explicit inspection")
    record = canonical_json_bytes({"pid": os.getpid(), "token": uuid4().hex})
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError("Exp451 run lock exists; stale locks require explicit inspection") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Exp451 run lock is unsafe")
        _write_all_antigravity(descriptor, record)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        details = path.lstat() if path.exists() or path.is_symlink() else None
        if details is not None and stat.S_ISREG(details.st_mode) and path.read_bytes() == record:
            path.unlink()
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def execute_resume_coordinates_antigravity(
    coordinates: Sequence[tuple[str, str]], *, scorer: Callable[[str, str], object],
    persist_success: Callable[[str, str, object], None],
    progress: Callable[[dict[str, object]], None] | None, completed: int, failed: int,
    total: int, wave: int, append_failure: Callable[[FailureRecordAntigravity], None],
    native_attempts: int, native_elapsed_seconds: float,
) -> tuple[int, int, int, float]:
    """Serialise MPS-heavy work and emit progress strictly after durable persistence."""
    if wave not in {1, 2}:
        raise ValueError("Exp451 execution wave differs")
    started = monotonic()
    current_completed, current_failed = completed, failed
    current_attempts, current_elapsed = native_attempts, float(native_elapsed_seconds)
    for dataset, condition in coordinates:
        try:
            result = scorer(dataset, condition)
            if (
                getattr(result, "dataset", None) != dataset
                or getattr(result, "condition_id", None) != condition
            ):
                raise ValueError("Exp451 scorer result coordinate differs")
            persist_success(dataset, condition, result)
        except Exception as error:
            append_failure(FailureRecordAntigravity(dataset, condition, wave, type(error).__name__, str(error)))
            current_failed += 1
            status = "failure"
        else:
            current_completed += 1
            status = "success"
        current_attempts += 1
        current_elapsed = float(native_elapsed_seconds) + (monotonic() - started)
        if progress is not None:
            progress(progress_event_antigravity(
                dataset=dataset, condition=condition, status=status,
                completed=current_completed, failed=current_failed, total=total,
                elapsed_seconds=current_elapsed, native_attempts=current_attempts,
                native_elapsed_seconds=current_elapsed,
            ))
    return current_completed, current_failed, current_attempts, current_elapsed


def execute_exp451_resume_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object],
    manifest: Exp451MigrationManifest, raw_datasets: Mapping[str, RawDataset],
    *, progress: Callable[[dict[str, object]], None] | None = None,
) -> Exp451ResumeInventoryAntigravity:
    """Run exactly the missing coordinates under one root-contained exclusive lock."""
    if Path(root) != context.run_root or manifest.target_context_sha256 != context.sha256:
        raise ValueError("Exp451 resume context differs")
    datasets = _catalog_datasets_antigravity(catalog)
    if tuple(raw_datasets) != datasets or any(
        not isinstance(raw_datasets[dataset], RawDataset) or raw_datasets[dataset].dataset != dataset
        for dataset in datasets
    ):
        raise ValueError("Exp451 resume raw inventory differs")
    with exclusive_run_lock_antigravity(Path(root)):
        return _execute_exp451_resume_locked_antigravity(
            Path(root), context, catalog, manifest, raw_datasets, progress=progress
        )


def _execute_exp451_resume_locked_antigravity(
    root: Path, context: Exp451RunContext, catalog: Mapping[str, object],
    manifest: Exp451MigrationManifest, raw_datasets: Mapping[str, RawDataset],
    *, progress: Callable[[dict[str, object]], None] | None,
) -> Exp451ResumeInventoryAntigravity:
    """Execute the two bounded waves after the public runner has acquired its lock."""
    schedule = build_execution_schedule_antigravity(context)
    if context.backend.tensor_backend == "mps" and schedule.maximum_concurrent_mps_jobs != 1:
        raise ValueError("Exp451 MPS serialization differs")

    def live_inventory() -> Exp451ResumeInventoryAntigravity:
        return read_resume_inventory_antigravity(root, context, catalog, manifest)

    def persist(dataset: str, condition: str, result: object) -> None:
        if not isinstance(result, Exp451DatasetConditionResult):
            raise ValueError("Exp451 native scorer result differs")
        write_native_success_antigravity(root, context, catalog, raw_datasets[dataset], result)

    def scorer(dataset: str, condition: str) -> Exp451DatasetConditionResult:
        chosen = next(item for item in context_conditions_antigravity() if item.condition_id == condition)
        return score_dataset_condition_exp451_antigravity(raw_datasets[dataset], chosen, context.backend)

    def record_failure(record: FailureRecordAntigravity) -> None:
        append_failure_antigravity(root, context, catalog, record)

    def emit(event: dict[str, object]) -> None:
        append_progress_event_antigravity(root, event)
        if progress is not None:
            progress(event)

    inventory = live_inventory()
    reconciliation = reconcile_resume_progress_antigravity(root, inventory, progress=progress)
    first_wave = wave_one_coordinates_antigravity(inventory)
    completed, failed, native_attempts, native_elapsed = execute_resume_coordinates_antigravity(
        first_wave, scorer=scorer, persist_success=persist, progress=emit,
        completed=inventory.completed_count, failed=len(inventory.failures),
        total=EXP451_TOTAL_COORDINATES, wave=1, append_failure=record_failure,
        native_attempts=reconciliation.native_attempts,
        native_elapsed_seconds=reconciliation.native_elapsed_seconds,
    )
    inventory = live_inventory()
    if inventory.wave_two:
        execute_resume_coordinates_antigravity(
            inventory.wave_two, scorer=scorer, persist_success=persist, progress=emit,
            completed=completed, failed=failed,
            total=EXP451_TOTAL_COORDINATES, wave=2, append_failure=record_failure,
            native_attempts=native_attempts, native_elapsed_seconds=native_elapsed,
        )
    inventory = live_inventory()
    if inventory.pending:
        raise Exp451IncompleteAntigravity(inventory.pending)
    return inventory


def context_conditions_antigravity() -> tuple[Exp451Condition, ...]:
    return (
        Exp451Condition("A", "linear", "linear"),
        Exp451Condition("B", "polyphase_pchip", "linear"),
        Exp451Condition("C", "linear", "area"),
        Exp451Condition("D", "polyphase_pchip", "area"),
    )


__all__ = [
    "Exp451ExecutionScheduleAntigravity", "Exp451IncompleteAntigravity",
    "Exp451ProgressStateAntigravity", "Exp451ResumeInventoryAntigravity",
    "FailureRecordAntigravity",
    "append_failure_antigravity", "append_progress_event_antigravity",
    "build_execution_schedule_antigravity", "build_resume_inventory_antigravity",
    "context_conditions_antigravity", "execute_exp451_resume_antigravity",
    "execute_resume_coordinates_antigravity", "exclusive_run_lock_antigravity",
    "progress_event_antigravity", "read_resume_inventory_antigravity",
    "reconcile_resume_progress_antigravity", "status_snapshot_antigravity",
    "validate_failure_records_antigravity", "wave_one_coordinates_antigravity",
    "write_native_success_antigravity",
]
