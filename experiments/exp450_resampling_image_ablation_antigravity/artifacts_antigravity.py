"""Canonical per-dataset-condition Exp450 checkpoints and recovery inventory.

The typed source and route payload closure below is copied from the verified
Task4/5 artifact serializer.  Keeping it local binds Exp450 results to its
four-condition identity without importing or changing Exp449's artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import numpy as np

from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    RouteLanes,
    SourceExpansion,
)
from paper.equipment_anomaly_detection.lineage_paper import LINEAGE_NODES
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    SOURCE_ALPHA_PAPER,
    HardSourceBundle,
    HardSourceResult,
)

from .contracts_antigravity import (
    Exp450Condition,
    Exp450DatasetConditionResult,
    Exp450RunContext,
    canonical_json_bytes,
)


SUCCESS_SCHEMA = "exp450-dataset-condition-success-antigravity.v1"
FAILURE_SCHEMA = "exp450-dataset-condition-failure-antigravity.v1"
SUCCESS_NAME = "dataset_condition_success_antigravity.json"
CONDITION_IDS = ("A", "B", "C", "D")
TASK4_TASK5_ARTIFACT_SOURCE_ANTIGRAVITY = {
    "path": "paper/equipment_anomaly_detection/artifacts_paper.py",
    "sha256": "aee497c04f527437df85d11006d254a34a8e82c9bc4a4187e73452ba5ac4e358",
}
_SOURCE_ORIGIN_IDS = {
    "rocket_exp40": "rocket_exp40",
    "exp55_best": "exp55_best",
    "exp56_best": "exp56_best",
    "exp84": "exp84_family_guard",
    "rocket_block_b": "rocket_block_b",
    "canonical_normal_state": "canonical_normal_state",
}


@dataclass(frozen=True, slots=True)
class DatasetConditionFailureAntigravity:
    dataset: str
    condition_id: str
    wave: int
    attempt: int
    exception_type: str
    exception_message: str


@dataclass(frozen=True, slots=True)
class DatasetConditionInventoryAntigravity:
    completed: tuple[tuple[str, str], ...]
    failures: tuple[DatasetConditionFailureAntigravity, ...]
    wave_two: tuple[tuple[str, str], ...]


def _sha(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _path(root: Path, *parts: str) -> Path:
    root = Path(root)
    if not root.is_absolute() or any(
        not part or "/" in part or part in {".", ".."} for part in parts
    ):
        raise ValueError("checkpoint path is invalid")
    return root.joinpath(*parts)


def _canonical_read(path: Path) -> object:
    details = _lstat(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("checkpoint is missing")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint JSON is invalid") from error
    if raw != canonical_json_bytes(value):
        raise ValueError("checkpoint is noncanonical")
    return value


def _write_create_only(path: Path, data: bytes) -> None:
    details = _lstat(path)
    if details is not None:
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise ValueError("checkpoint destination is invalid")
        if path.read_bytes() != data:
            raise ValueError("checkpoint already exists with different data")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("checkpoint already exists with different data") from None
        return
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _catalog_entry(catalog: object, dataset: str) -> dict[str, object]:
    if not isinstance(catalog, dict) or not isinstance(catalog.get("datasets"), list):
        raise ValueError("checkpoint catalog is invalid")
    entries = [entry for entry in catalog["datasets"] if isinstance(entry, dict) and entry.get("dataset") == dataset]
    if len(entries) != 1:
        raise ValueError("checkpoint catalog dataset differs")
    return entries[0]


def _check_context(context: object, catalog: object) -> Exp450RunContext:
    if not isinstance(context, Exp450RunContext) or not isinstance(catalog, dict):
        raise ValueError("checkpoint context is invalid")
    if catalog.get("catalog_sha256") != context.catalog_sha256:
        raise ValueError("checkpoint catalog context differs")
    return context


def _check_root(root: Path, context: Exp450RunContext) -> Path:
    root = Path(root)
    if root != context.run_root:
        raise ValueError("checkpoint root differs from bound context")
    return root


def _raw_identity(raw: object, entry: dict[str, object]) -> None:
    if not isinstance(raw, RawDataset) or raw.dataset != entry.get("dataset") or raw.metadata_length != entry.get("series_length"):
        raise ValueError("checkpoint raw identity differs")
    for indices, blobs, rows, identity in (
        (raw.train_indices, raw.train_blobs, raw.train_rows, entry.get("train")),
        (raw.test_indices, raw.test_blobs, raw.test_rows, entry.get("test")),
    ):
        if not isinstance(identity, dict):
            raise ValueError("checkpoint raw identity differs")
        if (
            tuple(indices) != tuple(identity.get("indices", ()))
            or len(blobs) != identity.get("count")
            or [sha256(bytes(blob)).hexdigest() for blob in blobs] != identity.get("values_blob_sha256")
            or len(rows) != len(blobs)
        ):
            raise ValueError("checkpoint raw identity differs")
        for blob, row in zip(blobs, rows, strict=True):
            decoded = np.frombuffer(blob, dtype="<f4")
            array = np.asarray(row)
            if array.dtype != np.dtype("<f4") or array.ndim != 1 or array.shape != decoded.shape or array.tobytes() != blob:
                raise ValueError("checkpoint raw identity differs")


def _lineage_record(origin: object) -> dict[str, object]:
    if origin not in LINEAGE_NODES:
        raise ValueError("checkpoint source origin differs")
    return {
        "node_id": origin.node_id,
        "stage": origin.stage,
        "input_kind": origin.input_kind,
        "module": origin.module,
        "symbols": list(origin.symbols),
        "source_sha256": origin.source_sha256,
        "paper_module": origin.paper_module,
        "label_access": origin.label_access,
        "parity_oracle": origin.parity_oracle,
    }


def _lineage_from_record(value: object):
    for origin in LINEAGE_NODES:
        if value == _lineage_record(origin):
            return origin
    raise ValueError("checkpoint source origin differs")


def _float_array(value: object, size: int, name: str, *, pvalue: bool = False) -> list[float]:
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or array.ndim != 1 or array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(f"checkpoint {name} differs")
    result = [float(item) for item in array]
    if pvalue and any(item <= 0.0 or item > 1.0 for item in result):
        raise ValueError(f"checkpoint {name} differs")
    return result


def _bool_array(value: object, size: int, name: str) -> list[bool]:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.ndim != 1 or array.size != size:
        raise ValueError(f"checkpoint {name} differs")
    return [bool(item) for item in array]


def _indices(value: object, n_test: int, name: str) -> list[int]:
    if not isinstance(value, frozenset) or any(type(index) is not int or index < 0 or index >= n_test for index in value):
        raise ValueError(f"checkpoint {name} differs")
    return sorted(value)


def _indices_from_payload(value: object, n_test: int, name: str) -> frozenset[int]:
    if not isinstance(value, list) or value != sorted(set(value)) or any(type(index) is not int or index < 0 or index >= n_test for index in value):
        raise ValueError(f"checkpoint {name} differs")
    return frozenset(value)


def _lanes(value: object, n_test: int, name: str) -> dict[str, list[int]]:
    if not isinstance(value, RouteLanes):
        raise ValueError(f"checkpoint {name} differs")
    return {"hard": _indices(value.hard, n_test, name), "standard": _indices(value.standard, n_test, name), "priority": _indices(value.priority, n_test, name)}


def _lanes_from_payload(value: object, n_test: int, name: str) -> RouteLanes:
    if not isinstance(value, dict) or set(value) != {"hard", "standard", "priority"}:
        raise ValueError(f"checkpoint {name} differs")
    return RouteLanes(
        _indices_from_payload(value["hard"], n_test, name),
        _indices_from_payload(value["standard"], n_test, name),
        _indices_from_payload(value["priority"], n_test, name),
    )


def _source_payload(sources: object, n_train: int, n_test: int) -> list[dict[str, object]]:
    if not isinstance(sources, HardSourceBundle) or tuple(sources.source_names) != EXPECTED_HARD_SOURCES or set(sources.by_name) != set(EXPECTED_HARD_SOURCES):
        raise ValueError("checkpoint sources differ")
    rows = []
    for order, name in enumerate(EXPECTED_HARD_SOURCES):
        result = sources.by_name[name]
        if not isinstance(result, HardSourceResult) or result.name != name or result.origin.node_id != _SOURCE_ORIGIN_IDS[name] or type(result.eligible) is not bool or type(result.eligibility_reason) is not str or not result.eligibility_reason:
            raise ValueError("checkpoint source differs")
        if type(result.threshold) not in (int, float) or not math.isfinite(float(result.threshold)) or float(result.threshold) != SOURCE_ALPHA_PAPER[name]:
            raise ValueError("checkpoint source threshold differs")
        pvalues = _float_array(result.test_pvalue, n_test, "source pvalue", pvalue=True)
        native = _bool_array(result.test_native_pass, n_test, "source pass")
        if result.eligible:
            train = _float_array(result.train_oof_score, n_train, "source TRAIN OOF score")
            test = _float_array(result.test_score, n_test, "source TEST score")
            if result.eligibility_reason != "eligible" or native != [item <= result.threshold for item in pvalues] or not np.array_equal(np.asarray(pvalues), upper_tail_pvalues(np.asarray(train), np.asarray(test))):
                raise ValueError("checkpoint eligible source relation differs")
        else:
            _float_array(result.train_oof_score, 0, "abstained source TRAIN OOF score")
            _float_array(result.test_score, 0, "abstained source TEST score")
            if pvalues != [1.0] * n_test or any(native):
                raise ValueError("checkpoint abstained source relation differs")
            train, test = [], []
        rows.append({"source_id": name, "order": order, "origin": _lineage_record(result.origin), "train_oof_score": train, "test_score": test, "test_pvalue": pvalues, "threshold": float(result.threshold), "test_native_pass": native, "eligible": result.eligible, "eligibility_reason": result.eligibility_reason})
    return rows


def _sources_from_payload(value: object, n_train: int, n_test: int) -> HardSourceBundle:
    required = {"source_id", "order", "origin", "train_oof_score", "test_score", "test_pvalue", "threshold", "test_native_pass", "eligible", "eligibility_reason"}
    if not isinstance(value, list) or [row.get("source_id") if isinstance(row, dict) else None for row in value] != list(EXPECTED_HARD_SOURCES):
        raise ValueError("checkpoint source order differs")
    results = {}
    for order, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != required or row["order"] != order or type(row["eligible"]) is not bool or type(row["eligibility_reason"]) is not str or not row["eligibility_reason"]:
            raise ValueError("checkpoint source schema differs")
        name = row["source_id"]
        if type(row["threshold"]) not in (int, float) or not math.isfinite(float(row["threshold"])) or float(row["threshold"]) != SOURCE_ALPHA_PAPER[name]:
            raise ValueError("checkpoint source threshold differs")
        pvalues = np.asarray(_float_array(np.asarray(row["test_pvalue"], dtype=np.float64), n_test, "source pvalue", pvalue=True), dtype=np.float64)
        native = np.asarray(_bool_array(np.asarray(row["test_native_pass"], dtype=bool), n_test, "source pass"), dtype=bool)
        if row["eligible"]:
            train = np.asarray(_float_array(np.asarray(row["train_oof_score"], dtype=np.float64), n_train, "source TRAIN OOF score"), dtype=np.float64)
            test = np.asarray(_float_array(np.asarray(row["test_score"], dtype=np.float64), n_test, "source TEST score"), dtype=np.float64)
            if row["eligibility_reason"] != "eligible" or not np.array_equal(native, pvalues <= float(row["threshold"])) or not np.array_equal(pvalues, upper_tail_pvalues(train, test)):
                raise ValueError("checkpoint eligible source relation differs")
        else:
            if row["train_oof_score"] != [] or row["test_score"] != [] or not np.array_equal(pvalues, np.ones(n_test)) or np.any(native):
                raise ValueError("checkpoint abstained source relation differs")
            train, test = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        for array in (train, test, pvalues, native):
            array.setflags(write=False)
        origin = _lineage_from_record(row["origin"])
        if origin.node_id != _SOURCE_ORIGIN_IDS[name]:
            raise ValueError("checkpoint source origin differs")
        results[name] = HardSourceResult(name, origin, train, test, pvalues, float(row["threshold"]), native, row["eligible"], row["eligibility_reason"])
    return HardSourceBundle(EXPECTED_HARD_SOURCES, results)


def _route_payload(route: object, n_test: int) -> dict[str, object]:
    if not isinstance(route, HardRouteResult) or route.n_test != n_test:
        raise ValueError("checkpoint route differs")
    return {"n_test": n_test, "exp74d": _lanes(route.exp74d, n_test, "exp74d"), "exp89": _indices(route.exp89, n_test, "exp89"), "exp90": _indices(route.exp90, n_test, "exp90"), "exp93": _indices(route.exp93, n_test, "exp93"), "exp137": _lanes(route.exp137, n_test, "exp137"), "exp234": {"minimum_two": _indices(route.exp234.minimum_two, n_test, "exp234"), "minimum_three": _indices(route.exp234.minimum_three, n_test, "exp234")}, "exp236": _lanes(route.exp236, n_test, "exp236"), "promoted": _indices(route.promoted, n_test, "promoted"), "exp253": _lanes(route.exp253, n_test, "exp253"), "exact_normal_removed": _indices(route.exact_normal_removed, n_test, "exact normal removed"), "automatic_hard": _indices(route.automatic_hard, n_test, "automatic hard")}


def _route_from_payload(value: object, n_test: int) -> HardRouteResult:
    required = {"n_test", "exp74d", "exp89", "exp90", "exp93", "exp137", "exp234", "exp236", "promoted", "exp253", "exact_normal_removed", "automatic_hard"}
    if not isinstance(value, dict) or set(value) != required or value["n_test"] != n_test:
        raise ValueError("checkpoint route schema differs")
    exp234 = value["exp234"]
    if not isinstance(exp234, dict) or set(exp234) != {"minimum_two", "minimum_three"}:
        raise ValueError("checkpoint route schema differs")
    return HardRouteResult(_lanes_from_payload(value["exp74d"], n_test, "exp74d"), _indices_from_payload(value["exp89"], n_test, "exp89"), _indices_from_payload(value["exp90"], n_test, "exp90"), _indices_from_payload(value["exp93"], n_test, "exp93"), _lanes_from_payload(value["exp137"], n_test, "exp137"), SourceExpansion(_indices_from_payload(exp234["minimum_two"], n_test, "exp234"), _indices_from_payload(exp234["minimum_three"], n_test, "exp234")), _lanes_from_payload(value["exp236"], n_test, "exp236"), _indices_from_payload(value["promoted"], n_test, "promoted"), _lanes_from_payload(value["exp253"], n_test, "exp253"), _indices_from_payload(value["exact_normal_removed"], n_test, "exact normal removed"), _indices_from_payload(value["automatic_hard"], n_test, "automatic hard"), n_test)


def _payload(raw: RawDataset, result: Exp450DatasetConditionResult, entry: dict[str, object]) -> dict[str, object]:
    _raw_identity(raw, entry)
    if not isinstance(result, Exp450DatasetConditionResult) or result.dataset != raw.dataset:
        raise ValueError("checkpoint result dataset differs")
    return {"sources": _source_payload(result.sources, entry["train"]["count"], entry["test"]["count"]), "route": _route_payload(result.route, entry["test"]["count"])}


def _success_path(root: Path, dataset: str, condition_id: str) -> Path:
    token = sha256(dataset.encode("utf-8")).hexdigest()
    return _path(root, "checkpoints_antigravity", f"{condition_id}_antigravity", f"{token}_antigravity", SUCCESS_NAME)


def _success_envelope(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw: RawDataset, result: Exp450DatasetConditionResult) -> dict[str, object]:
    _check_context(context, catalog)
    if result.condition_id not in CONDITION_IDS:
        raise ValueError("checkpoint condition differs")
    entry = _catalog_entry(catalog, raw.dataset)
    payload = _payload(raw, result, entry)
    return {"schema": SUCCESS_SCHEMA, "context_sha256": context.sha256, "catalog_sha256": context.catalog_sha256, "condition_id": result.condition_id, "dataset": result.dataset, "raw_identity": entry, "backend": context.backend.artifact_record(), "payload": payload, "payload_sha256": _sha(payload)}


def write_dataset_condition_success_antigravity(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw: RawDataset, result: Exp450DatasetConditionResult) -> Path:
    """Create one verified typed success checkpoint for a dataset and condition."""
    root = _check_root(root, context)
    envelope = _success_envelope(root, context, catalog, raw, result)
    entry = _catalog_entry(catalog, raw.dataset)
    reread_sources = _sources_from_payload(envelope["payload"]["sources"], entry["train"]["count"], entry["test"]["count"])
    reread_route = _route_from_payload(envelope["payload"]["route"], entry["test"]["count"])
    if canonical_json_bytes(_payload(raw, Exp450DatasetConditionResult(raw.dataset, result.condition_id, reread_sources, reread_route), entry)) != canonical_json_bytes(envelope["payload"]):
        raise ValueError("checkpoint typed round-trip differs")
    path = _success_path(root, raw.dataset, result.condition_id)
    _write_create_only(path, canonical_json_bytes(envelope))
    return path


def read_dataset_condition_success_antigravity(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw: RawDataset, condition: Exp450Condition) -> Exp450DatasetConditionResult:
    """Read and validate exactly one context-bound typed success checkpoint."""
    _check_context(context, catalog)
    root = _check_root(root, context)
    if not isinstance(condition, Exp450Condition):
        raise ValueError("checkpoint condition differs")
    entry = _catalog_entry(catalog, raw.dataset)
    _raw_identity(raw, entry)
    value = _canonical_read(_success_path(root, raw.dataset, condition.condition_id))
    required = {"schema", "context_sha256", "catalog_sha256", "condition_id", "dataset", "raw_identity", "backend", "payload", "payload_sha256"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SUCCESS_SCHEMA or value["dataset"] != raw.dataset or value["condition_id"] != condition.condition_id:
        raise ValueError("checkpoint schema differs")
    if value["context_sha256"] != context.sha256 or value["catalog_sha256"] != context.catalog_sha256 or value["raw_identity"] != entry or value["backend"] != context.backend.artifact_record():
        raise ValueError("checkpoint context differs")
    if type(value["payload_sha256"]) is not str or value["payload_sha256"] != _sha(value["payload"]):
        raise ValueError("checkpoint payload hash differs")
    if not isinstance(value["payload"], dict) or set(value["payload"]) != {"sources", "route"}:
        raise ValueError("checkpoint payload schema differs")
    sources = _sources_from_payload(value["payload"]["sources"], entry["train"]["count"], entry["test"]["count"])
    route = _route_from_payload(value["payload"]["route"], entry["test"]["count"])
    result = Exp450DatasetConditionResult(raw.dataset, condition.condition_id, sources, route)
    if canonical_json_bytes(_payload(raw, result, entry)) != canonical_json_bytes(value["payload"]):
        raise ValueError("checkpoint typed round-trip differs")
    return result


def canonical_result_bytes_antigravity(result: Exp450DatasetConditionResult) -> bytes:
    """Canonical byte representation of every serializable typed science field."""
    if not isinstance(result, Exp450DatasetConditionResult):
        raise TypeError("Exp450 dataset condition result is required")
    n_train = next((len(source.train_oof_score) for source in result.sources.by_name.values() if source.eligible), 0)
    return canonical_json_bytes({"dataset": result.dataset, "condition_id": result.condition_id, "payload": {"sources": _source_payload(result.sources, n_train, result.route.n_test), "route": _route_payload(result.route, result.route.n_test)}})


def _failure_directory(root: Path) -> Path:
    return _path(root, "checkpoints_antigravity", "failures_antigravity")


def _failure_path(root: Path, attempt: int) -> Path:
    return _path(root, "checkpoints_antigravity", "failures_antigravity", f"attempt_{attempt:06d}_antigravity.json")


def _failure_paths(root: Path) -> list[Path]:
    directory = _failure_directory(root)
    details = _lstat(directory)
    if details is None:
        return []
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("checkpoint failure ledger differs")
    paths = sorted(directory.iterdir())
    if [path.name for path in paths] != [f"attempt_{index:06d}_antigravity.json" for index in range(1, len(paths) + 1)]:
        raise ValueError("checkpoint failure ledger differs")
    return paths


def _failure_record_from_value(
    value: object,
    attempt: int,
    context: Exp450RunContext,
    catalog: dict[str, object],
    raw_by_dataset: dict[str, RawDataset] | None = None,
) -> DatasetConditionFailureAntigravity:
    required = {"schema", "context_sha256", "catalog_sha256", "dataset", "condition_id", "raw_identity", "backend", "wave", "attempt", "exception_type", "exception_message", "record_sha256"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != FAILURE_SCHEMA or value.get("attempt") != attempt:
        raise ValueError("checkpoint failure record differs")
    bare = {key: child for key, child in value.items() if key != "record_sha256"}
    if value["record_sha256"] != _sha(bare) or value["context_sha256"] != context.sha256 or value["catalog_sha256"] != context.catalog_sha256 or value["backend"] != context.backend.artifact_record():
        raise ValueError("checkpoint failure context differs")
    dataset, condition_id = value["dataset"], value["condition_id"]
    if condition_id not in CONDITION_IDS or value["wave"] not in {1, 2} or type(value["exception_type"]) is not str or not value["exception_type"] or type(value["exception_message"]) is not str:
        raise ValueError("checkpoint failure record differs")
    entry = _catalog_entry(catalog, dataset)
    if value["raw_identity"] != entry:
        raise ValueError("checkpoint failure raw identity differs")
    if raw_by_dataset is not None:
        if dataset not in raw_by_dataset:
            raise ValueError("checkpoint failure record differs")
        _raw_identity(raw_by_dataset[dataset], entry)
    return DatasetConditionFailureAntigravity(
        dataset, condition_id, value["wave"], attempt,
        value["exception_type"], value["exception_message"],
    )


def _validate_failure_ledger(records: tuple[DatasetConditionFailureAntigravity, ...]) -> None:
    """Accept only first-wave records followed by one optional retry per coordinate."""
    seen: set[tuple[str, str, int]] = set()
    first_wave: set[tuple[str, str]] = set()
    second_wave_started = False
    for record in records:
        coordinate = (record.dataset, record.condition_id)
        identity = (*coordinate, record.wave)
        if identity in seen:
            raise ValueError("checkpoint failure ledger has duplicate coordinate wave")
        if record.wave == 1:
            if second_wave_started:
                raise ValueError("checkpoint failure ledger has wave one after wave two")
            first_wave.add(coordinate)
        else:
            second_wave_started = True
            if coordinate not in first_wave:
                raise ValueError("checkpoint failure ledger has wave two without wave one")
        seen.add(identity)


def _append_failure_history(
    root: Path, context: Exp450RunContext, catalog: dict[str, object]
) -> tuple[DatasetConditionFailureAntigravity, ...]:
    records = tuple(
        _failure_record_from_value(_canonical_read(path), attempt, context, catalog)
        for attempt, path in enumerate(_failure_paths(root), 1)
    )
    _validate_failure_ledger(records)
    return records


def append_dataset_condition_failure_antigravity(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw: RawDataset, condition: Exp450Condition, wave: int, error: Exception) -> Path:
    """Append the next global failure ledger record; it never counts as success."""
    _check_context(context, catalog)
    root = _check_root(root, context)
    if not isinstance(condition, Exp450Condition) or type(wave) is not int or wave not in {1, 2} or not isinstance(error, Exception):
        raise ValueError("checkpoint failure identity differs")
    entry = _catalog_entry(catalog, raw.dataset)
    _raw_identity(raw, entry)
    history = _append_failure_history(root, context, catalog)
    attempt = len(history) + 1
    _validate_failure_ledger(
        (*history, DatasetConditionFailureAntigravity(
            raw.dataset, condition.condition_id, wave, attempt,
            type(error).__name__, str(error),
        ))
    )
    bare = {"schema": FAILURE_SCHEMA, "context_sha256": context.sha256, "catalog_sha256": context.catalog_sha256, "dataset": raw.dataset, "condition_id": condition.condition_id, "raw_identity": entry, "backend": context.backend.artifact_record(), "wave": wave, "attempt": attempt, "exception_type": type(error).__name__, "exception_message": str(error)}
    path = _failure_path(root, attempt)
    _write_create_only(path, canonical_json_bytes({**bare, "record_sha256": _sha(bare)}))
    return path


def _failure_records(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw_by_dataset: dict[str, RawDataset]) -> tuple[DatasetConditionFailureAntigravity, ...]:
    records = tuple(
        _failure_record_from_value(
            _canonical_read(path), attempt, context, catalog, raw_by_dataset
        )
        for attempt, path in enumerate(_failure_paths(root), 1)
    )
    _validate_failure_ledger(records)
    return records


def verify_dataset_condition_inventory_antigravity(root: Path, context: Exp450RunContext, catalog: dict[str, object], raw_by_dataset: dict[str, RawDataset]) -> DatasetConditionInventoryAntigravity:
    """Verify successes and reconstruct the one allowed second-wave work queue."""
    _check_context(context, catalog)
    root = _check_root(root, context)
    if not isinstance(raw_by_dataset, dict):
        raise ValueError("checkpoint raw inventory differs")
    entries = catalog.get("datasets")
    if not isinstance(entries, list) or {
        entry.get("dataset") for entry in entries if isinstance(entry, dict)
    } != set(raw_by_dataset) or len(entries) != len(raw_by_dataset):
        raise ValueError("checkpoint catalog order or raw inventory differs")
    completed = []
    for entry in entries:
        dataset = entry["dataset"]
        raw = raw_by_dataset[dataset]
        _raw_identity(raw, entry)
        for condition_id in CONDITION_IDS:
            condition = {"A": Exp450Condition("A", "linear", "linear"), "B": Exp450Condition("B", "polyphase_pchip", "linear"), "C": Exp450Condition("C", "linear", "area"), "D": Exp450Condition("D", "polyphase_pchip", "area")}[condition_id]
            path = _success_path(root, dataset, condition_id)
            if _lstat(path) is not None:
                read_dataset_condition_success_antigravity(root, context, catalog, raw, condition)
                completed.append((dataset, condition_id))
    failures = _failure_records(root, context, catalog, raw_by_dataset)
    complete = set(completed)
    exhausted = {
        (failure.dataset, failure.condition_id)
        for failure in failures
        if failure.wave == 2
    }
    all_coordinates = tuple((entry["dataset"], condition_id) for entry in entries for condition_id in CONDITION_IDS)
    return DatasetConditionInventoryAntigravity(
        tuple(completed),
        failures,
        tuple(
            coordinate
            for coordinate in all_coordinates
            if coordinate not in complete and coordinate not in exhausted
        ),
    )


__all__ = [
    "DatasetConditionFailureAntigravity",
    "DatasetConditionInventoryAntigravity",
    "TASK4_TASK5_ARTIFACT_SOURCE_ANTIGRAVITY",
    "append_dataset_condition_failure_antigravity",
    "canonical_result_bytes_antigravity",
    "read_dataset_condition_success_antigravity",
    "verify_dataset_condition_inventory_antigravity",
    "write_dataset_condition_success_antigravity",
]
