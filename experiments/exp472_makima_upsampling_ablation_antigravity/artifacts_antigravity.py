"""Durable, create-only condition-E score artifacts for Exp472."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

import numpy as np

from .contracts_antigravity import (
    Exp472RunContextAntigravity,
    Exp472ScoreManifestAntigravity,
    Exp472ScoreShardAntigravity,
    canonical_json_bytes_exp472_antigravity,
)


_SHARD_SCHEMA_ANTIGRAVITY = "exp472-score-shard-antigravity.v1"
_MANIFEST_SCHEMA_ANTIGRAVITY = "exp472-score-manifest-antigravity.v1"
_PROGRESS_SCHEMA_ANTIGRAVITY = "exp472-score-progress-antigravity.v1"
_FAILURE_SCHEMA_ANTIGRAVITY = "exp472-score-failure-antigravity.v1"
_SHARD_FILENAME_ANTIGRAVITY = "score_shard_antigravity.json"
_MANIFEST_FILENAME_ANTIGRAVITY = "exp472_score_manifest_antigravity.json"
_PROGRESS_FILENAME_ANTIGRAVITY = "progress_antigravity.jsonl"
_FAILURE_FILENAME_ANTIGRAVITY = "failures_antigravity.jsonl"


def _dataset_from_entry_antigravity(entry: object) -> str:
    value = entry.get("dataset") if isinstance(entry, Mapping) else getattr(entry, "dataset", None)
    if type(value) is not str or not value:
        raise ValueError("Exp472 catalog entry dataset is invalid")
    return value


def _entry_identity_antigravity(entry: object) -> object | None:
    value = entry.get("test_identity") if isinstance(entry, Mapping) else getattr(entry, "test_identity", None)
    return value


def _safe_root_antigravity(root: Path, *, create: bool) -> Path:
    target = Path(root)
    if not target.is_absolute():
        raise ValueError("Exp472 artifact root must be absolute")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    try:
        details = target.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 artifact root is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp472 artifact root is unsafe")
    return target


def _shard_path_antigravity(root: Path, dataset: str, *, create: bool) -> Path:
    safe_root = _safe_root_antigravity(root, create=create)
    token = sha256(dataset.encode("utf-8")).hexdigest()
    directory = safe_root / "score_shards_antigravity" / f"{token}_antigravity"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        details = directory.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 score shard parent is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp472 score shard parent is unsafe")
    return directory / _SHARD_FILENAME_ANTIGRAVITY


def _read_canonical_record_antigravity(path: Path) -> dict[str, object]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 artifact is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp472 artifact is unsafe")
    raw = path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp472 artifact JSON is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp472_antigravity(record):
        raise ValueError("Exp472 artifact is noncanonical")
    return record


def _fsync_directory_antigravity(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_bytes_antigravity(path: Path, data: bytes) -> None:
    """Create a shard with O_EXCL; an existing byte-identical shard is reusable."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _read_canonical_record_antigravity(path)
        if canonical_json_bytes_exp472_antigravity(existing) != data:
            raise ValueError("Exp472 create-only artifact has different bytes") from None
        return
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp472 artifact write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_antigravity(path.parent)


def _append_jsonl_antigravity(path: Path, record: Mapping[str, object]) -> None:
    data = canonical_json_bytes_exp472_antigravity(dict(record))
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp472 JSONL append failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_antigravity(path.parent)


def _condition_record_antigravity() -> dict[str, str]:
    return {"id": "E", "downsampling": "polyphase", "upsampling": "makima", "image_mode": "area"}


def score_shard_record_exp472_antigravity(
    context: Exp472RunContextAntigravity, shard: Exp472ScoreShardAntigravity
) -> dict[str, object]:
    """Serialize one digest-bound, finite condition-E score result."""
    if not isinstance(context, Exp472RunContextAntigravity) or not isinstance(shard, Exp472ScoreShardAntigravity):
        raise ValueError("Exp472 score shard inputs are invalid")
    bare: dict[str, object] = {
        "schema": _SHARD_SCHEMA_ANTIGRAVITY,
        "context_sha256": context.sha256,
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "code_closure": dict(context.code_closure),
        "condition": _condition_record_antigravity(),
        "backend": context.backend.artifact_record(),
        "dataset": shard.dataset,
        "test_identity": dict(shard.test_identity),
        "score_dtype": "float64",
        "scores": [float(value) for value in shard.scores],
        "alerts": sorted(shard.alerts),
        "upsampled_rows": shard.upsampled_rows,
        "diagnostic": dict(shard.diagnostic),
    }
    return {
        **bare,
        "shard_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(),
    }


def _verify_shard_record_antigravity(
    record: Mapping[str, object], context: Exp472RunContextAntigravity, entry: object
) -> dict[str, object]:
    expected = {
        "schema", "context_sha256", "database_sha256", "catalog_sha256", "code_closure",
        "condition", "backend", "dataset", "test_identity", "score_dtype", "scores",
        "alerts", "upsampled_rows", "diagnostic", "shard_sha256",
    }
    if set(record) != expected or record.get("schema") != _SHARD_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp472 score shard schema differs")
    if (
        record["context_sha256"] != context.sha256
        or record["database_sha256"] != context.database_sha256
        or record["catalog_sha256"] != context.catalog_sha256
        or record["code_closure"] != dict(context.code_closure)
        or record["condition"] != _condition_record_antigravity()
        or record["backend"] != context.backend.artifact_record()
        or record["dataset"] != _dataset_from_entry_antigravity(entry)
    ):
        raise ValueError("Exp472 score shard identity differs")
    entry_identity = _entry_identity_antigravity(entry)
    if entry_identity is not None and canonical_json_bytes_exp472_antigravity(record["test_identity"]) != canonical_json_bytes_exp472_antigravity(entry_identity):
        raise ValueError("Exp472 score shard TEST identity differs")
    if record["score_dtype"] != "float64" or not isinstance(record["scores"], list):
        raise ValueError("Exp472 score shard score encoding differs")
    try:
        reconstructed = Exp472ScoreShardAntigravity(
            dataset=str(record["dataset"]),
            test_identity=record["test_identity"],
            scores=np.asarray(record["scores"], dtype=np.float64),
            alerts=frozenset(record["alerts"]),
            upsampled_rows=record["upsampled_rows"],
            diagnostic=record["diagnostic"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exp472 score shard payload differs") from error
    bare = {key: value for key, value in record.items() if key != "shard_sha256"}
    if record["shard_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest():
        raise ValueError("Exp472 score shard digest differs")
    if score_shard_record_exp472_antigravity(context, reconstructed) != dict(record):
        raise ValueError("Exp472 score shard canonical payload differs")
    return dict(record)


def publish_score_shard_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, entry: object, shard: Exp472ScoreShardAntigravity
) -> Path:
    """Durably create one immutable score shard, or reuse identical bytes only."""
    if _dataset_from_entry_antigravity(entry) != shard.dataset:
        raise ValueError("Exp472 score shard dataset identity differs")
    entry_identity = _entry_identity_antigravity(entry)
    if entry_identity is not None and canonical_json_bytes_exp472_antigravity(entry_identity) != canonical_json_bytes_exp472_antigravity(shard.test_identity):
        raise ValueError("Exp472 score shard TEST identity differs")
    destination = _shard_path_antigravity(Path(root), shard.dataset, create=True)
    _create_only_bytes_antigravity(destination, canonical_json_bytes_exp472_antigravity(score_shard_record_exp472_antigravity(context, shard)))
    _verify_shard_record_antigravity(_read_canonical_record_antigravity(destination), context, entry)
    return destination


def verify_score_shard_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, entry: object
) -> dict[str, object]:
    """Reopen and verify a shard without invoking a scorer."""
    dataset = _dataset_from_entry_antigravity(entry)
    return _verify_shard_record_antigravity(
        _read_canonical_record_antigravity(_shard_path_antigravity(Path(root), dataset, create=False)), context, entry
    )


def _catalog_entries_antigravity(catalog: Sequence[object]) -> tuple[object, ...]:
    entries = tuple(catalog)
    datasets = tuple(_dataset_from_entry_antigravity(entry) for entry in entries)
    if not entries or len(set(datasets)) != len(datasets):
        raise ValueError("Exp472 catalog coverage is invalid")
    return entries


def _manifest_record_antigravity(
    context: Exp472RunContextAntigravity, entries: tuple[object, ...], shards: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    dataset_order = [_dataset_from_entry_antigravity(entry) for entry in entries]
    descriptors = [
        {"dataset": dataset, "shard_sha256": shard["shard_sha256"]}
        for dataset, shard in zip(dataset_order, shards, strict=True)
    ]
    bare: dict[str, object] = {
        "schema": _MANIFEST_SCHEMA_ANTIGRAVITY,
        "mode": context.mode,
        "context_sha256": context.sha256,
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "dataset_order": dataset_order,
        "descriptors": descriptors,
    }
    return {**bare, "manifest_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}


def _verify_manifest_record_antigravity(
    record: Mapping[str, object], root: Path, context: Exp472RunContextAntigravity, entries: tuple[object, ...]
) -> dict[str, object]:
    expected = {"schema", "mode", "context_sha256", "database_sha256", "catalog_sha256", "dataset_order", "descriptors", "manifest_sha256"}
    if set(record) != expected or record.get("schema") != _MANIFEST_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp472 score manifest schema differs")
    if record.get("mode") != context.mode or record.get("context_sha256") != context.sha256 or record.get("database_sha256") != context.database_sha256 or record.get("catalog_sha256") != context.catalog_sha256:
        raise ValueError("Exp472 score manifest identity differs")
    datasets = [_dataset_from_entry_antigravity(entry) for entry in entries]
    if record.get("dataset_order") != datasets or not isinstance(record.get("descriptors"), list):
        raise ValueError("Exp472 score manifest coverage differs")
    bare = {key: value for key, value in record.items() if key != "manifest_sha256"}
    if record.get("manifest_sha256") != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest():
        raise ValueError("Exp472 score manifest digest differs")
    descriptors = record["descriptors"]
    if len(descriptors) != len(entries):
        raise ValueError("Exp472 score manifest coverage differs")
    shard_hashes: dict[str, str] = {}
    for entry, descriptor in zip(entries, descriptors, strict=True):
        if not isinstance(descriptor, dict) or set(descriptor) != {"dataset", "shard_sha256"}:
            raise ValueError("Exp472 score manifest descriptor differs")
        shard = verify_score_shard_exp472_antigravity(root, context, entry)
        if descriptor["dataset"] != shard["dataset"] or descriptor["shard_sha256"] != shard["shard_sha256"]:
            raise ValueError("Exp472 score manifest shard differs")
        shard_hashes[str(shard["dataset"])] = str(shard["shard_sha256"])
    Exp472ScoreManifestAntigravity(
        mode=context.mode,
        context_sha256=context.sha256,
        dataset_order=tuple(datasets),
        shard_sha256=shard_hashes,
        manifest_sha256=str(record["manifest_sha256"]),
    )
    return dict(record)


def publish_manifest_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, catalog: Sequence[object], *, failures: set[str] | frozenset[str] = frozenset()
) -> Path:
    """Publish a terminal manifest only after complete scorer-free re-verification."""
    entries = _catalog_entries_antigravity(catalog)
    if failures:
        raise ValueError("Exp472 terminal manifest requires zero failures")
    shards = [verify_score_shard_exp472_antigravity(root, context, entry) for entry in entries]
    record = _manifest_record_antigravity(context, entries, shards)
    destination = _safe_root_antigravity(Path(root), create=True) / _MANIFEST_FILENAME_ANTIGRAVITY
    _create_only_bytes_antigravity(destination, canonical_json_bytes_exp472_antigravity(record))
    _verify_manifest_record_antigravity(_read_canonical_record_antigravity(destination), Path(root), context, entries)
    return destination


def verify_manifest_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, catalog: Sequence[object]
) -> dict[str, object]:
    """Independently reopen every listed shard before accepting a manifest."""
    entries = _catalog_entries_antigravity(catalog)
    path = _safe_root_antigravity(Path(root), create=False) / _MANIFEST_FILENAME_ANTIGRAVITY
    return _verify_manifest_record_antigravity(_read_canonical_record_antigravity(path), Path(root), context, entries)


def append_progress_exp472_antigravity(
    root: Path, *, completed: int, failed: int, remaining: int, elapsed_seconds: float, dataset: str, wave: int, status: str
) -> None:
    """Durably append one immutable lifecycle observation."""
    if status not in {"completed", "failed"} or wave not in {1, 2} or type(dataset) is not str or not dataset:
        raise ValueError("Exp472 progress record is invalid")
    if any(type(value) is not int or value < 0 for value in (completed, failed, remaining)) or type(elapsed_seconds) not in {int, float} or elapsed_seconds < 0:
        raise ValueError("Exp472 progress counters are invalid")
    safe_root = _safe_root_antigravity(Path(root), create=True)
    _append_jsonl_antigravity(safe_root / _PROGRESS_FILENAME_ANTIGRAVITY, {
        "schema": _PROGRESS_SCHEMA_ANTIGRAVITY, "completed": completed, "failed": failed,
        "remaining": remaining, "elapsed_seconds": float(elapsed_seconds), "dataset": dataset,
        "wave": wave, "status": status,
    })


def append_failure_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, *, dataset: str, wave: int, error: BaseException
) -> None:
    """Durably preserve a bounded-wave failure without stopping other work."""
    if type(dataset) is not str or not dataset or wave not in {1, 2}:
        raise ValueError("Exp472 failure record is invalid")
    safe_root = _safe_root_antigravity(Path(root), create=True)
    _append_jsonl_antigravity(safe_root / _FAILURE_FILENAME_ANTIGRAVITY, {
        "schema": _FAILURE_SCHEMA_ANTIGRAVITY, "context_sha256": context.sha256,
        "dataset": dataset, "wave": wave, "error_type": type(error).__name__, "error": str(error),
    })


def _jsonl_records_antigravity(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp472 JSONL artifact is unsafe")
    records: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines(keepends=True):
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp472 JSONL artifact is invalid") from error
        if not isinstance(record, dict) or canonical_json_bytes_exp472_antigravity(record) != line:
            raise ValueError("Exp472 JSONL artifact is noncanonical")
        records.append(record)
    return records


def status_exp472_antigravity(root: Path, *, total: int = 1117, now_monotonic: float | None = None) -> dict[str, object]:
    """Report current durable progress with rate and ETA from completed work."""
    if type(total) is not int or total < 1:
        raise ValueError("Exp472 status total is invalid")
    safe_root = _safe_root_antigravity(Path(root), create=False)
    records = _jsonl_records_antigravity(safe_root / _PROGRESS_FILENAME_ANTIGRAVITY)
    latest = records[-1] if records else None
    completed = int(latest["completed"]) if latest else 0
    failed = int(latest["failed"]) if latest else 0
    remaining = int(latest["remaining"]) if latest else total
    elapsed = float(latest["elapsed_seconds"]) if latest else 0.0
    del now_monotonic
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = remaining / rate if rate > 0 else None
    return {
        "schema": "exp472-score-status-antigravity.v1", "completed": completed, "failed": failed,
        "remaining": remaining, "total": total, "rate_per_second": rate, "eta_seconds": eta,
        "manifest_complete": (safe_root / _MANIFEST_FILENAME_ANTIGRAVITY).is_file(),
        "evaluation_complete": (safe_root / "exp472_evaluation_antigravity.json").is_file(),
        "report_complete": (safe_root / "exp472_report_antigravity.json").is_file(),
        "latest_progress": latest,
    }


__all__ = [
    "append_failure_exp472_antigravity", "append_progress_exp472_antigravity",
    "publish_manifest_exp472_antigravity", "publish_score_shard_exp472_antigravity",
    "score_shard_record_exp472_antigravity", "status_exp472_antigravity",
    "verify_manifest_exp472_antigravity", "verify_score_shard_exp472_antigravity",
]
