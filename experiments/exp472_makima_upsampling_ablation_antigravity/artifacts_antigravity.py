"""Durable, create-only condition-E score artifacts for Exp472."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
from time import monotonic

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


@dataclass(frozen=True, slots=True)
class Exp472CatalogEntryAntigravity:
    """Typed, immutable consumed boundary for one frozen score coordinate."""

    dataset: str
    test_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Exp472 catalog entry dataset is invalid")
        try:
            identity = json.loads(
                canonical_json_bytes_exp472_antigravity(self.test_identity).decode("utf-8")
            )
        except ValueError as error:
            raise ValueError("Exp472 catalog entry test_identity is invalid") from error
        if not isinstance(identity, dict) or identity.get("dataset") != self.dataset:
            raise ValueError("Exp472 catalog entry test_identity is invalid")
        object.__setattr__(self, "test_identity", identity)

    def artifact_record(self) -> dict[str, object]:
        return {"dataset": self.dataset, "test_identity": dict(self.test_identity)}


def _catalog_entry_antigravity(entry: object) -> Exp472CatalogEntryAntigravity:
    if not isinstance(entry, Exp472CatalogEntryAntigravity):
        raise ValueError("Exp472 catalog entry requires dataset and test_identity")
    return entry


def _dataset_from_entry_antigravity(entry: object) -> str:
    typed = _catalog_entry_antigravity(entry)
    if type(typed.dataset) is not str or not typed.dataset:
        raise ValueError("Exp472 catalog entry dataset is invalid")
    return typed.dataset


def _entry_identity_antigravity(entry: object) -> Mapping[str, object]:
    return _catalog_entry_antigravity(entry).test_identity


def catalog_sha256_exp472_antigravity(catalog: Sequence[object]) -> str:
    """Digest the exact ordered catalog coordinate and TEST-identity sequence."""
    entries = _catalog_entries_antigravity(catalog)
    return sha256(
        canonical_json_bytes_exp472_antigravity(
            [entry.artifact_record() for entry in entries]
        )
    ).hexdigest()


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


_TRACE_ORDER_ANTIGRAVITY = (
    ("hard_sources", "TRAIN"), ("hard_sources", "TEST"),
    ("minirocket", "TRAIN"), ("minirocket", "TEST"),
    ("legacy_exp40", "TRAIN"), ("legacy_exp40", "TEST"),
    ("legacy_route", "TRAIN"), ("legacy_route", "TEST"),
)


def _diagnostic_error_antigravity() -> None:
    raise ValueError("Exp472 score shard diagnostic differs")


def _is_count_antigravity(value: object) -> bool:
    return type(value) is int and value >= 0


def _validate_digest_antigravity(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_evidence_antigravity(
    value: object, row_count: int, target_len: int, *, output: bool,
) -> tuple[int, ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "dtype", "order", "shape", "row_boundaries", "row_bytes_sha256"
    }:
        _diagnostic_error_antigravity()
    shape, boundaries, digests = (
        value["shape"], value["row_boundaries"], value["row_bytes_sha256"]
    )
    if (
        value["dtype"] != "<f4" or value["order"] != "C" or not isinstance(shape, (list, tuple))
        or list(shape) != ([row_count, target_len] if output else [row_count])
        or not isinstance(boundaries, (list, tuple))
        or not isinstance(digests, (list, tuple))
        or len(boundaries) != row_count + 1 or len(digests) != row_count
        or boundaries[0] != 0
        or any(type(offset) is not int or offset < 0 for offset in boundaries)
        or any(right <= left for left, right in zip(boundaries, boundaries[1:]))
        or any(not _validate_digest_antigravity(digest) for digest in digests)
        or (output and tuple(boundaries) != tuple(index * target_len for index in range(row_count + 1)))
    ):
        _diagnostic_error_antigravity()
    return tuple(right - left for left, right in zip(boundaries, boundaries[1:]))


def _validate_diagnostic_antigravity(
    value: object, *, dataset: str | None = None, upsampled_rows: int | None = None,
) -> None:
    """Authenticate the complete sealed v3 observer payload before reuse."""
    required = {
        "schema", "dataset", "trace_state", "alignment_status", "invocation_count",
        "row_invocation_count", "makima_row_invocation_count", "upsampled_rows",
        "input_identity_sha256", "result_sha256",
        "invocations", "invocation_order_sha256", "invocation_aggregate_sha256",
        "diagnostic_sha256",
    }
    if (
        not isinstance(value, Mapping) or set(value) != required
        or value.get("schema") != "exp472-shape-diagnostic-antigravity.v3"
        or value.get("trace_state") != "sealed" or value.get("alignment_status") != "completed"
        or type(value.get("dataset")) is not str or not value["dataset"]
        or (dataset is not None and value["dataset"] != dataset)
        or not isinstance(value.get("invocations"), (list, tuple))
        or any(not _is_count_antigravity(value.get(name)) for name in (
            "invocation_count", "row_invocation_count", "makima_row_invocation_count", "upsampled_rows",
        ))
        or not _validate_digest_antigravity(value.get("diagnostic_sha256"))
        or not _validate_digest_antigravity(value.get("input_identity_sha256"))
        or not _validate_digest_antigravity(value.get("result_sha256"))
        or not _validate_digest_antigravity(value.get("invocation_order_sha256"))
        or not _validate_digest_antigravity(value.get("invocation_aggregate_sha256"))
        or (upsampled_rows is not None and value["upsampled_rows"] != upsampled_rows)
    ):
        _diagnostic_error_antigravity()
    bare = {key: item for key, item in value.items() if key != "diagnostic_sha256"}
    if value["diagnostic_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest():
        _diagnostic_error_antigravity()
    records = value["invocations"]
    slots: set[tuple[str, int]] = set()
    expected_order: list[tuple[str, str]] = []
    row_total = makima_total = 0
    record_required = {
        "ordinal", "source_tag", "phase_tag", "target_len", "row_count",
        "input_evidence", "input_sha256", "output_evidence", "output_sha256",
        "singleton_rows", "equal_rows", "polyphase_downsample_rows", "makima_rows",
        "makima_row_indices", "overshoot", "undershoot", "source_extrema",
        "output_extrema", "added_extrema", "invocation_sha256",
    }
    previous = -1
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != record_required:
            _diagnostic_error_antigravity()
        target_len, row_count = record["target_len"], record["row_count"]
        pair = (record["phase_tag"], record["source_tag"])
        if (
            not _is_count_antigravity(record["ordinal"])
            or record["ordinal"] != ordinal
            or not _is_count_antigravity(target_len) or target_len == 0
            or not _is_count_antigravity(row_count) or pair not in _TRACE_ORDER_ANTIGRAVITY
            or _TRACE_ORDER_ANTIGRAVITY.index(pair) <= previous
            or not _validate_digest_antigravity(record["input_sha256"])
            or not _validate_digest_antigravity(record["output_sha256"])
            or not _validate_digest_antigravity(record["invocation_sha256"])
        ):
            _diagnostic_error_antigravity()
        previous = _TRACE_ORDER_ANTIGRAVITY.index(pair)
        input_boundaries = _validate_evidence_antigravity(
            record["input_evidence"], row_count, target_len, output=False
        )
        _validate_evidence_antigravity(record["output_evidence"], row_count, target_len, output=True)
        if (
            record["input_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(record["input_evidence"])).hexdigest()
            or record["output_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(record["output_evidence"])).hexdigest()
            or record["invocation_sha256"] != sha256(canonical_json_bytes_exp472_antigravity({key: item for key, item in record.items() if key != "invocation_sha256"})).hexdigest()
        ):
            _diagnostic_error_antigravity()
        route_counts = ("singleton_rows", "equal_rows", "polyphase_downsample_rows", "makima_rows")
        if any(not _is_count_antigravity(record[name]) for name in route_counts):
            _diagnostic_error_antigravity()
        expected_routes = (
            sum(length == 1 for length in input_boundaries),
            sum(length == target_len for length in input_boundaries),
            sum(length > target_len for length in input_boundaries),
            sum(1 < length < target_len for length in input_boundaries),
        )
        if tuple(record[name] for name in route_counts) != expected_routes or sum(expected_routes) != row_count:
            _diagnostic_error_antigravity()
        makima_indices = record["makima_row_indices"]
        if (
            not isinstance(makima_indices, (list, tuple))
            or list(makima_indices) != [index for index, length in enumerate(input_boundaries) if 1 < length < target_len]
            or any(type(item) is not int for item in makima_indices)
        ):
            _diagnostic_error_antigravity()
        for name in ("overshoot", "undershoot"):
            item = record[name]
            if type(item) not in (int, float) or not math.isfinite(float(item)) or item < 0:
                _diagnostic_error_antigravity()
        for name in ("source_extrema", "output_extrema", "added_extrema"):
            if not _is_count_antigravity(record[name]):
                _diagnostic_error_antigravity()
        slots.update((record["source_tag"], index) for index in makima_indices)
        expected_order.append(pair)
        row_total += row_count
        makima_total += record["makima_rows"]
    aggregate = {
        "invocation_count": len(records), "row_invocation_count": row_total,
        "makima_row_invocation_count": makima_total, "upsampled_rows": len(slots),
    }
    if (
        any(value[name] != item for name, item in aggregate.items())
        or value["invocation_order_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(expected_order)).hexdigest()
        or value["invocation_aggregate_sha256"] != sha256(canonical_json_bytes_exp472_antigravity(aggregate)).hexdigest()
    ):
        _diagnostic_error_antigravity()


def validate_diagnostic_exp472_antigravity(
    value: object, *, dataset: str | None = None, upsampled_rows: int | None = None,
) -> None:
    """Validate a primitive v3 diagnostic before IPC reconstructs its typed shard."""
    _validate_diagnostic_antigravity(
        value, dataset=dataset, upsampled_rows=upsampled_rows,
    )


def score_shard_record_exp472_antigravity(
    context: Exp472RunContextAntigravity, shard: Exp472ScoreShardAntigravity
) -> dict[str, object]:
    """Serialize one digest-bound, finite condition-E score result."""
    if not isinstance(context, Exp472RunContextAntigravity) or not isinstance(shard, Exp472ScoreShardAntigravity):
        raise ValueError("Exp472 score shard inputs are invalid")
    _validate_diagnostic_antigravity(
        shard.diagnostic, dataset=shard.dataset, upsampled_rows=shard.upsampled_rows
    )
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
    if canonical_json_bytes_exp472_antigravity(record["test_identity"]) != canonical_json_bytes_exp472_antigravity(entry_identity):
        raise ValueError("Exp472 score shard TEST identity differs")
    if record["score_dtype"] != "float64" or not isinstance(record["scores"], list):
        raise ValueError("Exp472 score shard score encoding differs")
    _validate_diagnostic_antigravity(
        record["diagnostic"], dataset=record["dataset"], upsampled_rows=record["upsampled_rows"]
    )
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
    if canonical_json_bytes_exp472_antigravity(
        score_shard_record_exp472_antigravity(context, reconstructed)
    ) != canonical_json_bytes_exp472_antigravity(dict(record)):
        raise ValueError("Exp472 score shard canonical payload differs")
    return dict(record)


def publish_score_shard_exp472_antigravity(
    root: Path, context: Exp472RunContextAntigravity, entry: object, shard: Exp472ScoreShardAntigravity
) -> Path:
    """Durably create one immutable score shard, or reuse identical bytes only."""
    if _dataset_from_entry_antigravity(entry) != shard.dataset:
        raise ValueError("Exp472 score shard dataset identity differs")
    entry_identity = _entry_identity_antigravity(entry)
    if canonical_json_bytes_exp472_antigravity(entry_identity) != canonical_json_bytes_exp472_antigravity(shard.test_identity):
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


def _catalog_entries_antigravity(catalog: Sequence[object]) -> tuple[Exp472CatalogEntryAntigravity, ...]:
    entries = tuple(_catalog_entry_antigravity(entry) for entry in catalog)
    datasets = tuple(entry.dataset for entry in entries)
    if not entries or len(set(datasets)) != len(datasets):
        raise ValueError("Exp472 catalog coverage is invalid")
    return entries


def _validated_catalog_antigravity(
    context: Exp472RunContextAntigravity, catalog: Sequence[object]
) -> tuple[Exp472CatalogEntryAntigravity, ...]:
    entries = _catalog_entries_antigravity(catalog)
    if catalog_sha256_exp472_antigravity(entries) != context.catalog_sha256:
        raise ValueError("Exp472 catalog digest differs from run context")
    return entries


def _manifest_record_antigravity(
    context: Exp472RunContextAntigravity, entries: tuple[Exp472CatalogEntryAntigravity, ...], shards: Sequence[Mapping[str, object]]
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
    record: Mapping[str, object], root: Path, context: Exp472RunContextAntigravity, entries: tuple[Exp472CatalogEntryAntigravity, ...]
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
    entries = _validated_catalog_antigravity(context, catalog)
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
    entries = _validated_catalog_antigravity(context, catalog)
    path = _safe_root_antigravity(Path(root), create=False) / _MANIFEST_FILENAME_ANTIGRAVITY
    return _verify_manifest_record_antigravity(_read_canonical_record_antigravity(path), Path(root), context, entries)


def append_progress_exp472_antigravity(
    root: Path, *, completed: int, failed: int, remaining: int, elapsed_seconds: float,
    dataset: str, wave: int, status: str, observed_monotonic: float | None = None,
) -> None:
    """Durably append one immutable lifecycle observation."""
    if status not in {"completed", "failed"} or wave not in {1, 2} or type(dataset) is not str or not dataset:
        raise ValueError("Exp472 progress record is invalid")
    observed = monotonic() if observed_monotonic is None else observed_monotonic
    if (
        any(type(value) is not int or value < 0 for value in (completed, failed, remaining))
        or type(elapsed_seconds) not in {int, float} or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
        or type(observed) not in {int, float} or not math.isfinite(observed) or observed < elapsed_seconds
    ):
        raise ValueError("Exp472 progress counters are invalid")
    safe_root = _safe_root_antigravity(Path(root), create=True)
    _append_jsonl_antigravity(safe_root / _PROGRESS_FILENAME_ANTIGRAVITY, {
        "schema": _PROGRESS_SCHEMA_ANTIGRAVITY, "completed": completed, "failed": failed,
        "remaining": remaining, "elapsed_seconds": float(elapsed_seconds),
        "observed_monotonic": float(observed), "dataset": dataset, "wave": wave,
        "status": status,
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


def _regular_file_exists_antigravity(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode)


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
    now = monotonic() if now_monotonic is None else now_monotonic
    if type(now) not in {int, float} or not math.isfinite(now) or now < 0:
        raise ValueError("Exp472 status monotonic clock is invalid")
    elapsed = 0.0
    if latest:
        observed = latest.get("observed_monotonic")
        elapsed_recorded = latest.get("elapsed_seconds")
        if (
            type(observed) not in {int, float} or type(elapsed_recorded) not in {int, float}
            or not math.isfinite(observed) or not math.isfinite(elapsed_recorded)
            or elapsed_recorded < 0 or observed < elapsed_recorded
        ):
            raise ValueError("Exp472 progress monotonic data is invalid")
        started = float(observed) - float(elapsed_recorded)
        elapsed = max(float(elapsed_recorded), float(now) - started)
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = remaining / rate if rate > 0 else None
    return {
        "schema": "exp472-score-status-antigravity.v1", "completed": completed, "failed": failed,
        "remaining": remaining, "total": total, "rate_per_second": rate, "eta_seconds": eta,
        "manifest_complete": _regular_file_exists_antigravity(safe_root / _MANIFEST_FILENAME_ANTIGRAVITY),
        "latest_progress": latest,
    }


__all__ = [
    "Exp472CatalogEntryAntigravity",
    "append_failure_exp472_antigravity", "append_progress_exp472_antigravity",
    "catalog_sha256_exp472_antigravity",
    "publish_manifest_exp472_antigravity", "publish_score_shard_exp472_antigravity",
    "score_shard_record_exp472_antigravity", "status_exp472_antigravity",
    "validate_diagnostic_exp472_antigravity",
    "verify_manifest_exp472_antigravity", "verify_score_shard_exp472_antigravity",
]
