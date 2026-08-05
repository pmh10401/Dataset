"""Deterministic, label-free publication and independent replay for Exp426."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .checkpoint_antigravity import (
    Exp426CheckpointStore,
    _canonical_object as _checkpoint_object,
    _context_payload,
    _fsync_directory,
    _record_filename,
    _write_new,
)
from .contracts_antigravity import Exp426DatasetRecord, Exp426Provenance
from .runtime_antigravity import _load_events, inspect_exp426


PUBLICATION_DIRECTORY = "publication_antigravity"
PREDICTION_FILE = "prediction_antigravity.jsonl"
SUMMARY_FILE = "dataset_summary_antigravity.jsonl"
IMPORT_FILE = "import_ledger_antigravity.jsonl"
ABSTENTION_FILE = "train_abstention_ledger_antigravity.jsonl"
BACKEND_FILE = "backend_decision_antigravity.json"
MANIFEST_FILE = "manifest_antigravity.json"
STATUS_FILE = "status_antigravity.json"
HASH_FILE = "artifact_sha256_antigravity.json"

_SEMANTIC_FILES = (
    PREDICTION_FILE, SUMMARY_FILE, IMPORT_FILE, ABSTENTION_FILE, BACKEND_FILE,
)
_PUBLICATION_FILES = frozenset(
    (*_SEMANTIC_FILES, MANIFEST_FILE, STATUS_FILE, HASH_FILE)
)
_ALLOWED_GOVERNANCE_KEYS = frozenset({
    "contains_test_labels", "test_labels_opened",
})
_FORBIDDEN_KEY_TOKENS = (
    "label", "target", "groundtruth", "confusion", "precision", "recall",
    "f1", "anomalylocation", "familytest",
)


@dataclass(frozen=True)
class _PublicationExpectations:
    total: int = 1117
    imported: int = 983
    verified_train_abstentions: int = 134

    def __post_init__(self) -> None:
        values = (self.total, self.imported, self.verified_train_abstentions)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("publication expectations are invalid")
        if self.total <= 0 or self.imported + self.verified_train_abstentions != self.total:
            raise ValueError("publication expectations do not conserve population")


@dataclass(frozen=True)
class Exp426PublicationResult:
    publication: Path
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "publication", Path(self.publication))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _json_raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_raw(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _read_canonical_json(
    path: Path, label: str, *, trailing_newline: bool = True,
) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    expected = _json_raw(value) if trailing_newline else canonical_json_bytes(value)
    if not isinstance(value, dict) or expected != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _reject_forbidden_keys(value: object, *, artifact: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("publication JSON key is invalid")
            normalized = "".join(
                character for character in key.lower() if character.isalnum()
            )
            if key in _ALLOWED_GOVERNANCE_KEYS:
                if artifact not in {MANIFEST_FILE, STATUS_FILE} or child is not False:
                    raise ValueError("label-governance flags are invalid")
            elif any(token in normalized for token in _FORBIDDEN_KEY_TOKENS):
                raise ValueError(f"forbidden publication key: {key}")
            _reject_forbidden_keys(child, artifact=artifact)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child, artifact=artifact)


def _bound_checkpoint_records(
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
) -> tuple[tuple[Exp426DatasetRecord, ...], Mapping[str, str]]:
    checkpoint = Path(checkpoint)
    context_payload = _context_payload(checkpoint_context, provenance)
    context_raw = canonical_json_bytes(context_payload)
    bound = provenance.with_context_sha256(_digest(context_raw))
    actual_provenance = _checkpoint_object(
        (checkpoint / "checkpoint_provenance.json").read_bytes(),
        "checkpoint provenance",
    )
    actual_context = _checkpoint_object(
        (checkpoint / "run_context.json").read_bytes(), "checkpoint context",
    )
    if actual_provenance != bound.to_payload() or actual_context != context_payload:
        raise ValueError("checkpoint provenance or context is not the frozen input")
    inspection = Exp426CheckpointStore.inspect_read_only(
        checkpoint, expected_run_identity=bound.run_identity,
    )
    if (
        inspection.total != len(bound.expected_datasets)
        or set(inspection.record_digests) != set(bound.expected_datasets)
    ):
        raise ValueError("checkpoint population is incomplete")
    test_sizes = context_payload["test_sizes"]
    lanes = {tuple(value) for value in context_payload["lane_coordinates"]}
    records: list[Exp426DatasetRecord] = []
    for dataset in bound.expected_datasets:
        path = checkpoint / "datasets" / _record_filename(dataset)
        raw = path.read_bytes()
        if _digest(raw) != inspection.record_digests[dataset]:
            raise ValueError("checkpoint record SHA256 does not match")
        record = Exp426DatasetRecord.from_payload(
            _checkpoint_object(raw, "checkpoint dataset record")
        )
        if record.dataset != dataset:
            raise ValueError("checkpoint record dataset does not match")
        record.validate_against(
            test_sizes, lambda name, index: (name, index) in lanes,
        )
        records.append(record)
    return tuple(records), inspection.record_digests


def _validate_runtime_evidence(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    record_digests: Mapping[str, str],
) -> Mapping[str, object]:
    runtime = inspect_exp426(
        output, checkpoint, expected_run_identity=provenance.run_identity,
    )
    progress = runtime.progress
    counts = progress.get("counts")
    if (
        progress.get("state") != "complete_review_only"
        or progress.get("current_dataset") is not None
        or progress.get("backend") != "cpu"
        or progress.get("workers") != 12
        or not isinstance(counts, dict)
        or counts.get("completed") != len(record_digests)
        or counts.get("error") != 0
        or counts.get("failed") != 0
    ):
        raise ValueError("runtime completion evidence is invalid")
    events = _load_events(Path(output))
    completed = {
        item["dataset"]: item
        for item in events if item.get("kind") == "dataset_complete"
    }
    if set(completed) != set(record_digests):
        raise ValueError("runtime completion event population does not match checkpoint")
    for dataset, item in completed.items():
        if item.get("record_sha256") != record_digests[dataset]:
            raise ValueError("runtime completion event SHA256 does not match checkpoint")
    pid = _read_canonical_json(
        Path(output) / "run.pid", "runtime PID", trailing_newline=False,
    )
    if pid.get("run_identity") != provenance.run_identity or pid.get("active") is not False:
        raise ValueError("runtime PID evidence is still active or mismatched")
    return progress


def _count_records(
    records: tuple[Exp426DatasetRecord, ...], expectations: _PublicationExpectations,
) -> tuple[int, int]:
    imported = sum(record.origin == "exp424_import" for record in records)
    verified = sum(record.origin == "verified_train_abstention" for record in records)
    if (
        len(records) != expectations.total
        or imported != expectations.imported
        or verified != expectations.verified_train_abstentions
        or imported + verified != len(records)
    ):
        raise ValueError("publication population does not match frozen expectations")
    return imported, verified


def _build_publication_bytes(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
    expectations: _PublicationExpectations,
) -> dict[str, bytes]:
    records, record_digests = _bound_checkpoint_records(
        checkpoint, provenance, checkpoint_context,
    )
    progress = _validate_runtime_evidence(
        output, checkpoint, provenance, record_digests,
    )
    imported, verified = _count_records(records, expectations)

    prediction_rows = [
        {"dataset": dataset, "instance_index": index}
        for record in records for dataset, index in record.candidate_coordinates
    ]
    if prediction_rows != sorted(
        prediction_rows, key=lambda item: (item["dataset"], item["instance_index"]),
    ) or len({(item["dataset"], item["instance_index"]) for item in prediction_rows}) != len(prediction_rows):
        raise ValueError("prediction coordinates are not sorted and unique")
    summary_rows = [
        {
            "dataset": record.dataset,
            "origin": record.origin,
            "status": record.status,
            "candidate_count": len(record.candidate_coordinates),
            "reason": record.reason,
            "record_sha256": record_digests[record.dataset],
        }
        for record in records
    ]
    import_rows = [
        {
            "dataset": record.dataset,
            "source_record_sha256": record.source_record_sha256,
            "source_result_sha256": record.source_result_sha256,
            "record_sha256": record_digests[record.dataset],
        }
        for record in records if record.origin == "exp424_import"
    ]
    abstention_rows = [
        {
            "dataset": record.dataset,
            "source_record_sha256": record.source_record_sha256,
            "source_result_sha256": record.source_result_sha256,
            "train_identity_sha256": record.train_identity_sha256,
            "eligibility_sha256": record.eligibility_sha256,
            "reason": record.reason,
            "record_sha256": record_digests[record.dataset],
        }
        for record in records if record.origin == "verified_train_abstention"
    ]
    backend = {
        "schema": "exp426.backend-decision.v1",
        "selected_backend": "cpu",
        "workers": 12,
        "inner_threads": 1,
        "prefetch": 24,
        "mlx": {"assessed": True, "adopted": False, "reason": "no_eligible_compute_stage"},
        "mps": {"assessed": True, "adopted": False, "reason": "no_model_or_tensor_score_stage"},
    }
    semantic = {
        PREDICTION_FILE: _jsonl_raw(prediction_rows),
        SUMMARY_FILE: _jsonl_raw(summary_rows),
        IMPORT_FILE: _jsonl_raw(import_rows),
        ABSTENTION_FILE: _jsonl_raw(abstention_rows),
        BACKEND_FILE: _json_raw(backend),
    }
    semantic_hashes = {
        name: _digest(raw) for name, raw in sorted(semantic.items())
    }
    record_set_sha256 = _digest(canonical_json_bytes(dict(record_digests)))
    manifest = {
        "schema": "exp426.publication-manifest.v1",
        "run_identity": provenance.run_identity,
        "git_commit": provenance.git_commit,
        "database_sha256": provenance.database_sha256,
        "source_root_sha256": provenance.source_root_sha256,
        "source_progress_sha256": provenance.source_progress_sha256,
        "source_record_set_sha256": provenance.source_record_set_sha256,
        "checkpoint_record_set_sha256": record_set_sha256,
        "lane_artifact_sha256": dict(provenance.lane_artifact_sha256),
        "fold_contract_sha256": provenance.fold_contract_sha256,
        "prediction_sha256": semantic_hashes[PREDICTION_FILE],
        "completed_datasets": len(records),
        "imported_datasets": imported,
        "verified_train_abstentions": verified,
        "failed_datasets": 0,
        "candidate_count": len(prediction_rows),
        "workers": 12,
        "review_only": True,
        "operational_alert_eligible": False,
        "contains_test_labels": False,
        "test_labels_opened": False,
        "artifact_sha256": semantic_hashes,
    }
    _reject_forbidden_keys(manifest, artifact=MANIFEST_FILE)
    manifest_raw = _json_raw(manifest)
    status = {
        "schema": "exp426.publication-status.v1",
        "run_identity": provenance.run_identity,
        "state": "complete_review_only",
        "completed_datasets": len(records),
        "imported_datasets": imported,
        "verified_train_abstentions": verified,
        "failed_datasets": 0,
        "workers": 12,
        "backend": "cpu",
        "prediction_sha256": semantic_hashes[PREDICTION_FILE],
        "manifest_sha256": _digest(manifest_raw),
        "runtime_progress_sha256": _digest((Path(output) / "progress.json").read_bytes()),
        "review_only": True,
        "operational_alert_eligible": False,
        "contains_test_labels": False,
        "test_labels_opened": False,
    }
    _reject_forbidden_keys(status, artifact=STATUS_FILE)
    status_raw = _json_raw(status)
    before_hash_map = {
        **semantic,
        MANIFEST_FILE: manifest_raw,
        STATUS_FILE: status_raw,
    }
    artifact_map = {
        "schema": "exp426.artifact-sha256.v1",
        "artifact_sha256": {
            name: _digest(raw) for name, raw in sorted(before_hash_map.items())
        },
        "self_excluded": HASH_FILE,
    }
    payloads = {**before_hash_map, HASH_FILE: _json_raw(artifact_map)}
    for name, raw in payloads.items():
        if name.endswith(".json"):
            value = json.loads(raw)
            _reject_forbidden_keys(value, artifact=name)
        else:
            for line in raw.splitlines():
                _reject_forbidden_keys(json.loads(line), artifact=name)
    if set(payloads) != _PUBLICATION_FILES or progress.get("state") != "complete_review_only":
        raise ValueError("publication artifact inventory is invalid")
    return payloads


def _validated_publication_root(output: Path) -> Path:
    publication = Path(output) / PUBLICATION_DIRECTORY
    if not publication.is_dir() or publication.is_symlink():
        raise ValueError("publication directory is absent or invalid")
    entries = tuple(publication.iterdir())
    if {path.name for path in entries} != _PUBLICATION_FILES:
        raise ValueError("publication inventory is invalid")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("publication inventory contains an invalid file")
    return publication


def verify_exp426_publication(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
    *,
    _expectations: _PublicationExpectations = _PublicationExpectations(),
) -> Exp426PublicationResult:
    if not isinstance(provenance, Exp426Provenance) or not isinstance(
        _expectations, _PublicationExpectations
    ):
        raise ValueError("publication verification inputs are invalid")
    publication = _validated_publication_root(Path(output))
    expected = _build_publication_bytes(
        Path(output), Path(checkpoint), provenance, checkpoint_context, _expectations,
    )
    artifact_map = _read_canonical_json(
        publication / HASH_FILE, "publication artifact SHA256 map",
    )
    hashes = artifact_map.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != _PUBLICATION_FILES - {HASH_FILE}:
        raise ValueError("publication artifact SHA256 inventory is invalid")
    for name in sorted(_PUBLICATION_FILES):
        actual = (publication / name).read_bytes()
        if actual != expected[name]:
            raise ValueError(f"publication {name} does not match checkpoint replay")
        if name != HASH_FILE and hashes.get(name) != _digest(actual):
            raise ValueError(f"publication {name} SHA256 does not match")
    manifest = _read_canonical_json(
        publication / MANIFEST_FILE, "publication manifest",
    )
    return Exp426PublicationResult(publication, manifest)


def publish_exp426(
    output: Path,
    checkpoint: Path,
    provenance: Exp426Provenance,
    checkpoint_context: object,
    *,
    _expectations: _PublicationExpectations = _PublicationExpectations(),
) -> Exp426PublicationResult:
    output = Path(output)
    publication = output / PUBLICATION_DIRECTORY
    if publication.exists() or publication.is_symlink():
        return verify_exp426_publication(
            output, checkpoint, provenance, checkpoint_context,
            _expectations=_expectations,
        )
    if not output.is_dir() or output.is_symlink():
        raise ValueError("runtime output root is absent or invalid")
    payloads = _build_publication_bytes(
        output, Path(checkpoint), provenance, checkpoint_context, _expectations,
    )
    staging = output / f".publication_antigravity.{uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    _fsync_directory(output)
    published = False
    try:
        for name in sorted(payloads):
            _write_new(staging / name, payloads[name])
        _fsync_directory(staging)
        if publication.exists() or publication.is_symlink():
            raise ValueError("publication appeared during atomic staging")
        os.rename(staging, publication)
        _fsync_directory(output)
        published = True
    finally:
        if not published and staging.is_dir() and not staging.is_symlink():
            for name in sorted(payloads):
                path = staging / name
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            staging.rmdir()
            _fsync_directory(output)
    return verify_exp426_publication(
        output, checkpoint, provenance, checkpoint_context,
        _expectations=_expectations,
    )
