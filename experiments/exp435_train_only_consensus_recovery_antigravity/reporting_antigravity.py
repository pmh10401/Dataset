"""Independent, label-free publication and replay for Exp435.

The runtime checkpoint is the only scientific source for this module.  It is
read through the checkpoint's read-only API and replayed into a private
staging directory before a no-clobber rename.  No database, worker, or
evaluation module is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import ctypes
import errno
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Iterable, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .checkpoint_antigravity import Exp435CheckpointStore
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    BackendDecision,
    Exp435DatasetResult,
)
from .progress_antigravity import (
    _assert_label_free,
    _canonical_checkpoint,
    _validate_binding,
    _validate_events,
    _validate_output_inventory,
)
from .scoring_antigravity import CandidateBudgetExceeded


PUBLICATION_DIRECTORY = "publication_antigravity"
PREDICTION_FILE = "prediction_antigravity.jsonl"
SUMMARY_FILE = "dataset_summary_antigravity.jsonl"
EVIDENCE_FILE = "score_evidence_antigravity.jsonl"
ABSTENTION_FILE = "abstention_ledger_antigravity.jsonl"
BACKEND_FILE = "backend_decision_antigravity.json"
MANIFEST_FILE = "manifest_antigravity.json"
STATUS_FILE = "status_antigravity.json"
HASH_FILE = "artifact_sha256_antigravity.json"

_SEMANTIC_FILES = (
    PREDICTION_FILE,
    SUMMARY_FILE,
    EVIDENCE_FILE,
    ABSTENTION_FILE,
    BACKEND_FILE,
)
_PUBLICATION_FILES = frozenset((*_SEMANTIC_FILES, MANIFEST_FILE, STATUS_FILE, HASH_FILE))
_GOVERNANCE_KEYS = frozenset({
    "contains_test_labels",
    "test_labels_opened",
    "label_access_count",
    "operational_alert_eligible",
    "review_only",
})
_FORBIDDEN_KEY_TOKENS = (
    "target",
    "groundtruth",
    "confusion",
    "precision",
    "recall",
    "f1",
    "anomalylocation",
    "familytest",
)


@dataclass(frozen=True)
class _PublicationExpectations:
    """Frozen population expectations, with a tiny override for unit tests."""

    total: int = FROZEN_CONFIG.expected_datasets

    def __post_init__(self) -> None:
        if type(self.total) is not int or self.total <= 0:
            raise ValueError("publication total is invalid")
        if self.total == FROZEN_CONFIG.expected_datasets:
            return
        # Small expectations are useful for deterministic publication tests,
        # but production callers must use the exact 1,117 population.
        if self.total > FROZEN_CONFIG.expected_datasets:
            raise ValueError("publication total exceeds the frozen population")


PublicationExpectations = _PublicationExpectations


@dataclass(frozen=True)
class FrozenPublication:
    """A verified publication root and its immutable manifest."""

    publication: Path
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        publication = Path(self.publication)
        if publication.is_symlink() or not publication.is_dir():
            raise ValueError("publication root is absent or invalid")
        object.__setattr__(self, "publication", publication)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def prediction_sha256(self) -> str:
        value = self.manifest.get("prediction_sha256")
        if not isinstance(value, str):
            raise ValueError("publication prediction SHA256 is absent")
        return value

    @property
    def manifest_sha256(self) -> str:
        return _digest((self.publication / MANIFEST_FILE).read_bytes())

    @property
    def artifact_sha256(self) -> str:
        return _digest((self.publication / HASH_FILE).read_bytes())


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _json_raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_raw(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _read_json(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or raw not in {canonical_json_bytes(value), _json_raw(value)}:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_rename(source: Path, destination: Path) -> None:
    """Atomically move a directory without replacing a destination.

    ``os.rename`` replaces an existing directory on Darwin.  That behavior is
    unsafe for a publication that must never clobber a concurrent destination,
    so use the kernel's exclusive rename primitive and fail closed elsewhere.
    """

    source = Path(source)
    destination = Path(destination)
    if sys.platform != "darwin":
        raise RuntimeError("exclusive directory rename is unsupported on this platform")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
    except (AttributeError, OSError) as error:
        raise RuntimeError("Darwin renamex_np is unavailable") from error
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    result = renamex_np(
        os.fsencode(source),
        os.fsencode(destination),
        ctypes.c_uint(0x00000004),  # RENAME_EXCL from <sys/stdio.h>
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _reject_forbidden_keys(value: object, *, artifact: str) -> None:
    """Reject semantic fields that could smuggle TEST labels into artifacts."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("publication JSON key is invalid")
            normalized = "".join(character for character in key.lower() if character.isalnum())
            if normalized in {"contains test labels", "test labels opened"}:
                raise ValueError("publication governance key is malformed")
            if key in _GOVERNANCE_KEYS:
                if artifact not in {MANIFEST_FILE, STATUS_FILE}:
                    raise ValueError("governance keys are restricted to manifest/status")
                if key == "label_access_count" and child != 0:
                    raise ValueError("label access count must be zero")
                if key in {"contains_test_labels", "test_labels_opened", "operational_alert_eligible"} and child is not False:
                    raise ValueError("label governance flags must be false")
                if key == "review_only" and child is not True:
                    raise ValueError("review_only must be true")
            elif "label" in normalized or any(token in normalized for token in _FORBIDDEN_KEY_TOKENS):
                raise ValueError(f"forbidden publication key: {key}")
            _reject_forbidden_keys(child, artifact=artifact)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_forbidden_keys(child, artifact=artifact)


def _store_root(store: object) -> Path:
    root = getattr(store, "root", None)
    if root is None:
        raise ValueError("checkpoint store does not expose a root")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root is absent or invalid")
    return root


def _open_store(value: object) -> Exp435CheckpointStore:
    if isinstance(value, Exp435CheckpointStore):
        return value
    if isinstance(value, (str, Path)):
        return Exp435CheckpointStore.open_read_only(Path(value))
    # Test doubles intentionally use the same narrow public surface.  A real
    # store is required for production publication, while this adapter keeps
    # replay tests independent of filesystem internals.
    required = ("inspect", "completed_digests", "load_record", "datasets", "context")
    if all(hasattr(value, name) for name in required):
        return value  # type: ignore[return-value]
    raise ValueError("checkpoint store is invalid")


def _backend_from_context(context: Mapping[str, object]) -> BackendDecision:
    payload = context.get("backend_decision")
    if isinstance(payload, BackendDecision):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint backend decision is absent")
    raw = dict(payload)
    evidence_raw = raw.get("batch_evidence", ())
    timings_raw = raw.get("timings", ())
    if isinstance(evidence_raw, list):
        from .contracts_antigravity import BackendBatchEvidence

        raw["batch_evidence"] = tuple(BackendBatchEvidence(**dict(item)) for item in evidence_raw)
    else:
        raw["batch_evidence"] = tuple(evidence_raw)
    raw["timings"] = tuple(tuple(item) for item in timings_raw) if isinstance(timings_raw, list) else tuple(timings_raw)
    keys = (
        "backend", "adopted", "smoke_seed", "finite_outputs", "candidate_identity_match",
        "parity_passed", "normalized_error_max", "absolute_error_max", "speedup",
        "batch_size", "selected_batch_size", "cpu_median_seconds", "mlx_median_seconds",
        "mlx_version", "mlx_metal_version", "device", "device_info", "metal_available",
        "timings", "batch_evidence", "reason",
    )
    try:
        return BackendDecision(**{key: raw[key] for key in keys if key in raw})
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("checkpoint backend decision is invalid") from error


def _context_hash(context: Mapping[str, object]) -> str:
    return _digest(canonical_json_bytes(dict(context)))


def _input_hashes(context: Mapping[str, object]) -> dict[str, str]:
    database = context.get("database_sha256")
    if not isinstance(database, str):
        raise ValueError("checkpoint database hash is absent")
    values: dict[str, str] = {"database": database}
    lanes = context.get("lanes")
    if isinstance(lanes, Mapping):
        for name in (
            "exp291_prediction_sha256", "exp291_manifest_sha256",
            "exp292_prediction_sha256", "exp292_manifest_sha256",
        ):
            value = lanes.get(name)
            if isinstance(value, str):
                values[name] = value
    # Some test contexts expose lane hashes directly.  Keep the same names
    # and never invent a digest for absent evidence.
    for name in (
        "exp291_prediction_sha256", "exp291_manifest_sha256",
        "exp292_prediction_sha256", "exp292_manifest_sha256",
    ):
        value = context.get(name)
        if isinstance(value, str):
            values[name] = value
    return dict(sorted(values.items()))


def _record_data(
    store: object,
    expectations: _PublicationExpectations,
) -> tuple[tuple[Exp435DatasetResult, ...], Mapping[str, str], Mapping[str, object]]:
    inspection = store.inspect()
    datasets = tuple(getattr(store, "datasets"))
    if len(datasets) != expectations.total or tuple(sorted(datasets)) != datasets or len(set(datasets)) != len(datasets):
        raise ValueError("publication population does not match frozen expectations")
    if inspection.total != len(datasets) or inspection.completed != len(datasets):
        raise ValueError("publication requires a completed checkpoint population")
    digests = dict(store.completed_digests())
    if set(digests) != set(datasets):
        raise ValueError("checkpoint record population does not match context")
    records: list[Exp435DatasetResult] = []
    candidate_count = 0
    for dataset in datasets:
        result = store.load_record(dataset)
        if not isinstance(result, Exp435DatasetResult) or result.dataset != dataset:
            raise ValueError("checkpoint dataset record is invalid")
        if result.status == "failed":
            raise ValueError("publication cannot contain failed dataset records")
        candidate_count += result.candidate_count
        records.append(result)
    if candidate_count > FROZEN_CONFIG.global_candidate_budget:
        raise CandidateBudgetExceeded(
            observed=candidate_count,
            budget=FROZEN_CONFIG.global_candidate_budget,
        )
    context = getattr(store, "context")
    if not isinstance(context, Mapping):
        raise ValueError("checkpoint context is invalid")
    if expectations.total == FROZEN_CONFIG.expected_datasets:
        if context.get("database_sha256") != FROZEN_CONFIG.database_sha256:
            raise ValueError("checkpoint database hash does not match frozen input")
        if context.get("code_commit") is None:
            raise ValueError("checkpoint code commit is absent")
    return tuple(records), dict(sorted(digests.items())), dict(context)


def _validate_runtime(
    output: Path,
    store: object,
    records: tuple[Exp435DatasetResult, ...],
    record_digests: Mapping[str, str],
    context: Mapping[str, object],
) -> Mapping[str, object]:
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("runtime output inventory is invalid")
    runtime_names = {
        "progress_antigravity.json",
        "events_antigravity.jsonl",
        "parent_antigravity.log",
        "run_antigravity.pid",
    }
    names = {item.name for item in output.iterdir()}
    extras = names - runtime_names - {PUBLICATION_DIRECTORY}
    if extras:
        raise ValueError("runtime output inventory is invalid")
    for name in runtime_names:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime output inventory is invalid")
    publication = output / PUBLICATION_DIRECTORY
    if publication.exists() and (publication.is_symlink() or not publication.is_dir()):
        raise ValueError("publication directory is invalid")
    progress_path = output / "progress_antigravity.json"
    progress = _read_json(progress_path, "runtime progress")
    backend = _backend_from_context(context)
    root = _store_root(store)
    expected_checkpoint = _canonical_checkpoint(root)
    context_fingerprint = getattr(store, "context_fingerprint", None)
    if not isinstance(context_fingerprint, str):
        context_fingerprint = _context_hash(context)
    _assert_label_free(progress)
    _validate_binding(
        progress,
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(records),
        run_identity=str(getattr(store, "run_identity")),
        context_fingerprint=context_fingerprint,
        source="runtime progress",
        require_total=True,
    )
    # ``complete_review_only`` is a publication/result state, never a runtime
    # checkpoint state.  Accepting it here would allow a second publication
    # pass to masquerade as the first checkpoint-complete boundary.
    if progress.get("state") != "checkpoint_complete":
        raise ValueError("runtime is not checkpoint-complete")
    if progress.get("current_dataset") is not None:
        raise ValueError("runtime current dataset is not null")
    if progress.get("eta_seconds") not in {0, 0.0}:
        raise ValueError("runtime ETA is not zero")
    for field in ("completed", "completed_datasets", "success", "abstention", "error", "candidate_count"):
        if type(progress.get(field)) is not int:
            raise ValueError("runtime completion accounting is invalid")
    recorded_success = sum(result.status == "scored" for result in records)
    recorded_abstention = sum(result.status == "abstained" for result in records)
    recorded_error = sum(result.status == "failed" for result in records)
    if (
        progress["completed"] != len(records)
        or progress["completed_datasets"] != len(records)
        or progress["success"] != recorded_success
        or progress["abstention"] != recorded_abstention
        or progress["error"] != recorded_error
        or progress["success"] + progress["abstention"] + progress["error"] != progress["completed"]
    ):
        raise ValueError("runtime completion accounting is invalid")
    if progress["candidate_count"] != sum(result.candidate_count for result in records):
        raise ValueError("runtime candidate accounting does not match checkpoint")
    _validate_events(
        output / "events_antigravity.jsonl",
        str(getattr(store, "run_identity")),
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(records),
        context_fingerprint=context_fingerprint,
    )
    completed: dict[str, Mapping[str, object]] = {}
    for line in (output / "events_antigravity.jsonl").read_bytes().splitlines():
        item = json.loads(line)
        if item.get("kind") == "dataset_complete":
            dataset = item.get("dataset")
            if not isinstance(dataset, str) or dataset in completed:
                raise ValueError("runtime completion events are duplicated or invalid")
            completed[dataset] = item
    if set(completed) != set(record_digests):
        raise ValueError("runtime completion events do not cover the checkpoint")
    for dataset, item in completed.items():
        if item.get("record_sha256") != record_digests[dataset]:
            raise ValueError("runtime event record SHA256 does not match checkpoint")
    pid = _read_json(output / "run_antigravity.pid", "runtime PID")
    _assert_label_free(pid)
    _validate_binding(
        pid,
        checkpoint=expected_checkpoint,
        backend=backend.backend,
        total_datasets=len(records),
        run_identity=str(getattr(store, "run_identity")),
        context_fingerprint=context_fingerprint,
        source="runtime PID",
    )
    if pid.get("active") is not False:
        raise ValueError("runtime PID is still active")
    return progress


def _build_payloads(
    store: object,
    output: Path,
    expectations: _PublicationExpectations,
) -> dict[str, bytes]:
    records, record_digests, context = _record_data(store, expectations)
    progress = _validate_runtime(output, store, records, record_digests, context)
    backend = _backend_from_context(context)
    run_identity = str(getattr(store, "run_identity"))
    candidate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    abstention_rows: list[dict[str, object]] = []
    for result in records:
        for candidate in result.candidates:
            candidate_rows.append({
                "dataset": candidate.dataset,
                "instance_index": candidate.instance_index,
                "length": candidate.length,
                "evidence_digest": candidate.evidence_digest,
            })
        summary_rows.append({
            "dataset": result.dataset,
            "status": result.status,
            "test_size": result.test_size,
            "candidate_count": result.candidate_count,
            "abstention_reason": result.abstention_reason,
            "error": result.error,
            "record_sha256": record_digests[result.dataset],
        })
        for evidence in result.evidence:
            row = dict(evidence.to_payload())
            row["record_sha256"] = record_digests[result.dataset]
            evidence_rows.append(row)
            if evidence.status != "scored" or evidence.exclusion_reason is not None:
                abstention_rows.append({
                    "dataset": evidence.dataset,
                    "instance_index": evidence.instance_index,
                    "length": evidence.length,
                    "status": evidence.status,
                    "reason": evidence.exclusion_reason,
                    "record_sha256": record_digests[result.dataset],
                })
        if result.status == "abstained" and not result.evidence:
            abstention_rows.append({
                "dataset": result.dataset,
                "instance_index": None,
                "length": None,
                "status": result.status,
                "reason": result.abstention_reason,
                "record_sha256": record_digests[result.dataset],
            })
    candidate_rows.sort(key=lambda row: (row["dataset"], row["instance_index"]))
    summary_rows.sort(key=lambda row: row["dataset"])
    evidence_rows.sort(key=lambda row: (row["dataset"], row["instance_index"]))
    abstention_rows.sort(key=lambda row: (row["dataset"], -1 if row["instance_index"] is None else row["instance_index"]))
    semantic: dict[str, bytes] = {
        PREDICTION_FILE: _jsonl_raw(candidate_rows),
        SUMMARY_FILE: _jsonl_raw(summary_rows),
        EVIDENCE_FILE: _jsonl_raw(evidence_rows),
        ABSTENTION_FILE: _jsonl_raw(abstention_rows),
        BACKEND_FILE: _json_raw({
            "schema": "exp435.backend-decision.v1",
            "decision": backend.to_dict(),
        }),
    }
    semantic_hashes = {name: _digest(raw) for name, raw in sorted(semantic.items())}
    config_hash = _digest(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    context_sha256 = _context_hash(context)
    code_commit = context.get("code_commit")
    if not isinstance(code_commit, str):
        raise ValueError("checkpoint code commit is absent")
    manifest = {
        "schema": "exp435.publication-manifest.v1",
        "run_identity": run_identity,
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "git_commit": code_commit,
        "code_commit": code_commit,
        "code_commit_sha256": _digest(code_commit.encode("utf-8")),
        "config_sha256": config_hash,
        "input_context_sha256": context_sha256,
        "database_sha256": context.get("database_sha256"),
        "input_sha256": _input_hashes(context),
        "lane_artifact_sha256": {
            key: value for key, value in _input_hashes(context).items() if key != "database"
        },
        "checkpoint_context_sha256": context_sha256,
        "checkpoint_record_set_sha256": str(getattr(store, "record_set_sha256")),
        "runtime_versions": dict(sorted(dict(context.get("runtime_versions", {})).items())),
        "backend": backend.to_dict(),
        "backend_decision_sha256": _digest(canonical_json_bytes(backend.to_dict())),
        "prediction_sha256": semantic_hashes[PREDICTION_FILE],
        "completed_datasets": len(records),
        "failed_datasets": 0,
        "candidate_count": sum(result.candidate_count for result in records),
        "review_only": True,
        "operational_alert_eligible": False,
        "contains_test_labels": False,
        "test_labels_opened": False,
        "label_access_count": 0,
        "artifact_sha256": semantic_hashes,
        "artifact_sha256_self_excluded": HASH_FILE,
    }
    _reject_forbidden_keys(manifest, artifact=MANIFEST_FILE)
    manifest_raw = _json_raw(manifest)
    status = {
        "schema": "exp435.publication-status.v1",
        "run_identity": run_identity,
        "state": "complete_review_only",
        "completed_datasets": len(records),
        "failed_datasets": 0,
        "candidate_count": sum(result.candidate_count for result in records),
        "backend": backend.backend,
        "prediction_sha256": semantic_hashes[PREDICTION_FILE],
        "manifest_sha256": _digest(manifest_raw),
        "runtime_progress_sha256": _digest((Path(output) / "progress_antigravity.json").read_bytes()),
        "review_only": True,
        "operational_alert_eligible": False,
        "contains_test_labels": False,
        "test_labels_opened": False,
        "label_access_count": 0,
    }
    _reject_forbidden_keys(status, artifact=STATUS_FILE)
    status_raw = _json_raw(status)
    before_hash_map = {**semantic, MANIFEST_FILE: manifest_raw, STATUS_FILE: status_raw}
    artifact_map = {
        "schema": "exp435.artifact-sha256.v1",
        "artifact_sha256": {name: _digest(raw) for name, raw in sorted(before_hash_map.items())},
        "self_excluded": HASH_FILE,
        "self_hash_excluded": True,
    }
    payloads = {**before_hash_map, HASH_FILE: _json_raw(artifact_map)}
    for name, raw in payloads.items():
        if name.endswith(".json"):
            _reject_forbidden_keys(json.loads(raw), artifact=name)
        else:
            for line in raw.splitlines():
                _reject_forbidden_keys(json.loads(line), artifact=name)
    if set(payloads) != _PUBLICATION_FILES or progress.get("state") != "checkpoint_complete":
        raise ValueError("publication artifact inventory is invalid")
    return payloads


def _validated_root(output: Path) -> Path:
    publication = Path(output) / PUBLICATION_DIRECTORY
    if publication.is_symlink() or not publication.is_dir():
        raise ValueError("publication directory is absent or invalid")
    entries = tuple(publication.iterdir())
    if {item.name for item in entries} != _PUBLICATION_FILES:
        raise ValueError("publication inventory is invalid")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError("publication inventory contains an invalid file")
    return publication


def _verify_bytes(
    output: Path,
    store: object,
    expectations: _PublicationExpectations,
) -> FrozenPublication:
    publication = _validated_root(Path(output))
    expected = _build_payloads(store, Path(output), expectations)
    artifact_map = _read_json(publication / HASH_FILE, "artifact SHA256 map")
    hashes = artifact_map.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != _PUBLICATION_FILES - {HASH_FILE}:
        raise ValueError("artifact SHA256 map inventory is invalid")
    if artifact_map.get("self_excluded") != HASH_FILE or artifact_map.get("self_hash_excluded") is not True:
        raise ValueError("artifact SHA256 map self-exclusion is invalid")
    for name in sorted(_PUBLICATION_FILES):
        actual = (publication / name).read_bytes()
        if actual != expected[name]:
            raise ValueError(f"publication {name} does not match checkpoint replay")
        if name != HASH_FILE and hashes.get(name) != _digest(actual):
            raise ValueError(f"publication {name} SHA256 does not match")
    manifest = _read_json(publication / MANIFEST_FILE, "publication manifest")
    if manifest.get("prediction_sha256") != _digest((publication / PREDICTION_FILE).read_bytes()):
        raise ValueError("prediction SHA256 does not match manifest")
    return FrozenPublication(publication, manifest)


def inspect_publication(
    output: Path,
    completed_store: object,
    *,
    _expectations: _PublicationExpectations = _PublicationExpectations(),
) -> FrozenPublication:
    """Replay and verify bytes without opening DB, labels, or workers."""

    if not isinstance(_expectations, _PublicationExpectations):
        raise ValueError("publication expectations are invalid")
    store = _open_store(completed_store)
    return _verify_bytes(Path(output), store, _expectations)


def publish_exp435(
    completed_store: object,
    output: Path,
    *,
    _expectations: _PublicationExpectations = _PublicationExpectations(),
) -> FrozenPublication:
    """Publish a fully completed checkpoint exactly once, without clobbering."""

    if not isinstance(_expectations, _PublicationExpectations):
        raise ValueError("publication expectations are invalid")
    store = _open_store(completed_store)
    output = Path(output)
    # Validate every precondition, including budget, before allocating a
    # publication staging directory.  A failed run can therefore never leave
    # a partial publication behind.
    payloads = _build_payloads(store, output, _expectations)
    publication = output / PUBLICATION_DIRECTORY
    if publication.exists() or publication.is_symlink():
        return _verify_bytes(output, store, _expectations)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("runtime output root is absent or invalid")
    staging = output / f".{PUBLICATION_DIRECTORY}.{uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    _fsync_directory(output)
    published = False
    try:
        for name in sorted(payloads):
            _write_new(staging / name, payloads[name])
        _fsync_directory(staging)
        if publication.exists() or publication.is_symlink():
            raise ValueError("publication appeared during atomic staging")
        _exclusive_rename(staging, publication)
        _fsync_directory(output)
        published = True
    finally:
        if not published and staging.is_dir() and not staging.is_symlink():
            for child in tuple(staging.iterdir()):
                if child.is_file() and not child.is_symlink():
                    child.unlink()
            staging.rmdir()
            _fsync_directory(output)
    return _verify_bytes(output, store, _expectations)


__all__ = [
    "ABSTENTION_FILE",
    "BACKEND_FILE",
    "EVIDENCE_FILE",
    "FrozenPublication",
    "HASH_FILE",
    "MANIFEST_FILE",
    "PUBLICATION_DIRECTORY",
    "PREDICTION_FILE",
    "PublicationExpectations",
    "STATUS_FILE",
    "SUMMARY_FILE",
    "inspect_publication",
    "publish_exp435",
]
