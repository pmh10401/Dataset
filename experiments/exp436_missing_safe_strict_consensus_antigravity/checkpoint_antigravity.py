"""Ordered, single-writer Exp436 checkpoint authority.

The checkpoint deliberately stores Exp436's own strict records.  It has no
dependency on Exp435 so a malformed predecessor-shaped record cannot enter an
Exp436 recovery run.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from hashlib import sha256
import fcntl
import json
import os
import stat
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Iterator
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendDecision, ExcludedTrainRowAudit, Exp436Candidate, Exp436DatasetResult, InstanceScoreEvidence

_ROOT = frozenset({"run_context_antigravity.json", "record_set_antigravity.json", "records_antigravity", "intents_antigravity"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_SET_TEMP = re.compile(r"\.record_set_antigravity\.json\.[0-9a-f]{32}\.tmp\Z")

_CONTEXT_KEYS = frozenset({
    "schema", "context_sha256", "protocol_versions", "experiment_id", "run_identity",
    "reviewed_commit", "readiness_path", "readiness_artifact_sha256", "backend_decision",
    "backend_decision_sha256", "database_path", "database_sha256", "lane_artifacts",
    "lane_mappings", "lane_snapshot_sha256", "datasets", "dataset_order_sha256",
    "test_identity_manifest", "test_manifest_sha256", "frozen_config", "frozen_config_sha256",
    "python_implementation", "python_version", "python_executable", "numpy_version",
    "platform", "machine", "multiprocessing_start_method", "logical_cpus", "cpu_workers",
    "selected_framework_provenance", "backend_protocol", "selected_batch_size", "output_path",
    "checkpoint_path",
})


@dataclass(frozen=True)
class Exp436RunContext:
    """The only persisted Exp436 v1 context; no arbitrary object projection."""

    payload: Mapping[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "Exp436RunContext":
        if not isinstance(payload, Mapping) or set(payload) != _CONTEXT_KEYS:
            raise ValueError("run context keys are invalid")
        value = dict(payload)
        if value["schema"] != "exp436.run-context.v1" or value["experiment_id"] != FROZEN_CONFIG.experiment_id or value["run_identity"] != FROZEN_CONFIG.run_identity:
            raise ValueError("run context identity is invalid")
        if not isinstance(value["reviewed_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", value["reviewed_commit"]) is None:
            raise ValueError("run context reviewed commit is invalid")
        for name in ("readiness_artifact_sha256", "backend_decision_sha256", "database_sha256", "lane_snapshot_sha256", "dataset_order_sha256", "test_manifest_sha256", "frozen_config_sha256", "context_sha256"):
            if not isinstance(value[name], str) or _SHA.fullmatch(value[name]) is None:
                raise ValueError("run context digest is invalid")
        for name in ("readiness_path", "database_path", "output_path", "checkpoint_path", "python_executable"):
            if not isinstance(value[name], str) or not Path(value[name]).is_absolute() or ".." in Path(value[name]).parts:
                raise ValueError("run context path is invalid")
        protocols = value["protocol_versions"]
        if protocols != {"context": "v1", "checkpoint": "v1", "readiness": "v1"}:
            raise ValueError("run context protocol versions are invalid")
        try:
            decision = BackendDecision.from_payload(value["backend_decision"])
        except (TypeError, ValueError) as error:
            raise ValueError("run context backend decision is invalid") from error
        if decision.to_payload() != value["backend_decision"]:
            raise ValueError("run context backend decision is noncanonical")
        if value["frozen_config"] != json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG))):
            raise ValueError("run context frozen config is invalid")
        datasets = value["datasets"]
        if not isinstance(datasets, list) or len(datasets) != FROZEN_CONFIG.expected_datasets or any(type(x) is not str for x in datasets) or datasets != sorted(datasets) or len(set(datasets)) != len(datasets):
            raise ValueError("run context population is invalid")
        manifest = value["test_identity_manifest"]
        if not isinstance(manifest, Mapping) or set(manifest) != set(datasets):
            raise ValueError("run context TEST manifest is invalid")
        for dataset in datasets:
            rows = manifest[dataset]
            if not isinstance(rows, list) or not rows:
                raise ValueError("run context TEST manifest is invalid")
            indices = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 2 or type(row[0]) is not int or type(row[1]) is not int or row[0] < 0 or row[1] < 1:
                    raise ValueError("run context TEST manifest is invalid")
                indices.append(row[0])
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("run context TEST manifest has duplicate or unordered indices")
        if sha256(canonical_json_bytes(datasets)).hexdigest() != value["dataset_order_sha256"]:
            raise ValueError("run context dataset order digest is invalid")
        if sha256(canonical_json_bytes(manifest)).hexdigest() != value["test_manifest_sha256"]:
            raise ValueError("run context TEST manifest digest is invalid")
        lanes = value["lane_artifacts"]
        if not isinstance(lanes, Mapping) or set(lanes) != {"exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"}:
            raise ValueError("run context lane artifacts are invalid")
        for record in lanes.values():
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"} or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute() or not isinstance(record["sha256"], str) or _SHA.fullmatch(record["sha256"]) is None:
                raise ValueError("run context lane artifact is invalid")
        lane_mappings = value["lane_mappings"]
        if not isinstance(lane_mappings, Mapping) or set(lane_mappings) != {"hard", "standard", "priority", "automatic_normal"}:
            raise ValueError("run context lane mappings are invalid")
        seen: set[tuple[str, int]] = set()
        for lane in lane_mappings.values():
            if not isinstance(lane, Mapping): raise ValueError("run context lane mapping is invalid")
            for dataset, indices in lane.items():
                if dataset not in datasets or not isinstance(indices, list) or indices != sorted(indices) or len(indices) != len(set(indices)) or any(type(index) is not int or index < 0 for index in indices):
                    raise ValueError("run context lane mapping is invalid")
                for index in indices:
                    if (dataset, index) in seen: raise ValueError("run context lane mappings overlap")
                    seen.add((dataset, index))
        lane_bound = {"artifacts": lanes, "mappings": lane_mappings, "datasets": datasets}
        if sha256(canonical_json_bytes(lane_bound)).hexdigest() != value["lane_snapshot_sha256"]:
            raise ValueError("run context lane snapshot digest is invalid")
        expected_provenance = {"framework": decision.backend, "adopted": decision.adopted}
        if value["selected_framework_provenance"] != expected_provenance or value["backend_protocol"] != FROZEN_CONFIG.backend_protocol or value["selected_batch_size"] != decision.selected_batch_size:
            raise ValueError("run context selected framework provenance is invalid")
        if type(value["logical_cpus"]) is not int or value["logical_cpus"] < 1 or value["cpu_workers"] != FROZEN_CONFIG.cpu_workers or value["multiprocessing_start_method"] not in {"spawn", "fork", "forkserver"}:
            raise ValueError("run context topology is invalid")
        for name in ("protocol_versions", "lane_mappings", "backend_decision", "frozen_config", "selected_framework_provenance"):
            if not isinstance(value[name], Mapping): raise ValueError("run context nested object is invalid")
        if sha256(canonical_json_bytes(value["backend_decision"])).hexdigest() != value["backend_decision_sha256"] or sha256(canonical_json_bytes(value["frozen_config"])).hexdigest() != value["frozen_config_sha256"]:
            raise ValueError("run context nested digest is invalid")
        unsigned = dict(value); unsigned.pop("context_sha256")
        if sha256(canonical_json_bytes(unsigned)).hexdigest() != value["context_sha256"]:
            raise ValueError("run context digest is invalid")
        return cls(_freeze(value))

    def to_payload(self) -> dict[str, object]:
        value = _thaw(self.payload)
        if not isinstance(value, dict): raise ValueError("run context projection is invalid")
        return value

def _fsync_dir(path: Path) -> None:
    fd = _open_directory(path, "checkpoint directory")
    try: os.fsync(fd)
    finally: os.close(fd)

def _open_directory(path: Path, label: str) -> int:
    try:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode): raise ValueError(f"{label} is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"{label} is not a directory")
        return descriptor
    except OSError as error:
        raise ValueError(f"{label} is invalid") from error

@contextmanager
def _root_lock(root: Path, exclusive: bool) -> Iterator[None]:
    """Hold the only checkpoint-root lock across one coherent operation."""
    descriptor = _open_directory(Path(root), "checkpoint root")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

def _directory_names(path: Path, label: str) -> tuple[str, ...]:
    descriptor = _open_directory(path, label)
    try:
        return tuple(sorted(os.listdir(descriptor)))
    except OSError as error:
        raise ValueError(f"{label} is invalid") from error
    finally:
        os.close(descriptor)

def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode): raise ValueError(f"{label} is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode): raise ValueError(f"{label} is not a regular file")
            chunks: list[bytes] = []
            while True:
                part = os.read(descriptor, 1 << 20)
                if not part: break
                chunks.append(part)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ValueError(f"{label} is invalid") from error

def _write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw); os.fsync(fd)
    finally: os.close(fd)
    _fsync_dir(path.parent)

def _replace(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    _write_new(temporary, raw)
    try:
        os.replace(temporary, path); _fsync_dir(path.parent)
    finally: temporary.unlink(missing_ok=True)

def _object(path: Path, label: str) -> dict[str, object]:
    try:
        raw = _regular_bytes(path, label)
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw: raise ValueError(f"{label} is not canonical JSON")
    return value

def _digest(raw: bytes) -> str: return sha256(raw).hexdigest()
def _name(dataset: str) -> str: return sha256(dataset.encode()).hexdigest() + ".json"
def _set_hash(items: Mapping[str, str]) -> str: return _digest(canonical_json_bytes(dict(sorted(items.items()))))

def _normalise(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)): return value
    if isinstance(value, Path): return str(value)
    if isinstance(value, Mapping): return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)): return [_normalise(v) for v in value]
    payload = getattr(value, "to_payload", None)
    if callable(payload): return _normalise(payload())
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict): return {str(k): _normalise(v) for k, v in sorted(data.items()) if not str(k).startswith("_")}
    raise ValueError("context contains unsupported value")

def _freeze(value: object) -> object:
    if isinstance(value, Mapping): return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(item) for item in value)
    return value

def _thaw(value: object) -> object:
    if isinstance(value, Mapping): return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_thaw(item) for item in value]
    return value

def context_payload(context: object) -> dict[str, object]:
    value = _normalise(context)
    if not isinstance(value, dict): raise ValueError("run context must be an object")
    return value

def context_fingerprint(context: object) -> str: return _digest(canonical_json_bytes(context_payload(context)))

def _datasets(context: Mapping[str, object]) -> tuple[str, ...]:
    raw = context.get("datasets")
    if not isinstance(raw, list) or not raw or any(type(v) is not str or not v for v in raw): raise ValueError("context datasets are invalid")
    values = tuple(raw)
    if values != tuple(sorted(values)) or len(values) != len(set(values)): raise ValueError("context datasets are not sorted and unique")
    return values

def _manifest(context: Mapping[str, object], datasets: tuple[str, ...]) -> Mapping[str, tuple[tuple[int, int], ...]]:
    raw = context.get("test_identity_manifest")
    if not isinstance(raw, Mapping) or set(raw) != set(datasets): raise ValueError("context TEST manifest is incomplete")
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for dataset in datasets:
        items = raw[dataset]
        if not isinstance(items, list) or not items: raise ValueError("context TEST manifest is invalid")
        pairs: list[tuple[int, int]] = []
        for item in items:
            if not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int or type(item[1]) is not int or item[0] < 0 or item[1] < 1: raise ValueError("context TEST manifest is invalid")
            pairs.append((item[0], item[1]))
        if tuple(pairs) != tuple(sorted(pairs)) or len(set(pairs)) != len(pairs): raise ValueError("context TEST manifest is invalid")
        result[dataset] = tuple(pairs)
    return MappingProxyType(result)

def _dataset_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"train_count", "test_count", "train_length_histogram", "test_length_histogram"}:
        raise ValueError("dataset record metadata is invalid")
    result = dict(value)
    for name in ("train_count", "test_count"):
        if type(result[name]) is not int or result[name] < (1 if name == "train_count" else 0): raise ValueError("dataset record metadata is invalid")
    for name, count in (("train_length_histogram", result["train_count"]), ("test_length_histogram", result["test_count"])):
        histogram = result[name]
        if not isinstance(histogram, Mapping) or sum(histogram.values()) != count or any(type(key) is not str or not key.isdecimal() or key.startswith("0") or type(item) is not int or item < 0 for key, item in histogram.items()): raise ValueError("dataset record metadata is invalid")
        result[name] = dict(sorted(histogram.items(), key=lambda item: (len(item[0]), item[0])))
        if list(histogram) != list(result[name]): raise ValueError("dataset record metadata is invalid")
    return result

def _result_payload(result: Exp436DatasetResult, sequence_index: int, dataset_metadata: Mapping[str, object] | None = None) -> dict[str, object]:
    payload = {"schema": "exp436.dataset-record.v1", "sequence_index": sequence_index, "result": result.to_payload()}
    if dataset_metadata is not None: payload["dataset_metadata"] = _dataset_metadata(dataset_metadata)
    return payload

def _result_from_payload(payload: Mapping[str, object], *, expected_sequence: int | None = None) -> Exp436DatasetResult:
    if set(payload) not in ({"schema", "sequence_index", "result"}, {"schema", "sequence_index", "result", "dataset_metadata"}) or payload.get("schema") != "exp436.dataset-record.v1" or type(payload.get("sequence_index")) is not int or not isinstance(payload.get("result"), Mapping): raise ValueError("dataset record schema is invalid")
    if "dataset_metadata" in payload: _dataset_metadata(payload["dataset_metadata"])
    if expected_sequence is not None and payload["sequence_index"] != expected_sequence: raise ValueError("dataset record sequence is invalid")
    r = payload["result"]
    required = {"dataset", "status", "test_instance_indices", "candidates", "evidence", "train_exclusion_audits", "abstention_reason", "error"}
    if set(r) != required: raise ValueError("dataset record keys are invalid")
    evidence_keys = {"dataset", "instance_index", "length", "prototype_score", "prototype_threshold", "prototype_abnormal", "prototype_unavailability_reason", "gradient_score", "gradient_threshold", "gradient_abnormal", "gradient_unavailability_reason", "acf_score", "acf_threshold", "acf_abnormal", "acf_unavailability_reason", "consensus_candidate", "status", "exclusion_reason", "evidence_digest"}
    candidate_keys = {"dataset", "instance_index", "length", "evidence_digest"}
    audit_keys = {"schema", "dataset", "instance_index", "length", "nonfinite_count", "raw_blob_sha256", "audit_digest"}
    try:
        evidence_raw, candidates_raw, audits_raw = r["evidence"], r["candidates"], r["train_exclusion_audits"]
        if not all(isinstance(x, list) for x in (evidence_raw, candidates_raw, audits_raw)): raise ValueError("dataset record arrays are invalid")
        if any(not isinstance(x, Mapping) or set(x) != evidence_keys for x in evidence_raw): raise ValueError("evidence payload keys are invalid")
        if any(not isinstance(x, Mapping) or set(x) != candidate_keys for x in candidates_raw): raise ValueError("candidate payload keys are invalid")
        if any(not isinstance(x, Mapping) or set(x) != audit_keys or x.get("schema") != "exp436.excluded-train-row.v1" for x in audits_raw): raise ValueError("audit payload keys are invalid")
        evidence = tuple(InstanceScoreEvidence(**dict(x)) for x in evidence_raw)
        candidates = tuple(Exp436Candidate(**dict(x)) for x in candidates_raw)
        audits = tuple(ExcludedTrainRowAudit(**{key: value for key, value in x.items() if key != "schema"}) for x in audits_raw)
        if type(r["dataset"]) is not str or type(r["status"]) is not str or not isinstance(r["test_instance_indices"], list): raise ValueError("dataset record scalar types are invalid")
        return Exp436DatasetResult(r["dataset"], r["status"], tuple(r["test_instance_indices"]), candidates, evidence, audits, r["abstention_reason"], r["error"])
    except (TypeError, ValueError) as error: raise ValueError("dataset record payload is invalid") from error

@dataclass(frozen=True)
class Exp436CheckpointInspection:
    run_identity: str
    context_fingerprint: str
    completed: int
    total: int
    record_digests: Mapping[str, str]
    reconciliation_required: bool = False
    reconciliation_reasons: tuple[str, ...] = ()

@dataclass(frozen=True)
class _CheckpointScan:
    store: "Exp436CheckpointStore"
    inspection: Exp436CheckpointInspection

class Exp436CheckpointStore:
    def __init__(self, root: Path, context: Mapping[str, object], records: Mapping[str, str], *, fingerprint: str) -> None:
        self.root, self.context, self._records, self.context_fingerprint = Path(root), _freeze(context), dict(records), fingerprint
        self.datasets = _datasets(context); self.manifest = _manifest(context, self.datasets)
        self.run_identity = str(context.get("run_identity"))

    @classmethod
    def create(cls, root: Path, context: object, backend_decision: object | None = None) -> "Exp436CheckpointStore":
        root = Path(root)
        if root.exists() or root.is_symlink(): raise ValueError("checkpoint root already exists")
        if not isinstance(context, Exp436RunContext) or backend_decision is not None:
            raise ValueError("checkpoint requires an Exp436RunContext v1")
        projection = context.to_payload()
        datasets = _datasets(projection); _manifest(projection, datasets)
        if projection.get("run_identity") != FROZEN_CONFIG.run_identity: raise ValueError("checkpoint run identity does not match frozen identity")
        fingerprint = context_fingerprint(projection)
        root.mkdir(mode=0o700); _fsync_dir(root.parent)
        with _root_lock(root, True):
            (root / "records_antigravity").mkdir(mode=0o700); (root / "intents_antigravity").mkdir(mode=0o700); _fsync_dir(root)
            _write_new(root / "run_context_antigravity.json", canonical_json_bytes({"schema": "exp436.run-context.v1", "context_fingerprint": fingerprint, "context": projection}))
            empty = cls._set_payload_static(fingerprint, (), {})
            _write_new(root / "record_set_antigravity.json", canonical_json_bytes(empty))
        return cls(root, projection, {}, fingerprint=fingerprint)

    @classmethod
    def _load_locked(cls, root: Path, expected_context: object | None = None, *, mutable: bool = False, allow_temp: bool = False) -> "Exp436CheckpointStore":
        root = Path(root)
        descriptor = _open_directory(root, "checkpoint root")
        os.close(descriptor)
        names = set(_directory_names(root, "checkpoint root"))
        extras = names - _ROOT
        if extras and (not allow_temp or len(extras) != 1 or not _RECORD_SET_TEMP.fullmatch(next(iter(extras)))): raise ValueError("checkpoint inventory is invalid")
        context_file, record_set = _object(root / "run_context_antigravity.json", "run context"), _object(root / "record_set_antigravity.json", "record set")
        if set(context_file) != {"schema", "context_fingerprint", "context"} or context_file.get("schema") != "exp436.run-context.v1" or not isinstance(context_file.get("context"), Mapping): raise ValueError("run context schema is invalid")
        context = dict(context_file["context"])
        # Reconstruct through the explicit v1 parser before trusting any
        # stored population, TEST manifest, path, or provenance field.
        Exp436RunContext.from_payload(context)
        fingerprint = context_fingerprint(context)
        if context_file.get("context_fingerprint") != fingerprint: raise ValueError("context fingerprint is invalid")
        if expected_context is not None and context != context_payload(expected_context): raise ValueError("context fingerprint does not match verified context")
        required_set = {"schema", "context_fingerprint", "completed_datasets", "completed", "record_digests", "candidate_count", "excluded_train_rows", "nonfinite_test_rows", "success", "abstention", "record_set_sha256"}
        if set(record_set) != required_set or record_set.get("schema") != "exp436.record-set.v1" or record_set.get("context_fingerprint") != fingerprint: raise ValueError("record set schema is invalid")
        digests = record_set.get("record_digests")
        if not isinstance(digests, Mapping) or any(type(k) is not str or not isinstance(v, str) or not _SHA.fullmatch(v) for k, v in digests.items()) or record_set.get("completed") != len(digests): raise ValueError("record set is invalid")
        store = cls(root, context, dict(digests), fingerprint=fingerprint); store._validate_records(allow_pending=mutable); store._validate_set_counters(record_set)
        return store

    @classmethod
    def _inspection_locked(cls, store: "Exp436CheckpointStore") -> Exp436CheckpointInspection:
        """Classify only validated crash leftovers while the root lock is held."""
        reasons: list[str] = []
        intent_names = _directory_names(store.root / "intents_antigravity", "checkpoint intent directory")
        if intent_names:
            if len(intent_names) != 1: raise ValueError("checkpoint has multiple intents")
            intent = store.root / "intents_antigravity" / intent_names[0]
            dataset, digest = store._intent(intent)
            record_names = set(_directory_names(store.root / "records_antigravity", "checkpoint record directory"))
            record_exists = _name(dataset) in record_names
            next_dataset = store.datasets[len(store._records)] if len(store._records) < len(store.datasets) else None
            if dataset == next_dataset and not record_exists: reasons.append("intent_without_record")
            elif dataset == next_dataset and record_exists:
                store._read_record(dataset, digest); reasons.append("intent_with_record")
            elif dataset in store._records and store._records[dataset] == digest and record_exists:
                store._read_record(dataset, digest); reasons.append("stale_intent_after_record_set")
            else: raise ValueError("checkpoint intent state is impossible")
        for name in _directory_names(store.root, "checkpoint root"):
            if _RECORD_SET_TEMP.fullmatch(name):
                temporary = _object(store.root / name, "checkpoint temporary")
                current = _object(store.root / "record_set_antigravity.json", "record set")
                if temporary != current:
                    if "intent_with_record" not in reasons:
                        raise ValueError("checkpoint temporary is unrelated")
                    intent = store.root / "intents_antigravity" / intent_names[0]
                    dataset, digest = store._intent(intent)
                    updated = dict(store._records); updated[dataset] = digest
                    results = tuple(store._read_record(item, item_digest) for item, item_digest in updated.items())
                    expected = store._set_payload_static(store.context_fingerprint, tuple(updated), updated, results)
                    if temporary != expected:
                        raise ValueError("checkpoint temporary is unrelated")
                reasons.append("record_set_temp")
        return Exp436CheckpointInspection(store.run_identity, store.context_fingerprint, len(store._records), len(store.datasets), MappingProxyType(dict(store._records)), bool(reasons), tuple(reasons))

    @classmethod
    def _scan_locked(cls, root: Path, expected_context: object | None = None, *, mode: str) -> _CheckpointScan:
        """Read one complete snapshot. The caller owns the sole root flock."""
        if mode not in {"pristine", "reconcile"}: raise ValueError("checkpoint scan mode is invalid")
        store = cls._load_locked(root, expected_context, mutable=mode == "reconcile", allow_temp=mode == "reconcile")
        inspection = cls._inspection_locked(store) if mode == "reconcile" else Exp436CheckpointInspection(store.run_identity, store.context_fingerprint, len(store._records), len(store.datasets), MappingProxyType(dict(store._records)))
        return _CheckpointScan(store, inspection)

    @classmethod
    def _repair_locked(cls, scan: _CheckpointScan) -> None:
        """Repair exactly the classified crash state; never rescan or relock."""
        store, reasons = scan.store, scan.inspection.reconciliation_reasons
        if "record_set_temp" in reasons:
            temporary = [store.root / name for name in _directory_names(store.root, "checkpoint root") if _RECORD_SET_TEMP.fullmatch(name)]
            if len(temporary) != 1: raise ValueError("checkpoint temporary inventory is invalid")
            temporary[0].unlink()
            _fsync_dir(store.root)
        intent_names = _directory_names(store.root / "intents_antigravity", "checkpoint intent directory")
        if not intent_names: return
        if len(intent_names) != 1: raise ValueError("checkpoint has multiple intents")
        intent = store.root / "intents_antigravity" / intent_names[0]
        dataset, digest = store._intent(intent)
        record_names = set(_directory_names(store.root / "records_antigravity", "checkpoint record directory"))
        if dataset in store._records:
            if store._records[dataset] != digest or _name(dataset) not in record_names: raise ValueError("checkpoint intent disagrees with record set")
            store._read_record(dataset, digest)
        elif _name(dataset) in record_names:
            expected = store.datasets[len(store._records)] if len(store._records) < len(store.datasets) else None
            if dataset != expected: raise ValueError("checkpoint intent/record is not next prefix")
            store._read_record(dataset, digest)
            updated = dict(store._records); updated[dataset] = digest; store._write_set(updated); store._records = updated
        else:
            expected = store.datasets[len(store._records)] if len(store._records) < len(store.datasets) else None
            if dataset != expected: raise ValueError("checkpoint intent is not next prefix")
        intent.unlink(); _fsync_dir(intent.parent)

    @classmethod
    def open_existing(cls, root: Path, context: object, backend_decision: object | None = None) -> "Exp436CheckpointStore":
        if not isinstance(context, Exp436RunContext) or backend_decision is not None:
            raise ValueError("checkpoint requires an Exp436RunContext v1")
        expected = context.to_payload()
        with _root_lock(Path(root), True):
            scan = cls._scan_locked(root, expected, mode="reconcile")
            cls._repair_locked(scan)
            return cls._scan_locked(root, expected, mode="pristine").store

    @classmethod
    def open_read_only(cls, root: Path, *, expected_context: object | None = None) -> "Exp436CheckpointStore":
        with _root_lock(Path(root), False): return cls._scan_locked(root, expected_context, mode="pristine").store

    @classmethod
    def open_classified_read_only(cls, root: Path, *, expected_context: object | None = None) -> "Exp436CheckpointStore":
        """Read the durable prefix while classifying recognized crash leftovers."""
        with _root_lock(Path(root), False): return cls._scan_locked(root, expected_context, mode="reconcile").store

    @classmethod
    def inspect_read_only(cls, root: Path, *, expected_context: object | None = None) -> Exp436CheckpointInspection:
        # Inspection deliberately accepts only syntactically recognized crash
        # leftovers; it never unlinks, replaces, or reconciles them.
        with _root_lock(Path(root), False): return cls._scan_locked(root, expected_context, mode="reconcile").inspection

    @classmethod
    def read_context_projection(cls, root: Path) -> Mapping[str, object]:
        with _root_lock(Path(root), False): return cls._scan_locked(root, mode="pristine").store.context

    def completed_digests(self) -> Mapping[str, str]:
        with _root_lock(self.root, False):
            return MappingProxyType(dict(type(self)._scan_locked(self.root, self.context, mode="reconcile").store._records))

    def inspect(self) -> Exp436CheckpointInspection:
        with _root_lock(self.root, False): return type(self)._scan_locked(self.root, self.context, mode="reconcile").inspection

    def _validate_records(self, *, allow_pending: bool) -> None:
        expected_prefix = self.datasets[:len(self._records)]
        if tuple(self._records) != expected_prefix: raise ValueError("checkpoint records are not the ordered prefix")
        records, intents = self.root / "records_antigravity", self.root / "intents_antigravity"
        record_names = set(_directory_names(records, "checkpoint record directory"))
        intent_names = set(_directory_names(intents, "checkpoint intent directory"))
        expected = {_name(d) for d in self._records}
        for name in intent_names:
            if not allow_pending: raise ValueError("checkpoint requires reconciliation")
            _regular_bytes(intents / name, "checkpoint intent")
        # Intent-only is the recognized first crash window.  An intent's name
        # becomes an expected record only after the record file itself exists.
        pending_record_names = (intent_names & record_names) if allow_pending else set()
        if record_names != expected | pending_record_names: raise ValueError("checkpoint record inventory is invalid")
        for dataset, digest in self._records.items(): self._read_record(dataset, digest)

    def _intent(self, path: Path) -> tuple[str, str]:
        payload = _object(path, "checkpoint intent")
        if set(payload) != {"schema", "dataset", "record_sha256"} or payload.get("schema") != "exp436.commit-intent.v1": raise ValueError("checkpoint intent is invalid")
        dataset, digest = payload.get("dataset"), payload.get("record_sha256")
        if not isinstance(dataset, str) or not isinstance(digest, str) or path.name != _name(dataset) or _SHA.fullmatch(digest) is None:
            raise ValueError("checkpoint intent is invalid")
        return dataset, digest

    def _read_record(self, dataset: str, digest: str) -> Exp436DatasetResult:
        path = self.root / "records_antigravity" / _name(dataset)
        raw = _regular_bytes(path, "dataset record")
        if _digest(raw) != digest: raise ValueError("checkpoint record digest does not match")
        result = _result_from_payload(_object(path, "dataset record"), expected_sequence=self.datasets.index(dataset))
        if result.dataset != dataset or result.status not in {"scored", "abstained"}: raise ValueError("checkpoint result status is invalid")
        if tuple((e.instance_index, e.length) for e in result.evidence) != self.manifest[dataset]: raise ValueError("record TEST identities do not match context")
        return result

    def load_record(self, dataset: str) -> Exp436DatasetResult:
        with _root_lock(self.root, False):
            store = type(self)._scan_locked(self.root, self.context, mode="reconcile").store
            if dataset not in store._records: raise ValueError("checkpoint record is absent")
            return store._read_record(dataset, store._records[dataset])

    def load_record_metadata(self, dataset: str) -> Mapping[str, object]:
        with _root_lock(self.root, False):
            store = type(self)._scan_locked(self.root, self.context, mode="reconcile").store
            digest = store._records.get(dataset)
            if digest is None: raise ValueError("checkpoint record is absent")
            return store._read_record_metadata(dataset, digest)

    def _read_record_metadata(self, dataset: str, digest: str) -> Mapping[str, object]:
        payload = _object(self.root / "records_antigravity" / _name(dataset), "dataset record")
        if _digest(canonical_json_bytes(payload)) != digest or "dataset_metadata" not in payload:
            raise ValueError("checkpoint record lacks durable dataset metadata")
        return MappingProxyType(_dataset_metadata(payload["dataset_metadata"]))

    def completed_records(self) -> tuple[tuple[str, str, Exp436DatasetResult, Mapping[str, object]], ...]:
        """Return one locked, validated ordered snapshot for runtime replay."""
        with _root_lock(self.root, False):
            store = type(self)._scan_locked(self.root, self.context, mode="reconcile").store
            return tuple((dataset, digest, store._read_record(dataset, digest), store._read_record_metadata(dataset, digest)) for dataset, digest in store._records.items())

    @staticmethod
    def _set_payload_static(fingerprint: str, datasets: tuple[str, ...], records: Mapping[str, str], results: tuple[Exp436DatasetResult, ...] = ()) -> dict[str, object]:
        payload = {"schema": "exp436.record-set.v1", "context_fingerprint": fingerprint, "completed_datasets": list(datasets), "completed": len(datasets), "record_digests": dict(records), "candidate_count": sum(r.candidate_count for r in results), "excluded_train_rows": sum(len(r.train_exclusion_audits) for r in results), "nonfinite_test_rows": sum(sum(e.exclusion_reason == "nonfinite_test" for e in r.evidence) for r in results), "success": sum(r.status == "scored" for r in results), "abstention": sum(r.status == "abstained" for r in results)}
        payload["record_set_sha256"] = _digest(canonical_json_bytes(payload))
        return payload
    def _write_set(self, records: Mapping[str, str]) -> None:
        datasets = tuple(records)
        results = tuple(self._read_record(dataset, digest) for dataset, digest in records.items())
        _replace(self.root / "record_set_antigravity.json", canonical_json_bytes(self._set_payload_static(self.context_fingerprint, datasets, records, results)))
    def _validate_set_counters(self, record_set: Mapping[str, object]) -> None:
        datasets = tuple(self._records)
        results = tuple(self._read_record(dataset, digest) for dataset, digest in self._records.items())
        expected = self._set_payload_static(self.context_fingerprint, datasets, self._records, results)
        if dict(record_set) != expected or expected["candidate_count"] > FROZEN_CONFIG.global_candidate_budget:
            raise ValueError("record set counters or digest are invalid")

    def commit(self, result: Exp436DatasetResult, *, dataset_metadata: Mapping[str, object] | None = None, _failpoint: object | None = None) -> str:
        if not isinstance(result, Exp436DatasetResult) or result.status not in {"scored", "abstained"}: raise ValueError("checkpoint result is invalid")
        if result.dataset not in self.datasets: raise ValueError("record dataset is outside checkpoint universe")
        if tuple((e.instance_index, e.length) for e in result.evidence) != self.manifest[result.dataset]: raise ValueError("record TEST identities do not match context")
        raw, digest = canonical_json_bytes(_result_payload(result, self.datasets.index(result.dataset), dataset_metadata)), None
        digest = _digest(raw)
        with _root_lock(self.root, True):
            fresh = type(self)._scan_locked(self.root, self.context, mode="pristine").store
            self._records = dict(fresh._records)
            expected = self.datasets[len(self._records)] if len(self._records) < len(self.datasets) else None
            existing = self._records.get(result.dataset)
            path = self.root / "records_antigravity" / _name(result.dataset)
            if existing is not None:
                if existing != digest or _regular_bytes(path, "dataset record") != raw: raise ValueError("dataset already has a different checkpoint record")
                return digest
            if result.dataset != expected: raise ValueError("checkpoint commit must be the next ordered prefix")
            candidates = sum(fresh._read_record(dataset, item_digest).candidate_count for dataset, item_digest in self._records.items()) + result.candidate_count
            if candidates > FROZEN_CONFIG.global_candidate_budget: raise ValueError("Exp436 candidate budget exceeded before intent")
            intent = self.root / "intents_antigravity" / _name(result.dataset)
            _write_new(intent, canonical_json_bytes({"schema": "exp436.commit-intent.v1", "dataset": result.dataset, "record_sha256": digest}))
            if _failpoint is not None:
                if not callable(_failpoint): raise ValueError("checkpoint failpoint is invalid")
                _failpoint("after_intent_fsync")
            _write_new(path, raw)
            if _failpoint is not None: _failpoint("after_record_fsync")
            updated = dict(self._records); updated[result.dataset] = digest
            self._write_set(updated)
            if _failpoint is not None: _failpoint("after_record_set_fsync_before_intent_remove")
            intent.unlink(); _fsync_dir(intent.parent)
            self._records = updated
        return digest


__all__ = ["Exp436CheckpointStore", "Exp436CheckpointInspection", "context_payload", "context_fingerprint"]
