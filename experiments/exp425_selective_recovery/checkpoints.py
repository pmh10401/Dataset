"""Compact, provenance-bound, fail-closed local checkpoints for Exp425."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Iterator, Mapping
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import _context_from_payload, _context_payload
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext

from . import capabilities
from .contracts import Exp425Record, Exp425RunProvenance, ImportedRecordEnvelope, NativeRecordEnvelope, SourceProvenGroupEvidence


_ROOT_FILES = frozenset({"checkpoint_provenance.json", "run_context.json", "progress.json", "datasets", "transactions", "import_batches"})
_FORBIDDEN_RECORD_KEYS = frozenset({"context", "lanes", "datasets", "test_identity_manifest"})
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def record_filename(dataset: str) -> str:
    """A deterministic filename that cannot be interpreted as a path."""
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset name is invalid")
    return sha256(dataset.encode("utf-8")).hexdigest() + ".json"


@dataclass(frozen=True)
class Exp425CheckpointInspection:
    run_identity: str
    expected_datasets: tuple[str, ...]
    completed_count: int
    record_digests: Mapping[str, str]
    import_revalidation_required: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_digests", MappingProxyType(dict(self.record_digests)))


@dataclass(frozen=True)
class _ImportBatch:
    identity: str
    directory: Path
    inode: tuple[int, int]
    attestation_sha256: str
    nonce: str


class Exp425CheckpointStore:
    """A shared-context checkpoint root. Only private capability paths mutate it."""

    def __init__(self, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext, record_digests: Mapping[str, str]) -> None:
        self.root = Path(root)
        self.provenance = provenance
        self.context = context
        self.expected_datasets = provenance.expected_datasets
        self.run_context_sha256 = provenance.run_context_sha256
        if self.run_context_sha256 is None:
            raise ValueError("checkpoint context digest is absent")
        self._record_digests = MappingProxyType(dict(record_digests))
        self.import_revalidation_required: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(cls, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext) -> "Exp425CheckpointStore":
        if not isinstance(provenance, Exp425RunProvenance) or not isinstance(context, FrozenRunContext):
            raise ValueError("Exp425 checkpoint inputs are invalid")
        root = Path(root)
        if root.exists() or root.is_symlink():
            raise ValueError("Exp425 checkpoint root already exists")
        if tuple(context.datasets) != provenance.expected_datasets:
            raise ValueError("checkpoint dataset universe does not match context")
        context_bytes = canonical_json_bytes(_context_payload(context))
        digest = sha256(context_bytes).hexdigest()
        bound = provenance.with_run_context_sha256(digest)
        try:
            root.mkdir(mode=0o700)
            for name in ("datasets", "transactions", "import_batches"):
                (root / name).mkdir(mode=0o700)
            _write_new(root / "run_context.json", context_bytes)
            _write_new(root / "checkpoint_provenance.json", canonical_json_bytes(bound.to_payload()))
            _write_new(root / "progress.json", _progress_bytes(bound.expected_datasets, {}))
            _fsync_directory(root)
        except BaseException:
            # A partially-created root remains fail-closed and is never adopted.
            raise
        return cls(root, bound, context, {})

    @classmethod
    def open(cls, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext) -> "Exp425CheckpointStore":
        if not Path(root).exists():
            raise ValueError("Exp425 checkpoint root is absent")
        store = cls._validated(Path(root), provenance, context, exclusive=True)
        store._reconcile_native_intents()
        return store

    @classmethod
    def inspect_read_only(cls, root: Path, *, expected_run_identity: str | None = None) -> Exp425CheckpointInspection:
        """Validate metadata in place without recovery, quarantine, or writes."""
        root = Path(root)
        with _locked_root(root, exclusive=False):
            provenance, context_bytes, progress = _read_root_metadata(root)
            if expected_run_identity is not None and provenance["run_identity"] != expected_run_identity:
                raise ValueError("checkpoint run identity does not match")
            _validate_inventory(root, provenance, progress, allow_intents=True)
            pending = _import_intent_requirements(root)
            return Exp425CheckpointInspection(
                provenance["run_identity"], tuple(provenance["expected_datasets"]), progress["completed_count"], progress["record_digests"], pending
            )

    @classmethod
    def _validated(cls, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext, *, exclusive: bool) -> "Exp425CheckpointStore":
        if not isinstance(provenance, Exp425RunProvenance) or not isinstance(context, FrozenRunContext):
            raise ValueError("Exp425 checkpoint inputs are invalid")
        with _locked_root(root, exclusive=exclusive):
            stored, context_bytes, progress = _read_root_metadata(root)
            expected_context = canonical_json_bytes(_context_payload(context))
            digest = sha256(expected_context).hexdigest()
            expected = provenance.with_run_context_sha256(digest).to_payload()
            if stored != expected or context_bytes != expected_context:
                raise ValueError("Exp425 checkpoint provenance or context does not match")
            _validate_inventory(root, stored, progress, allow_intents=True)
            return cls(root, provenance.with_run_context_sha256(digest), context, progress["record_digests"])

    def completed_record_digests(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._record_digests))

    def load_record(self, dataset: str) -> Exp425Record:
        if dataset not in self._record_digests:
            raise ValueError("dataset is not completed")
        raw = _read_file(self.root / "datasets" / record_filename(dataset))
        record = _decode_record(raw, self.context)
        if record.dataset != dataset or _record_digest_from_raw(raw) != self._record_digests[dataset]:
            raise ValueError("checkpoint record digest does not match progress")
        return record

    def iter_records(self) -> Iterator[Exp425Record]:
        for dataset in self.expected_datasets:
            if dataset in self._record_digests:
                yield self.load_record(dataset)

    def _commit_native(self, capability: object, record: NativeRecordEnvelope) -> str:
        if not isinstance(record, NativeRecordEnvelope):
            raise ValueError("native checkpoint record is invalid")
        trace = _native_trace_sha256(self.provenance.run_identity, self.run_context_sha256, record)
        body = _native_record_body(self.run_context_sha256, record)
        digest = sha256(canonical_json_bytes(body)).hexdigest()
        if trace != record.trace_sha256 or digest != record.record_payload_sha256:
            raise ValueError("native record digest does not match its canonical preimage")
        capabilities._consume(capability, kind="native", run_identity=self.provenance.run_identity, dataset=record.dataset, context_sha256=self.run_context_sha256, trace_sha256=trace, record_sha256=digest, backend_sha256=record.backend_audit_sha256)
        self._write_record(record, body, digest, intent_extra={})
        return digest

    def _begin_import_batch(self, capability: object, batch_identity: str) -> object:
        claims = capabilities._consume(capability, kind="import_begin", run_identity=self.provenance.run_identity, batch_identity=batch_identity)
        attestation = claims.get("attestation_sha256")
        nonce = claims.get("process_nonce")
        if not isinstance(attestation, str) or not isinstance(nonce, str):
            raise ValueError("import batch authority lacks live attestation")
        directory = self.root / "import_batches" / sha256(batch_identity.encode()).hexdigest()
        directory.mkdir(mode=0o700)
        identity = _inode(directory)
        return _ImportBatch(batch_identity, directory, identity, attestation, nonce)

    def _stage_imported(self, capability: object, batch: object, record: ImportedRecordEnvelope) -> None:
        if not isinstance(batch, _ImportBatch) or not isinstance(record, ImportedRecordEnvelope) or _inode(batch.directory) != batch.inode:
            raise ValueError("import batch is invalid")
        payload = _import_record_body(self.run_context_sha256, record)
        digest = sha256(canonical_json_bytes(payload)).hexdigest()
        capabilities._consume(capability, kind="import_stage", run_identity=self.provenance.run_identity, batch_identity=batch.identity, dataset=record.dataset, source_sha256=record.source_record_sha256, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, record_sha256=digest)
        _write_new(batch.directory / record_filename(record.dataset), canonical_json_bytes(payload | {"record_payload_sha256": digest}))

    def _promote_import_batch(self, capability: object, batch: object) -> None:
        if not isinstance(batch, _ImportBatch) or _inode(batch.directory) != batch.inode:
            raise ValueError("import batch is invalid")
        staged = sorted(batch.directory.glob("*.json"))
        digests = {path.name: _record_digest_from_raw(_read_file(path)) for path in staged}
        final_digest = sha256(canonical_json_bytes({"batch_identity": batch.identity, "record_digests": digests, "attestation_sha256": batch.attestation_sha256})).hexdigest()
        capabilities._consume(capability, kind="import_promote", run_identity=self.provenance.run_identity, batch_identity=batch.identity, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, final_attestation_sha256=final_digest)
        for path in staged:
            record = _decode_record(_read_file(path), self.context)
            body = _import_record_body(self.run_context_sha256, record)
            self._write_record(record, body, sha256(canonical_json_bytes(body)).hexdigest(), intent_extra={"batch_identity": batch.identity, "attestation_sha256": batch.attestation_sha256})

    def _write_record(self, record: Exp425Record, body: dict[str, object], digest: str, *, intent_extra: Mapping[str, object]) -> None:
        if record.dataset not in self.expected_datasets or record.dataset in self._record_digests:
            raise ValueError("checkpoint dataset is invalid or already completed")
        encoded = canonical_json_bytes(body | {"record_payload_sha256": digest})
        _reject_forbidden_keys(body)
        with _locked_root(self.root, exclusive=True):
            _, _, progress = _read_root_metadata(self.root)
            if record.dataset in progress["record_digests"]:
                raise ValueError("checkpoint dataset is already completed")
            intent_name = uuid.uuid4().hex + ".json"
            intent = {"schema": "exp425.intent.v1", "origin": record.origin, "dataset": record.dataset, "record_payload_sha256": digest, **dict(intent_extra)}
            _write_new(self.root / "transactions" / intent_name, canonical_json_bytes(intent))
            _write_new(self.root / "datasets" / record_filename(record.dataset), encoded)
            digests = dict(progress["record_digests"]); digests[record.dataset] = digest
            _atomic_replace(self.root / "progress.json", _progress_bytes(self.expected_datasets, digests, record.dataset))
            (self.root / "transactions" / intent_name).unlink()
            _fsync_directory(self.root / "transactions")
            self._record_digests = MappingProxyType(digests)

    def _reconcile_native_intents(self) -> None:
        with _locked_root(self.root, exclusive=True):
            _, _, progress = _read_root_metadata(self.root)
            digests = dict(progress["record_digests"])
            pending: list[tuple[str, str]] = []
            changed = False
            for path in sorted((self.root / "transactions").glob("*.json")):
                intent = _canonical_object(_read_file(path), "intent")
                origin, dataset, digest = intent.get("origin"), intent.get("dataset"), intent.get("record_payload_sha256")
                if origin == "exp424_import":
                    pending.append((str(intent.get("batch_identity")), str(intent.get("attestation_sha256")))); continue
                if origin != "exp425_native" or not isinstance(dataset, str) or not isinstance(digest, str):
                    raise ValueError("checkpoint intent is invalid")
                target = self.root / "datasets" / record_filename(dataset)
                if not target.exists():
                    path.unlink(); changed = True; continue
                if _record_digest_from_raw(_read_file(target)) != digest:
                    raise ValueError("native checkpoint intent conflicts with record")
                if dataset not in digests:
                    digests[dataset] = digest; changed = True
                path.unlink(); changed = True
            if changed:
                _atomic_replace(self.root / "progress.json", _progress_bytes(self.expected_datasets, digests))
            self._record_digests = MappingProxyType(digests)
            self.import_revalidation_required = tuple(pending)


def _native_trace_sha256(run_identity: str, context_sha256: str, record: NativeRecordEnvelope) -> str:
    preimage = {"schema": "exp425.native-trace.v1", "run_identity": run_identity, "dataset": record.dataset, "run_context_sha256": context_sha256, "train_identity_sha256": record.train_identity_sha256, "eligibility_sha256": record.eligibility_sha256, "processed_groups_sha256": record.processed_groups_sha256, "model_evidence_sha256": record.model_evidence_sha256, "frozen_train_contract_sha256": record.frozen_train_contract_sha256, "scoring_evidence_sha256": record.scoring_evidence_sha256, "result_evidence_sha256": record.result_evidence_sha256, "backend_audit_sha256": record.backend_audit_sha256, "device": "mps"}
    return sha256(canonical_json_bytes(preimage)).hexdigest()


def _native_record_body(context_sha256: str, record: NativeRecordEnvelope) -> dict[str, object]:
    return {"schema_version": "exp425.dataset.v1", "dataset": record.dataset, "origin": record.origin, "run_context_sha256": context_sha256, "result_payload": dict(record.result_payload), "group_evidence": dict(record.group_evidence), "trace_sha256": record.trace_sha256, "model_hashes": dict(record.model_hashes), "calibration_hashes": dict(record.calibration_hashes), "backend_audit_sha256": record.backend_audit_sha256, "elapsed_seconds": record.elapsed_seconds, "reason": record.reason}


def _import_record_body(context_sha256: str, record: ImportedRecordEnvelope) -> dict[str, object]:
    return {"schema_version": "exp425.imported-dataset.v1", "dataset": record.dataset, "origin": record.origin, "run_context_sha256": context_sha256, "result_payload": dict(record.result_payload), "group_evidence": {"source_group_evidence": dict(record.group_evidence.source_group_evidence), "source_group_evidence_sha256": record.group_evidence.source_group_evidence_sha256}, "source_root_sha256": record.source_root_sha256, "source_progress_sha256": record.source_progress_sha256, "source_record_sha256": record.source_record_sha256, "source_result_sha256": record.source_result_sha256, "compatibility_attestation_sha256": record.compatibility_attestation_sha256, "elapsed_seconds": record.elapsed_seconds, "reason": record.reason}


def _progress_bytes(expected: tuple[str, ...], digests: Mapping[str, str], current: str | None = None) -> bytes:
    if any(dataset not in expected for dataset in digests) or len(digests) != len(set(digests)):
        raise ValueError("progress contains an unknown dataset")
    body = {"schema": "exp425.progress.v1", "record_digests": dict(sorted(digests.items())), "completed_count": len(digests), "expected_count": len(expected), "current_dataset": current}
    return canonical_json_bytes(body | {"payload_sha256": sha256(canonical_json_bytes(body)).hexdigest()})


def _read_root_metadata(root: Path) -> tuple[dict[str, object], bytes, dict[str, object]]:
    if not root.is_dir() or root.is_symlink() or {item.name for item in root.iterdir()} != _ROOT_FILES:
        raise ValueError("checkpoint root inventory is invalid")
    provenance = _canonical_object(_read_file(root / "checkpoint_provenance.json"), "provenance")
    context = _read_file(root / "run_context.json")
    progress = _canonical_object(_read_file(root / "progress.json"), "progress")
    required = {"schema", "run_identity", "git_commit", "database_sha256", "lane_artifact_sha256", "frozen_config_sha256", "fold_contract_sha256", "model_contract_sha256", "backend_contract_sha256", "model_seeds", "perturbation_seeds", "runtime_versions", "expected_datasets", "predecessor_compatibility_sha256", "run_context_sha256"}
    if set(provenance) != required or provenance["schema"] != "exp425.checkpoint.v1" or sha256(context).hexdigest() != provenance["run_context_sha256"]:
        raise ValueError("checkpoint provenance is invalid")
    _context_from_payload(_canonical_object(context, "run context"))
    _validate_progress(provenance, progress)
    return provenance, context, progress


def _validate_progress(provenance: Mapping[str, object], progress: Mapping[str, object]) -> None:
    required = {"schema", "record_digests", "completed_count", "expected_count", "current_dataset", "payload_sha256"}
    if set(progress) != required or progress["schema"] != "exp425.progress.v1": raise ValueError("checkpoint progress is invalid")
    body = {key: value for key, value in progress.items() if key != "payload_sha256"}
    if sha256(canonical_json_bytes(body)).hexdigest() != progress["payload_sha256"]: raise ValueError("checkpoint progress digest is invalid")
    records = progress["record_digests"]
    if not isinstance(records, dict) or list(records) != sorted(records) or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 for key, value in records.items()): raise ValueError("checkpoint progress records are invalid")
    expected = provenance["expected_datasets"]
    if not isinstance(expected, list) or set(records) - set(expected) or progress["completed_count"] != len(records) or progress["expected_count"] != len(expected): raise ValueError("checkpoint progress counts are invalid")


def _validate_inventory(root: Path, provenance: Mapping[str, object], progress: Mapping[str, object], *, allow_intents: bool) -> None:
    for directory in ("datasets", "transactions", "import_batches"):
        path = root / directory
        if not path.is_dir() or path.is_symlink(): raise ValueError("checkpoint directory is invalid")
    names = {item.name for item in (root / "datasets").iterdir()}
    pending_native: dict[str, str] = {}
    if allow_intents:
        for path in (root / "transactions").glob("*.json"):
            intent = _canonical_object(_read_file(path), "intent")
            if intent.get("origin") == "exp425_native":
                dataset, digest = intent.get("dataset"), intent.get("record_payload_sha256")
                if not isinstance(dataset, str) or not isinstance(digest, str):
                    raise ValueError("native checkpoint intent is invalid")
                if dataset in pending_native:
                    raise ValueError("duplicate native checkpoint intent")
                pending_native[dataset] = digest
    expected_names = {record_filename(dataset) for dataset in progress["record_digests"]} | {record_filename(dataset) for dataset in pending_native}
    if names != expected_names: raise ValueError("checkpoint dataset inventory is invalid")
    for dataset, digest in progress["record_digests"].items():
        raw = _read_file(root / "datasets" / record_filename(dataset))
        record = _decode_record(raw, None)
        if record.dataset != dataset or _record_digest_from_raw(raw) != digest: raise ValueError("checkpoint dataset record is invalid")
    for dataset, digest in pending_native.items():
        path = root / "datasets" / record_filename(dataset)
        if path.exists() and _record_digest_from_raw(_read_file(path)) != digest:
            raise ValueError("checkpoint pending record is invalid")
    if not allow_intents and any((root / "transactions").iterdir()): raise ValueError("checkpoint intents are present")


def _decode_record(raw: bytes, context: FrozenRunContext | None) -> Exp425Record:
    payload = _canonical_object(raw, "record")
    _reject_forbidden_keys(payload)
    origin = payload.get("origin")
    digest = payload.pop("record_payload_sha256", None)
    if not isinstance(digest, str) or sha256(canonical_json_bytes(payload)).hexdigest() != digest: raise ValueError("checkpoint record digest is invalid")
    if origin == "exp425_native":
        required = {"schema_version", "dataset", "origin", "run_context_sha256", "result_payload", "group_evidence", "trace_sha256", "model_hashes", "calibration_hashes", "backend_audit_sha256", "elapsed_seconds", "reason"}
        if set(payload) != required or payload["schema_version"] != "exp425.dataset.v1": raise ValueError("native checkpoint record keys are invalid")
        return NativeRecordEnvelope(payload["dataset"], payload["result_payload"], payload["group_evidence"], "0" * 64, "0" * 64, "0" * 64, "0" * 64, "0" * 64, "0" * 64, "0" * 64, payload["model_hashes"], payload["calibration_hashes"], payload["backend_audit_sha256"], payload["trace_sha256"], digest, payload["elapsed_seconds"], payload["reason"], context=context)
    if origin == "exp424_import":
        evidence = payload.get("group_evidence")
        if not isinstance(evidence, dict): raise ValueError("import checkpoint evidence is invalid")
        return ImportedRecordEnvelope(payload["dataset"], payload["result_payload"], SourceProvenGroupEvidence(evidence.get("source_group_evidence"), evidence.get("source_group_evidence_sha256")), payload["source_root_sha256"], payload["source_progress_sha256"], payload["source_record_sha256"], payload["source_result_sha256"], payload["compatibility_attestation_sha256"], payload["elapsed_seconds"], payload["reason"], context=context)
    raise ValueError("checkpoint record origin is invalid")


def _record_digest_from_raw(raw: bytes) -> str:
    return str(_canonical_object(raw, "record").get("record_payload_sha256", ""))


def _canonical_object(raw: bytes, name: str) -> dict[str, object]:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        data: dict[str, object] = {}
        for key, value in pairs:
            if key in data: raise ValueError(f"duplicate JSON key in {name}")
            data[key] = value
        return data
    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate, parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(f"nonfinite JSON constant in {name}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint {name}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw: raise ValueError(f"noncanonical checkpoint {name}")
    return value


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        if set(value) & _FORBIDDEN_RECORD_KEYS: raise ValueError("checkpoint record embeds forbidden context")
        for nested in value.values(): _reject_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value: _reject_forbidden_keys(nested)
    elif isinstance(value, float) and value != value:
        raise ValueError("checkpoint record contains a nonfinite number")
    elif isinstance(value, float) and value in (float("inf"), float("-inf")):
        raise ValueError("checkpoint record contains a nonfinite number")


def _read_file(path: Path) -> bytes:
    descriptor = os.open(path, _OPEN_FLAGS)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode): raise ValueError("checkpoint path is not a regular file")
        chunks: list[bytes] = []
        while True:
            data = os.read(descriptor, 1024 * 1024)
            if not data: break
            chunks.append(data)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(descriptor, data); os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_replace(path: Path, data: bytes) -> None:
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex)
    _write_new(temporary, data)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _inode(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode): raise ValueError("checkpoint path is a symlink")
    return info.st_dev, info.st_ino


class _locked_root:
    def __init__(self, root: Path, *, exclusive: bool) -> None: self.root = root; self.exclusive = exclusive; self.descriptor: int | None = None
    def __enter__(self) -> "_locked_root":
        self.descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        fcntl.flock(self.descriptor, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
        return self
    def __exit__(self, *unused: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN); os.close(self.descriptor)


def _import_intent_requirements(root: Path) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for path in sorted((root / "transactions").glob("*.json")):
        intent = _canonical_object(_read_file(path), "intent")
        if intent.get("origin") == "exp424_import":
            batch, attestation = intent.get("batch_identity"), intent.get("attestation_sha256")
            if not isinstance(batch, str) or not isinstance(attestation, str): raise ValueError("import intent is invalid")
            result.append((batch, attestation))
    return tuple(result)
