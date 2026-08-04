"""Compact, provenance-bound, fail-closed local checkpoints for Exp425."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
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
_after_import_phase = lambda phase: None


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
            with _locked_root(root, exclusive=True) as binding:
                assert binding.descriptor is not None
                for name in ("datasets", "transactions", "import_batches"):
                    os.mkdir(name, 0o700, dir_fd=binding.descriptor)
                os.fsync(binding.descriptor)
                _write_new_at(binding.descriptor, "run_context.json", context_bytes)
                _write_new_at(binding.descriptor, "checkpoint_provenance.json", canonical_json_bytes(bound.to_payload()))
                _write_new_at(binding.descriptor, "progress.json", _progress_bytes(bound.expected_datasets, {}))
                binding.verify()
        except BaseException:
            # A partially-created root remains fail-closed and is never adopted.
            raise
        return cls(root, bound, context, {})

    @classmethod
    def open(cls, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext) -> "Exp425CheckpointStore":
        if not Path(root).exists():
            raise ValueError("Exp425 checkpoint root is absent")
        return cls._validated(Path(root), provenance, context, exclusive=True)

    @classmethod
    def inspect_read_only(cls, root: Path, *, expected_run_identity: str | None = None) -> Exp425CheckpointInspection:
        """Validate metadata in place without recovery, quarantine, or writes."""
        root = Path(root)
        with _locked_root(root, exclusive=False) as binding:
            provenance, context_bytes, progress = _read_root_metadata(root, binding=binding)
            if expected_run_identity is not None and provenance["run_identity"] != expected_run_identity:
                raise ValueError("checkpoint run identity does not match")
            _validate_inventory(root, provenance, progress, allow_intents=True, binding=binding)
            pending = _import_intent_requirements(root, binding=binding)
            return Exp425CheckpointInspection(
                provenance["run_identity"], tuple(provenance["expected_datasets"]), progress["completed_count"], progress["record_digests"], pending
            )

    @classmethod
    def _validated(cls, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext, *, exclusive: bool) -> "Exp425CheckpointStore":
        if not isinstance(provenance, Exp425RunProvenance) or not isinstance(context, FrozenRunContext):
            raise ValueError("Exp425 checkpoint inputs are invalid")
        with _locked_root(root, exclusive=exclusive) as binding:
            try:
                stored, context_bytes, progress = _read_root_metadata(root, binding=binding)
                expected_context = canonical_json_bytes(_context_payload(context))
                digest = sha256(expected_context).hexdigest()
                expected = provenance.with_run_context_sha256(digest).to_payload()
                if stored != expected or context_bytes != expected_context:
                    raise ValueError("Exp425 checkpoint provenance or context does not match")
                _validate_inventory(root, stored, progress, allow_intents=True, binding=binding)
                store = cls(root, provenance.with_run_context_sha256(digest), context, progress["record_digests"])
                if exclusive:
                    store._reconcile_native_intents(binding=binding)
                binding.verify()
                return store
            except ValueError as error:
                if exclusive:
                    try:
                        quarantined = _quarantine_owned(root, binding)
                    except ValueError:
                        raise error
                    raise ValueError(f"Exp425 checkpoint was quarantined at {quarantined}") from error
                raise

    def completed_record_digests(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._record_digests))

    def load_record(self, dataset: str) -> Exp425Record:
        if dataset not in self._record_digests:
            raise ValueError("dataset is not completed")
        with _locked_root(self.root, exclusive=False) as binding:
            datasets_fd, _ = _open_directory_at(binding, "datasets")
            try:
                raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
            finally:
                os.close(datasets_fd)
            binding.verify()
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
        name = _batch_name(batch_identity)
        with _locked_root(self.root, exclusive=True) as binding:
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                try:
                    os.mkdir(name, 0o700, dir_fd=imports_fd)
                    batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                    try:
                        state = _batch_state(self.provenance.run_identity, self.run_context_sha256, batch_identity, attestation, {})
                        _write_new_at(batch_fd, "batch.json", canonical_json_bytes(state))
                    finally:
                        os.close(batch_fd)
                    os.fsync(imports_fd)
                except FileExistsError:
                    # A revalidated process resumes only an identical durable batch.
                    batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                    try:
                        state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                    finally:
                        os.close(batch_fd)
                    if state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch_identity, attestation, state.get("staged_record_digests", {})):
                        raise ValueError("import batch durable state does not match authority")
                info = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                identity = (info.st_dev, info.st_ino)
                _verify_directory_at(binding, "import_batches", imports_identity)
            finally:
                os.close(imports_fd)
        directory = self.root / "import_batches" / name
        return _ImportBatch(batch_identity, directory, identity, attestation, nonce)

    def _stage_imported(self, capability: object, batch: object, record: ImportedRecordEnvelope) -> None:
        if not isinstance(batch, _ImportBatch) or not isinstance(record, ImportedRecordEnvelope):
            raise ValueError("import batch is invalid")
        payload = _import_record_body(self.run_context_sha256, record)
        digest = sha256(canonical_json_bytes(payload)).hexdigest()
        capabilities._consume(capability, kind="import_stage", run_identity=self.provenance.run_identity, batch_identity=batch.identity, dataset=record.dataset, source_sha256=record.source_record_sha256, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, record_sha256=digest)
        with _locked_root(self.root, exclusive=True) as binding:
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            batch_fd = _open_batch_at(imports_fd, batch)
            try:
                state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                staged = dict(state["staged_record_digests"])
                if record.dataset in staged and staged[record.dataset] != digest:
                    raise ValueError("import batch has conflicting staged dataset")
                if record.dataset not in staged:
                    _write_new_at(batch_fd, record_filename(record.dataset), canonical_json_bytes(payload | {"record_payload_sha256": digest}))
                    staged[record.dataset] = digest
                    _atomic_replace_at(batch_fd, "batch.json", canonical_json_bytes(_batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged)))
                _verify_directory_at(binding, "import_batches", imports_identity)
            finally:
                os.close(batch_fd); os.close(imports_fd)

    def _promote_import_batch(self, capability: object, batch: object) -> None:
        if not isinstance(batch, _ImportBatch):
            raise ValueError("import batch is invalid")
        with _locked_root(self.root, exclusive=False) as binding:
            imports_fd, _ = _open_directory_at(binding, "import_batches")
            batch_fd = _open_batch_at(imports_fd, batch)
            try:
                state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                staged = dict(state["staged_record_digests"])
                raw_records = {dataset: _read_file(record_filename(dataset), dir_fd=batch_fd) for dataset in staged}
            finally:
                os.close(batch_fd); os.close(imports_fd)
        final_digest = sha256(canonical_json_bytes({"batch_identity": batch.identity, "record_digests": dict(sorted(staged.items())), "attestation_sha256": batch.attestation_sha256})).hexdigest()
        capabilities._consume(capability, kind="import_promote", run_identity=self.provenance.run_identity, batch_identity=batch.identity, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, final_attestation_sha256=final_digest)
        for dataset in sorted(staged):
            record = _decode_record(raw_records[dataset], self.context)
            body = _import_record_body(self.run_context_sha256, record)
            self._promote_imported_record(record, body, sha256(canonical_json_bytes(body)).hexdigest(), batch.identity, final_digest)
        self._finalize_import_batch(batch, staged, final_digest)

    def _finalize_import_batch(self, batch: _ImportBatch, staged: Mapping[str, str], final_attestation: str) -> None:
        with _locked_root(self.root, exclusive=True) as binding:
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            batch_fd = _open_batch_at(imports_fd, batch)
            try:
                state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                if state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged):
                    raise ValueError("import batch final state does not match")
                expected_names = {"batch.json"} | {record_filename(dataset) for dataset in staged}
                if set(os.listdir(batch_fd)) != expected_names:
                    raise ValueError("import batch has unknown cleanup entries")
                expected_final = sha256(canonical_json_bytes({"batch_identity": batch.identity, "record_digests": dict(sorted(staged.items())), "attestation_sha256": batch.attestation_sha256})).hexdigest()
                if expected_final != final_attestation or any(progress["record_digests"].get(dataset) != digest for dataset, digest in staged.items()):
                    raise ValueError("import batch promotion is incomplete")
                for dataset in staged:
                    os.unlink(record_filename(dataset), dir_fd=batch_fd)
                os.unlink("batch.json", dir_fd=batch_fd)
                os.fsync(batch_fd)
            finally:
                os.close(batch_fd)
            _verify_directory_at(binding, "import_batches", imports_identity)
            info = os.stat(_batch_name(batch.identity), dir_fd=imports_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != batch.inode:
                raise ValueError("import batch directory was replaced before cleanup")
            os.rmdir(_batch_name(batch.identity), dir_fd=imports_fd)
            os.fsync(imports_fd)
            os.close(imports_fd)

    def _promote_imported_record(self, record: ImportedRecordEnvelope, body: dict[str, object], digest: str, batch_identity: str, final_attestation: str) -> None:
        """Advance one imported record through durable A→E publication phases."""
        if record.dataset not in self.expected_datasets:
            raise ValueError("imported dataset is outside checkpoint universe")
        intent_name = "import-" + sha256((batch_identity + "\0" + record.dataset).encode()).hexdigest() + ".json"
        encoded = canonical_json_bytes(body | {"record_payload_sha256": digest})
        with _locked_root(self.root, exclusive=True) as binding:
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
            try:
                intent = {"schema": "exp425.intent.v1", "origin": "exp424_import", "dataset": record.dataset, "record_payload_sha256": digest, "batch_identity": batch_identity, "final_attestation_sha256": final_attestation}
                try:
                    existing = _canonical_object(_read_file(intent_name, dir_fd=transactions_fd), "import intent")
                    if existing != intent:
                        raise ValueError("import intent conflicts with durable state")
                except FileNotFoundError:
                    _write_new_at(transactions_fd, intent_name, canonical_json_bytes(intent))
                    _after_import_phase("intent_durable")
                try:
                    raw = _read_file(record_filename(record.dataset), dir_fd=datasets_fd)
                    if _record_digest_from_raw(raw) != digest:
                        raise ValueError("import record conflicts with durable state")
                except FileNotFoundError:
                    _write_new_at(datasets_fd, record_filename(record.dataset), encoded)
                    _after_import_phase("record_durable")
                digests = dict(progress["record_digests"])
                if record.dataset not in digests:
                    digests[record.dataset] = digest
                    assert binding.descriptor is not None
                    _atomic_replace_at(binding.descriptor, "progress.json", _progress_bytes(self.expected_datasets, digests, record.dataset))
                    _after_import_phase("progress_durable")
                elif digests[record.dataset] != digest:
                    raise ValueError("import progress conflicts with durable state")
                _verify_directory_at(binding, "transactions", transactions_identity)
                os.unlink(intent_name, dir_fd=transactions_fd)
                os.fsync(transactions_fd)
                _after_import_phase("intent_removed")
                _verify_directory_at(binding, "datasets", datasets_identity)
                self._record_digests = MappingProxyType(digests)
            finally:
                os.close(transactions_fd); os.close(datasets_fd)

    def _write_record(self, record: Exp425Record, body: dict[str, object], digest: str, *, intent_extra: Mapping[str, object]) -> None:
        if record.dataset not in self.expected_datasets or record.dataset in self._record_digests:
            raise ValueError("checkpoint dataset is invalid or already completed")
        encoded = canonical_json_bytes(body | {"record_payload_sha256": digest})
        _reject_forbidden_keys(body)
        with _locked_root(self.root, exclusive=True) as binding:
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            if record.dataset in progress["record_digests"]:
                raise ValueError("checkpoint dataset is already completed")
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
            try:
                intent_name = uuid.uuid4().hex + ".json"
                intent = {"schema": "exp425.intent.v1", "origin": record.origin, "dataset": record.dataset, "record_payload_sha256": digest, **dict(intent_extra)}
                _write_new_at(transactions_fd, intent_name, canonical_json_bytes(intent))
                _write_new_at(datasets_fd, record_filename(record.dataset), encoded)
                digests = dict(progress["record_digests"]); digests[record.dataset] = digest
                assert binding.descriptor is not None
                _atomic_replace_at(binding.descriptor, "progress.json", _progress_bytes(self.expected_datasets, digests, record.dataset))
                _verify_directory_at(binding, "transactions", transactions_identity)
                os.unlink(intent_name, dir_fd=transactions_fd)
                os.fsync(transactions_fd)
                _verify_directory_at(binding, "datasets", datasets_identity)
                binding.verify()
                self._record_digests = MappingProxyType(digests)
            finally:
                os.close(transactions_fd); os.close(datasets_fd)

    def _reconcile_native_intents(self, *, binding: "_locked_root | None" = None) -> None:
        if binding is None:
            with _locked_root(self.root, exclusive=True) as owned:
                self._reconcile_native_intents(binding=owned)
            return
        with nullcontext(binding):
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            digests = dict(progress["record_digests"])
            changed = False
            remove_after_progress: list[str] = []
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
            try:
                for name in sorted(os.listdir(transactions_fd)):
                    if not name.endswith(".json"):
                        raise ValueError("checkpoint transaction inventory is invalid")
                    intent = _canonical_object(_read_file(name, dir_fd=transactions_fd), "intent")
                    origin, dataset, digest = intent.get("origin"), intent.get("dataset"), intent.get("record_payload_sha256")
                    if origin == "exp424_import":
                        continue
                    if origin != "exp425_native" or not isinstance(dataset, str) or not isinstance(digest, str):
                        raise ValueError("checkpoint intent is invalid")
                    try:
                        raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
                    except FileNotFoundError:
                        _verify_directory_at(binding, "transactions", transactions_identity)
                        os.unlink(name, dir_fd=transactions_fd); changed = True; continue
                    if _record_digest_from_raw(raw) != digest:
                        raise ValueError("native checkpoint intent conflicts with record")
                    decoded = _decode_record(raw, self.context)
                    if decoded.dataset != dataset:
                        raise ValueError("native checkpoint record conflicts with intent")
                    if dataset not in digests:
                        digests[dataset] = digest; changed = True
                    remove_after_progress.append(name)
                if changed:
                    assert binding.descriptor is not None
                    _atomic_replace_at(binding.descriptor, "progress.json", _progress_bytes(self.expected_datasets, digests))
                for name in remove_after_progress:
                    _verify_directory_at(binding, "transactions", transactions_identity)
                    os.unlink(name, dir_fd=transactions_fd)
                if remove_after_progress:
                    os.fsync(transactions_fd)
                _verify_directory_at(binding, "datasets", datasets_identity)
                _verify_directory_at(binding, "transactions", transactions_identity)
                binding.verify()
                self._record_digests = MappingProxyType(digests)
                self.import_revalidation_required = _import_intent_requirements(self.root, binding=binding)
            finally:
                os.close(transactions_fd); os.close(datasets_fd)


def _native_trace_sha256(run_identity: str, context_sha256: str, record: NativeRecordEnvelope) -> str:
    preimage = {"schema": "exp425.native-trace.v1", "run_identity": run_identity, "dataset": record.dataset, "run_context_sha256": context_sha256, "train_identity_sha256": record.train_identity_sha256, "eligibility_sha256": record.eligibility_sha256, "processed_groups_sha256": record.processed_groups_sha256, "model_evidence_sha256": record.model_evidence_sha256, "frozen_train_contract_sha256": record.frozen_train_contract_sha256, "scoring_evidence_sha256": record.scoring_evidence_sha256, "result_evidence_sha256": record.result_evidence_sha256, "backend_audit_sha256": record.backend_audit_sha256, "device": "mps"}
    payload = dict(record.result_payload)
    if set(payload) != {"effective_result", "native_trace_body"} or not isinstance(payload["effective_result"], Mapping) or not isinstance(payload["native_trace_body"], Mapping):
        raise ValueError("native result_payload schema is invalid")
    if dict(payload["native_trace_body"]) != preimage:
        raise ValueError("native trace body does not agree with record evidence")
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


def _batch_name(batch_identity: str) -> str:
    if not isinstance(batch_identity, str) or not batch_identity or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in batch_identity):
        raise ValueError("import batch identity is not a safe directory name")
    return batch_identity


def _batch_state(run_identity: str, context_sha256: str, batch_identity: str, attestation: str, staged: Mapping[str, str]) -> dict[str, object]:
    body = {"schema": "exp425.import-batch.v1", "run_identity": run_identity, "run_context_sha256": context_sha256, "batch_identity": batch_identity, "source_attestation_sha256": attestation, "staged_record_digests": dict(sorted(staged.items()))}
    return body | {"payload_sha256": sha256(canonical_json_bytes(body)).hexdigest()}


def _open_batch_at(imports_fd: int, batch: _ImportBatch) -> int:
    descriptor = os.open(_batch_name(batch.identity), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
    info = os.fstat(descriptor)
    if (info.st_dev, info.st_ino) != batch.inode:
        os.close(descriptor)
        raise ValueError("import batch directory was replaced")
    return descriptor


def _read_root_metadata(root: Path, *, binding: "_locked_root | None" = None) -> tuple[dict[str, object], bytes, dict[str, object]]:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            return _read_root_metadata(root, binding=owned)
    binding.verify()
    assert binding.descriptor is not None
    if set(os.listdir(binding.descriptor)) != _ROOT_FILES:
        raise ValueError("checkpoint root inventory is invalid")
    provenance = _canonical_object(_read_file("checkpoint_provenance.json", dir_fd=binding.descriptor), "provenance")
    context = _read_file("run_context.json", dir_fd=binding.descriptor)
    progress = _canonical_object(_read_file("progress.json", dir_fd=binding.descriptor), "progress")
    required = {"schema", "run_identity", "git_commit", "database_sha256", "lane_artifact_sha256", "frozen_config_sha256", "fold_contract_sha256", "model_contract_sha256", "backend_contract_sha256", "model_seeds", "perturbation_seeds", "runtime_versions", "expected_datasets", "predecessor_compatibility_sha256", "run_context_sha256"}
    if set(provenance) != required or provenance["schema"] != "exp425.checkpoint.v1" or sha256(context).hexdigest() != provenance["run_context_sha256"]:
        raise ValueError("checkpoint provenance is invalid")
    _context_from_payload(_canonical_object(context, "run context"))
    _validate_progress(provenance, progress)
    binding.verify()
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


def _validate_inventory(root: Path, provenance: Mapping[str, object], progress: Mapping[str, object], *, allow_intents: bool, binding: "_locked_root | None" = None) -> None:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            _validate_inventory(root, provenance, progress, allow_intents=allow_intents, binding=owned)
            return
    binding.verify()
    datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
    transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
    imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
    try:
        names = set(os.listdir(datasets_fd))
        transaction_names = sorted(os.listdir(transactions_fd))
        if any(not name.endswith(".json") for name in transaction_names):
            raise ValueError("checkpoint transaction inventory is invalid")
        pending_native: dict[str, str] = {}
        pending_imported: dict[str, str] = {}
        if allow_intents:
            for name in transaction_names:
                intent = _canonical_object(_read_file(name, dir_fd=transactions_fd), "intent")
                if intent.get("origin") == "exp425_native":
                    dataset, digest = intent.get("dataset"), intent.get("record_payload_sha256")
                    if not isinstance(dataset, str) or not isinstance(digest, str):
                        raise ValueError("native checkpoint intent is invalid")
                    if dataset in pending_native:
                        raise ValueError("duplicate native checkpoint intent")
                    pending_native[dataset] = digest
                elif intent.get("origin") == "exp424_import":
                    dataset, digest = intent.get("dataset"), intent.get("record_payload_sha256")
                    batch, final = intent.get("batch_identity"), intent.get("final_attestation_sha256")
                    if not all(isinstance(value, str) for value in (dataset, digest, batch, final)):
                        raise ValueError("import checkpoint intent is invalid")
                    if dataset in pending_imported:
                        raise ValueError("duplicate imported checkpoint intent")
                    pending_imported[dataset] = digest
        # A fsynced native intent may legitimately precede its record.  Only
        # records actually present need inventory entries at validation time.
        pending_present = {
            dataset for dataset in (*pending_native, *pending_imported)
            if record_filename(dataset) in names
        }
        expected_names = {record_filename(dataset) for dataset in progress["record_digests"]} | {record_filename(dataset) for dataset in pending_present}
        if names != expected_names: raise ValueError("checkpoint dataset inventory is invalid")
        for dataset, digest in progress["record_digests"].items():
            raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
            record = _decode_record(raw, None)
            if record.dataset != dataset or _record_digest_from_raw(raw) != digest: raise ValueError("checkpoint dataset record is invalid")
        for dataset, digest in {**pending_native, **pending_imported}.items():
            try:
                raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
            except FileNotFoundError:
                continue
            if _record_digest_from_raw(raw) != digest: raise ValueError("checkpoint pending record is invalid")
        if not allow_intents and transaction_names: raise ValueError("checkpoint intents are present")
        _verify_directory_at(binding, "datasets", datasets_identity)
        _verify_directory_at(binding, "transactions", transactions_identity)
        _verify_directory_at(binding, "import_batches", imports_identity)
        binding.verify()
    finally:
        os.close(imports_fd); os.close(transactions_fd); os.close(datasets_fd)


def _decode_record(raw: bytes, context: FrozenRunContext | None) -> Exp425Record:
    payload = _canonical_object(raw, "record")
    _reject_forbidden_keys(payload)
    origin = payload.get("origin")
    digest = payload.pop("record_payload_sha256", None)
    if not isinstance(digest, str) or sha256(canonical_json_bytes(payload)).hexdigest() != digest: raise ValueError("checkpoint record digest is invalid")
    if origin == "exp425_native":
        required = {"schema_version", "dataset", "origin", "run_context_sha256", "result_payload", "group_evidence", "trace_sha256", "model_hashes", "calibration_hashes", "backend_audit_sha256", "elapsed_seconds", "reason"}
        if set(payload) != required or payload["schema_version"] != "exp425.dataset.v1": raise ValueError("native checkpoint record keys are invalid")
        result_payload = payload["result_payload"]
        if not isinstance(result_payload, dict) or set(result_payload) != {"effective_result", "native_trace_body"}:
            raise ValueError("native checkpoint result payload is invalid")
        trace = result_payload["native_trace_body"]
        if not isinstance(trace, dict) or set(trace) != {"schema", "run_identity", "dataset", "run_context_sha256", "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256", "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256", "result_evidence_sha256", "backend_audit_sha256", "device"} or sha256(canonical_json_bytes(trace)).hexdigest() != payload["trace_sha256"]:
            raise ValueError("native checkpoint trace is invalid")
        return NativeRecordEnvelope(payload["dataset"], result_payload, payload["group_evidence"], trace["train_identity_sha256"], trace["eligibility_sha256"], trace["processed_groups_sha256"], trace["model_evidence_sha256"], trace["frozen_train_contract_sha256"], trace["scoring_evidence_sha256"], trace["result_evidence_sha256"], payload["model_hashes"], payload["calibration_hashes"], payload["backend_audit_sha256"], payload["trace_sha256"], digest, payload["elapsed_seconds"], payload["reason"], context=context)
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


def _read_file(path: Path | str, *, dir_fd: int | None = None) -> bytes:
    descriptor = os.open(path, _OPEN_FLAGS, dir_fd=dir_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode): raise ValueError("checkpoint path is not a regular file")
        chunks: list[bytes] = []
        while True:
            data = os.read(descriptor, 1024 * 1024)
            if not data: break
            chunks.append(data)
        result = b"".join(chunks)
        if dir_fd is not None:
            current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("checkpoint entry was replaced during read")
        return result
    finally:
        os.close(descriptor)


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        os.write(descriptor, data); os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_new_at(directory_fd: int, name: str, data: bytes) -> None:
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _atomic_replace(path: Path, data: bytes) -> None:
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex)
    _write_new(temporary, data)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_replace_at(directory_fd: int, name: str, data: bytes) -> None:
    temporary = "." + name + "." + uuid.uuid4().hex
    _write_new_at(directory_fd, temporary, data)
    os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    os.fsync(directory_fd)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _inode(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode): raise ValueError("checkpoint path is a symlink")
    return info.st_dev, info.st_ino


def _open_directory_at(binding: "_locked_root", name: str) -> tuple[int, tuple[int, int]]:
    binding.verify()
    assert binding.descriptor is not None
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=binding.descriptor)
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ValueError("checkpoint directory is invalid")
    identity = (info.st_dev, info.st_ino)
    _verify_directory_at(binding, name, identity)
    return descriptor, identity


def _verify_directory_at(binding: "_locked_root", name: str, identity: tuple[int, int]) -> None:
    binding.verify()
    assert binding.descriptor is not None
    info = os.stat(name, dir_fd=binding.descriptor, follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise ValueError("checkpoint directory was replaced")


class _locked_root:
    def __init__(self, root: Path, *, exclusive: bool) -> None:
        self.root = Path(root); self.exclusive = exclusive; self.descriptor: int | None = None
        self.identity: tuple[int, int] | None = None
    def __enter__(self) -> "_locked_root":
        self.descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(self.descriptor)
        self.identity = (info.st_dev, info.st_ino)
        fcntl.flock(self.descriptor, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
        self.verify()
        return self

    def verify(self) -> None:
        """Ensure the public pathname still resolves to this locked directory."""
        if self.identity is None:
            raise ValueError("checkpoint root lock is not live")
        try:
            info = os.stat(self.root, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("checkpoint root pathname was replaced") from error
        if stat.S_ISLNK(info.st_mode) or (info.st_dev, info.st_ino) != self.identity:
            raise ValueError("checkpoint root pathname was replaced")
    def __exit__(self, *unused: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN); os.close(self.descriptor)


def _quarantine_owned(root: Path, binding: _locked_root, *, destination: Path | None = None) -> Path:
    """Move only the exact locked root inode to an unused sibling quarantine."""
    binding.verify()
    root = Path(root)
    if root != binding.root:
        raise ValueError("quarantine root does not match locked root")
    if destination is None:
        destination = root.with_name(root.name + ".quarantine-" + uuid.uuid4().hex)
    destination = Path(destination)
    if destination.parent != root.parent or destination.exists() or destination.is_symlink():
        raise ValueError("quarantine destination is unavailable")
    parent_fd = os.open(root.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        # The check is intentionally adjacent to the descriptor-relative rename.
        binding.verify()
        source = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        if (source.st_dev, source.st_ino) != binding.identity:
            raise ValueError("checkpoint root pathname was replaced")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("quarantine destination is unavailable")
        os.rename(root.name, destination.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        moved = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != binding.identity:
            raise ValueError("quarantine move did not retain root identity")
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return destination


def _import_intent_requirements(root: Path, *, binding: _locked_root | None = None) -> tuple[tuple[str, str], ...]:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            return _import_intent_requirements(root, binding=owned)
    stored, _, _ = _read_root_metadata(root, binding=binding)
    transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
    imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
    try:
        batches: dict[str, str] = {}
        for name in sorted(os.listdir(imports_fd)):
            batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
            try:
                state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                batch_identity = state.get("batch_identity")
                attestation = state.get("source_attestation_sha256")
                staged = state.get("staged_record_digests")
                if not isinstance(batch_identity, str) or _batch_name(batch_identity) != name or not isinstance(attestation, str) or not isinstance(staged, dict):
                    raise ValueError("import batch is invalid")
                if state != _batch_state(str(stored["run_identity"]), str(stored["run_context_sha256"]), batch_identity, attestation, staged):
                    raise ValueError("import batch durable state is invalid")
                if set(os.listdir(batch_fd)) != {"batch.json"} | {record_filename(dataset) for dataset in staged}:
                    raise ValueError("import batch inventory is invalid")
                for dataset, digest in staged.items():
                    if not isinstance(dataset, str) or not isinstance(digest, str) or len(digest) != 64:
                        raise ValueError("import batch staged digests are invalid")
                    raw = _read_file(record_filename(dataset), dir_fd=batch_fd)
                    if _record_digest_from_raw(raw) != digest:
                        raise ValueError("import batch staged record is invalid")
                current = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                opened = os.fstat(batch_fd)
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ValueError("import batch directory was replaced")
                batches[batch_identity] = sha256(canonical_json_bytes({
                    "batch_identity": batch_identity,
                    "record_digests": dict(sorted(staged.items())),
                    "attestation_sha256": attestation,
                })).hexdigest()
            finally:
                os.close(batch_fd)
        result: set[tuple[str, str]] = set()
        for name in sorted(os.listdir(transactions_fd)):
            if not name.endswith(".json"):
                raise ValueError("checkpoint transaction inventory is invalid")
            intent = _canonical_object(_read_file(name, dir_fd=transactions_fd), "intent")
            if intent.get("origin") == "exp424_import":
                batch, final = intent.get("batch_identity"), intent.get("final_attestation_sha256")
                if not isinstance(batch, str) or not isinstance(final, str) or batches.get(batch) != final:
                    raise ValueError("import intent is invalid")
                result.add((batch, final))
        result.update(batches.items())
        _verify_directory_at(binding, "transactions", transactions_identity)
        _verify_directory_at(binding, "import_batches", imports_identity)
        return tuple(sorted(result))
    finally:
        os.close(imports_fd)
        os.close(transactions_fd)
