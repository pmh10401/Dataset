"""Compact, provenance-bound, fail-closed local checkpoints for Exp425."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager, nullcontext
import ctypes
import errno
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Iterator, Mapping
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import _context_from_payload, _context_payload
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext

from . import capabilities
from .contracts import Exp425Record, Exp425RunProvenance, ImportedRecordEnvelope, NativeRecordEnvelope, SourceProvenGroupEvidence, thaw_payload


_ROOT_FILES = frozenset({"checkpoint_provenance.json", "run_context.json", "progress.json", "datasets", "transactions", "import_batches"})
_FORBIDDEN_RECORD_KEYS = frozenset({"context", "lanes", "datasets", "test_identity_manifest", "governance"})
_FORBIDDEN_RECORD_KEY_TOKENS = frozenset({"target", "groundtruth", "anomalylocation", "confusion", "precision", "recall", "f1", "familytestperformance"})
_KNOWN_RESULT_STATUSES = frozenset({"abstained", "scored", "failed"})
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_after_import_phase = lambda phase: None
_after_import_cleanup_phase = lambda phase: None
_after_native_phase = lambda phase: None


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
    inode: tuple[int, int] | None
    attestation_sha256: str
    nonce: str


class Exp425CheckpointStore:
    """A shared-context checkpoint root. Only private capability paths mutate it."""

    def __init__(self, root: Path, provenance: Exp425RunProvenance, context: FrozenRunContext, record_digests: Mapping[str, str], root_identity: tuple[int, int]) -> None:
        self.root = Path(root)
        self.provenance = provenance
        self.context = context
        self.expected_datasets = provenance.expected_datasets
        self.run_context_sha256 = provenance.run_context_sha256
        if self.run_context_sha256 is None:
            raise ValueError("checkpoint context digest is absent")
        self._record_digests = MappingProxyType(dict(record_digests))
        self._root_identity = root_identity
        self.import_revalidation_required: tuple[tuple[str, str], ...] = ()

    @contextmanager
    def _store_binding(self, *, exclusive: bool) -> Iterator["_locked_root"]:
        """Acquire this instance's root only if the original inode still owns its path."""
        with _locked_root(self.root, exclusive=exclusive) as binding:
            if binding.identity != self._root_identity:
                raise ValueError("checkpoint store root identity was replaced")
            yield binding

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
        root_identity: tuple[int, int] | None = None
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
                assert binding.identity is not None
                root_identity = binding.identity
        except BaseException:
            # A partially-created root remains fail-closed and is never adopted.
            raise
        assert root_identity is not None
        return cls(root, bound, context, {}, root_identity)

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
                assert binding.identity is not None
                store = cls(root, provenance.with_run_context_sha256(digest), context, progress["record_digests"], binding.identity)
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
        with self._store_binding(exclusive=False):
            return MappingProxyType(dict(self._record_digests))

    def load_record(self, dataset: str) -> Exp425Record:
        with self._store_binding(exclusive=False) as binding:
            if dataset not in self._record_digests:
                raise ValueError("dataset is not completed")
            datasets_fd, _ = _open_directory_at(binding, "datasets")
            try:
                raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
            finally:
                os.close(datasets_fd)
            binding.verify()
        record = _decode_record(raw, self.context, native_run_identity=self.provenance.run_identity, native_context_sha256=self.run_context_sha256)
        if record.dataset != dataset or _record_digest_from_raw(raw) != self._record_digests[dataset]:
            raise ValueError("checkpoint record digest does not match progress")
        return record

    def iter_records(self) -> Iterator[Exp425Record]:
        with self._store_binding(exclusive=False):
            pass
        for dataset in self.expected_datasets:
            if dataset in self._record_digests:
                yield self.load_record(dataset)

    def _commit_native(self, capability: object, record: NativeRecordEnvelope) -> str:
        with self._store_binding(exclusive=True) as binding:
            if not isinstance(record, NativeRecordEnvelope):
                raise ValueError("native checkpoint record is invalid")
            trace = _native_trace_sha256(self.provenance.run_identity, self.run_context_sha256, record)
            body = _native_record_body(self.run_context_sha256, record)
            digest = sha256(canonical_json_bytes(body)).hexdigest()
            if trace != record.trace_sha256 or digest != record.record_payload_sha256:
                raise ValueError("native record digest does not match its canonical preimage")
            _decode_record(canonical_json_bytes(body | {"record_payload_sha256": digest}), self.context, native_run_identity=self.provenance.run_identity, native_context_sha256=self.run_context_sha256)
            capabilities._consume(capability, kind="native", run_identity=self.provenance.run_identity, dataset=record.dataset, context_sha256=self.run_context_sha256, trace_sha256=trace, record_sha256=digest, backend_sha256=record.backend_audit_sha256)
            self._write_record(record, body, digest, intent_extra={
                "run_identity": self.provenance.run_identity,
                "run_context_sha256": self.run_context_sha256,
                "backend_contract_sha256": self.provenance.backend_contract_sha256,
                "backend_audit_sha256": record.backend_audit_sha256,
            }, binding=binding)
        return digest

    def _begin_import_batch(self, capability: object, batch_identity: str) -> object:
        with self._store_binding(exclusive=True) as binding:
            claims = capabilities._consume(capability, kind="import_begin", run_identity=self.provenance.run_identity, batch_identity=batch_identity)
            attestation = claims.get("attestation_sha256")
            nonce = claims.get("process_nonce")
            if not isinstance(attestation, str) or not isinstance(nonce, str):
                raise ValueError("import batch authority lacks live attestation")
            name = _batch_name(batch_identity)
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                cleanup = _read_cleanup_at(imports_fd, self.provenance.run_identity, self.run_context_sha256, batch_identity)
                if cleanup is not None:
                    if cleanup["source_attestation_sha256"] != attestation:
                        raise ValueError("import cleanup durable state does not match authority")
                    try:
                        info = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                        if not stat.S_ISDIR(info.st_mode):
                            raise ValueError("import batch cleanup entry is invalid")
                        identity = _cleanup_batch_inode(cleanup)
                        if (info.st_dev, info.st_ino) != identity:
                            raise ValueError("import batch directory was replaced before cleanup")
                    except FileNotFoundError:
                        identity = None
                    _verify_directory_at(binding, "import_batches", imports_identity)
                    return _ImportBatch(batch_identity, self.root / "import_batches" / name, identity, attestation, nonce)
                try:
                    os.mkdir(name, 0o700, dir_fd=imports_fd)
                    batch_fd: int | None = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                    try:
                        state = _batch_state(self.provenance.run_identity, self.run_context_sha256, batch_identity, attestation, {})
                        _write_new_at(batch_fd, "batch.json", canonical_json_bytes(state))
                    finally:
                        if batch_fd is not None: os.close(batch_fd)
                    os.fsync(imports_fd)
                except FileExistsError:
                    # A revalidated process resumes only an identical durable batch.
                    batch_fd = None
                    try:
                        batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                        state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                    finally:
                        if batch_fd is not None: os.close(batch_fd)
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
        _decode_record(canonical_json_bytes(payload | {"record_payload_sha256": digest}), self.context, native_context_sha256=self.run_context_sha256)
        with self._store_binding(exclusive=True) as binding:
            capabilities._consume(capability, kind="import_stage", run_identity=self.provenance.run_identity, batch_identity=batch.identity, dataset=record.dataset, source_sha256=record.source_record_sha256, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, record_sha256=digest)
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                cleanup = _read_cleanup_at(imports_fd, self.provenance.run_identity, self.run_context_sha256, batch.identity)
                if cleanup is not None:
                    staged_cleanup = cleanup["staged_record_digests"]
                    if cleanup["source_attestation_sha256"] != batch.attestation_sha256 or staged_cleanup.get(record.dataset) != digest:
                        raise ValueError("import cleanup durable state does not match staged record")
                    _verify_directory_at(binding, "import_batches", imports_identity)
                    return
                batch_fd: int | None = None
                try:
                    batch_fd = _open_batch_at(imports_fd, batch)
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
                    if batch_fd is not None: os.close(batch_fd)
            finally:
                os.close(imports_fd)

    def _promote_import_batch(self, capability: object, batch: object) -> None:
        if not isinstance(batch, _ImportBatch):
            raise ValueError("import batch is invalid")
        with self._store_binding(exclusive=True) as binding:
            imports_fd, _ = _open_directory_at(binding, "import_batches")
            try:
                cleanup = _read_cleanup_at(imports_fd, self.provenance.run_identity, self.run_context_sha256, batch.identity)
                if cleanup is None:
                    batch_fd: int | None = None
                    try:
                        batch_fd = _open_batch_at(imports_fd, batch)
                        state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                        staged = dict(state["staged_record_digests"])
                        if state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged):
                            raise ValueError("import batch durable state does not match authority")
                        raw_records = {dataset: _read_file(record_filename(dataset), dir_fd=batch_fd) for dataset in staged}
                    finally:
                        if batch_fd is not None: os.close(batch_fd)
                else:
                    if cleanup["source_attestation_sha256"] != batch.attestation_sha256:
                        raise ValueError("import cleanup durable state does not match authority")
                    staged = dict(cleanup["staged_record_digests"])
                    raw_records = {}
            finally:
                os.close(imports_fd)
            final_digest = _import_final_attestation(batch.identity, batch.attestation_sha256, staged)
            capabilities._consume(capability, kind="import_promote", run_identity=self.provenance.run_identity, batch_identity=batch.identity, attestation_sha256=batch.attestation_sha256, process_nonce=batch.nonce, final_attestation_sha256=final_digest)
            if cleanup is None:
                for dataset in sorted(staged):
                    record = _decode_record(raw_records[dataset], self.context)
                    body = _import_record_body(self.run_context_sha256, record)
                    self._promote_imported_record(record, body, sha256(canonical_json_bytes(body)).hexdigest(), batch.identity, final_digest, binding=binding)
            self._finalize_import_batch(batch, staged, final_digest, binding=binding)

    def _finalize_import_batch(self, batch: _ImportBatch, staged: Mapping[str, str], final_attestation: str, *, binding: "_locked_root | None" = None) -> None:
        if binding is None:
            with self._store_binding(exclusive=True) as owned:
                self._finalize_import_batch(batch, staged, final_attestation, binding=owned)
            return
        with nullcontext(binding):
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                cleanup = _read_cleanup_at(imports_fd, self.provenance.run_identity, self.run_context_sha256, batch.identity)
                expected_cleanup: dict[str, object] | None = None
                if cleanup is None:
                    batch_fd: int | None = None
                    try:
                        batch_fd = _open_batch_at(imports_fd, batch)
                        state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                        if state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged):
                            raise ValueError("import batch final state does not match")
                        expected_names = {"batch.json"} | {record_filename(dataset) for dataset in staged}
                        if set(os.listdir(batch_fd)) != expected_names:
                            raise ValueError("import batch has unknown cleanup entries")
                        if _import_final_attestation(batch.identity, batch.attestation_sha256, staged) != final_attestation or any(progress["record_digests"].get(dataset) != digest for dataset, digest in staged.items()):
                            raise ValueError("import batch promotion is incomplete")
                        if batch.inode is None:
                            raise ValueError("import batch directory is absent before cleanup")
                        expected_cleanup = _cleanup_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged, final_attestation, batch.inode)
                        _write_new_at(imports_fd, _cleanup_name(batch.identity), canonical_json_bytes(expected_cleanup))
                        _after_import_cleanup_phase("cleanup_tombstone_durable")
                    finally:
                        if batch_fd is not None: os.close(batch_fd)
                elif (
                    cleanup["source_attestation_sha256"] != batch.attestation_sha256
                    or cleanup["staged_record_digests"] != dict(sorted(staged.items()))
                    or cleanup["final_attestation_sha256"] != final_attestation
                ):
                    raise ValueError("import cleanup durable state does not match finalization")
                if any(progress["record_digests"].get(dataset) != digest for dataset, digest in staged.items()):
                    raise ValueError("import batch promotion is incomplete")
                try:
                    batch_fd = _open_batch_at(imports_fd, batch)
                except FileNotFoundError:
                    batch_fd = None
                if batch_fd is not None:
                    try:
                        expected_inode = _cleanup_batch_inode(cleanup) if cleanup is not None else batch.inode
                        opened = os.fstat(batch_fd)
                        if expected_inode is None or (opened.st_dev, opened.st_ino) != expected_inode:
                            raise ValueError("import batch directory was replaced before cleanup")
                        names = set(os.listdir(batch_fd))
                        expected_names = {"batch.json"} | {record_filename(dataset) for dataset in staged}
                        if not names <= expected_names:
                            raise ValueError("import batch has unknown cleanup entries")
                        for dataset in sorted(staged):
                            name = record_filename(dataset)
                            if name in names:
                                if _record_digest_from_raw(_read_file(name, dir_fd=batch_fd)) != staged[dataset]:
                                    raise ValueError("import batch staged record was replaced before cleanup")
                                os.unlink(name, dir_fd=batch_fd)
                                os.fsync(batch_fd)
                                _after_import_cleanup_phase("cleanup_staged_unlinked")
                        if "batch.json" in names:
                            state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                            if state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged):
                                raise ValueError("import batch metadata was replaced before cleanup")
                            os.unlink("batch.json", dir_fd=batch_fd)
                            os.fsync(batch_fd)
                            _after_import_cleanup_phase("cleanup_metadata_removed")
                        if os.listdir(batch_fd):
                            raise ValueError("import batch has unknown cleanup entries")
                    finally:
                        os.close(batch_fd)
                    _verify_directory_at(binding, "import_batches", imports_identity)
                    info = os.stat(_batch_name(batch.identity), dir_fd=imports_fd, follow_symlinks=False)
                    expected_inode = _cleanup_batch_inode(cleanup) if cleanup is not None else batch.inode
                    if expected_inode is None or (info.st_dev, info.st_ino) != expected_inode:
                        raise ValueError("import batch directory was replaced before cleanup")
                    os.rmdir(_batch_name(batch.identity), dir_fd=imports_fd)
                    _after_import_cleanup_phase("cleanup_directory_removed")
                    os.fsync(imports_fd)
                    _after_import_cleanup_phase("cleanup_directory_fsynced")
                _verify_directory_at(binding, "import_batches", imports_identity)
                os.unlink(_cleanup_name(batch.identity), dir_fd=imports_fd)
                os.fsync(imports_fd)
                binding.verify()
            finally:
                os.close(imports_fd)

    def _promote_imported_record(self, record: ImportedRecordEnvelope, body: dict[str, object], digest: str, batch_identity: str, final_attestation: str, *, binding: "_locked_root | None" = None) -> None:
        """Advance one imported record through durable A→E publication phases."""
        if record.dataset not in self.expected_datasets:
            raise ValueError("imported dataset is outside checkpoint universe")
        intent_name = "import-" + sha256((batch_identity + "\0" + record.dataset).encode()).hexdigest() + ".json"
        encoded = canonical_json_bytes(body | {"record_payload_sha256": digest})
        if binding is None:
            with self._store_binding(exclusive=True) as owned:
                self._promote_imported_record(record, body, digest, batch_identity, final_attestation, binding=owned)
            return
        with nullcontext(binding):
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
                    _atomic_replace_at(binding.descriptor, "progress.json", _progress_bytes(self.expected_datasets, digests))
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

    def _write_record(self, record: Exp425Record, body: dict[str, object], digest: str, *, intent_extra: Mapping[str, object], binding: "_locked_root | None" = None) -> None:
        if record.dataset not in self.expected_datasets or record.dataset in self._record_digests:
            raise ValueError("checkpoint dataset is invalid or already completed")
        if binding is None:
            with self._store_binding(exclusive=True) as owned:
                self._write_record(record, body, digest, intent_extra=intent_extra, binding=owned)
            return
        encoded = canonical_json_bytes(body | {"record_payload_sha256": digest})
        _reject_forbidden_keys(body)
        with nullcontext(binding):
            _, _, progress = _read_root_metadata(self.root, binding=binding)
            if record.dataset in progress["record_digests"]:
                raise ValueError("checkpoint dataset is already completed")
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
            try:
                intent_name = uuid.uuid4().hex + ".json"
                intent = {"schema": "exp425.intent.v1", "operation": "commit_native", "origin": record.origin, "dataset": record.dataset, "record_payload_sha256": digest, **dict(intent_extra)}
                _write_new_at(transactions_fd, intent_name, canonical_json_bytes(intent))
                _after_native_phase("intent_durable")
                _write_new_at(datasets_fd, record_filename(record.dataset), encoded)
                _after_native_phase("record_durable")
                digests = dict(progress["record_digests"]); digests[record.dataset] = digest
                assert binding.descriptor is not None
                _atomic_replace_at(binding.descriptor, "progress.json", _progress_bytes(self.expected_datasets, digests))
                _after_native_phase("progress_durable")
                _verify_directory_at(binding, "transactions", transactions_identity)
                os.unlink(intent_name, dir_fd=transactions_fd)
                os.fsync(transactions_fd)
                _after_native_phase("intent_removed")
                _verify_directory_at(binding, "datasets", datasets_identity)
                binding.verify()
                self._record_digests = MappingProxyType(digests)
            finally:
                os.close(transactions_fd); os.close(datasets_fd)

    def _reconcile_native_intents(self, *, binding: "_locked_root | None" = None) -> None:
        if binding is None:
            with self._store_binding(exclusive=True) as owned:
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
                    origin = intent.get("origin")
                    if origin == "exp424_import":
                        continue
                    dataset, digest, run_identity, context_sha256, backend_contract, backend_audit = _native_intent_claims(intent)
                    if (
                        dataset not in self.expected_datasets
                        or run_identity != self.provenance.run_identity
                        or context_sha256 != self.run_context_sha256
                        or backend_contract != self.provenance.backend_contract_sha256
                    ):
                        raise ValueError("native checkpoint intent does not match root provenance")
                    try:
                        raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
                    except FileNotFoundError:
                        _verify_directory_at(binding, "transactions", transactions_identity)
                        os.unlink(name, dir_fd=transactions_fd)
                        os.fsync(transactions_fd)
                        continue
                    if _record_digest_from_raw(raw) != digest:
                        raise ValueError("native checkpoint intent conflicts with record")
                    decoded = _decode_record(raw, self.context, native_run_identity=self.provenance.run_identity, native_context_sha256=self.run_context_sha256)
                    if decoded.dataset != dataset or decoded.backend_audit_sha256 != backend_audit:
                        raise ValueError("native checkpoint record conflicts with intent")
                    if dataset not in digests:
                        digests[dataset] = digest; changed = True
                    elif digests[dataset] != digest:
                        raise ValueError("native checkpoint intent conflicts with progress")
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
    return {"schema_version": "exp425.dataset.v1", "dataset": record.dataset, "origin": record.origin, "run_context_sha256": context_sha256, "result_payload": thaw_payload(record.result_payload), "group_evidence": thaw_payload(record.group_evidence), "trace_sha256": record.trace_sha256, "model_hashes": thaw_payload(record.model_hashes), "calibration_hashes": thaw_payload(record.calibration_hashes), "backend_audit_sha256": record.backend_audit_sha256, "elapsed_seconds": record.elapsed_seconds, "reason": record.reason}


def _native_intent_claims(intent: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    required = {"schema", "operation", "origin", "dataset", "record_payload_sha256", "run_identity", "run_context_sha256", "backend_contract_sha256", "backend_audit_sha256"}
    if (
        set(intent) != required
        or intent.get("schema") != "exp425.intent.v1"
        or intent.get("operation") != "commit_native"
        or intent.get("origin") != "exp425_native"
    ):
        raise ValueError("native checkpoint intent is invalid")
    dataset, digest, run_identity, context_sha256, backend_contract, backend_audit = (intent["dataset"], intent["record_payload_sha256"], intent["run_identity"], intent["run_context_sha256"], intent["backend_contract_sha256"], intent["backend_audit_sha256"])
    if not isinstance(dataset, str) or not dataset or not isinstance(run_identity, str) or not run_identity or any(not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in (digest, context_sha256, backend_contract, backend_audit)):
        raise ValueError("native checkpoint intent is invalid")
    return dataset, digest, run_identity, context_sha256, backend_contract, backend_audit


def _validate_native_record_binding(payload: Mapping[str, object], trace: Mapping[str, object], *, native_run_identity: str | None, native_context_sha256: str | None) -> None:
    if (
        trace.get("schema") != "exp425.native-trace.v1"
        or trace.get("dataset") != payload.get("dataset")
        or trace.get("run_context_sha256") != payload.get("run_context_sha256")
        or trace.get("device") != "mps"
        or trace.get("backend_audit_sha256") != payload.get("backend_audit_sha256")
        or native_run_identity is not None and trace.get("run_identity") != native_run_identity
        or native_context_sha256 is not None and payload.get("run_context_sha256") != native_context_sha256
    ):
        raise ValueError("native checkpoint trace literals are invalid")
    result_payload = payload.get("result_payload")
    group_evidence = payload.get("group_evidence")
    model_hashes = payload.get("model_hashes")
    calibration_hashes = payload.get("calibration_hashes")
    if not all(isinstance(value, dict) for value in (result_payload, group_evidence, model_hashes, calibration_hashes)):
        raise ValueError("native checkpoint evidence is invalid")
    assert isinstance(result_payload, dict)
    effective_result = result_payload.get("effective_result")
    if not isinstance(effective_result, dict):
        raise ValueError("native checkpoint result evidence is invalid")
    if effective_result.get("status") not in _KNOWN_RESULT_STATUSES:
        raise ValueError("native checkpoint result status is invalid")
    assert isinstance(group_evidence, dict) and isinstance(model_hashes, dict) and isinstance(calibration_hashes, dict)
    evidence_locations = (
        (effective_result, "result_evidence_sha256"),
        (group_evidence, "eligibility_sha256"),
        (group_evidence, "processed_groups_sha256"),
        (model_hashes, "train_identity_sha256"),
        (model_hashes, "model_evidence_sha256"),
        (model_hashes, "frozen_train_contract_sha256"),
        (calibration_hashes, "scoring_evidence_sha256"),
    )
    if any(container.get(key) != trace.get(key) for container, key in evidence_locations):
        raise ValueError("native checkpoint evidence does not match trace")


def _import_record_body(context_sha256: str, record: ImportedRecordEnvelope) -> dict[str, object]:
    return {"schema_version": "exp425.imported-dataset.v1", "dataset": record.dataset, "origin": record.origin, "run_context_sha256": context_sha256, "result_payload": thaw_payload(record.result_payload), "group_evidence": {"source_group_evidence": thaw_payload(record.group_evidence.source_group_evidence), "source_group_evidence_sha256": record.group_evidence.source_group_evidence_sha256}, "source_root_sha256": record.source_root_sha256, "source_progress_sha256": record.source_progress_sha256, "source_record_sha256": record.source_record_sha256, "source_result_sha256": record.source_result_sha256, "compatibility_attestation_sha256": record.compatibility_attestation_sha256, "elapsed_seconds": record.elapsed_seconds, "reason": record.reason}


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


def _cleanup_name(batch_identity: str) -> str:
    return "cleanup-" + _batch_name(batch_identity) + ".json"


def _cleanup_state(run_identity: str, context_sha256: str, batch_identity: str, attestation: str, staged: Mapping[str, str], final_attestation: str, batch_inode: tuple[int, int]) -> dict[str, object]:
    body = {"schema": "exp425.import-cleanup.v1", "run_identity": run_identity, "run_context_sha256": context_sha256, "batch_identity": batch_identity, "source_attestation_sha256": attestation, "staged_record_digests": dict(sorted(staged.items())), "final_attestation_sha256": final_attestation, "batch_directory_identity": list(batch_inode)}
    return body | {"payload_sha256": sha256(canonical_json_bytes(body)).hexdigest()}


def _import_final_attestation(batch_identity: str, attestation: str, staged: Mapping[str, str]) -> str:
    return sha256(canonical_json_bytes({
        "batch_identity": batch_identity,
        "record_digests": dict(sorted(staged.items())),
        "attestation_sha256": attestation,
    })).hexdigest()


def _read_cleanup_at(imports_fd: int, run_identity: str, context_sha256: str, batch_identity: str) -> dict[str, object] | None:
    try:
        cleanup = _canonical_object(_read_file(_cleanup_name(batch_identity), dir_fd=imports_fd), "import cleanup")
    except FileNotFoundError:
        return None
    attestation = cleanup.get("source_attestation_sha256")
    staged = cleanup.get("staged_record_digests")
    final = cleanup.get("final_attestation_sha256")
    inode = cleanup.get("batch_directory_identity")
    if not isinstance(attestation, str) or not isinstance(staged, dict) or not isinstance(final, str) or not isinstance(inode, list) or len(inode) != 2 or any(not isinstance(value, int) or value < 0 for value in inode):
        raise ValueError("import cleanup durable state is invalid")
    if cleanup != _cleanup_state(run_identity, context_sha256, batch_identity, attestation, staged, final, (inode[0], inode[1])):
        raise ValueError("import cleanup durable state is invalid")
    if final != _import_final_attestation(batch_identity, attestation, staged):
        raise ValueError("import cleanup final attestation is invalid")
    if any(not isinstance(dataset, str) or not isinstance(digest, str) or len(digest) != 64 for dataset, digest in staged.items()):
        raise ValueError("import cleanup staged digests are invalid")
    return cleanup


def _cleanup_batch_inode(cleanup: Mapping[str, object]) -> tuple[int, int]:
    inode = cleanup.get("batch_directory_identity")
    if not isinstance(inode, list) or len(inode) != 2 or any(not isinstance(value, int) or value < 0 for value in inode):
        raise ValueError("import cleanup durable state is invalid")
    return inode[0], inode[1]


def _open_batch_at(imports_fd: int, batch: _ImportBatch) -> int:
    descriptor = os.open(_batch_name(batch.identity), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
    info = os.fstat(descriptor)
    if batch.inode is not None and (info.st_dev, info.st_ino) != batch.inode:
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
    try:
        persisted_provenance = _provenance_from_payload(provenance)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint provenance is invalid") from error
    if persisted_provenance.to_payload() != provenance or sha256(context).hexdigest() != persisted_provenance.run_context_sha256:
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
    if not isinstance(records, dict) or list(records) != sorted(records) or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for key, value in records.items()): raise ValueError("checkpoint progress records are invalid")
    expected = provenance["expected_datasets"]
    current = progress["current_dataset"]
    if not isinstance(expected, list) or any(not isinstance(dataset, str) for dataset in expected) or type(progress["completed_count"]) is not int or type(progress["expected_count"]) is not int or set(records) - set(expected) or progress["completed_count"] != len(records) or progress["expected_count"] != len(expected) or progress["completed_count"] > progress["expected_count"] or current is not None and (not isinstance(current, str) or current not in set(expected) - set(records)):
        raise ValueError("checkpoint progress counts are invalid")


def _provenance_from_payload(payload: Mapping[str, object]) -> Exp425RunProvenance:
    required = {"schema", "run_identity", "git_commit", "database_sha256", "lane_artifact_sha256", "frozen_config_sha256", "fold_contract_sha256", "model_contract_sha256", "backend_contract_sha256", "model_seeds", "perturbation_seeds", "runtime_versions", "expected_datasets", "predecessor_compatibility_sha256", "run_context_sha256"}
    if set(payload) != required or payload.get("schema") != "exp425.checkpoint.v1":
        raise ValueError("checkpoint provenance is invalid")
    return Exp425RunProvenance(
        payload["run_identity"], payload["git_commit"], payload["database_sha256"], payload["lane_artifact_sha256"],
        payload["frozen_config_sha256"], payload["fold_contract_sha256"], payload["model_contract_sha256"], payload["backend_contract_sha256"],
        payload["model_seeds"], payload["perturbation_seeds"], payload["runtime_versions"], payload["expected_datasets"],
        payload["predecessor_compatibility_sha256"], payload["run_context_sha256"],
    )


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
                    dataset, digest, run_identity, context_sha256, backend_contract, _ = _native_intent_claims(intent)
                    if dataset not in provenance["expected_datasets"] or run_identity != provenance["run_identity"] or context_sha256 != provenance["run_context_sha256"] or backend_contract != provenance["backend_contract_sha256"]:
                        raise ValueError("native checkpoint intent does not match root provenance")
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
        for dataset, digest in pending_native.items():
            if dataset in progress["record_digests"] and progress["record_digests"][dataset] != digest:
                raise ValueError("native checkpoint intent conflicts with progress")
        for dataset, digest in pending_imported.items():
            if dataset in progress["record_digests"] and progress["record_digests"][dataset] != digest:
                raise ValueError("import checkpoint intent conflicts with progress")
        if not allow_intents and transaction_names: raise ValueError("checkpoint intents are present")
        _verify_directory_at(binding, "datasets", datasets_identity)
        _verify_directory_at(binding, "transactions", transactions_identity)
        _verify_directory_at(binding, "import_batches", imports_identity)
        binding.verify()
    finally:
        os.close(imports_fd); os.close(transactions_fd); os.close(datasets_fd)


def _decode_record(raw: bytes, context: FrozenRunContext | None, *, native_run_identity: str | None = None, native_context_sha256: str | None = None) -> Exp425Record:
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
        if not isinstance(payload["group_evidence"], dict):
            raise ValueError("native checkpoint group evidence is invalid")
        if not isinstance(payload["model_hashes"], dict):
            raise ValueError("native checkpoint model evidence is invalid")
        if not isinstance(payload["calibration_hashes"], dict):
            raise ValueError("native checkpoint calibration evidence is invalid")
        effective = result_payload["effective_result"]
        if not isinstance(effective, dict) or not {"status", "result_evidence_sha256"} <= set(effective):
            raise ValueError("native checkpoint effective result is invalid")
        if "candidate_coordinates" in effective and not isinstance(effective["candidate_coordinates"], list):
            raise ValueError("native checkpoint candidate coordinates are invalid")
        trace = result_payload["native_trace_body"]
        if not isinstance(trace, dict) or set(trace) != {"schema", "run_identity", "dataset", "run_context_sha256", "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256", "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256", "result_evidence_sha256", "backend_audit_sha256", "device"} or sha256(canonical_json_bytes(trace)).hexdigest() != payload["trace_sha256"]:
            raise ValueError("native checkpoint trace is invalid")
        _validate_native_record_binding(payload, trace, native_run_identity=native_run_identity, native_context_sha256=native_context_sha256)
        return NativeRecordEnvelope(payload["dataset"], result_payload, payload["group_evidence"], trace["train_identity_sha256"], trace["eligibility_sha256"], trace["processed_groups_sha256"], trace["model_evidence_sha256"], trace["frozen_train_contract_sha256"], trace["scoring_evidence_sha256"], trace["result_evidence_sha256"], payload["model_hashes"], payload["calibration_hashes"], payload["backend_audit_sha256"], payload["trace_sha256"], digest, payload["elapsed_seconds"], payload["reason"], context=context)
    if origin == "exp424_import":
        evidence = payload.get("group_evidence")
        required = {"schema_version", "dataset", "origin", "run_context_sha256", "result_payload", "group_evidence", "source_root_sha256", "source_progress_sha256", "source_record_sha256", "source_result_sha256", "compatibility_attestation_sha256", "elapsed_seconds", "reason"}
        if set(payload) != required or payload["schema_version"] != "exp425.imported-dataset.v1":
            raise ValueError("import checkpoint record keys are invalid")
        if native_context_sha256 is not None and payload["run_context_sha256"] != native_context_sha256:
            raise ValueError("import checkpoint context binding is invalid")
        if not isinstance(evidence, dict) or set(evidence) != {"source_group_evidence", "source_group_evidence_sha256"} or not isinstance(evidence["source_group_evidence"], dict):
            raise ValueError("import checkpoint evidence is invalid")
        if sha256(canonical_json_bytes(evidence["source_group_evidence"])).hexdigest() != evidence["source_group_evidence_sha256"]:
            raise ValueError("import checkpoint source evidence digest is invalid")
        result = payload["result_payload"]
        if not isinstance(result, dict) or result.get("status") not in _KNOWN_RESULT_STATUSES:
            raise ValueError("import checkpoint result status is invalid")
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
        for key in value:
            if not isinstance(key, str): raise ValueError("checkpoint record keys must be strings")
            normalized = "".join(character for character in key.lower() if character.isalnum())
            if normalized in {"context", "lanes", "datasets", "testidentitymanifest", "governance", "test"} or normalized.startswith("evaluation") or "testevaluation" in normalized or "testlabel" in normalized or any(token in normalized for token in _FORBIDDEN_RECORD_KEY_TOKENS):
                raise ValueError("checkpoint record embeds forbidden evaluation or context data")
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
        _rename_exclusive_at(parent_fd, root.name, destination.name)
        moved = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (moved.st_dev, moved.st_ino) != binding.identity:
            raise ValueError("quarantine move did not retain root identity")
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return destination


def _rename_exclusive_at(parent_fd: int, source: str, destination: str) -> None:
    """macOS-only atomic directory rename that refuses an occupied destination."""
    if sys.platform != "darwin":
        raise ValueError("atomic quarantine rename is unavailable")
    try:
        renameatx_np = ctypes.CDLL(None, use_errno=True).renameatx_np
    except AttributeError as error:
        raise ValueError("atomic quarantine rename is unavailable") from error
    renameatx_np.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameatx_np.restype = ctypes.c_int
    if renameatx_np(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 0x00000004) == 0:
        return
    failure = ctypes.get_errno()
    if failure in (errno.EEXIST, errno.ENOTEMPTY):
        raise ValueError("quarantine destination is unavailable")
    raise ValueError(f"atomic quarantine rename failed: {os.strerror(failure)}")


def _import_intent_requirements(root: Path, *, binding: _locked_root | None = None) -> tuple[tuple[str, str], ...]:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            return _import_intent_requirements(root, binding=owned)
    stored, _, _ = _read_root_metadata(root, binding=binding)
    transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
    imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
    try:
        batches: dict[str, str] = {}
        cleanup: dict[str, dict[str, object]] = {}
        directories: set[str] = set()
        for name in sorted(os.listdir(imports_fd)):
            if name.startswith("cleanup-") and name.endswith(".json"):
                batch_identity = name[len("cleanup-"):-len(".json")]
                if _cleanup_name(batch_identity) != name or batch_identity in cleanup:
                    raise ValueError("import cleanup inventory is invalid")
                cleanup[batch_identity] = _read_cleanup_at(imports_fd, str(stored["run_identity"]), str(stored["run_context_sha256"]), batch_identity) or {}
                if not cleanup[batch_identity]:
                    raise ValueError("import cleanup inventory is invalid")
                continue
            info = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or _batch_name(name) != name:
                raise ValueError("import batch inventory is invalid")
            directories.add(name)
        for name in sorted(directories):
            batch_fd: int | None = None
            try:
                batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                batch_identity = name
                pending_cleanup = cleanup.get(batch_identity)
                if pending_cleanup is not None:
                    staged = pending_cleanup["staged_record_digests"]
                    attestation = pending_cleanup["source_attestation_sha256"]
                    assert isinstance(staged, dict) and isinstance(attestation, str)
                    opened = os.fstat(batch_fd)
                    if (opened.st_dev, opened.st_ino) != _cleanup_batch_inode(pending_cleanup):
                        raise ValueError("import batch directory was replaced before cleanup")
                    names = set(os.listdir(batch_fd))
                    expected_names = {"batch.json"} | {record_filename(dataset) for dataset in staged}
                    if not names <= expected_names or "batch.json" not in names and names:
                        raise ValueError("import batch cleanup inventory is invalid")
                    if "batch.json" in names:
                        state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                        if state != _batch_state(str(stored["run_identity"]), str(stored["run_context_sha256"]), batch_identity, attestation, staged):
                            raise ValueError("import batch cleanup metadata is invalid")
                    for dataset, digest in staged.items():
                        filename = record_filename(dataset)
                        if filename in names and _record_digest_from_raw(_read_file(filename, dir_fd=batch_fd)) != digest:
                            raise ValueError("import batch staged record is invalid")
                    current = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != _cleanup_batch_inode(pending_cleanup):
                        raise ValueError("import batch directory was replaced before cleanup")
                    batches[batch_identity] = str(pending_cleanup["final_attestation_sha256"])
                    continue
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
                if batch_identity in batches:
                    raise ValueError("duplicate import batch identity")
                batches[batch_identity] = _import_final_attestation(batch_identity, attestation, staged)
            finally:
                if batch_fd is not None: os.close(batch_fd)
        for batch_identity, state in cleanup.items():
            if batch_identity not in batches:
                batches[batch_identity] = str(state["final_attestation_sha256"])
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
