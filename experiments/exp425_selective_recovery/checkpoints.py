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
    record_summaries: tuple[tuple[str, str, str, str | None, float], ...] = ()
    provenance_payload: Mapping[str, object] = MappingProxyType({})
    run_context_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_digests", MappingProxyType(dict(self.record_digests)))
        object.__setattr__(self, "provenance_payload", MappingProxyType(dict(self.provenance_payload)))


@dataclass(frozen=True)
class _ImportBatch:
    identity: str
    directory: Path
    inode: tuple[int, int] | None
    attestation_sha256: str
    nonce: str


@dataclass(frozen=True)
class _ImportBatchQuarantine:
    batch_identity: str
    source_attestation_sha256: str
    reason_sha256: str
    directory: Path
    identity: tuple[int, int]


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
            _validate_inventory(root, provenance, progress, allow_intents=True, full_records=True, binding=binding)
            pending = _import_intent_requirements(root, binding=binding)
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            try:
                summaries: list[tuple[str, str, str, str | None, float]] = []
                for dataset in provenance["expected_datasets"]:
                    digest = progress["record_digests"].get(dataset)
                    if digest is None:
                        continue
                    record = _decode_record(
                        _read_file(record_filename(dataset), dir_fd=datasets_fd), None,
                        native_run_identity=provenance["run_identity"],
                        native_context_sha256=provenance["run_context_sha256"],
                    )
                    effective = thaw_payload(record.result_payload)
                    if record.origin == "exp425_native":
                        effective = effective.get("effective_result") if isinstance(effective, Mapping) else None
                    if not isinstance(effective, Mapping) or effective.get("status") not in _KNOWN_RESULT_STATUSES:
                        raise ValueError("checkpoint compact effective result is invalid")
                    summaries.append((dataset, record.origin, str(effective["status"]), record.reason, record.elapsed_seconds))
                _verify_directory_at(binding, "datasets", datasets_identity)
            finally:
                os.close(datasets_fd)
            return Exp425CheckpointInspection(
                provenance["run_identity"], tuple(provenance["expected_datasets"]), progress["completed_count"], progress["record_digests"], pending, tuple(summaries), provenance, provenance["run_context_sha256"]
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
            # Validate all cleanup representations before deciding whether a
            # second batch may start; a moved cleanup-owned directory is still
            # durable work for its original logical batch.
            _import_intent_requirements(self.root, binding=binding)
            claims = capabilities._consume(capability, kind="import_begin", run_identity=self.provenance.run_identity, batch_identity=batch_identity)
            attestation = claims.get("attestation_sha256")
            nonce = claims.get("process_nonce")
            if not isinstance(attestation, str) or not isinstance(nonce, str):
                raise ValueError("import batch authority lacks live attestation")
            name = _batch_name(batch_identity)
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                active: set[str] = set()
                cleanup_states: dict[str, dict[str, object]] = {}
                for entry in os.listdir(imports_fd):
                    if entry.startswith(".quarantined-"):
                        continue
                    if entry.startswith("cleanup-") and entry.endswith(".json"):
                        cleanup_identity = entry[len("cleanup-"):-len(".json")]
                        if _cleanup_name(cleanup_identity) != entry or cleanup_identity in cleanup_states:
                            raise ValueError("import cleanup inventory is invalid")
                        cleanup = _read_cleanup_at(imports_fd, self.provenance.run_identity, self.run_context_sha256, cleanup_identity)
                        if cleanup is None:
                            raise ValueError("import cleanup inventory is invalid")
                        cleanup_states[cleanup_identity] = cleanup
                        active.add(cleanup_identity)
                        continue
                    if entry.startswith(".cleanup-owned-"):
                        continue
                    info = os.stat(entry, dir_fd=imports_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(info.st_mode) or _batch_name(entry) != entry:
                        raise ValueError("import batch inventory is invalid")
                    active.add(entry)
                if active and active != {name}:
                    raise ValueError("a different import batch is already active")
                cleanup = cleanup_states.get(batch_identity)
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

    def _quarantine_import_batch(self, batch: object, *, final_attestation_sha256: str | None, reason_sha256: str) -> _ImportBatchQuarantine:
        """Move only one verified owned staging directory aside after an aborted import."""
        if not isinstance(batch, _ImportBatch) or not _is_lower_sha256(reason_sha256):
            raise ValueError("import batch quarantine inputs are invalid")
        with self._store_binding(exclusive=True) as binding:
            imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
            try:
                active = {name for name in os.listdir(imports_fd) if not name.startswith(".quarantined-") and not name.startswith("cleanup-")}
                if active != {_batch_name(batch.identity)}:
                    raise ValueError("import batch quarantine requires one exact active batch")
                batch_fd = _open_batch_at(imports_fd, batch)
                try:
                    state = _canonical_object(_read_file("batch.json", dir_fd=batch_fd), "import batch")
                    staged = state.get("staged_record_digests")
                    if not isinstance(staged, dict) or state != _batch_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged):
                        raise ValueError("import batch quarantine state is invalid")
                    expected_names = {"batch.json"} | {record_filename(dataset) for dataset in staged}
                    if set(os.listdir(batch_fd)) != expected_names:
                        raise ValueError("import batch quarantine inventory is invalid")
                    for dataset, digest in staged.items():
                        if not isinstance(dataset, str) or _record_digest_from_raw(_read_file(record_filename(dataset), dir_fd=batch_fd)) != digest:
                            raise ValueError("import batch quarantine staged record is invalid")
                    computed = _import_final_attestation(batch.identity, batch.attestation_sha256, staged)
                    if final_attestation_sha256 is not None and final_attestation_sha256 != computed:
                        raise ValueError("import batch quarantine attestation is invalid")
                    opened = os.fstat(batch_fd)
                finally:
                    os.close(batch_fd)
                if batch.inode is None or (opened.st_dev, opened.st_ino) != batch.inode:
                    raise ValueError("import batch quarantine directory was replaced")
                source_name = _batch_name(batch.identity)
                path_state = os.stat(source_name, dir_fd=imports_fd, follow_symlinks=False)
                if (path_state.st_dev, path_state.st_ino) != batch.inode:
                    raise ValueError("import batch quarantine directory was replaced")
                destination = ".quarantined-" + sha256(canonical_json_bytes({"schema": "exp425.import-quarantine.v1", "batch_identity": batch.identity, "inode": list(batch.inode), "reason_sha256": reason_sha256})).hexdigest()
                _rename_exclusive_at(imports_fd, source_name, destination)
                moved = os.stat(destination, dir_fd=imports_fd, follow_symlinks=False)
                if (moved.st_dev, moved.st_ino) != batch.inode:
                    raise ValueError("import batch quarantine move did not retain identity")
                quarantined_fd = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                try:
                    _write_new_at(quarantined_fd, "quarantine.json", canonical_json_bytes(_quarantine_state(self.provenance.run_identity, self.run_context_sha256, batch, reason_sha256, computed)))
                    os.fsync(quarantined_fd)
                finally:
                    os.close(quarantined_fd)
                os.fsync(imports_fd)
                _verify_directory_at(binding, "import_batches", imports_identity)
                return _ImportBatchQuarantine(batch.identity, batch.attestation_sha256, reason_sha256, self.root / "import_batches" / destination, batch.inode)
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
                        entries = _cleanup_entries_at(imports_fd, batch_fd, batch.identity, staged, batch.inode)
                        expected_cleanup = _cleanup_state(self.provenance.run_identity, self.run_context_sha256, batch.identity, batch.attestation_sha256, staged, final_attestation, entries)
                        _write_new_at(imports_fd, _cleanup_name(batch.identity), canonical_json_bytes(expected_cleanup))
                        cleanup = expected_cleanup
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
                        entries = cleanup["entries"]
                        assert isinstance(entries, dict)
                        _validate_cleanup_plan_at(batch_fd, cleanup, imports_fd=imports_fd)
                        expected_names = {entry[name_key] for entry in entries.values() if isinstance(entry, dict) and entry["entry_type"] == "file" for name_key in ("original_name", "isolated_name")}
                        if not names <= expected_names:
                            raise ValueError("import batch has unknown cleanup entries")
                        def verify_batch_parent() -> None:
                            _verify_directory_at(binding, "import_batches", imports_identity)
                            binding.verify()

                        for dataset in sorted(staged):
                            name = record_filename(dataset)
                            cleanup = _advance_cleanup_entry_at(batch_fd, imports_fd, cleanup, "staged:" + dataset, phase_label="staged", verify_parent=verify_batch_parent)
                            _after_import_cleanup_phase("cleanup_staged_unlinked")
                        cleanup = _advance_cleanup_entry_at(batch_fd, imports_fd, cleanup, "batch.json", phase_label="metadata", verify_parent=verify_batch_parent)
                        _after_import_cleanup_phase("cleanup_metadata_removed")
                        if os.listdir(batch_fd):
                            raise ValueError("import batch has unknown cleanup entries")
                    finally:
                        os.close(batch_fd)
                def verify_imports_parent() -> None:
                    _verify_directory_at(binding, "import_batches", imports_identity)
                    binding.verify()

                cleanup = _advance_cleanup_entry_at(imports_fd, imports_fd, cleanup, "batch-directory", phase_label="directory", verify_parent=verify_imports_parent)
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
            stored, _, progress = _read_root_metadata(self.root, binding=binding)
            _validate_inventory(self.root, stored, progress, allow_intents=True, binding=binding)
            digests = dict(progress["record_digests"])
            changed = False
            remove_after_progress: list[str] = []
            datasets_fd, datasets_identity = _open_directory_at(binding, "datasets")
            transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
            try:
                # Complete the read-only preflight first.  In particular, a later
                # malformed import must never authorize removal of an earlier
                # record-absent native intent.
                pending_imports = _import_intent_requirements(self.root, binding=binding)
                native_actions: list[tuple[str, str, str | None, str | None]] = []
                for name in sorted(os.listdir(transactions_fd)):
                    if not name.endswith(".json"):
                        raise ValueError("checkpoint transaction inventory is invalid")
                    intent = _canonical_object(_read_file(name, dir_fd=transactions_fd), "intent")
                    origin = intent.get("origin")
                    if origin == "exp424_import":
                        dataset, _, _, _ = _import_intent_claims(intent)
                        if dataset not in self.expected_datasets:
                            raise ValueError("import checkpoint intent does not match root provenance")
                        continue
                    if origin != "exp425_native":
                        raise ValueError("checkpoint intent origin is invalid")
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
                        if dataset in digests:
                            raise ValueError("native checkpoint intent conflicts with progress")
                        native_actions.append((name, dataset, None, None))
                        continue
                    if _record_digest_from_raw(raw) != digest:
                        raise ValueError("native checkpoint intent conflicts with record")
                    decoded = _decode_record(raw, self.context, native_run_identity=self.provenance.run_identity, native_context_sha256=self.run_context_sha256)
                    if decoded.dataset != dataset or decoded.backend_audit_sha256 != backend_audit:
                        raise ValueError("native checkpoint record conflicts with intent")
                    if dataset in digests and digests[dataset] != digest:
                        raise ValueError("native checkpoint intent conflicts with progress")
                    native_actions.append((name, dataset, digest, backend_audit))

                # Only this second pass changes durable state, after every intent,
                # root claim, and durable record state has been accepted.
                for name, dataset, digest, _ in native_actions:
                    if digest is None:
                        _verify_directory_at(binding, "transactions", transactions_identity)
                        os.unlink(name, dir_fd=transactions_fd)
                        os.fsync(transactions_fd)
                        continue
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
                self.import_revalidation_required = pending_imports
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


def _import_intent_claims(intent: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Decode the complete durable import-intent schema before reconciliation."""
    required = {"schema", "origin", "dataset", "record_payload_sha256", "batch_identity", "final_attestation_sha256"}
    if set(intent) != required or intent.get("schema") != "exp425.intent.v1" or intent.get("origin") != "exp424_import":
        raise ValueError("import checkpoint intent is invalid")
    dataset, digest, batch, final = (intent["dataset"], intent["record_payload_sha256"], intent["batch_identity"], intent["final_attestation_sha256"])
    if (
        not isinstance(dataset, str) or not dataset
        or not isinstance(batch, str) or _batch_name(batch) != batch
        or any(not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in (digest, final))
    ):
        raise ValueError("import checkpoint intent is invalid")
    return dataset, digest, batch, final


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
    contract = model_hashes.get("frozen_train_contract")
    if contract is not None and (
        not isinstance(contract, dict)
        or set(contract) != {"dataset", "groups", "backend", "train_identity_sha256", "near_repeat_status"}
        or contract.get("dataset") != payload.get("dataset")
        or contract.get("train_identity_sha256") != trace.get("train_identity_sha256")
        or sha256(canonical_json_bytes(contract)).hexdigest() != trace.get("frozen_train_contract_sha256")
    ):
        raise ValueError("native checkpoint frozen contract binding is invalid")
    if group_evidence.get("schema") == "exp425.runtime-group-evidence.v1":
        _validate_runtime_native_evidence(payload, trace, effective_result, group_evidence, model_hashes, calibration_hashes)
        return
    model_evidence = model_hashes.get("model_evidence")
    scoring_evidence = calibration_hashes.get("scoring_evidence")
    if model_evidence is None and scoring_evidence is None:
        return
    if not isinstance(model_evidence, dict) or not isinstance(scoring_evidence, dict) or sha256(canonical_json_bytes(model_evidence)).hexdigest() != trace.get("model_evidence_sha256") or sha256(canonical_json_bytes(scoring_evidence)).hexdigest() != trace.get("scoring_evidence_sha256"):
        raise ValueError("native checkpoint nested evidence digest is invalid")
    model_final = model_evidence.get("final_model_hashes")
    scoring_final = scoring_evidence.get("final_model_hashes")
    if model_evidence.get("status") == "captured":
        if not isinstance(model_final, dict) or not isinstance(scoring_final, dict) or model_final != scoring_final:
            raise ValueError("native checkpoint scoring final models do not match TRAIN")
    elif model_evidence.get("status") == "not_applicable":
        if scoring_evidence.get("status") != "not_applicable" or model_final != {}:
            raise ValueError("native checkpoint no-model evidence is inconsistent")
    else:
        raise ValueError("native checkpoint model evidence status is invalid")
def _validate_runtime_native_evidence(
    payload: Mapping[str, object], trace: Mapping[str, object], effective: Mapping[str, object],
    group: Mapping[str, object], model_hashes: Mapping[str, object], calibration_hashes: Mapping[str, object],
) -> None:
    """Validate the explicit compact schema emitted only by runtime._native_record.

    This deliberately recomputes each nested canonical preimage.  The older
    compact envelope remains extensible; it has no runtime schema marker and is
    therefore not mistaken for a complete native execution attestation.
    """
    def digest(value: object) -> str:
        return sha256(canonical_json_bytes(value)).hexdigest()

    def sha(value: object) -> bool:
        return _is_lower_sha256(value)

    if set(group) != {"schema", "eligibility_sha256", "processed_groups_sha256", "eligibility", "processed"}:
        raise ValueError("runtime-native group evidence keys are invalid")
    eligibility, processed = group.get("eligibility"), group.get("processed")
    if not isinstance(eligibility, dict) or set(eligibility) != {"payload", "sha256"} or not isinstance(eligibility.get("payload"), dict):
        raise ValueError("runtime-native eligibility evidence is invalid")
    if eligibility.get("sha256") != digest(eligibility["payload"]) or eligibility.get("sha256") != group.get("eligibility_sha256") or eligibility.get("sha256") != trace.get("eligibility_sha256"):
        raise ValueError("runtime-native eligibility digest is invalid")
    if not isinstance(processed, dict) or set(processed) != {"manifest", "manifest_sha256", "identity_sha256"} or not isinstance(processed.get("manifest"), dict):
        raise ValueError("runtime-native processed evidence is invalid")
    manifest = processed["manifest"]
    if set(manifest) != {"schema", "groups"} or manifest.get("schema") != "exp425.processed-manifest.v1" or not isinstance(manifest.get("groups"), list):
        raise ValueError("runtime-native processed manifest is invalid")
    groups = manifest["groups"]
    if any(not isinstance(item, dict) or set(item) != {"length", "status", "series_count", "instance_indices"} or type(item["length"]) is not int or type(item["series_count"]) is not int or not isinstance(item["status"], str) or not isinstance(item["instance_indices"], list) or any(type(index) is not int for index in item["instance_indices"]) for item in groups) or groups != sorted(groups, key=lambda item: item["length"]) or len({item["length"] for item in groups}) != len(groups):
        raise ValueError("runtime-native processed manifest groups are invalid")
    if processed.get("manifest_sha256") != digest(manifest) or not sha(processed.get("identity_sha256")) or processed.get("identity_sha256") != group.get("processed_groups_sha256") or processed.get("identity_sha256") != trace.get("processed_groups_sha256"):
        raise ValueError("runtime-native processed identity is invalid")

    contract = model_hashes.get("frozen_train_contract")
    if not isinstance(contract, dict) or set(contract) != {"dataset", "groups", "backend", "train_identity_sha256", "near_repeat_status"} or contract.get("dataset") != payload.get("dataset") or contract.get("near_repeat_status") != "not_available" or not isinstance(contract.get("groups"), list) or not isinstance(contract.get("backend"), dict):
        raise ValueError("runtime-native frozen contract is invalid")
    contract_groups = contract["groups"]
    required_group = {"length", "series_count", "status", "oof_candidate_rate", "wilson_upper", "pooled_synthetic_sensitivity", "per_type_synthetic_sensitivity", "seed_agreement", "finite", "backend_fallback"}
    required_types = {"level_shift", "gradual_drift", "amplitude_scaling", "period_change"}
    if any(not isinstance(item, dict) or set(item) != required_group or type(item["length"]) is not int or type(item["series_count"]) is not int or not isinstance(item["status"], str) or set(item["per_type_synthetic_sensitivity"]) != required_types or any(value is not None and type(value) not in {int, float} for value in item["per_type_synthetic_sensitivity"].values()) or any(item[name] is not None and type(item[name]) not in {int, float} for name in {"oof_candidate_rate", "wilson_upper", "pooled_synthetic_sensitivity"}) or any(item[name] is not None and type(item[name]) is not bool for name in {"seed_agreement", "finite", "backend_fallback"}) for item in contract_groups) or contract_groups != sorted(contract_groups, key=lambda item: item["length"]) or len({item["length"] for item in contract_groups}) != len(contract_groups):
        raise ValueError("runtime-native frozen contract groups are invalid")
    backend = contract["backend"]
    if set(backend) != {"backend", "candidate_identity", "max_normalized_error", "spearman", "speedup", "fallback", "fallback_reason"} or digest(backend) != payload.get("backend_audit_sha256") or contract.get("train_identity_sha256") != trace.get("train_identity_sha256") or digest(contract) != trace.get("frozen_train_contract_sha256"):
        raise ValueError("runtime-native frozen contract binding is invalid")

    models, scoring = model_hashes.get("model_evidence"), calibration_hashes.get("scoring_evidence")
    if not isinstance(models, dict) or not isinstance(scoring, dict):
        raise ValueError("runtime-native model or scoring evidence is invalid")
    if set(models) != {"schema", "status", "processed_identity_sha256", "oof_models", "final_models"} or models.get("schema") != "exp425.runtime-model-evidence.v1" or not isinstance(models.get("oof_models"), list) or not isinstance(models.get("final_models"), list) or models.get("processed_identity_sha256") != trace.get("processed_groups_sha256"):
        raise ValueError("runtime-native model evidence schema is invalid")
    def model_entries(entries: object, *, oof: bool) -> bool:
        fields = {"length", "seed", "fold", "model_sha256"} if oof else {"length", "seed", "model_sha256"}
        return isinstance(entries, list) and all(isinstance(item, dict) and set(item) == fields and type(item["length"]) is int and type(item["seed"]) is int and (not oof or type(item["fold"]) is int) and sha(item["model_sha256"]) for item in entries) and entries == sorted(entries, key=lambda item: (item["length"], item["seed"], item.get("fold", -1))) and len({tuple(item[field] for field in ("length", "seed", "fold") if field in item) for item in entries}) == len(entries)
    if not model_entries(models["oof_models"], oof=True) or not model_entries(models["final_models"], oof=False) or digest(models) != trace.get("model_evidence_sha256"):
        raise ValueError("runtime-native model identities are invalid")
    if set(scoring) != {"schema", "status", "calibrations", "final_models"} or scoring.get("schema") != "exp425.runtime-scoring-evidence.v1" or not isinstance(scoring.get("calibrations"), list) or not model_entries(scoring.get("final_models"), oof=False) or digest(scoring) != trace.get("scoring_evidence_sha256"):
        raise ValueError("runtime-native scoring evidence schema is invalid")
    calibrations = scoring["calibrations"]
    calibration_keys: set[tuple[int, int, int]] = set()
    for item in calibrations:
        if not isinstance(item, dict) or set(item) != {"length", "seed", "horizon", "digest", "statistic"} or not all(type(item[name]) is int for name in ("length", "seed", "horizon")) or not isinstance(item["statistic"], dict) or item["digest"] != digest(item["statistic"]):
            raise ValueError("runtime-native calibration evidence is invalid")
        statistic = item["statistic"]
        if set(statistic) != {"horizon", "oof_count", "length", "center", "scale", "threshold", "oof_scores_dtype", "oof_scores_shape", "oof_scores_sha256"} or statistic.get("length") != item["length"] or statistic.get("horizon") != item["horizon"] or not sha(statistic.get("oof_scores_sha256")):
            raise ValueError("runtime-native calibration statistic is invalid")
        key = (item["length"], item["seed"], item["horizon"])
        if key in calibration_keys:
            raise ValueError("runtime-native calibration keys are duplicate")
        calibration_keys.add(key)
    if calibrations != sorted(calibrations, key=lambda item: (item["length"], item["seed"], item["horizon"])):
        raise ValueError("runtime-native calibration order is invalid")
    oof_pairs = {(item["length"], item["seed"]) for item in models["oof_models"]}
    if {(length, seed) for length, seed, _ in calibration_keys} != oof_pairs:
        raise ValueError("runtime-native calibration seeds do not match OOF models")
    if set(effective) != {"status", "candidate_coordinates", "result_evidence_sha256", "result_source_sha256"}:
        raise ValueError("runtime-native effective result keys are invalid")
    candidates = effective.get("candidate_coordinates")
    if not isinstance(candidates, list) or any(not isinstance(item, list) or len(item) != 2 or item[0] != payload.get("dataset") or type(item[1]) is not int for item in candidates) or candidates != sorted(candidates, key=lambda item: item[1]) or len({item[1] for item in candidates}) != len(candidates) or effective.get("result_source_sha256") != digest({"status": effective.get("status"), "candidate_coordinates": candidates}):
        raise ValueError("runtime-native effective candidates are invalid")
    model_status, scoring_status = models.get("status"), scoring.get("status")
    finals = {(item["length"], item["seed"], item["model_sha256"]) for item in models["final_models"]}
    scored_finals = {(item["length"], item["seed"], item["model_sha256"]) for item in scoring["final_models"]}
    if model_status == "not_applicable_oof_identity_insufficient":
        if scoring_status != "not_applicable_oof_identity_insufficient" or models["oof_models"] or models["final_models"] or calibrations or scoring["final_models"] or effective.get("status") != "abstained" or candidates or payload.get("reason") != "oof_identity_insufficient":
            raise ValueError("runtime-native no-model evidence is inconsistent")
    elif model_status == "captured":
        if not models["oof_models"] or not calibrations:
            raise ValueError("runtime-native captured model evidence is incomplete")
        if scoring_status == "not_applicable_no_qualified_group":
            if scoring["final_models"] or candidates or payload.get("reason") != "native_no_qualified_group":
                raise ValueError("runtime-native no-qualified scoring evidence is inconsistent")
        elif scoring_status == "captured":
            if finals != scored_finals or payload.get("reason") != "native_mps":
                raise ValueError("runtime-native scoring final models do not match TRAIN")
        else:
            raise ValueError("runtime-native scoring status is invalid")
    else:
        raise ValueError("runtime-native model status is invalid")


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
    if not _is_lower_sha256(attestation) or any(not isinstance(dataset, str) or not _is_lower_sha256(digest) for dataset, digest in staged.items()):
        raise ValueError("import batch staged digests are invalid")
    body = {"schema": "exp425.import-batch.v1", "run_identity": run_identity, "run_context_sha256": context_sha256, "batch_identity": batch_identity, "source_attestation_sha256": attestation, "staged_record_digests": dict(sorted(staged.items()))}
    return body | {"payload_sha256": sha256(canonical_json_bytes(body)).hexdigest()}


def _quarantine_state(run_identity: str, context_sha256: str, batch: _ImportBatch, reason_sha256: str, final_attestation_sha256: str) -> dict[str, object]:
    body = {"schema": "exp425.import-quarantine.v1", "run_identity": run_identity, "run_context_sha256": context_sha256, "batch_identity": batch.identity, "source_attestation_sha256": batch.attestation_sha256, "original_inode": list(batch.inode or ()), "reason_sha256": reason_sha256, "final_attestation_sha256": final_attestation_sha256}
    if batch.inode is None or not _is_lower_sha256(reason_sha256) or not _is_lower_sha256(final_attestation_sha256):
        raise ValueError("import quarantine evidence is invalid")
    return body | {"payload_sha256": sha256(canonical_json_bytes(body)).hexdigest()}


def _cleanup_name(batch_identity: str) -> str:
    return "cleanup-" + _batch_name(batch_identity) + ".json"


def _cleanup_isolated_name(batch_identity: str, key: str, digest: str) -> str:
    return ".cleanup-owned-" + sha256((batch_identity + "\0" + key + "\0" + digest).encode()).hexdigest()


def _cleanup_state(run_identity: str, context_sha256: str, batch_identity: str, attestation: str, staged: Mapping[str, str], final_attestation: str, entries: Mapping[str, object]) -> dict[str, object]:
    if (
        not _is_lower_sha256(attestation)
        or not _is_lower_sha256(final_attestation)
        or any(not isinstance(dataset, str) or not _is_lower_sha256(digest) for dataset, digest in staged.items())
    ):
        raise ValueError("import cleanup staged digests are invalid")
    body = {"schema": "exp425.import-cleanup.v2", "run_identity": run_identity, "run_context_sha256": context_sha256, "batch_identity": batch_identity, "source_attestation_sha256": attestation, "staged_record_digests": dict(sorted(staged.items())), "final_attestation_sha256": final_attestation, "entries": dict(sorted(entries.items()))}
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
    entries = cleanup.get("entries")
    if not isinstance(attestation, str) or not isinstance(staged, dict) or not isinstance(final, str) or not isinstance(entries, dict):
        raise ValueError("import cleanup durable state is invalid")
    if cleanup != _cleanup_state(run_identity, context_sha256, batch_identity, attestation, staged, final, entries):
        raise ValueError("import cleanup durable state is invalid")
    if final != _import_final_attestation(batch_identity, attestation, staged):
        raise ValueError("import cleanup final attestation is invalid")
    if not _is_lower_sha256(attestation) or not _is_lower_sha256(final) or any(not isinstance(dataset, str) or not _is_lower_sha256(digest) for dataset, digest in staged.items()):
        raise ValueError("import cleanup staged digests are invalid")
    required = {*("staged:" + dataset for dataset in staged), "batch.json", "batch-directory"}
    if set(entries) != required:
        raise ValueError("import cleanup entries are invalid")
    for key, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"original_name", "isolated_name", "expected_identity", "entry_type", "digest", "phase"}:
            raise ValueError("import cleanup entry is invalid")
        inode = entry["expected_identity"]
        if not isinstance(inode, list) or len(inode) != 2 or any(type(value) is not int or value < 0 for value in inode) or entry["entry_type"] not in {"file", "directory"} or entry["phase"] not in {"planned", "isolated", "removed"} or not isinstance(entry["original_name"], str) or not isinstance(entry["isolated_name"], str):
            raise ValueError("import cleanup entry is invalid")
        if key.startswith("staged:"):
            dataset = key[len("staged:"):]
            digest = staged.get(dataset)
            if (
                not _is_lower_sha256(digest)
                or entry["original_name"] != record_filename(dataset)
                or entry["isolated_name"] != _cleanup_isolated_name(batch_identity, key, digest)
                or entry["entry_type"] != "file" or entry["digest"] != digest
            ):
                raise ValueError("import cleanup entry plan is invalid")
        elif key == "batch.json":
            digest = entry["digest"]
            if (
                not _is_lower_sha256(digest)
                or entry["original_name"] != "batch.json"
                or entry["isolated_name"] != _cleanup_isolated_name(batch_identity, "batch.json", digest)
                or entry["entry_type"] != "file"
            ):
                raise ValueError("import cleanup metadata plan is invalid")
        else:
            if (
                entry["original_name"] != batch_identity
                or entry["isolated_name"] != _cleanup_isolated_name(batch_identity, "batch-directory", "directory")
                or entry["entry_type"] != "directory" or entry["digest"] is not None
            ):
                raise ValueError("import cleanup batch-directory plan is invalid")
    return cleanup


def _cleanup_batch_inode(cleanup: Mapping[str, object]) -> tuple[int, int]:
    entries = cleanup.get("entries")
    inode = entries.get("batch-directory", {}).get("expected_identity") if isinstance(entries, dict) else None
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


def _validate_inventory(root: Path, provenance: Mapping[str, object], progress: Mapping[str, object], *, allow_intents: bool, full_records: bool = False, binding: "_locked_root | None" = None) -> None:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            _validate_inventory(root, provenance, progress, allow_intents=allow_intents, full_records=full_records, binding=owned)
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
                    dataset, digest, _, _ = _import_intent_claims(intent)
                    if dataset not in provenance["expected_datasets"]:
                        raise ValueError("import checkpoint intent does not match root provenance")
                    if dataset in pending_imported:
                        raise ValueError("duplicate imported checkpoint intent")
                    pending_imported[dataset] = digest
                else:
                    raise ValueError("checkpoint intent origin is invalid")
        # A fsynced native intent may legitimately precede its record.  Only
        # records actually present need inventory entries at validation time.
        pending_present = {
            dataset for dataset in (*pending_native, *pending_imported)
            if record_filename(dataset) in names
        }
        expected_names = {record_filename(dataset) for dataset in progress["record_digests"]} | {record_filename(dataset) for dataset in pending_present}
        if names != expected_names: raise ValueError("checkpoint dataset inventory is invalid")
        # The descriptor-safe inspector is a full-record validator, not just a
        # filename/progress checker.  This remains strictly read-only: no
        # reconciliation, quarantine, temporary file, fsync, or write occurs.
        if full_records:
            for dataset, digest in progress["record_digests"].items():
                raw = _read_file(record_filename(dataset), dir_fd=datasets_fd)
                if _record_digest_from_raw(raw) != digest:
                    raise ValueError("checkpoint record digest does not match progress")
                _decode_record(
                    raw, None,
                    native_run_identity=provenance["run_identity"],
                    native_context_sha256=provenance["run_context_sha256"],
                )
        for dataset, digest in pending_native.items():
            if dataset in progress["record_digests"] and progress["record_digests"][dataset] != digest:
                raise ValueError("native checkpoint intent conflicts with progress")
        for dataset, digest in pending_imported.items():
            if dataset in progress["record_digests"] and progress["record_digests"][dataset] != digest:
                raise ValueError("import checkpoint intent conflicts with progress")
        if set(pending_native) & set(pending_imported):
            raise ValueError("native and import checkpoint intents duplicate a dataset")
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
    digest = _record_payload_digest(payload)
    payload.pop("record_payload_sha256")
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
    return _record_payload_digest(_canonical_object(raw, "record"))


def _is_lower_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _record_payload_digest(payload: Mapping[str, object]) -> str:
    digest = payload.get("record_payload_sha256")
    body = dict(payload)
    body.pop("record_payload_sha256", None)
    if not _is_lower_sha256(digest) or sha256(canonical_json_bytes(body)).hexdigest() != digest:
        raise ValueError("checkpoint record digest is invalid")
    return digest


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
            if normalized in {"context", "lanes", "datasets", "testidentitymanifest", "governance", "test"} or "evaluation" in normalized or "testevaluation" in normalized or "testlabel" in normalized or any(token in normalized for token in _FORBIDDEN_RECORD_KEY_TOKENS):
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
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("checkpoint directory is invalid")
        identity = (info.st_dev, info.st_ino)
        _verify_directory_at(binding, name, identity)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


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
        try:
            info = os.fstat(self.descriptor)
            self.identity = (info.st_dev, info.st_ino)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
            self.verify()
            return self
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = None
            self.identity = None
            raise

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


def _entry_inode_at(directory_fd: int, name: str) -> tuple[int, int]:
    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("cleanup entry is a symlink")
    return info.st_dev, info.st_ino


def _cleanup_entries_at(imports_fd: int, batch_fd: int, batch_identity: str, staged: Mapping[str, str], batch_inode: tuple[int, int]) -> dict[str, object]:
    entries: dict[str, object] = {}
    for dataset, digest in staged.items():
        name = record_filename(dataset)
        inode = _entry_inode_at(batch_fd, name)
        key = "staged:" + dataset
        entries[key] = {"original_name": name, "isolated_name": _cleanup_isolated_name(batch_identity, key, digest), "expected_identity": list(inode), "entry_type": "file", "digest": digest, "phase": "planned"}
    metadata_inode = _entry_inode_at(batch_fd, "batch.json")
    metadata_digest = sha256(_read_file("batch.json", dir_fd=batch_fd)).hexdigest()
    entries["batch.json"] = {"original_name": "batch.json", "isolated_name": _cleanup_isolated_name(batch_identity, "batch.json", metadata_digest), "expected_identity": list(metadata_inode), "entry_type": "file", "digest": metadata_digest, "phase": "planned"}
    entries["batch-directory"] = {"original_name": batch_identity, "isolated_name": _cleanup_isolated_name(batch_identity, "batch-directory", "directory"), "expected_identity": list(batch_inode), "entry_type": "directory", "digest": None, "phase": "planned"}
    return entries


def _cleanup_entry_inode(entry: Mapping[str, object]) -> tuple[int, int]:
    inode = entry.get("expected_identity")
    if not isinstance(inode, list) or len(inode) != 2 or any(type(value) is not int for value in inode):
        raise ValueError("import cleanup entry is invalid")
    return inode[0], inode[1]


def _cleanup_entry_matches_at(parent_fd: int, entry: Mapping[str, object], name: str) -> bool:
    """Whether this exact directory entry still has the tombstoned identity and content."""
    entry_type, digest = entry.get("entry_type"), entry.get("digest")
    if entry_type not in {"file", "directory"} or digest is not None and not isinstance(digest, str):
        raise ValueError("import cleanup entry is invalid")
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(current.st_mode):
        raise ValueError("cleanup entry is a symlink")
    if (current.st_dev, current.st_ino) != _cleanup_entry_inode(entry):
        return False
    if entry_type == "file" and not stat.S_ISREG(current.st_mode):
        return False
    if entry_type == "directory" and not stat.S_ISDIR(current.st_mode):
        return False
    if digest is not None:
        try:
            raw = _read_file(name, dir_fd=parent_fd)
            actual = sha256(raw).hexdigest() if entry.get("original_name") == "batch.json" else _record_digest_from_raw(raw)
            return actual == digest
        except ValueError:
            return False
    return True


def _validate_cleanup_plan_at(batch_fd: int, cleanup: Mapping[str, object], *, imports_fd: int | None = None) -> None:
    """Recheck every extant cleanup entry before any further deletion."""
    entries = cleanup.get("entries")
    staged = cleanup.get("staged_record_digests")
    if not isinstance(entries, dict) or not isinstance(staged, dict):
        raise ValueError("import cleanup durable state is invalid")
    names = set(os.listdir(batch_fd))
    allowed: set[str] = set()
    metadata_name: str | None = None
    for key, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("entry_type") == "directory":
            continue
        original, isolated, phase = entry.get("original_name"), entry.get("isolated_name"), entry.get("phase")
        if not isinstance(original, str) or not isinstance(isolated, str):
            raise ValueError("import cleanup entry is invalid")
        original_present, isolated_present = original in names, isolated in names
        if phase == "removed":
            if original_present or isolated_present:
                raise ValueError("import cleanup removed entry inventory is invalid")
            continue
        if phase == "planned":
            if original_present == isolated_present:
                raise ValueError("import cleanup planned entry inventory is invalid")
        elif phase == "isolated":
            if original_present or isolated_present:
                if original_present or not isolated_present:
                    raise ValueError("import cleanup isolated entry inventory is invalid")
            # Both names absent is only the crash-after-removal boundary.
        else:
            raise ValueError("import cleanup phase is invalid")
        if original_present:
            allowed.add(original)
            current_name = original
        elif isolated_present:
            allowed.add(isolated)
            current_name = isolated
        else:
            continue
        if not _cleanup_entry_matches_at(batch_fd, entry, current_name):
            raise ValueError("import cleanup entry was replaced")
        if key == "batch.json":
            metadata_name = current_name
    if names != allowed:
        raise ValueError("import batch has unknown cleanup entries")
    if metadata_name is not None:
        state = _canonical_object(_read_file(metadata_name, dir_fd=batch_fd), "import batch")
        if state != _batch_state(
            str(cleanup["run_identity"]), str(cleanup["run_context_sha256"]), str(cleanup["batch_identity"]),
            str(cleanup["source_attestation_sha256"]), staged,
        ):
            raise ValueError("import batch cleanup metadata is invalid")
    if imports_fd is not None:
        directory_entry = entries.get("batch-directory")
        if not isinstance(directory_entry, dict):
            raise ValueError("import cleanup batch-directory entry is invalid")
        original = directory_entry.get("original_name")
        isolated = directory_entry.get("isolated_name")
        phase = directory_entry.get("phase")
        if not isinstance(original, str) or not isinstance(isolated, str):
            raise ValueError("import cleanup batch-directory entry is invalid")
        try:
            os.stat(original, dir_fd=imports_fd, follow_symlinks=False)
            original_exists = True
        except FileNotFoundError:
            original_exists = False
        try:
            os.stat(isolated, dir_fd=imports_fd, follow_symlinks=False)
            isolated_exists = True
        except FileNotFoundError:
            isolated_exists = False
        original_present = _cleanup_entry_matches_at(imports_fd, directory_entry, original)
        isolated_present = _cleanup_entry_matches_at(imports_fd, directory_entry, isolated)
        if phase == "planned":
            # A crash may happen after the exclusive move but before its phase
            # fsync.  Both legal planned representations remain exact-owned.
            if not ((original_present and not isolated_exists) or (isolated_present and not original_exists)):
                raise ValueError("import cleanup planned batch-directory inventory is invalid")
        elif phase == "isolated":
            # The all-absent form is the retained crash-after-removal boundary.
            if original_exists or (isolated_exists and not isolated_present):
                raise ValueError("import cleanup isolated batch-directory inventory is invalid")
        elif phase == "removed":
            if original_exists or isolated_exists:
                raise ValueError("import cleanup removed batch-directory inventory is invalid")
        else:
            raise ValueError("import cleanup batch-directory phase is invalid")


def _persist_cleanup_at(imports_fd: int, cleanup: Mapping[str, object]) -> dict[str, object]:
    entries = cleanup["entries"]
    assert isinstance(entries, dict)
    state = _cleanup_state(str(cleanup["run_identity"]), str(cleanup["run_context_sha256"]), str(cleanup["batch_identity"]), str(cleanup["source_attestation_sha256"]), cleanup["staged_record_digests"], str(cleanup["final_attestation_sha256"]), entries)
    _atomic_replace_at(imports_fd, _cleanup_name(str(cleanup["batch_identity"])), canonical_json_bytes(state))
    return state


def _advance_cleanup_entry_at(parent_fd: int, imports_fd: int, cleanup: Mapping[str, object], key: str, *, phase_label: str, verify_parent: object) -> dict[str, object]:
    """Advance one v2 tombstoned cleanup entry through planned→isolated→removed."""
    entries = cleanup.get("entries")
    if not isinstance(entries, dict) or not isinstance(entries.get(key), dict):
        raise ValueError("import cleanup entry is invalid")
    entry = entries[key]
    assert isinstance(entry, dict)
    original, isolated, expected = entry["original_name"], entry["isolated_name"], _cleanup_entry_inode(entry)
    if not isinstance(original, str) or not isinstance(isolated, str):
        raise ValueError("import cleanup entry is invalid")
    entry_type = entry.get("entry_type")
    if entry_type not in {"file", "directory"}:
        raise ValueError("import cleanup entry is invalid")

    def info(name: str) -> os.stat_result | None:
        try:
            result = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(result.st_mode):
            raise ValueError("cleanup entry is a symlink")
        return result

    def inode(name: str) -> tuple[int, int] | None:
        current = info(name)
        return None if current is None else (current.st_dev, current.st_ino)

    def expected_entry(name: str) -> bool:
        return _cleanup_entry_matches_at(parent_fd, entry, name)

    if not callable(verify_parent):
        raise ValueError("import cleanup parent binding is invalid")

    def restore_foreign_isolated() -> None:
        """Return an untrusted isolated object to its public name without deleting it."""
        moved = info(isolated)
        if moved is None or inode(original) is not None:
            raise ValueError("cleanup replacement requires quarantine")
        moved_identity = (moved.st_dev, moved.st_ino)
        moved_mode = stat.S_IFMT(moved.st_mode)
        moved_bytes = _read_file(isolated, dir_fd=parent_fd) if stat.S_ISREG(moved.st_mode) else None
        try:
            _rename_exclusive_at(parent_fd, isolated, original)
        except ValueError as error:
            raise ValueError("cleanup replacement requires quarantine") from error
        restored = info(original)
        if (restored is None or (restored.st_dev, restored.st_ino) != moved_identity
                or stat.S_IFMT(restored.st_mode) != moved_mode or info(isolated) is not None
                or moved_bytes is not None and _read_file(original, dir_fd=parent_fd) != moved_bytes):
            raise ValueError("cleanup replacement restoration failed")
        os.fsync(parent_fd)
        raise ValueError("cleanup entry was replaced during isolation")

    phase = entry.get("phase")
    original_inode, isolated_inode = inode(original), inode(isolated)
    if phase == "planned":
        if original_inode is not None and isolated_inode is None:
            _after_import_cleanup_phase("cleanup_before_" + phase_label + "_isolation")
            # Persisted planned state names the deterministic isolated target before the move.
            verify_parent()
            cleanup = _persist_cleanup_at(imports_fd, cleanup)
            _rename_exclusive_at(parent_fd, original, isolated)
            if not expected_entry(isolated):
                restore_foreign_isolated()
            _after_import_cleanup_phase("cleanup_after_" + phase_label + "_isolation_before_phase_update")
        elif original_inode is None and isolated_inode is not None:
            if not expected_entry(isolated):
                restore_foreign_isolated()
        else:
            raise ValueError("import cleanup planned entry inventory is invalid")
        entry["phase"] = "isolated"
        cleanup = _persist_cleanup_at(imports_fd, cleanup)
        entries = cleanup["entries"]; assert isinstance(entries, dict); entry = entries[key]; assert isinstance(entry, dict)
        phase = "isolated"
    if phase == "isolated":
        original_inode, isolated_inode = inode(original), inode(isolated)
        if original_inode is None and isolated_inode is None:
            # The removal and parent fsync may have completed just before a
            # process death.  This precise durable state is safely advanced
            # only while the retained descriptor/root binding still verifies.
            verify_parent()
            os.fsync(parent_fd)
            entry["phase"] = "removed"
            return _persist_cleanup_at(imports_fd, cleanup)
        if original_inode is not None or not expected_entry(isolated):
            raise ValueError("import cleanup isolated entry inventory is invalid")
        _after_import_cleanup_phase("cleanup_after_" + phase_label + "_phase_before_remove")
        # The hook is intentionally an adversarial boundary: revalidate the
        # exact inode/type/content immediately before descriptor-relative unlink.
        verify_parent()
        if not expected_entry(isolated):
            restore_foreign_isolated()
        if entry["entry_type"] == "directory":
            os.rmdir(isolated, dir_fd=parent_fd)
        else:
            os.unlink(isolated, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _after_import_cleanup_phase("cleanup_after_" + phase_label + "_remove_before_phase_update")
        verify_parent()
        entry["phase"] = "removed"
        cleanup = _persist_cleanup_at(imports_fd, cleanup)
    elif phase == "removed":
        if inode(original) is not None or inode(isolated) is not None:
            raise ValueError("import cleanup removed entry inventory is invalid")
    else:
        raise ValueError("import cleanup phase is invalid")
    return dict(cleanup)


def _import_intent_requirements(root: Path, *, binding: _locked_root | None = None) -> tuple[tuple[str, str], ...]:
    if binding is None:
        with _locked_root(root, exclusive=False) as owned:
            return _import_intent_requirements(root, binding=owned)
    stored, _, _ = _read_root_metadata(root, binding=binding)
    transactions_fd, transactions_identity = _open_directory_at(binding, "transactions")
    imports_fd, imports_identity = _open_directory_at(binding, "import_batches")
    try:
        batches: dict[str, tuple[str, Mapping[str, str]]] = {}
        cleanup: dict[str, dict[str, object]] = {}
        directories: set[str] = set()
        for name in sorted(os.listdir(imports_fd)):
            if name.startswith(".quarantined-"):
                digest = name[len(".quarantined-"):]
                info = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                if not _is_lower_sha256(digest) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError("import quarantine inventory is invalid")
                quarantine_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                try:
                    if not {"quarantine.json", "batch.json"} <= set(os.listdir(quarantine_fd)):
                        raise ValueError("import quarantine inventory is invalid")
                    evidence = _canonical_object(_read_file("quarantine.json", dir_fd=quarantine_fd), "import quarantine")
                    batch_identity = evidence.get("batch_identity")
                    attestation = evidence.get("source_attestation_sha256")
                    reason = evidence.get("reason_sha256")
                    final = evidence.get("final_attestation_sha256")
                    inode = evidence.get("original_inode")
                    if not isinstance(batch_identity, str) or not isinstance(attestation, str) or not isinstance(reason, str) or not isinstance(final, str) or not isinstance(inode, list) or len(inode) != 2:
                        raise ValueError("import quarantine evidence is invalid")
                    owned = _ImportBatch(batch_identity, Path(name), (inode[0], inode[1]), attestation, "quarantined")
                    if evidence != _quarantine_state(str(stored["run_identity"]), str(stored["run_context_sha256"]), owned, reason, final):
                        raise ValueError("import quarantine evidence is invalid")
                    if (info.st_dev, info.st_ino) != owned.inode:
                        raise ValueError("import quarantine directory was replaced")
                    state = _canonical_object(_read_file("batch.json", dir_fd=quarantine_fd), "quarantined import batch")
                    staged = state.get("staged_record_digests")
                    if not isinstance(staged, dict) or state != _batch_state(str(stored["run_identity"]), str(stored["run_context_sha256"]), batch_identity, attestation, staged):
                        raise ValueError("quarantined import batch state is invalid")
                    if final != _import_final_attestation(batch_identity, attestation, staged):
                        raise ValueError("quarantined import batch attestation is invalid")
                    if set(os.listdir(quarantine_fd)) != {"quarantine.json", "batch.json"} | {record_filename(dataset) for dataset in staged}:
                        raise ValueError("quarantined import batch inventory is invalid")
                    for dataset, digest in staged.items():
                        if not isinstance(dataset, str) or _record_digest_from_raw(_read_file(record_filename(dataset), dir_fd=quarantine_fd)) != digest:
                            raise ValueError("quarantined import staged record is invalid")
                finally:
                    os.close(quarantine_fd)
                continue
            if name.startswith("cleanup-") and name.endswith(".json"):
                batch_identity = name[len("cleanup-"):-len(".json")]
                if _cleanup_name(batch_identity) != name or batch_identity in cleanup:
                    raise ValueError("import cleanup inventory is invalid")
                cleanup[batch_identity] = _read_cleanup_at(imports_fd, str(stored["run_identity"]), str(stored["run_context_sha256"]), batch_identity) or {}
                if not cleanup[batch_identity]:
                    raise ValueError("import cleanup inventory is invalid")
                continue
            info = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or not (name.startswith(".cleanup-owned-") or _batch_name(name) == name):
                raise ValueError("import batch inventory is invalid")
            directories.add(name)
        for name in sorted(directories):
            batch_fd: int | None = None
            try:
                batch_fd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=imports_fd)
                batch_identity = name
                pending_cleanup = cleanup.get(batch_identity)
                if pending_cleanup is None:
                    for candidate, state in cleanup.items():
                        entries = state.get("entries")
                        if isinstance(entries, dict) and isinstance(entries.get("batch-directory"), dict) and entries["batch-directory"].get("isolated_name") == name:
                            batch_identity, pending_cleanup = candidate, state
                            break
                if pending_cleanup is not None:
                    staged = pending_cleanup["staged_record_digests"]
                    attestation = pending_cleanup["source_attestation_sha256"]
                    assert isinstance(staged, dict) and isinstance(attestation, str)
                    opened = os.fstat(batch_fd)
                    if (opened.st_dev, opened.st_ino) != _cleanup_batch_inode(pending_cleanup):
                        raise ValueError("import batch directory was replaced before cleanup")
                    entries = pending_cleanup["entries"]
                    assert isinstance(entries, dict)
                    _validate_cleanup_plan_at(batch_fd, pending_cleanup, imports_fd=imports_fd)
                    directory_entry = entries["batch-directory"]
                    assert isinstance(directory_entry, dict)
                    if not _cleanup_entry_matches_at(imports_fd, directory_entry, name):
                        raise ValueError("import batch directory was replaced before cleanup")
                    current = os.stat(name, dir_fd=imports_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != _cleanup_batch_inode(pending_cleanup):
                        raise ValueError("import batch directory was replaced before cleanup")
                    batches[batch_identity] = (str(pending_cleanup["final_attestation_sha256"]), staged)
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
                    if not isinstance(dataset, str) or not _is_lower_sha256(digest):
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
                batches[batch_identity] = (_import_final_attestation(batch_identity, attestation, staged), staged)
            finally:
                if batch_fd is not None: os.close(batch_fd)
        for batch_identity, state in cleanup.items():
            if batch_identity not in batches:
                staged = state["staged_record_digests"]
                assert isinstance(staged, dict)
                batches[batch_identity] = (str(state["final_attestation_sha256"]), staged)
        result: set[tuple[str, str]] = set()
        for name in sorted(os.listdir(transactions_fd)):
            if not name.endswith(".json"):
                raise ValueError("checkpoint transaction inventory is invalid")
            intent = _canonical_object(_read_file(name, dir_fd=transactions_fd), "intent")
            if intent.get("origin") == "exp424_import":
                dataset, digest, batch, final = _import_intent_claims(intent)
                batch_state = batches.get(batch)
                if (
                    dataset not in stored["expected_datasets"]
                    or batch_state is None
                    or batch_state[0] != final
                    or batch_state[1].get(dataset) != digest
                ):
                    raise ValueError("import intent is invalid")
                result.add((batch, final))
            elif intent.get("origin") != "exp425_native":
                raise ValueError("checkpoint intent origin is invalid")
        result.update((batch_identity, state[0]) for batch_identity, state in batches.items())
        _verify_directory_at(binding, "transactions", transactions_identity)
        _verify_directory_at(binding, "import_batches", imports_identity)
        return tuple(sorted(result))
    finally:
        os.close(imports_fd)
        os.close(transactions_fd)
