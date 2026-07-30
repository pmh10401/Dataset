"""Fail-closed, provenance-bound local checkpoints for Exp417.

The store is deliberately local-only.  It persists no source data; a record is
accepted only when its immutable decision evidence can be reconstructed from
canonical JSON and all run provenance still agrees with the root manifest.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import errno
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Any
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config import FROZEN_CONFIG
from .contracts import (
    BackendAudit, DatasetDecisionEvidence, DatasetResult, DatasetTerminalEvidence, FrozenLaneSnapshot,
    FrozenRunContext, FrozenTrainContract, GroupQualification, HorizonScore,
    SeedDecision, ShadowCandidate, TestInstanceDecision,
)


_SCHEMA_VERSION = 3
_ROOT_FILE = "checkpoint_provenance.json"
_PROGRESS_FILE = "progress.json"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_LINUX_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_RENAME_SWAP = 0x00000002
_RENAME_EXCL = 0x00000004


class _CommittedWriteError(BaseException):
    """Carry inode ownership when an atomic rename committed before failure."""

    def __init__(
        self, original: BaseException, identity: tuple[int, int]
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.identity = identity


class _DirectoryBindingError(ValueError):
    """A pathname no longer names the directory inode opened for the operation."""


@dataclass(frozen=True)
class Exp417CheckpointProvenance:
    """Every immutable input and algorithm contract required to resume safely."""

    git_commit: str
    database_sha256: str
    exp291_prediction_sha256: str
    exp291_manifest_sha256: str
    exp292_prediction_sha256: str
    exp292_manifest_sha256: str
    frozen_config_sha256: str
    preprocessing_contract_sha256: str
    fold_contract_sha256: str
    model_contract_sha256: str
    model_seeds: tuple[int, ...]
    perturbation_seeds: tuple[int, ...]
    runtime_versions: Mapping[str, str]
    backend_contract_sha256: str

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "git_commit": self.git_commit, "database_sha256": self.database_sha256,
            "exp291_prediction_sha256": self.exp291_prediction_sha256,
            "exp291_manifest_sha256": self.exp291_manifest_sha256,
            "exp292_prediction_sha256": self.exp292_prediction_sha256,
            "exp292_manifest_sha256": self.exp292_manifest_sha256,
            "frozen_config_sha256": self.frozen_config_sha256,
            "preprocessing_contract_sha256": self.preprocessing_contract_sha256,
            "fold_contract_sha256": self.fold_contract_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "model_seeds": self.model_seeds,
            "perturbation_seeds": self.perturbation_seeds,
            "runtime_versions": dict(self.runtime_versions),
            "backend_contract_sha256": self.backend_contract_sha256,
        }
        _validate_provenance(payload)
        return payload


def canonical_result_sha256(result: DatasetResult) -> str:
    """Hash the complete canonical DatasetResult checkpoint payload."""
    if not isinstance(result, DatasetResult):
        raise ValueError("Exp417 checkpoint result is invalid")
    return sha256(canonical_json_bytes(_result_payload(result))).hexdigest()


class Exp417CheckpointStore:
    """An exclusive per-dataset checkpoint store with a fixed dataset universe."""

    def __init__(self, root: Path, provenance: Exp417CheckpointProvenance,
                 expected_datasets: Sequence[str]) -> None:
        if not isinstance(provenance, Exp417CheckpointProvenance):
            raise ValueError("Exp417 checkpoint provenance is invalid")
        self.root = Path(root)
        self.datasets_dir = self.root / "datasets"
        self.provenance = provenance
        self.expected_datasets = _dataset_universe(expected_datasets)
        self._root_payload = {
            "schema_version": _SCHEMA_VERSION,
            "provenance": provenance.to_payload(),
            "expected_datasets": list(self.expected_datasets),
        }
        if _path_exists_lstat(self.root):
            self._validate_existing_root_or_quarantine()
        else:
            self._create_root()

    def write_dataset(
        self,
        result: DatasetResult,
        *,
        raw_train_identity_sha256: str | None,
        repaired_train_identity_sha256: str | None,
        length_group_identity_sha256: Mapping[int, str],
        canonical_model_hashes: Mapping[int, Mapping[int, str]],
        score_calibration_hashes: Mapping[int, str],
        backend_audit: BackendAudit | None,
        elapsed_seconds: float,
    ) -> None:
        if not isinstance(result, DatasetResult):
            raise ValueError("Exp417 checkpoint result is invalid")
        _validate_result_provenance(result, self.provenance)
        if result.dataset not in self.expected_datasets:
            raise ValueError("Exp417 checkpoint dataset is outside the universe")
        terminal = isinstance(result.decision_evidence, DatasetTerminalEvidence)
        if terminal:
            if any((raw_train_identity_sha256 is not None, repaired_train_identity_sha256 is not None,
                    length_group_identity_sha256, canonical_model_hashes,
                    score_calibration_hashes, backend_audit is not None)):
                raise ValueError("terminal checkpoints must not invent training or scoring evidence")
        elif raw_train_identity_sha256 is None:
            raise ValueError("decision checkpoints require raw TRAIN identity")
        if raw_train_identity_sha256 is not None:
            _digest(raw_train_identity_sha256, "raw TRAIN identity")
        if repaired_train_identity_sha256 is not None:
            _digest(repaired_train_identity_sha256, "repaired TRAIN identity")
        identities = _length_digest_mapping(length_group_identity_sha256, "length-group identities")
        model_hashes = _model_hash_mapping(canonical_model_hashes)
        calibrations = _length_digest_mapping(score_calibration_hashes, "score calibration hashes")
        if not terminal:
            _validate_length_evidence(
                result, identities, model_hashes, calibrations
            )
        contract = result.decision_evidence.frozen_train_contract
        if terminal:
            pass
        elif not isinstance(backend_audit, BackendAudit) or contract is None or backend_audit != contract.backend:
            raise ValueError("Exp417 checkpoint backend audit does not match result")
        elapsed = _finite_nonnegative(elapsed_seconds, "elapsed time")
        record_name = self._dataset_path(result.dataset).name
        body: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "dataset": result.dataset,
            "raw_train_identity_sha256": raw_train_identity_sha256,
            "repaired_train_identity_sha256": repaired_train_identity_sha256,
            "length_group_identity_sha256": {str(key): value for key, value in identities.items()},
            "canonical_model_hashes": {
                str(length): {str(seed): digest for seed, digest in hashes.items()}
                for length, hashes in model_hashes.items()
            },
            "score_calibration_hashes": {str(key): value for key, value in calibrations.items()},
            "qualification": _result_payload(result),
            "candidate_coordinates": [[candidate.dataset, candidate.instance_index] for candidate in result.candidates],
            "backend_audit": None if backend_audit is None else _backend_payload(backend_audit),
            "elapsed_seconds": elapsed,
        }
        body["payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
        record_bytes = canonical_json_bytes(body) + b"\n"
        with self._root_descriptor_lock(exclusive=True) as (
            root_descriptor,
            root_identity,
        ):
            datasets_descriptor: int | None = None
            try:
                try:
                    datasets_descriptor, datasets_identity = (
                        _open_directory_at_verified(
                            root_descriptor,
                            "datasets",
                            "Exp417 checkpoint datasets directory",
                        )
                    )
                    (
                        old_provenance,
                        old_provenance_identity,
                        old_progress,
                        old_progress_identity,
                        digests,
                        dataset_names,
                    ) = self._read_write_metadata_at(
                        root_descriptor, datasets_descriptor
                    )
                except _DirectoryBindingError:
                    raise
                except ValueError as error:
                    if datasets_descriptor is not None:
                        _verify_directory_entry_identity_at(
                            root_descriptor,
                            "datasets",
                            datasets_identity,
                            "Exp417 checkpoint datasets directory",
                        )
                    self._verify_root_path_identity(root_identity)
                    quarantine = self._quarantine_root(
                        root_identity,
                        root_descriptor,
                    )
                    raise ValueError(
                        "Exp417 checkpoint state was quarantined at "
                        f"{quarantine}"
                    ) from error
                # The locked directory descriptors are the transaction
                # authority.  Every read and mutation remains bound to
                # those inodes even if their pathnames undergo ABA swaps.
                if (
                    result.dataset in digests
                    or record_name in dataset_names
                ):
                    raise ValueError(
                        "Exp417 checkpoint already exists for dataset: "
                        f"{result.dataset}"
                    )
                published = False
                progress_published = False
                progress_identity: tuple[int, int] | None = None
                try:
                    try:
                        published_identity = _atomic_write_exclusive_at(
                            datasets_descriptor,
                            record_name,
                            record_bytes,
                            recovery_root_descriptor=root_descriptor,
                        )
                        published = True
                    except _CommittedWriteError as committed:
                        published_identity = committed.identity
                        published = True
                        raise committed.original
                    os.fsync(datasets_descriptor)
                    digests[result.dataset] = body["payload_sha256"]
                    progress_bytes = self._progress_bytes(
                        completed=len(digests),
                        current_dataset=None,
                        record_digests=digests,
                    )
                    try:
                        progress_identity = self._replace_owned_progress_at(
                            root_descriptor,
                            progress_bytes,
                            old_progress_identity,
                        )
                        progress_published = True
                    except _CommittedWriteError as committed:
                        progress_identity = committed.identity
                        progress_published = True
                        raise committed.original
                    os.fsync(root_descriptor)
                    observed_record, observed_record_identity = (
                        _read_regular_entry_at(
                            datasets_descriptor,
                            record_name,
                            "Exp417 published dataset checkpoint",
                        )
                    )
                    if (
                        observed_record_identity != published_identity
                        or observed_record != record_bytes
                    ):
                        raise ValueError(
                            "Exp417 published dataset checkpoint changed "
                            "before commit verification"
                        )
                    observed_progress, observed_progress_identity = (
                        _read_regular_entry_at(
                            root_descriptor,
                            _PROGRESS_FILE,
                            "Exp417 published checkpoint progress",
                        )
                    )
                    if (
                        observed_progress_identity != progress_identity
                        or observed_progress != progress_bytes
                    ):
                        raise ValueError(
                            "Exp417 published checkpoint progress changed "
                            "before commit verification"
                        )
                    (
                        observed_provenance,
                        observed_provenance_identity,
                    ) = _read_regular_entry_at(
                        root_descriptor,
                        _ROOT_FILE,
                        "Exp417 published checkpoint provenance",
                    )
                    if (
                        observed_provenance_identity
                        != old_provenance_identity
                        or observed_provenance != old_provenance
                    ):
                        raise ValueError(
                            "Exp417 checkpoint provenance changed "
                            "during commit"
                        )
                    final_root_inventory = set(
                        _list_directory_descriptor(
                            root_descriptor,
                            "Exp417 checkpoint root",
                        )
                    )
                    if final_root_inventory != {
                        _ROOT_FILE,
                        _PROGRESS_FILE,
                        "datasets",
                    }:
                        raise ValueError(
                            "Exp417 checkpoint root inventory changed "
                            "during commit"
                        )
                    final_dataset_names = (
                        self._validate_write_dataset_inventory_at(
                            datasets_descriptor,
                            digests,
                        )
                    )
                    if final_dataset_names != dataset_names | {record_name}:
                        raise ValueError(
                            "Exp417 checkpoint dataset inventory changed "
                            "during commit"
                        )
                    _verify_directory_entry_identity_at(
                        root_descriptor,
                        "datasets",
                        datasets_identity,
                        "Exp417 checkpoint datasets directory",
                    )
                    self._verify_root_path_identity(root_identity)
                except BaseException as error:
                    try:
                        self._rollback_transaction_at(
                            datasets_descriptor,
                            record_name if published else None,
                            published_identity if published else None,
                            root_descriptor,
                            old_provenance,
                            old_provenance_identity,
                            old_progress,
                            old_progress_identity,
                            progress_identity if progress_published else None,
                            root_identity,
                            datasets_identity,
                        )
                    except BaseException as rollback_error:
                        raise BaseExceptionGroup(
                            "Exp417 checkpoint transaction and rollback both failed",
                            [error, rollback_error],
                        )
                    raise
            finally:
                if datasets_descriptor is not None:
                    os.close(datasets_descriptor)

    def load_completed(self) -> Mapping[str, DatasetResult]:
        records, _ = self._read_state_snapshot()
        return MappingProxyType(
            {
                name: records[name]["qualification"]
                for name in self.expected_datasets
                if name in records
            }
        )

    def progress(self) -> Mapping[str, object]:
        _, payload = self._read_state_snapshot()
        return MappingProxyType(
            {
                "completed_datasets": payload["completed_datasets"],
                "current_dataset": payload["current_dataset"],
                "expected_datasets": payload["expected_datasets"],
            }
        )

    def completed_record_digests(self) -> Mapping[str, str]:
        """Return validated canonical record digests under a shared root lock."""
        _, payload = self._read_state_snapshot()
        digests = _progress_digests(payload)
        return MappingProxyType(dict(sorted(digests.items())))

    def require_complete_snapshot(
        self,
    ) -> tuple[Mapping[str, DatasetResult], Mapping[str, str]]:
        """Return complete results and record digests from one shared-lock snapshot."""
        records, progress = self._read_state_snapshot()
        if set(records) != set(self.expected_datasets):
            raise ValueError("Exp417 checkpoint dataset coverage is incomplete")
        completed = {
            name: records[name]["qualification"]
            for name in self.expected_datasets
        }
        progress_digests = _progress_digests(progress)
        digests = {
            name: _digest(
                progress_digests[name],
                "checkpoint record payload",
            )
            for name in self.expected_datasets
        }
        return MappingProxyType(completed), MappingProxyType(digests)

    def _read_state_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        with self._root_descriptor_lock(exclusive=False) as (
            root_descriptor,
            root_identity,
        ):
            datasets_descriptor, datasets_identity = (
                _open_directory_at_verified(
                    root_descriptor,
                    "datasets",
                    "Exp417 checkpoint datasets directory",
                )
            )
            try:
                records = self._load_records_from_descriptor(
                    root_descriptor, datasets_descriptor
                )
                progress = self._read_progress_from_descriptor(
                    root_descriptor
                )
                progress_digests = _progress_digests(progress)
                observed_digests = {
                    name: record["payload_sha256"]
                    for name, record in records.items()
                }
                if progress_digests != observed_digests:
                    raise ValueError(
                        "Exp417 checkpoint progress accounting is invalid"
                    )
                _verify_directory_entry_identity_at(
                    root_descriptor,
                    "datasets",
                    datasets_identity,
                    "Exp417 checkpoint datasets directory",
                )
                self._verify_root_path_identity(root_identity)
                return records, progress
            finally:
                os.close(datasets_descriptor)

    def require_complete(self) -> tuple[DatasetResult, ...]:
        completed, _ = self.require_complete_snapshot()
        return tuple(completed[name] for name in self.expected_datasets)

    def _create_root(self) -> None:
        root_name = _entry_name(
            self.root.name, "Exp417 checkpoint root name"
        )
        staging_name = f".exp417-init-{uuid.uuid4().hex}"
        try:
            parent_descriptor = os.open(
                self.root.parent, _DIRECTORY_OPEN_FLAGS
            )
        except OSError as error:
            raise ValueError("Exp417 checkpoint parent is invalid") from error
        root_descriptor: int | None = None
        datasets_descriptor: int | None = None
        try:
            parent_metadata = os.fstat(parent_descriptor)
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise ValueError("Exp417 checkpoint parent is invalid")
            parent_identity = (
                parent_metadata.st_dev,
                parent_metadata.st_ino,
            )
            try:
                named_parent = os.stat(
                    self.root.parent, follow_symlinks=False
                )
            except OSError as error:
                raise ValueError(
                    "Exp417 checkpoint parent is invalid"
                ) from error
            if (
                not stat.S_ISDIR(named_parent.st_mode)
                or (named_parent.st_dev, named_parent.st_ino)
                != parent_identity
            ):
                raise ValueError(
                    "Exp417 checkpoint parent identity changed"
                )
            baseline_inventory = _directory_inventory_at(
                parent_descriptor,
                "Exp417 checkpoint parent",
            )
            baseline_identities = {
                (metadata.st_dev, metadata.st_ino)
                for metadata in baseline_inventory.values()
            }
            os.mkdir(
                staging_name,
                0o700,
                dir_fd=parent_descriptor,
            )
            initialized_inventory = _directory_inventory_at(
                parent_descriptor,
                "Exp417 checkpoint parent",
            )
            staged_metadata = initialized_inventory.get(staging_name)
            new_identities = {
                (metadata.st_dev, metadata.st_ino)
                for metadata in initialized_inventory.values()
                if (metadata.st_dev, metadata.st_ino)
                not in baseline_identities
            }
            if (
                staged_metadata is None
                or not stat.S_ISDIR(staged_metadata.st_mode)
                or new_identities
                != {(staged_metadata.st_dev, staged_metadata.st_ino)}
            ):
                raise ValueError(
                    "Exp417 checkpoint staging root ownership is invalid"
                )
            staged_identity = (
                staged_metadata.st_dev,
                staged_metadata.st_ino,
            )
            root_descriptor = os.open(
                staging_name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=parent_descriptor,
            )
            root_metadata = os.fstat(root_descriptor)
            if not stat.S_ISDIR(root_metadata.st_mode):
                raise ValueError("Exp417 checkpoint root is invalid")
            root_identity = (
                root_metadata.st_dev,
                root_metadata.st_ino,
            )
            if root_identity != staged_identity:
                raise ValueError(
                    "Exp417 checkpoint staging root identity changed"
                )
            _verify_directory_entry_identity_at(
                parent_descriptor,
                staging_name,
                root_identity,
                "Exp417 checkpoint staging root",
            )
            os.fchmod(root_descriptor, 0o700)
            os.mkdir("datasets", 0o700, dir_fd=root_descriptor)
            (
                datasets_descriptor,
                datasets_identity,
            ) = _open_directory_at_verified(
                root_descriptor,
                "datasets",
                "Exp417 checkpoint datasets directory",
            )
            os.fchmod(datasets_descriptor, 0o700)
            provenance_bytes = (
                canonical_json_bytes(self._root_payload) + b"\n"
            )
            progress_bytes = self._progress_bytes(
                completed=0,
                current_dataset=None,
                record_digests={},
            )
            provenance_identity = _atomic_write_exclusive_at(
                root_descriptor,
                _ROOT_FILE,
                provenance_bytes,
                recovery_root_descriptor=root_descriptor,
            )
            progress_identity = _atomic_write_exclusive_at(
                root_descriptor,
                _PROGRESS_FILE,
                progress_bytes,
                recovery_root_descriptor=root_descriptor,
            )
            os.fsync(datasets_descriptor)
            os.fsync(root_descriptor)
            os.fsync(parent_descriptor)
            if set(_list_directory_descriptor(
                root_descriptor, "Exp417 checkpoint root"
            )) != {_ROOT_FILE, _PROGRESS_FILE, "datasets"}:
                raise ValueError(
                    "Exp417 checkpoint root inventory is invalid"
                )
            observed_provenance, observed_provenance_identity = (
                _read_regular_entry_at(
                    root_descriptor,
                    _ROOT_FILE,
                    "Exp417 checkpoint provenance",
                )
            )
            observed_progress, observed_progress_identity = (
                _read_regular_entry_at(
                    root_descriptor,
                    _PROGRESS_FILE,
                    "Exp417 checkpoint progress",
                )
            )
            if (
                observed_provenance != provenance_bytes
                or observed_provenance_identity != provenance_identity
                or observed_progress != progress_bytes
                or observed_progress_identity != progress_identity
            ):
                raise ValueError(
                    "Exp417 checkpoint initialization changed during publish"
                )
            _verify_directory_entry_identity_at(
                root_descriptor,
                "datasets",
                datasets_identity,
                "Exp417 checkpoint datasets directory",
            )
            _verify_directory_entry_identity_at(
                parent_descriptor,
                staging_name,
                root_identity,
                "Exp417 checkpoint staging root",
            )
            try:
                publish_returned = False
                _atomic_rename_noreplace_at(
                    parent_descriptor,
                    staging_name,
                    parent_descriptor,
                    root_name,
                )
                publish_returned = True
                os.fsync(parent_descriptor)
                published_root_metadata = os.fstat(root_descriptor)
                if (
                    not stat.S_ISDIR(published_root_metadata.st_mode)
                    or (
                        published_root_metadata.st_dev,
                        published_root_metadata.st_ino,
                    )
                    != root_identity
                ):
                    raise ValueError(
                        "Exp417 checkpoint published root identity changed"
                    )
                _verify_directory_entry_identity_at(
                    parent_descriptor,
                    root_name,
                    root_identity,
                    "Exp417 checkpoint root",
                )
                if set(_list_directory_descriptor(
                    root_descriptor,
                    "Exp417 checkpoint root",
                )) != {_ROOT_FILE, _PROGRESS_FILE, "datasets"}:
                    raise ValueError(
                        "Exp417 checkpoint published root inventory is "
                        "invalid"
                    )
                (
                    published_provenance,
                    published_provenance_identity,
                ) = _read_regular_entry_at(
                    root_descriptor,
                    _ROOT_FILE,
                    "Exp417 checkpoint provenance",
                )
                (
                    published_progress,
                    published_progress_identity,
                ) = _read_regular_entry_at(
                    root_descriptor,
                    _PROGRESS_FILE,
                    "Exp417 checkpoint progress",
                )
                if (
                    published_provenance != provenance_bytes
                    or published_provenance_identity
                    != provenance_identity
                    or published_progress != progress_bytes
                    or published_progress_identity != progress_identity
                ):
                    raise ValueError(
                        "Exp417 checkpoint initialization changed during "
                        "root publish"
                    )
                _verify_directory_entry_identity_at(
                    root_descriptor,
                    "datasets",
                    datasets_identity,
                    "Exp417 checkpoint datasets directory",
                )
                if _list_directory_descriptor(
                    datasets_descriptor,
                    "Exp417 checkpoint datasets directory",
                ):
                    raise ValueError(
                        "Exp417 checkpoint published datasets inventory is "
                        "invalid"
                    )
                _verify_directory_entry_identity_at(
                    parent_descriptor,
                    root_name,
                    root_identity,
                    "Exp417 checkpoint root",
                )
                self._verify_root_path_identity(root_identity)
                try:
                    final_parent = os.stat(
                        self.root.parent, follow_symlinks=False
                    )
                except OSError as error:
                    raise ValueError(
                        "Exp417 checkpoint parent identity changed"
                    ) from error
                if (
                    not stat.S_ISDIR(final_parent.st_mode)
                    or (final_parent.st_dev, final_parent.st_ino)
                    != parent_identity
                ):
                    raise ValueError(
                        "Exp417 checkpoint parent identity changed"
                    )
            except BaseException as error:
                failures: list[BaseException] = [error]
                try:
                    source_identity = _entry_identity_at_optional(
                        parent_descriptor,
                        staging_name,
                        "Exp417 checkpoint staging root",
                    )
                except BaseException as probe_error:
                    failures.append(probe_error)
                    source_identity = None
                target_probe_failed = False
                try:
                    target_identity = _entry_identity_at_optional(
                        parent_descriptor,
                        root_name,
                        "Exp417 checkpoint root",
                    )
                except BaseException as probe_error:
                    failures.append(probe_error)
                    target_identity = None
                    target_probe_failed = True
                should_capture = (
                    publish_returned
                    or target_probe_failed
                    or (
                        target_identity is not None
                        and source_identity != root_identity
                    )
                )
                captured: tuple[str, tuple[int, int]] | None = None
                if should_capture:
                    try:
                        captured = _preserve_root_publish_target_at(
                            parent_descriptor,
                            root_name,
                        )
                    except BaseException as preserve_error:
                        failures.append(preserve_error)
                    if (
                        captured is not None
                        and captured[1] != root_identity
                    ):
                        failures.append(ValueError(
                            "Exp417 checkpoint foreign root publish target "
                            f"was preserved as {captured[0]}"
                        ))
                try:
                    canonical_identity = _entry_identity_at_optional(
                        parent_descriptor,
                        root_name,
                        "Exp417 checkpoint root",
                    )
                except BaseException as probe_error:
                    failures.append(probe_error)
                    canonical_identity = None
                if should_capture and canonical_identity is not None:
                    failures.append(ValueError(
                        "Exp417 checkpoint canonical root remains after "
                        "publish recovery"
                    ))
                if len(failures) == 1:
                    raise failures[0]
                raise BaseExceptionGroup(
                    "Exp417 checkpoint root publish recovery failed",
                    failures,
                )
        finally:
            if datasets_descriptor is not None:
                os.close(datasets_descriptor)
            if root_descriptor is not None:
                os.close(root_descriptor)
            os.close(parent_descriptor)

    def _validate_existing_root_or_quarantine(self) -> None:
        with self._root_descriptor_lock(exclusive=True) as (
            root_descriptor,
            root_identity,
        ):
            # Provenance establishes ownership.  A root belonging to a
            # different run is foreign and must be rejected without moving it.
            provenance, _ = _read_regular_entry_at(
                root_descriptor,
                _ROOT_FILE,
                "Exp417 checkpoint provenance",
            )
            if provenance != canonical_json_bytes(self._root_payload) + b"\n":
                raise ValueError("Exp417 checkpoint provenance mismatch")
            datasets_descriptor: int | None = None
            try:
                try:
                    datasets_descriptor, datasets_identity = (
                        _open_directory_at_verified(
                            root_descriptor,
                            "datasets",
                            "Exp417 checkpoint datasets directory",
                        )
                    )
                    records = self._load_records_from_descriptor(
                        root_descriptor,
                        datasets_descriptor,
                    )
                    progress = self._read_progress_from_descriptor(
                        root_descriptor
                    )
                    digests = _progress_digests(progress)
                    observed = {
                        name: record["payload_sha256"]
                        for name, record in records.items()
                    }
                    if digests != observed:
                        raise ValueError(
                            "Exp417 checkpoint progress accounting is invalid"
                        )
                except _DirectoryBindingError:
                    raise
                except ValueError as error:
                    if datasets_descriptor is not None:
                        _verify_directory_entry_identity_at(
                            root_descriptor,
                            "datasets",
                            datasets_identity,
                            "Exp417 checkpoint datasets directory",
                        )
                    self._verify_root_path_identity(root_identity)
                    quarantine = self._quarantine_root(
                        root_identity,
                        root_descriptor,
                    )
                    raise ValueError(
                        "Exp417 checkpoint state was quarantined at "
                        f"{quarantine}"
                    ) from error
                _verify_directory_entry_identity_at(
                    root_descriptor,
                    "datasets",
                    datasets_identity,
                    "Exp417 checkpoint datasets directory",
                )
                self._verify_root_path_identity(root_identity)
            finally:
                if datasets_descriptor is not None:
                    os.close(datasets_descriptor)

    def _validate_root(self) -> None:
        _require_directory(self.root, "Exp417 checkpoint root")
        _require_directory(self.datasets_dir, "Exp417 checkpoint datasets directory")
        inventory = {entry.name for entry in _scandir(self.root)}
        if inventory != {_ROOT_FILE, _PROGRESS_FILE, "datasets"}:
            raise ValueError("Exp417 checkpoint root inventory is invalid")
        raw = _read_regular(self.root / _ROOT_FILE, "Exp417 checkpoint provenance")
        expected = canonical_json_bytes(self._root_payload) + b"\n"
        if raw != expected:
            raise ValueError("Exp417 checkpoint provenance mismatch")
        self._read_progress()

    def _load_records(self) -> dict[str, dict[str, object]]:
        self._validate_root()
        entries = _scandir(self.datasets_dir)
        expected_names = {self._dataset_path(dataset).name for dataset in self.expected_datasets}
        result: dict[str, dict[str, object]] = {}
        for entry in entries:
            if entry.name not in expected_names or not entry.name.endswith(".json"):
                raise ValueError("Exp417 checkpoint dataset inventory is invalid")
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError("Exp417 checkpoint dataset entry is invalid")
            record = _record_from_bytes(_read_regular(Path(entry.path), "Exp417 dataset checkpoint"))
            dataset = record["dataset"]
            if not isinstance(dataset, str) or dataset not in self.expected_datasets:
                raise ValueError("Exp417 checkpoint contains a foreign dataset")
            if entry.name != self._dataset_path(dataset).name or dataset in result:
                raise ValueError("Exp417 checkpoint dataset identity is invalid")
            _validate_result_provenance(record["qualification"], self.provenance)
            result[dataset] = record
        return result

    def _load_records_from_descriptor(
        self,
        root_descriptor: int,
        datasets_descriptor: int | None = None,
    ) -> dict[str, dict[str, object]]:
        inventory = set(
            _list_directory_descriptor(
                root_descriptor, "Exp417 checkpoint root"
            )
        )
        if inventory != {_ROOT_FILE, _PROGRESS_FILE, "datasets"}:
            raise ValueError("Exp417 checkpoint root inventory is invalid")
        root_payload = _read_regular_at(
            root_descriptor,
            _ROOT_FILE,
            "Exp417 checkpoint provenance",
        )
        if root_payload != canonical_json_bytes(self._root_payload) + b"\n":
            raise ValueError("Exp417 checkpoint provenance mismatch")
        owns_datasets_descriptor = datasets_descriptor is None
        if datasets_descriptor is None:
            datasets_descriptor, datasets_identity = (
                _open_directory_at_verified(
                    root_descriptor,
                    "datasets",
                    "Exp417 checkpoint datasets directory",
                )
            )
        else:
            metadata = os.fstat(datasets_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    "Exp417 checkpoint datasets directory is invalid"
                )
            datasets_identity = (metadata.st_dev, metadata.st_ino)
            _verify_directory_entry_identity_at(
                root_descriptor,
                "datasets",
                datasets_identity,
                "Exp417 checkpoint datasets directory",
            )
        try:
            names = _list_directory_descriptor(
                datasets_descriptor, "Exp417 checkpoint datasets directory"
            )
            expected_names = {
                self._dataset_path(dataset).name
                for dataset in self.expected_datasets
            }
            result: dict[str, dict[str, object]] = {}
            for name in names:
                if name not in expected_names or not name.endswith(".json"):
                    raise ValueError(
                        "Exp417 checkpoint dataset inventory is invalid"
                    )
                record = _record_from_bytes(
                    _read_regular_at(
                        datasets_descriptor,
                        name,
                        "Exp417 dataset checkpoint",
                    )
                )
                dataset = record["dataset"]
                if (
                    not isinstance(dataset, str)
                    or dataset not in self.expected_datasets
                    or name != self._dataset_path(dataset).name
                    or dataset in result
                ):
                    raise ValueError(
                        "Exp417 checkpoint dataset identity is invalid"
                    )
                _validate_result_provenance(
                    record["qualification"], self.provenance
                )
                result[dataset] = record
            _verify_directory_entry_identity_at(
                root_descriptor,
                "datasets",
                datasets_identity,
                "Exp417 checkpoint datasets directory",
            )
            return result
        finally:
            if owns_datasets_descriptor:
                os.close(datasets_descriptor)

    def _read_write_metadata_at(
        self,
        root_descriptor: int,
        datasets_descriptor: int,
    ) -> tuple[
        bytes,
        tuple[int, int],
        bytes,
        tuple[int, int],
        dict[str, str],
        frozenset[str],
    ]:
        """Reconcile append metadata without reconstructing prior results."""
        inventory = set(
            _list_directory_descriptor(
                root_descriptor, "Exp417 checkpoint root"
            )
        )
        if inventory != {_ROOT_FILE, _PROGRESS_FILE, "datasets"}:
            raise ValueError("Exp417 checkpoint root inventory is invalid")
        root_payload, root_payload_identity = _read_regular_entry_at(
            root_descriptor,
            _ROOT_FILE,
            "Exp417 checkpoint provenance",
        )
        if root_payload != canonical_json_bytes(self._root_payload) + b"\n":
            raise ValueError("Exp417 checkpoint provenance mismatch")
        metadata = os.fstat(datasets_descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "Exp417 checkpoint datasets directory is invalid"
            )
        datasets_identity = (metadata.st_dev, metadata.st_ino)
        _verify_directory_entry_identity_at(
            root_descriptor,
            "datasets",
            datasets_identity,
            "Exp417 checkpoint datasets directory",
        )
        old_progress, old_progress_identity = _read_regular_entry_at(
            root_descriptor,
            _PROGRESS_FILE,
            "Exp417 checkpoint progress",
        )
        progress = self._progress_from_bytes(old_progress)
        digests = _progress_digests(progress)
        if set(digests) - set(self.expected_datasets):
            raise ValueError(
                "Exp417 checkpoint progress contains a foreign dataset"
            )
        names = self._validate_write_dataset_inventory_at(
            datasets_descriptor,
            digests,
        )
        _verify_directory_entry_identity_at(
            root_descriptor,
            "datasets",
            datasets_identity,
            "Exp417 checkpoint datasets directory",
        )
        return (
            root_payload,
            root_payload_identity,
            old_progress,
            old_progress_identity,
            digests,
            names,
        )

    def _validate_write_dataset_inventory_at(
        self,
        datasets_descriptor: int,
        record_digests: Mapping[str, str],
    ) -> frozenset[str]:
        names = _list_directory_descriptor(
            datasets_descriptor,
            "Exp417 checkpoint datasets directory",
        )
        if len(names) != len(set(names)):
            raise ValueError(
                "Exp417 checkpoint dataset inventory is invalid"
            )
        expected_names = {
            self._dataset_path(dataset).name
            for dataset in record_digests
        }
        if (
            len(expected_names) != len(record_digests)
            or set(names) != expected_names
        ):
            raise ValueError(
                "Exp417 checkpoint progress accounting is invalid"
            )
        for name in names:
            try:
                metadata = os.stat(
                    name,
                    dir_fd=datasets_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ValueError(
                    "Exp417 checkpoint dataset entry is invalid"
                ) from error
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "Exp417 checkpoint dataset entry is invalid"
                )
        return frozenset(names)

    def _dataset_path(self, dataset: str) -> Path:
        return self.datasets_dir / f"{sha256(dataset.encode('utf-8')).hexdigest()}.json"

    def _progress_bytes(
        self,
        *,
        completed: int,
        current_dataset: str | None,
        record_digests: Mapping[str, object],
    ) -> bytes:
        if type(completed) is not int or not 0 <= completed <= len(self.expected_datasets):
            raise ValueError("Exp417 checkpoint progress is invalid")
        if set(record_digests) - set(self.expected_datasets) or len(record_digests) != completed:
            raise ValueError("Exp417 checkpoint progress is invalid")
        digests = {name: _digest(value, "progress record digest") for name, value in sorted(record_digests.items())}
        payload = {"completed_datasets": completed, "current_dataset": current_dataset,
                   "expected_datasets": len(self.expected_datasets), "completed_record_digests": digests}
        return canonical_json_bytes(payload) + b"\n"

    def _read_progress(self) -> dict[str, object]:
        raw = _read_regular(self.root / _PROGRESS_FILE, "Exp417 checkpoint progress")
        return self._progress_from_bytes(raw)

    def _read_progress_from_descriptor(
        self, root_descriptor: int
    ) -> dict[str, object]:
        return self._progress_from_bytes(
            _read_regular_at(
                root_descriptor,
                _PROGRESS_FILE,
                "Exp417 checkpoint progress",
            )
        )

    def _progress_from_bytes(self, raw: bytes) -> dict[str, object]:
        payload = _json_object(raw, "Exp417 checkpoint progress")
        if set(payload) != {"completed_datasets", "current_dataset", "expected_datasets", "completed_record_digests"}:
            raise ValueError("Exp417 checkpoint progress is invalid")
        if (type(payload["completed_datasets"]) is not int or type(payload["expected_datasets"]) is not int
                or not 0 <= payload["completed_datasets"] <= len(self.expected_datasets)
                or payload["expected_datasets"] != len(self.expected_datasets)
                or payload["current_dataset"] is not None):
            raise ValueError("Exp417 checkpoint progress is invalid")
        _progress_digests(payload)
        if len(_progress_digests(payload)) != payload["completed_datasets"]:
            raise ValueError("Exp417 checkpoint progress is invalid")
        if raw != canonical_json_bytes(payload) + b"\n":
            raise ValueError("Exp417 checkpoint progress is noncanonical")
        return payload

    def _assert_progress_matches_records(self, records: Mapping[str, Mapping[str, object]]) -> None:
        digests = _progress_digests(self._read_progress())
        observed = {name: record["payload_sha256"] for name, record in records.items()}
        if digests != observed:
            raise ValueError("Exp417 checkpoint progress accounting is invalid")

    @contextmanager
    def _root_lock(
        self, *, exclusive: bool
    ) -> Iterator[tuple[int, int]]:
        with self._root_descriptor_lock(exclusive=exclusive) as (
            _,
            root_identity,
        ):
            yield root_identity

    @contextmanager
    def _root_descriptor_lock(
        self, *, exclusive: bool
    ) -> Iterator[tuple[int, tuple[int, int]]]:
        try:
            descriptor = os.open(self.root, _DIRECTORY_OPEN_FLAGS)
        except OSError as error:
            raise ValueError("Exp417 checkpoint root is invalid") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("Exp417 checkpoint root is invalid")
            root_identity = (metadata.st_dev, metadata.st_ino)
            fcntl.flock(
                descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            )
            try:
                self._verify_root_path_identity(root_identity)
                yield descriptor, root_identity
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _verify_root_path_identity(
        self, root_identity: tuple[int, int]
    ) -> None:
        try:
            metadata = os.stat(self.root, follow_symlinks=False)
        except OSError as error:
            raise ValueError("Exp417 checkpoint root identity changed") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != root_identity
        ):
            raise ValueError("Exp417 checkpoint root identity changed")

    def _replace_owned_progress_at(
        self,
        root_descriptor: int,
        payload: bytes,
        expected_identity: tuple[int, int],
    ) -> tuple[int, int]:
        captured_name, captured_identity = self._capture_rollback_entry(
            root_descriptor,
            _PROGRESS_FILE,
            root_descriptor,
            "progress-previous",
        )
        if captured_identity != expected_identity:
            raise ValueError(
                "Exp417 checkpoint foreign progress replacement was "
                f"preserved as {captured_name}"
            )
        return _atomic_write_at(
            root_descriptor,
            _PROGRESS_FILE,
            payload,
            replace=False,
            recovery_root_descriptor=root_descriptor,
        )

    def _rollback_transaction_at(
        self,
        datasets_descriptor: int,
        record_name: str | None,
        record_identity: tuple[int, int] | None,
        root_descriptor: int,
        old_provenance: bytes,
        old_provenance_identity: tuple[int, int],
        old_progress: bytes,
        old_progress_identity: tuple[int, int],
        progress_identity: tuple[int, int] | None,
        root_identity: tuple[int, int],
        datasets_identity: tuple[int, int],
    ) -> None:
        rollback_errors: list[BaseException] = []
        def rollback_entry_exists(
            descriptor: int, name: str
        ) -> bool:
            try:
                return _entry_exists_at(descriptor, name)
            except BaseException as error:
                rollback_errors.append(error)
                # A failed probe is not evidence of absence.  Continue with
                # descriptor-bound read/capture so later cleanup still runs.
                return True

        if not rollback_entry_exists(root_descriptor, _ROOT_FILE):
            current_provenance = None
            current_provenance_identity = None
        else:
            try:
                (
                    current_provenance,
                    current_provenance_identity,
                ) = _read_regular_entry_at(
                    root_descriptor,
                    _ROOT_FILE,
                    "Exp417 checkpoint provenance",
                )
            except BaseException as error:
                rollback_errors.append(error)
                current_provenance = None
                current_provenance_identity = None
        if (
            current_provenance != old_provenance
            or current_provenance_identity != old_provenance_identity
        ):
            if rollback_entry_exists(root_descriptor, _ROOT_FILE):
                try:
                    captured_name, captured_identity = (
                        self._capture_rollback_entry(
                            root_descriptor,
                            _ROOT_FILE,
                            root_descriptor,
                            "provenance",
                        )
                    )
                    if captured_identity != old_provenance_identity:
                        rollback_errors.append(ValueError(
                            "Exp417 checkpoint foreign provenance replacement "
                            f"was preserved as {captured_name}"
                        ))
                except BaseException as error:
                    rollback_errors.append(error)
            try:
                _atomic_write_at(
                    root_descriptor,
                    _ROOT_FILE,
                    old_provenance,
                    replace=False,
                    recovery_root_descriptor=root_descriptor,
                )
                os.fsync(root_descriptor)
            except BaseException as error:
                rollback_errors.append(error)
        if record_name is not None:
            if record_identity is None:
                rollback_errors.append(ValueError(
                    "Exp417 checkpoint published record identity is missing"
                ))
            elif rollback_entry_exists(datasets_descriptor, record_name):
                try:
                    captured_name, captured_identity = (
                        self._capture_rollback_entry(
                            datasets_descriptor,
                            record_name,
                            root_descriptor,
                            "record",
                        )
                    )
                    if captured_identity != record_identity:
                        rollback_errors.append(ValueError(
                            "Exp417 checkpoint foreign record replacement was "
                            f"preserved as {captured_name}"
                        ))
                except BaseException as error:
                    rollback_errors.append(error)
            else:
                rollback_errors.append(ValueError(
                    "Exp417 checkpoint published record disappeared before rollback"
                ))
        if not rollback_entry_exists(root_descriptor, _PROGRESS_FILE):
            current_progress = None
            current_progress_identity = None
        else:
            try:
                (
                    current_progress,
                    current_progress_identity,
                ) = _read_regular_entry_at(
                    root_descriptor,
                    _PROGRESS_FILE,
                    "Exp417 checkpoint progress",
                )
            except BaseException as error:
                rollback_errors.append(error)
                current_progress = None
                current_progress_identity = None
        if (
            current_progress != old_progress
            or current_progress_identity != old_progress_identity
        ):
            expected_current_identity = (
                progress_identity
                if progress_identity is not None
                else old_progress_identity
            )
            if rollback_entry_exists(root_descriptor, _PROGRESS_FILE):
                try:
                    captured_name, captured_identity = (
                        self._capture_rollback_entry(
                            root_descriptor,
                            _PROGRESS_FILE,
                            root_descriptor,
                            "progress",
                        )
                    )
                    if captured_identity != expected_current_identity:
                        rollback_errors.append(ValueError(
                            "Exp417 checkpoint foreign progress replacement was "
                            f"preserved as {captured_name}"
                        ))
                except BaseException as error:
                    rollback_errors.append(error)
            try:
                _atomic_write_at(
                    root_descriptor,
                    _PROGRESS_FILE,
                    old_progress,
                    replace=False,
                    recovery_root_descriptor=root_descriptor,
                )
                os.fsync(root_descriptor)
            except BaseException as error:
                rollback_errors.append(error)
        try:
            _verify_directory_entry_identity_at(
                root_descriptor,
                "datasets",
                datasets_identity,
                "Exp417 checkpoint datasets directory",
            )
        except BaseException as error:
            rollback_errors.append(error)
        try:
            self._verify_root_path_identity(root_identity)
        except BaseException as error:
            rollback_errors.append(error)
        if len(rollback_errors) == 1:
            raise rollback_errors[0]
        if rollback_errors:
            raise BaseExceptionGroup(
                "Exp417 checkpoint rollback encountered multiple failures",
                rollback_errors,
            )

    def _capture_rollback_entry(
        self,
        source_descriptor: int,
        source_name: str,
        root_descriptor: int,
        kind: str,
    ) -> tuple[str, tuple[int, int]]:
        try:
            parent_descriptor = os.open(
                "..", _DIRECTORY_OPEN_FLAGS, dir_fd=root_descriptor
            )
        except OSError as error:
            raise ValueError(
                "Exp417 checkpoint rollback parent is invalid"
            ) from error
        capture_name = (
            f".{self.root.name}.rollback-{kind}.{uuid.uuid4().hex}.json"
        )
        try:
            _atomic_rename_noreplace_at(
                source_descriptor,
                source_name,
                parent_descriptor,
                capture_name,
            )
            os.fsync(source_descriptor)
            os.fsync(parent_descriptor)
            metadata = os.stat(
                capture_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "Exp417 checkpoint rollback capture is invalid"
                )
            return capture_name, (metadata.st_dev, metadata.st_ino)
        finally:
            os.close(parent_descriptor)

    def _quarantine_root(
        self,
        root_identity: tuple[int, int],
        root_descriptor: int,
    ) -> Path:
        """Atomically capture only the provenance-owned root inode."""
        locked = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(locked.st_mode)
            or (locked.st_dev, locked.st_ino) != root_identity
        ):
            raise ValueError(
                "Exp417 checkpoint quarantine root identity changed"
            )
        provenance, _ = _read_regular_entry_at(
            root_descriptor,
            _ROOT_FILE,
            "Exp417 checkpoint provenance",
        )
        if provenance != canonical_json_bytes(self._root_payload) + b"\n":
            raise ValueError("Exp417 checkpoint provenance mismatch")
        self._verify_root_path_identity(root_identity)
        parent = self.root.parent
        _require_directory(parent, "Exp417 checkpoint parent")
        token = uuid.uuid4().hex
        quarantine = parent / f".{self.root.name}.quarantine.{token}"
        release = parent / f".{self.root.name}.recovery-placeholder.{token}"
        os.mkdir(quarantine, 0o700)
        placeholder = os.lstat(quarantine)
        placeholder_identity = (placeholder.st_dev, placeholder.st_ino)
        _fsync_directory(parent)
        _atomic_exchange(self.root, quarantine)
        captured = os.lstat(quarantine)
        captured_identity = (captured.st_dev, captured.st_ino)
        if not stat.S_ISDIR(captured.st_mode) or captured_identity != root_identity:
            # A foreign pathname replacement won the race.  Restore that exact
            # entry atomically; retain our empty capture slot and touch nothing.
            _atomic_exchange(self.root, quarantine)
            _fsync_directory(parent)
            raise ValueError(
                "Exp417 checkpoint root identity changed; foreign entry restored"
            )
        try:
            _atomic_rename_noreplace(self.root, release)
        except BaseException:
            # Restore the owned checkpoint to its original name when release is
            # unavailable.  Both exchange operands are still known identities.
            _atomic_exchange(self.root, quarantine)
            _fsync_directory(parent)
            raise
        released = os.lstat(release)
        released_identity = (released.st_dev, released.st_ino)
        if (
            not stat.S_ISDIR(released.st_mode)
            or released_identity != placeholder_identity
        ):
            # Preserve and restore a racing foreign replacement.  The owned
            # checkpoint remains intact under its unique quarantine name.
            _atomic_rename_noreplace(release, self.root)
            _fsync_directory(parent)
            raise ValueError(
                "Exp417 checkpoint quarantine restored a foreign replacement"
            )
        # The caller still holds the original directory descriptor locked;
        # opening the quarantine pathname confirms it names that same inode.
        descriptor = os.open(quarantine, _DIRECTORY_OPEN_FLAGS)
        try:
            locked = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(locked.st_mode)
                or (locked.st_dev, locked.st_ino) != root_identity
            ):
                raise ValueError(
                    "Exp417 checkpoint quarantine inode identity changed"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(parent)
        return quarantine


def _record_from_bytes(raw: bytes) -> dict[str, object]:
    payload = _json_object(raw, "Exp417 dataset checkpoint")
    expected = {"schema_version", "dataset", "raw_train_identity_sha256", "repaired_train_identity_sha256",
                "length_group_identity_sha256", "canonical_model_hashes", "score_calibration_hashes",
                "qualification", "candidate_coordinates", "backend_audit", "elapsed_seconds", "payload_sha256"}
    if set(payload) != expected or type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Exp417 checkpoint dataset schema is invalid")
    digest = payload["payload_sha256"]
    body = dict(payload)
    body.pop("payload_sha256")
    if not isinstance(digest, str) or sha256(canonical_json_bytes(body)).hexdigest() != digest:
        raise ValueError("Exp417 checkpoint payload hash mismatch")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ValueError("Exp417 checkpoint record is noncanonical")
    dataset = _string(payload["dataset"], "dataset")
    result = _result_from_payload(payload["qualification"])
    terminal = isinstance(result.decision_evidence, DatasetTerminalEvidence)
    raw_identity = payload["raw_train_identity_sha256"]
    if raw_identity is not None:
        _digest(raw_identity, "raw TRAIN identity")
    elif not terminal:
        raise ValueError("Exp417 decision checkpoint lacks raw TRAIN identity")
    repaired = payload["repaired_train_identity_sha256"]
    if repaired is not None:
        _digest(repaired, "repaired TRAIN identity")
    identities = _length_digest_mapping(payload["length_group_identity_sha256"], "length-group identities")
    model_hashes = _model_hash_mapping(payload["canonical_model_hashes"])
    calibrations = _length_digest_mapping(payload["score_calibration_hashes"], "score calibration hashes")
    if terminal and (raw_identity is not None or repaired is not None or identities or model_hashes or calibrations):
        raise ValueError("Exp417 terminal checkpoint must not include fabricated stage evidence")
    if result.dataset != dataset:
        raise ValueError("Exp417 checkpoint result identity is invalid")
    if not terminal:
        _validate_length_evidence(result, identities, model_hashes, calibrations)
    coordinates = payload["candidate_coordinates"]
    if not isinstance(coordinates, list) or coordinates != [[item.dataset, item.instance_index] for item in result.candidates]:
        raise ValueError("Exp417 checkpoint candidate coordinates are invalid")
    audit_payload = payload["backend_audit"]
    audit = None if audit_payload is None else _backend_from_payload(audit_payload)
    contract = result.decision_evidence.frozen_train_contract
    if terminal:
        if audit is not None:
            raise ValueError("Exp417 terminal checkpoint must not include backend audit")
    elif audit != contract.backend:
        raise ValueError("Exp417 checkpoint backend audit does not match result")
    elapsed = _finite_nonnegative(payload["elapsed_seconds"], "elapsed time")
    return {**payload, "qualification": result, "backend_audit": audit, "elapsed_seconds": elapsed}


def _result_payload(value: DatasetResult) -> dict[str, object]:
    return {"dataset": value.dataset, "test_size": value.test_size, "group_count": value.group_count,
            "groups": [_group_payload(group) for group in value.groups],
            "candidates": [_candidate_payload(candidate) for candidate in value.candidates],
            "decision_evidence": _evidence_payload(value.decision_evidence),
            "lanes": _lanes_payload(value.lanes), "context": _context_payload(value.context), "status": value.status}


def _result_from_payload(value: object) -> DatasetResult:
    data = _object_keys(value, {"dataset", "test_size", "group_count", "groups", "candidates", "decision_evidence", "lanes", "context", "status"}, "result")
    context = _context_from_payload(data["context"])
    lanes = _lanes_from_payload(data["lanes"])
    return DatasetResult(_string(data["dataset"], "dataset"), _integer(data["test_size"], "test size", 1),
        _integer(data["group_count"], "group count"), tuple(_group_from_payload(item) for item in _list(data["groups"], "groups")),
        tuple(_candidate_from_payload(item) for item in _list(data["candidates"], "candidates")),
        _evidence_from_payload(data["decision_evidence"]), lanes, context, _string(data["status"], "status"))


def _context_payload(value: FrozenRunContext) -> dict[str, object]:
    return {"database_sha256": value.database_sha256, "datasets": list(value.datasets), "lanes": _lanes_payload(value.lanes),
            "test_identity_manifest": {name: [[index, length] for index, length in entries] for name, entries in value.test_identity_manifest.items()},
            "code_commit": value.code_commit, "runtime_versions": dict(value.runtime_versions)}


def _context_from_payload(value: object) -> FrozenRunContext:
    data = _object_keys(value, {"database_sha256", "datasets", "lanes", "test_identity_manifest", "code_commit", "runtime_versions"}, "context")
    datasets = tuple(_string(item, "dataset") for item in _list(data["datasets"], "datasets"))
    manifest_data = _object(data["test_identity_manifest"], "test identity manifest")
    manifest = {name: tuple((_integer(item[0], "test index"), _integer(item[1], "test length", 1)) for item in _list(entries, "test identity entries") if isinstance(item, list) and len(item) == 2)
                for name, entries in manifest_data.items()}
    if set(manifest) != set(datasets) or any(len(manifest[name]) != len(_list(manifest_data[name], "test identity entries")) for name in manifest):
        raise ValueError("Exp417 checkpoint test identity manifest is invalid")
    versions = { _string(key, "runtime version name"): _string(item, "runtime version") for key, item in _object(data["runtime_versions"], "runtime versions").items() }
    return FrozenRunContext(_string(data["database_sha256"], "database hash"), datasets, _lanes_from_payload(data["lanes"]), manifest,
                            _string(data["code_commit"], "code commit"), versions)


def _lanes_payload(value: FrozenLaneSnapshot) -> dict[str, object]:
    return {name: {dataset: list(indices) for dataset, indices in getattr(value, name).items()} for name in ("hard", "standard", "priority", "automatic_normal")}


def _lanes_from_payload(value: object) -> FrozenLaneSnapshot:
    data = _object_keys(value, {"hard", "standard", "priority", "automatic_normal"}, "lanes")
    def lane(name: str) -> dict[str, tuple[int, ...]]:
        return {_string(dataset, "lane dataset"): tuple(_integer(index, "lane index") for index in _list(indices, "lane indices")) for dataset, indices in _object(data[name], "lane").items()}
    return FrozenLaneSnapshot(hard=lane("hard"), standard=lane("standard"), priority=lane("priority"), automatic_normal=lane("automatic_normal"))


def _group_payload(value: GroupQualification) -> dict[str, object]:
    return {"length": value.length, "series_count": value.series_count, "status": value.status, "oof_candidate_rate": value.oof_candidate_rate,
            "wilson_upper": value.wilson_upper, "pooled_synthetic_sensitivity": value.pooled_synthetic_sensitivity,
            "per_type_synthetic_sensitivity": dict(value.per_type_synthetic_sensitivity), "seed_agreement": value.seed_agreement,
            "finite": value.finite, "backend_fallback": value.backend_fallback}


def _group_from_payload(value: object) -> GroupQualification:
    data = _object_keys(value, {"length", "series_count", "status", "oof_candidate_rate", "wilson_upper", "pooled_synthetic_sensitivity", "per_type_synthetic_sensitivity", "seed_agreement", "finite", "backend_fallback"}, "group")
    return GroupQualification(_integer(data["length"], "length", 1), _integer(data["series_count"], "series count"), _string(data["status"], "group status"),
        _optional_finite(data["oof_candidate_rate"], "oof rate"),
        _optional_finite(data["wilson_upper"], "wilson upper"),
        _optional_finite(data["pooled_synthetic_sensitivity"], "sensitivity"),
        {
            _string(key, "perturbation"): _optional_finite(item, "sensitivity")
            for key, item in _object(
                data["per_type_synthetic_sensitivity"], "sensitivity map"
            ).items()
        },
        _optional_bool(data["seed_agreement"], "seed agreement"),
        _optional_bool(data["finite"], "finite"),
        _optional_bool(data["backend_fallback"], "backend fallback"))


def _backend_payload(value: BackendAudit) -> dict[str, object]:
    return asdict(value)


def _backend_from_payload(value: object) -> BackendAudit:
    data = _object_keys(value, {"backend", "candidate_identity", "max_normalized_error", "spearman", "speedup", "fallback", "fallback_reason"}, "backend audit")
    reason = data["fallback_reason"]
    if reason is not None:
        reason = _string(reason, "fallback reason")
    return BackendAudit(_string(data["backend"], "backend"), _bool(data["candidate_identity"], "candidate identity"),
        _finite(data["max_normalized_error"], "max normalized error"), _finite(data["spearman"], "spearman"), _finite(data["speedup"], "speedup"), _bool(data["fallback"], "fallback"), reason)


def _seed_payload(value: SeedDecision) -> dict[str, object]:
    return {"seed": value.seed, "scores": {str(horizon): {"horizon": score.horizon, "score": score.score, "p_value": score.p_value, "persistence": score.persistence} for horizon, score in value.scores.items()}, "candidate": value.candidate}


def _seed_from_payload(value: object) -> SeedDecision:
    data = _object_keys(value, {"seed", "scores", "candidate"}, "seed decision")
    scores: dict[int, HorizonScore] = {}
    raw_scores = _object(data["scores"], "scores")
    for key in sorted(raw_scores, key=lambda item: _mapping_integer(item, "horizon", 1)):
        item = raw_scores[key]
        score = _object_keys(item, {"horizon", "score", "p_value", "persistence"}, "horizon score")
        horizon = _mapping_integer(key, "horizon", 1)
        scores[horizon] = HorizonScore(_integer(score["horizon"], "horizon", 1), _finite(score["score"], "score"), _finite(score["p_value"], "p value"), _finite(score["persistence"], "persistence"))
    return SeedDecision(_integer(data["seed"], "seed", 1), scores, _bool(data["candidate"], "candidate"))


def _candidate_payload(value: ShadowCandidate) -> dict[str, object]:
    return {"dataset": value.dataset, "instance_index": value.instance_index, "length": value.length, "seed_decisions": [_seed_payload(item) for item in value.seed_decisions]}


def _candidate_from_payload(value: object) -> ShadowCandidate:
    data = _object_keys(value, {"dataset", "instance_index", "length", "seed_decisions"}, "candidate")
    return ShadowCandidate(_string(data["dataset"], "dataset"), _integer(data["instance_index"], "instance index"), _integer(data["length"], "length", 1), tuple(_seed_from_payload(item) for item in _list(data["seed_decisions"], "seed decisions")))


def _contract_payload(value: FrozenTrainContract) -> dict[str, object]:
    return {"dataset": value.dataset, "groups": [_group_payload(item) for item in value.groups],
            "backend": _backend_payload(value.backend), "train_identity_sha256": value.train_identity_sha256,
            "near_repeat_status": value.near_repeat_status}


def _contract_from_payload(value: object) -> FrozenTrainContract:
    train = _object_keys(value, {"dataset", "groups", "backend", "train_identity_sha256", "near_repeat_status"}, "frozen train contract")
    return FrozenTrainContract(_string(train["dataset"], "dataset"), tuple(_group_from_payload(item) for item in _list(train["groups"], "groups")), _backend_from_payload(train["backend"]), _string(train["train_identity_sha256"], "train identity"), _string(train["near_repeat_status"], "near repeat status"))


def _evidence_payload(value: DatasetDecisionEvidence | DatasetTerminalEvidence) -> dict[str, object]:
    if isinstance(value, DatasetTerminalEvidence):
        return {"kind": "terminal", "dataset": value.dataset, "context": _context_payload(value.context),
                "stage": value.stage, "reason": value.reason,
                "frozen_train_contract": None if value.frozen_train_contract is None else _contract_payload(value.frozen_train_contract)}
    return {"kind": "decision", "dataset": value.dataset, "context": _context_payload(value.context), "frozen_train_contract": _contract_payload(value.frozen_train_contract),
            "database_sha256": value.database_sha256, "train_identity_sha256": value.train_identity_sha256,
            "decisions": [{"dataset": item.dataset, "instance_index": item.instance_index, "length": item.length, "seed_decisions": [_seed_payload(seed) for seed in item.seed_decisions]} for item in value.decisions]}


def _evidence_from_payload(value: object) -> DatasetDecisionEvidence | DatasetTerminalEvidence:
    raw = _object(value, "evidence")
    kind = raw.get("kind")
    if kind == "terminal":
        data = _object_keys(raw, {"kind", "dataset", "context", "stage", "reason", "frozen_train_contract"}, "terminal evidence")
        contract_raw = data["frozen_train_contract"]
        return DatasetTerminalEvidence(_string(data["dataset"], "dataset"), _context_from_payload(data["context"]),
            _string(data["stage"], "terminal stage"), _string(data["reason"], "terminal reason"),
            None if contract_raw is None else _contract_from_payload(contract_raw))
    if kind != "decision":
        raise ValueError("Exp417 checkpoint evidence kind is invalid")
    data = _object_keys(raw, {"kind", "dataset", "context", "frozen_train_contract", "database_sha256", "train_identity_sha256", "decisions"}, "decision evidence")
    contract = _contract_from_payload(data["frozen_train_contract"])
    decisions = []
    for item in _list(data["decisions"], "decisions"):
        decision = _object_keys(item, {"dataset", "instance_index", "length", "seed_decisions"}, "instance decision")
        decisions.append(TestInstanceDecision(_string(decision["dataset"], "dataset"), _integer(decision["instance_index"], "instance index"), _integer(decision["length"], "length", 1), tuple(_seed_from_payload(seed) for seed in _list(decision["seed_decisions"], "seed decisions"))))
    return DatasetDecisionEvidence(_string(data["dataset"], "dataset"), _context_from_payload(data["context"]), contract, _string(data["database_sha256"], "database hash"), _string(data["train_identity_sha256"], "train identity"), tuple(decisions))


def _validate_provenance(value: object) -> None:
    data = _object_keys(value, {"git_commit", "database_sha256", "exp291_prediction_sha256", "exp291_manifest_sha256", "exp292_prediction_sha256", "exp292_manifest_sha256", "frozen_config_sha256", "preprocessing_contract_sha256", "fold_contract_sha256", "model_contract_sha256", "model_seeds", "perturbation_seeds", "runtime_versions", "backend_contract_sha256"}, "provenance")
    _string(data["git_commit"], "git commit")
    for field in ("database_sha256", "exp291_prediction_sha256", "exp291_manifest_sha256", "exp292_prediction_sha256", "exp292_manifest_sha256", "frozen_config_sha256", "preprocessing_contract_sha256", "fold_contract_sha256", "model_contract_sha256", "backend_contract_sha256"):
        _digest(data[field], field)
    for field in ("model_seeds", "perturbation_seeds"):
        values = tuple(_integer(item, field, 1) for item in _sequence(data[field], field))
        expected = FROZEN_CONFIG.model_seeds if field == "model_seeds" else FROZEN_CONFIG.perturbation_seeds
        if values != expected: raise ValueError("Exp417 checkpoint provenance seeds are invalid")
    versions = _object(data["runtime_versions"], "runtime versions")
    if not versions or any(not isinstance(key, str) or not isinstance(item, str) or not key or not item for key, item in versions.items()):
        raise ValueError("Exp417 checkpoint provenance runtime versions are invalid")


def _validate_result_provenance(result: object, provenance: Exp417CheckpointProvenance) -> None:
    if not isinstance(result, DatasetResult):
        raise ValueError("Exp417 checkpoint result is invalid")
    context = result.context
    lanes = context.lanes
    if (context.database_sha256 != provenance.database_sha256
            or context.code_commit != provenance.git_commit
            or dict(context.runtime_versions) != dict(provenance.runtime_versions)
            or lanes.exp291_prediction_sha256 != provenance.exp291_prediction_sha256
            or lanes.exp291_manifest_sha256 != provenance.exp291_manifest_sha256
            or lanes.exp292_prediction_sha256 != provenance.exp292_prediction_sha256
            or lanes.exp292_manifest_sha256 != provenance.exp292_manifest_sha256):
        raise ValueError("Exp417 checkpoint result provenance mismatch")


def _validate_length_evidence(
    result: DatasetResult,
    identities: Mapping[int, str],
    model_hashes: Mapping[int, Mapping[int, str]],
    calibrations: Mapping[int, str],
) -> None:
    lengths = tuple(group.length for group in result.groups)
    identity_lengths = set(identities)
    model_lengths = set(model_hashes)
    calibration_lengths = set(calibrations)
    qualified_lengths = {
        group.length for group in result.groups if group.status == "qualified"
    }
    diagnostic_lengths = {
        group.length for group in result.groups if group.status == "diagnostic"
    }
    failed_lengths = {
        group.length for group in result.groups if group.status == "failed"
    }
    expected_calibrations = (
        qualified_lengths | diagnostic_lengths | failed_lengths
    )
    expected_models = qualified_lengths | failed_lengths
    if (
        tuple(identities) != lengths
        or identity_lengths != set(lengths)
        or calibration_lengths != expected_calibrations
        or model_lengths != expected_models
    ):
        raise ValueError("Exp417 checkpoint length-group accounting is invalid")


def _dataset_universe(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence): raise ValueError("Exp417 checkpoint dataset universe is invalid")
    names = tuple(values)
    if not names or any(not isinstance(item, str) or not item for item in names) or names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("Exp417 checkpoint dataset universe is invalid")
    return names


def _length_digest_mapping(value: object, name: str) -> dict[int, str]:
    data = _integer_key_mapping(value, name)
    result: dict[int, str] = {}
    for key, item in data.items():
        length = _mapping_integer(key, name, 1)
        if length in result:
            raise ValueError(f"Exp417 checkpoint {name} contain an alias")
        result[length] = _digest(item, name)
    return dict(sorted(result.items()))


def _model_hash_mapping(value: object) -> dict[int, dict[int, str]]:
    data = _integer_key_mapping(value, "canonical model hashes"); result: dict[int, dict[int, str]] = {}
    for raw_length, raw_hashes in data.items():
        length = _mapping_integer(raw_length, "model length", 1)
        if length in result:
            raise ValueError("Exp417 checkpoint model lengths contain an alias")
        hashes: dict[int, str] = {}
        for raw_seed, digest in _integer_key_mapping(
            raw_hashes, "model hashes"
        ).items():
            seed = _mapping_integer(raw_seed, "model seed", 1)
            if seed in hashes:
                raise ValueError(
                    "Exp417 checkpoint model seeds contain an alias"
                )
            hashes[seed] = _digest(digest, "model hash")
        hashes = dict(sorted(hashes.items()))
        if tuple(hashes) != FROZEN_CONFIG.model_seeds:
            raise ValueError("Exp417 checkpoint model seed accounting is invalid")
        result[length] = hashes
    return dict(sorted(result.items()))


def _entry_name(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
    ):
        raise ValueError(f"{context} is invalid")
    return value


def _entry_exists_at(directory_descriptor: int, name: str) -> bool:
    checked = _entry_name(name, "Exp417 checkpoint entry")
    try:
        os.stat(checked, dir_fd=directory_descriptor, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("Exp417 checkpoint entry is invalid") from error


def _entry_identity_at_optional(
    directory_descriptor: int, name: str, context: str
) -> tuple[int, int] | None:
    checked = _entry_name(name, context)
    try:
        metadata = os.stat(
            checked,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"{context} is invalid") from error
    return metadata.st_dev, metadata.st_ino


def _preserve_root_publish_target_at(
    parent_descriptor: int,
    root_name: str,
) -> tuple[str, tuple[int, int]] | None:
    checked_root = _entry_name(
        root_name, "Exp417 checkpoint root recovery source"
    )
    capture_name = (
        ".exp417-checkpoint.rollback-root-publish-target."
        f"{uuid.uuid4().hex}.entry"
    )
    try:
        _atomic_rename_noreplace_at(
            parent_descriptor,
            checked_root,
            parent_descriptor,
            capture_name,
        )
    except FileNotFoundError:
        return None
    os.fsync(parent_descriptor)
    identity = _entry_identity_at_optional(
        parent_descriptor,
        capture_name,
        "Exp417 checkpoint root recovery entry",
    )
    if identity is None:
        raise ValueError(
            "Exp417 checkpoint root recovery entry is missing"
        )
    return capture_name, identity


def _regular_identity_at(
    directory_descriptor: int, name: str, context: str
) -> tuple[int, int]:
    checked = _entry_name(name, context)
    try:
        metadata = os.stat(
            checked,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ValueError(f"{context} is invalid") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{context} is invalid")
    return metadata.st_dev, metadata.st_ino


def _preserve_entry_at(
    directory_descriptor: int,
    source_name: str,
    recovery_root_descriptor: int,
    kind: str,
) -> tuple[str, tuple[int, int]] | None:
    """Atomically move any basename occupant out; never unlink by name."""
    checked_kind = _entry_name(
        kind, "Exp417 checkpoint recovery kind"
    )
    try:
        parent_descriptor = os.open(
            "..",
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=recovery_root_descriptor,
        )
    except OSError as error:
        raise ValueError(
            "Exp417 checkpoint temporary recovery parent is invalid"
        ) from error
    capture_name = (
        f".exp417-checkpoint.rollback-{checked_kind}."
        f"{uuid.uuid4().hex}.bin"
    )
    try:
        try:
            _atomic_rename_noreplace_at(
                directory_descriptor,
                source_name,
                parent_descriptor,
                capture_name,
            )
        except FileNotFoundError:
            return None
        os.fsync(directory_descriptor)
        os.fsync(parent_descriptor)
        metadata = os.stat(
            capture_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                "Exp417 checkpoint recovery entry is invalid"
            )
        return capture_name, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(parent_descriptor)


def _preserve_temporary_entry_at(
    directory_descriptor: int,
    temporary: str,
    recovery_root_descriptor: int,
) -> tuple[str, tuple[int, int]] | None:
    return _preserve_entry_at(
        directory_descriptor,
        temporary,
        recovery_root_descriptor,
        "temporary",
    )


def _write_temporary_at(
    directory_descriptor: int,
    target_name: str,
    data: bytes,
    *,
    recovery_root_descriptor: int,
) -> tuple[str, tuple[int, int]]:
    if type(data) is not bytes:
        raise ValueError("Exp417 checkpoint payload is invalid")
    temporary = f".checkpoint-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ValueError("Exp417 checkpoint temporary file is invalid") from error
    identity: tuple[int, int] | None = None
    failures: list[BaseException] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Exp417 checkpoint temporary file is invalid")
        identity = (metadata.st_dev, metadata.st_ino)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("Exp417 checkpoint write made no progress")
            offset += written
        os.fsync(descriptor)
    except BaseException as error:
        failures.append(error)
    try:
        os.close(descriptor)
    except BaseException as error:
        failures.append(error)
    if failures:
        preserve_succeeded = False
        try:
            captured = _preserve_temporary_entry_at(
                directory_descriptor,
                temporary,
                recovery_root_descriptor,
            )
            preserve_succeeded = True
        except BaseException as preserve_error:
            failures.append(preserve_error)
            captured = None
        if (
            identity is not None
            and captured is not None
            and captured[1] != identity
        ):
            failures.append(ValueError(
                "Exp417 checkpoint foreign temporary replacement was "
                f"preserved as {captured[0]}"
            ))
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup(
            "Exp417 checkpoint temporary write recovery failed",
            failures,
        )
    if identity is None:
        raise ValueError(
            "Exp417 checkpoint temporary identity is unavailable"
        )
    return temporary, identity


def _atomic_write_exclusive_at(
    directory_descriptor: int,
    name: str,
    data: bytes,
    *,
    recovery_root_descriptor: int,
) -> tuple[int, int]:
    checked = _entry_name(name, "Exp417 checkpoint record name")
    temporary, identity = _write_temporary_at(
        directory_descriptor,
        checked,
        data,
        recovery_root_descriptor=recovery_root_descriptor,
    )
    try:
        _atomic_rename_noreplace_at(
            directory_descriptor,
            temporary,
            directory_descriptor,
            checked,
        )
        return identity
    except BaseException as error:
        failures: list[BaseException] = [error]
        try:
            target_identity = _regular_identity_at(
                directory_descriptor,
                checked,
                "Exp417 checkpoint published record",
            )
        except ValueError:
            target_identity = None
        except BaseException as target_error:
            failures.append(target_error)
            target_identity = None
        try:
            source_exists: bool | None = _entry_exists_at(
                directory_descriptor, temporary
            )
        except BaseException as probe_error:
            failures.append(probe_error)
            source_exists = None
        try:
            captured = _preserve_temporary_entry_at(
                directory_descriptor,
                temporary,
                recovery_root_descriptor,
            )
        except BaseException as preserve_error:
            failures.append(preserve_error)
            captured = None
        if captured is not None and captured[1] != identity:
            failures.append(ValueError(
                "Exp417 checkpoint foreign temporary replacement was "
                f"preserved as {captured[0]}"
            ))
        target_preserve_failed = False
        try:
            captured_target = _preserve_entry_at(
                directory_descriptor,
                checked,
                recovery_root_descriptor,
                "publish-target",
            )
        except BaseException as preserve_error:
            failures.append(preserve_error)
            captured_target = None
            target_preserve_failed = True
        if (
            captured_target is not None
            and target_identity is not None
            and captured_target[1] != target_identity
        ):
            failures.append(ValueError(
                "Exp417 checkpoint publish target changed during recovery; "
                f"latest entry was preserved as {captured_target[0]}"
            ))
        if (
            captured_target is not None
            and captured_target[1] != identity
        ):
            failures.append(ValueError(
                "Exp417 checkpoint foreign publish target was preserved as "
                f"{captured_target[0]}"
            ))
        if len(failures) == 1:
            combined_error = failures[0]
        else:
            combined_error = BaseExceptionGroup(
                "Exp417 checkpoint publish recovery failed",
                failures,
            )
        if target_preserve_failed and target_identity is not None:
            raise _CommittedWriteError(
                combined_error,
                target_identity,
            ) from combined_error
        raise combined_error


def _atomic_write_at(
    directory_descriptor: int,
    name: str,
    data: bytes,
    *,
    replace: bool,
    recovery_root_descriptor: int,
) -> tuple[int, int]:
    checked = _entry_name(name, "Exp417 checkpoint target name")
    if replace is not False:
        raise ValueError(
            "Exp417 checkpoint descriptor writes must not replace entries"
        )
    return _atomic_write_exclusive_at(
        directory_descriptor,
        checked,
        data,
        recovery_root_descriptor=recovery_root_descriptor,
    )


def _atomic_exchange(first: Path, second: Path) -> None:
    """Atomically exchange two directory entries without a pathname race."""
    first_bytes = os.fsencode(os.path.abspath(first))
    second_bytes = os.fsencode(os.path.abspath(second))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            exchange = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic directory exchange is unavailable"
            ) from error
        exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        exchange.restype = ctypes.c_int
        if exchange(first_bytes, second_bytes, _RENAME_SWAP) != 0:
            _raise_rename_error(first)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            exchange = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic directory exchange is unavailable"
            ) from error
        exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        exchange.restype = ctypes.c_int
        if (
            exchange(
                _LINUX_AT_FDCWD,
                first_bytes,
                _LINUX_AT_FDCWD,
                second_bytes,
                _RENAME_EXCHANGE,
            )
            != 0
        ):
            _raise_rename_error(first)
        return
    raise OSError(errno.ENOTSUP, "atomic directory exchange is unavailable")


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename while refusing to overwrite a competing entry."""
    source_bytes = os.fsencode(os.path.abspath(source))
    destination_bytes = os.fsencode(os.path.abspath(destination))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(source_bytes, destination_bytes, _RENAME_EXCL) != 0:
            _raise_rename_error(destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if (
            rename(
                _LINUX_AT_FDCWD,
                source_bytes,
                _LINUX_AT_FDCWD,
                destination_bytes,
                _RENAME_NOREPLACE,
            )
            != 0
        ):
            _raise_rename_error(destination)
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")


def _atomic_rename_noreplace_at(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    source_bytes = os.fsencode(
        _entry_name(source_name, "Exp417 checkpoint rename source")
    )
    destination_bytes = os.fsencode(
        _entry_name(
            destination_name, "Exp417 checkpoint rename destination"
        )
    )
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameatx_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if (
            rename(
                source_descriptor,
                source_bytes,
                destination_descriptor,
                destination_bytes,
                _RENAME_EXCL,
            )
            != 0
        ):
            _raise_rename_error(Path(destination_name))
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP, "atomic no-replace rename is unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if (
            rename(
                source_descriptor,
                source_bytes,
                destination_descriptor,
                destination_bytes,
                _RENAME_NOREPLACE,
            )
            != 0
        ):
            _raise_rename_error(Path(destination_name))
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")


def _raise_rename_error(path: Path) -> None:
    number = ctypes.get_errno()
    if number == errno.ENOENT:
        raise FileNotFoundError(number, os.strerror(number), str(path))
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(number, os.strerror(number), str(path))
    raise OSError(number, os.strerror(number), str(path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _scandir(path: Path) -> list[os.DirEntry[str]]:
    _require_directory(path, "Exp417 checkpoint directory")
    try:
        with os.scandir(path) as scan: return list(scan)
    except OSError as error: raise ValueError("Exp417 checkpoint inventory is unreadable") from error


def _list_directory_descriptor(descriptor: int, name: str) -> list[str]:
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{name} is invalid")
        entries = os.listdir(descriptor)
    except OSError as error:
        raise ValueError(f"{name} inventory is unreadable") from error
    if any(not isinstance(entry, str) or not entry for entry in entries):
        raise ValueError(f"{name} inventory is invalid")
    return entries


def _directory_inventory_at(
    descriptor: int, name: str
) -> dict[str, os.stat_result]:
    inventory: dict[str, os.stat_result] = {}
    for entry in _list_directory_descriptor(descriptor, name):
        try:
            inventory[entry] = os.stat(
                entry,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ValueError(f"{name} inventory changed") from error
    return inventory


def _open_directory_at_verified(
    parent_descriptor: int, name: str, context: str
) -> tuple[int, tuple[int, int]]:
    checked = _entry_name(name, context)
    try:
        descriptor = os.open(
            checked, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
        )
    except OSError as error:
        raise ValueError(f"{context} is invalid") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{context} is invalid")
        identity = (metadata.st_dev, metadata.st_ino)
        _verify_directory_entry_identity_at(
            parent_descriptor, checked, identity, context
        )
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory_entry_identity_at(
    parent_descriptor: int,
    name: str,
    identity: tuple[int, int],
    context: str,
) -> None:
    checked = _entry_name(name, context)
    try:
        metadata = os.stat(
            checked,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _DirectoryBindingError(
            f"{context} identity changed"
        ) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise _DirectoryBindingError(
            f"{context} identity changed"
        )


def _read_regular_entry_at(
    directory_descriptor: int, name: str, context: str
) -> tuple[bytes, tuple[int, int]]:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError(f"{context} is invalid")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ValueError(f"{context} is invalid") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{context} is invalid")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 65536):
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"{context} changed while being read")
        try:
            named = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ValueError(f"{context} identity changed") from error
        if (
            not stat.S_ISREG(named.st_mode)
            or (named.st_dev, named.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{context} identity changed")
        return b"".join(chunks), (before.st_dev, before.st_ino)
    except OSError as error:
        raise ValueError(f"{context} is invalid") from error
    finally:
        os.close(descriptor)


def _read_regular_at(directory_descriptor: int, name: str, context: str) -> bytes:
    payload, _ = _read_regular_entry_at(directory_descriptor, name, context)
    return payload


def _read_regular(path: Path, name: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error: raise ValueError(f"{name} is invalid") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode): raise ValueError(f"{name} is invalid")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 65536): chunks.append(block)
        return b"".join(chunks)
    except OSError as error: raise ValueError(f"{name} is invalid") from error
    finally: os.close(descriptor)


def _require_directory(path: Path, name: str) -> None:
    try: metadata = os.lstat(path)
    except OSError as error: raise ValueError(f"{name} is invalid") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode): raise ValueError(f"{name} is invalid")


def _path_exists_lstat(path: Path) -> bool:
    try: os.lstat(path); return True
    except FileNotFoundError: return False




def _json_object(raw: bytes, name: str) -> dict[str, object]:
    try: value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error: raise ValueError(f"{name} is corrupt") from error
    return _object(value, name)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _integer_key_mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping) or any(type(key) is not int and not isinstance(key, str) for key in value):
        raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _object_keys(value: object, keys: set[str], name: str) -> dict[str, object]:
    data = _object(value, name)
    if set(data) != keys: raise ValueError(f"Exp417 checkpoint {name} schema is invalid")
    return data


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value: raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum: raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _mapping_integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is int:
        number = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        number = int(value)
    else:
        raise ValueError(f"Exp417 checkpoint {name} is invalid")
    if number < minimum or (isinstance(value, str) and value != str(number)):
        raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return number


def _progress_digests(payload: Mapping[str, object]) -> dict[str, str]:
    value = payload.get("completed_record_digests")
    if not isinstance(value, dict):
        raise ValueError("Exp417 checkpoint progress is invalid")
    result: dict[str, str] = {}
    for dataset, digest in value.items():
        name = _string(dataset, "progress dataset")
        if name in result:
            raise ValueError("Exp417 checkpoint progress is invalid")
        result[name] = _digest(digest, "progress record digest")
    if tuple(result) != tuple(sorted(result)):
        raise ValueError("Exp417 checkpoint progress is invalid")
    return result


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool: raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, name)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return float(value)


def _optional_finite(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name)


def _finite_nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0: raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return number


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


__all__ = [
    "Exp417CheckpointProvenance",
    "Exp417CheckpointStore",
    "canonical_result_sha256",
]
