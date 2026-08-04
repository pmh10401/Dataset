"""Read-only, descriptor-bound import of verified Exp424 checkpoint evidence.

This module deliberately does no model or database work.  It only validates the
sealed Exp424 checkpoint bytes and uses Task 2's private capability broker to
publish compatible, non-terminal evidence into an Exp425 checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from hashlib import sha256
import copy
import fcntl
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any
import uuid

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    _record_from_bytes,
    canonical_result_sha256,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetDecisionEvidence,
    DatasetResult,
    DatasetTerminalEvidence,
    FrozenRunContext,
)

from . import capabilities
from .checkpoints import Exp425CheckpointStore, _import_record_body, record_filename
from .contracts import ImportedRecordEnvelope, SourceProvenGroupEvidence


EXP424_SOURCE_ROOT = Path("/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint")
EXP424_ROOT_PROVENANCE_SHA256 = "f7526bf638f4c1051a5970271bdbbccd753986f7d067ec1260817da884c1b757"
EXP424_PROGRESS_SHA256 = "5e9aa4a4b7240ac89c078bdc1dce4b41bda6b6dbc505dd9ad49782556da7bd84"
EXP424_RECORD_SET_SHA256 = "d74a0def7788db89a1d3e32f46662abf07c44c7cbbce5e02612b510e2fdf4258"
EXP424_SCORE_CACHE_CONTRACT = {
    "schema": "exp424.score-cache-contract.v1",
    "components": ["oof_score_calibration_cache", "reusable_batch_residual_predictor"],
    "score_formula": "Exp417.score_horizon.v1", "oof_residual_dtype": "float64",
    "percentile_method": "linear", "residual_batch_size": 32, "query_chunk_size": 256,
    "model_materialization": "once_per_seed_predictor_per_scoring_stage",
    "candidate_coordinates": "exact_exp417_parity",
}
EXP424_SCORE_CACHE_CONTRACT_SHA256 = "82d6f3b431b2d57f45f2aae33ee4f16729709b0090f8c4984bbdf5557baff257"
EXP424_MPS_SYNC_CONTRACT = {
    "schema": "exp424.mps-synchronization-contract.v1", "device": "mps",
    "device_guard": "mps_and_mps0_equivalent",
    "input_transfer": "contiguous_float32_numpy_copy_to_torch_then_to_mps",
    "output_transfer": "per_prediction_horizon_detach_cpu_numpy",
    "implicit_sync_boundary": "mps_to_cpu_materialization",
    "explicit_torch_mps_synchronize_calls": 0, "output_residual_dtype": "float64",
}
EXP424_MPS_SYNC_CONTRACT_SHA256 = "e1e928ddde0a4793a63cb95a6a643888251e4ddabba9b2bbedd12859df95133a"

_OPEN_DIRECTORY = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_OPEN_FILE = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_CONSTRUCTION_SENTINEL = object()


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _canonical_object(raw: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON")
    return value


@dataclass(frozen=True)
class Exp424CompatibilityContract:
    """Frozen source descriptor and numerical/runtime contract for Exp424."""

    source_real_path: Path
    root_provenance_sha256: str
    progress_sha256: str
    record_set_sha256: str
    database_sha256: str | None = None
    lane_hashes: Mapping[str, str] = field(default_factory=dict)
    config_sha256: str | None = None
    fold_contract_sha256: str | None = None
    model_contract_sha256: str | None = None
    model_seeds: tuple[int, ...] = ()
    perturbation_seeds: tuple[int, ...] = ()
    backend_contract_sha256: str | None = None
    runtime_versions: Mapping[str, str] = field(default_factory=dict)
    device: str = "mps"
    mps_device_guard: str = "mps_and_mps0_equivalent"
    batch_size: int = 32
    query_chunk_size: int = 256
    score_formula: str = "Exp417.score_horizon.v1"
    score_cache_contract_sha256: str = EXP424_SCORE_CACHE_CONTRACT_SHA256
    percentile_method: str = "linear"
    score_dtype: str = "float64"
    thresholds_payload: Mapping[str, object] = field(default_factory=dict)
    alpha_values: tuple[float, ...] = ()
    candidate_formula_sha256: str | None = None
    routing_contract_sha256: str | None = None
    synchronization_contract_sha256: str = EXP424_MPS_SYNC_CONTRACT_SHA256
    entrypoint_sha256: str = "98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1"
    entrypoint_git_blob: str = "da5374b0bf38803aa506dfda664d76395d7f28f9"
    source_git_commit: str = "4a398374d880f859f285a271ad63de4a5e55b028"
    _fixture: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        path = Path(self.source_real_path).resolve(strict=False)
        object.__setattr__(self, "source_real_path", path)
        for name in ("root_provenance_sha256", "progress_sha256", "record_set_sha256", "score_cache_contract_sha256", "synchronization_contract_sha256", "entrypoint_sha256"):
            _sha256(getattr(self, name), name)
        if self.entrypoint_sha256 != "98c909f6100dd2814e3fe102c3f95054a3d6bb42f85657d19b17780ae4934fe1" or self.entrypoint_git_blob != "da5374b0bf38803aa506dfda664d76395d7f28f9":
            raise ValueError("Exp424 entrypoint contract is incompatible")
        if self.source_git_commit != "4a398374d880f859f285a271ad63de4a5e55b028":
            raise ValueError("source_git_commit is incompatible")
        if self.device != "mps" or self.mps_device_guard != "mps_and_mps0_equivalent" or self.batch_size != 32 or self.query_chunk_size != 256:
            raise ValueError("Exp424 runtime contract is incompatible")
        if self.score_formula != "Exp417.score_horizon.v1" or self.percentile_method != "linear" or self.score_dtype != "float64":
            raise ValueError("Exp424 score contract is incompatible")
        if self.score_cache_contract_sha256 != EXP424_SCORE_CACHE_CONTRACT_SHA256 or self.synchronization_contract_sha256 != EXP424_MPS_SYNC_CONTRACT_SHA256:
            raise ValueError("Exp424 fixed contract digest is incompatible")
        object.__setattr__(self, "lane_hashes", MappingProxyType(dict(self.lane_hashes)))
        object.__setattr__(self, "runtime_versions", MappingProxyType(dict(self.runtime_versions)))
        object.__setattr__(self, "thresholds_payload", MappingProxyType(dict(self.thresholds_payload)))
        object.__setattr__(self, "model_seeds", tuple(self.model_seeds))
        object.__setattr__(self, "perturbation_seeds", tuple(self.perturbation_seeds))
        object.__setattr__(self, "alpha_values", tuple(self.alpha_values))

    @classmethod
    def for_fixture(cls, root: Path, *, root_provenance_sha256: str, progress_sha256: str, record_set_sha256: str) -> "Exp424CompatibilityContract":
        """Build a derived test-fixture descriptor without production digests."""
        return cls(Path(root).resolve(), root_provenance_sha256, progress_sha256, record_set_sha256, _fixture=True)

    @classmethod
    def production(cls) -> "Exp424CompatibilityContract":
        return cls(
            EXP424_SOURCE_ROOT, EXP424_ROOT_PROVENANCE_SHA256, EXP424_PROGRESS_SHA256,
            EXP424_RECORD_SET_SHA256,
            database_sha256="8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
            lane_hashes={
                "exp291_prediction": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
                "exp291_manifest": "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
                "exp292_prediction": "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
                "exp292_manifest": "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0",
            },
            config_sha256="6da423d0135a9607f04751e5f145153a5c45bbb19ff8d30dcd7863b77373286e",
            fold_contract_sha256="ed67b46a7a777ba3e6ef319344338d5b2f72aa4bbdec678af1db7ef6f29e0998",
            model_contract_sha256="116a2ccf896b89da6ea21b10f822d53c18d936ee78b2e0a1b0005262e7906c8c",
            model_seeds=(41701, 41702, 41703), perturbation_seeds=(41711, 41712, 41713),
            backend_contract_sha256="d6c62aaeb5f9b1c6e92eb515809e8ea3fb9b8c84b9165e3f9ced0e28690a0c1c",
            runtime_versions={"numba": "0.66.0", "numpy": "2.4.6", "python": "3.14.6", "torch": "2.13.0"},
        )

    def attestation_sha256(self) -> str:
        payload = {
            "schema": "exp425.exp424-compatibility.v1", "source_experiment": "Exp424-MPS-SCORE-CACHE",
            "source_git_commit": self.source_git_commit, "entrypoint_sha256": self.entrypoint_sha256,
            "entrypoint_git_blob": self.entrypoint_git_blob, "database_sha256": self.database_sha256,
            "lane_hashes": dict(sorted(self.lane_hashes.items())), "config_sha256": self.config_sha256,
            "fold_contract_sha256": self.fold_contract_sha256, "model_contract_sha256": self.model_contract_sha256,
            "model_seeds": list(self.model_seeds), "perturbation_seeds": list(self.perturbation_seeds),
            "backend_contract_sha256": self.backend_contract_sha256, "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "device": self.device, "mps_device_guard": self.mps_device_guard, "batch_size": self.batch_size,
            "query_chunk_size": self.query_chunk_size, "score_formula": self.score_formula,
            "score_cache_contract_sha256": self.score_cache_contract_sha256, "percentile_method": self.percentile_method,
            "score_dtype": self.score_dtype, "thresholds": dict(self.thresholds_payload), "alpha_values": list(self.alpha_values),
            "candidate_formula_sha256": self.candidate_formula_sha256, "routing_contract_sha256": self.routing_contract_sha256,
            "synchronization_contract_sha256": self.synchronization_contract_sha256,
        }
        return _digest(canonical_json_bytes(payload))


@dataclass(frozen=True)
class Exp424ImportReport:
    imported: tuple[str, ...]
    rejected_terminal: tuple[str, ...]
    rejected_incompatible: tuple[str, ...]
    source_root_sha256: str
    source_progress_sha256: str
    source_record_set_sha256: str
    compatibility_attestation_sha256: str
    batch_attestation_sha256: str | None = None


@dataclass(frozen=True)
class _SourceRecord:
    dataset: str
    filename: str
    digest: str
    identity: tuple[int, int]


def _source_record_digest(raw: bytes, *, fixture: bool) -> str:
    """Return Exp424's payload digest, retaining raw-byte hashing for fixtures."""
    payload = _canonical_object(raw, "Exp424 dataset record")
    declared = payload.get("payload_sha256")
    if fixture and declared is None:
        return _digest(raw)
    body = dict(payload)
    body.pop("payload_sha256", None)
    if not isinstance(declared, str) or _digest(canonical_json_bytes(body)) != declared:
        raise ValueError("Exp424 dataset record payload digest is invalid")
    return declared


def _payload_digest_from_canonical_bytes(raw: bytes) -> str:
    """Verify the schema-v3 embedded digest without materializing huge evidence."""
    if not raw.endswith(b"\n"):
        raise ValueError("Exp424 dataset record is not canonical")
    marker = b',"payload_sha256":"'
    start = raw.find(marker)
    if start < 0:
        raise ValueError("Exp424 dataset record payload digest is absent")
    value_start = start + len(marker)
    value_end = value_start + 64
    if raw[value_end:value_end + 2] != b'",':
        raise ValueError("Exp424 dataset record payload digest is malformed")
    declared = raw[value_start:value_end].decode("ascii")
    _sha256(declared, "Exp424 dataset record payload digest")
    body = raw[:start] + b"," + raw[value_end + 2:-1]
    if _digest(body) != declared:
        raise ValueError("Exp424 dataset record payload digest is invalid")
    return declared


class ValidatedExp424Source:
    """Opaque live descriptor authority; only the context manager can create it."""

    __slots__ = ("_root", "_contract", "_root_fd", "_datasets_fd", "_root_identity", "_datasets_identity", "_records", "_nonce", "_live")

    def __init__(self, sentinel: object, **values: object) -> None:
        if sentinel is not _CONSTRUCTION_SENTINEL:
            raise TypeError("ValidatedExp424Source construction is private")
        for key, value in values.items():
            setattr(self, key, value)

    def __copy__(self) -> object: raise TypeError("ValidatedExp424Source cannot be copied")
    def __deepcopy__(self, memo: object) -> object: raise TypeError("ValidatedExp424Source cannot be copied")
    def __reduce__(self) -> object: raise TypeError("ValidatedExp424Source cannot be pickled")

    @property
    def datasets(self) -> tuple[str, ...]:
        self._require_live()
        return tuple(record.dataset for record in self._records)

    @property
    def nonce(self) -> str:
        self._require_live()
        return self._nonce

    def _require_live(self) -> None:
        if not self._live:
            raise ValueError("Exp424 source authority is closed")

    def _iter_records(self) -> Iterator[tuple[_SourceRecord, dict[str, object]]]:
        self._require_live()
        for record in self._records:
            raw, identity = _read_at(self._datasets_fd, record.filename, "Exp424 dataset record")
            if identity != record.identity or _source_record_digest(raw, fixture=self._contract._fixture) != record.digest:
                raise ValueError("Exp424 source record changed after validation")
            yield record, _record_from_bytes(raw)

    def _revalidate(self) -> None:
        self._require_live()
        _validate_source(self._root, self._contract, self._root_fd, self._datasets_fd, self._root_identity, self._datasets_identity, self._records, full_records=True)

    def _mint(self, kind: str, **claims: object) -> object:
        self._require_live()
        # The final source check is deliberately adjacent to promotion.  This
        # closes the otherwise observable gap between the bulk revalidation and
        # consuming the one-use publish capability.
        if kind == "import_promote":
            self._revalidate()
        return capabilities._mint(self, kind=kind, process_nonce=self._nonce, **claims)


def _read_at(directory_fd: int, name: str, label: str) -> tuple[bytes, tuple[int, int]]:
    fd = os.open(name, _OPEN_FILE, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            part = os.read(fd, 1024 * 1024)
            if not part: break
            chunks.append(part)
        return b"".join(chunks), (info.st_dev, info.st_ino)
    finally:
        os.close(fd)


def _validate_source(root: Path, contract: Exp424CompatibilityContract, root_fd: int, datasets_fd: int, root_identity: tuple[int, int], datasets_identity: tuple[int, int], old_records: tuple[_SourceRecord, ...] | None = None, *, full_records: bool = False) -> tuple[_SourceRecord, ...]:
    current = os.fstat(root_fd)
    if (current.st_dev, current.st_ino) != root_identity or Path(root).resolve(strict=True) != contract.source_real_path:
        raise ValueError("Exp424 source root was replaced")
    current_ds = os.fstat(datasets_fd)
    if (current_ds.st_dev, current_ds.st_ino) != datasets_identity:
        raise ValueError("Exp424 datasets directory was replaced")
    if set(os.listdir(root_fd)) != {"checkpoint_provenance.json", "progress.json", "datasets"}:
        raise ValueError("Exp424 source root inventory is invalid")
    provenance_raw, _ = _read_at(root_fd, "checkpoint_provenance.json", "Exp424 provenance")
    progress_raw, _ = _read_at(root_fd, "progress.json", "Exp424 progress")
    if _digest(provenance_raw) != contract.root_provenance_sha256 or _digest(progress_raw) != contract.progress_sha256:
        raise ValueError("Exp424 source metadata digest does not match")
    provenance = _canonical_object(provenance_raw, "Exp424 provenance")
    progress = _canonical_object(progress_raw, "Exp424 progress")
    if provenance.get("schema_version") != 3 or progress.get("current_dataset") is not None:
        raise ValueError("Exp424 source schema is invalid")
    expected = provenance.get("expected_datasets")
    digests = progress.get("completed_record_digests")
    if not isinstance(expected, list) or tuple(expected) != tuple(sorted(expected)) or len(expected) != len(set(expected)) or not isinstance(digests, dict):
        raise ValueError("Exp424 source inventory is invalid")
    if set(digests) != set(expected) or progress.get("completed_datasets") != len(expected):
        raise ValueError("Exp424 source progress is inconsistent")
    if _digest(canonical_json_bytes(sorted(digests.items()))) != contract.record_set_sha256:
        raise ValueError("Exp424 source record-set digest does not match")
    names = set(os.listdir(datasets_fd))
    records: list[_SourceRecord] = []
    for dataset in expected:
        value = digests[dataset]
        _sha256(value, "Exp424 record digest")
        filename = record_filename(dataset)
        if contract._fixture and filename not in names and len(names) == 1:
            filename = next(iter(names))
        if filename not in names:
            raise ValueError("Exp424 dataset inventory is incomplete")
        info = os.stat(filename, dir_fd=datasets_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Exp424 dataset record is not a regular file")
        identity = (info.st_dev, info.st_ino)
        if full_records:
            raw, checked_identity = _read_at(datasets_fd, filename, "Exp424 dataset record")
            if checked_identity != identity or _source_record_digest(raw, fixture=contract._fixture) != value:
                raise ValueError("Exp424 dataset record digest does not match")
        records.append(_SourceRecord(dataset, filename, value, identity))
    if names != {record.filename for record in records}:
        raise ValueError("Exp424 dataset inventory contains unknown files")
    ordered = tuple(sorted(records, key=lambda item: item.dataset))
    if old_records is not None and ordered != old_records:
        raise ValueError("Exp424 source record inventory changed after validation")
    if not contract._fixture:
        _validate_production_provenance(provenance, contract)
    return ordered


def _validate_production_provenance(root: Mapping[str, object], contract: Exp424CompatibilityContract) -> None:
    provenance = root.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Exp424 provenance is invalid")
    required = {
        "git_commit": contract.source_git_commit, "database_sha256": contract.database_sha256,
        "frozen_config_sha256": contract.config_sha256, "fold_contract_sha256": contract.fold_contract_sha256,
        "model_contract_sha256": contract.model_contract_sha256, "backend_contract_sha256": contract.backend_contract_sha256,
    }
    if any(value is not None and provenance.get(key) != value for key, value in required.items()):
        raise ValueError("Exp424 provenance compatibility mismatch")
    if contract.model_seeds and tuple(provenance.get("model_seeds", ())) != contract.model_seeds:
        raise ValueError("Exp424 model seed compatibility mismatch")
    if contract.perturbation_seeds and tuple(provenance.get("perturbation_seeds", ())) != contract.perturbation_seeds:
        raise ValueError("Exp424 perturbation seed compatibility mismatch")
    if contract.runtime_versions and provenance.get("runtime_versions") != dict(contract.runtime_versions):
        raise ValueError("Exp424 runtime compatibility mismatch")
    lane_fields = {
        "exp291_prediction": "exp291_prediction_sha256", "exp291_manifest": "exp291_manifest_sha256",
        "exp292_prediction": "exp292_prediction_sha256", "exp292_manifest": "exp292_manifest_sha256",
    }
    if any(provenance.get(field) != contract.lane_hashes.get(name) for name, field in lane_fields.items() if name in contract.lane_hashes):
        raise ValueError("Exp424 lane compatibility mismatch")


@contextmanager
def open_validated_exp424_source(root: Path, contract: Exp424CompatibilityContract) -> Iterator[ValidatedExp424Source]:
    if not isinstance(contract, Exp424CompatibilityContract):
        raise ValueError("Exp424 compatibility contract is invalid")
    root = Path(root)
    if root.is_symlink() or root.resolve(strict=True) != contract.source_real_path:
        raise ValueError("Exp424 source path is not the contract path")
    root_fd = os.open(root, _OPEN_DIRECTORY)
    datasets_fd: int | None = None
    source: ValidatedExp424Source | None = None
    try:
        root_info = os.fstat(root_fd)
        fcntl.flock(root_fd, fcntl.LOCK_SH)
        datasets_fd = os.open("datasets", _OPEN_DIRECTORY, dir_fd=root_fd)
        records = _validate_source(root, contract, root_fd, datasets_fd, (root_info.st_dev, root_info.st_ino), tuple(os.fstat(datasets_fd)[i] for i in (stat.ST_DEV, stat.ST_INO)))
        source = ValidatedExp424Source(_CONSTRUCTION_SENTINEL, _root=root, _contract=contract, _root_fd=root_fd, _datasets_fd=datasets_fd, _root_identity=(root_info.st_dev, root_info.st_ino), _datasets_identity=tuple(os.fstat(datasets_fd)[i] for i in (stat.ST_DEV, stat.ST_INO)), _records=records, _nonce=uuid.uuid4().hex, _live=True)
        capabilities._register_authority(source, "exp424_import")
        yield source
    finally:
        if source is not None:
            source._live = False
            capabilities._close_authority(source)
        if datasets_fd is not None: os.close(datasets_fd)
        fcntl.flock(root_fd, fcntl.LOCK_UN)
        os.close(root_fd)


def _coordinates(result: DatasetResult) -> tuple[tuple[str, int], ...]:
    return tuple((candidate.dataset, candidate.instance_index) for candidate in result.candidates)


def adapt_imported_result(source: DatasetResult, target_context: FrozenRunContext) -> DatasetResult:
    """Rebind only context-bearing decision evidence, preserving effective output."""
    if not isinstance(source, DatasetResult) or not isinstance(target_context, FrozenRunContext):
        raise ValueError("Exp424 result or target context is invalid")
    if not isinstance(source.decision_evidence, DatasetDecisionEvidence):
        raise ValueError("terminal Exp424 result cannot be adapted")
    if source.context.database_sha256 != target_context.database_sha256 or source.dataset not in target_context.test_sizes:
        raise ValueError("Exp424 result is target-incompatible")
    if source.context.datasets != target_context.datasets or source.context.lanes != target_context.lanes or source.context.test_identity_manifest != target_context.test_identity_manifest:
        raise ValueError("Exp424 result context is target-incompatible")
    adapted = replace(source, decision_evidence=replace(source.decision_evidence, context=target_context), lanes=target_context.lanes, context=target_context)
    if _coordinates(adapted) != _coordinates(source):
        raise ValueError("Exp424 imported candidate coordinates changed")
    return adapted


def _batch_identity(source: ValidatedExp424Source, target: Exp425CheckpointStore) -> str:
    return _digest(canonical_json_bytes({"schema": "exp425.import-batch.v1", "run_identity": target.provenance.run_identity, "run_context_sha256": target.run_context_sha256, "source_real_path": str(source._contract.source_real_path), "source_root_sha256": source._contract.root_provenance_sha256, "source_progress_sha256": source._contract.progress_sha256, "source_record_set_sha256": source._contract.record_set_sha256, "compatibility_attestation_sha256": source._contract.attestation_sha256()}))


def _durable_attestation(source: ValidatedExp424Source, target: Exp425CheckpointStore, batch_identity: str, staged: Mapping[str, str]) -> str:
    return _digest(canonical_json_bytes({"schema": "exp425.exp424-import-attestation.v1", "target_run_identity": target.provenance.run_identity, "target_context_sha256": target.run_context_sha256, "batch_identity": batch_identity, "source_real_path": str(source._contract.source_real_path), "source_root_sha256": source._contract.root_provenance_sha256, "source_progress_sha256": source._contract.progress_sha256, "source_record_set_sha256": source._contract.record_set_sha256, "compatibility_attestation_sha256": source._contract.attestation_sha256(), "staged_record_digests": dict(sorted(staged.items()))}))


def import_exp424_nonterminal(source: ValidatedExp424Source, target: Exp425CheckpointStore, target_context: FrozenRunContext) -> Exp424ImportReport:
    if type(source) is not ValidatedExp424Source or not isinstance(target, Exp425CheckpointStore) or target.context != target_context:
        raise ValueError("Exp424 import inputs are invalid")
    source._require_live()
    imported: list[str] = []; terminal: list[str] = []; incompatible: list[str] = []
    pending: list[tuple[ImportedRecordEnvelope, str]] = []
    compatibility = source._contract.attestation_sha256()
    for descriptor, payload in source._iter_records():
        result = payload.get("qualification")
        if not isinstance(result, DatasetResult) or result.dataset != descriptor.dataset:
            raise ValueError("Exp424 dataset result is structurally invalid")
        if isinstance(result.decision_evidence, DatasetTerminalEvidence):
            terminal.append(descriptor.dataset); continue
        try:
            adapted = adapt_imported_result(result, target_context)
            if descriptor.dataset not in target.expected_datasets:
                raise ValueError("dataset is outside target universe")
        except ValueError:
            incompatible.append(descriptor.dataset); continue
        source_result = canonical_result_sha256(result)
        source_groups = {"schema": "exp425.source-proven-group-evidence.v1", "source_qualification_status": result.status, "oof_final_model_and_calibration_evidence": True, "source_record_sha256": descriptor.digest, "source_result_sha256": source_result, "compatibility_attestation_sha256": compatibility, "groups": [{"length": group.length, "status": group.status, "split_counts": "unavailable"} for group in result.groups]}
        evidence = SourceProvenGroupEvidence(source_groups, _digest(canonical_json_bytes(source_groups)))
        effective_status = "scored" if adapted.status == "qualified" else adapted.status
        record = ImportedRecordEnvelope(descriptor.dataset, {"status": effective_status, "candidate_coordinates": [list(item) for item in _coordinates(adapted)], "source_result_sha256": source_result}, evidence, source._contract.root_provenance_sha256, source._contract.progress_sha256, descriptor.digest, source_result, compatibility, float(payload["elapsed_seconds"]), "verified Exp424 nonterminal evidence")
        stage_digest = _digest(canonical_json_bytes(_import_record_body(target.run_context_sha256, record)))
        pending.append((record, stage_digest)); imported.append(descriptor.dataset)
    batch_id = _batch_identity(source, target)
    attestation = _durable_attestation(source, target, batch_id, dict((record.dataset, digest) for record, digest in pending))
    batch = target._begin_import_batch(source._mint("import_begin", run_identity=target.provenance.run_identity, batch_identity=batch_id, attestation_sha256=attestation), batch_id)
    for record, digest in pending:
        target._stage_imported(source._mint("import_stage", run_identity=target.provenance.run_identity, batch_identity=batch_id, dataset=record.dataset, source_sha256=record.source_record_sha256, attestation_sha256=attestation, record_sha256=digest), batch, record)
    source._revalidate()
    target._promote_import_batch(source._mint("import_promote", run_identity=target.provenance.run_identity, batch_identity=batch_id, attestation_sha256=attestation, final_attestation_sha256=_digest(canonical_json_bytes({"batch_identity": batch_id, "record_digests": dict(sorted((record.dataset, digest) for record, digest in pending)), "attestation_sha256": attestation}))), batch)
    return Exp424ImportReport(tuple(sorted(imported)), tuple(sorted(terminal)), tuple(sorted(incompatible)), source._contract.root_provenance_sha256, source._contract.progress_sha256, source._contract.record_set_sha256, compatibility, attestation)


def inspect_exp424_source(source: ValidatedExp424Source) -> Mapping[str, object]:
    """Stream structured records for the read-only Exp424 inventory smoke."""
    if type(source) is not ValidatedExp424Source:
        raise ValueError("Exp424 inspection source is invalid")
    source._require_live()
    importable = terminal = 0
    terminal_reasons: set[str] = set()
    for descriptor in source._records:
        raw, identity = _read_at(source._datasets_fd, descriptor.filename, "Exp424 dataset record")
        if identity != descriptor.identity or _payload_digest_from_canonical_bytes(raw) != descriptor.digest:
            raise ValueError("Exp424 source record changed after validation")
        # Schema-v3 terminal records are the only records with no raw TRAIN
        # identity.  Reconstruct those compact terminal records to read their
        # structured evidence/reason; nonterminals stay streaming-only here.
        if b'"raw_train_identity_sha256":null' in raw:
            payload = _record_from_bytes(raw)
            result = payload.get("qualification")
            if not isinstance(result, DatasetResult) or result.dataset != descriptor.dataset or not isinstance(result.decision_evidence, DatasetTerminalEvidence):
                raise ValueError("Exp424 terminal record is structurally invalid")
            terminal += 1
            terminal_reasons.add(result.decision_evidence.reason)
        else:
            importable += 1
    return MappingProxyType({"records": importable + terminal, "importable": importable, "terminal": terminal, "terminal_reasons": tuple(sorted(terminal_reasons)), "source_mutated": False})


if _digest(canonical_json_bytes(EXP424_SCORE_CACHE_CONTRACT)) != EXP424_SCORE_CACHE_CONTRACT_SHA256:
    raise RuntimeError("Exp424 score-cache contract literal digest is incorrect")
if _digest(canonical_json_bytes(EXP424_MPS_SYNC_CONTRACT)) != EXP424_MPS_SYNC_CONTRACT_SHA256:
    raise RuntimeError("Exp424 MPS synchronization contract literal digest is incorrect")
