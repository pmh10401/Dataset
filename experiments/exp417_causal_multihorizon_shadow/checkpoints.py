"""Fail-closed, provenance-bound local checkpoints for Exp417.

The store is deliberately local-only.  It persists no source data; a record is
accepted only when its immutable decision evidence can be reconstructed from
canonical JSON and all run provenance still agrees with the root manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Any

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config import FROZEN_CONFIG
from .contracts import (
    BackendAudit, DatasetDecisionEvidence, DatasetResult, DatasetTerminalEvidence, FrozenLaneSnapshot,
    FrozenRunContext, FrozenTrainContract, GroupQualification, HorizonScore,
    SeedDecision, ShadowCandidate, TestInstanceDecision,
)


_SCHEMA_VERSION = 2
_ROOT_FILE = "checkpoint_provenance.json"
_PROGRESS_FILE = "progress.json"


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
            self._validate_root()
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
        self._validate_root()
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
        lengths = tuple(group.length for group in result.groups)
        if not terminal and (tuple(identities) != lengths or tuple(model_hashes) != lengths or tuple(calibrations) != lengths):
            raise ValueError("Exp417 checkpoint length-group accounting is invalid")
        contract = result.decision_evidence.frozen_train_contract
        if terminal:
            pass
        elif not isinstance(backend_audit, BackendAudit) or contract is None or backend_audit != contract.backend:
            raise ValueError("Exp417 checkpoint backend audit does not match result")
        elapsed = _finite_nonnegative(elapsed_seconds, "elapsed time")
        path = self._dataset_path(result.dataset)
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
        # The exclusive link publication below is the concurrency authority;
        # scanning every completed record here made a valid 1,117-dataset
        # resume quadratic without adding safety.
        if _path_exists_lstat(path):
            raise ValueError(f"Exp417 checkpoint already exists for dataset: {result.dataset}")
        old_progress = _read_regular(self.root / _PROGRESS_FILE, "Exp417 checkpoint progress")
        self._begin_transaction(result.dataset, old_progress)
        published = False
        try:
            _atomic_write_exclusive(path, canonical_json_bytes(body) + b"\n")
            published = True
            _fsync_directory(self.datasets_dir)
            digests = _progress_digests(self._read_progress())
            digests[result.dataset] = body["payload_sha256"]
            self._write_progress(completed=len(digests), current_dataset=None, record_digests=digests)
            self._finish_transaction()
        except BaseException:
            self._rollback_transaction(path if published else None, old_progress)
            raise

    def load_completed(self) -> Mapping[str, DatasetResult]:
        records = self._load_records()
        self._assert_progress_matches_records(records)
        return MappingProxyType({name: records[name]["qualification"] for name in self.expected_datasets if name in records})

    def progress(self) -> Mapping[str, object]:
        records = self._load_records()
        payload = self._read_progress()
        self._assert_progress_matches_records(records)
        return MappingProxyType({"completed_datasets": payload["completed_datasets"],
                                 "current_dataset": payload["current_dataset"],
                                 "expected_datasets": payload["expected_datasets"]})

    def require_complete(self) -> tuple[DatasetResult, ...]:
        completed = self.load_completed()
        if tuple(completed) != self.expected_datasets:
            raise ValueError("Exp417 checkpoint dataset coverage is incomplete")
        return tuple(completed[name] for name in self.expected_datasets)

    def _create_root(self) -> None:
        if not self.root.parent.is_dir() or stat.S_ISLNK(os.lstat(self.root.parent).st_mode):
            raise ValueError("Exp417 checkpoint parent is invalid")
        try:
            os.mkdir(self.root, 0o700)
            os.chmod(self.root, 0o700)
            os.mkdir(self.datasets_dir, 0o700)
            os.chmod(self.datasets_dir, 0o700)
            _atomic_write(self.root / _ROOT_FILE, canonical_json_bytes(self._root_payload) + b"\n", replace=False)
            self._write_progress(completed=0, current_dataset=None, record_digests={})
            _fsync_directory(self.root)
        except BaseException:
            # A partially initialized root must never be mistaken for a checkpoint.
            raise

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

    def _dataset_path(self, dataset: str) -> Path:
        return self.datasets_dir / f"{sha256(dataset.encode('utf-8')).hexdigest()}.json"

    def _write_progress(self, *, completed: int, current_dataset: str | None,
                        record_digests: Mapping[str, object]) -> None:
        if type(completed) is not int or not 0 <= completed <= len(self.expected_datasets):
            raise ValueError("Exp417 checkpoint progress is invalid")
        if set(record_digests) - set(self.expected_datasets) or len(record_digests) != completed:
            raise ValueError("Exp417 checkpoint progress is invalid")
        digests = {name: _digest(value, "progress record digest") for name, value in sorted(record_digests.items())}
        payload = {"completed_datasets": completed, "current_dataset": current_dataset,
                   "expected_datasets": len(self.expected_datasets), "completed_record_digests": digests}
        _atomic_write(self.root / _PROGRESS_FILE, canonical_json_bytes(payload) + b"\n", replace=True)
        _fsync_directory(self.root)

    def _read_progress(self) -> dict[str, object]:
        raw = _read_regular(self.root / _PROGRESS_FILE, "Exp417 checkpoint progress")
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

    def _begin_transaction(self, dataset: str, old_progress: bytes) -> None:
        lock = self.root / ".checkpoint.lock"
        marker = self.root / ".checkpoint-transaction.json"
        acquired_lock = False
        wrote_marker = False
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            os.close(descriptor)
            acquired_lock = True
            _fsync_directory(self.root)
            marker_payload = {"dataset": dataset, "previous_progress_sha256": sha256(old_progress).hexdigest()}
            _atomic_write(marker, canonical_json_bytes(marker_payload) + b"\n", replace=False)
            wrote_marker = True
            _fsync_directory(self.root)
        except BaseException:
            # Never remove a competing writer's lock or marker.
            try:
                if wrote_marker and _path_exists_lstat(marker):
                    _unlink_regular(marker)
                if acquired_lock and _path_exists_lstat(lock):
                    _unlink_regular(lock)
                if acquired_lock:
                    _fsync_directory(self.root)
            except OSError:
                pass
            raise

    def _finish_transaction(self) -> None:
        self._clear_transaction()

    def _rollback_transaction(self, path: Path | None, old_progress: bytes) -> None:
        try:
            if path is not None and _path_exists_lstat(path):
                _unlink_regular(path)
                _fsync_directory(self.datasets_dir)
            _atomic_write(self.root / _PROGRESS_FILE, old_progress, replace=True)
            _fsync_directory(self.root)
        finally:
            self._clear_transaction()

    def _clear_transaction(self) -> None:
        for path in (self.root / ".checkpoint-transaction.json", self.root / ".checkpoint.lock"):
            if _path_exists_lstat(path):
                _unlink_regular(path)
        _fsync_directory(self.root)


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
    lengths = tuple(group.length for group in result.groups)
    if not terminal and (tuple(identities) != lengths or tuple(model_hashes) != lengths or tuple(calibrations) != lengths):
        raise ValueError("Exp417 checkpoint length-group accounting is invalid")
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
        _finite(data["oof_candidate_rate"], "oof rate"), _finite(data["wilson_upper"], "wilson upper"), _finite(data["pooled_synthetic_sensitivity"], "sensitivity"),
        {_string(key, "perturbation"): _finite(item, "sensitivity") for key, item in _object(data["per_type_synthetic_sensitivity"], "sensitivity map").items()},
        _bool(data["seed_agreement"], "seed agreement"), _bool(data["finite"], "finite"), _bool(data["backend_fallback"], "backend fallback"))


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


def _dataset_universe(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence): raise ValueError("Exp417 checkpoint dataset universe is invalid")
    names = tuple(values)
    if not names or any(not isinstance(item, str) or not item for item in names) or names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("Exp417 checkpoint dataset universe is invalid")
    return names


def _length_digest_mapping(value: object, name: str) -> dict[int, str]:
    data = _integer_key_mapping(value, name); result = {_mapping_integer(key, name, 1): _digest(item, name) for key, item in data.items()}
    if tuple(result) != tuple(sorted(result)): raise ValueError(f"Exp417 checkpoint {name} are not sorted")
    return result


def _model_hash_mapping(value: object) -> dict[int, dict[int, str]]:
    data = _integer_key_mapping(value, "canonical model hashes"); result: dict[int, dict[int, str]] = {}
    for raw_length, raw_hashes in data.items():
        length = _mapping_integer(raw_length, "model length", 1)
        hashes = {_mapping_integer(seed, "model seed", 1): _digest(digest, "model hash") for seed, digest in _integer_key_mapping(raw_hashes, "model hashes").items()}
        if tuple(hashes) != FROZEN_CONFIG.model_seeds: raise ValueError("Exp417 checkpoint model seed accounting is invalid")
        result[length] = hashes
    if tuple(result) != tuple(sorted(result)): raise ValueError("Exp417 checkpoint model hashes are not sorted")
    return result


def _atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    _require_directory(path.parent, "Exp417 checkpoint parent")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    temporary_path = Path(temporary)
    committed = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        if not replace and _path_exists_lstat(path): raise FileExistsError(path)
        os.replace(temporary_path, path); committed = True
    except BaseException:
        if committed and _path_exists_lstat(path):
            try: _unlink_regular(path)
            except OSError: pass
        raise
    finally:
        if _path_exists_lstat(temporary_path):
            try: _unlink_regular(temporary_path)
            except OSError: pass


def _atomic_write_exclusive(path: Path, data: bytes) -> None:
    """Publish a new record with link(2), which can never overwrite a rival."""
    _require_directory(path.parent, "Exp417 checkpoint parent")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    temporary_path = Path(temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary_path, path)
        linked = True
    except BaseException:
        if linked and _path_exists_lstat(path):
            try: _unlink_regular(path)
            except OSError: pass
        raise
    finally:
        if _path_exists_lstat(temporary_path):
            try: _unlink_regular(temporary_path)
            except OSError: pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _scandir(path: Path) -> list[os.DirEntry[str]]:
    _require_directory(path, "Exp417 checkpoint directory")
    try:
        with os.scandir(path) as scan: return list(scan)
    except OSError as error: raise ValueError("Exp417 checkpoint inventory is unreadable") from error


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


def _unlink_regular(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode): raise OSError("not a regular file")
    os.unlink(path)


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


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return float(value)


def _finite_nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0: raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return number


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value): raise ValueError(f"Exp417 checkpoint {name} is invalid")
    return value


__all__ = ["Exp417CheckpointProvenance", "Exp417CheckpointStore"]
