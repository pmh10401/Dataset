"""Crash-safe, canonical checkpoints for Exp435.

The checkpoint is deliberately independent from the Exp426 record schemas.  A
single parent process owns this store; workers return immutable
``Exp435DatasetResult`` values and never open this directory.  Every record is
written once, and the sorted record digest set is updated atomically after the
record bytes have been fsynced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    BackendDecision,
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
)


_ROOT_FILES = frozenset(
    {
        "run_context_antigravity.json",
        "record_set_antigravity.json",
        "records_antigravity",
        "intents_antigravity",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
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


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _write_new(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _normalise(value: object) -> object:
    """Project a verified context to deterministic scalar JSON.

    ``FrozenRunContext`` contains mapping proxies and frozen dataclasses.  A
    recursive projection keeps the checkpoint independent of object identity
    while retaining the frozen dataset, TEST identity and code provenance.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, frozenset, set)):
        values = [_normalise(item) for item in value]
        if isinstance(value, (frozenset, set)):
            values.sort(key=lambda item: canonical_json_bytes(item))
        return values
    if is_dataclass(value):
        return {
            field.name: _normalise(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _normalise(to_payload())
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(attrs.items())
            if not str(key).startswith("_")
        }
    raise ValueError(f"context contains unsupported value: {type(value).__name__}")


def context_payload(context: object) -> dict[str, object]:
    """Return a canonical JSON-safe context projection."""

    projected = _normalise(context)
    if not isinstance(projected, dict):
        raise ValueError("verified run context must project to an object")
    return projected


def context_fingerprint(context: object) -> str:
    return _digest(canonical_json_bytes(context_payload(context)))


def _context_datasets(context: Mapping[str, object]) -> tuple[str, ...]:
    datasets = context.get("datasets")
    if not isinstance(datasets, list) or any(not isinstance(item, str) for item in datasets):
        raise ValueError("verified context datasets are invalid")
    result = tuple(datasets)
    if not result or result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError("verified context datasets are not sorted and unique")
    return result


def _context_test_sizes(context: Mapping[str, object]) -> Mapping[str, int]:
    sizes = context.get("test_sizes")
    if sizes is None:
        return MappingProxyType({})
    if not isinstance(sizes, dict):
        raise ValueError("verified context test_sizes are invalid")
    checked: dict[str, int] = {}
    for dataset, size in sizes.items():
        if not isinstance(dataset, str) or type(size) is not int or size <= 0:
            raise ValueError("verified context test_sizes are invalid")
        checked[dataset] = size
    return MappingProxyType(checked)


def _record_filename(dataset: str) -> str:
    return f"{sha256(dataset.encode('utf-8')).hexdigest()}.json"


def _record_set_digest(record_digests: Mapping[str, str]) -> str:
    canonical = canonical_json_bytes(dict(sorted(record_digests.items())))
    return _digest(canonical)


def _result_payload(result: Exp435DatasetResult) -> dict[str, object]:
    return {
        "schema": "exp435.dataset-record.v1",
        "result": result.to_payload(),
    }


def _result_from_payload(payload: Mapping[str, object]) -> Exp435DatasetResult:
    if set(payload) != {"schema", "result"} or payload.get("schema") != "exp435.dataset-record.v1":
        raise ValueError("dataset record schema is invalid")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("dataset record result is invalid")
    candidates_raw = result.get("candidates")
    evidence_raw = result.get("evidence")
    if not isinstance(candidates_raw, list) or not isinstance(evidence_raw, list):
        raise ValueError("dataset record arrays are invalid")
    try:
        candidates = tuple(Exp435Candidate(**dict(item)) for item in candidates_raw)
        evidence = tuple(
            InstanceScoreEvidence(**{
                key: item.get(key)
                for key in (
                    "dataset", "instance_index", "length", "prototype_score",
                    "prototype_threshold", "prototype_abnormal", "gradient_score",
                    "gradient_threshold", "gradient_abnormal", "acf_score",
                    "acf_threshold", "acf_abnormal", "consensus_candidate",
                    "status", "exclusion_reason", "evidence_digest",
                )
                if key in item
            })
            for item in evidence_raw
            if isinstance(item, Mapping)
        )
        kwargs: dict[str, object] = {
            "dataset": result["dataset"],
            "status": result["status"],
            "test_size": result["test_size"],
            "candidates": candidates,
            "evidence": evidence,
        }
        if result.get("abstention_reason") is not None:
            kwargs["abstention_reason"] = result["abstention_reason"]
        if result.get("error") is not None:
            kwargs["error"] = result["error"]
        return Exp435DatasetResult(**kwargs)  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dataset record payload is invalid") from error


@dataclass(frozen=True)
class Exp435CheckpointInspection:
    run_identity: str
    context_fingerprint: str
    completed: int
    total: int
    record_digests: Mapping[str, str]
    record_set_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_digests", MappingProxyType(dict(self.record_digests)))


@dataclass(frozen=True)
class Exp435RuntimeAccounting:
    completed: int
    total: int
    missing_events: tuple[str, ...]
    missing_progress: tuple[str, ...] = ()
    record_digests: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_digests", MappingProxyType(dict(self.record_digests)))


class Exp435CheckpointStore:
    """Single-writer canonical checkpoint with crash-window reconciliation."""

    def __init__(
        self,
        root: Path,
        context: Mapping[str, object],
        record_digests: Mapping[str, str],
        *,
        run_identity: str,
        context_hash: str,
    ) -> None:
        self.root = Path(root)
        self.context = MappingProxyType(dict(context))
        self.run_identity = run_identity
        self.context_fingerprint = context_hash
        self._record_digests = dict(record_digests)
        self._root_identity = self._identity_of(self.root)
        self._datasets = _context_datasets(self.context)
        self._test_sizes = _context_test_sizes(self.context)

    @staticmethod
    def _identity_of(path: Path) -> tuple[int, int]:
        info = path.stat()
        return info.st_dev, info.st_ino

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[int]:
        descriptor = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != self._root_identity:
                raise ValueError("checkpoint root identity changed")
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @classmethod
    def create(
        cls,
        root: Path,
        context: object,
        backend_decision: BackendDecision | None = None,
    ) -> "Exp435CheckpointStore":
        root = Path(root)
        if root.exists() or root.is_symlink():
            raise ValueError("checkpoint root already exists")
        projected = context_payload(context)
        datasets = _context_datasets(projected)
        context_hash = _digest(canonical_json_bytes(projected))
        run_identity = str(projected.get("run_identity") or f"exp435-{context_hash[:24]}")
        if not run_identity:
            raise ValueError("checkpoint run identity is invalid")
        if backend_decision is not None:
            if not isinstance(backend_decision, BackendDecision):
                raise ValueError("backend decision is invalid")
            projected = dict(projected)
            projected["backend_decision"] = backend_decision.to_dict()
            context_hash = _digest(canonical_json_bytes(projected))
        root.mkdir(mode=0o700)
        _fsync_directory(root.parent)
        (root / "records_antigravity").mkdir(mode=0o700)
        (root / "intents_antigravity").mkdir(mode=0o700)
        _fsync_directory(root)
        context_file = {
            "schema": "exp435.run-context.v1",
            "run_identity": run_identity,
            "context_fingerprint": context_hash,
            "context": projected,
        }
        _write_new(root / "run_context_antigravity.json", canonical_json_bytes(context_file))
        _write_new(
            root / "record_set_antigravity.json",
            canonical_json_bytes({
                "schema": "exp435.record-set.v1",
                "run_identity": run_identity,
                "context_fingerprint": context_hash,
                "completed": 0,
                "record_digests": {},
                "record_set_sha256": _record_set_digest({}),
            }),
        )
        return cls(root, projected, {}, run_identity=run_identity, context_hash=context_hash)

    @classmethod
    def open_existing(
        cls,
        root: Path,
        context: object,
        backend_decision: BackendDecision | None = None,
    ) -> "Exp435CheckpointStore":
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("checkpoint root is absent or invalid")
        projected = context_payload(context)
        if backend_decision is not None:
            if not isinstance(backend_decision, BackendDecision):
                raise ValueError("backend decision is invalid")
            projected = dict(projected)
            projected["backend_decision"] = backend_decision.to_dict()
        expected_hash = _digest(canonical_json_bytes(projected))
        info = root.stat()
        store = cls._load(root, (info.st_dev, info.st_ino), expected_context=projected, expected_hash=expected_hash)
        store._reconcile_intents()
        store._refresh()
        return store

    @classmethod
    def inspect_read_only(cls, root: Path, *, expected_run_identity: str | None = None) -> Exp435CheckpointInspection:
        root = Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("checkpoint root is absent or invalid")
        info = root.stat()
        store = cls._load(root, (info.st_dev, info.st_ino), expected_context=None, expected_hash=None)
        if expected_run_identity is not None and store.run_identity != expected_run_identity:
            raise ValueError("checkpoint run identity does not match")
        return store.inspect()

    @classmethod
    def _load(
        cls,
        root: Path,
        identity: tuple[int, int],
        *,
        expected_context: Mapping[str, object] | None,
        expected_hash: str | None,
    ) -> "Exp435CheckpointStore":
        descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if expected_context is not None else fcntl.LOCK_SH)
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != identity:
                raise ValueError("checkpoint root identity changed")
            if set(os.listdir(descriptor)) != set(_ROOT_FILES):
                raise ValueError("checkpoint inventory is invalid")
            for name in _ROOT_FILES:
                path = root / name
                if path.is_symlink():
                    raise ValueError("checkpoint inventory contains a symlink")
            context_file = _canonical_object(root / "run_context_antigravity.json", "run context")
            record_set = _canonical_object(root / "record_set_antigravity.json", "record set")
            if set(context_file) != {"schema", "run_identity", "context_fingerprint", "context"} or context_file.get("schema") != "exp435.run-context.v1":
                raise ValueError("run context schema is invalid")
            projected = context_file.get("context")
            if not isinstance(projected, dict):
                raise ValueError("run context payload is invalid")
            computed_hash = _digest(canonical_json_bytes(projected))
            if context_file.get("context_fingerprint") != computed_hash or not _is_digest(computed_hash):
                raise ValueError("context fingerprint is invalid")
            if expected_context is not None and projected != dict(expected_context):
                raise ValueError("context fingerprint does not match verified context")
            if expected_hash is not None and computed_hash != expected_hash:
                raise ValueError("context fingerprint does not match verified context")
            run_identity = context_file.get("run_identity")
            if not isinstance(run_identity, str) or not run_identity:
                raise ValueError("run identity is invalid")
            records = _validate_record_set(record_set, run_identity, computed_hash, projected)
            store = cls(root, projected, records, run_identity=run_identity, context_hash=computed_hash)
            store._validate_inventory(allow_pending=expected_context is not None)
            return store
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @property
    def datasets(self) -> tuple[str, ...]:
        return self._datasets

    @property
    def record_set_sha256(self) -> str:
        return _record_set_digest(self._record_digests)

    def completed_digests(self) -> Mapping[str, str]:
        with self._locked(exclusive=False):
            self._refresh()
            return MappingProxyType(dict(sorted(self._record_digests.items())))

    def inspect(self) -> Exp435CheckpointInspection:
        self._refresh()
        return Exp435CheckpointInspection(
            self.run_identity,
            self.context_fingerprint,
            len(self._record_digests),
            len(self._datasets),
            dict(sorted(self._record_digests.items())),
            _record_set_digest(self._record_digests),
        )

    def commit(self, result: Exp435DatasetResult) -> str:
        if not isinstance(result, Exp435DatasetResult):
            raise ValueError("checkpoint result is invalid")
        if result.dataset not in self._datasets:
            raise ValueError("record dataset is outside checkpoint universe")
        expected_size = self._test_sizes.get(result.dataset)
        if expected_size is not None and result.test_size != expected_size:
            raise ValueError("record TEST size does not match context")
        if result.candidate_count > FROZEN_CONFIG.global_candidate_budget:
            from .scoring_antigravity import CandidateBudgetExceeded

            raise CandidateBudgetExceeded(
                observed=result.candidate_count,
                budget=FROZEN_CONFIG.global_candidate_budget,
            )
        raw = canonical_json_bytes(_result_payload(result))
        digest = _digest(raw)
        with self._locked(exclusive=True):
            self._refresh()
            existing = self._record_digests.get(result.dataset)
            record_path = self.root / "records_antigravity" / _record_filename(result.dataset)
            if existing is not None:
                if existing != digest or record_path.read_bytes() != raw:
                    raise ValueError("dataset already has a different checkpoint record")
                return digest
            prospective = dict(self._record_digests)
            prospective[result.dataset] = digest
            candidates = 0
            for dataset in prospective:
                if dataset == result.dataset:
                    continue
                candidates += _result_from_payload(
                    _canonical_object(
                        self.root / "records_antigravity" / _record_filename(dataset),
                        "dataset record",
                    )
                ).candidate_count
            candidates += result.candidate_count
            if candidates > FROZEN_CONFIG.global_candidate_budget:
                from .scoring_antigravity import CandidateBudgetExceeded

                raise CandidateBudgetExceeded(
                    observed=candidates,
                    budget=FROZEN_CONFIG.global_candidate_budget,
                )
            intent_path = self.root / "intents_antigravity" / _record_filename(result.dataset)
            _write_new(
                intent_path,
                canonical_json_bytes({
                    "schema": "exp435.commit-intent.v1",
                    "dataset": result.dataset,
                    "record_sha256": digest,
                }),
            )
            _write_new(record_path, raw)
            _write_record_set(self.root, self.run_identity, self.context_fingerprint, prospective)
            self._record_digests = prospective
            intent_path.unlink()
            _fsync_directory(intent_path.parent)
        return digest

    def load_record(self, dataset: str) -> Exp435DatasetResult:
        if dataset not in self._datasets:
            raise ValueError("record dataset is outside checkpoint universe")
        with self._locked(exclusive=False):
            self._refresh()
            if dataset not in self._record_digests:
                raise ValueError("checkpoint record is absent")
            path = self.root / "records_antigravity" / _record_filename(dataset)
            raw = path.read_bytes()
            if _digest(raw) != self._record_digests[dataset]:
                raise ValueError("dataset record digest does not match")
            return _result_from_payload(_canonical_object(path, "dataset record"))

    def recover_runtime_accounting(self, output: Path | None = None) -> Exp435RuntimeAccounting:
        """Report durable records lacking an event or progress acknowledgement."""

        self._refresh()
        output_candidates = []
        if output is not None:
            output_candidates.append(Path(output))
        output_candidates.extend(
            [
                self.root.parent / "output_antigravity",
                self.root.parent / "output",
                self.root / "output_antigravity",
            ]
        )
        events: set[str] = set()
        progress: set[str] = set()
        event_path = next((path / "events_antigravity.jsonl" for path in output_candidates if (path / "events_antigravity.jsonl").is_file()), None)
        progress_path = next((path / "progress_antigravity.json" for path in output_candidates if (path / "progress_antigravity.json").is_file()), None)
        if event_path is not None:
            for line in event_path.read_bytes().splitlines():
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and item.get("kind") == "dataset_complete" and isinstance(item.get("dataset"), str):
                    events.add(item["dataset"])
        if progress_path is not None:
            item = _canonical_object(progress_path, "runtime progress")
            raw_completed = item.get("completed_datasets", item.get("completed"))
            if type(raw_completed) is int:
                # Progress does not carry a set of completed names; a current
                # dataset is the only recoverable point in a crash window.
                current = item.get("current_dataset")
                if isinstance(current, str):
                    progress.add(current)
        return Exp435RuntimeAccounting(
            completed=len(self._record_digests),
            total=len(self._datasets),
            missing_events=tuple(sorted(set(self._record_digests) - events)),
            missing_progress=tuple(sorted(set(self._record_digests) - progress)) if progress_path is not None else tuple(sorted(self._record_digests)),
            record_digests=dict(sorted(self._record_digests.items())),
        )

    def _refresh(self) -> None:
        record_set = _canonical_object(self.root / "record_set_antigravity.json", "record set")
        records = _validate_record_set(record_set, self.run_identity, self.context_fingerprint, self.context)
        self._record_digests = records
        self._validate_inventory()

    def _reconcile_intents(self) -> None:
        with self._locked(exclusive=True):
            intents = sorted((self.root / "intents_antigravity").iterdir(), key=lambda path: path.name)
            if not intents:
                return
            record_set = _canonical_object(self.root / "record_set_antigravity.json", "record set")
            self._record_digests = _validate_record_set(
                record_set, self.run_identity, self.context_fingerprint, self.context
            )
            self._validate_inventory(allow_pending=True)
            updated = dict(self._record_digests)
            for path in intents:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("checkpoint intent inventory is invalid")
                payload = _canonical_object(path, "checkpoint intent")
                if set(payload) != {"schema", "dataset", "record_sha256"} or payload.get("schema") != "exp435.commit-intent.v1":
                    raise ValueError("checkpoint intent inventory is invalid")
                dataset = payload.get("dataset")
                expected = payload.get("record_sha256")
                if not isinstance(dataset, str) or dataset not in self._datasets or not _is_digest(expected) or path.name != _record_filename(dataset):
                    raise ValueError("checkpoint intent inventory is invalid")
                record_path = self.root / "records_antigravity" / _record_filename(dataset)
                if record_path.exists():
                    raw = record_path.read_bytes()
                    if _digest(raw) != expected:
                        raise ValueError("checkpoint intent and record disagree")
                    updated[dataset] = expected
                path.unlink()
            if updated != self._record_digests:
                _write_record_set(self.root, self.run_identity, self.context_fingerprint, updated)
                self._record_digests = updated
            _fsync_directory(self.root / "intents_antigravity")

    def _validate_inventory(self, *, allow_pending: bool = False) -> None:
        records_root = self.root / "records_antigravity"
        intents_root = self.root / "intents_antigravity"
        if records_root.is_symlink() or intents_root.is_symlink() or not records_root.is_dir() or not intents_root.is_dir():
            raise ValueError("checkpoint inventory is invalid")
        pending: dict[str, str] = {}
        for path in intents_root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("checkpoint intent inventory is invalid")
            payload = _canonical_object(path, "checkpoint intent")
            if set(payload) != {"schema", "dataset", "record_sha256"} or payload.get("schema") != "exp435.commit-intent.v1":
                raise ValueError("checkpoint intent inventory is invalid")
            dataset = payload.get("dataset")
            digest = payload.get("record_sha256")
            if not isinstance(dataset, str) or dataset not in self._datasets or path.name != _record_filename(dataset) or not _is_digest(digest):
                raise ValueError("checkpoint intent inventory is invalid")
            pending[dataset] = digest
        expected_datasets = set(self._record_digests)
        if allow_pending:
            expected_datasets.update(dataset for dataset in pending if (records_root / _record_filename(dataset)).exists())
        expected = {_record_filename(dataset) for dataset in expected_datasets}
        names = {path.name for path in records_root.iterdir()}
        if names != expected:
            raise ValueError("checkpoint record inventory is invalid")
        if any(path.is_symlink() or not path.is_file() for path in records_root.iterdir()):
            raise ValueError("checkpoint record inventory is invalid")
        total_candidates = 0
        for dataset in expected_datasets:
            digest = self._record_digests.get(dataset, pending.get(dataset))
            if digest is None:
                raise ValueError("checkpoint record digest is missing")
            path = records_root / _record_filename(dataset)
            raw = path.read_bytes()
            if _digest(raw) != digest:
                raise ValueError("checkpoint record digest does not match")
            result = _result_from_payload(_canonical_object(path, "dataset record"))
            if result.dataset != dataset:
                raise ValueError("checkpoint record dataset does not match filename")
            total_candidates += result.candidate_count
        if total_candidates > FROZEN_CONFIG.global_candidate_budget:
            raise ValueError("Exp435 candidate budget exceeded")


def _validate_record_set(
    payload: Mapping[str, object],
    run_identity: str,
    context_hash: str,
    context: Mapping[str, object],
) -> dict[str, str]:
    if set(payload) != {"schema", "run_identity", "context_fingerprint", "completed", "record_digests", "record_set_sha256"} or payload.get("schema") != "exp435.record-set.v1":
        raise ValueError("record set schema is invalid")
    if payload.get("run_identity") != run_identity or payload.get("context_fingerprint") != context_hash:
        raise ValueError("record set provenance does not match context")
    digests = payload.get("record_digests")
    if not isinstance(digests, dict) or type(payload.get("completed")) is not int or payload.get("completed") != len(digests) or payload.get("record_set_sha256") != _record_set_digest(digests):
        raise ValueError("record set accounting is invalid")
    datasets = set(_context_datasets(context))
    if not set(digests).issubset(datasets) or any(not _is_digest(value) for value in digests.values()):
        raise ValueError("record set dataset or digest is invalid")
    return dict(digests)


def _write_record_set(root: Path, run_identity: str, context_hash: str, digests: Mapping[str, str]) -> None:
    _atomic_replace(
        root / "record_set_antigravity.json",
        canonical_json_bytes({
            "schema": "exp435.record-set.v1",
            "run_identity": run_identity,
            "context_fingerprint": context_hash,
            "completed": len(digests),
            "record_digests": dict(sorted(digests.items())),
            "record_set_sha256": _record_set_digest(digests),
        }),
    )


__all__ = [
    "Exp435CheckpointInspection",
    "Exp435CheckpointStore",
    "Exp435RuntimeAccounting",
    "context_fingerprint",
    "context_payload",
]
