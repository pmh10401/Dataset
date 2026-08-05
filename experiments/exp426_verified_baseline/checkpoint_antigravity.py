"""Single-writer, crash-safe checkpoint storage for Exp426."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .contracts_antigravity import Exp426DatasetRecord, Exp426Provenance


_ROOT_FILES = {"checkpoint_provenance.json", "run_context.json", "progress.json", "datasets", "intents"}


def _record_filename(dataset: str) -> str:
    return sha256(dataset.encode("utf-8")).hexdigest() + ".json"


def _canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_new(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, raw: bytes) -> None:
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    _write_new(temporary, raw)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _context_payload(value: object, provenance: Exp426Provenance) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("frozen context is invalid")
    payload = dict(value)
    if set(payload) != {"schema", "database_sha256", "datasets", "test_sizes", "lane_coordinates", "code_commit"}:
        raise ValueError("frozen context fields are invalid")
    if payload.get("schema") != "exp426.frozen-context.v1":
        raise ValueError("frozen context schema is invalid")
    datasets = payload.get("datasets")
    sizes = payload.get("test_sizes")
    lanes = payload.get("lane_coordinates")
    if datasets != list(provenance.expected_datasets) or payload.get("database_sha256") != provenance.database_sha256 or payload.get("code_commit") != provenance.git_commit:
        raise ValueError("frozen context does not match provenance")
    if not isinstance(sizes, dict) or set(sizes) != set(provenance.expected_datasets) or any(type(item) is not int or item <= 0 for item in sizes.values()):
        raise ValueError("frozen context TEST sizes are invalid")
    if not isinstance(lanes, list):
        raise ValueError("frozen context lane coordinates are invalid")
    checked_lanes: list[list[object]] = []
    for item in lanes:
        if not isinstance(item, list) or len(item) != 2 or item[0] not in sizes or type(item[1]) is not int or item[1] < 0 or item[1] >= sizes[item[0]]:
            raise ValueError("frozen context lane coordinate is invalid")
        checked_lanes.append([item[0], item[1]])
    if checked_lanes != sorted(checked_lanes) or len({tuple(item) for item in checked_lanes}) != len(checked_lanes):
        raise ValueError("frozen context lanes must be sorted and unique")
    return {
        "schema": "exp426.frozen-context.v1",
        "database_sha256": payload["database_sha256"],
        "datasets": list(datasets),
        "test_sizes": dict(sorted(sizes.items())),
        "lane_coordinates": checked_lanes,
        "code_commit": payload["code_commit"],
    }


def _progress_payload(provenance: Exp426Provenance, digests: Mapping[str, str]) -> dict[str, object]:
    return {
        "schema": "exp426.checkpoint-progress.v1",
        "run_identity": provenance.run_identity,
        "completed": len(digests),
        "total": len(provenance.expected_datasets),
        "record_digests": dict(sorted(digests.items())),
        "current_dataset": None,
    }


@dataclass(frozen=True)
class Exp426CheckpointInspection:
    run_identity: str
    completed: int
    total: int
    record_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_digests", MappingProxyType(dict(self.record_digests)))


class Exp426CheckpointStore:
    def __init__(
        self, root: Path, provenance: Exp426Provenance, context: Mapping[str, object],
        record_digests: Mapping[str, str], root_identity: tuple[int, int],
    ) -> None:
        self.root = Path(root)
        self.provenance = provenance
        self.context = MappingProxyType(dict(context))
        self._record_digests = dict(record_digests)
        self._root_identity = root_identity
        self._test_sizes = dict(context["test_sizes"])  # type: ignore[arg-type]
        self._lanes = {tuple(item) for item in context["lane_coordinates"]}  # type: ignore[union-attr]

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[int]:
        descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
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
    def create(cls, root: Path, provenance: Exp426Provenance, context: object) -> "Exp426CheckpointStore":
        if not isinstance(provenance, Exp426Provenance):
            raise ValueError("checkpoint provenance is invalid")
        root = Path(root)
        if root.exists() or root.is_symlink():
            raise ValueError("checkpoint root already exists")
        payload = _context_payload(context, provenance)
        context_raw = canonical_json_bytes(payload)
        bound = provenance.with_context_sha256(sha256(context_raw).hexdigest())
        root.mkdir(mode=0o700)
        _fsync_directory(root.parent)
        (root / "datasets").mkdir(mode=0o700)
        (root / "intents").mkdir(mode=0o700)
        _fsync_directory(root)
        _write_new(root / "checkpoint_provenance.json", canonical_json_bytes(bound.to_payload()))
        _write_new(root / "run_context.json", context_raw)
        _write_new(root / "progress.json", canonical_json_bytes(_progress_payload(bound, {})))
        info = root.stat()
        return cls(root, bound, payload, {}, (info.st_dev, info.st_ino))

    @classmethod
    def open(cls, root: Path, provenance: Exp426Provenance, context: object) -> "Exp426CheckpointStore":
        root = Path(root)
        if not root.exists() or root.is_symlink():
            raise ValueError("checkpoint root is absent or invalid")
        payload = _context_payload(context, provenance)
        context_raw = canonical_json_bytes(payload)
        bound = provenance.with_context_sha256(sha256(context_raw).hexdigest())
        info = root.stat()
        return cls._load(
            root, (info.st_dev, info.st_ino),
            expected=bound, context=payload, exclusive=True,
        )

    @classmethod
    def _load(
        cls, root: Path, root_identity: tuple[int, int], *, expected: Exp426Provenance | None,
        context: Mapping[str, object] | None, exclusive: bool,
    ) -> "Exp426CheckpointStore":
        descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != root_identity or set(os.listdir(descriptor)) != _ROOT_FILES:
                raise ValueError("checkpoint root inventory is invalid")
            for name in _ROOT_FILES:
                if (root / name).is_symlink():
                    raise ValueError("checkpoint inventory contains a symlink")
            provenance_payload = _canonical_object((root / "checkpoint_provenance.json").read_bytes(), "checkpoint provenance")
            context_payload = _canonical_object((root / "run_context.json").read_bytes(), "run context")
            progress = _canonical_object((root / "progress.json").read_bytes(), "checkpoint progress")
            provenance = _provenance_from_payload(provenance_payload)
            if expected is not None and provenance != expected:
                raise ValueError("checkpoint provenance or context does not match")
            if context is not None and context_payload != dict(context):
                raise ValueError("checkpoint context does not match")
            digests = _validate_progress(progress, provenance)
            _validate_dataset_inventory(
                root, provenance, digests, allow_pending_records=exclusive,
            )
            store = cls(root, provenance, context_payload, digests, root_identity)
            if exclusive:
                store._reconcile_intents_locked()
                _validate_dataset_inventory(root, provenance, store._record_digests)
            return store
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @classmethod
    def inspect_read_only(cls, root: Path, *, expected_run_identity: str | None = None) -> Exp426CheckpointInspection:
        root = Path(root)
        if not root.exists() or root.is_symlink():
            raise ValueError("checkpoint root is absent or invalid")
        info = root.stat()
        store = cls._load(root, (info.st_dev, info.st_ino), expected=None, context=None, exclusive=False)
        if expected_run_identity is not None and store.provenance.run_identity != expected_run_identity:
            raise ValueError("checkpoint run identity does not match")
        return Exp426CheckpointInspection(
            store.provenance.run_identity, len(store._record_digests),
            len(store.provenance.expected_datasets), store._record_digests,
        )

    def completed_digests(self) -> Mapping[str, str]:
        with self._locked(exclusive=False):
            self._refresh_read_only_locked()
            return MappingProxyType(dict(self._record_digests))

    def commit(self, record: Exp426DatasetRecord) -> str:
        if not isinstance(record, Exp426DatasetRecord):
            raise ValueError("checkpoint record is invalid")
        if record.dataset not in self.provenance.expected_datasets:
            raise ValueError("record dataset is outside checkpoint universe")
        record.validate_against(self._test_sizes, lambda dataset, index: (dataset, index) in self._lanes)
        raw = canonical_json_bytes(record.scientific_payload())
        digest = sha256(raw).hexdigest()
        if digest != record.sha256:
            raise ValueError("record digest is inconsistent")
        with self._locked(exclusive=True):
            self._refresh_writable_locked()
            existing = self._record_digests.get(record.dataset)
            if existing is not None:
                existing_raw = (
                    self.root / "datasets" / _record_filename(record.dataset)
                ).read_bytes()
                if existing != digest or existing_raw != raw:
                    raise ValueError("dataset already has a different checkpoint record")
                return digest
            intent = self.root / "intents" / _record_filename(record.dataset)
            dataset_path = self.root / "datasets" / _record_filename(record.dataset)
            _write_new(intent, canonical_json_bytes({"schema": "exp426.commit-intent.v1", "dataset": record.dataset, "record_sha256": digest}))
            _write_new(dataset_path, raw)
            updated = dict(self._record_digests)
            updated[record.dataset] = digest
            _atomic_replace(
                self.root / "progress.json",
                canonical_json_bytes(_progress_payload(self.provenance, updated)),
            )
            self._record_digests = updated
            intent.unlink()
            _fsync_directory(intent.parent)
        return digest

    def load_record(self, dataset: str) -> Exp426DatasetRecord:
        with self._locked(exclusive=False):
            self._refresh_read_only_locked()
            if dataset not in self._record_digests:
                raise ValueError("checkpoint record is absent")
            raw = (self.root / "datasets" / _record_filename(dataset)).read_bytes()
            if sha256(raw).hexdigest() != self._record_digests[dataset]:
                raise ValueError("checkpoint record digest does not match")
            return Exp426DatasetRecord.from_payload(
                _canonical_object(raw, "dataset record")
            )

    def _refresh_read_only_locked(self) -> None:
        progress = _canonical_object(
            (self.root / "progress.json").read_bytes(), "checkpoint progress"
        )
        digests = _validate_progress(progress, self.provenance)
        _validate_dataset_inventory(self.root, self.provenance, digests)
        self._record_digests = digests

    def _refresh_writable_locked(self) -> None:
        progress = _canonical_object(
            (self.root / "progress.json").read_bytes(), "checkpoint progress"
        )
        self._record_digests = _validate_progress(progress, self.provenance)
        _validate_dataset_inventory(
            self.root, self.provenance, self._record_digests,
            allow_pending_records=True,
        )
        self._reconcile_intents_locked()
        _validate_dataset_inventory(
            self.root, self.provenance, self._record_digests,
        )

    def _reconcile_intents_locked(self) -> None:
        intents = _validated_intents(self.root, self.provenance)
        updated = dict(self._record_digests)
        for dataset, (path, expected_digest) in intents.items():
            dataset_path = self.root / "datasets" / _record_filename(dataset)
            if dataset_path.exists():
                if dataset_path.is_symlink() or not dataset_path.is_file():
                    raise ValueError("checkpoint pending dataset record is invalid")
                raw = dataset_path.read_bytes()
                digest = sha256(raw).hexdigest()
                if digest != expected_digest:
                    raise ValueError("checkpoint intent and record disagree")
                record = Exp426DatasetRecord.from_payload(
                    _canonical_object(raw, "dataset record")
                )
                if record.dataset != dataset:
                    raise ValueError("checkpoint intent record dataset disagrees")
                existing = updated.get(dataset)
                if existing is not None and existing != digest:
                    raise ValueError("checkpoint intent and progress disagree")
                updated[dataset] = digest
        if updated != self._record_digests:
            _atomic_replace(
                self.root / "progress.json",
                canonical_json_bytes(_progress_payload(self.provenance, updated)),
            )
            self._record_digests = updated
        for path, _digest_value in intents.values():
            path.unlink()
        if intents:
            _fsync_directory(self.root / "intents")


def _provenance_from_payload(value: Mapping[str, object]) -> Exp426Provenance:
    if set(value) != {
        "schema", "run_identity", "git_commit", "database_sha256", "source_root_sha256",
        "source_progress_sha256", "source_record_set_sha256", "lane_artifact_sha256",
        "fold_contract_sha256", "expected_datasets", "workers", "context_sha256",
    } or value.get("schema") != "exp426.checkpoint-provenance.v1":
        raise ValueError("checkpoint provenance payload is invalid")
    lanes = value["lane_artifact_sha256"]
    datasets = value["expected_datasets"]
    if not isinstance(lanes, dict) or not isinstance(datasets, list):
        raise ValueError("checkpoint provenance payload is invalid")
    return Exp426Provenance(
        value["run_identity"], value["git_commit"], value["database_sha256"],
        value["source_root_sha256"], value["source_progress_sha256"],
        value["source_record_set_sha256"], lanes, value["fold_contract_sha256"],
        tuple(datasets), value["workers"], value["context_sha256"],
    )  # type: ignore[arg-type]


def _validate_progress(value: Mapping[str, object], provenance: Exp426Provenance) -> dict[str, str]:
    if set(value) != {"schema", "run_identity", "completed", "total", "record_digests", "current_dataset"} or value.get("schema") != "exp426.checkpoint-progress.v1" or value.get("run_identity") != provenance.run_identity or value.get("total") != len(provenance.expected_datasets) or value.get("current_dataset") is not None:
        raise ValueError("checkpoint progress is invalid")
    digests = value.get("record_digests")
    if not isinstance(digests, dict) or value.get("completed") != len(digests) or not set(digests).issubset(provenance.expected_datasets):
        raise ValueError("checkpoint progress accounting is invalid")
    for dataset, digest in digests.items():
        if not isinstance(dataset, str) or not _is_lower_sha256(digest):
            raise ValueError("checkpoint progress digest must be a lowercase SHA256")
    return dict(digests)


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_intents(
    root: Path, provenance: Exp426Provenance,
) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for path in sorted((root / "intents").iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint intent inventory is invalid")
        payload = _canonical_object(path.read_bytes(), "checkpoint intent")
        if (
            set(payload) != {"schema", "dataset", "record_sha256"}
            or payload.get("schema") != "exp426.commit-intent.v1"
        ):
            raise ValueError("checkpoint intent inventory is invalid")
        dataset = payload.get("dataset")
        digest = payload.get("record_sha256")
        if (
            not isinstance(dataset, str)
            or dataset not in provenance.expected_datasets
            or path.name != _record_filename(dataset)
            or not _is_lower_sha256(digest)
            or dataset in result
        ):
            raise ValueError("checkpoint intent inventory is invalid")
        result[dataset] = (path, digest)
    return result


def _validate_dataset_inventory(
    root: Path, provenance: Exp426Provenance, digests: Mapping[str, str],
    *, allow_pending_records: bool = False,
) -> None:
    intents = _validated_intents(root, provenance)
    entries = list((root / "datasets").iterdir())
    names = {path.name for path in entries}
    expected_datasets = set(digests)
    if allow_pending_records:
        expected_datasets.update(
            dataset
            for dataset in intents
            if (root / "datasets" / _record_filename(dataset)).exists()
        )
    expected = {_record_filename(dataset) for dataset in expected_datasets}
    if names != expected:
        raise ValueError("checkpoint dataset inventory is invalid")
    for dataset in expected_datasets:
        path = root / "datasets" / _record_filename(dataset)
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint dataset record is invalid")
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        expected_digest = digests.get(dataset)
        if expected_digest is None:
            expected_digest = intents[dataset][1]
        if digest != expected_digest:
            raise ValueError("checkpoint dataset digest does not match")
        record = Exp426DatasetRecord.from_payload(_canonical_object(raw, "dataset record"))
        if record.dataset != dataset:
            raise ValueError("checkpoint dataset record name does not match")
