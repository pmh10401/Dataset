"""Ordered, fail-closed checkpoint storage owned only by Exp438's parent process."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    AbstentionEvidence,
    ExcludedTrainRowAudit,
    Exp438DatasetResult,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
    reject_label_like_fields_antigravity,
)


_SHA256_ANTIGRAVITY = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}\Z")
_ROOT_FILES_ANTIGRAVITY = {
    "context_antigravity.json",
    "state_antigravity.json",
    "records_antigravity",
}


def _digest_antigravity(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical_object_antigravity(path: Path, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_antigravity(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_antigravity(path.parent)


def _replace_antigravity(path: Path, raw: bytes) -> None:
    temporary = path.with_name(
        f".antigravity_{path.name}_{uuid4().hex}.tmp"
    )
    _write_new_antigravity(temporary, raw)
    try:
        os.replace(temporary, path)
        _fsync_directory_antigravity(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _context_payload_antigravity(context: object) -> dict[str, object]:
    if not isinstance(context, Mapping):
        raise ValueError("Exp438 context must be an object")
    try:
        payload = json.loads(canonical_json_bytes(dict(context)))
    except (TypeError, ValueError) as error:
        raise ValueError("Exp438 context is not JSON-compatible") from error
    reject_label_like_fields_antigravity(payload)
    if (
        payload.get("experiment_id") != FROZEN_CONFIG.experiment_id
        or payload.get("run_identity") != FROZEN_CONFIG.run_identity
    ):
        raise ValueError("Exp438 identity is invalid")
    if not isinstance(payload.get("code_commit"), str) or _COMMIT_ANTIGRAVITY.fullmatch(
        str(payload["code_commit"])
    ) is None:
        raise ValueError("Exp438 code commit is invalid")
    if not isinstance(payload.get("database_sha256"), str) or _SHA256_ANTIGRAVITY.fullmatch(
        str(payload["database_sha256"])
    ) is None:
        raise ValueError("Exp438 database digest is invalid")
    if payload["database_sha256"] != FROZEN_CONFIG.database_sha256:
        raise ValueError("Exp438 database digest does not match the frozen config")
    datasets = payload.get("datasets")
    if (
        not isinstance(datasets, list)
        or not datasets
        or any(type(item) is not str or not item for item in datasets)
        or datasets != sorted(datasets)
        or len(datasets) != len(set(datasets))
    ):
        raise ValueError("Exp438 datasets must be sorted and unique")
    if payload.get("backend") not in {"cpu", "numba", "mlx", "mps"}:
        raise ValueError("Exp438 backend is invalid")
    return payload


def context_fingerprint_antigravity(context: object) -> str:
    """Return the stable fingerprint after validating the frozen Exp438 identity."""
    return _digest_antigravity(canonical_json_bytes(_context_payload_antigravity(context)))


def _dataset_metadata_antigravity(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    keys = {
        "train_count", "test_count", "train_length_histogram", "test_length_histogram",
        "train_value_count", "test_value_count", "train_bytes", "test_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Exp438 dataset metadata is invalid")
    result = dict(value)
    for name in (
        "train_count", "test_count", "train_value_count", "test_value_count",
        "train_bytes", "test_bytes",
    ):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError("Exp438 dataset metadata is invalid")
    for name, expected in (
        ("train_length_histogram", int(result["train_count"])),
        ("test_length_histogram", int(result["test_count"])),
    ):
        histogram = result[name]
        if (
            not isinstance(histogram, Mapping)
            or any(
                type(key) is not str
                or not key.isdecimal()
                or int(key) < 1
                or type(count) is not int
                or count < 0
                for key, count in histogram.items()
            )
            or sum(histogram.values()) != expected
        ):
            raise ValueError("Exp438 dataset metadata is invalid")
        result[name] = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
    return result


def _channel_payload_antigravity(value: InstanceChannelEvidence) -> dict[str, object]:
    return {
        "channel": value.channel,
        "score": value.score,
        "threshold": value.threshold,
        "abnormal": value.abnormal,
        "available": value.available,
        "reason": value.reason,
    }


def _result_payload_antigravity(
    result: Exp438DatasetResult,
    sequence_index: int,
    dataset_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if type(result) is not Exp438DatasetResult:
        raise ValueError("Exp438 dataset result is invalid")
    payload: dict[str, object] = {
        "schema": "exp438.dataset-record.v1",
        "sequence_index": sequence_index,
        "result": {
            "dataset": result.dataset,
            "evidence": [
                {
                    "dataset": item.dataset,
                    "instance_index": item.instance_index,
                    "raw_length": item.raw_length,
                    "channels": [_channel_payload_antigravity(channel) for channel in item.channels],
                    "outside_all_frozen_lanes": item.outside_all_frozen_lanes,
                }
                for item in result.evidence
            ],
            "abstentions": [
                {
                    "dataset": item.dataset,
                    "instance_index": item.instance_index,
                    "raw_length": item.raw_length,
                    "reason": item.reason,
                }
                for item in result.abstentions
            ],
            "train_exclusions": [
                {
                    "dataset": item.dataset,
                    "instance_index": item.instance_index,
                    "raw_length": item.raw_length,
                    "nonfinite_positions": list(item.nonfinite_positions),
                    "values_blob_sha256": item.values_blob_sha256,
                }
                for item in result.train_exclusions
            ],
        },
    }
    if dataset_metadata is not None:
        payload["dataset_metadata"] = _dataset_metadata_antigravity(dataset_metadata)
    reject_label_like_fields_antigravity(payload)
    return payload


def _result_from_payload_antigravity(
    payload: object, *, expected_sequence: int
) -> Exp438DatasetResult:
    if (
        not isinstance(payload, Mapping)
        or set(payload) not in (
            {"schema", "sequence_index", "result"},
            {"schema", "sequence_index", "result", "dataset_metadata"},
        )
        or payload.get("schema") != "exp438.dataset-record.v1"
        or payload.get("sequence_index") != expected_sequence
        or not isinstance(payload.get("result"), Mapping)
    ):
        raise ValueError("Exp438 dataset record schema is invalid")
    reject_label_like_fields_antigravity(payload)
    if "dataset_metadata" in payload:
        _dataset_metadata_antigravity(payload["dataset_metadata"])
    result = payload["result"]
    if set(result) != {"dataset", "evidence", "abstentions", "train_exclusions"}:
        raise ValueError("Exp438 dataset result keys are invalid")
    try:
        evidence = tuple(
            InstanceScoreEvidence(
                dataset=item["dataset"],
                instance_index=item["instance_index"],
                raw_length=item["raw_length"],
                channels=tuple(InstanceChannelEvidence(**channel) for channel in item["channels"]),
                outside_all_frozen_lanes=item["outside_all_frozen_lanes"],
            )
            for item in result["evidence"]
        )
        abstentions = tuple(AbstentionEvidence(**item) for item in result["abstentions"])
        exclusions = tuple(
            ExcludedTrainRowAudit(
                dataset=item["dataset"],
                instance_index=item["instance_index"],
                raw_length=item["raw_length"],
                nonfinite_positions=tuple(item["nonfinite_positions"]),
                values_blob_sha256=item["values_blob_sha256"],
            )
            for item in result["train_exclusions"]
        )
        return Exp438DatasetResult(str(result["dataset"]), evidence, abstentions, exclusions)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp438 dataset record is invalid") from error


class Exp438CheckpointStore:
    """A single-writer ordered record store with strict inventory validation."""

    def __init__(
        self,
        root: Path,
        context: dict[str, object],
        state: dict[str, object],
        *,
        mutable: bool,
    ) -> None:
        self.root = Path(root)
        self._context = context
        self._state = state
        self._mutable = mutable

    @classmethod
    def create(cls, root: Path, context: object) -> "Exp438CheckpointStore":
        root = Path(root)
        payload = _context_payload_antigravity(context)
        if root.exists() or root.is_symlink():
            raise ValueError("Exp438 checkpoint root already exists")
        root.mkdir(mode=0o700, parents=False)
        records = root / "records_antigravity"
        records.mkdir(mode=0o700)
        _fsync_directory_antigravity(root.parent)
        context_raw = canonical_json_bytes(payload)
        context_hash = _digest_antigravity(context_raw)
        state: dict[str, object] = {
            "schema": "exp438.checkpoint-state.v1",
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "run_identity": FROZEN_CONFIG.run_identity,
            "context_hash": context_hash,
            "status": "running",
            "committed": [],
            "record_set_hash": _digest_antigravity(canonical_json_bytes([])),
            "failure": None,
        }
        _write_new_antigravity(root / "context_antigravity.json", context_raw)
        _write_new_antigravity(root / "state_antigravity.json", canonical_json_bytes(state))
        return cls(root, payload, state, mutable=True)

    @classmethod
    def inspect_existing(cls, root: Path) -> "Exp438CheckpointStore":
        """Open an Exp438 checkpoint read-only using only its own bound context."""
        root = Path(root)
        context = _canonical_object_antigravity(
            root / "context_antigravity.json", "Exp438 context"
        )
        return cls.open_existing(root, context, mutable=False)

    @classmethod
    def open_existing(
        cls, root: Path, context: object, *, mutable: bool
    ) -> "Exp438CheckpointStore":
        root = Path(root)
        supplied = _context_payload_antigravity(context)
        try:
            names = {path.name for path in root.iterdir()}
        except OSError as error:
            raise ValueError("Exp438 checkpoint root is invalid") from error
        if names != _ROOT_FILES_ANTIGRAVITY or not (root / "records_antigravity").is_dir():
            raise ValueError("Exp438 checkpoint inventory is invalid")
        stored = _canonical_object_antigravity(root / "context_antigravity.json", "Exp438 context")
        if stored != supplied:
            raise ValueError("Exp438 context does not match")
        state = _canonical_object_antigravity(root / "state_antigravity.json", "Exp438 state")
        cls._validate_state_antigravity(root, stored, state)
        if mutable and state["status"] == "failed":
            raise ValueError("failed checkpoint is immutable")
        if mutable and state["status"] == "complete":
            raise ValueError("complete checkpoint is immutable")
        return cls(root, stored, state, mutable=mutable)

    @staticmethod
    def _validate_state_antigravity(
        root: Path, context: Mapping[str, object], state: Mapping[str, object]
    ) -> None:
        required = {
            "schema", "experiment_id", "run_identity", "context_hash",
            "status", "committed", "record_set_hash", "failure",
        }
        if (
            set(state) != required
            or state.get("schema") != "exp438.checkpoint-state.v1"
            or state.get("experiment_id") != FROZEN_CONFIG.experiment_id
            or state.get("run_identity") != FROZEN_CONFIG.run_identity
            or state.get("context_hash") != _digest_antigravity(canonical_json_bytes(context))
            or state.get("status") not in {"running", "interrupted", "complete", "failed"}
            or not isinstance(state.get("committed"), list)
            or state.get("record_set_hash") != _digest_antigravity(
                canonical_json_bytes(state.get("committed"))
            )
        ):
            raise ValueError("Exp438 checkpoint state is invalid")
        failure = state.get("failure")
        if (state["status"] == "failed") != (type(failure) is str and bool(failure)):
            raise ValueError("Exp438 checkpoint failure state is invalid")
        datasets = tuple(context["datasets"])
        committed = state["committed"]
        if len(committed) > len(datasets):
            raise ValueError("Exp438 checkpoint prefix is invalid")
        expected_files: set[str] = set()
        for sequence, item in enumerate(committed):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"dataset", "file", "sha256"}
                or item.get("dataset") != datasets[sequence]
                or type(item.get("file")) is not str
                or _SHA256_ANTIGRAVITY.fullmatch(str(item.get("sha256"))) is None
            ):
                raise ValueError("Exp438 checkpoint prefix is invalid")
            filename = str(item["file"])
            expected_files.add(filename)
            raw_path = root / "records_antigravity" / filename
            try:
                raw = raw_path.read_bytes()
                record = json.loads(raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Exp438 record inventory is invalid") from error
            if canonical_json_bytes(record) != raw or _digest_antigravity(raw) != item["sha256"]:
                raise ValueError("Exp438 dataset record digest is invalid")
            parsed = _result_from_payload_antigravity(record, expected_sequence=sequence)
            if parsed.dataset != item["dataset"]:
                raise ValueError("Exp438 dataset record identity is invalid")
        actual_files = {path.name for path in (root / "records_antigravity").iterdir()}
        if actual_files != expected_files:
            raise ValueError("Exp438 record inventory is invalid")
        if state["status"] == "complete" and len(committed) != len(datasets):
            raise ValueError("Exp438 complete checkpoint is incomplete")

    @property
    def context_hash(self) -> str:
        return str(self._state["context_hash"])

    @property
    def committed_datasets(self) -> tuple[str, ...]:
        return tuple(str(item["dataset"]) for item in self._state["committed"])

    @property
    def status(self) -> str:
        return str(self._state["status"])

    @property
    def context_payload(self) -> dict[str, object]:
        return json.loads(canonical_json_bytes(self._context))

    def _require_mutable_antigravity(self) -> None:
        if not self._mutable:
            raise ValueError("checkpoint is read-only")
        if self._state["status"] in {"complete", "failed"}:
            raise ValueError(f"{self._state['status']} checkpoint is immutable")

    def _save_state_antigravity(self) -> None:
        _replace_antigravity(
            self.root / "state_antigravity.json",
            canonical_json_bytes(self._state),
        )

    def commit(
        self,
        result: Exp438DatasetResult,
        *,
        dataset_metadata: Mapping[str, object] | None = None,
    ) -> str:
        self._require_mutable_antigravity()
        sequence = len(self._state["committed"])
        datasets = tuple(self._context["datasets"])
        if sequence >= len(datasets) or result.dataset != datasets[sequence]:
            raise ValueError("result does not match the next frozen dataset")
        payload = _result_payload_antigravity(result, sequence, dataset_metadata)
        raw = canonical_json_bytes(payload)
        filename = (
            f"{sequence:04d}_{sha256(result.dataset.encode('utf-8')).hexdigest()[:16]}"
            "_antigravity.json"
        )
        _write_new_antigravity(self.root / "records_antigravity" / filename, raw)
        committed = list(self._state["committed"])
        committed.append({
            "dataset": result.dataset,
            "file": filename,
            "sha256": _digest_antigravity(raw),
        })
        self._state["committed"] = committed
        self._state["record_set_hash"] = _digest_antigravity(
            canonical_json_bytes(committed)
        )
        self._state["status"] = "running"
        self._save_state_antigravity()
        return _digest_antigravity(raw)

    def results(self) -> tuple[Exp438DatasetResult, ...]:
        results: list[Exp438DatasetResult] = []
        for sequence, item in enumerate(self._state["committed"]):
            record = _canonical_object_antigravity(
                self.root / "records_antigravity" / str(item["file"]),
                "Exp438 dataset record",
            )
            results.append(_result_from_payload_antigravity(record, expected_sequence=sequence))
        return tuple(results)

    def metadata_for_dataset(self, dataset: str) -> dict[str, object] | None:
        """Return a detached copy of one committed record's non-label size audit."""
        for item in self._state["committed"]:
            if item["dataset"] != dataset:
                continue
            record = _canonical_object_antigravity(
                self.root / "records_antigravity" / str(item["file"]),
                "Exp438 dataset record",
            )
            metadata = _dataset_metadata_antigravity(record.get("dataset_metadata"))
            return None if metadata is None else json.loads(canonical_json_bytes(metadata))
        return None

    def mark_interrupted(self) -> None:
        self._require_mutable_antigravity()
        self._state["status"] = "interrupted"
        self._save_state_antigravity()

    def mark_complete(self) -> None:
        self._require_mutable_antigravity()
        if len(self._state["committed"]) != len(self._context["datasets"]):
            raise ValueError("cannot complete an incomplete Exp438 checkpoint")
        self._state["status"] = "complete"
        self._save_state_antigravity()

    def mark_failed(self, failure: str) -> None:
        self._require_mutable_antigravity()
        if type(failure) is not str or not failure:
            raise ValueError("failure code is invalid")
        self._state["status"] = "failed"
        self._state["failure"] = failure
        self._save_state_antigravity()
