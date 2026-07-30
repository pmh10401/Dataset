"""Atomic local checkpoints for resumable Exp411 qualification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .contracts import DatasetBackendAudit, DatasetQualification
from .perturbations import perturbation_types


_SCHEMA_VERSION = 1
_ROOT_FILE = "checkpoint_provenance.json"
_PROGRESS_FILE = "progress.json"


@dataclass(frozen=True)
class CheckpointProvenance:
    git_commit: str
    database_snapshot: Mapping[str, str]
    population_manifest_sha256: str
    frozen_config_sha256: str
    perturbation_contract_sha256: str
    acceleration_contract_sha256: str

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        _validate_provenance_payload(payload)
        return payload


class CheckpointStore:
    def __init__(
        self,
        root: Path,
        provenance: CheckpointProvenance,
        expected_datasets: Sequence[str],
    ) -> None:
        self.root = Path(root)
        self.datasets_dir = self.root / "datasets"
        self.provenance = provenance
        self.expected_datasets = tuple(str(value) for value in expected_datasets)
        if (
            not self.expected_datasets
            or self.expected_datasets != tuple(sorted(self.expected_datasets))
            or len(set(self.expected_datasets)) != len(self.expected_datasets)
        ):
            raise ValueError("Exp411 checkpoint dataset universe is invalid")
        root_payload = {
            "schema_version": _SCHEMA_VERSION,
            "provenance": provenance.to_payload(),
            "expected_datasets": list(self.expected_datasets),
        }
        if self.root.exists():
            self._validate_existing_root(root_payload)
        else:
            if not self.root.parent.is_dir():
                raise FileNotFoundError(
                    "Exp411 checkpoint parent directory does not exist"
                )
            self.root.mkdir()
            self.datasets_dir.mkdir()
            _atomic_write(
                self.root / _ROOT_FILE,
                canonical_json_bytes(root_payload) + b"\n",
                replace_existing=False,
            )
            _fsync_directory(self.root)

    def load_completed(self) -> Mapping[str, DatasetQualification]:
        records = self._load_records()
        return MappingProxyType(
            {
                dataset: record["qualification"]
                for dataset, record in sorted(records.items())
            }
        )

    def load_dataset_record(self, dataset: str) -> Mapping[str, object]:
        name = str(dataset)
        if name not in self.expected_datasets:
            raise ValueError(
                "Exp411 checkpoint dataset is outside the universe"
            )
        records = self._load_records()
        if name not in records:
            raise ValueError(
                f"Exp411 checkpoint dataset is not completed: {name}"
            )
        return MappingProxyType(dict(records[name]))

    def write_dataset(
        self,
        row: DatasetQualification,
        audit: DatasetBackendAudit,
        dataset_train_identity_sha256: str,
        elapsed_seconds: float,
    ) -> None:
        if row.dataset not in self.expected_datasets:
            raise ValueError("Exp411 checkpoint dataset is outside the universe")
        _validate_hex_digest(
            dataset_train_identity_sha256,
            "dataset TRAIN identity",
        )
        if (
            not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0.0
        ):
            raise ValueError("Exp411 checkpoint elapsed time is invalid")
        qualification = _qualification_to_payload(row)
        backend_audit = _audit_to_payload(audit)
        body: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "dataset": row.dataset,
            "dataset_train_identity_sha256": dataset_train_identity_sha256,
            "qualification": qualification,
            "backend_audit": backend_audit,
            "elapsed_seconds": float(elapsed_seconds),
        }
        body["payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
        path = self._dataset_path(row.dataset)
        if path.exists():
            raise ValueError(
                f"Exp411 checkpoint already exists for dataset: {row.dataset}"
            )
        _atomic_write(
            path,
            canonical_json_bytes(body) + b"\n",
            replace_existing=False,
        )
        _fsync_directory(self.datasets_dir)

    def write_progress(self, *, current_dataset: str | None) -> None:
        if (
            current_dataset is not None
            and current_dataset not in self.expected_datasets
        ):
            raise ValueError("Exp411 checkpoint progress dataset is invalid")
        payload = {
            "completed_datasets": len(self._load_records()),
            "current_dataset": current_dataset,
            "expected_datasets": len(self.expected_datasets),
        }
        _atomic_write(
            self.root / _PROGRESS_FILE,
            canonical_json_bytes(payload) + b"\n",
            replace_existing=True,
        )
        _fsync_directory(self.root)

    def require_complete(
        self,
        expected_families: int,
    ) -> tuple[DatasetQualification, ...]:
        completed = self.load_completed()
        if set(completed) != set(self.expected_datasets):
            raise ValueError("Exp411 checkpoint dataset coverage is incomplete")
        rows = tuple(completed[dataset] for dataset in self.expected_datasets)
        families = {row.family for row in rows}
        if len(families) != int(expected_families):
            raise ValueError("Exp411 checkpoint family coverage is invalid")
        return rows

    def _validate_existing_root(
        self,
        expected_payload: Mapping[str, object],
    ) -> None:
        if not self.root.is_dir() or not self.datasets_dir.is_dir():
            raise ValueError("Exp411 checkpoint root is invalid")
        path = self.root / _ROOT_FILE
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Exp411 checkpoint provenance is invalid") from error
        if observed != dict(expected_payload):
            raise ValueError("Exp411 checkpoint provenance mismatch")

    def _load_records(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for path in sorted(self.datasets_dir.glob("*.json")):
            record = _load_record(path)
            dataset = str(record["dataset"])
            if dataset not in self.expected_datasets:
                raise ValueError(
                    "Exp411 checkpoint contains a foreign dataset"
                )
            if dataset in result:
                raise ValueError(
                    f"Exp411 checkpoint dataset is duplicated: {dataset}"
                )
            result[dataset] = record
        return result

    def _dataset_path(self, dataset: str) -> Path:
        name = sha256(dataset.encode("utf-8")).hexdigest()
        return self.datasets_dir / f"{name}.json"


def _load_record(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Exp411 dataset checkpoint is corrupt: {path.name}"
        ) from error
    expected_keys = {
        "schema_version",
        "dataset",
        "dataset_train_identity_sha256",
        "qualification",
        "backend_audit",
        "elapsed_seconds",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Exp411 dataset checkpoint schema is invalid")
    digest = payload.pop("payload_sha256")
    if not isinstance(digest, str) or sha256(
        canonical_json_bytes(payload)
    ).hexdigest() != digest:
        raise ValueError("Exp411 dataset checkpoint payload hash mismatch")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("Exp411 dataset checkpoint version is invalid")
    dataset = str(payload["dataset"])
    _validate_hex_digest(
        str(payload["dataset_train_identity_sha256"]),
        "dataset TRAIN identity",
    )
    elapsed = float(payload["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("Exp411 dataset checkpoint elapsed time is invalid")
    qualification = _qualification_from_payload(payload["qualification"])
    if qualification.dataset != dataset:
        raise ValueError("Exp411 dataset checkpoint identity mismatch")
    audit = _audit_from_payload(payload["backend_audit"])
    return {
        **payload,
        "qualification": qualification,
        "backend_audit": audit,
        "payload_sha256": digest,
    }


def _qualification_to_payload(
    row: DatasetQualification,
) -> dict[str, object]:
    payload = asdict(row)
    _qualification_from_payload(payload)
    return payload


def _qualification_from_payload(value: object) -> DatasetQualification:
    expected_keys = {
        "dataset",
        "family",
        "identity_group_count",
        "normal_total",
        "normal_abstained",
        "normal_near",
        "synthetic_total",
        "synthetic_near",
        "perturbation_total",
        "perturbation_near",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Exp411 checkpoint qualification schema is invalid")
    dataset = str(value["dataset"])
    family = str(value["family"])
    if dataset.count("_normal_") != 1 or dataset.split("_normal_", 1)[0] != family:
        raise ValueError("Exp411 checkpoint qualification identity is invalid")
    keys = ("0.01", "0.05", "0.10")
    types = perturbation_types()
    normal_near = _integer_mapping(value["normal_near"], keys)
    synthetic_total = _integer_mapping(value["synthetic_total"], keys)
    synthetic_near = _integer_mapping(value["synthetic_near"], keys)
    perturbation_total = _nested_integer_mapping(
        value["perturbation_total"],
        keys,
        types,
    )
    perturbation_near = _nested_integer_mapping(
        value["perturbation_near"],
        keys,
        types,
    )
    identity_group_count = _strict_counter(value["identity_group_count"])
    normal_total = _strict_counter(value["normal_total"])
    normal_abstained = _strict_counter(value["normal_abstained"])
    if (
        identity_group_count < 1
        or normal_total < 1
        or not 0 <= normal_abstained <= normal_total
        or any(
            normal_near[key] > normal_total - normal_abstained
            for key in keys
        )
    ):
        raise ValueError("Exp411 checkpoint qualification counts are invalid")
    for key in keys:
        if (
            synthetic_total[key] != sum(perturbation_total[key].values())
            or synthetic_near[key] != sum(perturbation_near[key].values())
            or synthetic_near[key] > synthetic_total[key]
            or any(
                perturbation_near[key][name]
                > perturbation_total[key][name]
                for name in types
            )
        ):
            raise ValueError(
                "Exp411 checkpoint perturbation counts are invalid"
            )
    return DatasetQualification(
        dataset=dataset,
        family=family,
        identity_group_count=identity_group_count,
        normal_total=normal_total,
        normal_abstained=normal_abstained,
        normal_near=normal_near,
        synthetic_total=synthetic_total,
        synthetic_near=synthetic_near,
        perturbation_total=perturbation_total,
        perturbation_near=perturbation_near,
    )


def _audit_to_payload(audit: DatasetBackendAudit) -> dict[str, object]:
    payload = asdict(audit)
    _audit_from_payload(payload)
    return payload


def _audit_from_payload(value: object) -> DatasetBackendAudit:
    expected_keys = {
        "backend",
        "sampled_pairs",
        "max_abs_error",
        "fallback_reason",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("Exp411 checkpoint backend audit schema is invalid")
    backend = str(value["backend"])
    if backend not in {"numpy_cpu", "numba_cpu"}:
        raise ValueError("Exp411 checkpoint backend audit is invalid")
    try:
        sampled_pairs = tuple(
            (int(pair[0]), int(pair[1]))
            for pair in value["sampled_pairs"]
        )
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError(
            "Exp411 checkpoint backend audit pairs are invalid"
        ) from error
    if any(
        left < 0 or right < 0 or left == right
        for left, right in sampled_pairs
    ):
        raise ValueError("Exp411 checkpoint backend audit pairs are invalid")
    max_abs_error = float(value["max_abs_error"])
    fallback = value["fallback_reason"]
    if (
        not math.isfinite(max_abs_error)
        or max_abs_error < 0.0
        or (fallback is not None and not isinstance(fallback, str))
    ):
        raise ValueError("Exp411 checkpoint backend audit values are invalid")
    return DatasetBackendAudit(
        backend=backend,
        sampled_pairs=sampled_pairs,
        max_abs_error=max_abs_error,
        fallback_reason=fallback,
    )


def _integer_mapping(
    value: object,
    expected_keys: Sequence[str],
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError("Exp411 checkpoint counter mapping is invalid")
    return {
        key: _strict_counter(value[key])
        for key in expected_keys
    }


def _strict_counter(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Exp411 checkpoint counter is invalid")
    return value


def _nested_integer_mapping(
    value: object,
    outer_keys: Sequence[str],
    inner_keys: Sequence[str],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict) or set(value) != set(outer_keys):
        raise ValueError("Exp411 checkpoint nested mapping is invalid")
    return {
        key: _integer_mapping(value[key], inner_keys)
        for key in outer_keys
    }


def _validate_provenance_payload(payload: Mapping[str, object]) -> None:
    git_commit = str(payload["git_commit"])
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit
    ):
        raise ValueError("Exp411 checkpoint Git provenance is invalid")
    snapshot = payload["database_snapshot"]
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Exp411 checkpoint database provenance is invalid")
    for name, digest in snapshot.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Exp411 checkpoint database key is invalid")
        _validate_hex_digest(str(digest), f"database {name}")
    for name in (
        "population_manifest_sha256",
        "frozen_config_sha256",
        "perturbation_contract_sha256",
        "acceleration_contract_sha256",
    ):
        _validate_hex_digest(str(payload[name]), name)


def _validate_hex_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"Exp411 checkpoint {name} SHA256 is invalid")


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    replace_existing: bool,
) -> None:
    if path.exists() and not replace_existing:
        raise FileExistsError(f"Exp411 checkpoint path exists: {path}")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not replace_existing:
            raise FileExistsError(f"Exp411 checkpoint path exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CheckpointProvenance",
    "CheckpointStore",
]
