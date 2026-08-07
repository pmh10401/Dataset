"""Read-only, exact-byte validation of the immutable 89-record Exp438 prefix."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .checkpoint_antigravity import (
    _dataset_metadata_antigravity,
    _result_from_payload_antigravity,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    Exp439DatasetResult,
    reject_label_like_fields_antigravity,
)


EXP438_PREFIX_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp438_frequency_complexity_evidence_antigravity_20260807"
)
EXP438_REVIEWED_COMMIT_ANTIGRAVITY = (
    "d1da286e9c74c64abe76e0c255c86c30141813bd"
)
EXP438_READINESS_SHA256_ANTIGRAVITY = (
    "8ac3c67395002e064f295c5aa8cd5472412579bcbe65d7c268934243a4ec1c0a"
)
EXP438_CONTEXT_SHA256_ANTIGRAVITY = (
    "0630e884ab51d9d175bae3c82715694fa582375cf559f316cd83963fbe9f2fb6"
)
EXP438_STATE_SHA256_ANTIGRAVITY = (
    "9bbbbc58569e18dddbab211ffa1e2f267e23d7b8688256d0b9e3100093100fdc"
)
EXP438_PROGRESS_SHA256_ANTIGRAVITY = (
    "13a978b8f042d6345cd9fd000e32a9532930355e1320c8d3729dafffaa5791e3"
)
EXP438_RECORD_SET_SHA256_ANTIGRAVITY = (
    "a97422f66338d1a2a633c3c8441560b525d364199aa8f2681ff7b07d3826eb8d"
)
EXP438_RUN_IDENTITY_ANTIGRAVITY = (
    "exp438-frequency-complexity-evidence-antigravity-20260807"
)
EXP438_PREFIX_COUNT_ANTIGRAVITY = 89
EXP438_PREFIX_LAST_DATASET_ANTIGRAVITY = "BeetleFly_normal_1"
EXP438_PREFIX_NEXT_DATASET_ANTIGRAVITY = "BeetleFly_normal_2"

_ROOT_INVENTORY_ANTIGRAVITY = {
    "checkpoint_antigravity",
    "launcher_antigravity.pid",
    "output_antigravity",
    "readiness_antigravity.json",
}
_CHECKPOINT_INVENTORY_ANTIGRAVITY = {
    "context_antigravity.json",
    "records_antigravity",
    "state_antigravity.json",
}
_OUTPUT_INVENTORY_ANTIGRAVITY = {
    "events_antigravity.jsonl",
    "parent_antigravity.log",
    "progress_antigravity.json",
    "run_antigravity.pid",
}


def _digest_antigravity(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _canonical_object_antigravity(path: Path, expected_sha256: str, label: str) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Exp438 {label} source is invalid")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Exp438 {label} source is invalid") from error
    if (
        _digest_antigravity(raw) != expected_sha256
        or not isinstance(payload, dict)
        or canonical_json_bytes(payload) != raw
    ):
        raise ValueError(f"Exp438 {label} source hash or canonical JSON is invalid")
    return raw, payload


def _inventory_antigravity(path: Path, expected: set[str], label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Exp438 {label} inventory is invalid")
    try:
        names = {item.name for item in path.iterdir()}
    except OSError as error:
        raise ValueError(f"Exp438 {label} inventory is invalid") from error
    if names != expected:
        raise ValueError(f"Exp438 {label} inventory is invalid")


@dataclass(frozen=True)
class Exp438PrefixRecordAntigravity:
    sequence_index: int
    dataset: str
    source_path: Path
    source_sha256: str
    source_bytes: bytes
    dataset_metadata_bytes: bytes
    result: Exp439DatasetResult

    @property
    def dataset_metadata(self) -> dict[str, object] | None:
        if not self.dataset_metadata_bytes:
            return None
        value = json.loads(self.dataset_metadata_bytes)
        if not isinstance(value, dict):
            raise ValueError("Exp438 imported dataset metadata is invalid")
        return value


@dataclass(frozen=True)
class Exp438PrefixImportAntigravity:
    source_root: Path
    reviewed_commit: str
    database_sha256: str
    readiness_sha256: str
    context_sha256: str
    state_sha256: str
    progress_sha256: str
    record_set_sha256: str
    datasets: tuple[str, ...]
    records: tuple[Exp438PrefixRecordAntigravity, ...]
    prediction_label_access_count: int

    def manifest_payload_antigravity(self) -> dict[str, object]:
        return {
            "schema": "exp439.exp438-prefix-import.v1",
            "predecessor_experiment_id": "Exp438",
            "predecessor_run_identity": EXP438_RUN_IDENTITY_ANTIGRAVITY,
            "predecessor_root": str(self.source_root),
            "reviewed_commit": self.reviewed_commit,
            "database_sha256": self.database_sha256,
            "readiness_sha256": self.readiness_sha256,
            "context_sha256": self.context_sha256,
            "state_sha256": self.state_sha256,
            "progress_sha256": self.progress_sha256,
            "record_set_sha256": self.record_set_sha256,
            "record_count": len(self.records),
            "last_dataset": self.records[-1].dataset,
            "next_dataset": self.datasets[len(self.records)],
            "prediction_label_access_count": self.prediction_label_access_count,
            "records": [
                {
                    "sequence_index": record.sequence_index,
                    "dataset": record.dataset,
                    "source_path": str(record.source_path),
                    "source_sha256": record.source_sha256,
                }
                for record in self.records
            ],
        }


def _validate_exp438_prefix_source_antigravity(
    root: Path,
) -> Exp438PrefixImportAntigravity:
    """Validate an exact-byte mirror without mutating it; production fixes its path."""
    source_root = Path(root)
    _inventory_antigravity(source_root, _ROOT_INVENTORY_ANTIGRAVITY, "root")
    checkpoint = source_root / "checkpoint_antigravity"
    output = source_root / "output_antigravity"
    _inventory_antigravity(checkpoint, _CHECKPOINT_INVENTORY_ANTIGRAVITY, "checkpoint")
    _inventory_antigravity(output, _OUTPUT_INVENTORY_ANTIGRAVITY, "output")
    records_root = checkpoint / "records_antigravity"
    if records_root.is_symlink() or not records_root.is_dir():
        raise ValueError("Exp438 record inventory is invalid")

    readiness_raw, readiness = _canonical_object_antigravity(
        source_root / "readiness_antigravity.json",
        EXP438_READINESS_SHA256_ANTIGRAVITY,
        "readiness",
    )
    context_raw, context = _canonical_object_antigravity(
        checkpoint / "context_antigravity.json",
        EXP438_CONTEXT_SHA256_ANTIGRAVITY,
        "context",
    )
    state_raw, state = _canonical_object_antigravity(
        checkpoint / "state_antigravity.json",
        EXP438_STATE_SHA256_ANTIGRAVITY,
        "state",
    )
    progress_raw, progress = _canonical_object_antigravity(
        output / "progress_antigravity.json",
        EXP438_PROGRESS_SHA256_ANTIGRAVITY,
        "progress",
    )

    datasets_raw = readiness.get("datasets")
    if (
        readiness.get("schema") != "exp438.readiness.v1"
        or readiness.get("experiment_id") != "Exp438"
        or readiness.get("run_identity") != EXP438_RUN_IDENTITY_ANTIGRAVITY
        or readiness.get("reviewed_commit") != EXP438_REVIEWED_COMMIT_ANTIGRAVITY
        or readiness.get("database_sha256") != FROZEN_CONFIG.database_sha256
        or readiness.get("run_parent") != str(EXP438_PREFIX_ROOT_ANTIGRAVITY)
        or readiness.get("prediction_label_access_count") != 0
        or readiness.get("test_labels_opened") != 0
        or readiness.get("test_values_opened") != 0
        or not isinstance(datasets_raw, list)
        or len(datasets_raw) != FROZEN_CONFIG.dataset_count
        or datasets_raw != sorted(datasets_raw)
        or len(set(datasets_raw)) != len(datasets_raw)
    ):
        raise ValueError("Exp438 readiness identity or label boundary is invalid")
    datasets = tuple(str(item) for item in datasets_raw)

    if (
        context.get("experiment_id") != "Exp438"
        or context.get("run_identity") != EXP438_RUN_IDENTITY_ANTIGRAVITY
        or context.get("code_commit") != EXP438_REVIEWED_COMMIT_ANTIGRAVITY
        or context.get("database_sha256") != FROZEN_CONFIG.database_sha256
        or context.get("datasets") != list(datasets)
        or context.get("backend") != "cpu"
        or context.get("prediction_label_access_count") != 0
        or context.get("readiness_sha256") != _digest_antigravity(readiness_raw)
        or context.get("source_paths") != readiness.get("source_paths")
        or context.get("source_sha256") != readiness.get("source_sha256")
        or context.get("frozen_config") != readiness.get("frozen_config")
        or context.get("backend_decision") != readiness.get("backend_decision")
        or context.get("backend_provenance") != readiness.get("backend_provenance")
    ):
        raise ValueError("Exp438 context binding is invalid")

    committed = state.get("committed")
    if (
        state.get("schema") != "exp438.checkpoint-state.v1"
        or state.get("experiment_id") != "Exp438"
        or state.get("run_identity") != EXP438_RUN_IDENTITY_ANTIGRAVITY
        or state.get("context_hash") != _digest_antigravity(context_raw)
        or state.get("status") != "failed"
        or state.get("failure") != "ValueError"
        or state.get("record_set_hash") != EXP438_RECORD_SET_SHA256_ANTIGRAVITY
        or not isinstance(committed, list)
        or len(committed) != EXP438_PREFIX_COUNT_ANTIGRAVITY
        or _digest_antigravity(canonical_json_bytes(committed))
        != EXP438_RECORD_SET_SHA256_ANTIGRAVITY
    ):
        raise ValueError("Exp438 failed checkpoint binding is invalid")

    if (
        progress.get("schema") != "exp438.runtime-progress.v1"
        or progress.get("state") != "failed"
        or progress.get("backend") != "cpu"
        or progress.get("context_hash") != _digest_antigravity(context_raw)
        or progress.get("completed") != EXP438_PREFIX_COUNT_ANTIGRAVITY
        or progress.get("total") != FROZEN_CONFIG.dataset_count
        or progress.get("import_count") != 0
        or progress.get("recompute_count") != EXP438_PREFIX_COUNT_ANTIGRAVITY
        or progress.get("success_count") != 74
        or progress.get("abstention_count") != 15
        or progress.get("error_count") != 1
        or progress.get("checkpoint") != EXP438_PREFIX_LAST_DATASET_ANTIGRAVITY
        or progress.get("latest_error") != {"code": "ValueError"}
        or progress.get("current_dataset") is not None
    ):
        raise ValueError("Exp438 failed progress binding is invalid")

    expected_files: set[str] = set()
    imported_records: list[Exp438PrefixRecordAntigravity] = []
    for sequence_index, item in enumerate(committed):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"dataset", "file", "sha256"}
            or item.get("dataset") != datasets[sequence_index]
            or type(item.get("file")) is not str
            or type(item.get("sha256")) is not str
        ):
            raise ValueError("Exp438 ordered prefix is invalid")
        filename = str(item["file"])
        expected_files.add(filename)
        source_path = records_root / filename
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("Exp438 record inventory is invalid")
        source_bytes = source_path.read_bytes()
        if _digest_antigravity(source_bytes) != item["sha256"]:
            raise ValueError("Exp438 record SHA256 is invalid")
        try:
            source_payload = json.loads(source_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp438 record JSON is invalid") from error
        if (
            not isinstance(source_payload, dict)
            or canonical_json_bytes(source_payload) != source_bytes
            or source_payload.get("schema") != "exp438.dataset-record.v1"
            or source_payload.get("sequence_index") != sequence_index
            or not isinstance(source_payload.get("result"), Mapping)
            or source_payload["result"].get("dataset") != datasets[sequence_index]
        ):
            raise ValueError("Exp438 record payload is invalid")
        reject_label_like_fields_antigravity(source_payload)
        metadata = _dataset_metadata_antigravity(source_payload.get("dataset_metadata"))
        converted = json.loads(source_bytes)
        converted["schema"] = "exp439.dataset-record.v1"
        result = _result_from_payload_antigravity(
            converted,
            expected_sequence=sequence_index,
        )
        metadata_bytes = b"" if metadata is None else canonical_json_bytes(metadata)
        imported_records.append(
            Exp438PrefixRecordAntigravity(
                sequence_index=sequence_index,
                dataset=datasets[sequence_index],
                source_path=source_path.resolve(),
                source_sha256=str(item["sha256"]),
                source_bytes=bytes(source_bytes),
                dataset_metadata_bytes=metadata_bytes,
                result=result,
            )
        )

    actual_files = {item.name for item in records_root.iterdir()}
    if actual_files != expected_files:
        raise ValueError("Exp438 record inventory is invalid")
    if (
        imported_records[-1].dataset != EXP438_PREFIX_LAST_DATASET_ANTIGRAVITY
        or datasets[EXP438_PREFIX_COUNT_ANTIGRAVITY]
        != EXP438_PREFIX_NEXT_DATASET_ANTIGRAVITY
    ):
        raise ValueError("Exp438 prefix boundary is invalid")

    return Exp438PrefixImportAntigravity(
        source_root=source_root.resolve(),
        reviewed_commit=EXP438_REVIEWED_COMMIT_ANTIGRAVITY,
        database_sha256=FROZEN_CONFIG.database_sha256,
        readiness_sha256=_digest_antigravity(readiness_raw),
        context_sha256=_digest_antigravity(context_raw),
        state_sha256=_digest_antigravity(state_raw),
        progress_sha256=_digest_antigravity(progress_raw),
        record_set_sha256=EXP438_RECORD_SET_SHA256_ANTIGRAVITY,
        datasets=datasets,
        records=tuple(imported_records),
        prediction_label_access_count=0,
    )


def validate_exp438_prefix_antigravity(
    root: Path = EXP438_PREFIX_ROOT_ANTIGRAVITY,
) -> Exp438PrefixImportAntigravity:
    source = Path(root)
    if source.is_symlink() or source.resolve() != EXP438_PREFIX_ROOT_ANTIGRAVITY.resolve():
        raise ValueError("Exp438 predecessor root is invalid")
    return _validate_exp438_prefix_source_antigravity(source)


__all__ = [
    "EXP438_PREFIX_ROOT_ANTIGRAVITY",
    "Exp438PrefixImportAntigravity",
    "Exp438PrefixRecordAntigravity",
    "validate_exp438_prefix_antigravity",
]
