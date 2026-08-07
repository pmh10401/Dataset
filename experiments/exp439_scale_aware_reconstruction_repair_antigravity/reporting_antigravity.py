"""Deterministic review-only publication and replay for frozen Exp439 results."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from types import MappingProxyType
from typing import Iterable, Mapping
from uuid import uuid4

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .checkpoint_antigravity import Exp439CheckpointStore
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CHANNELS_ANTIGRAVITY,
    Exp439DatasetResult,
    reject_label_like_fields_antigravity,
)
from .runtime_antigravity import inspect_exp439_antigravity


PUBLICATION_DIRECTORY_ANTIGRAVITY = "publication_antigravity"
PREDICTION_FILE_ANTIGRAVITY = "prediction_antigravity.jsonl"
SUMMARY_FILE_ANTIGRAVITY = "dataset_summary_antigravity.jsonl"
ABSTENTION_FILE_ANTIGRAVITY = "abstention_ledger_antigravity.jsonl"
TRAIN_AUDIT_FILE_ANTIGRAVITY = "train_exclusion_audit_antigravity.jsonl"
BACKEND_FILE_ANTIGRAVITY = "backend_decision_antigravity.json"
MANIFEST_FILE_ANTIGRAVITY = "manifest_antigravity.json"
STATUS_FILE_ANTIGRAVITY = "status_antigravity.json"
HASH_FILE_ANTIGRAVITY = "artifact_sha256_antigravity.json"
PUBLICATION_FILES_ANTIGRAVITY = (
    PREDICTION_FILE_ANTIGRAVITY,
    SUMMARY_FILE_ANTIGRAVITY,
    ABSTENTION_FILE_ANTIGRAVITY,
    TRAIN_AUDIT_FILE_ANTIGRAVITY,
    BACKEND_FILE_ANTIGRAVITY,
    MANIFEST_FILE_ANTIGRAVITY,
    STATUS_FILE_ANTIGRAVITY,
    HASH_FILE_ANTIGRAVITY,
)


def _digest_antigravity(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _json_antigravity(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_antigravity(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_antigravity(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_rename_antigravity(source: Path, destination: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("exclusive publication rename requires Darwin")
    try:
        renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    except (AttributeError, OSError) as error:
        raise RuntimeError("Darwin exclusive rename is unavailable") from error
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), 0x00000004) != 0:
        number = ctypes.get_errno() or errno.EIO
        raise OSError(number, os.strerror(number), str(destination))


def _channel_antigravity(value) -> dict[str, object]:
    return {
        "score": value.score,
        "threshold": value.threshold,
        "abnormal": value.abnormal,
        "available": value.available,
        "reason": value.reason,
    }


def _prediction_rows_antigravity(
    results: tuple[Exp439DatasetResult, ...]
) -> tuple[dict[str, object], ...]:
    rows = []
    for result in results:
        for item in result.evidence:
            rows.append({
                "dataset": item.dataset,
                "instance_index": item.instance_index,
                "raw_length": item.raw_length,
                "outside_all_frozen_lanes": item.outside_all_frozen_lanes,
                "channels": list(CHANNELS_ANTIGRAVITY),
                "channel_evidence": {
                    channel.channel: _channel_antigravity(channel)
                    for channel in item.channels
                },
            })
    return tuple(rows)


def _summary_rows_antigravity(
    results: tuple[Exp439DatasetResult, ...]
) -> tuple[dict[str, object], ...]:
    rows = []
    for result in results:
        rows.append({
            "dataset": result.dataset,
            "scored_test_rows": len(result.evidence),
            "abstained_test_rows": len(result.abstentions),
            "excluded_train_rows": len(result.train_exclusions),
            "per_channel_candidate_counts": {
                channel: len(result.candidate_coordinates[channel])
                for channel in CHANNELS_ANTIGRAVITY
            },
        })
    return tuple(rows)


def _abstention_rows_antigravity(results: tuple[Exp439DatasetResult, ...]):
    return tuple(
        {
            "dataset": item.dataset,
            "instance_index": item.instance_index,
            "raw_length": item.raw_length,
            "reason": item.reason,
        }
        for result in results
        for item in result.abstentions
    )


def _train_audit_rows_antigravity(results: tuple[Exp439DatasetResult, ...]):
    return tuple(
        {
            "dataset": item.dataset,
            "instance_index": item.instance_index,
            "raw_length": item.raw_length,
            "nonfinite_positions": list(item.nonfinite_positions),
            "values_blob_sha256": item.values_blob_sha256,
        }
        for result in results
        for item in result.train_exclusions
    )


def _build_payloads_antigravity(
    output_root: Path,
    checkpoint_root: Path,
    *,
    expected_datasets_antigravity: int,
) -> tuple[dict[str, bytes], dict[str, object]]:
    if (
        type(expected_datasets_antigravity) is not int
        or expected_datasets_antigravity < 1
        or expected_datasets_antigravity > FROZEN_CONFIG.dataset_count
    ):
        raise ValueError("Exp439 publication population is invalid")
    runtime_summary = inspect_exp439_antigravity(output_root, checkpoint_root)
    store = Exp439CheckpointStore.inspect_existing(checkpoint_root)
    results = store.results()
    context = store.context_payload
    if (
        len(results) != expected_datasets_antigravity
        or runtime_summary.total_datasets != expected_datasets_antigravity
        or runtime_summary.completed_datasets != expected_datasets_antigravity
        or tuple(result.dataset for result in results) != tuple(context["datasets"])
        or runtime_summary.failed_datasets != 0
        or runtime_summary.prediction_label_access_count != 0
    ):
        raise ValueError("Exp439 publication requires a complete conserved population")
    prediction = _jsonl_antigravity(_prediction_rows_antigravity(results))
    summary = _jsonl_antigravity(_summary_rows_antigravity(results))
    abstentions = _jsonl_antigravity(_abstention_rows_antigravity(results))
    train_audit = _jsonl_antigravity(_train_audit_rows_antigravity(results))
    backend_payload = context["backend_decision"]
    backend = _json_antigravity(backend_payload)
    semantic = {
        PREDICTION_FILE_ANTIGRAVITY: prediction,
        SUMMARY_FILE_ANTIGRAVITY: summary,
        ABSTENTION_FILE_ANTIGRAVITY: abstentions,
        TRAIN_AUDIT_FILE_ANTIGRAVITY: train_audit,
        BACKEND_FILE_ANTIGRAVITY: backend,
    }
    semantic_hashes = {name: _digest_antigravity(raw) for name, raw in semantic.items()}
    manifest: dict[str, object] = {
        "schema": "exp439.publication-manifest.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "code_commit": context["code_commit"],
        "context_sha256": store.context_hash,
        "database_sha256": context["database_sha256"],
        "dataset_count": expected_datasets_antigravity,
        "prediction_sha256": semantic_hashes[PREDICTION_FILE_ANTIGRAVITY],
        "semantic_sha256": semantic_hashes,
        "backend_decision": backend_payload,
        "backend_provenance": context.get("backend_provenance", {"synthetic_or_legacy": True}),
        "readiness_sha256": context.get("readiness_sha256"),
        "source_sha256": context.get("source_sha256", {}),
        "source_provenance": {
            "aspe_url": "https://github.com/Brioal/ASPE",
            "aspe_commit": "f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89",
            "aspe_license": "Apache-2.0",
            "physdiff_repository": "https://github.com/vut-hfut/PhysDiff",
            "physdiff_code_adopted": False,
        },
        "review_only": True,
        "operational_alert_eligible": False,
        "prediction_label_access_count": 0,
        "channels": list(CHANNELS_ANTIGRAVITY),
    }
    reject_label_like_fields_antigravity(manifest)
    manifest_raw = _json_antigravity(manifest)
    status_payload = {
        "schema": "exp439.publication-status.v1",
        "state": "frozen",
        "review_only": True,
        "operational_alert_eligible": False,
        "prediction_sha256": manifest["prediction_sha256"],
        "manifest_sha256": _digest_antigravity(manifest_raw),
    }
    status_raw = _json_antigravity(status_payload)
    payloads = {
        **semantic,
        MANIFEST_FILE_ANTIGRAVITY: manifest_raw,
        STATUS_FILE_ANTIGRAVITY: status_raw,
    }
    hash_payload = {
        "schema": "exp439.artifact-hashes.v1",
        "files": {name: _digest_antigravity(raw) for name, raw in payloads.items()},
    }
    payloads[HASH_FILE_ANTIGRAVITY] = _json_antigravity(hash_payload)
    return payloads, manifest


@dataclass(frozen=True)
class FrozenPublicationAntigravity:
    publication: Path
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        root = Path(self.publication)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Exp439 publication root is invalid")
        object.__setattr__(self, "publication", root)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def prediction_sha256(self) -> str:
        return str(self.manifest["prediction_sha256"])


def publish_exp439_antigravity(
    output_root: Path,
    checkpoint_root: Path,
    *,
    expected_datasets_antigravity: int = FROZEN_CONFIG.dataset_count,
) -> FrozenPublicationAntigravity:
    output_root = Path(output_root)
    destination = output_root.parent / PUBLICATION_DIRECTORY_ANTIGRAVITY
    if destination.exists() or destination.is_symlink():
        raise ValueError("Exp439 publication already exists")
    payloads, manifest = _build_payloads_antigravity(
        output_root,
        Path(checkpoint_root),
        expected_datasets_antigravity=expected_datasets_antigravity,
    )
    staging = destination.with_name(
        f".publication_antigravity_{uuid4().hex}_staging_antigravity"
    )
    staging.mkdir(mode=0o700)
    try:
        for name in PUBLICATION_FILES_ANTIGRAVITY:
            _write_new_antigravity(staging / name, payloads[name])
        _fsync_directory_antigravity(staging)
        try:
            _exclusive_rename_antigravity(staging, destination)
        except FileExistsError as error:
            raise ValueError("Exp439 publication already exists") from error
        _fsync_directory_antigravity(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return inspect_publication_antigravity(
        destination,
        output_root,
        Path(checkpoint_root),
        expected_datasets_antigravity=expected_datasets_antigravity,
    )


def inspect_publication_antigravity(
    publication: Path,
    output_root: Path,
    checkpoint_root: Path,
    *,
    expected_datasets_antigravity: int = FROZEN_CONFIG.dataset_count,
) -> FrozenPublicationAntigravity:
    publication = Path(publication)
    if (
        not publication.is_dir()
        or publication.is_symlink()
        or {path.name for path in publication.iterdir()} != set(PUBLICATION_FILES_ANTIGRAVITY)
    ):
        raise ValueError("Exp439 publication inventory is invalid")
    expected, manifest = _build_payloads_antigravity(
        Path(output_root),
        Path(checkpoint_root),
        expected_datasets_antigravity=expected_datasets_antigravity,
    )
    for name, raw in expected.items():
        if (publication / name).read_bytes() != raw:
            raise ValueError(f"Exp439 publication hash or bytes differ: {name}")
    return FrozenPublicationAntigravity(publication, manifest)


__all__ = [
    "FrozenPublicationAntigravity",
    "PUBLICATION_FILES_ANTIGRAVITY",
    "inspect_publication_antigravity",
    "publish_exp439_antigravity",
]
