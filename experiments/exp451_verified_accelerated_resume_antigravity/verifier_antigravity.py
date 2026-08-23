"""Label-free Exp451 completion verifier and frozen condition-manifest authority.

The module deliberately does not import a database or TEST-label accessor.  It
first checks every typed checkpoint envelope and only returns an authority that
the evaluation module may use after all four condition inventories are complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType

from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    _route_from_payload,
    _route_payload,
    _source_payload,
    _sources_from_payload,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450DatasetConditionResult,
)
from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import (
    compose_all_conditions_antigravity,
)

from .artifacts_antigravity import (
    SUCCESS_SCHEMA_ANTIGRAVITY,
    success_relative_path_antigravity,
)
from .config_antigravity import (
    EXP450_SOURCE_CONTEXT_SHA256,
    EXP451_FINAL_COUNTS_BY_CONDITION,
    EXP451_MIGRATED_COUNTS_BY_CONDITION,
    EXP451_TOTAL_COORDINATES,
)
from .contracts_antigravity import (
    Exp451MigrationManifest,
    Exp451RunContext,
    canonical_json_bytes,
)
from .runner_antigravity import (
    _NATIVE_ORIGIN_ANTIGRAVITY,
    _NATIVE_SCHEMA_ANTIGRAVITY,
    _native_relative_path_antigravity,
)


_CONDITIONS_ANTIGRAVITY = ("A", "B", "C", "D")
_IMPORTED_ORIGIN_ANTIGRAVITY = "imported_exp450_mps"
_CONDITION_MODES_ANTIGRAVITY = {
    "A": ("linear", "linear"), "B": ("polyphase_pchip", "linear"),
    "C": ("linear", "area"), "D": ("polyphase_pchip", "area"),
}
_CONDITION_MANIFEST_SCHEMA_ANTIGRAVITY = "exp451-verified-condition-manifest-antigravity.v1"


@dataclass(frozen=True, slots=True)
class Exp451VerificationAntigravity:
    """The only completion authority accepted by Exp451 evaluation/reporting."""

    total_successes: int
    imported_by_condition: Mapping[str, int]
    native_by_condition: Mapping[str, int]
    condition_manifests: Mapping[str, Mapping[str, object]]


def _sha256_antigravity(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _canonical_object_antigravity(path: Path) -> tuple[dict[str, object], bytes]:
    detail = path.lstat() if path.exists() or path.is_symlink() else None
    if detail is None or stat.S_ISLNK(detail.st_mode) or not stat.S_ISREG(detail.st_mode):
        raise ValueError("Exp451 checkpoint is missing or unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp451 checkpoint JSON is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("Exp451 checkpoint is noncanonical")
    return value, raw


def _catalog_entries_antigravity(catalog: Mapping[str, object], context: Exp451RunContext) -> tuple[dict[str, object], ...]:
    entries = catalog.get("datasets") if isinstance(catalog, Mapping) else None
    if (
        not isinstance(entries, list) or len(entries) != 1117
        or catalog.get("catalog_sha256") != context.catalog_sha256
    ):
        raise ValueError("Exp451 catalog differs")
    datasets = tuple(entry.get("dataset") if isinstance(entry, dict) else None for entry in entries)
    if (
        any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(set(datasets)) != len(datasets) or tuple(sorted(datasets)) != datasets
    ):
        raise ValueError("Exp451 catalog dataset inventory differs")
    return tuple(entries)


def _typed_payload_antigravity(payload: object, entry: Mapping[str, object]) -> None:
    """Round-trip source/route payload through the frozen typed Exp450 closure."""
    train = entry.get("train")
    test = entry.get("test")
    if (
        not isinstance(payload, dict) or set(payload) != {"sources", "route"}
        or not isinstance(train, Mapping) or not isinstance(test, Mapping)
        or type(train.get("count")) is not int or type(test.get("count")) is not int
    ):
        raise ValueError("Exp451 checkpoint typed payload differs")
    try:
        sources = _sources_from_payload(payload["sources"], train["count"], test["count"])
        route = _route_from_payload(payload["route"], test["count"])
        round_trip = {
            "sources": _source_payload(sources, train["count"], test["count"]),
            "route": _route_payload(route, test["count"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Exp451 checkpoint typed payload differs") from error
    if canonical_json_bytes(round_trip) != canonical_json_bytes(payload):
        raise ValueError("Exp451 checkpoint typed payload round-trip differs")


def _check_imported_envelope_antigravity(
    value: Mapping[str, object], *, entry: Mapping[str, object], condition: str,
    context: Exp451RunContext, migration_entry: object,
) -> None:
    required = {
        "schema", "context_sha256", "source_context_sha256", "catalog_sha256",
        "condition_id", "dataset", "raw_identity", "backend", "execution_origin",
        "payload", "payload_sha256", "source_envelope_sha256",
    }
    if (
        set(value) != required or value.get("schema") != SUCCESS_SCHEMA_ANTIGRAVITY
        or value.get("context_sha256") != context.sha256
        or value.get("source_context_sha256") != EXP450_SOURCE_CONTEXT_SHA256
        or value.get("catalog_sha256") != context.catalog_sha256
        or value.get("dataset") != entry.get("dataset") or value.get("condition_id") != condition
        or value.get("raw_identity") != entry or value.get("backend") != context.backend.artifact_record()
        or value.get("execution_origin") != _IMPORTED_ORIGIN_ANTIGRAVITY
        or not _sha256_antigravity(value.get("payload_sha256"))
        or sha256(canonical_json_bytes(value.get("payload"))).hexdigest() != value.get("payload_sha256")
        or value.get("payload_sha256") != getattr(migration_entry, "payload_sha256", None)
        or value.get("source_envelope_sha256") != getattr(migration_entry, "source_envelope_sha256", None)
    ):
        raise ValueError("Exp451 imported checkpoint binding differs")
    _typed_payload_antigravity(value["payload"], entry)


def _check_native_envelope_antigravity(
    value: Mapping[str, object], *, entry: Mapping[str, object], condition: str,
    context: Exp451RunContext,
) -> None:
    required = {
        "schema", "context_sha256", "catalog_sha256", "condition_id", "dataset",
        "raw_identity", "backend", "execution_origin", "payload", "payload_sha256",
    }
    if (
        set(value) != required or value.get("schema") != _NATIVE_SCHEMA_ANTIGRAVITY
        or value.get("context_sha256") != context.sha256
        or value.get("catalog_sha256") != context.catalog_sha256
        or value.get("dataset") != entry.get("dataset") or value.get("condition_id") != condition
        or value.get("raw_identity") != entry or value.get("backend") != context.backend.artifact_record()
        or value.get("execution_origin") != _NATIVE_ORIGIN_ANTIGRAVITY
        or not _sha256_antigravity(value.get("payload_sha256"))
        or sha256(canonical_json_bytes(value.get("payload"))).hexdigest() != value.get("payload_sha256")
    ):
        raise ValueError("Exp451 native checkpoint binding differs")
    _typed_payload_antigravity(value["payload"], entry)


def _condition_manifest_antigravity(
    *, condition: str, entries: Sequence[Mapping[str, object]], context: Exp451RunContext,
    checkpoint_hashes: Sequence[Mapping[str, str]], imported: int, native: int,
) -> Mapping[str, object]:
    length_mode, image_mode = _CONDITION_MODES_ANTIGRAVITY[condition]
    bare = {
        "schema": _CONDITION_MANIFEST_SCHEMA_ANTIGRAVITY,
        "database_sha256": context.database_sha256,
        "context_sha256": context.sha256,
        "catalog_sha256": context.catalog_sha256,
        "condition": {"condition_id": condition, "length_mode": length_mode, "image_mode": image_mode},
        "backend": context.backend.artifact_record(),
        "datasets": list(entries),
        "test_identities": [entry["test"] for entry in entries],
        "checkpoint_hashes": list(checkpoint_hashes),
        "origin_counts": {"imported_exp450_mps": imported, "exp451_native_cpu_pool_fallback": native},
    }
    return MappingProxyType({**bare, "manifest_sha256": sha256(canonical_json_bytes(bare)).hexdigest()})


def verify_exp451_completion_antigravity(
    root: Path, *, context: Exp451RunContext, catalog: Mapping[str, object],
    migration_manifest: Exp451MigrationManifest,
) -> Exp451VerificationAntigravity:
    """Verify all 4,468 checkpoint envelopes before evaluation may see TEST labels."""
    if (
        Path(root) != context.run_root or not isinstance(migration_manifest, Exp451MigrationManifest)
        or migration_manifest.source_context_sha256 != EXP450_SOURCE_CONTEXT_SHA256
        or migration_manifest.target_context_sha256 != context.sha256
        or migration_manifest.counts_by_condition != EXP451_MIGRATED_COUNTS_BY_CONDITION
    ):
        raise ValueError("Exp451 completion migration authority differs")
    entries = _catalog_entries_antigravity(catalog, context)
    migration = migration_manifest.entry_map
    imported_counts = {condition: 0 for condition in _CONDITIONS_ANTIGRAVITY}
    native_counts = {condition: 0 for condition in _CONDITIONS_ANTIGRAVITY}
    manifests: dict[str, Mapping[str, object]] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        hashes: list[Mapping[str, str]] = []
        for entry in entries:
            dataset = entry["dataset"]
            coordinate = (dataset, condition)
            migrated = migration.get(coordinate)
            if migrated is not None:
                relative = success_relative_path_antigravity(dataset, condition)
                if relative != migrated.target_relative_path:
                    raise ValueError("Exp451 imported checkpoint path differs")
                path = Path(root) / relative
                value, raw = _canonical_object_antigravity(path)
                _check_imported_envelope_antigravity(
                    value, entry=entry, condition=condition, context=context,
                    migration_entry=migrated,
                )
                if sha256(raw).hexdigest() != migrated.target_envelope_sha256:
                    raise ValueError("Exp451 imported checkpoint hash differs")
                imported_counts[condition] += 1
            else:
                if condition not in {"C", "D"}:
                    raise ValueError("Exp451 non-native condition is missing an import")
                path = Path(root) / _native_relative_path_antigravity(dataset, condition)
                value, raw = _canonical_object_antigravity(path)
                _check_native_envelope_antigravity(value, entry=entry, condition=condition, context=context)
                native_counts[condition] += 1
            hashes.append(MappingProxyType({"dataset": dataset, "sha256": sha256(raw).hexdigest()}))
        if imported_counts[condition] + native_counts[condition] != EXP451_FINAL_COUNTS_BY_CONDITION[condition]:
            raise ValueError("Exp451 condition completion count differs")
        manifests[condition] = _condition_manifest_antigravity(
            condition=condition, entries=entries, context=context, checkpoint_hashes=hashes,
            imported=imported_counts[condition], native=native_counts[condition],
        )
    total = sum(imported_counts.values()) + sum(native_counts.values())
    if total != EXP451_TOTAL_COORDINATES:
        raise ValueError("Exp451 completion count differs")
    return Exp451VerificationAntigravity(
        total_successes=total,
        imported_by_condition=MappingProxyType(imported_counts),
        native_by_condition=MappingProxyType(native_counts),
        condition_manifests=MappingProxyType(manifests),
    )


def provenance_summary_antigravity(verification: Exp451VerificationAntigravity) -> dict[str, object]:
    """Return origin-separated counts for reports without reopening any checkpoints."""
    if not isinstance(verification, Exp451VerificationAntigravity):
        raise ValueError("Exp451 verification authority is invalid")
    imported = dict(verification.imported_by_condition)
    native = dict(verification.native_by_condition)
    if set(imported) != set(_CONDITIONS_ANTIGRAVITY) or set(native) != set(_CONDITIONS_ANTIGRAVITY):
        raise ValueError("Exp451 origin inventory differs")
    if any(type(value) is not int or value < 0 for value in (*imported.values(), *native.values())):
        raise ValueError("Exp451 origin count is invalid")
    return {
        "total_successes": verification.total_successes,
        "imported_by_condition": imported,
        "native_by_condition": native,
        "imported_total": sum(imported.values()),
        "native_total": sum(native.values()),
    }


def _write_create_only_antigravity(path: Path, payload: bytes) -> None:
    """Persist one manifest without permitting a later conflicting replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Exp451 condition manifest already exists with different bytes") from None
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp451 condition manifest write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exp451_condition_manifests_antigravity(
    root: Path, verification: Exp451VerificationAntigravity,
) -> Mapping[str, Path]:
    """Publish four already-verified condition manifests as create-only evidence leaves."""
    summary = provenance_summary_antigravity(verification)
    if (
        summary["total_successes"] != EXP451_TOTAL_COORDINATES
        or tuple(verification.condition_manifests) != _CONDITIONS_ANTIGRAVITY
    ):
        raise ValueError("Exp451 cannot publish incomplete condition manifests")
    root = Path(root)
    if not root.is_absolute():
        raise ValueError("Exp451 condition manifest root must be absolute")
    results: dict[str, Path] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        manifest = verification.condition_manifests[condition]
        if not isinstance(manifest, Mapping):
            raise ValueError("Exp451 condition manifest is invalid")
        payload = canonical_json_bytes(dict(manifest))
        path = root / "condition_manifests_antigravity" / f"{condition}_condition_manifest_antigravity.json"
        _write_create_only_antigravity(path, payload)
        results[condition] = path
    return MappingProxyType(results)


def build_exp451_score_manifests_antigravity(
    root: Path, *, context: Exp451RunContext, catalog: Mapping[str, object],
    migration_manifest: Exp451MigrationManifest, raw_datasets: Mapping[str, object],
) -> tuple[Exp451VerificationAntigravity, Mapping[str, Mapping[str, object]]]:
    """Compose label-free continuous scores using the unchanged Exp450 closure.

    The function is intentionally separate from ``verify_exp451_completion``:
    checkpoint verification must finish before values-only global composition,
    and neither operation obtains TEST labels.
    """
    verification = verify_exp451_completion_antigravity(
        root, context=context, catalog=catalog, migration_manifest=migration_manifest,
    )
    entries = _catalog_entries_antigravity(catalog, context)
    if tuple(raw_datasets) != tuple(entry["dataset"] for entry in entries):
        raise ValueError("Exp451 raw dataset inventory differs")
    migration = migration_manifest.entry_map
    results: dict[tuple[str, str], Exp450DatasetConditionResult] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        for entry in entries:
            dataset = entry["dataset"]
            if (dataset, condition) in migration:
                path = Path(root) / success_relative_path_antigravity(dataset, condition)
            else:
                path = Path(root) / _native_relative_path_antigravity(dataset, condition)
            value, _raw = _canonical_object_antigravity(path)
            payload = value["payload"]
            train, test = entry["train"], entry["test"]
            try:
                sources = _sources_from_payload(payload["sources"], train["count"], test["count"])
                route = _route_from_payload(payload["route"], test["count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Exp451 checkpoint typed payload differs") from error
            results[(dataset, condition)] = Exp450DatasetConditionResult(dataset, condition, sources, route)
    backend = context.backend
    frozen_backend = Exp450BackendDecision(
        backend.tensor_backend, backend.torch_version, backend.mps_available,
        backend.mlx_available, backend.logical_cpu_count, backend.cpu_workers,
        backend.inner_numerical_threads,
    )
    scores = compose_all_conditions_antigravity(results, raw_datasets, frozen_backend)
    score_manifests: dict[str, Mapping[str, object]] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        source = verification.condition_manifests[condition]
        policy = next(
            item for item in scores[condition].policies.policies
            if item.name == "hard_plus_stable"
        )
        rows = {
            dataset: [float(value) for value in policy.scores[dataset]]
            for dataset in raw_datasets
        }
        bare = dict(source)
        bare.pop("manifest_sha256", None)
        bare["score_rows"] = rows
        score_manifests[condition] = MappingProxyType({
            **bare, "manifest_sha256": sha256(canonical_json_bytes(bare)).hexdigest(),
        })
    return verification, MappingProxyType(score_manifests)


__all__ = [
    "Exp451VerificationAntigravity", "provenance_summary_antigravity",
    "verify_exp451_completion_antigravity", "write_exp451_condition_manifests_antigravity",
    "build_exp451_score_manifests_antigravity",
]
