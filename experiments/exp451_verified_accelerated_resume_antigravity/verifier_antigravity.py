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
from pathlib import Path
import stat
from types import MappingProxyType
from weakref import WeakSet

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
    _checked_target_path_antigravity,
    _publish_create_only_antigravity,
    success_relative_path_antigravity,
)
from .config_antigravity import (
    EXP450_SOURCE_CONTEXT_SHA256,
    EXP451_FINAL_COUNTS_BY_CONDITION,
    EXP451_MIGRATED_COUNTS_BY_CONDITION,
    EXP451_TOTAL_COORDINATES,
    FROZEN_EXP451_CONFIG,
    OFFICIAL_ROOT,
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
_SCORE_MANIFEST_SCHEMA_ANTIGRAVITY = "exp451-verified-score-manifest-antigravity.v1"

def _identity_registry_antigravity():
    """Return closure-private issue/validate operations; the set is not a module value."""
    issued: WeakSet[object] = WeakSet()
    def issue(value: object) -> object:
        issued.add(value)
        return value
    def validates(value: object) -> bool:
        return value in issued
    return issue, validates


_issue_verified_authority_antigravity, _is_verified_authority_antigravity = _identity_registry_antigravity()


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp451VerificationAntigravity:
    """The only completion authority accepted by Exp451 evaluation/reporting."""

    total_successes: int
    imported_by_condition: Mapping[str, int]
    native_by_condition: Mapping[str, int]
    condition_manifests: Mapping[str, Mapping[str, object]]
    context_sha256: str = ""
    catalog_sha256: str = ""
    database_sha256: str = ""
    inventory_sha256: str = ""


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
        or context.database_sha256 != FROZEN_EXP451_CONFIG.database_sha256
        or not isinstance(catalog.get("database"), Mapping)
        or catalog["database"].get("sha256") != FROZEN_EXP451_CONFIG.database_sha256
        or catalog.get("mode") != "full"
    ):
        raise ValueError("Exp451 catalog differs")
    datasets = tuple(entry.get("dataset") if isinstance(entry, dict) else None for entry in entries)
    if (
        any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(set(datasets)) != len(datasets) or tuple(sorted(datasets)) != datasets
    ):
        raise ValueError("Exp451 catalog dataset inventory differs")
    if sum(entry.get("test", {}).get("count", -1) for entry in entries if isinstance(entry, Mapping)) != 195900:
        raise ValueError("Exp451 catalog TEST total differs")
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


def _inventory_sha256_antigravity(manifests: Mapping[str, Mapping[str, object]]) -> str:
    return sha256(canonical_json_bytes({
        condition: dict(manifests[condition]) for condition in manifests
    })).hexdigest()


def _validate_verification_authority_antigravity(verification: object) -> Exp451VerificationAntigravity:
    """Validate the opaque issued capability and all frozen manifest contents."""
    if (
        not isinstance(verification, Exp451VerificationAntigravity)
        or not _is_verified_authority_antigravity(verification)
    ):
        raise ValueError("Exp451 verification authority is invalid")
    if verification.total_successes != EXP451_TOTAL_COORDINATES:
        raise ValueError("Exp451 verification authority requires exactly 4468 successes")
    imported, native = dict(verification.imported_by_condition), dict(verification.native_by_condition)
    expected_imported = {"A": 1117, "B": 1117, "C": 1000, "D": 1000}
    expected_native = {"A": 0, "B": 0, "C": 117, "D": 117}
    if imported != expected_imported or native != expected_native:
        raise ValueError("Exp451 verification authority origin distribution differs")
    manifests = verification.condition_manifests
    if tuple(manifests) != _CONDITIONS_ANTIGRAVITY:
        raise ValueError("Exp451 verification authority requires four frozen condition manifests")
    if (
        verification.database_sha256 != FROZEN_EXP451_CONFIG.database_sha256
        or not _sha256_antigravity(verification.context_sha256)
        or not _sha256_antigravity(verification.catalog_sha256)
        or not _sha256_antigravity(verification.inventory_sha256)
    ):
        raise ValueError("Exp451 verification authority database/context differs")
    expected_datasets: tuple[str, ...] | None = None
    expected_tests: list[object] | None = None
    for condition in _CONDITIONS_ANTIGRAVITY:
        manifest = manifests[condition]
        required = {
            "schema", "database_sha256", "context_sha256", "catalog_sha256", "condition",
            "backend", "datasets", "test_identities", "checkpoint_hashes", "origin_counts",
            "manifest_sha256",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != required:
            raise ValueError("Exp451 verification authority manifest schema differs")
        bare = {key: manifest[key] for key in manifest if key != "manifest_sha256"}
        if (
            manifest["schema"] != _CONDITION_MANIFEST_SCHEMA_ANTIGRAVITY
            or manifest["manifest_sha256"] != sha256(canonical_json_bytes(bare)).hexdigest()
            or manifest["database_sha256"] != verification.database_sha256
            or manifest["context_sha256"] != verification.context_sha256
            or manifest["catalog_sha256"] != verification.catalog_sha256
            or manifest["condition"] != {
                "condition_id": condition,
                "length_mode": _CONDITION_MODES_ANTIGRAVITY[condition][0],
                "image_mode": _CONDITION_MODES_ANTIGRAVITY[condition][1],
            }
            or manifest["origin_counts"] != {
                _IMPORTED_ORIGIN_ANTIGRAVITY: imported[condition],
                _NATIVE_ORIGIN_ANTIGRAVITY: native[condition],
            }
            or not isinstance(manifest["datasets"], list) or len(manifest["datasets"]) != 1117
            or not isinstance(manifest["test_identities"], list) or len(manifest["test_identities"]) != 1117
            or not isinstance(manifest["checkpoint_hashes"], list) or len(manifest["checkpoint_hashes"]) != 1117
        ):
            raise ValueError("Exp451 verification authority manifest differs")
        datasets = tuple(item.get("dataset") if isinstance(item, Mapping) else None for item in manifest["datasets"])
        hashes = manifest["checkpoint_hashes"]
        if (
            any(type(dataset) is not str or not dataset for dataset in datasets)
            or len(set(datasets)) != 1117
            or [item.get("dataset") if isinstance(item, Mapping) else None for item in hashes] != list(datasets)
            or any(not _sha256_antigravity(item.get("sha256")) for item in hashes if isinstance(item, Mapping))
        ):
            raise ValueError("Exp451 verification authority checkpoint inventory differs")
        tests = manifest["test_identities"]
        if [item.get("dataset") if isinstance(item, Mapping) else None for item in tests] != list(datasets):
            raise ValueError("Exp451 verification authority TEST identity differs")
        if expected_datasets is None:
            expected_datasets, expected_tests = datasets, tests
        elif datasets != expected_datasets or tests != expected_tests:
            raise ValueError("Exp451 verification authority cross-condition identity differs")
    assert expected_tests is not None
    if sum(item.get("count", -1) for item in expected_tests if isinstance(item, Mapping)) != 195900:
        raise ValueError("Exp451 verification authority TEST total differs")
    if _inventory_sha256_antigravity(manifests) != verification.inventory_sha256:
        raise ValueError("Exp451 verification authority inventory hash differs")
    return verification


def verify_exp451_completion_antigravity(
    root: Path, *, context: Exp451RunContext, catalog: Mapping[str, object],
    migration_manifest: Exp451MigrationManifest,
) -> Exp451VerificationAntigravity:
    """Verify all 4,468 checkpoint envelopes before evaluation may see TEST labels."""
    if (
        Path(root) != OFFICIAL_ROOT or context.run_root != OFFICIAL_ROOT
        or not isinstance(migration_manifest, Exp451MigrationManifest)
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
    authority = Exp451VerificationAntigravity(
        total, MappingProxyType(imported_counts), MappingProxyType(native_counts),
        MappingProxyType(manifests), context.sha256, context.catalog_sha256,
        context.database_sha256, _inventory_sha256_antigravity(manifests),
    )
    return _issue_verified_authority_antigravity(authority)


def provenance_summary_antigravity(verification: Exp451VerificationAntigravity) -> dict[str, object]:
    """Return origin-separated counts for reports without reopening any checkpoints."""
    verification = _validate_verification_authority_antigravity(verification)
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
    if root != OFFICIAL_ROOT or root.is_symlink():
        raise ValueError("Exp451 condition manifest root differs from the frozen configured root")
    results: dict[str, Path] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        manifest = verification.condition_manifests[condition]
        if not isinstance(manifest, Mapping):
            raise ValueError("Exp451 condition manifest is invalid")
        payload = canonical_json_bytes(dict(manifest))
        relative = f"condition_manifests_antigravity/{condition}_condition_manifest_antigravity.json"
        path = _checked_target_path_antigravity(root, relative, create=True)
        _publish_create_only_antigravity(path, payload)
        results[condition] = path
    return MappingProxyType(results)


def read_persisted_exp451_condition_manifests_antigravity(
    root: Path, verification: Exp451VerificationAntigravity,
) -> Mapping[str, Mapping[str, object]]:
    """Reopen the four create-only leaves and bind their exact bytes to authority."""
    _validate_verification_authority_antigravity(verification)
    root = Path(root)
    if root != OFFICIAL_ROOT or root.is_symlink():
        raise ValueError("Exp451 condition manifest root differs from the frozen configured root")
    opened: dict[str, Mapping[str, object]] = {}
    for condition in _CONDITIONS_ANTIGRAVITY:
        path = _checked_target_path_antigravity(
            root, f"condition_manifests_antigravity/{condition}_condition_manifest_antigravity.json", create=False,
        )
        value, raw = _canonical_object_antigravity(path)
        expected = canonical_json_bytes(dict(verification.condition_manifests[condition]))
        if raw != expected or value != dict(verification.condition_manifests[condition]):
            raise ValueError("Exp451 persisted condition manifest differs")
        opened[condition] = MappingProxyType(value)
    return MappingProxyType(opened)


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
        base = dict(source)
        score_bare = {
            "score_schema": _SCORE_MANIFEST_SCHEMA_ANTIGRAVITY,
            "base_manifest_sha256": base["manifest_sha256"],
            "score_rows": rows,
            "score_rows_sha256": sha256(canonical_json_bytes(rows)).hexdigest(),
        }
        score_manifests[condition] = MappingProxyType({
            **base, **score_bare,
            "score_manifest_sha256": sha256(canonical_json_bytes(score_bare)).hexdigest(),
        })
    return verification, MappingProxyType(score_manifests)


__all__ = [
    "Exp451VerificationAntigravity", "provenance_summary_antigravity",
    "verify_exp451_completion_antigravity", "write_exp451_condition_manifests_antigravity",
    "read_persisted_exp451_condition_manifests_antigravity", "build_exp451_score_manifests_antigravity",
]
