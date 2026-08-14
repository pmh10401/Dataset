"""Exp449-only create-only proposal shards and outer-manifest authority."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time

import numpy as np

from paper.equipment_anomaly_detection.artifacts_paper import (
    _typed_payload,
    _validate_shard,
    canonical_json_bytes,
)

from .config_antigravity import FROZEN_EXP449_CONFIG, FROZEN_EXP449_POLICY_KEYS
from .contracts_antigravity import Exp449Context, Exp449ProposalManifest, Exp449ShardDescriptor


_ARTIFACT_DIRECTORY = "stage_artifacts_antigravity"
_FREEZE_DIRECTORY = "proposal_freeze_antigravity"
_SHARDS_DIRECTORY = "shards_antigravity"
_MANIFEST_NAME = "proposal_manifest_antigravity.json"
_SHARD_SCHEMA = "exp449-proposal-shard-antigravity.v1"
_MANIFEST_SCHEMA = "exp449-proposal-manifest-antigravity.v1"


def canonical_float64_bytes_antigravity(values: object) -> bytes:
    """Return the platform-independent bytes used by every Candidate C digest."""
    array = np.asarray(values, dtype=np.dtype("<f8"))
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Exp449 score vector differs")
    return np.ascontiguousarray(array).tobytes(order="C")


def _plain_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_antigravity(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_antigravity(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_antigravity(data: bytes) -> str:
    return sha256(data).hexdigest()


def _lstat_antigravity(path: Path):
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _safe_root_antigravity(root: Path) -> Path:
    raw = Path(root)
    if not raw.is_absolute() or ".." in raw.parts or "exp448" in raw.as_posix().casefold():
        raise ValueError("Exp449 root authority differs")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        if part in {"", ".", ".."}:
            raise ValueError("Exp449 root authority differs")
        current /= part
        details = _lstat_antigravity(current)
        if details is not None and stat.S_ISLNK(details.st_mode):
            raise ValueError("Exp449 symlinked root component is forbidden")
    return raw


def _safe_child_antigravity(root: Path, *parts: str) -> Path:
    target = _safe_root_antigravity(root)
    for part in parts:
        if type(part) is not str or not part or "/" in part or "\\" in part or part in {".", ".."} or "antigravity" not in part:
            raise ValueError("Exp449 runtime artifact name differs")
        target /= part
        details = _lstat_antigravity(target)
        if details is not None and stat.S_ISLNK(details.st_mode):
            raise ValueError("Exp449 symlinked artifact component is forbidden")
    return target


def _mkdirs_antigravity(path: Path) -> None:
    pending: list[Path] = []
    current = path
    while _lstat_antigravity(current) is None:
        pending.append(current)
        current = current.parent
    details = _lstat_antigravity(current)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp449 artifact parent is unsafe")
    for directory in reversed(pending):
        os.mkdir(directory)


def _fsync_directory_antigravity(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only_antigravity(path: Path, data: bytes, *, reuse_identical: bool = False) -> None:
    """Atomically publish bytes once, permitting only byte-identical resume reuse."""
    destination = Path(path)
    existing = _lstat_antigravity(destination)
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise ValueError("Exp449 create-only destination is unsafe")
        if not reuse_identical:
            raise FileExistsError(f"Exp449 create-only artifact already exists: {destination}")
        if destination.read_bytes() != data:
            raise ValueError("Exp449 existing create-only artifact differs")
        return
    _mkdirs_antigravity(destination.parent)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp_antigravity")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("Exp449 create-only write returned no bytes")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        if not reuse_identical:
            raise FileExistsError(f"Exp449 create-only artifact already exists: {destination}") from error
        existing = _lstat_antigravity(destination)
        if existing is None or stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode) or destination.read_bytes() != data:
            raise ValueError("Exp449 existing create-only artifact differs") from error
    else:
        _fsync_directory_antigravity(destination.parent)
        temporary.unlink(missing_ok=True)
        _fsync_directory_antigravity(destination.parent)


def _strict_json_antigravity(path: Path) -> object:
    details = _lstat_antigravity(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp449 artifact is missing or symlinked")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=lambda pairs: _unique_object_antigravity(pairs))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Exp449 artifact JSON differs") from error
    if canonical_json_bytes(value) != raw:
        raise ValueError("Exp449 artifact is noncanonical")
    return value


def _unique_object_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("Exp449 duplicate JSON key")
        value[key] = child
    return value


def validate_exp449_catalog_antigravity(catalog: object, *, mode: str) -> tuple[str, ...]:
    """Check the exact published inventory without loading a database or any labels."""
    if mode not in {"smoke", "official"} or not isinstance(catalog, Mapping):
        raise ValueError("Exp449 catalog differs")
    if type(catalog.get("catalog_sha256")) is not str or len(catalog["catalog_sha256"]) != 64:
        raise ValueError("Exp449 catalog differs")
    entries = catalog.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Exp449 catalog differs")
    names = tuple(entry.get("dataset") if isinstance(entry, Mapping) else None for entry in entries)
    if any(type(name) is not str or not name or "/" in name or "\\" in name or name in {".", ".."} for name in names) or names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("Exp449 catalog differs")
    if mode == "smoke":
        if catalog.get("mode") != "smoke" or names != FROZEN_EXP449_CONFIG.smoke_datasets:
            raise ValueError("Exp449 smoke catalog differs")
    else:
        if catalog.get("mode") != "full" or len(names) != FROZEN_EXP449_CONFIG.dataset_count:
            raise ValueError("Exp449 full catalog differs")
        counts = [entry.get("test", {}).get("count") if isinstance(entry, Mapping) and isinstance(entry.get("test"), Mapping) else None for entry in entries]
        if any(type(count) is not int or count < 0 for count in counts) or sum(counts) != FROZEN_EXP449_CONFIG.test_row_count:
            raise ValueError("Exp449 full catalog differs")
    return names


def _context_payload_antigravity(context: Exp449Context) -> dict[str, object]:
    if not isinstance(context, Exp449Context) or context.experiment_id != "Exp449" or context.historical_parity_status != "not_used":
        raise ValueError("Exp449 context differs")
    return {
        "experiment_id": context.experiment_id,
        "mode": context.mode,
        "run_root": str(context.run_root),
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "code_closure": _plain_antigravity(context.code_closure),
        "lineage_sha256": context.lineage_sha256,
        "codemap_sha256": context.codemap_sha256,
        "interpreter": context.interpreter,
        "environment": _plain_antigravity(context.environment),
        "backend_record": _plain_antigravity(context.backend_record),
        "seeds_and_constants": _plain_antigravity(context.seeds_and_constants),
        "historical_parity_status": context.historical_parity_status,
    }


def context_sha256_antigravity(context: Exp449Context) -> str:
    return _sha256_antigravity(canonical_json_bytes(_context_payload_antigravity(context)))


def _root_for_context_antigravity(root: Path, context: Exp449Context) -> Path:
    checked = _safe_root_antigravity(root)
    if "antigravity" not in checked.name or Path(context.run_root).resolve(strict=False) != checked.resolve(strict=False):
        raise ValueError("Exp449 root authority differs")
    return checked


def _entry_for_dataset_antigravity(catalog: Mapping[str, object], dataset: str) -> dict[str, object]:
    entries = [entry for entry in catalog["datasets"] if isinstance(entry, dict) and entry.get("dataset") == dataset]
    if len(entries) != 1:
        raise ValueError("Exp449 catalog shard identity differs")
    return entries[0]


def _shard_relative_path_antigravity(dataset: str) -> str:
    token = sha256(dataset.encode("utf-8")).hexdigest()
    return f"{_SHARDS_DIRECTORY}/{token}_antigravity/{dataset}_proposal_shard_antigravity.json"


def _policy_digests_antigravity(scientific_payload: Mapping[str, object], dataset: str) -> tuple[dict[str, str], str]:
    policies = scientific_payload.get("policies")
    if not isinstance(policies, list) or [row.get("policy_id") if isinstance(row, Mapping) else None for row in policies] != list(FROZEN_EXP449_POLICY_KEYS):
        raise ValueError("Exp449 policy closure differs")
    coordinates: dict[str, str] = {}
    candidate_scores: bytes | None = None
    for row in policies:
        assert isinstance(row, Mapping)
        alerts = row.get("alerts")
        if not isinstance(alerts, list) or any(type(index) is not int for index in alerts):
            raise ValueError("Exp449 policy coordinates differ")
        coordinates[str(row["policy_id"])] = _sha256_antigravity(canonical_json_bytes([[dataset, index] for index in alerts]))
        if row["policy_id"] == FROZEN_EXP449_CONFIG.policy_id:
            candidate_scores = canonical_float64_bytes_antigravity(row.get("scores"))
    if candidate_scores is None:
        raise ValueError("Exp449 Candidate C differs")
    return coordinates, _sha256_antigravity(candidate_scores)


def _envelope_antigravity(scientific_payload: dict[str, object], context: Exp449Context, catalog: Mapping[str, object]) -> dict[str, object]:
    """Keep shard bytes scientific; the outer manifest exclusively binds run-root context."""
    return {
        "schema": _SHARD_SCHEMA,
        "catalog_sha256": context.catalog_sha256,
        "backend_record": _plain_antigravity(context.backend_record),
        "scientific_payload": scientific_payload,
    }


def validate_exp449_proposal_shard_antigravity(
    value: object, *, entry: dict[str, object], context: Exp449Context, catalog: Mapping[str, object]
) -> tuple[dict[str, str], str]:
    """Validate the Exp449 envelope plus the frozen paper scientific payload."""
    if not isinstance(value, dict) or set(value) != {"schema", "catalog_sha256", "backend_record", "scientific_payload"}:
        raise ValueError("Exp449 proposal shard schema differs")
    if value["schema"] != _SHARD_SCHEMA or value["catalog_sha256"] != context.catalog_sha256 or context.catalog_sha256 != catalog.get("catalog_sha256"):
        raise ValueError("Exp449 proposal shard authority differs")
    if value["backend_record"] != _plain_antigravity(context.backend_record) or not isinstance(value["scientific_payload"], dict):
        raise ValueError("Exp449 proposal shard backend differs")
    scientific = _validate_shard(value["scientific_payload"], entry)
    if scientific["backend"] != value["backend_record"]:
        raise ValueError("Exp449 proposal shard backend differs")
    return _policy_digests_antigravity(scientific, entry["dataset"])


def _manifest_payload_antigravity(manifest: Exp449ProposalManifest) -> dict[str, object]:
    return {
        "schema": manifest.schema,
        "context_sha256": manifest.context_sha256,
        "catalog_sha256": manifest.catalog_sha256,
        "backend_record": _plain_antigravity(manifest.backend_record),
        "entries": [
            {
                "dataset": entry.dataset,
                "relative_path": entry.relative_path,
                "size": entry.size,
                "sha256": entry.sha256,
                "schema": entry.schema,
                "policy_coordinate_sha256": _plain_antigravity(entry.policy_coordinate_sha256),
                "candidate_c_score_sha256": entry.candidate_c_score_sha256,
            }
            for entry in manifest.entries
        ],
        "historical_parity_status": manifest.historical_parity_status,
    }


def _manifest_from_payload_antigravity(value: object) -> Exp449ProposalManifest:
    if not isinstance(value, dict) or set(value) != {"schema", "context_sha256", "catalog_sha256", "backend_record", "entries", "historical_parity_status"} or not isinstance(value.get("entries"), list):
        raise ValueError("Exp449 proposal manifest schema differs")
    return Exp449ProposalManifest(
        schema=value["schema"], context_sha256=value["context_sha256"], catalog_sha256=value["catalog_sha256"],
        backend_record=value["backend_record"], entries=tuple(Exp449ShardDescriptor(**entry) for entry in value["entries"]),
        historical_parity_status=value["historical_parity_status"],
    )


def _freeze_root_antigravity(root: Path) -> Path:
    return _safe_child_antigravity(root, _ARTIFACT_DIRECTORY, _FREEZE_DIRECTORY)


def freeze_exp449_proposal_antigravity(score: object, *, root: Path, context: Exp449Context, catalog: dict[str, object]) -> Exp449ProposalManifest:
    """Create only Exp449 typed shards and the one independently verifiable manifest."""
    root = _root_for_context_antigravity(root, context)
    names = validate_exp449_catalog_antigravity(catalog, mode=context.mode)
    if context.catalog_sha256 != catalog.get("catalog_sha256"):
        raise ValueError("Exp449 catalog authority differs")
    required = ("raw", "sources", "routes", "low_frequency", "stability", "policies", "backend")
    if any(not hasattr(score, name) for name in required) or score.backend.artifact_record() != _plain_antigravity(context.backend_record):
        raise ValueError("Exp449 typed score authority differs")
    entries: list[Exp449ShardDescriptor] = []
    for dataset in names:
        entry = _entry_for_dataset_antigravity(catalog, dataset)
        scientific = _typed_payload(score.raw[dataset], score.sources[dataset], score.routes[dataset], score.low_frequency, score.stability, score.policies, score.backend, entry)
        envelope = _envelope_antigravity(scientific, context, catalog)
        coordinates, candidate_digest = validate_exp449_proposal_shard_antigravity(envelope, entry=entry, context=context, catalog=catalog)
        relative = _shard_relative_path_antigravity(dataset)
        path = _freeze_root_antigravity(root) / relative
        _write_create_only_antigravity(path, canonical_json_bytes(envelope), reuse_identical=True)
        entries.append(Exp449ShardDescriptor(dataset, relative, path.stat().st_size, _sha256_antigravity(path.read_bytes()), _SHARD_SCHEMA, coordinates, candidate_digest))
    manifest = Exp449ProposalManifest(_MANIFEST_SCHEMA, context_sha256_antigravity(context), context.catalog_sha256, context.backend_record, tuple(entries), "not_used")
    payload = _manifest_payload_antigravity(manifest)
    _write_create_only_antigravity(_freeze_root_antigravity(root) / _MANIFEST_NAME, canonical_json_bytes(payload), reuse_identical=True)
    verify_exp449_freeze_inventory_antigravity(root, context=context, catalog=catalog, manifest=manifest)
    return manifest


def verify_exp449_freeze_inventory_antigravity(
    root: Path, *, context: Exp449Context, catalog: dict[str, object], manifest: Exp449ProposalManifest | None = None
) -> Exp449ProposalManifest:
    """Verify exactly the declared antigravity files without any scorer dependency."""
    root = _root_for_context_antigravity(root, context)
    names = validate_exp449_catalog_antigravity(catalog, mode=context.mode)
    freeze_root = _freeze_root_antigravity(root)
    if manifest is None:
        manifest = _manifest_from_payload_antigravity(_strict_json_antigravity(freeze_root / _MANIFEST_NAME))
    if manifest.context_sha256 != context_sha256_antigravity(context) or manifest.catalog_sha256 != context.catalog_sha256 or manifest.catalog_sha256 != catalog.get("catalog_sha256") or _plain_antigravity(manifest.backend_record) != _plain_antigravity(context.backend_record):
        raise ValueError("Exp449 proposal manifest authority differs")
    if tuple(entry.dataset for entry in manifest.entries) != names:
        raise ValueError("Exp449 proposal manifest inventory differs")
    declared = {_MANIFEST_NAME}
    for descriptor in manifest.entries:
        relative = PurePosixPath(descriptor.relative_path)
        if relative.is_absolute() or ".." in relative.parts or descriptor.relative_path != _shard_relative_path_antigravity(descriptor.dataset):
            raise ValueError("Exp449 proposal shard path differs")
        declared.add(descriptor.relative_path)
        path = freeze_root / descriptor.relative_path
        details = _lstat_antigravity(path)
        if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise ValueError("Exp449 proposal shard is missing or symlinked")
        if details.st_size != descriptor.size or _sha256_antigravity(path.read_bytes()) != descriptor.sha256:
            raise ValueError("Exp449 proposal shard hash differs")
        coordinates, candidate_digest = validate_exp449_proposal_shard_antigravity(_strict_json_antigravity(path), entry=_entry_for_dataset_antigravity(catalog, descriptor.dataset), context=context, catalog=catalog)
        if dict(descriptor.policy_coordinate_sha256) != coordinates or descriptor.candidate_c_score_sha256 != candidate_digest:
            raise ValueError("Exp449 proposal shard digest differs")
    found = {item.relative_to(freeze_root).as_posix() for item in freeze_root.rglob("*") if item.is_file() or item.is_symlink()}
    if found != declared:
        raise ValueError("Exp449 proposal freeze inventory differs")
    for item in freeze_root.rglob("*"):
        if "antigravity" not in item.name:
            raise ValueError("Exp449 runtime artifact name differs")
    return manifest


__all__ = [
    "canonical_float64_bytes_antigravity",
    "context_sha256_antigravity",
    "freeze_exp449_proposal_antigravity",
    "validate_exp449_catalog_antigravity",
    "validate_exp449_proposal_shard_antigravity",
    "verify_exp449_freeze_inventory_antigravity",
]
