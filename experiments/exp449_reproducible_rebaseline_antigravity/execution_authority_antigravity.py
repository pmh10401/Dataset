"""Pure canonical contracts for a separately reviewed Exp449 Smoke A grant.

This module deliberately has no operations, scorer, readiness, Git, or file I/O
dependency.  Later authority consumers can therefore verify retained bytes without
giving this boundary any execution capability of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import stat

from .config_antigravity import (
    EXP449_DATASET_WORKERS_ANTIGRAVITY,
    EXP449_LOGICAL_CPU_COUNT_ANTIGRAVITY,
    FROZEN_EXP449_CONFIG,
    SMOKE_A_DIGEST_PATH_ANTIGRAVITY,
    SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
    SMOKE_A_GRANT_PATH_ANTIGRAVITY,
)
from .contracts_antigravity import freeze_canonical_json_mapping_antigravity


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_NONCE_ANTIGRAVITY = re.compile(r"^[0-9a-f]{32}$")
_CREATED_AT_ANTIGRAVITY = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_GRANT_SCHEMA_ANTIGRAVITY = "exp449-smoke-a-execution-grant-antigravity.v1"
_DIGEST_SCHEMA_ANTIGRAVITY = "exp449-smoke-a-execution-grant-digest-antigravity.v1"
_GRANT_FIELDS_ANTIGRAVITY = frozenset({
    "schema", "experiment_id", "target", "policy_id", "implementation_commit", "scientific_root", "operations_root", "claim_path",
    "nonce", "maximum_waves", "label_access", "logical_cpu_count", "dataset_workers", "database_path",
    "database_size", "database_sha256", "smoke_catalog_sha256", "smoke_datasets", "context_sha256", "scientific_code_closure",
    "authority_code_closure", "lineage_lock_sha256", "codemap_lock_sha256", "acceleration_assessment_sha256",
    "canonical_backend", "package_inventory", "seeds_and_constants", "acceleration", "design_facts", "grant_created_at",
})
_DIGEST_FIELDS_ANTIGRAVITY = frozenset({"schema", "implementation_commit", "grant_path", "grant_size", "grant_sha256"})
_APPROVED_AUTHORITY_FILENAMES_ANTIGRAVITY = frozenset({
    SMOKE_A_GRANT_PATH_ANTIGRAVITY.name,
    SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name,
})
_DESIGN_FACTS_ANTIGRAVITY = {
    "dataset_count": FROZEN_EXP449_CONFIG.dataset_count,
    "test_row_count": FROZEN_EXP449_CONFIG.test_row_count,
    "preprocessing_contracts": FROZEN_EXP449_CONFIG.preprocessing_contracts,
    "cpu_repeat_tolerance": FROZEN_EXP449_CONFIG.cpu_repeat_tolerance,
    "accelerated_tolerance": FROZEN_EXP449_CONFIG.accelerated_tolerance,
    "minimum_accelerated_speedup": FROZEN_EXP449_CONFIG.minimum_accelerated_speedup,
}


def _sha256_antigravity(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _immutable_sha_map_antigravity(value: object, name: str) -> Mapping[str, str]:
    frozen = freeze_canonical_json_mapping_antigravity(value, name)
    if not frozen:
        raise ValueError(f"{name} must be nonempty")
    checked = {key: _sha256_antigravity(item, f"{name}.{key}") for key, item in frozen.items()}
    return freeze_canonical_json_mapping_antigravity(checked, name)  # type: ignore[return-value]


def _immutable_json_map_antigravity(value: object, name: str) -> Mapping[str, object]:
    frozen = freeze_canonical_json_mapping_antigravity(value, name)
    if not frozen:
        raise ValueError(f"{name} must be nonempty")
    return frozen


def _canonical_json_bytes_antigravity(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Smoke A authority must be canonical finite JSON") from error


def _plain_json_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json_antigravity(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_antigravity(nested) for nested in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class Exp449SmokeAGrantAntigravity:
    """Complete, data-only binding for one yet-unconsumed Smoke A run."""

    schema: str
    experiment_id: str
    target: str
    policy_id: str
    implementation_commit: str
    scientific_root: Path
    operations_root: Path
    claim_path: Path
    nonce: str
    maximum_waves: int
    label_access: bool
    logical_cpu_count: int
    dataset_workers: int
    database_path: str
    database_size: int
    database_sha256: str
    smoke_catalog_sha256: str
    smoke_datasets: tuple[str, ...]
    context_sha256: str
    scientific_code_closure: Mapping[str, str]
    authority_code_closure: Mapping[str, str]
    lineage_lock_sha256: str
    codemap_lock_sha256: str
    acceleration_assessment_sha256: str
    canonical_backend: str
    package_inventory: Mapping[str, object]
    seeds_and_constants: Mapping[str, object]
    acceleration: Mapping[str, object]
    design_facts: Mapping[str, object]
    grant_created_at: str

    def __post_init__(self) -> None:
        if self.schema != _GRANT_SCHEMA_ANTIGRAVITY:
            raise ValueError("Smoke A grant schema is invalid")
        if self.experiment_id != FROZEN_EXP449_CONFIG.experiment_id or self.target != "smoke-a":
            raise ValueError("Smoke A grant identity is invalid")
        if self.policy_id != FROZEN_EXP449_CONFIG.policy_id:
            raise ValueError("Smoke A grant policy is invalid")
        if type(self.implementation_commit) is not str or _COMMIT_ANTIGRAVITY.fullmatch(self.implementation_commit) is None:
            raise ValueError("Smoke A grant implementation commit is invalid")
        if Path(self.scientific_root).resolve(strict=False) != FROZEN_EXP449_CONFIG.smoke_a_scientific_root.resolve(strict=False):
            raise ValueError("Smoke A grant scientific root is invalid")
        if Path(self.operations_root).resolve(strict=False) != FROZEN_EXP449_CONFIG.smoke_a_operations_root.resolve(strict=False):
            raise ValueError("Smoke A grant operations root is invalid")
        if Path(self.claim_path) != SMOKE_A_CLAIM_PATH_ANTIGRAVITY:
            raise ValueError("Smoke A grant claim path is invalid")
        if type(self.nonce) is not str or _NONCE_ANTIGRAVITY.fullmatch(self.nonce) is None:
            raise ValueError("Smoke A grant nonce is invalid")
        if type(self.maximum_waves) is not int or self.maximum_waves != FROZEN_EXP449_CONFIG.maximum_waves:
            raise ValueError("Smoke A grant wave count is invalid")
        if self.label_access is not False:
            raise ValueError("Smoke A grant label access must be false")
        if type(self.logical_cpu_count) is not int or type(self.dataset_workers) is not int or (self.logical_cpu_count, self.dataset_workers) != (
            EXP449_LOGICAL_CPU_COUNT_ANTIGRAVITY,
            EXP449_DATASET_WORKERS_ANTIGRAVITY,
        ):
            raise ValueError("Smoke A grant CPU worker contract is invalid")
        if type(self.database_path) is not str or not Path(self.database_path).is_absolute():
            raise ValueError("Smoke A grant database path is invalid")
        if type(self.database_size) is not int or self.database_size < 0:
            raise ValueError("Smoke A grant database size is invalid")
        datasets = tuple(self.smoke_datasets)
        if datasets != FROZEN_EXP449_CONFIG.smoke_datasets:
            raise ValueError("Smoke A grant datasets must be canonically ordered")
        object.__setattr__(self, "scientific_root", FROZEN_EXP449_CONFIG.smoke_a_scientific_root)
        object.__setattr__(self, "operations_root", FROZEN_EXP449_CONFIG.smoke_a_operations_root)
        object.__setattr__(self, "claim_path", SMOKE_A_CLAIM_PATH_ANTIGRAVITY)
        object.__setattr__(self, "smoke_datasets", datasets)
        for field_name in ("database_sha256", "smoke_catalog_sha256", "context_sha256", "lineage_lock_sha256", "codemap_lock_sha256", "acceleration_assessment_sha256"):
            object.__setattr__(self, field_name, _sha256_antigravity(getattr(self, field_name), field_name))
        if self.canonical_backend != "cpu":
            raise ValueError("Smoke A grant canonical backend is invalid")
        object.__setattr__(self, "scientific_code_closure", _immutable_sha_map_antigravity(self.scientific_code_closure, "scientific_code_closure"))
        object.__setattr__(self, "authority_code_closure", _immutable_sha_map_antigravity(self.authority_code_closure, "authority_code_closure"))
        for field_name in ("package_inventory", "seeds_and_constants", "acceleration"):
            object.__setattr__(self, field_name, _immutable_json_map_antigravity(getattr(self, field_name), field_name))
        design_facts = _immutable_json_map_antigravity(self.design_facts, "design_facts")
        if _plain_json_antigravity(design_facts) != _plain_json_antigravity(_DESIGN_FACTS_ANTIGRAVITY):
            raise ValueError("Smoke A grant frozen design facts are invalid")
        object.__setattr__(self, "design_facts", design_facts)
        if type(self.grant_created_at) is not str or _CREATED_AT_ANTIGRAVITY.fullmatch(self.grant_created_at) is None:
            raise ValueError("Smoke A grant creation timestamp is invalid")
        try:
            datetime.strptime(self.grant_created_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError("Smoke A grant creation timestamp is invalid") from error


@dataclass(frozen=True, slots=True)
class Exp449SmokeAGrantDigestAntigravity:
    """Canonical pointer and digest for a grant committed in a later delta."""

    schema: str
    implementation_commit: str
    grant_path: Path
    grant_size: int
    grant_sha256: str

    def __post_init__(self) -> None:
        if self.schema != _DIGEST_SCHEMA_ANTIGRAVITY:
            raise ValueError("Smoke A grant digest schema is invalid")
        if type(self.implementation_commit) is not str or _COMMIT_ANTIGRAVITY.fullmatch(self.implementation_commit) is None:
            raise ValueError("Smoke A grant digest implementation commit is invalid")
        if Path(self.grant_path) != SMOKE_A_GRANT_PATH_ANTIGRAVITY:
            raise ValueError("Smoke A grant digest path is invalid")
        if type(self.grant_size) is not int or self.grant_size < 1:
            raise ValueError("Smoke A grant digest size is invalid")
        object.__setattr__(self, "grant_path", SMOKE_A_GRANT_PATH_ANTIGRAVITY)
        object.__setattr__(self, "grant_sha256", _sha256_antigravity(self.grant_sha256, "grant_sha256"))


def _grant_payload_antigravity(grant: Exp449SmokeAGrantAntigravity) -> dict[str, object]:
    if not isinstance(grant, Exp449SmokeAGrantAntigravity):
        raise TypeError("Exp449 Smoke A grant is required")
    return {name: _plain_json_antigravity(getattr(grant, name)) for name in _GRANT_FIELDS_ANTIGRAVITY}


def _digest_payload_antigravity(digest: Exp449SmokeAGrantDigestAntigravity) -> dict[str, object]:
    if not isinstance(digest, Exp449SmokeAGrantDigestAntigravity):
        raise TypeError("Exp449 Smoke A grant digest is required")
    return {name: _plain_json_antigravity(getattr(digest, name)) for name in _DIGEST_FIELDS_ANTIGRAVITY}


def canonical_grant_bytes_antigravity(grant: Exp449SmokeAGrantAntigravity) -> bytes:
    """Serialize a previously validated grant with the repository authority convention."""

    return _canonical_json_bytes_antigravity(_grant_payload_antigravity(grant))


def canonical_grant_digest_bytes_antigravity(digest: Exp449SmokeAGrantDigestAntigravity) -> bytes:
    """Serialize a previously validated grant digest with the same convention."""

    return _canonical_json_bytes_antigravity(_digest_payload_antigravity(digest))


def _parse_object_antigravity(data: bytes, fields: frozenset[str], name: str) -> dict[str, object]:
    if type(data) is not bytes:
        raise TypeError(f"{name} bytes are required")
    try:
        decoded = data.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} JSON is invalid") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def verify_authority_directory_structure_antigravity(
    authority_directory: Path = SMOKE_A_GRANT_PATH_ANTIGRAVITY.parent,
) -> tuple[str, ...]:
    """Accept only the two reserved regular JSON authority names, or absence.

    This is structural validation only.  It reads neither authority content nor
    Git state, and it deliberately permits an absent directory before the later
    review creates the two files.
    """

    directory = Path(authority_directory)
    try:
        details = directory.lstat()
    except FileNotFoundError:
        return ()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Smoke A authority directory is unsafe")
    names: list[str] = []
    for entry in directory.iterdir():
        entry_details = entry.lstat()
        if (
            entry.name not in _APPROVED_AUTHORITY_FILENAMES_ANTIGRAVITY
            or stat.S_ISLNK(entry_details.st_mode)
            or not stat.S_ISREG(entry_details.st_mode)
        ):
            raise ValueError("Smoke A authority directory contains an unapproved entry")
        names.append(entry.name)
    return tuple(sorted(names))


def parse_grant_bytes_antigravity(data: bytes) -> Exp449SmokeAGrantAntigravity:
    """Parse only a byte-for-byte canonical Smoke A grant."""

    value = _parse_object_antigravity(data, _GRANT_FIELDS_ANTIGRAVITY, "Smoke A grant")
    try:
        grant = Exp449SmokeAGrantAntigravity(
            schema=value["schema"], experiment_id=value["experiment_id"], target=value["target"], policy_id=value["policy_id"], implementation_commit=value["implementation_commit"],
            scientific_root=Path(value["scientific_root"]), operations_root=Path(value["operations_root"]), claim_path=Path(value["claim_path"]),
            nonce=value["nonce"], maximum_waves=value["maximum_waves"], label_access=value["label_access"],
            logical_cpu_count=value["logical_cpu_count"], dataset_workers=value["dataset_workers"], database_path=value["database_path"],
            database_size=value["database_size"], database_sha256=value["database_sha256"], smoke_catalog_sha256=value["smoke_catalog_sha256"], smoke_datasets=tuple(value["smoke_datasets"]),
            context_sha256=value["context_sha256"], scientific_code_closure=value["scientific_code_closure"],
            authority_code_closure=value["authority_code_closure"], lineage_lock_sha256=value["lineage_lock_sha256"],
            codemap_lock_sha256=value["codemap_lock_sha256"], acceleration_assessment_sha256=value["acceleration_assessment_sha256"], canonical_backend=value["canonical_backend"], package_inventory=value["package_inventory"],
            seeds_and_constants=value["seeds_and_constants"], acceleration=value["acceleration"], design_facts=value["design_facts"], grant_created_at=value["grant_created_at"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Smoke A grant is invalid") from error
    if canonical_grant_bytes_antigravity(grant) != data:
        raise ValueError("Smoke A grant bytes are not canonical")
    return grant


def parse_grant_digest_bytes_antigravity(data: bytes) -> Exp449SmokeAGrantDigestAntigravity:
    """Parse only a byte-for-byte canonical Smoke A grant digest."""

    value = _parse_object_antigravity(data, _DIGEST_FIELDS_ANTIGRAVITY, "Smoke A grant digest")
    try:
        digest = Exp449SmokeAGrantDigestAntigravity(
            schema=value["schema"], implementation_commit=value["implementation_commit"], grant_path=Path(value["grant_path"]),
            grant_size=value["grant_size"], grant_sha256=value["grant_sha256"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Smoke A grant digest is invalid") from error
    if canonical_grant_digest_bytes_antigravity(digest) != data:
        raise ValueError("Smoke A grant digest bytes are not canonical")
    return digest
