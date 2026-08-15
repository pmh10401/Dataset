"""Canonical contracts and scorer-free verification for Exp449 Smoke A authority.

The verifier reads only immutable Git and authority-file evidence.  It cannot
prepare a grant, claim a run, create a root, inspect a database, or score data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from .config_antigravity import (
    EXP449_DATASET_WORKERS_ANTIGRAVITY,
    EXP449_LOGICAL_CPU_COUNT_ANTIGRAVITY,
    FROZEN_EXP449_CONFIG,
    REPOSITORY_ROOT_ANTIGRAVITY,
    SMOKE_A_DIGEST_PATH_ANTIGRAVITY,
    SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
    SMOKE_A_GRANT_PATH_ANTIGRAVITY,
)
from .contracts_antigravity import Exp449Context, freeze_canonical_json_mapping_antigravity
from .artifacts_antigravity import context_sha256_antigravity


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
_SMOKE_A_GRANT_RELATIVE_PATH_ANTIGRAVITY = Path("docs/authorities") / SMOKE_A_GRANT_PATH_ANTIGRAVITY.name
_SMOKE_A_DIGEST_RELATIVE_PATH_ANTIGRAVITY = Path("docs/authorities") / SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name
_LINEAGE_LOCK_RELATIVE_PATH_ANTIGRAVITY = Path("paper/equipment_anomaly_detection/lineage_lock_paper.json")
_CODEMAP_LOCK_RELATIVE_PATH_ANTIGRAVITY = Path("docs/codemap/codemap_exp449_antigravity.lock")


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


@dataclass(frozen=True, slots=True)
class VerifiedExp449SmokeAAuthorityAntigravity:
    """Read-only, single-snapshot evidence for a separately reviewed A -> I delta."""

    approved_authority_commit: str
    implementation_commit: str
    grant: Exp449SmokeAGrantAntigravity
    grant_sha256: str
    grant_bytes: bytes
    digest_bytes: bytes
    grant_stat_identity: tuple[int, int, int, int]
    digest_stat_identity: tuple[int, int, int, int]


class _Exp449ExecutionCapabilityAntigravity:
    """Opaque registry key; it intentionally exposes no mutable authority facts."""

    __slots__ = ()

    def __copy__(self): raise TypeError("Exp449 execution capability cannot be copied")
    def __deepcopy__(self, memo: object): raise TypeError("Exp449 execution capability cannot be copied")
    def __reduce__(self): raise TypeError("Exp449 execution capability cannot be serialized")
    def __reduce_ex__(self, protocol: int): raise TypeError("Exp449 execution capability cannot be serialized")


@dataclass(frozen=True, slots=True)
class _ExecutionCapabilityFactsAntigravity:
    capability: object
    grant_sha256: str
    approved_authority_commit: str
    implementation_commit: str
    context_sha256: str
    target: str
    scientific_root: Path
    operations_root: Path
    claim_path: Path
    owner_pid: int
    claim_fd: int
    claim_device: int
    claim_inode: int
    receipt_path: Path
    receipt_sha256: str


def _claim_lock_contended_antigravity(path: Path, *, expected_device: int, expected_inode: int) -> bool:
    """Observe lock ownership in a fresh process without touching the owner FD."""
    child = os.fork()
    if child == 0:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            details = os.fstat(descriptor)
            path_details = path.lstat()
            if (not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(path_details.st_mode)
                or not stat.S_ISREG(path_details.st_mode)
                or (details.st_dev, details.st_ino) != (expected_device, expected_inode)
                or (path_details.st_dev, path_details.st_ino) != (expected_device, expected_inode)):
                os._exit(2)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                os._exit(0 if error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK} else 2)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os._exit(1)
            finally:
                os.close(descriptor)
        except BaseException:
            os._exit(2)
    try:
        _pid, status = os.waitpid(child, 0)
    except OSError:
        return False
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def _execution_capability_registry_antigravity():
    """Keep validation facts in a closure, keyed by the issued object identity."""
    records: dict[int, _ExecutionCapabilityFactsAntigravity] = {}

    def mint(verified: VerifiedExp449SmokeAAuthorityAntigravity, *, context: Exp449Context, target: str, scientific_root: Path, operations_root: Path, claim_path: Path, claim_fd: int) -> object:
        if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity or type(context) is not Exp449Context:
            raise TypeError("Exp449 execution capability inputs are invalid")
        details = os.fstat(claim_fd)
        receipt_path = Path(operations_root) / "execution_authority_consumed_antigravity.json"
        receipt_details = receipt_path.lstat()
        receipt_bytes = receipt_path.read_bytes()
        if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(receipt_details.st_mode) or not stat.S_ISREG(receipt_details.st_mode):
            raise ValueError("Exp449 execution capability inputs are unsafe")
        capability = _Exp449ExecutionCapabilityAntigravity()
        records[id(capability)] = _ExecutionCapabilityFactsAntigravity(
            capability, verified.grant_sha256, verified.approved_authority_commit, verified.implementation_commit,
            context_sha256_antigravity(context), target, Path(scientific_root), Path(operations_root), Path(claim_path),
            os.getpid(), claim_fd, details.st_dev, details.st_ino, receipt_path, sha256(receipt_bytes).hexdigest(),
        )
        return capability

    def revoke(capability: object) -> None:
        facts = records.get(id(capability))
        if facts is not None and facts.capability is capability:
            del records[id(capability)]

    def claim_fd(capability: object) -> int:
        """Return the registered descriptor without making it a capability attribute."""
        facts = records.get(id(capability))
        if type(capability) is not _Exp449ExecutionCapabilityAntigravity or facts is None or facts.capability is not capability:
            return -1
        return facts.claim_fd

    def require(capability: object, *, target: str, context: Exp449Context, claim_fd: int) -> None:
        facts = records.get(id(capability))
        if type(capability) is not _Exp449ExecutionCapabilityAntigravity or facts is None or facts.capability is not capability:
            raise ValueError("Exp449 execution capability is invalid")
        if type(context) is not Exp449Context:
            raise ValueError("Exp449 execution context is invalid")
        if facts.owner_pid != os.getpid() or facts.claim_fd != claim_fd or facts.target != target or facts.context_sha256 != context_sha256_antigravity(context) or facts.scientific_root != Path(context.run_root):
            raise ValueError("Exp449 execution capability binding differs")
        try:
            details = os.fstat(claim_fd); path_details = facts.claim_path.lstat(); receipt_details = facts.receipt_path.lstat(); receipt_bytes = facts.receipt_path.read_bytes()
        except OSError as error:
            raise ValueError("Exp449 execution claim descriptor is closed") from error
        if (not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(path_details.st_mode) or not stat.S_ISREG(path_details.st_mode) or (details.st_dev, details.st_ino) != (facts.claim_device, facts.claim_inode) or (path_details.st_dev, path_details.st_ino) != (facts.claim_device, facts.claim_inode) or stat.S_ISLNK(receipt_details.st_mode) or not stat.S_ISREG(receipt_details.st_mode) or sha256(receipt_bytes).hexdigest() != facts.receipt_sha256):
            raise ValueError("Exp449 execution claim or consumption receipt differs")
        # Same-account replacement while another process deliberately holds the
        # same claim is outside this process-local capability threat model.
        if not _claim_lock_contended_antigravity(
            facts.claim_path, expected_device=facts.claim_device, expected_inode=facts.claim_inode,
        ):
            raise ValueError("Exp449 execution claim lock is not live")

    return mint, revoke, require, claim_fd


_mint_execution_capability_impl_antigravity, _revoke_execution_capability_impl_antigravity, _require_execution_capability_impl_antigravity, _execution_capability_claim_fd_impl_antigravity = _execution_capability_registry_antigravity()


def _mint_execution_capability_antigravity(*args: object, **kwargs: object) -> object:
    return _mint_execution_capability_impl_antigravity(*args, **kwargs)


def _revoke_execution_capability_antigravity(capability: object) -> None:
    _revoke_execution_capability_impl_antigravity(capability)


def _execution_capability_claim_fd_antigravity(capability: object) -> int:
    return _execution_capability_claim_fd_impl_antigravity(capability)


def _require_execution_capability_antigravity(capability: object, *, target: str, context: Exp449Context, claim_fd: int) -> None:
    _require_execution_capability_impl_antigravity(capability, target=target, context=context, claim_fd=claim_fd)


def _git_authority_antigravity(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ("git", *arguments), cwd=repository_root, check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Smoke A authorization Git evidence is unavailable") from error


def _git_text_authority_antigravity(repository_root: Path, *arguments: str) -> str:
    try:
        return _git_authority_antigravity(repository_root, *arguments).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Smoke A authorization Git text is invalid") from error


def _tracked_regular_blob_antigravity(repository_root: Path, commit: str, relative: Path) -> bytes:
    entry = _git_text_authority_antigravity(repository_root, "ls-tree", commit, "--", str(relative))
    parts = entry.split("\t", 1)
    if len(parts) != 2 or parts[1] != str(relative):
        raise ValueError("Smoke A authorization path is not tracked exactly")
    metadata = parts[0].split()
    if len(metadata) != 3 or metadata[0] != "100644" or metadata[1] != "blob" or not _lower_hex_antigravity(metadata[2], 40):
        raise ValueError("Smoke A authorization path is not a tracked regular file")
    return _git_authority_antigravity(repository_root, "show", f"{commit}:{relative}")


def _lower_hex_antigravity(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _read_no_follow_authority_snapshot_antigravity(path: Path) -> tuple[bytes, tuple[int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Smoke A authority file is unsafe") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError("Smoke A authority file is not a single regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)
    finally:
        os.close(descriptor)


def _require_exact_authority_delta_antigravity(repository_root: Path, implementation: str, approved: str) -> None:
    parents = _git_text_authority_antigravity(repository_root, "rev-list", "--parents", "-n", "1", approved).split()
    if len(parents) != 2 or parents[0] != approved or parents[1] != implementation:
        raise ValueError("Smoke A authority commit is not a direct child of its implementation")
    expected = {
        f"A\t{_SMOKE_A_GRANT_RELATIVE_PATH_ANTIGRAVITY}",
        f"A\t{_SMOKE_A_DIGEST_RELATIVE_PATH_ANTIGRAVITY}",
    }
    observed = set(_git_text_authority_antigravity(
        repository_root, "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", implementation, approved,
    ).splitlines())
    if observed != expected:
        raise ValueError("Smoke A authority delta must add exactly the grant and digest")


def _require_identical_tree_binding_antigravity(
    repository_root: Path, implementation: str, approved: str, relative: Path, expected_sha256: str,
) -> None:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Smoke A authority closure path is unsafe")
    implementation_bytes = _tracked_regular_blob_antigravity(repository_root, implementation, relative)
    approved_bytes = _tracked_regular_blob_antigravity(repository_root, approved, relative)
    if implementation_bytes != approved_bytes or sha256(implementation_bytes).hexdigest() != expected_sha256:
        raise ValueError("Smoke A authorization commit changes a bound closure or lock")


def verify_smoke_a_authorization_commit_antigravity(
    approved_authority_commit: str,
    *,
    repository_root: Path = REPOSITORY_ROOT_ANTIGRAVITY,
) -> VerifiedExp449SmokeAAuthorityAntigravity:
    """Verify only the exact clean two-file authority child commit and retain its bytes.

    This is evidence verification, not a grant issuer or an execution capability.
    """
    if not _lower_hex_antigravity(approved_authority_commit, 40):
        raise ValueError("Smoke A approved authority commit is invalid")
    repository = Path(repository_root)
    head = _git_text_authority_antigravity(repository, "rev-parse", "HEAD")
    dirty = _git_authority_antigravity(repository, "status", "--porcelain", "--untracked-files=all")
    if head != approved_authority_commit or dirty:
        raise ValueError("Smoke A authorization requires the clean approved HEAD")
    authority_directory = repository / _SMOKE_A_GRANT_RELATIVE_PATH_ANTIGRAVITY.parent
    if verify_authority_directory_structure_antigravity(authority_directory) != tuple(sorted(_APPROVED_AUTHORITY_FILENAMES_ANTIGRAVITY)):
        raise ValueError("Smoke A authority directory does not contain exactly two approved files")
    grant_bytes, grant_identity = _read_no_follow_authority_snapshot_antigravity(
        repository / _SMOKE_A_GRANT_RELATIVE_PATH_ANTIGRAVITY
    )
    digest_bytes, digest_identity = _read_no_follow_authority_snapshot_antigravity(
        repository / _SMOKE_A_DIGEST_RELATIVE_PATH_ANTIGRAVITY
    )
    if grant_bytes != _tracked_regular_blob_antigravity(repository, approved_authority_commit, _SMOKE_A_GRANT_RELATIVE_PATH_ANTIGRAVITY):
        raise ValueError("Smoke A grant snapshot differs from its committed blob")
    if digest_bytes != _tracked_regular_blob_antigravity(repository, approved_authority_commit, _SMOKE_A_DIGEST_RELATIVE_PATH_ANTIGRAVITY):
        raise ValueError("Smoke A digest snapshot differs from its committed blob")
    grant = parse_grant_bytes_antigravity(grant_bytes)
    digest = parse_grant_digest_bytes_antigravity(digest_bytes)
    grant_sha256 = sha256(grant_bytes).hexdigest()
    implementation = grant.implementation_commit
    if digest.implementation_commit != implementation or digest.grant_size != len(grant_bytes) or digest.grant_sha256 != grant_sha256:
        raise ValueError("Smoke A digest does not bind the retained grant snapshot")
    _require_exact_authority_delta_antigravity(repository, implementation, approved_authority_commit)
    closures = dict(grant.scientific_code_closure)
    if set(closures).intersection(grant.authority_code_closure):
        raise ValueError("Smoke A authority closures overlap")
    closures.update(grant.authority_code_closure)
    for relative, expected_sha256 in closures.items():
        _require_identical_tree_binding_antigravity(
            repository, implementation, approved_authority_commit, Path(relative), expected_sha256,
        )
    _require_identical_tree_binding_antigravity(
        repository, implementation, approved_authority_commit, _LINEAGE_LOCK_RELATIVE_PATH_ANTIGRAVITY, grant.lineage_lock_sha256,
    )
    _require_identical_tree_binding_antigravity(
        repository, implementation, approved_authority_commit, _CODEMAP_LOCK_RELATIVE_PATH_ANTIGRAVITY, grant.codemap_lock_sha256,
    )
    return VerifiedExp449SmokeAAuthorityAntigravity(
        approved_authority_commit=approved_authority_commit,
        implementation_commit=implementation,
        grant=grant,
        grant_sha256=grant_sha256,
        grant_bytes=grant_bytes,
        digest_bytes=digest_bytes,
        grant_stat_identity=grant_identity,
        digest_stat_identity=digest_identity,
    )
