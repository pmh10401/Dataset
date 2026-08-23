"""Historical Exp450 authority reconstruction and create-only Exp451 migration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType
from typing import Mapping
from types import MappingProxyType

from .artifacts_antigravity import (
    SUCCESS_NAME_ANTIGRAVITY,
    read_exp451_dataset_condition_success_antigravity,
    success_relative_path_antigravity,
    write_exp451_dataset_condition_success_antigravity,
)
from .config_antigravity import (
    EXP450_SOURCE_CATALOG_SHA256,
    EXP450_SOURCE_COMMIT,
    EXP450_SOURCE_CONTEXT_SHA256,
    EXP450_SOURCE_ROOT,
    EXP450_SOURCE_TRANSFORMS_SHA256,
)
from .contracts_antigravity import (
    Exp451MigrationEntry,
    Exp451MigrationManifest,
    canonical_json_bytes,
)


TRANSFORMS_PATH_ANTIGRAVITY = "experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py"
_EXP450_COMMIT_FULL_ANTIGRAVITY = "eac0d3b45577f08cae67482d015da6af707e6fa4"
_EXP450_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/backend_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/config_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/imaging_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py",
    TRANSFORMS_PATH_ANTIGRAVITY,
    "experiments/exp450_resampling_image_ablation_antigravity/verifier_antigravity.py",
    "paper/equipment_anomaly_detection/data_paper.py",
)
_EXP450_BACKEND_RECORD_ANTIGRAVITY = {
    "tensor_backend": "mps", "torch_version": "2.13.0", "mps_available": True,
    "mlx_available": False, "logical_cpu_count": 12, "cpu_workers": 11,
    "inner_numerical_threads": 1,
}
_EXP450_CONFIG_SHA256_ANTIGRAVITY = "9251e3a84f96cb7f51b26e2c1295263561ed0dd8f1adf70aaad67df4c5352720"
_EXP450_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_SOURCE_SUCCESS_SCHEMA_ANTIGRAVITY = "exp450-dataset-condition-success-antigravity.v1"


def _digest_antigravity(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class Exp450SourceContextAntigravity:
    """The historical Exp450 context shape, kept separate from Exp451 contracts."""

    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Mapping[str, object]
    config_sha256: str

    def __post_init__(self) -> None:
        if self.mode != "full" or Path(self.run_root) != EXP450_SOURCE_ROOT:
            raise ValueError("Exp450 source context root differs")
        if not all(_digest_antigravity(item) for item in (
            self.database_sha256, self.catalog_sha256, self.config_sha256
        )) or self.catalog_sha256 != EXP450_SOURCE_CATALOG_SHA256:
            raise ValueError("Exp450 source context digest differs")
        closure = dict(self.code_closure)
        if tuple(sorted(closure)) != _EXP450_CLOSURE_PATHS_ANTIGRAVITY or not all(
            _digest_antigravity(item) for item in closure.values()
        ):
            raise ValueError("Exp450 source context closure differs")
        if dict(self.backend) != _EXP450_BACKEND_RECORD_ANTIGRAVITY:
            raise ValueError("Exp450 source context backend differs")
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "code_closure", MappingProxyType(dict(sorted(closure.items()))))
        object.__setattr__(self, "backend", MappingProxyType(dict(self.backend)))

    def artifact_record(self) -> dict[str, object]:
        return {
            "mode": self.mode, "run_root": str(self.run_root),
            "database_sha256": self.database_sha256, "catalog_sha256": self.catalog_sha256,
            "code_closure": dict(self.code_closure), "backend": dict(self.backend),
            "config_sha256": self.config_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(self.artifact_record())).hexdigest()


@dataclass(frozen=True, slots=True)
class FixtureMigrationAuthorityAntigravity:
    """Explicit test-only source authority; production never infers this authority."""

    source_root: Path
    source_context_sha256: str
    catalog_sha256: str
    backend: Mapping[str, object] = MappingProxyType(_EXP450_BACKEND_RECORD_ANTIGRAVITY)

    def __post_init__(self) -> None:
        root = Path(self.source_root)
        if not root.is_absolute() or root == EXP450_SOURCE_ROOT:
            raise ValueError("Exp451 fixture source authority root is invalid")
        if not _digest_antigravity(self.source_context_sha256) or not _digest_antigravity(self.catalog_sha256):
            raise ValueError("Exp451 fixture source authority digest is invalid")
        if dict(self.backend) != _EXP450_BACKEND_RECORD_ANTIGRAVITY:
            raise ValueError("Exp451 fixture source authority backend is invalid")
        object.__setattr__(self, "source_root", root)
        object.__setattr__(self, "backend", MappingProxyType(dict(self.backend)))


@dataclass(frozen=True, slots=True)
class _FixtureSourceContextAntigravity:
    sha256: str
    catalog_sha256: str
    backend: Mapping[str, object]


def _historical_git_bytes_antigravity(repository: Path, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{_EXP450_COMMIT_FULL_ANTIGRAVITY}:{path}"],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("Exp450 historical closure path is unavailable")
    return result.stdout


def _require_historical_commit_antigravity(repository: Path) -> Path:
    repository = Path(repository).resolve()
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", f"{EXP450_SOURCE_COMMIT}^{{commit}}"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != _EXP450_COMMIT_FULL_ANTIGRAVITY:
        raise ValueError("Exp450 fixed source commit is unavailable")
    return repository


def reconstruct_exp450_source_context_antigravity(
    repository: Path, catalog: Mapping[str, object]
) -> Exp450SourceContextAntigravity:
    """Rebuild the full historical source context only from fixed-commit Git bytes."""
    if not isinstance(catalog, Mapping) or catalog.get("catalog_sha256") != EXP450_SOURCE_CATALOG_SHA256:
        raise ValueError("Exp450 source catalog differs")
    repository = _require_historical_commit_antigravity(Path(repository))
    closure = {
        path: sha256(_historical_git_bytes_antigravity(repository, path)).hexdigest()
        for path in _EXP450_CLOSURE_PATHS_ANTIGRAVITY
    }
    context = Exp450SourceContextAntigravity(
        "full", EXP450_SOURCE_ROOT, _EXP450_DATABASE_SHA256_ANTIGRAVITY,
        EXP450_SOURCE_CATALOG_SHA256, closure, _EXP450_BACKEND_RECORD_ANTIGRAVITY,
        _EXP450_CONFIG_SHA256_ANTIGRAVITY,
    )
    if context.code_closure[TRANSFORMS_PATH_ANTIGRAVITY] != EXP450_SOURCE_TRANSFORMS_SHA256:
        raise ValueError("Exp450 historical transforms digest differs")
    if context.sha256 != EXP450_SOURCE_CONTEXT_SHA256:
        raise ValueError("Exp450 reconstructed context digest differs")
    return context


@lru_cache(maxsize=1)
def _historical_artifact_parser_antigravity(repository: Path) -> ModuleType:
    """Load the frozen Exp450 serializer from its fixed Git object, never the worktree."""
    source = _historical_git_bytes_antigravity(
        repository,
        "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py",
    )
    module = ModuleType("_exp450_historical_artifacts_antigravity")
    module.__package__ = "experiments.exp450_resampling_image_ablation_antigravity"
    sys.modules[module.__name__] = module
    try:
        exec(
            compile(source, "eac0d3b4:exp450/artifacts_antigravity.py", "exec"),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def _validate_historical_typed_payload_antigravity(
    repository: Path, payload: object, entry: Mapping[str, object]
) -> None:
    """Deserialize and serialise the frozen typed source/route closure exactly once."""
    if not isinstance(payload, dict) or set(payload) != {"sources", "route"}:
        raise ValueError("Exp450 source checkpoint payload differs")
    train, test = entry.get("train"), entry.get("test")
    if not isinstance(train, dict) or not isinstance(test, dict) or type(train.get("count")) is not int or type(test.get("count")) is not int:
        raise ValueError("Exp450 source checkpoint raw identity differs")
    parser = _historical_artifact_parser_antigravity(repository)
    try:
        sources = parser._sources_from_payload(payload["sources"], train["count"], test["count"])
        route = parser._route_from_payload(payload["route"], test["count"])
        round_trip = {
            "sources": parser._source_payload(sources, train["count"], test["count"]),
            "route": parser._route_payload(route, test["count"]),
        }
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("Exp450 source checkpoint typed payload differs") from error
    if canonical_json_bytes(round_trip) != canonical_json_bytes(payload):
        raise ValueError("Exp450 source checkpoint typed payload round-trip differs")


def _canonical_json_file_antigravity(path: Path, *, source: bool) -> dict[str, object]:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp450 source checkpoint is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp450 source checkpoint is invalid")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp450 source checkpoint JSON is invalid") from error
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise ValueError("Exp450 source checkpoint is noncanonical")
    return value


def _catalog_entry_antigravity(catalog: Mapping[str, object], dataset: str) -> dict[str, object]:
    datasets = catalog.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Exp450 source catalog dataset inventory differs")
    matches = [item for item in datasets if isinstance(item, dict) and item.get("dataset") == dataset]
    if len(matches) != 1:
        raise ValueError("Exp450 source catalog raw identity differs")
    return matches[0]


def _verify_source_catalog_antigravity(catalog: Mapping[str, object]) -> None:
    """Bind the supplied catalog bytes before any source checkpoint is opened."""
    required = {"schema", "mode", "database", "datasets", "catalog_sha256"}
    if not isinstance(catalog, Mapping) or set(catalog) != required:
        raise ValueError("Exp450 source catalog schema differs")
    bare = {key: catalog[key] for key in catalog if key != "catalog_sha256"}
    if (
        catalog["mode"] != "full"
        or catalog["catalog_sha256"] != EXP450_SOURCE_CATALOG_SHA256
        or sha256(canonical_json_bytes(bare)).hexdigest() != EXP450_SOURCE_CATALOG_SHA256
    ):
        raise ValueError("Exp450 source catalog differs")
    if not isinstance(catalog["datasets"], list) or not isinstance(catalog["database"], dict):
        raise ValueError("Exp450 source catalog schema differs")


def _source_relative_path_antigravity(dataset: str, condition_id: str) -> str:
    return success_relative_path_antigravity(dataset, condition_id)


def _source_envelope_antigravity(
    repository: Path, source_root: Path, relative_path: str, context: Exp450SourceContextAntigravity | _FixtureSourceContextAntigravity,
    catalog: Mapping[str, object],
) -> tuple[dict[str, object], bytes]:
    source_root = Path(source_root)
    if not source_root.is_absolute() or source_root.is_symlink():
        raise ValueError("Exp450 source root is invalid")
    path = source_root / relative_path
    try:
        path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("Exp450 source path escapes root") from error
    value = _canonical_json_file_antigravity(path, source=True)
    raw = path.read_bytes()
    required = {
        "schema", "context_sha256", "catalog_sha256", "condition_id", "dataset",
        "raw_identity", "backend", "payload", "payload_sha256",
    }
    if set(value) != required or value["schema"] != _SOURCE_SUCCESS_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp450 source checkpoint schema differs")
    if value["context_sha256"] != context.sha256 or value["catalog_sha256"] != context.catalog_sha256:
        raise ValueError("Exp450 source checkpoint context differs")
    if value["condition_id"] not in {"A", "B", "C", "D"} or type(value["dataset"]) is not str:
        raise ValueError("Exp450 source checkpoint coordinate differs")
    if value["backend"] != dict(context.backend):
        raise ValueError("Exp450 source checkpoint backend differs")
    if value["raw_identity"] != _catalog_entry_antigravity(catalog, value["dataset"]):
        raise ValueError("Exp450 source checkpoint raw identity differs")
    if value["payload_sha256"] != sha256(canonical_json_bytes(value["payload"])).hexdigest():
        raise ValueError("Exp450 source checkpoint payload digest differs")
    _validate_historical_typed_payload_antigravity(
        Path(repository), value["payload"], value["raw_identity"]
    )
    return value, raw


def _source_authority_antigravity(
    repository: Path, source_root: Path, catalog: Mapping[str, object],
    fixture_authority: FixtureMigrationAuthorityAntigravity | None,
) -> tuple[Path, Exp450SourceContextAntigravity | _FixtureSourceContextAntigravity]:
    """Select frozen official authority, or an explicit isolated fixture authority."""
    source_root = Path(source_root)
    if fixture_authority is None:
        if source_root.resolve(strict=False) != EXP450_SOURCE_ROOT:
            raise ValueError("Exp451 migration source root must be the frozen official Exp450 root")
        context = reconstruct_exp450_source_context_antigravity(repository, catalog)
        _verify_source_catalog_antigravity(catalog)
        return source_root, context
    if source_root.resolve(strict=False) != fixture_authority.source_root.resolve(strict=False):
        raise ValueError("Exp451 fixture source root differs from explicit authority")
    if source_root.resolve(strict=False) == EXP450_SOURCE_ROOT:
        raise ValueError("Exp451 fixture authority cannot authorize the official source root")
    if not isinstance(catalog, Mapping) or catalog.get("catalog_sha256") != fixture_authority.catalog_sha256:
        raise ValueError("Exp451 fixture source catalog differs")
    return source_root, _FixtureSourceContextAntigravity(
        fixture_authority.source_context_sha256,
        fixture_authority.catalog_sha256,
        fixture_authority.backend,
    )


def _source_checkpoint_paths_antigravity(source_root: Path) -> tuple[str, ...]:
    source_root = Path(source_root)
    try:
        source_details = source_root.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp450 source root is missing") from error
    if stat.S_ISLNK(source_details.st_mode) or not stat.S_ISDIR(source_details.st_mode):
        raise ValueError("Exp450 source root is invalid")
    base = source_root / "checkpoints_antigravity"
    try:
        base_details = base.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp450 source checkpoint directory is missing") from error
    if stat.S_ISLNK(base_details.st_mode) or not stat.S_ISDIR(base_details.st_mode):
        raise ValueError("Exp450 source checkpoint directory is invalid")
    failures = base / "failures_antigravity"
    if failures.exists() or failures.is_symlink():
        raise ValueError("Exp450 source failure ledger forbids migration")
    paths: list[str] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("Exp450 source checkpoint is symlinked")
        if path.is_file():
            if path.name != SUCCESS_NAME_ANTIGRAVITY:
                raise ValueError("Exp450 source checkpoint inventory contains an unexpected file")
            paths.append(path.relative_to(source_root).as_posix())
    if not paths:
        raise ValueError("Exp450 source checkpoint inventory is empty")
    return tuple(paths)


def _target_envelope_antigravity(
    source: Mapping[str, object], source_context: Exp450SourceContextAntigravity,
    target_context_sha256: str, source_envelope_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "exp451-dataset-condition-success-antigravity.v1",
        "context_sha256": target_context_sha256,
        "source_context_sha256": source_context.sha256,
        "catalog_sha256": source_context.catalog_sha256,
        "condition_id": source["condition_id"], "dataset": source["dataset"],
        "raw_identity": source["raw_identity"], "backend": source["backend"],
        "execution_origin": "imported_exp450_mps", "payload": source["payload"],
        "payload_sha256": source["payload_sha256"], "source_envelope_sha256": source_envelope_sha256,
    }


def _write_manifest_antigravity(target_root: Path, manifest: Exp451MigrationManifest) -> Path:
    envelope = {**manifest.artifact_record(), "manifest_sha256": manifest.sha256}
    from .artifacts_antigravity import _checked_target_path_antigravity, _publish_create_only_antigravity
    destination = _checked_target_path_antigravity(
        Path(target_root), "migration_manifest_antigravity.json", create=True
    )
    _publish_create_only_antigravity(destination, canonical_json_bytes(envelope))
    return destination


def migrate_verified_exp450_results_antigravity(
    *, repository: Path, source_root: Path, target_root: Path, catalog: Mapping[str, object],
    target_context_sha256: str, fixture_authority: FixtureMigrationAuthorityAntigravity | None = None,
) -> Exp451MigrationManifest:
    """Verify every historical source leaf, then create-only publish its Exp451 twin."""
    if not _digest_antigravity(target_context_sha256):
        raise ValueError("Exp451 target context digest is invalid")
    repository = _require_historical_commit_antigravity(Path(repository))
    source_root, source_context = _source_authority_antigravity(
        repository, source_root, catalog, fixture_authority
    )
    target_root = Path(target_root)
    if not source_root.is_absolute() or not target_root.is_absolute() or source_root == target_root:
        raise ValueError("Exp451 migration roots are invalid")
    source_real, target_real = source_root.resolve(strict=False), target_root.resolve(strict=False)
    if source_real.is_relative_to(target_real) or target_real.is_relative_to(source_real):
        raise ValueError("Exp451 source and target roots may not overlap")
    entries: list[Exp451MigrationEntry] = []
    seen: set[tuple[str, str]] = set()
    for relative_path in _source_checkpoint_paths_antigravity(source_root):
        source, raw = _source_envelope_antigravity(repository, source_root, relative_path, source_context, catalog)
        coordinate = (source["dataset"], source["condition_id"])
        if coordinate in seen:
            raise ValueError("Exp450 source checkpoint coordinate is duplicated")
        expected_source = _source_relative_path_antigravity(*coordinate)
        if relative_path != expected_source:
            raise ValueError("Exp450 source checkpoint path differs")
        seen.add(coordinate)
        source_hash = sha256(raw).hexdigest()
        target_relative = success_relative_path_antigravity(*coordinate)
        target = _target_envelope_antigravity(source, source_context, target_context_sha256, source_hash)
        target_path = write_exp451_dataset_condition_success_antigravity(target_root, target, target_relative)
        entries.append(Exp451MigrationEntry(
            dataset=coordinate[0], condition_id=coordinate[1], source_relative_path=relative_path,
            source_envelope_sha256=source_hash, payload_sha256=source["payload_sha256"],
            target_relative_path=target_path.relative_to(target_root).as_posix(),
            target_envelope_sha256=sha256(target_path.read_bytes()).hexdigest(),
        ))
    manifest = Exp451MigrationManifest(
        source_context_sha256=source_context.sha256, entries=tuple(entries),
        target_context_sha256=target_context_sha256,
    )
    _write_manifest_antigravity(target_root, manifest)
    return manifest


def verify_migration_manifest_antigravity(
    *, repository: Path, source_root: Path, target_root: Path, catalog: Mapping[str, object],
    target_context_sha256: str, fixture_authority: FixtureMigrationAuthorityAntigravity | None = None,
) -> Exp451MigrationManifest:
    """Revalidate the complete source-to-target mapping without modifying either root."""
    repository = _require_historical_commit_antigravity(Path(repository))
    source_root, source_context = _source_authority_antigravity(
        repository, source_root, catalog, fixture_authority
    )
    target_root = Path(target_root)
    source_real, target_real = source_root.resolve(strict=False), target_root.resolve(strict=False)
    if source_real.is_relative_to(target_real) or target_real.is_relative_to(source_real):
        raise ValueError("Exp451 source and target roots may not overlap")
    manifest_path = target_root / "migration_manifest_antigravity.json"
    value = _canonical_json_file_antigravity(manifest_path, source=False)
    expected = {"source_context_sha256", "target_context_sha256", "total_entries", "counts_by_condition", "entries", "manifest_sha256"}
    if set(value) != expected or value["manifest_sha256"] != sha256(canonical_json_bytes({key: value[key] for key in value if key != "manifest_sha256"})).hexdigest():
        raise ValueError("Exp451 migration manifest differs")
    manifest = Exp451MigrationManifest(
        source_context_sha256=value["source_context_sha256"], target_context_sha256=value["target_context_sha256"],
        entries=tuple(Exp451MigrationEntry(**entry) for entry in value["entries"]),
    )
    if manifest.sha256 != value["manifest_sha256"] or manifest.source_context_sha256 != source_context.sha256 or manifest.target_context_sha256 != target_context_sha256:
        raise ValueError("Exp451 migration manifest context differs")
    for entry in manifest.entries:
        if entry.source_relative_path != _source_relative_path_antigravity(entry.dataset, entry.condition_id):
            raise ValueError("Exp451 migration source path differs")
        if entry.target_relative_path != success_relative_path_antigravity(entry.dataset, entry.condition_id):
            raise ValueError("Exp451 migration target path differs")
        source, raw = _source_envelope_antigravity(repository, source_root, entry.source_relative_path, source_context, catalog)
        target = read_exp451_dataset_condition_success_antigravity(target_root, entry.target_relative_path)
        expected_target = _target_envelope_antigravity(
            source, source_context, target_context_sha256, entry.source_envelope_sha256
        )
        if (
            sha256(raw).hexdigest() != entry.source_envelope_sha256
            or target["source_envelope_sha256"] != entry.source_envelope_sha256
            or sha256(canonical_json_bytes(target)).hexdigest() != entry.target_envelope_sha256
            or target["payload_sha256"] != entry.payload_sha256
            or target["payload"] != source["payload"]
            or target["dataset"] != entry.dataset
            or target["condition_id"] != entry.condition_id
            or canonical_json_bytes(target) != canonical_json_bytes(expected_target)
        ):
            raise ValueError("Exp451 migration entry differs")
    return manifest


__all__ = [
    "Exp450SourceContextAntigravity", "TRANSFORMS_PATH_ANTIGRAVITY",
    "migrate_verified_exp450_results_antigravity", "reconstruct_exp450_source_context_antigravity",
    "verify_migration_manifest_antigravity",
]
