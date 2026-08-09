"""Fail-closed, label-free readiness authority for Exp443 Task 8.

Readiness deliberately imports only frozen configuration, source, database and
environment boundaries.  It does not import the scorer, routing evaluator, or
runtime scheduler and therefore cannot accidentally score or inspect TEST
labels while creating its evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    scientific_config_sha256_antigravity,
)


_SOURCE_HASH_NAMES_ANTIGRAVITY = (
    "database", "exp249_prediction", "exp249_prediction_manifest",
    "exp249_artifact_manifest", "exp249_cache_manifest", "exp249_audit",
    "exp291_routing", "exp291_manifest", "exp414_repair_audit",
    "exp414_manifest", "exp404_hard", "exp404_manifest",
)
_THREAD_VARIABLES_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class ReadinessDependenciesAntigravity:
    """Injectable, label-free seams used by fixture tests and review tooling."""

    clean_head: Callable[[], str]
    verify_configuration: Callable[[], object]
    verify_sources: Callable[[], object]
    verify_environment: Callable[[], object]

    def __post_init__(self) -> None:
        if any(not callable(getattr(self, name)) for name in self.__dataclass_fields__):
            raise ValueError("Exp443 readiness dependency is invalid")


def _repo_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def current_clean_head_antigravity() -> str:
    """Return the only commit identity accepted by production readiness."""
    repository = _repo_root_antigravity()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    if status:
        raise ValueError("Exp443 readiness requires a clean working tree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Exp443 readiness clean HEAD is invalid")
    return commit


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe_antigravity(item) for item in value]
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _json_safe_antigravity(to_payload())
    to_payload = getattr(value, "to_payload_antigravity", None)
    if callable(to_payload):
        return _json_safe_antigravity(to_payload())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe_antigravity(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    raise ValueError(f"readiness evidence contains unsupported value: {type(value).__name__}")


def _field_antigravity(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _source_evidence_antigravity(source: object) -> dict[str, object]:
    """Detach the exact source inventory/routing authority without arrays."""
    datasets_value = _field_antigravity(source, "datasets", _field_antigravity(source, "context_datasets", ()))
    affected_value = _field_antigravity(source, "affected_datasets", ())
    families_value = _field_antigravity(source, "affected_families", ())
    datasets = tuple(datasets_value or ())
    affected = tuple(affected_value or ())
    families = tuple(families_value or ())
    if len(datasets) != FROZEN_CONFIG.dataset_count or datasets != tuple(sorted(set(datasets))):
        raise ValueError("readiness source dataset population differs")
    if len(affected) != FROZEN_CONFIG.affected_dataset_count or affected != tuple(sorted(set(affected)) or ()) or not set(affected).issubset(datasets):
        raise ValueError("readiness affected dataset population differs")
    if len(families) != FROZEN_CONFIG.affected_family_count or families != tuple(sorted(set(families))):
        raise ValueError("readiness affected family population differs")
    label_access = _field_antigravity(source, "label_access_count", 0)
    if type(label_access) is not int or label_access != 0:
        raise ValueError("readiness source label access is nonzero")
    for name in ("contains_test_labels", "contains_evaluation_metrics"):
        value = _field_antigravity(source, name, False)
        if type(value) is not bool or value is not False:
            raise ValueError(f"readiness source {name} is not false")
    scalar_names = (
        "affected_rows", "nonfinite_scalars", "all_missing_rows",
        "automatic_hard_count", "exact_normal_count", "standard_count", "priority_count",
    )
    expected_scalars = {
        "affected_rows": FROZEN_CONFIG.affected_normal_train_rows,
        "nonfinite_scalars": FROZEN_CONFIG.nonfinite_scalar_count,
        "all_missing_rows": FROZEN_CONFIG.all_missing_row_count,
        "automatic_hard_count": 2218,
        "exact_normal_count": 35,
        "standard_count": 806,
        "priority_count": 31,
    }
    scalars: dict[str, int] = {}
    for name in scalar_names:
        value = _field_antigravity(source, name, expected_scalars[name])
        if type(value) is not int or value != expected_scalars[name]:
            raise ValueError(f"readiness source {name} differs")
        scalars[name] = value
    routing = tuple(_field_antigravity(source, "routing_algebra", (2116, 137, 35, 2218)) or ())
    if routing != (2116, 137, 35, 2218):
        raise ValueError("readiness routing algebra differs")
    raw_hashes = _field_antigravity(source, "source_hashes", {})
    if not isinstance(raw_hashes, Mapping):
        raise ValueError("readiness source hashes are missing")
    source_hashes = {str(name): str(value) for name, value in raw_hashes.items()}
    if set(source_hashes) != set(_SOURCE_HASH_NAMES_ANTIGRAVITY):
        raise ValueError("readiness source hash inventory is incomplete")
    for name, value in source_hashes.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"readiness source hash is invalid: {name}")
    if source_hashes["database"] != FROZEN_CONFIG.database_sha256:
        raise ValueError("readiness database SHA256 differs")
    affected_sha = _field_antigravity(source, "affected_datasets_sha256", None)
    if affected_sha is None:
        affected_sha = sha256(json.dumps(list(affected), separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    if type(affected_sha) is not str or len(affected_sha) != 64 or any(character not in "0123456789abcdef" for character in affected_sha):
        raise ValueError("readiness affected dataset SHA256 is invalid")
    return {
        "dataset_count": len(datasets),
        "affected_dataset_count": len(affected),
        "affected_family_count": len(families),
        "affected_datasets": list(affected),
        "affected_families": list(families),
        "affected_rows": scalars["affected_rows"],
        "nonfinite_scalars": scalars["nonfinite_scalars"],
        "all_missing_rows": scalars["all_missing_rows"],
        "automatic_hard_count": scalars["automatic_hard_count"],
        "exact_normal_count": scalars["exact_normal_count"],
        "standard_count": scalars["standard_count"],
        "priority_count": scalars["priority_count"],
        "routing_algebra": list(routing),
        "affected_datasets_sha256": affected_sha,
        "source_hashes": source_hashes,
        "label_access_count": 0,
    }


def _configuration_evidence_antigravity(configuration: object) -> dict[str, object]:
    if configuration is False or configuration is None:
        raise ValueError("approved Exp443 configuration is unavailable")
    if isinstance(configuration, Mapping):
        payload = dict(_json_safe_antigravity(configuration))
        if payload.get("approved", True) is not True:
            raise ValueError("approved Exp443 configuration is not bound")
    else:
        payload = {"approved": True}
    payload.update({
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_design_sha256": FROZEN_CONFIG.reviewed_design_sha256,
        "scientific_config_sha256": scientific_config_sha256_antigravity(),
        "environment_lock_sha256": FROZEN_CONFIG.environment_lock_sha256,
        "environment_lock_path": FROZEN_CONFIG.environment_lock_path,
    })
    return payload


def _environment_evidence_antigravity(environment: object) -> dict[str, object]:
    payload = _json_safe_antigravity(environment)
    if not isinstance(payload, Mapping):
        raise ValueError("Exp443 environment evidence is invalid")
    copied = dict(payload)
    logical = copied.get("logical_cpu_count")
    workers = copied.get("outer_workers")
    inner = copied.get("inner_threads")
    if logical != FROZEN_CONFIG.logical_cpu_count or workers != FROZEN_CONFIG.outer_workers or inner != FROZEN_CONFIG.inner_threads:
        raise ValueError("Exp443 readiness CPU topology differs")
    return copied


def validate_readiness_payload_antigravity(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the compact readiness identity before runtime scheduling."""
    if not isinstance(payload, Mapping):
        raise ValueError("readiness payload is invalid")
    required = {
        "schema", "experiment_id", "run_identity", "reviewed_commit", "reviewed_design_sha256",
        "scientific_config_sha256", "environment_lock_sha256", "database_sha256", "source_snapshot_sha256",
        "source", "environment", "configuration", "dataset_count", "affected_dataset_count", "workers", "logical_cpu_count", "inner_threads",
        "contains_test_labels", "contains_evaluation_metrics", "label_access_count", "output_root",
    }
    if set(payload) != required:
        raise ValueError("readiness payload schema is invalid")
    if payload["schema"] != "exp443.readiness.v1" or payload["experiment_id"] != FROZEN_CONFIG.experiment_id or payload["run_identity"] != FROZEN_CONFIG.run_identity:
        raise ValueError("readiness identity differs")
    commit = payload["reviewed_commit"]
    if type(commit) is not str or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("readiness reviewed commit is invalid")
    for name, expected in (("reviewed_design_sha256", FROZEN_CONFIG.reviewed_design_sha256), ("scientific_config_sha256", scientific_config_sha256_antigravity()), ("environment_lock_sha256", FROZEN_CONFIG.environment_lock_sha256), ("database_sha256", FROZEN_CONFIG.database_sha256)):
        if payload[name] != expected:
            raise ValueError("readiness frozen identity differs")
    if payload["dataset_count"] != FROZEN_CONFIG.dataset_count or payload["affected_dataset_count"] != FROZEN_CONFIG.affected_dataset_count:
        raise ValueError("readiness population differs")
    if payload["workers"] != FROZEN_CONFIG.outer_workers or payload["logical_cpu_count"] != FROZEN_CONFIG.logical_cpu_count or payload["inner_threads"] != FROZEN_CONFIG.inner_threads:
        raise ValueError("readiness topology differs")
    if payload["contains_test_labels"] is not False or payload["contains_evaluation_metrics"] is not False or payload["label_access_count"] != 0:
        raise ValueError("readiness label gate failed")
    source = payload["source"]
    if not isinstance(source, Mapping) or source.get("dataset_count") != FROZEN_CONFIG.dataset_count or source.get("affected_dataset_count") != FROZEN_CONFIG.affected_dataset_count or source.get("label_access_count") != 0:
        raise ValueError("readiness source evidence differs")
    source_digest = sha256(canonical_json_bytes_antigravity(source)).hexdigest()
    if payload["source_snapshot_sha256"] != source_digest:
        raise ValueError("readiness source digest differs")
    return dict(payload)


def create_readiness_antigravity(
    run_root: Path | str,
    *,
    reviewed_commit: str,
    dependencies: ReadinessDependenciesAntigravity | None = None,
) -> dict[str, object]:
    """Create one immutable readiness file after all label-free gates pass."""
    root = Path(run_root)
    if root.exists() or root.is_symlink():
        raise ValueError("Exp443 readiness output root already exists")
    deps = dependencies or ReadinessDependenciesAntigravity(
        clean_head=current_clean_head_antigravity,
        verify_configuration=lambda: (FROZEN_CONFIG.validate() or True),
        verify_sources=lambda: __import__(
            "experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity",
            fromlist=["verify_frozen_sources_antigravity"],
        ).verify_frozen_sources_antigravity(),
        verify_environment=lambda: __import__(
            "experiments.exp443_missing_normal_train_policy_comparison_antigravity.environment_antigravity",
            fromlist=["verify_environment_antigravity"],
        ).verify_environment_antigravity(),
    )
    commit = deps.clean_head()
    if type(commit) is not str or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit) or commit != reviewed_commit:
        raise ValueError("readiness requires the exact clean reviewed commit")
    configuration = _configuration_evidence_antigravity(deps.verify_configuration())
    source = _source_evidence_antigravity(deps.verify_sources())
    environment = _environment_evidence_antigravity(deps.verify_environment())
    payload: dict[str, object] = {
        "schema": "exp443.readiness.v1",
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_commit": reviewed_commit,
        "reviewed_design_sha256": FROZEN_CONFIG.reviewed_design_sha256,
        "scientific_config_sha256": scientific_config_sha256_antigravity(),
        "environment_lock_sha256": FROZEN_CONFIG.environment_lock_sha256,
        "database_sha256": FROZEN_CONFIG.database_sha256,
        "source_snapshot_sha256": sha256(canonical_json_bytes_antigravity(source)).hexdigest(),
        "source": source,
        "environment": environment,
        "configuration": configuration,
        "dataset_count": source["dataset_count"],
        "affected_dataset_count": source["affected_dataset_count"],
        "workers": FROZEN_CONFIG.outer_workers,
        "logical_cpu_count": FROZEN_CONFIG.logical_cpu_count,
        "inner_threads": FROZEN_CONFIG.inner_threads,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "label_access_count": 0,
        "output_root": str(root.resolve()),
    }
    validate_readiness_payload_antigravity(payload)
    root.mkdir(mode=0o700)
    try:
        write_create_only_antigravity(root / "readiness_antigravity.json", canonical_json_bytes_antigravity(payload))
    except BaseException:
        try:
            root.rmdir()
        except OSError:
            pass
        raise
    return json.loads(canonical_json_bytes_antigravity(payload))


def run_readiness_antigravity(*, reviewed_commit: str, run_root: Path | str | None = None) -> dict[str, object]:
    """Production alias bound to the frozen Exp443 run root."""
    return create_readiness_antigravity(
        FROZEN_CONFIG.run_root if run_root is None else run_root,
        reviewed_commit=reviewed_commit,
    )


__all__ = [
    "ReadinessDependenciesAntigravity", "create_readiness_antigravity",
    "run_readiness_antigravity", "current_clean_head_antigravity",
    "validate_readiness_payload_antigravity",
]
