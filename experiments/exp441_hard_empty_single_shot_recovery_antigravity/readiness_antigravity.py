"""Atomic, label-free readiness ownership for the fixed Exp441 lifecycle."""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import platform
import stat
import subprocess
from typing import Mapping
from uuid import uuid4

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    exclusive_rename_antigravity,
    fsync_directory_antigravity,
    pin_exact_regular_inventory_antigravity,
    parse_canonical_json_bytes_antigravity,
    write_create_only_antigravity,
)
from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity, governance_payload_antigravity
from .contracts_antigravity import (
    BackendAssessmentAntigravity,
    ReadinessBindingAntigravity,
    validate_production_selection_antigravity,
)
from .lifecycle_profile_antigravity import (
    EXP441_PROFILE_ANTIGRAVITY,
    RecoveryLifecycleProfileAntigravity,
)
from .selector_antigravity import select_hard_empty_single_shot_antigravity
from .sources_antigravity import verify_frozen_sources_antigravity


_RUN_ROOT_ANTIGRAVITY = EXP441_PROFILE_ANTIGRAVITY.root_antigravity
_READINESS_NAME_ANTIGRAVITY = "readiness_antigravity.json"
_GOVERNANCE_ANTIGRAVITY = governance_payload_antigravity()
_READINESS_KEYS_ANTIGRAVITY = frozenset({
    "schema", "experiment_id", "run_identity", "reviewed_commit", "database_path",
    "database_sha256", "database_access", "source_snapshot_sha256", "source",
    "selection", "backend", "label_access_count", "reviewed_preregistration_sha256",
    "reviewed_design_sha256", "scientific_config_sha256",
} | set(_GOVERNANCE_ANTIGRAVITY))


def _repo_root_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _lower_hex_antigravity(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(item in "0123456789abcdef" for item in value)


def _current_clean_head_antigravity() -> str:
    """Read the only launch identity accepted by the frozen lifecycle."""
    root = _repo_root_antigravity()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    if not _lower_hex_antigravity(head, 40) or dirty:
        raise ValueError("readiness requires a clean lowercase 40-hex HEAD")
    return head


def current_clean_head_antigravity() -> str:
    """Public fixed-head reader used by the CLI without exposing a path override."""
    return _current_clean_head_antigravity()


def _require_reviewed_head_antigravity(reviewed_commit: str) -> str:
    if not _lower_hex_antigravity(reviewed_commit, 40):
        raise ValueError("reviewed commit is invalid")
    if _current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("reviewed commit is not the current clean HEAD")
    return reviewed_commit


def _environment_antigravity() -> dict[str, object]:
    """Collect complete provenance; this selector deliberately has no tensor stage."""
    try:
        import numpy as numpy_module
    except ImportError as error:
        raise ValueError("NumPy is unavailable") from error
    if numpy_module.__version__ != FROZEN_CONFIG.numpy_version:
        raise ValueError("NumPy version differs from the frozen contract")
    try:
        import torch
        pytorch_version: str | None = str(torch.__version__)
        mps_built = bool(torch.backends.mps.is_built())
        mps_available = bool(torch.backends.mps.is_available())
    except ImportError:
        pytorch_version, mps_built, mps_available = None, False, False
    try:
        import mlx.core as mlx_core
        mlx_version: str | None = importlib.metadata.version("mlx")
        mlx_default_device: str | None = str(mlx_core.default_device())
    except ImportError:
        mlx_version, mlx_default_device = None, None
    try:
        mlx_metal_version: str | None = importlib.metadata.version("mlx-metal")
    except importlib.metadata.PackageNotFoundError:
        mlx_metal_version = None
    return {
        "python_version": platform.python_version(), "numpy_version": numpy_module.__version__,
        "pytorch_version": pytorch_version, "mps_built": mps_built, "mps_available": mps_available,
        "mlx_version": mlx_version, "mlx_metal_version": mlx_metal_version,
        "mlx_default_device": mlx_default_device,
        "acceleration_reason": "selector_has_no_tensor_stage",
    }


def _backend_antigravity() -> BackendAssessmentAntigravity:
    logical_cpus = os.cpu_count()
    if logical_cpus != 12 or cpu_worker_count_antigravity(logical_cpus) != 11:
        raise ValueError("Exp441 requires exactly 12 logical CPUs and 11 workers")
    return BackendAssessmentAntigravity(
        canonical_backend="cpu", workers=11, tensor_eligible=False,
        acceleration_benchmark_status="not_run_inapplicable",
        pytorch="inapplicable_no_tensor_stage", mlx="inapplicable_no_tensor_stage",
        mps="inapplicable_no_tensor_stage", environment=_environment_antigravity(),
    )


def _source_payload_antigravity(snapshot: object) -> dict[str, object]:
    """Detach the complete frozen label-free source evidence into a strict payload."""
    required = (
        "context_datasets", "record_digest_by_dataset", "hard_coordinates", "provisional_candidates",
        "context_sha256", "state_sha256", "retained_inventory_sha256", "publication_prediction_sha256",
        "publication_manifest_sha256", "publication_artifact_map_sha256", "hard_prediction_sha256",
        "hard_manifest_sha256", "hard_coordinate_identity_sha256", "label_access_count",
    )
    if any(not hasattr(snapshot, name) for name in required):
        raise ValueError("frozen source snapshot is invalid")
    source_hashes = {
        "context": snapshot.context_sha256, "state": snapshot.state_sha256,
        "retained_inventory": snapshot.retained_inventory_sha256,
        "publication_prediction": snapshot.publication_prediction_sha256,
        "publication_manifest": snapshot.publication_manifest_sha256,
        "publication_artifact_map": snapshot.publication_artifact_map_sha256,
        "hard_prediction": snapshot.hard_prediction_sha256, "hard_manifest": snapshot.hard_manifest_sha256,
        "hard_coordinate_identity": snapshot.hard_coordinate_identity_sha256,
    }
    expected = {
        "context": FROZEN_CONFIG.checkpoint_context_sha256, "state": FROZEN_CONFIG.checkpoint_state_sha256,
        "retained_inventory": FROZEN_CONFIG.checkpoint_retained_inventory_sha256,
        "publication_prediction": FROZEN_CONFIG.publication_prediction_sha256,
        "publication_manifest": FROZEN_CONFIG.publication_manifest_sha256,
        "publication_artifact_map": FROZEN_CONFIG.publication_artifact_map_sha256,
        "hard_prediction": FROZEN_CONFIG.hard_prediction_sha256, "hard_manifest": FROZEN_CONFIG.hard_manifest_sha256,
        "hard_coordinate_identity": FROZEN_CONFIG.hard_coordinate_identity_sha256,
    }
    if source_hashes != expected or snapshot.label_access_count != 0:
        raise ValueError("frozen source evidence differs")
    if len(snapshot.context_datasets) != 1117 or len(snapshot.record_digest_by_dataset) != 1117 or len(snapshot.hard_coordinates) != 2218:
        raise ValueError("frozen source counts differ")
    return {
        "context_dataset_count": len(snapshot.context_datasets), "record_digest_count": len(snapshot.record_digest_by_dataset),
        "hard_coordinate_count": len(snapshot.hard_coordinates), "provisional_candidate_count": len(snapshot.provisional_candidates),
        "label_access_count": 0, "source_hashes": source_hashes,
    }


def _selection_payload_antigravity(snapshot: object) -> dict[str, int]:
    result = validate_production_selection_antigravity(select_hard_empty_single_shot_antigravity(snapshot))
    return {
        "provisional_candidate_count": len(result.provisional_candidates),
        "outside_hard_candidate_count": len(result.outside_hard_candidates),
        "eligible_candidate_count": len(result.eligible_candidates), "winner_count": len(result.winners),
        "selected_candidate_count": len(result.selected),
    }


def _require_absent_output_roots_antigravity(
    run_root: Path | None = None,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> None:
    root = _RUN_ROOT_ANTIGRAVITY if run_root is None else Path(run_root)
    outputs = (root, root / "attempts_antigravity", root / "publication_antigravity", root / "evaluation_antigravity")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError(f"{profile.experiment_id} fixed run parent and output roots must be absent")
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"{profile.experiment_id} run parent container is invalid")


def _readiness_payload_antigravity(
    reviewed_commit: str, source: Mapping[str, object], selection: Mapping[str, object],
    backend: BackendAssessmentAntigravity,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> dict[str, object]:
    source_bytes = canonical_json_bytes_antigravity(dict(source))
    payload: dict[str, object] = {
        "schema": profile.schema_antigravity("readiness"), "experiment_id": profile.experiment_id,
        "run_identity": profile.run_identity, "reviewed_commit": reviewed_commit,
        "reviewed_preregistration_sha256": profile.reviewed_preregistration_sha256,
        "reviewed_design_sha256": profile.reviewed_design_sha256,
        "scientific_config_sha256": profile.scientific_config_sha256,
        "database_path": FROZEN_CONFIG.database_path, "database_sha256": FROZEN_CONFIG.database_sha256,
        "database_access": "unevaluated_literal_only", "source_snapshot_sha256": sha256(source_bytes).hexdigest(),
        "source": dict(source), "selection": dict(selection), "backend": backend.to_payload(),
        "label_access_count": 0,
    }
    payload.update(_GOVERNANCE_ANTIGRAVITY)
    _validate_readiness_payload_antigravity(payload, profile)
    return payload


def _validate_readiness_payload_antigravity(
    payload: object,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> tuple[dict[str, object], BackendAssessmentAntigravity]:
    if not isinstance(payload, dict) or set(payload) != _READINESS_KEYS_ANTIGRAVITY:
        raise ValueError("readiness payload schema is invalid")
    if (payload["schema"], payload["experiment_id"], payload["run_identity"], payload["database_path"],
            payload["database_sha256"], payload["database_access"], payload["label_access_count"]) != (
                profile.schema_antigravity("readiness"), profile.experiment_id, profile.run_identity,
                FROZEN_CONFIG.database_path, FROZEN_CONFIG.database_sha256, "unevaluated_literal_only", 0):
        raise ValueError("readiness frozen literals differ")
    if (
        payload["reviewed_preregistration_sha256"], payload["reviewed_design_sha256"],
        payload["scientific_config_sha256"],
    ) != (
        profile.reviewed_preregistration_sha256, profile.reviewed_design_sha256,
        profile.scientific_config_sha256,
    ):
        raise ValueError("readiness lifecycle profile binding differs")
    if not _lower_hex_antigravity(payload["reviewed_commit"], 40) or not _lower_hex_antigravity(payload["source_snapshot_sha256"], 64):
        raise ValueError("readiness identity is invalid")
    governance = {name: payload[name] for name in _GOVERNANCE_ANTIGRAVITY}
    if governance != _GOVERNANCE_ANTIGRAVITY or any(type(governance[name]) is not type(value) for name, value in _GOVERNANCE_ANTIGRAVITY.items()):
        raise ValueError("readiness governance is invalid")
    source, selection = payload["source"], payload["selection"]
    if not isinstance(source, dict) or not isinstance(selection, dict):
        raise ValueError("readiness replay evidence is invalid")
    source_expected = {"context_dataset_count", "record_digest_count", "hard_coordinate_count", "provisional_candidate_count", "label_access_count", "source_hashes"}
    selection_expected = {"provisional_candidate_count", "outside_hard_candidate_count", "eligible_candidate_count", "winner_count", "selected_candidate_count"}
    if set(source) != source_expected or set(selection) != selection_expected:
        raise ValueError("readiness replay evidence schema is invalid")
    if any(type(source[name]) is not int for name in (
        "context_dataset_count", "record_digest_count", "hard_coordinate_count",
        "provisional_candidate_count", "label_access_count",
    )) or any(type(selection[name]) is not int for name in selection_expected):
        raise ValueError("readiness replay evidence types are invalid")
    if (source["context_dataset_count"], source["record_digest_count"], source["hard_coordinate_count"], source["provisional_candidate_count"], source["label_access_count"]) != (1117, 1117, 2218, 840, 0):
        raise ValueError("readiness source counts differ")
    if selection != {"provisional_candidate_count": 840, "outside_hard_candidate_count": 394, "eligible_candidate_count": 73, "winner_count": 49, "selected_candidate_count": 30}:
        raise ValueError("readiness selection counts differ")
    expected_hashes = {
        "context": FROZEN_CONFIG.checkpoint_context_sha256, "state": FROZEN_CONFIG.checkpoint_state_sha256,
        "retained_inventory": FROZEN_CONFIG.checkpoint_retained_inventory_sha256,
        "publication_prediction": FROZEN_CONFIG.publication_prediction_sha256,
        "publication_manifest": FROZEN_CONFIG.publication_manifest_sha256,
        "publication_artifact_map": FROZEN_CONFIG.publication_artifact_map_sha256,
        "hard_prediction": FROZEN_CONFIG.hard_prediction_sha256, "hard_manifest": FROZEN_CONFIG.hard_manifest_sha256,
        "hard_coordinate_identity": FROZEN_CONFIG.hard_coordinate_identity_sha256,
    }
    if source["source_hashes"] != expected_hashes:
        raise ValueError("readiness source hashes differ")
    if sha256(canonical_json_bytes_antigravity(source)).hexdigest() != payload["source_snapshot_sha256"]:
        raise ValueError("readiness source byte identity differs")
    backend = BackendAssessmentAntigravity.from_payload(payload["backend"])
    if backend.workers != 11:
        raise ValueError("readiness worker count differs")
    return payload, backend


def _failure_payload_antigravity(
    reviewed_commit: str, error: BaseException,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": profile.schema_antigravity("readiness_failure"), "reviewed_commit": reviewed_commit,
        "error": str(error) or type(error).__name__,
    }
    payload.update(_GOVERNANCE_ANTIGRAVITY)
    if (set(payload) != {"schema", "reviewed_commit", "error"} | set(_GOVERNANCE_ANTIGRAVITY) or
            payload["schema"] != profile.schema_antigravity("readiness_failure") or not _lower_hex_antigravity(payload["reviewed_commit"], 40) or
            not isinstance(payload["error"], str) or not payload["error"]):
        raise ValueError("readiness failure evidence is invalid")
    return payload


def _failed_sibling_antigravity(parent: Path, run_root: Path | None = None) -> Path:
    root = _RUN_ROOT_ANTIGRAVITY if run_root is None else Path(run_root)
    return parent / f".{root.name}_failed_{uuid4().hex}_antigravity"


def _write_failure_evidence_antigravity(
    root: Path, reviewed_commit: str, error: BaseException,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
) -> None:
    write_create_only_antigravity(root / "failure_antigravity.json", canonical_json_bytes_antigravity(_failure_payload_antigravity(reviewed_commit, error, profile)))
    fsync_directory_antigravity(root)


def _preserve_failed_staging_antigravity(
    staging: Path, reviewed_commit: str, error: BaseException,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
    run_root: Path | None = None,
) -> None:
    """Durably retain a private pre-rename failure without replacing any winner."""
    _write_failure_evidence_antigravity(staging, reviewed_commit, error, profile)
    failed = _failed_sibling_antigravity(staging.parent, run_root)
    os.rename(staging, failed)
    fsync_directory_antigravity(failed.parent)


def _directory_identity_antigravity(path: Path) -> tuple[int, int, int]:
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode) or stat.S_IMODE(status.st_mode) != 0o700:
        raise ValueError("owned readiness parent is invalid")
    return status.st_dev, status.st_ino, status.st_mode


def _rollback_owned_parent_antigravity(
    identity: tuple[int, int, int], reviewed_commit: str, error: BaseException,
    profile: RecoveryLifecycleProfileAntigravity = EXP441_PROFILE_ANTIGRAVITY,
    run_root: Path | None = None,
) -> None:
    """Undo only our provisional post-rename parent through a no-clobber failed rename."""
    root = _RUN_ROOT_ANTIGRAVITY if run_root is None else Path(run_root)
    if root.is_symlink() or _directory_identity_antigravity(root) != identity:
        raise ValueError("readiness parent changed before rollback")
    failed = _failed_sibling_antigravity(root.parent, root)
    exclusive_rename_antigravity(root, failed)
    _write_failure_evidence_antigravity(failed, reviewed_commit, error, profile)
    fsync_directory_antigravity(failed.parent)
    if root.exists() or root.is_symlink():
        raise ValueError("readiness rollback left the fixed parent present")


def _run_readiness_at_root_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, reviewed_commit: str, root: Path,
) -> ReadinessBindingAntigravity:
    """Internal test seam beneath the fixed production-profile root boundary."""
    reviewed_commit = _require_reviewed_head_antigravity(reviewed_commit)
    root = Path(root)
    _require_absent_output_roots_antigravity(root, profile)
    staging = root.with_name(f".{root.name}_staging_{uuid4().hex}_antigravity")
    owned_identity: tuple[int, int, int] | None = None
    try:
        staging.mkdir(mode=0o700)
        snapshot = verify_frozen_sources_antigravity()
        source = _source_payload_antigravity(snapshot)
        selection = _selection_payload_antigravity(snapshot)
        backend = _backend_antigravity()
        _require_reviewed_head_antigravity(reviewed_commit)
        payload = _readiness_payload_antigravity(reviewed_commit, source, selection, backend, profile)
        encoded = canonical_json_bytes_antigravity(payload)
        write_create_only_antigravity(staging / _READINESS_NAME_ANTIGRAVITY, encoded)
        fsync_directory_antigravity(staging)
        with pin_exact_regular_inventory_antigravity(staging, (_READINESS_NAME_ANTIGRAVITY,)) as pinned:
            if root.exists() or root.is_symlink():
                raise ValueError(f"{profile.experiment_id} readiness ownership raced")
            pre_rename_identity = (pinned._root_metadata_antigravity.device, pinned._root_metadata_antigravity.inode, pinned._root_metadata_antigravity.mode)
            exclusive_rename_antigravity(staging, root)
            owned_identity = pre_rename_identity
            pinned.revalidate_at_antigravity(root, renamed_antigravity=True)
            fsync_directory_antigravity(root.parent)
            binding = ReadinessBindingAntigravity(reviewed_commit, str(payload["source_snapshot_sha256"]), backend, sha256(encoded).hexdigest())
            pinned.revalidate_at_antigravity(root, renamed_antigravity=True)
            return binding
    except BaseException as error:
        try:
            if owned_identity is not None:
                _rollback_owned_parent_antigravity(owned_identity, reviewed_commit, error, profile, root)
            elif staging.exists():
                _preserve_failed_staging_antigravity(staging, reviewed_commit, error, profile, root)
        except BaseException as preservation_error:
            error.add_note(f"readiness failure preservation also failed: {type(preservation_error).__name__}: {preservation_error}")
            raise error from preservation_error
        raise


def _run_readiness_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, reviewed_commit: str,
) -> ReadinessBindingAntigravity:
    """Replay label-free authority at the sole root bound into the closed profile."""
    return _run_readiness_at_root_antigravity(
        profile, reviewed_commit, profile.root_antigravity,
    )


def _validate_readiness_at_root_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, reviewed_commit: str, root: Path,
) -> ReadinessBindingAntigravity:
    """Internal test seam beneath the fixed production-profile root boundary."""
    reviewed_commit = _require_reviewed_head_antigravity(reviewed_commit)
    root = Path(root)
    artifact = root / _READINESS_NAME_ANTIGRAVITY
    with pin_exact_regular_inventory_antigravity(root, (_READINESS_NAME_ANTIGRAVITY,)) as pinned:
        encoded = pinned.payloads[_READINESS_NAME_ANTIGRAVITY]
        parsed = parse_canonical_json_bytes_antigravity(encoded)
        payload, backend = _validate_readiness_payload_antigravity(parsed, profile)
        if canonical_json_bytes_antigravity(payload) != encoded or payload["reviewed_commit"] != reviewed_commit:
            raise ValueError("readiness bytes or reviewed commit differ")
        snapshot = verify_frozen_sources_antigravity()
        source = _source_payload_antigravity(snapshot)
        selection = _selection_payload_antigravity(snapshot)
        expected = _readiness_payload_antigravity(reviewed_commit, source, selection, backend, profile)
        if canonical_json_bytes_antigravity(expected) != encoded:
            raise ValueError("readiness source replay differs")
        if _backend_antigravity().to_payload() != backend.to_payload():
            raise ValueError("readiness worker or environment differs")
        binding = ReadinessBindingAntigravity(reviewed_commit, str(payload["source_snapshot_sha256"]), backend, sha256(encoded).hexdigest())
        pinned.revalidate_antigravity()
        return binding


def _validate_readiness_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity, reviewed_commit: str,
) -> ReadinessBindingAntigravity:
    """Recompute readiness only at the sole root bound into the closed profile."""
    return _validate_readiness_at_root_antigravity(
        profile, reviewed_commit, profile.root_antigravity,
    )


def run_readiness_antigravity(reviewed_commit: str) -> ReadinessBindingAntigravity:
    """Run the original fixed Exp441 readiness lifecycle."""
    return _run_readiness_at_root_antigravity(
        EXP441_PROFILE_ANTIGRAVITY, reviewed_commit, _RUN_ROOT_ANTIGRAVITY,
    )


def validate_readiness_for_run_antigravity(reviewed_commit: str) -> ReadinessBindingAntigravity:
    """Validate the original fixed Exp441 readiness lifecycle."""
    return _validate_readiness_at_root_antigravity(
        EXP441_PROFILE_ANTIGRAVITY, reviewed_commit, _RUN_ROOT_ANTIGRAVITY,
    )


__all__ = [
    "run_readiness_antigravity", "validate_readiness_for_run_antigravity",
    "current_clean_head_antigravity",
]
