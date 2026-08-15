"""Exp449 parent-only two-wave recovery and result-neutral operations records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import argparse
import ast
import fcntl
import importlib.metadata
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from time import monotonic, time

from paper.equipment_anomaly_detection.artifacts_paper import (
    Task4Task5SuccessAuthority,
    canonical_json_bytes,
    read_canonical,
    verify_task4_task5_attempt_namespace,
    verify_task4_task5_success_namespace,
    write_create_only,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
from paper.equipment_anomaly_detection.acceleration_paper import (
    build_acceleration_manifest,
    verify_acceleration_manifest,
)
from paper.equipment_anomaly_detection.catalog_paper import build_catalog, verify_catalog
from paper.equipment_anomaly_detection.data_paper import load_dataset_values
from paper.equipment_anomaly_detection.config_paper import (
    FROZEN_CONFIG as FROZEN_PAPER_CONFIG,
    LINEAGE_LOCK_PATH,
)
from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
    verify_pycatch22_environment_paper,
)
from paper.equipment_anomaly_detection.sources_paper.multirocket_hydra_paper import (
    verify_exp84_environment_paper,
)

from .config_antigravity import FROZEN_EXP449_CONFIG, FROZEN_EXP449_POLICY_KEYS, SMOKE_A_CLAIM_PATH_ANTIGRAVITY
from .contracts_antigravity import Exp449Context, Exp449ProposalManifest, Exp449TerminalStatus
from .execution_authority_antigravity import (
    Exp449SmokeAGrantAntigravity,
    VerifiedExp449SmokeAAuthorityAntigravity,
    _execution_capability_claim_fd_antigravity,
    _mint_execution_capability_antigravity,
    _require_execution_capability_antigravity,
    _revoke_execution_capability_antigravity,
    verify_smoke_a_authorization_commit_antigravity,
)
from .artifacts_antigravity import (
    context_sha256_antigravity,
    freeze_exp449_proposal_antigravity,
)


compose_exp449_proposal_score_antigravity = None


_TERMINAL_COMPLETE_NAME = "terminal_complete_antigravity.json"
_TERMINAL_INCOMPLETE_NAME = "terminal_incomplete_antigravity.json"
_PROGRESS_NAME = "progress_antigravity.jsonl"
_STATUS_NAME = "status_antigravity.json"
_WORKER_STDOUT_NAME = "worker_stdout_antigravity.log"
_WORKER_STDERR_NAME = "worker_stderr_antigravity.log"
_SMOKE_COMPARISON_NAME = "smoke_comparison_antigravity.json"
_EXECUTION_AUTHORITY_CONSUMED_NAME = "execution_authority_consumed_antigravity.json"
_TERMINAL_SCHEMA = "exp449-proposal-score-incomplete-antigravity.v1"
_PROGRESS_KEYS = frozenset(
    {
        "schema",
        "timestamp",
        "event",
        "wave",
        "dataset",
        "completed",
        "failed",
        "pending",
        "elapsed_seconds",
        "throughput_per_second",
        "eta_seconds",
        "latest_error",
        "worker_pid",
    }
)
_REQUIRED_PROGRESS_KEYS = _PROGRESS_KEYS
_FORBIDDEN_PROGRESS_KEYS = frozenset({
    "label", "labels", "raw", "raw_value", "raw_values", "score", "scores",
    "p_value", "p_values", "rank", "ranks", "coordinate", "coordinates",
    "trace", "traceback", "exception_trace", "exception_traceback",
})
_SMOKE_OPERATIONS_ROOT_PAIRS_ANTIGRAVITY = {
    (
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root.resolve(strict=False),
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root.resolve(strict=False),
    ): (
        FROZEN_EXP449_CONFIG.smoke_a_operations_root.resolve(strict=False),
        FROZEN_EXP449_CONFIG.smoke_b_operations_root.resolve(strict=False),
    ),
}
_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_CAPABLE_INTERPRETER_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python"
)
_LABEL_MODULE_PREFIXES_ANTIGRAVITY = (
    "paper.equipment_anomaly_detection.label_access_paper",
    "paper.equipment_anomaly_detection.labels_paper",
)
# These are the formerly authorized Exp448/R4 execution destinations recorded
# before Exp449.  They are path authorities, not spelling fragments: an
# unrelated temporary directory may quite legitimately contain e.g. ``r4``.
# Inspection deliberately compares raw lexical paths and lstat's each component
# before any resolving operation, so a symlink cannot turn a benign spelling
# into an old authority between checks.
_HISTORICAL_ROOTS_ANTIGRAVITY = (
    Path("/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_v2_antigravity_20260814_ops_antigravity"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813"),
    Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v3_antigravity_20260814"),
)
_HISTORICAL_ROOT_COMPONENTS_ANTIGRAVITY = frozenset({"exp448", "r4", "historical"})
_EXPECTED_PREPROCESSING_CONTRACTS_ANTIGRAVITY = (
    "task4_train_only",
    "task5_label_free_train_test_value_length_transductive",
)
_NORMALIZED_PROJECT_NAME_ANTIGRAVITY = re.compile(r"[-_.]+")
_AUTHORIZATION_DELTA_READ_ISSUER_ANTIGRAVITY = object()
_FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/validation_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py",
    "paper/equipment_anomaly_detection/acceleration_paper.py",
    "paper/equipment_anomaly_detection/artifacts_paper.py",
    "paper/equipment_anomaly_detection/calibration_paper.py",
    "paper/equipment_anomaly_detection/catalog_paper.py",
    "paper/equipment_anomaly_detection/config_paper.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/hard_route_paper.py",
    "paper/equipment_anomaly_detection/lineage_paper.py",
    "paper/equipment_anomaly_detection/proposal_score_paper.py",
    "paper/equipment_anomaly_detection/routing_paper.py",
    "paper/equipment_anomaly_detection/score_adapter_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/canonical_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/knn_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
)


class _Task45ContextPayloadAntigravity(dict[str, object]):
    """Paper-compatible dict retaining the frozen mode for Task3's gate."""

    __slots__ = ("mode",)

    def __init__(self, payload: Mapping[str, object], mode: str) -> None:
        super().__init__(payload)
        self.mode = mode


@dataclass(frozen=True, slots=True)
class Exp449WaveConfig:
    """Verified inputs for one fixed-root Exp449 recovery lifecycle.

    The configuration deliberately carries data and authorities only.  Scoring,
    record verification, and success publication remain the existing paper
    boundaries; callers cannot replace them with executable adapters.
    """

    context: Exp449Context
    catalog: dict[str, object]
    raw_datasets: Mapping[str, object]
    acceleration_manifest: object

    def __post_init__(self) -> None:
        if not isinstance(self.context, Exp449Context) or not isinstance(self.catalog, dict):
            raise ValueError("Exp449 recovery context is invalid")
        if self.catalog.get("catalog_sha256") != self.context.catalog_sha256:
            raise ValueError("Exp449 recovery catalog identity differs")
        datasets = _catalog_datasets_antigravity(self.catalog)
        raw = dict(self.raw_datasets)
        if tuple(raw) != datasets:
            raise ValueError("Exp449 recovery raw identity differs")
        object.__setattr__(self, "raw_datasets", raw)


@dataclass(frozen=True, slots=True)
class Exp449AuthorizedWaveResultAntigravity:
    """Verified dataset-local state plus the only score eligible for a freeze.

    The score remains opaque here: the scientific composer exclusively defines
    its type and fields.  Incomplete collection must never hand a partial
    scientific object to an authorized publisher.
    """

    status: Exp449TerminalStatus
    score: object | None


@dataclass(slots=True)
class Exp449SmokeAExecutionLeaseAntigravity:
    """Own the one process-local claim descriptor for a fresh Smoke A launch."""

    verified_authority: VerifiedExp449SmokeAAuthorityAntigravity
    context: Exp449Context
    claim_fd: int
    capability: object

    def __enter__(self) -> "Exp449SmokeAExecutionLeaseAntigravity":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the descriptor only; the consumed claim remains permanent."""
        if self.claim_fd < 0:
            return
        _revoke_execution_capability_antigravity(self.capability)
        try:
            fcntl.flock(self.claim_fd, fcntl.LOCK_UN)
        finally:
            os.close(self.claim_fd)
            self.claim_fd = -1


@dataclass(frozen=True, slots=True)
class _RecoverySnapshot:
    successes: tuple[str, ...]
    attempts: tuple[dict[str, object], ...]
    failed: tuple[str, ...]
    never_attempted: tuple[str, ...]
    wave_count: int


def _json_plain_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_plain_antigravity(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_plain_antigravity(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def task45_context_payload_antigravity(context: Exp449Context) -> dict[str, object]:
    """Make the sole deterministic bridge to the paper checkpoint context.

    It serializes the already-frozen Exp449 context, then hashes exactly the
    same canonical JSON bytes required by the paper Task4/5 authority.  This is
    a bridge, not a second provenance contract.
    """
    if not isinstance(context, Exp449Context):
        raise TypeError("Exp449 context is required")
    payload: dict[str, object] = {
        "experiment_id": context.experiment_id,
        "mode": context.mode,
        "run_root": str(context.run_root),
        "database_sha256": context.database_sha256,
        "catalog_sha256": context.catalog_sha256,
        "code_closure": _json_plain_antigravity(context.code_closure),
        "lineage_sha256": context.lineage_sha256,
        "codemap_sha256": context.codemap_sha256,
        "interpreter": context.interpreter,
        "environment": _json_plain_antigravity(context.environment),
        "backend_record": _json_plain_antigravity(context.backend_record),
        "seeds_and_constants": _json_plain_antigravity(context.seeds_and_constants),
        "historical_parity_status": context.historical_parity_status,
    }
    payload["context_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return _Task45ContextPayloadAntigravity(payload, context.mode)


def _catalog_datasets_antigravity(catalog: dict[str, object]) -> tuple[str, ...]:
    entries = catalog.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Exp449 recovery catalog is invalid")
    names = tuple(entry.get("dataset") if isinstance(entry, dict) else None for entry in entries)
    if not names or any(type(name) is not str or not name for name in names) or names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError("Exp449 recovery catalog is invalid")
    expected = FROZEN_EXP449_CONFIG.smoke_datasets if len(names) == len(FROZEN_EXP449_CONFIG.smoke_datasets) else None
    if expected is not None and names != expected:
        raise ValueError("Exp449 smoke recovery catalog differs")
    if len(names) == FROZEN_EXP449_CONFIG.dataset_count:
        return names
    if expected is None:
        raise ValueError("Exp449 recovery catalog count differs")
    return names


def _operations_root_for_context_antigravity(context: Exp449Context) -> Path:
    roots = {
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root: FROZEN_EXP449_CONFIG.smoke_a_operations_root,
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root: FROZEN_EXP449_CONFIG.smoke_b_operations_root,
        FROZEN_EXP449_CONFIG.official_scientific_root: FROZEN_EXP449_CONFIG.official_operations_root,
    }
    try:
        root = roots[Path(context.run_root)]
    except KeyError as error:
        raise ValueError("Exp449 scientific root is not an operations authority") from error
    if root == context.run_root or context.run_root in root.parents or root in context.run_root.parents:
        raise ValueError("Exp449 operations root must be separate from scientific root")
    return root


def _lstat_absent_path_antigravity(path: Path, *, name: str) -> None:
    """Reject traversal, symlink components, and every pre-existing destination."""
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"Exp449 {name} path is unsafe")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError(f"Exp449 {name} path is unsafe")
        if current != raw and not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"Exp449 {name} parent is unsafe")
        if current == raw:
            raise ValueError(f"Exp449 {name} must be absent")


def _write_fd_all_antigravity(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("Exp449 create-only write returned no bytes")
        offset += written
    os.fsync(descriptor)


def _read_canonical_fd_mapping_antigravity(descriptor: int) -> dict[str, object]:
    """Read one already-open regular record without a second pathname lookup."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(descriptor, 65536)
        except InterruptedError:
            continue
        if not chunk:
            break
        chunks.append(chunk)
    data = b"".join(chunks)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp449 persisted authority record is invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise ValueError("Exp449 persisted authority record is noncanonical")
    return value


def _create_execution_receipt_antigravity(
    root: Path,
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    context: Exp449Context,
    claim_path: Path,
    claim_details: os.stat_result,
) -> None:
    """Create the authenticated one-shot consumption record without root healing."""
    receipt = root / _EXECUTION_AUTHORITY_CONSUMED_NAME
    try:
        details = receipt.lstat()
    except FileNotFoundError:
        details = None
    if details is not None:
        raise ValueError("Exp449 execution authority consumption already exists")
    payload = _receipt_payload_antigravity(verified, context, claim_path, claim_details)
    descriptor = os.open(
        receipt,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_fd_all_antigravity(descriptor, canonical_json_bytes(payload))
    finally:
        os.close(descriptor)
    directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _claim_payload_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    context: Exp449Context,
) -> dict[str, object]:
    return {
        "schema": "exp449-smoke-a-execution-claim-antigravity.v1",
        "grant_sha256": verified.grant_sha256,
        "approved_authority_commit": verified.approved_authority_commit,
        "implementation_commit": verified.implementation_commit,
        "target": verified.grant.target,
        "context_sha256": context_sha256_antigravity(context),
    }


def _receipt_payload_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    context: Exp449Context,
    claim_path: Path,
    claim_details: os.stat_result,
) -> dict[str, object]:
    return {
        "schema": "exp449-smoke-a-execution-consumed-antigravity.v1",
        "grant_sha256": verified.grant_sha256,
        "approved_authority_commit": verified.approved_authority_commit,
        "implementation_commit": verified.implementation_commit,
        "target": verified.grant.target,
        "context_sha256": context_sha256_antigravity(context),
        "claim_path": str(claim_path),
        "claim_device": claim_details.st_dev,
        "claim_inode": claim_details.st_ino,
    }


def acquire_smoke_a_execution_lease_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    context: Exp449Context,
    *,
    _test_paths_antigravity: tuple[Path, Path, Path] | None = None,
) -> Exp449SmokeAExecutionLeaseAntigravity:
    """Verify, exclusively claim, consume, then mint Smoke A mutation authority.

    The optional private path triple exists only for in-memory authority tests.
    Its normal-public-call value is always the exact grant-bound fixed paths.
    """
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity or type(context) is not Exp449Context:
        raise TypeError("Exp449 verified authority and context are required")
    grant = verified.grant
    if grant.target != "smoke-a" or context.mode != "smoke":
        raise ValueError("Exp449 Smoke A lease target differs")
    if _test_paths_antigravity is None:
        scientific_root, operations_root, claim_path = (
            Path(grant.scientific_root), Path(grant.operations_root), Path(grant.claim_path),
        )
        if (
            scientific_root != FROZEN_EXP449_CONFIG.smoke_a_scientific_root
            or operations_root != FROZEN_EXP449_CONFIG.smoke_a_operations_root
        ):
            raise ValueError("Exp449 Smoke A fixed roots differ")
    else:
        if len(_test_paths_antigravity) != 3:
            raise ValueError("Exp449 test path seam is invalid")
        scientific_root, operations_root, claim_path = map(Path, _test_paths_antigravity)
        if (
            scientific_root != Path(grant.scientific_root)
            or operations_root != Path(grant.operations_root)
            or claim_path != Path(grant.claim_path)
        ):
            raise ValueError("Exp449 test path seam binding differs")
    if Path(context.run_root) != scientific_root or context_sha256_antigravity(context) != grant.context_sha256:
        raise ValueError("Exp449 Smoke A context binding differs")
    # Readiness is deliberately complete before the first claim/root mutation.
    readiness_for_authorized_implementation_antigravity(verified)
    _lstat_absent_path_antigravity(scientific_root, name="scientific root")
    _lstat_absent_path_antigravity(operations_root, name="operations root")
    _lstat_absent_path_antigravity(claim_path, name="claim")
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError as error:
            raise ValueError("Exp449 execution claim already exists") from error
        claim_payload = _claim_payload_antigravity(verified, context)
        _write_fd_all_antigravity(descriptor, canonical_json_bytes(claim_payload))
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim_details = os.fstat(descriptor)
        path_details = claim_path.lstat()
        if (
            not stat.S_ISREG(claim_details.st_mode) or stat.S_ISLNK(path_details.st_mode)
            or not stat.S_ISREG(path_details.st_mode)
            or (claim_details.st_dev, claim_details.st_ino) != (path_details.st_dev, path_details.st_ino)
        ):
            raise ValueError("Exp449 execution claim inode differs")
        os.mkdir(operations_root, 0o700)
        root_details = operations_root.lstat()
        if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
            raise ValueError("Exp449 operations root is unsafe")
        _create_execution_receipt_antigravity(operations_root, verified, context, claim_path, claim_details)
        capability = _mint_execution_capability_antigravity(
            verified, context=context, target="smoke-a", scientific_root=scientific_root,
            operations_root=operations_root, claim_path=claim_path, claim_fd=descriptor,
        )
        return Exp449SmokeAExecutionLeaseAntigravity(verified, context, descriptor, capability)
    except BaseException:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        raise


def _acquire_smoke_a_resume_lease_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    config: Exp449WaveConfig,
) -> Exp449SmokeAExecutionLeaseAntigravity:
    """Reclaim only the exact inactive claim and consumed-root identity.

    This is intentionally private: callers receive no root or claim override,
    and a mismatched persisted namespace is an error rather than something the
    lifecycle can heal or migrate.
    """
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
        raise TypeError("Exp449 verified Smoke A authority is required")
    context = config.context
    grant = verified.grant
    scientific_root, operations_root, claim_path = (
        Path(grant.scientific_root), Path(grant.operations_root), Path(grant.claim_path),
    )
    if (
        scientific_root != FROZEN_EXP449_CONFIG.smoke_a_scientific_root
        or operations_root != FROZEN_EXP449_CONFIG.smoke_a_operations_root
        or claim_path != SMOKE_A_CLAIM_PATH_ANTIGRAVITY
        or context.run_root != scientific_root
    ):
        raise ValueError("Exp449 Smoke A resume roots differ")
    readiness_for_authorized_resume_antigravity(verified)
    for root, name in ((scientific_root, "scientific root"), (operations_root, "operations root")):
        details = root.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"Exp449 Smoke A resume {name} is unsafe")
    descriptor = -1
    try:
        descriptor = os.open(claim_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        details = os.fstat(descriptor)
        path_details = claim_path.lstat()
        if (
            not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(path_details.st_mode)
            or not stat.S_ISREG(path_details.st_mode)
            or (details.st_dev, details.st_ino) != (path_details.st_dev, path_details.st_ino)
            or _read_canonical_fd_mapping_antigravity(descriptor) != _claim_payload_antigravity(verified, context)
        ):
            raise ValueError("Exp449 Smoke A resume claim differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("Exp449 Smoke A resume claim is active") from error
        receipt = operations_root / _EXECUTION_AUTHORITY_CONSUMED_NAME
        receipt_fd = os.open(receipt, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            receipt_details = os.fstat(receipt_fd)
            receipt_path_details = receipt.lstat()
            if (
                stat.S_ISLNK(receipt_path_details.st_mode) or not stat.S_ISREG(receipt_details.st_mode)
                or (receipt_details.st_dev, receipt_details.st_ino) != (receipt_path_details.st_dev, receipt_path_details.st_ino)
                or _read_canonical_fd_mapping_antigravity(receipt_fd) != _receipt_payload_antigravity(verified, context, claim_path, details)
            ):
                raise ValueError("Exp449 Smoke A consumption receipt differs")
        finally:
            os.close(receipt_fd)
        capability = _mint_execution_capability_antigravity(
            verified, context=context, target="smoke-a", scientific_root=scientific_root,
            operations_root=operations_root, claim_path=claim_path, claim_fd=descriptor,
        )
        return Exp449SmokeAExecutionLeaseAntigravity(verified, context, descriptor, capability)
    except BaseException:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        raise


def _execution_capability_for_config_antigravity(config: Exp449WaveConfig, capability: object) -> None:
    """Gate only configured publication destinations; fixture composition stays pure."""
    configured = {
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root: "smoke-a",
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root: "smoke-b",
        FROZEN_EXP449_CONFIG.official_scientific_root: "official",
    }
    target = configured.get(Path(config.context.run_root))
    if target is None:
        return
    claim_fd = _execution_capability_claim_fd_antigravity(capability)
    _require_execution_capability_antigravity(
        capability, target=target, context=config.context, claim_fd=claim_fd,
    )


def _authority_antigravity(config: Exp449WaveConfig) -> tuple[dict[str, object], Task4Task5SuccessAuthority]:
    context_payload = task45_context_payload_antigravity(config.context)
    authority = Task4Task5SuccessAuthority.from_verified_context(
        context_payload, config.catalog, config.acceleration_manifest
    )
    return context_payload, authority


def _attempt_tuple_antigravity(record: object) -> tuple[str, int, int, str, str]:
    dataset = getattr(record, "dataset", None)
    wave = getattr(record, "wave", None)
    attempt = getattr(record, "attempt", None)
    exception_type = getattr(record, "exception_type", None)
    exception_message = getattr(record, "exception_message", None)
    if type(dataset) is not str or not dataset or type(wave) is not int or wave not in {1, 2} or type(attempt) is not int or attempt < 1 or type(exception_type) is not str or not exception_type or type(exception_message) is not str:
        raise ValueError("Exp449 verified attempt record differs")
    return dataset, wave, attempt, exception_type, exception_message


def _read_verified_attempt_payloads_antigravity(config: Exp449WaveConfig, attempts: tuple[tuple[str, int, int, str, str], ...]) -> tuple[dict[str, object], ...]:
    """Bind the terminal ledger to every already-verified on-disk record byte."""
    if not attempts:
        return ()
    root = config.context.run_root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_attempts"
    try:
        paths = sorted(root.iterdir())
    except FileNotFoundError as error:
        raise ValueError("Exp449 verified attempt ledger differs") from error
    expected_names = [f"attempt_{index:06d}_antigravity.json" for index in range(1, len(attempts) + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("Exp449 verified attempt ledger differs")
    payloads: list[dict[str, object]] = []
    for expected, path in zip(attempts, paths, strict=True):
        value = read_canonical(path)
        if not isinstance(value, dict):
            raise ValueError("Exp449 verified attempt ledger differs")
        dataset, wave, attempt, exception_type, exception_message = expected
        if (value.get("dataset"), value.get("wave"), value.get("attempt"), value.get("exception_type"), value.get("exception_message")) != expected or type(value.get("record_sha256")) is not str:
            raise ValueError("Exp449 verified attempt ledger differs")
        payloads.append(dict(value))
    return tuple(payloads)


def _snapshot_antigravity(config: Exp449WaveConfig) -> _RecoverySnapshot:
    """Reopen the exact paper namespaces before every status or scheduling step."""
    _context_payload, authority = _authority_antigravity(config)
    names = _catalog_datasets_antigravity(config.catalog)
    verified = tuple(verify_task4_task5_success_namespace(
        config.context.run_root, config.catalog, authority, dict(config.raw_datasets)
    ))
    if verified != tuple(sorted(set(verified))) or any(dataset not in names for dataset in verified):
        raise ValueError("Exp449 verified success namespace differs")
    attempt_tuples = tuple(_attempt_tuple_antigravity(record) for record in verify_task4_task5_attempt_namespace(
        config.context.run_root, config.catalog, authority, dict(config.raw_datasets)
    ))
    if tuple(attempt for _dataset, _wave, attempt, _type, _message in attempt_tuples) != tuple(range(1, len(attempt_tuples) + 1)):
        raise ValueError("Exp449 verified attempt identities differ")
    if len({(dataset, wave) for dataset, wave, _attempt, _type, _message in attempt_tuples}) != len(attempt_tuples):
        raise ValueError("Exp449 verified attempt identities differ")
    global_waves = [wave for _dataset, wave, _attempt, _type, _message in attempt_tuples]
    if global_waves and (global_waves[0] != 1 or global_waves != sorted(global_waves)):
        raise ValueError("Exp449 verified attempt identities differ")
    per_dataset_waves: dict[str, list[int]] = {}
    for dataset, wave, _attempt, _type, _message in attempt_tuples:
        per_dataset_waves.setdefault(dataset, []).append(wave)
    if any(waves != sorted(waves) for waves in per_dataset_waves.values()):
        raise ValueError("Exp449 verified attempt identities differ")
    if any(dataset not in names for dataset, _wave, _attempt, _type, _message in attempt_tuples):
        raise ValueError("Exp449 verified attempt namespace differs")
    attempts = _read_verified_attempt_payloads_antigravity(config, attempt_tuples)
    maximum_wave = max((wave for _dataset, wave, _attempt, _type, _message in attempt_tuples), default=0)
    pending = tuple(dataset for dataset in names if dataset not in set(verified))
    attempted_pending = {dataset for dataset, _wave, _attempt, _type, _message in attempt_tuples if dataset in pending}
    failed = tuple(dataset for dataset in pending if dataset in attempted_pending)
    never_attempted = tuple(dataset for dataset in pending if dataset not in attempted_pending)
    return _RecoverySnapshot(verified, attempts, failed, never_attempted, maximum_wave)


def _terminal_status_antigravity(config: Exp449WaveConfig, snapshot: _RecoverySnapshot) -> Exp449TerminalStatus:
    names = _catalog_datasets_antigravity(config.catalog)
    if len(snapshot.successes) == len(names):
        # A success-only first wave has no attempt ledger entry.  After a retry,
        # the highest failed wave plus one is the completed terminal wave.
        wave_count = min(FROZEN_EXP449_CONFIG.maximum_waves, max(1, snapshot.wave_count + 1))
        return Exp449TerminalStatus(config.context.mode, "dataset_local_complete", len(names), 0, 0, wave_count)
    return Exp449TerminalStatus(
        config.context.mode,
        "proposal_score_incomplete",
        len(snapshot.successes),
        len(snapshot.failed),
        len(snapshot.never_attempted),
        min(snapshot.wave_count, FROZEN_EXP449_CONFIG.maximum_waves),
    )


def _terminal_record_antigravity(config: Exp449WaveConfig, snapshot: _RecoverySnapshot) -> dict[str, object]:
    context_payload = task45_context_payload_antigravity(config.context)
    ledger = [
        _json_plain_antigravity(attempt)
        for attempt in snapshot.attempts
    ]
    payload: dict[str, object] = {
        "schema": _TERMINAL_SCHEMA,
        "context_sha256": context_payload["context_sha256"],
        "catalog_sha256": config.context.catalog_sha256,
        "verified_successes": list(snapshot.successes),
        "failed": list(snapshot.failed),
        "never_attempted": list(snapshot.never_attempted),
        "attempt_ledger_sha256": sha256(canonical_json_bytes(ledger)).hexdigest(),
    }
    payload["record_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def verify_terminal_incomplete_antigravity(
    config: Exp449WaveConfig,
    *,
    _execution_capability: object = None,
) -> Exp449TerminalStatus:
    """Create or byte-verify the only allowed terminal incomplete record."""
    _execution_capability_for_config_antigravity(config, _execution_capability)
    snapshot = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, snapshot)
    if status.status != "proposal_score_incomplete" or snapshot.wave_count < FROZEN_EXP449_CONFIG.maximum_waves:
        raise ValueError("Exp449 terminal incomplete state is not established")
    root = _operations_root_for_context_antigravity(config.context)
    write_create_only(root / _TERMINAL_INCOMPLETE_NAME, canonical_json_bytes(_terminal_record_antigravity(config, snapshot)), reuse_identical=True)
    return status


def _publish_terminal_complete_antigravity(
    config: Exp449WaveConfig,
    snapshot: _RecoverySnapshot,
    *,
    _execution_capability: object = None,
) -> Exp449TerminalStatus:
    """Publish the create-only completion counterpart after verified closure."""
    _execution_capability_for_config_antigravity(config, _execution_capability)
    status = _terminal_status_antigravity(config, snapshot)
    if status.status != "dataset_local_complete":
        raise ValueError("Exp449 terminal complete state is not established")
    root = _operations_root_for_context_antigravity(config.context)
    write_create_only(root / _TERMINAL_COMPLETE_NAME, canonical_json_bytes(_terminal_record_antigravity(config, snapshot)), reuse_identical=True)
    return status


def _existing_terminal_status_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus | None:
    """Treat any present terminal artifact as permanent, byte-verified authority."""
    root = _operations_root_for_context_antigravity(config.context)
    present: list[tuple[Path, str]] = []
    for name, expected_status in (
        (_TERMINAL_COMPLETE_NAME, "dataset_local_complete"),
        (_TERMINAL_INCOMPLETE_NAME, "proposal_score_incomplete"),
    ):
        path = root / name
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        present.append((path, expected_status))
    if not present:
        return None
    if len(present) != 1:
        raise ValueError("Exp449 terminal authority is ambiguous")
    path, expected_status = present[0]
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp449 terminal authority is unsafe")
    snapshot = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, snapshot)
    if status.status != expected_status or (
        expected_status == "proposal_score_incomplete" and snapshot.wave_count != FROZEN_EXP449_CONFIG.maximum_waves
    ):
        raise ValueError("Exp449 terminal authority differs from recovery state")
    expected = canonical_json_bytes(_terminal_record_antigravity(config, snapshot))
    if path.read_bytes() != expected:
        raise ValueError("Exp449 terminal authority differs")
    return status


def _mkdirs_without_symlinks_antigravity(directory: Path) -> None:
    directory = Path(directory)
    current = Path(directory.anchor) if directory.is_absolute() else Path.cwd()
    parts = directory.parts[1:] if directory.is_absolute() else directory.parts
    for part in parts:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            details = current.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp449 operations root is unsafe")


def _validate_progress_event_antigravity(event: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(event, Mapping):
        raise ValueError("Exp449 progress event keys differ")
    if set(event) & _FORBIDDEN_PROGRESS_KEYS:
        raise ValueError("Exp449 progress event contains forbidden scientific fields")
    if not (_REQUIRED_PROGRESS_KEYS <= set(event) <= _PROGRESS_KEYS):
        raise ValueError("Exp449 progress event keys differ")
    normalized = dict(event)
    if (
        normalized["schema"] != "exp449-operations-progress-antigravity.v1"
        or type(normalized["timestamp"]) is not str
        or not normalized["timestamp"]
        or type(normalized["event"]) is not str
        or not normalized["event"]
    ):
        raise ValueError("Exp449 progress event identity differs")
    for key in ("wave", "completed", "failed", "pending"):
        if type(normalized[key]) is not int or normalized[key] < 0:
            raise ValueError("Exp449 progress event count differs")
    for key in ("elapsed_seconds", "throughput_per_second", "eta_seconds"):
        if type(normalized[key]) not in (int, float) or not isfinite(float(normalized[key])) or float(normalized[key]) < 0.0:
            raise ValueError("Exp449 progress event timing differs")
        normalized[key] = float(normalized[key])
    if normalized["dataset"] is not None and (type(normalized["dataset"]) is not str or not normalized["dataset"]):
        raise ValueError("Exp449 progress event dataset differs")
    if normalized["latest_error"] is not None:
        latest_error = normalized["latest_error"]
        if type(latest_error) is not str or not latest_error or len(latest_error) > 256 or "\n" in latest_error or "\r" in latest_error:
            raise ValueError("Exp449 progress event error differs")
    if type(normalized["worker_pid"]) is not int or normalized["worker_pid"] < 1:
        raise ValueError("Exp449 progress event worker differs")
    return normalized


def _append_progress_event_antigravity(operations_root: Path, event: Mapping[str, object], *, create_root: bool) -> None:
    """Append and fsync one validated record; callers choose their authority level."""
    normalized = _validate_progress_event_antigravity(event)
    root = Path(operations_root)
    if create_root:
        _mkdirs_without_symlinks_antigravity(root)
    else:
        details = root.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp449 operations root is unsafe")
    path = root / _PROGRESS_NAME
    try:
        details = path.lstat()
    except FileNotFoundError:
        details = None
    if details is not None and (stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode)):
        raise ValueError("Exp449 progress log is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        line = memoryview(canonical_json_bytes(normalized))
        offset = 0
        while offset < len(line):
            try:
                written = os.write(descriptor, line[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("Exp449 progress log short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exp449_progress_event_antigravity(operations_root: Path, event: Mapping[str, object]) -> None:
    """Best-effort composition seam; official mutation uses the lease wrapper."""
    _append_progress_event_antigravity(operations_root, event, create_root=True)


def _authorized_operations_root_for_lease_antigravity(lease: Exp449SmokeAExecutionLeaseAntigravity) -> Path:
    """Recheck the active one-shot capability before every durable operation write."""
    if type(lease) is not Exp449SmokeAExecutionLeaseAntigravity or lease.claim_fd < 0:
        raise ValueError("Exp449 active Smoke A execution lease is required")
    _require_execution_capability_antigravity(
        lease.capability, target="smoke-a", context=lease.context, claim_fd=lease.claim_fd,
    )
    root = Path(lease.verified_authority.grant.operations_root)
    details = root.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp449 operations root is unsafe")
    return root


def append_authorized_progress_antigravity(
    lease: Exp449SmokeAExecutionLeaseAntigravity,
    event: Mapping[str, object],
) -> None:
    """Durably append one authorized, result-neutral progress event or raise."""
    _append_progress_event_antigravity(_authorized_operations_root_for_lease_antigravity(lease), event, create_root=False)


def _status_cache_payload_antigravity(root: Path) -> dict[str, object]:
    """Derive mutable monitoring data solely from existing operations metadata."""
    progress = root / _PROGRESS_NAME
    tail: object = None
    try:
        with progress.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 65536), os.SEEK_SET)
            lines = handle.read().splitlines()
        if lines:
            tail = _validate_progress_event_antigravity(json.loads(lines[-1].decode("utf-8")))
    except FileNotFoundError:
        pass
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        tail = {"state": "unreadable"}
    terminal = next((name for name in (_TERMINAL_COMPLETE_NAME, _TERMINAL_INCOMPLETE_NAME) if (root / name).is_file()), None)
    return {
        "schema": "exp449-derived-status-antigravity.v1",
        "timestamp": f"{time():.6f}",
        "execution_authority_consumed": (root / _EXECUTION_AUTHORITY_CONSUMED_NAME).is_file(),
        "progress_tail": tail,
        "terminal": terminal,
    }


def refresh_derived_status_antigravity(lease: Exp449SmokeAExecutionLeaseAntigravity) -> None:
    """Atomically replace only the nonauthoritative operational status cache."""
    root = _authorized_operations_root_for_lease_antigravity(lease)
    temporary = root / f".{_STATUS_NAME}.{os.getpid()}.tmp_antigravity"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        _write_fd_all_antigravity(descriptor, canonical_json_bytes(_status_cache_payload_antigravity(root)))
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, root / _STATUS_NAME)
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _emit_progress_antigravity(config: Exp449WaveConfig, *, wave: int, snapshot: _RecoverySnapshot, started: float, completed: int | None = None, total: int | None = None, failed: int | None = None, dataset: str | None = None, _authorized_lease: Exp449SmokeAExecutionLeaseAntigravity | None = None) -> None:
    """Emit test-composition progress best-effort, but never downgrade lease failures."""
    completed_count = len(snapshot.successes) if completed is None else completed
    failed_count = len(snapshot.failed) if failed is None else failed
    total_count = len(_catalog_datasets_antigravity(config.catalog)) if total is None else total
    elapsed = float(monotonic() - started)
    rate = completed_count / elapsed if elapsed > 0.0 else 0.0
    event: dict[str, object] = {
        "schema": "exp449-operations-progress-antigravity.v1",
        "timestamp": f"{time():.6f}",
        "event": "dataset_progress" if dataset is not None else "wave_progress",
        "wave": wave,
        "dataset": dataset,
        "completed": completed_count,
        "failed": failed_count,
        "pending": max(0, total_count - completed_count),
        "elapsed_seconds": elapsed,
        "throughput_per_second": rate,
        "eta_seconds": (max(0, total_count - completed_count) / rate) if rate > 0.0 else 0.0,
        "latest_error": "dataset_failure" if failed_count else None,
        "worker_pid": os.getpid(),
    }
    if _authorized_lease is not None:
        append_authorized_progress_antigravity(_authorized_lease, event)
        try:
            refresh_derived_status_antigravity(_authorized_lease)
        except OSError:
            # The cache is expressly nonauthoritative: a replace/fsync failure
            # may leave it stale but must not block checkpoint/terminal truth.
            pass
        return
    try:
        _write_exp449_progress_event_antigravity(_operations_root_for_context_antigravity(config.context), event)
    except Exception:
        return


def status_exp449_antigravity(config: Exp449WaveConfig) -> Exp449TerminalStatus:
    """Return status from freshly verified records; attempts never imply success."""
    terminal = _existing_terminal_status_antigravity(config)
    if terminal is not None:
        return terminal
    return _terminal_status_antigravity(config, _snapshot_antigravity(config))


def _configured_execution_target_antigravity(config: Exp449WaveConfig) -> str | None:
    return {
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root: "smoke-a",
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root: "smoke-b",
        FROZEN_EXP449_CONFIG.official_scientific_root: "official",
    }.get(Path(config.context.run_root))


def _authorized_lease_for_config_antigravity(
    config: Exp449WaveConfig,
    lease: Exp449SmokeAExecutionLeaseAntigravity | None,
) -> Exp449SmokeAExecutionLeaseAntigravity | None:
    """Only the Smoke A fixed root can obtain durable progress from its lease."""
    target = _configured_execution_target_antigravity(config)
    if target is None:
        return None
    if target != "smoke-a" or type(lease) is not Exp449SmokeAExecutionLeaseAntigravity:
        raise ValueError("Exp449 active Smoke A execution lease is required")
    if lease.context is not config.context:
        raise ValueError("Exp449 execution lease context differs")
    _authorized_operations_root_for_lease_antigravity(lease)
    return lease


def run_exp449_wave_antigravity(
    config: Exp449WaveConfig,
    *,
    _execution_capability: object = None,
    _execution_lease: Exp449SmokeAExecutionLeaseAntigravity | None = None,
) -> Exp449AuthorizedWaveResultAntigravity:
    """Execute at most one of the two preregistered dataset-local recovery waves."""
    global compose_exp449_proposal_score_antigravity
    _execution_capability_for_config_antigravity(config, _execution_capability)
    authorized_lease = _authorized_lease_for_config_antigravity(config, _execution_lease)
    started = monotonic()
    terminal = _existing_terminal_status_antigravity(config)
    if terminal is not None:
        return Exp449AuthorizedWaveResultAntigravity(terminal, None)
    before = _snapshot_antigravity(config)
    before_status = _terminal_status_antigravity(config, before)
    # A verified all-success Task4/5 namespace still needs the existing
    # composer to reconstruct Task6/7 and Candidate C.  Its collector reuses
    # those verified records and must not re-score any dataset.
    next_wave = before.wave_count + 1 if before_status.status != "dataset_local_complete" else max(1, before.wave_count)
    if next_wave > FROZEN_EXP449_CONFIG.maximum_waves:
        return Exp449AuthorizedWaveResultAntigravity(
            verify_terminal_incomplete_antigravity(config, _execution_capability=_execution_capability), None,
        )
    _emit_progress_antigravity(config, wave=next_wave, snapshot=before, started=started, _authorized_lease=authorized_lease)
    context_payload, _authority = _authority_antigravity(config)
    if compose_exp449_proposal_score_antigravity is None:
        # Keeping scorer import here makes the public status command a genuine
        # read-only inspection path rather than a latent scoring import.
        from .runner_antigravity import compose_exp449_proposal_score_antigravity as composer
        compose_exp449_proposal_score_antigravity = composer
    from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScoreIncomplete
    progress_failures = 0
    progress_completed = len(before.successes)
    def fixed_progress(event: Mapping[str, object]) -> None:
        nonlocal progress_completed, progress_failures
        if event.get("stage") != "task4_task5" or event.get("phase") not in {"start", "tick", "error", "end"}:
            return
        if event.get("phase") == "tick" and authorized_lease is None:
            return
        if event.get("phase") == "tick":
            progress_completed += 1
        if event.get("phase") == "error":
            progress_failures += 1
        _emit_progress_antigravity(
            config, wave=next_wave, snapshot=before, started=started,
            completed=progress_completed, total=len(_catalog_datasets_antigravity(config.catalog)),
            failed=progress_failures, dataset=event.get("dataset") if event.get("phase") in {"tick", "error"} and type(event.get("dataset")) is str else None,
            _authorized_lease=authorized_lease,
        )
    try:
        score = compose_exp449_proposal_score_antigravity(
            config.raw_datasets,
            config.acceleration_manifest,
            root=config.context.run_root,
            context=context_payload,
            catalog=config.catalog,
            _progress_callback=fixed_progress,
            _strict_operations_progress=authorized_lease is not None,
        )
    except ProposalScoreIncomplete:
        after = _snapshot_antigravity(config)
        if after.wave_count < next_wave:
            raise ValueError("Exp449 incomplete collection lacks an append-only attempt record")
        for index, dataset in enumerate((name for name in after.successes if name not in before.successes), len(before.successes) + 1):
            _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, completed=index, total=len(_catalog_datasets_antigravity(config.catalog)), failed=len(after.failed), dataset=dataset, _authorized_lease=authorized_lease)
        _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, _authorized_lease=authorized_lease)
        if next_wave == FROZEN_EXP449_CONFIG.maximum_waves:
            return Exp449AuthorizedWaveResultAntigravity(
                verify_terminal_incomplete_antigravity(config, _execution_capability=_execution_capability), None,
            )
        return Exp449AuthorizedWaveResultAntigravity(_terminal_status_antigravity(config, after), None)
    after = _snapshot_antigravity(config)
    status = _terminal_status_antigravity(config, after)
    if status.status != "dataset_local_complete":
        raise ValueError("Exp449 composition returned before verified dataset-local closure")
    for index, dataset in enumerate((name for name in after.successes if name not in before.successes), len(before.successes) + 1):
        _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, completed=index, total=len(_catalog_datasets_antigravity(config.catalog)), failed=0, dataset=dataset, _authorized_lease=authorized_lease)
    _emit_progress_antigravity(config, wave=next_wave, snapshot=after, started=started, _authorized_lease=authorized_lease)
    # Completion is deliberately not terminal until the exact returned score
    # has been frozen and independently verified by the authorized parent.
    return Exp449AuthorizedWaveResultAntigravity(status, score)


def freeze_authorized_exp449_proposal_antigravity(
    score: object,
    *,
    config: Exp449WaveConfig,
    _execution_capability: object,
) -> Exp449ProposalManifest:
    """Capability-gated official wrapper around the unchanged scientific freeze."""
    _execution_capability_for_config_antigravity(config, _execution_capability)
    return freeze_exp449_proposal_antigravity(
        score, root=config.context.run_root, context=config.context, catalog=config.catalog,
    )


def _authorized_smoke_a_wave_config_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
    *,
    resume: bool,
) -> Exp449WaveConfig:
    """Derive the sole Smoke A execution inputs from reverified readonly facts."""
    facts = (
        readiness_for_authorized_resume_antigravity(verified)
        if resume else readiness_for_authorized_implementation_antigravity(verified)
    )
    grant = verified.grant
    catalogs, database = _verified_catalogs_antigravity()
    catalog = catalogs["smoke"]
    if (
        catalog.get("catalog_sha256") != grant.smoke_catalog_sha256
        or _catalog_datasets_antigravity(catalog) != grant.smoke_datasets
        or database.get("sha256") != grant.database_sha256
        or database.get("size") != grant.database_size
        or database.get("unresolved_path") != grant.database_path
    ):
        raise ValueError("Exp449 authorized Smoke A catalog or database drift differs")
    acceleration = verify_acceleration_manifest(
        build_acceleration_manifest(
            grant.acceleration_assessment_sha256,
            logical_cpu_count=grant.logical_cpu_count,
        ),
        grant.acceleration_assessment_sha256,
    )
    backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration)
    if backend.selected_backend != grant.canonical_backend:
        raise ValueError("Exp449 authorized Smoke A backend drift differs")
    required_facts = ("interpreter", "dependencies", "package_inventory", "seeds_and_constants", "code_closure", "lineage_sha256", "codemap_sha256")
    if any(name not in facts for name in required_facts):
        raise ValueError("Exp449 authorized Smoke A readiness facts are incomplete")
    context = Exp449Context.from_verified_inputs(
        run_root=FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
        mode="smoke",
        catalog=catalog,
        acceleration=backend,
        database_sha256=grant.database_sha256,
        code_closure=facts["code_closure"],  # type: ignore[arg-type]
        lineage_sha256=facts["lineage_sha256"],  # type: ignore[arg-type]
        codemap_sha256=facts["codemap_sha256"],  # type: ignore[arg-type]
        interpreter=facts["interpreter"]["executable"],  # type: ignore[index,arg-type]
        environment={
            "interpreter": facts["interpreter"],
            "dependencies": facts["dependencies"],
            "package_inventory": facts["package_inventory"],
        },
        seeds_and_constants=facts["seeds_and_constants"],  # type: ignore[arg-type]
    )
    if context_sha256_antigravity(context) != grant.context_sha256:
        raise ValueError("Exp449 authorized Smoke A context drift differs")
    raw = {
        dataset: load_dataset_values(Path(grant.database_path), dataset)
        for dataset in grant.smoke_datasets
    }
    return Exp449WaveConfig(
        context=context, catalog=catalog, raw_datasets=raw,
        acceleration_manifest=acceleration,
    )


def _freeze_exists_antigravity(config: Exp449WaveConfig) -> bool:
    """Accept either a fully verifiable freeze or no freeze directory at all."""
    freeze_root = config.context.run_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
    try:
        details = freeze_root.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp449 proposal freeze root is unsafe")
    from .verifier_antigravity import verify_exp449_proposal_antigravity
    verify_exp449_proposal_antigravity(config.context.run_root, context=config.context, catalog=config.catalog)
    return True


def _finalize_authorized_complete_antigravity(
    config: Exp449WaveConfig,
    result: Exp449AuthorizedWaveResultAntigravity,
    lease: Exp449SmokeAExecutionLeaseAntigravity,
) -> Exp449TerminalStatus:
    """Freeze, independently verify, then publish the sole complete terminal."""
    if result.status.status != "dataset_local_complete" or result.score is None:
        raise ValueError("Exp449 authorized completion lacks an exact ProposalScore")
    freeze_authorized_exp449_proposal_antigravity(
        result.score, config=config, _execution_capability=lease.capability,
    )
    from .verifier_antigravity import verify_exp449_proposal_antigravity
    verify_exp449_proposal_antigravity(config.context.run_root, context=config.context, catalog=config.catalog)
    terminal = _publish_terminal_complete_antigravity(
        config, _snapshot_antigravity(config), _execution_capability=lease.capability,
    )
    try:
        refresh_derived_status_antigravity(lease)
    except OSError:
        pass
    return terminal


def _execute_authorized_smoke_a_waves_antigravity(
    config: Exp449WaveConfig,
    lease: Exp449SmokeAExecutionLeaseAntigravity,
) -> Exp449TerminalStatus:
    """Run no more than the frozen two dataset-local waves in one parent."""
    result = run_exp449_wave_antigravity(
        config, _execution_capability=lease.capability, _execution_lease=lease,
    )
    if result.status.status == "proposal_score_incomplete" and result.status.wave_count < FROZEN_EXP449_CONFIG.maximum_waves:
        result = run_exp449_wave_antigravity(
            config, _execution_capability=lease.capability, _execution_lease=lease,
        )
    if result.status.status == "proposal_score_incomplete":
        return result.status
    return _finalize_authorized_complete_antigravity(config, result, lease)


def execute_smoke_a_fresh_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> Exp449TerminalStatus:
    """Perform the one permitted fresh Smoke A lifecycle with no caller inputs."""
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
        raise TypeError("Exp449 verified Smoke A authority is required")
    config = _authorized_smoke_a_wave_config_antigravity(verified, resume=False)
    with acquire_smoke_a_execution_lease_antigravity(verified, config.context) as lease:
        return _execute_authorized_smoke_a_waves_antigravity(config, lease)


def execute_smoke_a_resume_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> Exp449TerminalStatus:
    """Resume only an exact inactive consumed Smoke A namespace."""
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
        raise TypeError("Exp449 verified Smoke A authority is required")
    config = _authorized_smoke_a_wave_config_antigravity(verified, resume=True)
    with _acquire_smoke_a_resume_lease_antigravity(verified, config) as lease:
        terminal = _existing_terminal_status_antigravity(config)
        if terminal is not None:
            if terminal.status == "dataset_local_complete":
                _freeze_exists_antigravity(config)
            elif _freeze_exists_antigravity(config):
                raise ValueError("Exp449 incomplete terminal must not have a proposal freeze")
            return terminal
        if _freeze_exists_antigravity(config):
            return _publish_terminal_complete_antigravity(
                config, _snapshot_antigravity(config), _execution_capability=lease.capability,
            )
        return _execute_authorized_smoke_a_waves_antigravity(config, lease)


def _smoke_operations_roots_antigravity(root_a: Path, root_b: Path) -> tuple[Path, Path]:
    """Return the sole fixed operations authorities for an approved smoke pair."""
    try:
        operations_a, operations_b = _SMOKE_OPERATIONS_ROOT_PAIRS_ANTIGRAVITY[
            (Path(root_a).resolve(strict=False), Path(root_b).resolve(strict=False))
        ]
    except KeyError as error:
        raise ValueError("Exp449 smoke operations authority differs") from error
    if operations_a == operations_b or operations_a in operations_b.parents or operations_b in operations_a.parents:
        raise ValueError("Exp449 smoke operations roots must be independent")
    return operations_a, operations_b


def record_exp449_smoke_comparison_antigravity(root_a: Path, root_b: Path, authority: object):
    """Create the one Smoke B operations receipt after independent comparison.

    The verifier is imported only here, after operations has been imported; the
    verifier itself therefore remains a fresh scorer-free inspection boundary.
    """
    from .verifier_antigravity import _smoke_comparison_evidence_antigravity

    comparison, payload = _smoke_comparison_evidence_antigravity(root_a, root_b, authority)
    operations_a, operations_b = _smoke_operations_roots_antigravity(root_a, root_b)
    payload = dict(payload)
    payload.update({
        "smoke_a_operations_root": str(operations_a),
        "smoke_b_operations_root": str(operations_b),
        "smoke_a_operations_root_sha256": sha256(canonical_json_bytes({"root": str(operations_a)})).hexdigest(),
        "smoke_b_operations_root_sha256": sha256(canonical_json_bytes({"root": str(operations_b)})).hexdigest(),
    })
    _mkdirs_without_symlinks_antigravity(operations_b)
    write_create_only(
        operations_b / _SMOKE_COMPARISON_NAME,
        canonical_json_bytes(payload),
        reuse_identical=True,
    )
    return comparison


def _lower_hex_antigravity(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _current_clean_head_antigravity() -> str:
    """Read the sole review identity without inspecting any experiment input."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout
    if not _lower_hex_antigravity(head, 40) or dirty:
        raise ValueError("Exp449 readiness requires a clean lowercase 40-hex HEAD")
    return head


class _AuthorizationDeltaReadCapabilityAntigravity:
    """Unforgeable-in-normal-use proof that the verifier approved A -> I evidence."""

    __slots__ = ("approved_authority_commit", "implementation_commit")

    def __init__(self, verified: VerifiedExp449SmokeAAuthorityAntigravity, *, issuer: object) -> None:
        if issuer is not _AUTHORIZATION_DELTA_READ_ISSUER_ANTIGRAVITY or type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
            raise TypeError("Exp449 authorization delta read capability is private")
        self.approved_authority_commit = verified.approved_authority_commit
        self.implementation_commit = verified.implementation_commit


def _require_reviewed_head_antigravity(
    reviewed_commit: str,
    *,
    authorization_delta_capability: object | None = None,
) -> str:
    if not _lower_hex_antigravity(reviewed_commit, 40):
        raise ValueError("Exp449 reviewed commit is invalid")
    head = _current_clean_head_antigravity()
    if authorization_delta_capability is None:
        if head != reviewed_commit:
            raise ValueError("Exp449 reviewed commit is not the current clean HEAD")
        return reviewed_commit
    if type(authorization_delta_capability) is not _AuthorizationDeltaReadCapabilityAntigravity:
        raise ValueError("Exp449 authorized readiness capability is invalid")
    capability = authorization_delta_capability
    if capability.implementation_commit != reviewed_commit or head != capability.approved_authority_commit:
        raise ValueError("Exp449 reviewed commit is not the current clean HEAD")
    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", head], cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout.split()
    if len(parents) != 2 or parents != [head, reviewed_commit]:
        raise ValueError("Exp449 authorized readiness commit delta differs")
    return reviewed_commit


def _fixed_roots_antigravity() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return only the six config-bound destinations; callers cannot supply roots."""
    return (
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
        FROZEN_EXP449_CONFIG.smoke_a_operations_root,
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root,
        FROZEN_EXP449_CONFIG.smoke_b_operations_root,
        FROZEN_EXP449_CONFIG.official_scientific_root,
        FROZEN_EXP449_CONFIG.official_operations_root,
    )


def _lstat_or_none_antigravity(path: Path):
    try:
        return Path(path).lstat()
    except FileNotFoundError:
        return None


def _inspect_fixed_roots_antigravity(roots: tuple[Path, ...]) -> list[dict[str, str]]:
    """Inspect raw path components only; this review helper never resolves or creates.

    Its private ``roots`` argument exists solely to exercise hostile temporary
    fixtures.  Every public readiness/status/CLI path obtains the fixed tuple
    above, so an operator cannot redirect an Exp449 root.
    """
    if len(roots) != 6 or len({str(path) for path in roots}) != 6:
        raise ValueError("Exp449 fixed roots differ")
    result: list[dict[str, str]] = []
    for root in roots:
        raw = Path(root)
        if not raw.is_absolute() or ".." in raw.parts:
            raise ValueError("Exp449 fixed roots differ")
        normalized_components = frozenset(part.casefold() for part in raw.parts[1:])
        if (
            any(historical_root == raw or historical_root in raw.parents for historical_root in _HISTORICAL_ROOTS_ANTIGRAVITY)
            or normalized_components & _HISTORICAL_ROOT_COMPONENTS_ANTIGRAVITY
        ):
            raise ValueError("Exp449 historical root is forbidden")
        current = Path(raw.anchor)
        for part in raw.parts[1:]:
            current /= part
            details = _lstat_or_none_antigravity(current)
            if details is not None and stat.S_ISLNK(details.st_mode):
                raise ValueError("Exp449 symlinked root component is forbidden")
        details = _lstat_or_none_antigravity(raw)
        if details is not None:
            raise ValueError("Exp449 fixed root must be absent")
        result.append({"path": str(raw), "state": "absent"})
    return result


def _status_fixed_roots_antigravity(roots: tuple[Path, ...]) -> list[dict[str, str]]:
    """Read raw fixed-root lstat facts without treating absent roots as errors."""
    result: list[dict[str, str]] = []
    for root in roots:
        raw = Path(root)
        if not raw.is_absolute() or ".." in raw.parts:
            raise ValueError("Exp449 fixed roots differ")
        current = Path(raw.anchor)
        for part in raw.parts[1:]:
            current /= part
            details = _lstat_or_none_antigravity(current)
            if details is not None and stat.S_ISLNK(details.st_mode):
                raise ValueError("Exp449 symlinked root component is forbidden")
        details = _lstat_or_none_antigravity(raw)
        state = "absent" if details is None else "symlink" if stat.S_ISLNK(details.st_mode) else "present"
        result.append({"path": str(raw), "state": state})
    return result


def _require_capable_interpreter_antigravity() -> dict[str, str]:
    expected = _CAPABLE_INTERPRETER_ANTIGRAVITY
    observed = os.path.abspath(sys.executable)
    if not expected.is_file() or observed != str(expected) or platform.python_version() != "3.14.6":
        raise ValueError("Exp449 capable interpreter differs")
    return {
        "executable": observed,
        "pinned_executable": str(expected),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
    }


def _dependency_facts_antigravity() -> dict[str, str]:
    """Call the frozen paper preflights instead of recreating their semantics."""
    pycatch = verify_pycatch22_environment_paper()
    exp84 = verify_exp84_environment_paper()
    try:
        aeon = importlib.metadata.version("aeon")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("dependency_unavailable:aeon==1.5.0") from error
    if aeon != "1.5.0":
        raise ValueError("dependency_unavailable:aeon==1.5.0")
    if pycatch.get("pycatch22") != "0.5.0" or exp84.get("hydra") != "constructed" or exp84.get("torch", "").split("+", 1)[0] != "2.13.0":
        raise ValueError("Exp449 frozen dependency preflight differs")
    return {"pycatch22": pycatch["pycatch22"], "aeon": aeon, "hydra": exp84["hydra"], "torch": exp84["torch"]}


def _package_inventory_facts_antigravity(
    distributions: object | None = None,
) -> dict[str, object]:
    """Return the complete normalized installed-distribution identity, not only required packages."""
    observed = importlib.metadata.distributions() if distributions is None else distributions
    try:
        iterator = iter(observed)
    except TypeError as error:
        raise ValueError("Exp449 package inventory is invalid") from error
    normalized: dict[str, str] = {}
    for distribution in iterator:
        metadata = getattr(distribution, "metadata", None)
        name = metadata.get("Name") if metadata is not None else None
        version = getattr(distribution, "version", None)
        if type(name) is not str or not name.strip() or type(version) is not str or not version:
            raise ValueError("Exp449 package inventory is invalid")
        project = _NORMALIZED_PROJECT_NAME_ANTIGRAVITY.sub("-", name.strip()).lower()
        existing = normalized.get(project)
        if existing is not None and existing != version:
            raise ValueError("Exp449 package inventory has conflicting normalized project versions")
        normalized[project] = version
    if not normalized:
        raise ValueError("Exp449 package inventory is empty")
    records = [{"project": project, "version": normalized[project]} for project in sorted(normalized)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    return {
        "schema": "exp449-installed-distributions-antigravity.v1",
        "algorithm": "packaging-normalized-project-name+canonical-json-sha256.v1",
        "records": records,
        "sha256": sha256(canonical).hexdigest(),
    }


def _readiness_catalog_builder_antigravity(mode: str) -> dict[str, object]:
    """The sole read-only paper catalog builder; test fixtures replace only this seam."""
    database = Path(FROZEN_PAPER_CONFIG.database_path)
    if mode == "smoke":
        value = build_catalog(database, mode="smoke", smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets)
        return verify_catalog(value, database, "smoke", FROZEN_EXP449_CONFIG.smoke_datasets)
    if mode == "official":
        value = build_catalog(database, mode="full")
        return verify_catalog(value, database, "full")
    raise ValueError("Exp449 readiness catalog mode is invalid")


def _verified_catalogs_antigravity() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    catalogs = {"smoke": _readiness_catalog_builder_antigravity("smoke"), "official": _readiness_catalog_builder_antigravity("official")}
    expected_path = FROZEN_PAPER_CONFIG.database_path
    expected_resolved = str(Path(expected_path).resolve(strict=False))
    observed_database: dict[str, object] | None = None
    for catalog in catalogs.values():
        database = catalog.get("database") if isinstance(catalog, dict) else None
        if not isinstance(database, dict) or database.get("unresolved_path") != expected_path or database.get("resolved_path") != expected_resolved or type(database.get("size")) is not int or database["size"] < 0 or database.get("sha256") != FROZEN_PAPER_CONFIG.database_sha256:
            raise ValueError("Exp449 verified catalog database SHA256 or identity differs")
        if observed_database is None:
            observed_database = dict(database)
        elif database != observed_database:
            raise ValueError("Exp449 verified catalog database identities differ")
    assert observed_database is not None
    return catalogs, observed_database


def _preprocessing_contracts_antigravity() -> tuple[str, str]:
    contracts = tuple(FROZEN_EXP449_CONFIG.preprocessing_contracts)
    if contracts != _EXPECTED_PREPROCESSING_CONTRACTS_ANTIGRAVITY or len(set(contracts)) != 2:
        raise ValueError("Exp449 preprocessing contracts differ")
    return contracts


def _label_firewall_antigravity() -> None:
    loaded = tuple(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _LABEL_MODULE_PREFIXES_ANTIGRAVITY)
    )
    if loaded:
        raise ValueError("Exp449 label module is loaded")


def _validated_dependency_facts_antigravity(value: object) -> dict[str, str]:
    required = {"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "2.13.0"}
    if not isinstance(value, dict) or set(value) != set(required) or any(type(value[key]) is not str for key in required) or value != required:
        raise ValueError("Exp449 dependency facts differ")
    return dict(value)


def _sha256_file_antigravity(path: Path) -> str:
    details = _lstat_or_none_antigravity(path)
    if details is None or stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp449 closure input is unsafe")
    return sha256(path.read_bytes()).hexdigest()


def _closure_facts_antigravity() -> dict[str, object]:
    if tuple(sorted(_FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY)) != _FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY:
        raise ValueError("Exp449 frozen scientific closure order differs")
    tracked = subprocess.run(
        ("git", "ls-files", "--", *_FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY),
        cwd=_REPOSITORY_ROOT_ANTIGRAVITY, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if tuple(tracked) != _FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY:
        raise ValueError("Exp449 frozen scientific closure paths are not tracked exactly")
    closure: dict[str, str] = {}
    for relative in _FROZEN_SCIENTIFIC_CLOSURE_PATHS_ANTIGRAVITY:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != ".py":
            raise ValueError("Exp449 frozen scientific closure path is unsafe")
        resolved = (_REPOSITORY_ROOT_ANTIGRAVITY / candidate).resolve(strict=False)
        if _REPOSITORY_ROOT_ANTIGRAVITY not in resolved.parents:
            raise ValueError("Exp449 frozen scientific closure path escapes repository")
        closure[relative] = _sha256_file_antigravity(resolved)
    lineage_lock_paper_sha256 = _sha256_file_antigravity(LINEAGE_LOCK_PATH)
    codemap_json_path = _REPOSITORY_ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp449_antigravity.json"
    codemap_lock_path = _REPOSITORY_ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp449_antigravity.lock"
    return {
        "code_closure": closure,
        "lineage_lock_paper_sha256": lineage_lock_paper_sha256,
        "lineage_sha256": lineage_lock_paper_sha256,
        "codemap_json_sha256": _sha256_file_antigravity(codemap_json_path),
        "codemap_lock_sha256": _sha256_file_antigravity(codemap_lock_path),
        "codemap_sha256": _sha256_file_antigravity(codemap_lock_path),
    }


def _ast_literal_assignments_antigravity(relative: str, names: tuple[str, ...]) -> dict[str, object]:
    """Read only literal source facts; no score source is imported or invoked."""
    source = _REPOSITORY_ROOT_ANTIGRAVITY / relative
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: dict[str, object] = {}

    def static_value(node: ast.AST, bindings: Mapping[str, object]) -> object:
        if isinstance(node, ast.Name) and node.id in bindings:
            return bindings[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = static_value(node.left, bindings), static_value(node.right, bindings)
            if type(left) is tuple and type(right) is tuple:
                return left + right
        return ast.literal_eval(node)

    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,) if isinstance(node, ast.AnnAssign) else ()
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    values[target.id] = static_value(node.value, values)
                except ValueError as error:
                    raise ValueError(f"Exp449 constants literal is invalid: {relative}:{target.id}") from error
    if set(values) != set(names):
        raise ValueError(f"Exp449 constants literal is incomplete: {relative}")
    return values


def _seeds_and_constants_antigravity() -> dict[str, object]:
    """Bind every scientific literal category using import-safe config and AST-only source reads."""
    hard = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/hard_route_paper.py",
        ("MAX_TOP_PAPER", "EXP93_WEIGHTS_PAPER", "EXP234_SOURCES_PAPER"),
    )
    sources = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
        ("RAW_PRIMARY_SOURCES", "EXPECTED_HARD_SOURCES", "SOURCE_ALPHA_PAPER"),
    )
    low = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
        ("MINIMUM_RAW_LENGTH_PAPER", "MAXIMUM_DEPTH_PAPER", "EMBEDDING_DIMENSION_PAPER", "DELAYS_PAPER", "NEAREST_K_PAPER", "QUANTILE_PAPER", "MINIMUM_TRAIN_ROWS_PAPER", "MAXIMUM_TRAIN_ROWS_PAPER", "STABILITY_RATIO_MAX_PAPER", "GLOBAL_CANDIDATE_BUDGET_PAPER"),
    )
    mini = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
        ("SEEDS_PAPER", "CROSSFIT_FOLDS_PAPER", "KNN_NEIGHBORS_PAPER", "MAX_TRAIN_REFERENCE_PAPER", "QUERY_CHUNK_SIZE_PAPER", "MINIMUM_TRAIN_PAPER"),
    )
    artifact = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/artifacts_paper.py",
        ("SHARD_SCHEMA", "MANIFEST_SCHEMA", "PARITY_SCHEMA", "TASK4_TASK5_SUCCESS_SCHEMA", "TASK4_TASK5_ATTEMPT_SCHEMA"),
    )
    preprocessing = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py", ("PAPER_SEED",)
    )
    canonical = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/canonical_paper.py", ("CANONICAL_SEED_PAPER", "PINNED_PYCATCH22_VERSION_PAPER")
    )
    rocket = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py", ("ROCKET_SEED_PAPER", "MAX_TRAIN_REFERENCE_PAPER")
    )
    imaging = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py", ("RNG_SEED_PAPER", "EXP55_CONFIG_PAPER", "EXP56_CONFIG_PAPER")
    )
    multirocket = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py", ("EXP84_CONFIG_PAPER", "PINNED_TORCH_VERSION_PAPER")
    )
    routing = _ast_literal_assignments_antigravity(
        "paper/equipment_anomaly_detection/routing_paper.py", ("POLICY_ORDER", "EXPECTED_COUNTS"))
    return {
        "schema": "exp449-seeds-and-constants-antigravity.v1",
        "algorithm": "ast-literal-and-import-safe-config.v1",
        "seeds": {
            "paper_config_seed": FROZEN_PAPER_CONFIG.seed, "paper_seed": preprocessing["PAPER_SEED"],
            "canonical_seed": canonical["CANONICAL_SEED_PAPER"], "rocket_seed": rocket["ROCKET_SEED_PAPER"],
            "imaging_seed": imaging["RNG_SEED_PAPER"], "minirocket_seeds": mini["SEEDS_PAPER"],
            "multirocket_random_state": multirocket["EXP84_CONFIG_PAPER"]["random_state"],
        },
        "alpha_and_thresholds": {"source_alpha": sources["SOURCE_ALPHA_PAPER"], "low_frequency_quantile": low["QUANTILE_PAPER"], "stability_ratio_max": low["STABILITY_RATIO_MAX_PAPER"]},
        "route_promotion_topk_and_rate_rules": {"max_top": hard["MAX_TOP_PAPER"], "exp93_weights": hard["EXP93_WEIGHTS_PAPER"], "exp234_sources": hard["EXP234_SOURCES_PAPER"], "low_frequency": low, "minirocket": mini},
        "policy_ids_and_order": {"exp449_policy_id": FROZEN_EXP449_CONFIG.policy_id, "exp449_policy_order": list(FROZEN_EXP449_POLICY_KEYS), "paper_policy_order": routing["POLICY_ORDER"], "paper_expected_counts": routing["EXPECTED_COUNTS"], "hard_sources": sources["EXPECTED_HARD_SOURCES"]},
        "preprocessing_contracts": list(_preprocessing_contracts_antigravity()),
        "dataset_and_test_counts": {"exp449_dataset_count": FROZEN_EXP449_CONFIG.dataset_count, "exp449_test_row_count": FROZEN_EXP449_CONFIG.test_row_count, "paper_dataset_count": FROZEN_PAPER_CONFIG.dataset_count, "paper_test_row_count": FROZEN_PAPER_CONFIG.test_row_count, "smoke_dataset_count": len(FROZEN_EXP449_CONFIG.smoke_datasets)},
        "tolerances": {"cpu_repeat_tolerance": FROZEN_EXP449_CONFIG.cpu_repeat_tolerance, "accelerated_tolerance": FROZEN_EXP449_CONFIG.accelerated_tolerance, "minimum_accelerated_speedup": FROZEN_EXP449_CONFIG.minimum_accelerated_speedup},
        "task4_task5_recovery": {"maximum_waves": FROZEN_EXP449_CONFIG.maximum_waves, "task4_train_only": "task4_train_only", "task5_value_length_contract": "task5_label_free_train_test_value_length_transductive"},
        "serialization_and_verifier_schemas": {"exp449_shard_schema": "exp449-proposal-shard-antigravity.v1", "exp449_manifest_schema": "exp449-proposal-manifest-antigravity.v1", "exp449_terminal_schema": _TERMINAL_SCHEMA, "task45_success_schema": artifact["TASK4_TASK5_SUCCESS_SCHEMA"], "task45_attempt_schema": artifact["TASK4_TASK5_ATTEMPT_SCHEMA"], "paper_shard_schema": artifact["SHARD_SCHEMA"], "paper_manifest_schema": artifact["MANIFEST_SCHEMA"], "paper_parity_schema": artifact["PARITY_SCHEMA"], "acceleration_schema": "paper-task9-acceleration.v2", "root_snapshot_verifier": "verify_exp449_proposal_root_snapshot_antigravity"},
        "source_hash_only_constants": [],
    }


def _validate_seeds_and_constants_antigravity(value: object) -> dict[str, object]:
    expected = _seeds_and_constants_antigravity()
    if not isinstance(value, dict) or value != expected:
        raise ValueError("Exp449 seeds and constants differ")
    return expected


def _plain_json_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_antigravity(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json_antigravity(child) for child in value]
    if isinstance(value, list):
        return [_plain_json_antigravity(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _readiness_contexts_antigravity(
    *, reviewed_commit: str, interpreter: dict[str, str], dependencies: dict[str, str],
    contracts: tuple[str, str], catalogs: dict[str, dict[str, object]], database: dict[str, object],
    closure: dict[str, object], package_inventory: dict[str, object], logical_cpu_count: int | None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    constants = _validate_seeds_and_constants_antigravity(_seeds_and_constants_antigravity())
    entries = (
        ("smoke_a", "smoke", FROZEN_EXP449_CONFIG.smoke_a_scientific_root, catalogs["smoke"]),
        ("smoke_b", "smoke", FROZEN_EXP449_CONFIG.smoke_b_scientific_root, catalogs["smoke"]),
        ("official", "official", FROZEN_EXP449_CONFIG.official_scientific_root, catalogs["official"]),
    )
    result: dict[str, dict[str, object]] = {}
    official_acceleration: dict[str, object] | None = None
    for name, mode, root, catalog in entries:
        assessment_inputs = {
            "schema": "exp449-readiness-assessment-inputs-antigravity.v1", "reviewed_commit": reviewed_commit,
            "mode": mode, "run_root": str(root), "database": database,
            "catalog_sha256": catalog["catalog_sha256"], "interpreter": interpreter,
            "dependencies": dependencies, "package_inventory": package_inventory, "closure": closure, "constants": constants,
        }
        assessment_sha256 = sha256(json.dumps(assessment_inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        acceleration = verify_acceleration_manifest(build_acceleration_manifest(assessment_sha256, logical_cpu_count=logical_cpu_count), assessment_sha256)
        decision = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration)
        context = Exp449Context.from_verified_inputs(
            run_root=root, mode=mode, catalog=catalog, acceleration=decision,
            database_sha256=str(database["sha256"]), code_closure=closure["code_closure"],
            lineage_sha256=str(closure["lineage_sha256"]), codemap_sha256=str(closure["codemap_sha256"]),
            interpreter=interpreter["executable"], environment={"interpreter": interpreter, "dependencies": dependencies, "package_inventory": package_inventory},
            seeds_and_constants=constants,
        )
        result[name] = {
            "context_sha256": context_sha256_antigravity(context), "mode": context.mode,
            "run_root": str(context.run_root), "catalog_sha256": context.catalog_sha256,
            "database_sha256": context.database_sha256, "historical_parity_status": context.historical_parity_status,
            "backend_record": _plain_json_antigravity(context.backend_record),
            "acceleration_assessment_sha256": assessment_sha256,
        }
        if name == "official":
            official_acceleration = acceleration
    assert official_acceleration is not None
    return result, official_acceleration


def _readiness_facts_antigravity(
    reviewed_commit: str,
    *,
    logical_cpu_count: int | None = None,
    authorization_delta_capability: object | None = None,
    authorized_grant: Exp449SmokeAGrantAntigravity | None = None,
    allow_smoke_a_resume: bool = False,
) -> dict[str, object]:
    """The implementation seam for readonly fixture testing; never creates or scores."""
    reviewed_commit = _require_reviewed_head_antigravity(
        reviewed_commit, authorization_delta_capability=authorization_delta_capability,
    )
    interpreter = _require_capable_interpreter_antigravity()
    _label_firewall_antigravity()
    dependencies = _validated_dependency_facts_antigravity(_dependency_facts_antigravity())
    package_inventory = _package_inventory_facts_antigravity()
    contracts = _preprocessing_contracts_antigravity()
    constants = _validate_seeds_and_constants_antigravity(_seeds_and_constants_antigravity())
    if contracts != _EXPECTED_PREPROCESSING_CONTRACTS_ANTIGRAVITY or len(set(contracts)) != 2:
        raise ValueError("Exp449 preprocessing contracts differ")
    catalogs, database = _verified_catalogs_antigravity()
    if authorized_grant is not None:
        if (
            authorized_grant.database_path != database["unresolved_path"]
            or authorized_grant.database_size != database["size"]
            or authorized_grant.database_sha256 != database["sha256"]
            or authorized_grant.smoke_catalog_sha256 != catalogs["smoke"].get("catalog_sha256")
            or authorized_grant.smoke_datasets != _catalog_datasets_antigravity(catalogs["smoke"])
        ):
            raise ValueError("Exp449 authorized Smoke A database or catalog facts differ")
    roots = (
        _status_fixed_roots_antigravity(tuple(_fixed_roots_antigravity()))
        if allow_smoke_a_resume else _inspect_fixed_roots_antigravity(tuple(_fixed_roots_antigravity()))
    )
    closure = _closure_facts_antigravity()
    contexts, acceleration = _readiness_contexts_antigravity(
        reviewed_commit=reviewed_commit, interpreter=interpreter, dependencies=dependencies, contracts=contracts,
        catalogs=catalogs, database=database, closure=closure, package_inventory=package_inventory, logical_cpu_count=logical_cpu_count,
    )
    workers = dict(acceleration["workers"])
    benchmark_requirement = "same_input_and_seed; finite_outputs; identical_candidate_indices_at_every_preregistered_alpha; numerical_tolerance; at_least_1.1x_end_to_end_including_transfer_compilation_and_synchronization"
    facts: dict[str, object] = {
        "schema": "exp449-readiness-antigravity.v1",
        "experiment_id": "Exp449",
        "reviewed_commit": reviewed_commit,
        "interpreter": interpreter,
        "dependencies": dependencies,
        "package_inventory": package_inventory,
        "database": {"path": database["unresolved_path"], "resolved_path": database["resolved_path"], "observed_size": database["size"], "sha256": database["sha256"], "access": "read_only_verified_catalog"},
        "catalogs": {mode: {"catalog_sha256": catalog["catalog_sha256"], "dataset_count": len(catalog["datasets"])} for mode, catalog in catalogs.items()},
        "workers": {"logical_cpu_count": workers["logical_cpu_count"], "dataset_workers": workers["dataset_workers"]},
        "preprocessing_contracts": list(contracts),
        "seeds_and_constants": constants,
        "acceleration": {
            "canonical_backend": acceleration["selected_backend"],
            "fallback_reason": acceleration["fallback_reason"],
            "backends": acceleration["backends"],
            "stages": acceleration["stages"],
            "benchmark_requirement": benchmark_requirement,
            "cpu_vs_accelerated_tolerance": FROZEN_EXP449_CONFIG.accelerated_tolerance,
            "cpu_repeat_tolerance": FROZEN_EXP449_CONFIG.cpu_repeat_tolerance,
            "minimum_accelerated_speedup": FROZEN_EXP449_CONFIG.minimum_accelerated_speedup,
            "one_heavy_process_only": True,
            "no_outer_pool_with_heavy_process": True,
        },
        "roots": roots,
        "contexts": contexts,
        "label_modules_loaded": [],
        **closure,
    }
    plain = _plain_json_antigravity(facts)
    try:
        json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp449 readiness facts are not canonical JSON-ready") from error
    assert isinstance(plain, dict)
    return plain


def readiness_exp449_antigravity(reviewed_commit: str) -> dict[str, object]:
    """Return canonical facts only after clean-head, capability, and no-side-effect gates."""
    return _readiness_facts_antigravity(reviewed_commit)


def _require_authorized_smoke_a_readiness_facts_antigravity(
    facts: dict[str, object], grant: Exp449SmokeAGrantAntigravity,
    *,
    allow_smoke_a_resume: bool = False,
) -> None:
    """Require every recomputed readonly fact to match the immutable Smoke A grant."""
    contexts = facts.get("contexts")
    roots = facts.get("roots")
    database = facts.get("database")
    acceleration = facts.get("acceleration")
    expected_roots = tuple(str(path) for path in _fixed_roots_antigravity())
    authority_relative = "experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py"
    expected_authority_closure = {
        authority_relative: _sha256_file_antigravity(_REPOSITORY_ROOT_ANTIGRAVITY / authority_relative),
    }
    if dict(grant.authority_code_closure) != expected_authority_closure:
        raise ValueError("Exp449 authorized Smoke A authority closure differs from the grant")
    if (
        not isinstance(contexts, dict) or not isinstance(roots, list) or not isinstance(database, dict)
        or not isinstance(acceleration, dict) or facts.get("reviewed_commit") != grant.implementation_commit
        or database.get("path") != grant.database_path or database.get("observed_size") != grant.database_size
        or database.get("sha256") != grant.database_sha256 or facts.get("code_closure") != dict(grant.scientific_code_closure)
        or facts.get("lineage_lock_paper_sha256") != grant.lineage_lock_sha256
        or facts.get("codemap_lock_sha256") != grant.codemap_lock_sha256
        or facts.get("package_inventory") != _plain_json_antigravity(grant.package_inventory)
        or facts.get("seeds_and_constants") != _plain_json_antigravity(grant.seeds_and_constants)
        or acceleration != _plain_json_antigravity(grant.acceleration)
        or facts.get("workers") != {"logical_cpu_count": grant.logical_cpu_count, "dataset_workers": grant.dataset_workers}
        or acceleration.get("canonical_backend") != grant.canonical_backend
        or tuple(item.get("path") for item in roots if isinstance(item, dict)) != expected_roots
        or tuple(item.get("state") for item in roots if isinstance(item, dict)) != (
            ("present", "present", "absent", "absent", "absent", "absent")
            if allow_smoke_a_resume else ("absent",) * 6
        )
        or len(roots) != 6 or str(grant.scientific_root) != expected_roots[0] or str(grant.operations_root) != expected_roots[1]
        or grant.maximum_waves != FROZEN_EXP449_CONFIG.maximum_waves or grant.label_access is not False
    ):
        raise ValueError("Exp449 authorized Smoke A readiness facts differ from the grant")
    smoke_context = contexts.get("smoke_a")
    if (
        not isinstance(smoke_context, dict)
        or smoke_context.get("context_sha256") != grant.context_sha256
        or smoke_context.get("catalog_sha256") != grant.smoke_catalog_sha256
        or smoke_context.get("acceleration_assessment_sha256") != grant.acceleration_assessment_sha256
        or facts.get("label_modules_loaded") != []
    ):
        raise ValueError("Exp449 authorized Smoke A context facts differ from the grant")


def readiness_for_authorized_implementation_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> dict[str, object]:
    """Recompute readonly Smoke A facts for I only after fresh exact A -> I verification."""
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
        raise TypeError("Exp449 verified Smoke A authority is required")
    reverified = verify_smoke_a_authorization_commit_antigravity(verified.approved_authority_commit)
    if reverified != verified:
        raise ValueError("Exp449 verified Smoke A authority snapshot changed")
    capability = _AuthorizationDeltaReadCapabilityAntigravity(
        reverified, issuer=_AUTHORIZATION_DELTA_READ_ISSUER_ANTIGRAVITY,
    )
    facts = _readiness_facts_antigravity(
        reverified.implementation_commit,
        authorization_delta_capability=capability,
        authorized_grant=reverified.grant,
    )
    _require_authorized_smoke_a_readiness_facts_antigravity(facts, reverified.grant)
    return facts


def readiness_for_authorized_resume_antigravity(
    verified: VerifiedExp449SmokeAAuthorityAntigravity,
) -> dict[str, object]:
    """Recompute the immutable Smoke A identity while its owned root exists."""
    if type(verified) is not VerifiedExp449SmokeAAuthorityAntigravity:
        raise TypeError("Exp449 verified Smoke A authority is required")
    reverified = verify_smoke_a_authorization_commit_antigravity(verified.approved_authority_commit)
    if reverified != verified:
        raise ValueError("Exp449 verified Smoke A authority snapshot changed")
    capability = _AuthorizationDeltaReadCapabilityAntigravity(
        reverified, issuer=_AUTHORIZATION_DELTA_READ_ISSUER_ANTIGRAVITY,
    )
    facts = _readiness_facts_antigravity(
        reverified.implementation_commit,
        authorization_delta_capability=capability,
        authorized_grant=reverified.grant,
        allow_smoke_a_resume=True,
    )
    _require_authorized_smoke_a_readiness_facts_antigravity(
        facts, reverified.grant, allow_smoke_a_resume=True,
    )
    return facts


def status_exp449_read_only_antigravity() -> dict[str, object]:
    """Inspect pre-existing Smoke A operations metadata without healing any path."""
    roots = _status_fixed_roots_antigravity(tuple(_fixed_roots_antigravity()))
    smoke_a = _read_only_smoke_a_operations_status_antigravity(roots)
    return {
        "schema": "exp449-read-only-status-antigravity.v1",
        "experiment_id": "Exp449",
        "status": "not_started" if all(item["state"] == "absent" for item in roots) else "root_present",
        "roots": roots,
        "execution_authority": _execution_authority_status_antigravity(),
        "smoke_a_operations": smoke_a,
    }


def _read_only_path_state_antigravity(path: Path) -> str:
    """Classify a path with lstat only; status never follows or repairs it."""
    try:
        details = Path(path).lstat()
    except FileNotFoundError:
        return "absent"
    if stat.S_ISLNK(details.st_mode):
        return "symlink"
    if stat.S_ISREG(details.st_mode):
        return "file"
    if stat.S_ISDIR(details.st_mode):
        return "directory"
    return "other"


def _read_only_count_children_antigravity(directory: Path) -> int | None:
    """Count only immediate checkpoint records and leave malformed paths observable."""
    try:
        details = directory.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        return None
    try:
        return sum(1 for child in directory.iterdir() if _read_only_path_state_antigravity(child) == "file")
    except OSError:
        return None


def _read_only_progress_tail_antigravity(path: Path) -> dict[str, object] | None:
    """Read at most one bounded trailing line; malformed telemetry is nonauthoritative."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        size = os.lseek(descriptor, 0, os.SEEK_END)
        os.lseek(descriptor, max(0, size - 65536), os.SEEK_SET)
        tail = os.read(descriptor, min(65536, size)).splitlines()
    except OSError:
        return {"state": "unreadable"}
    finally:
        os.close(descriptor)
    if not tail:
        return None
    try:
        return _validate_progress_event_antigravity(json.loads(tail[-1].decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {"state": "unreadable"}


def _read_only_smoke_a_operations_status_antigravity(roots: list[dict[str, str]]) -> dict[str, object]:
    """Report presence/count metadata only; no checkpoint payload is opened."""
    scientific_root = Path(roots[0]["path"])
    operations_root = Path(roots[1]["path"])
    work = scientific_root / "stage_artifacts_paper" / "proposal_work"
    manifest = scientific_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity" / "proposal_manifest_antigravity.json"
    return {
        "claim": _read_only_path_state_antigravity(SMOKE_A_CLAIM_PATH_ANTIGRAVITY),
        "execution_authority_consumed": _read_only_path_state_antigravity(operations_root / _EXECUTION_AUTHORITY_CONSUMED_NAME),
        "progress_tail": _read_only_progress_tail_antigravity(operations_root / _PROGRESS_NAME),
        "checkpoint_counts": {
            "success": _read_only_count_children_antigravity(work / "task4_task5_success"),
            "attempt": _read_only_count_children_antigravity(work / "task4_task5_attempts"),
        },
        "proposal_manifest": _read_only_path_state_antigravity(manifest),
        "terminal_complete": _read_only_path_state_antigravity(operations_root / _TERMINAL_COMPLETE_NAME),
        "terminal_incomplete": _read_only_path_state_antigravity(operations_root / _TERMINAL_INCOMPLETE_NAME),
        "status_cache": _read_only_path_state_antigravity(operations_root / _STATUS_NAME),
    }


def verify_exp449_proposal_read_only_antigravity(target: str) -> dict[str, object]:
    """Report whether a fixed target is available; verification itself remains scorer-free."""
    names = {"smoke-a": 0, "smoke-b": 2, "official": 4}
    if target not in names:
        raise ValueError("Exp449 proposal target is invalid")
    roots = _status_fixed_roots_antigravity(tuple(_fixed_roots_antigravity()))
    root = roots[names[target]]
    if root["state"] == "absent":
        return {"schema": "exp449-proposal-verify-status-antigravity.v1", "experiment_id": "Exp449", "target": target, "status": "absent", "root": root["path"]}
    raise ValueError("Exp449 proposal verification requires separately verified immutable context and catalog inputs")


def _execution_authority_status_antigravity() -> dict[str, str]:
    return {"status": "execution_authority_not_configured", "enablement_requirement": "later_separately_reviewed_code_and_config_change_must_pin_a_trusted_authority_digest_or_signature"}


def _require_execution_authority_antigravity(target: str, reviewed_commit: str) -> None:
    """Fail closed: no trusted issuer or pinned digest exists in the frozen contract."""
    if target not in {"smoke_a", "smoke_b", "official"}:
        raise ValueError("Exp449 execution target is invalid")
    _require_reviewed_head_antigravity(reviewed_commit)
    raise ValueError(_execution_authority_status_antigravity()["status"])


def main_antigravity(argv: list[str] | None = None) -> int:
    """Thin non-executing CLI; future authority does not make this implementation launch."""
    parser = argparse.ArgumentParser(prog="run_exp449_reproducible_rebaseline_antigravity")
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("readiness")
    readiness.add_argument("--reviewed-commit", required=True)
    commands.add_parser("status")
    verify = commands.add_parser("verify-proposal")
    verify.add_argument("--target", choices=("smoke-a", "smoke-b", "official"), required=True)
    for command in ("run-smoke-a", "run-smoke-b", "run-official"):
        run = commands.add_parser(command)
        run.add_argument("--reviewed-commit", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "readiness":
            payload = readiness_exp449_antigravity(args.reviewed_commit)
        elif args.command == "status":
            payload = status_exp449_read_only_antigravity()
        elif args.command == "verify-proposal":
            payload = verify_exp449_proposal_read_only_antigravity(args.target)
        else:
            target = {"run-smoke-a": "smoke_a", "run-smoke-b": "smoke_b", "run-official": "official"}[args.command]
            _require_execution_authority_antigravity(target, args.reviewed_commit)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


__all__ = [
    "Exp449SmokeAExecutionLeaseAntigravity",
    "Exp449WaveConfig",
    "acquire_smoke_a_execution_lease_antigravity",
    "append_authorized_progress_antigravity",
    "freeze_authorized_exp449_proposal_antigravity",
    "main_antigravity",
    "readiness_for_authorized_implementation_antigravity",
    "readiness_exp449_antigravity",
    "run_exp449_wave_antigravity",
    "record_exp449_smoke_comparison_antigravity",
    "refresh_derived_status_antigravity",
    "status_exp449_read_only_antigravity",
    "status_exp449_antigravity",
    "task45_context_payload_antigravity",
    "verify_exp449_proposal_read_only_antigravity",
    "verify_terminal_incomplete_antigravity",
]
