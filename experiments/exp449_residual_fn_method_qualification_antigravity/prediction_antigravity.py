"""Frozen, label-free Exp449 prediction lifecycle; official entry is intentionally guarded."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import stat
from pathlib import Path
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates

from .config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY, FROZEN_CONFIG_ANTIGRAVITY
from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .progress_antigravity import (
    commit_candidate_leaf_antigravity, initialise_attempt_antigravity,
    record_verified_progress_antigravity, verified_prefix_antigravity,
    write_failure_audit_antigravity,
)


_OFFICIAL_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812")
_LOGICAL_CPUS_ANTIGRAVITY = 12
_WORKERS_ANTIGRAVITY = 11


def prepare_attempt_root_antigravity(root: Path) -> None:
    """Reject every preexisting namespace before a lifecycle can create its attempt root."""
    if not isinstance(root, Path):
        raise ValueError("attempt root is invalid")
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        root.mkdir(mode=0o700)
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or os.listdir(root):
        raise ValueError("official attempt root already exists or is unsafe")
    raise ValueError("official attempt root already exists")


def _synthetic_worker_antigravity(sequence: int) -> int:
    """Picklable process worker with deterministic inverse latency for scheduling regression."""
    time.sleep((95 - sequence % 96) * 0.0001)
    return sequence


def _production_worker_antigravity(sequence: int, dataset: str, family: str, method: str) -> tuple[int, str, dict[str, object]]:
    """Picklable label-free Task-3/4 worker; parent alone writes every durable artifact."""
    from .contracts_antigravity import MethodCandidateAntigravity
    from .scoring_antigravity import score_candidate_antigravity
    from .sources_antigravity import load_target_bundle_antigravity

    bundle = load_target_bundle_antigravity(dataset)
    candidate = MethodCandidateAntigravity(method, method, tuple(FROZEN_CONFIG_ANTIGRAVITY["seeds"]), (0.005,))
    rows = tuple(score_candidate_antigravity(bundle.train_normal, bundle.test_values, candidate, seed=seed) for seed in FROZEN_CONFIG_ANTIGRAVITY["seeds"])
    train_median = np.median(np.stack([row.train_oof for row in rows]), axis=0)
    test_median = np.median(np.stack([row.test_scores for row in rows]), axis=0)
    if not np.isfinite(train_median).all() or not np.isfinite(test_median).all():
        raise ValueError("non-finite score")
    indices = [int(index) for index in empirical_candidates(train_median, test_median, 0.005).indices]
    name = f"{dataset}__{method}"
    return sequence, name, {
        "sequence": sequence, "candidate_name": name, "family": family, "method": method,
        "contains_test_labels": False, "target_length": bundle.target_length,
        "train_count": int(len(bundle.train_normal)), "test_count": int(len(bundle.test_values)),
        "score_identities": [{"seed": row.seed, "feature_dimension": row.feature_dimension} for row in rows],
        "indices": indices, "alpha": 0.005,
    }


def schedule_synthetic_units_antigravity(sequences: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, int]]:
    """Exercise the exact fixed-size process scheduler; parent emits only contiguous order."""
    if type(sequences) is not tuple or sequences != tuple(range(96)):
        raise ValueError("synthetic scheduler requires the frozen 96 sequences")
    completed: list[int] = []
    pending: dict[int, int] = {}
    ordered: list[int] = []
    next_sequence = 0
    with ProcessPoolExecutor(max_workers=_WORKERS_ANTIGRAVITY) as executor:
        futures = {executor.submit(_synthetic_worker_antigravity, sequence): sequence for sequence in sequences}
        for future in as_completed(futures):
            sequence = future.result()
            completed.append(sequence)
            pending[sequence] = sequence
            while next_sequence in pending:
                ordered.append(pending.pop(next_sequence))
                next_sequence += 1
    if tuple(ordered) != sequences:
        raise RuntimeError("scheduler did not commit a contiguous frozen prefix")
    return tuple(ordered), tuple(completed), {"logical_cpus": _LOGICAL_CPUS_ANTIGRAVITY, "workers": _WORKERS_ANTIGRAVITY}


def _set_pid_inactive_antigravity(root: Path) -> None:
    """Close the lifecycle marker with a no-follow, durable replacement payload."""
    path = root / "pid_antigravity.json"
    payload = canonical_json_bytes_antigravity({"active": False})
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short PID marker write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_synthetic_lifecycle_antigravity(root: Path, reviewed_commit: str, *, fail_after_leaf: int | None = None, audit_path: Path | None = None) -> "FrozenPredictionAuthorityAntigravity":
    """Private test lifecycle with real fixed workers and parent-only ordered durable commits."""
    if not isinstance(root, Path) or type(reviewed_commit) is not str or len(reviewed_commit) != 40:
        raise ValueError("synthetic lifecycle inputs are invalid")
    if fail_after_leaf is not None and (type(fail_after_leaf) is not int or not 0 <= fail_after_leaf < 96):
        raise ValueError("synthetic failure sequence is invalid")
    inventory = tuple(f"{sequence:03d}_synthetic_{sequence:03d}.json" for sequence in range(96))
    try:
        initialise_attempt_antigravity(root, {"schema": "exp449", "reviewed_commit": reviewed_commit, "logical_cpus": _LOGICAL_CPUS_ANTIGRAVITY, "workers": _WORKERS_ANTIGRAVITY, "label_access_count": 0})
        ordered, _completed, _provenance = schedule_synthetic_units_antigravity(tuple(range(96)))
        leaves = root / "candidates_antigravity"
        digests: list[tuple[str, str]] = []
        for sequence in ordered:
            digest = commit_candidate_leaf_antigravity(leaves, inventory[sequence], {"sequence": sequence, "candidate_name": f"synthetic_{sequence:03d}", "contains_test_labels": False, "scores": {"20260721": [0.0], "20261721": [0.0]}, "alpha": 0.005})
            digests.append(("candidates_antigravity/" + inventory[sequence], digest))
            record_verified_progress_antigravity(root, inventory, sequence + 1)
            if sequence == fail_after_leaf:
                raise RuntimeError(f"synthetic post-leaf failure exact_next_sequence={sequence + 1}")
        _set_pid_inactive_antigravity(root)
        if verified_prefix_antigravity(leaves, inventory) != 96:
            raise RuntimeError("terminal candidate prefix is incomplete")
        aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
        manifest = {"schema": "exp449_prediction_manifest", "reviewed_commit": reviewed_commit, "inventory": [name for name, _ in digests], "leaf_sha256": dict(digests), "aggregate_sha256": aggregate, "label_access_count": 0, "dataset_units": 96}
        write_create_only_antigravity(root / "prediction_manifest_antigravity.json", canonical_json_bytes_antigravity(manifest))
        return FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(f"synthetic_{sequence:03d}" for sequence in range(96)), tuple(name for name, _ in digests), tuple(digests), aggregate, 0)
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            if root.exists() and (root / "pid_antigravity.json").exists():
                _set_pid_inactive_antigravity(root)
        except BaseException as error:
            cleanup_error = error
        target_audit = audit_path if audit_path is not None else root.parent / "exp449_failure_audit_antigravity.json"
        try:
            prefix = verified_prefix_antigravity(root / "candidates_antigravity", tuple(f"{sequence:03d}_synthetic_{sequence:03d}.json" for sequence in range(96))) if (root / "candidates_antigravity").is_dir() else 0
            write_failure_audit_antigravity(target_audit, primary_error, cleanup_error, exact_next_sequence=prefix)
        except BaseException as audit_error:
            raise RuntimeError(f"{primary_error}; failure_audit_error={audit_error}") from primary_error
        raise


@dataclass(frozen=True, slots=True)
class FrozenPredictionAuthorityAntigravity:
    reviewed_commit: str
    family_method_pairs: int
    dataset_units: int
    candidate_names: tuple[str, ...]
    inventory: tuple[str, ...]
    leaf_sha256: tuple[tuple[str, str], ...]
    aggregate_sha256: str
    label_access_count: int


@dataclass(frozen=True, slots=True)
class PredictionDependenciesAntigravity:
    root: Path
    exp448_authority: object
    load_bundle: Callable[[int, str], tuple[str, np.ndarray, np.ndarray]]
    score: Callable[[np.ndarray, np.ndarray, str, int], tuple[np.ndarray, np.ndarray]]
    method_audit: Callable[[], object]
    mlx_decision: Callable[[], object]
    executor_factory: object | None = None


def _families_antigravity(authority: object) -> tuple[str, ...]:
    rows = getattr(authority, "target_baseline_rows", None)
    if type(rows) is not tuple or len(rows) != 48:
        raise ValueError("Exp448 target authority differs")
    families = tuple(getattr(row, "family", None) for row in rows)
    expected = tuple(family for family, _ in CANDIDATE_MATRIX_ANTIGRAVITY)
    if any(type(family) is not str for family in families) or tuple(sorted(set(families), key=expected.index)) != expected:
        raise ValueError("Exp448 family order differs")
    return families


def predict_with_dependencies_antigravity(dependencies: PredictionDependenciesAntigravity, *, reviewed_commit: str = "synthetic") -> FrozenPredictionAuthorityAntigravity:
    """Run synthetic/test dependencies in frozen sequence, committing only canonical leaves."""
    if type(dependencies) is not PredictionDependenciesAntigravity or type(reviewed_commit) is not str or not reviewed_commit:
        raise ValueError("prediction dependencies are invalid")
    families = _families_antigravity(dependencies.exp448_authority)
    candidates = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
    root = dependencies.root
    leaves = root / "candidates_antigravity"
    names: list[str] = []
    inventory: list[str] = []
    digests: list[tuple[str, str]] = []
    for sequence, family in enumerate(families):
        dataset, train, test = dependencies.load_bundle(sequence, family)
        for method in candidates[family]:
            name = f"{dataset}__{method}"
            relative = f"{sequence:03d}_{name}.json"
            score_rows = [dependencies.score(train, test, method, seed) for seed in FROZEN_CONFIG_ANTIGRAVITY["seeds"]]
            train_median = np.median(np.stack([item[0] for item in score_rows]), axis=0)
            test_median = np.median(np.stack([item[1] for item in score_rows]), axis=0)
            if not np.isfinite(train_median).all() or not np.isfinite(test_median).all():
                raise ValueError("non-finite score")
            indices = tuple(int(item) for item in empirical_candidates(train_median, test_median, 0.005).indices)
            payload = {"sequence": sequence, "candidate_name": name, "family": family, "method": method, "indices": indices, "contains_test_labels": False, "target_length": int(len(test)), "train_count": int(len(train)), "test_count": int(len(test)), "seeds": list(FROZEN_CONFIG_ANTIGRAVITY["seeds"]), "alpha": 0.005}
            digest = commit_candidate_leaf_antigravity(leaves, relative, payload)
            names.append(name)
            inventory.append("candidates_antigravity/" + relative)
            digests.append(("candidates_antigravity/" + relative, digest))
    aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
    return FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(names), tuple(inventory), tuple(digests), aggregate, 0)


def predict_antigravity(reviewed_commit: str) -> FrozenPredictionAuthorityAntigravity:
    """Single-use fixed-root production lifecycle; its only dependencies are frozen Task 3/4 authorities."""
    if type(reviewed_commit) is not str or len(reviewed_commit) != 40:
        raise ValueError("reviewed commit is invalid")
    from .method_audit_antigravity import audit_methods_antigravity
    from .mlx_gate_antigravity import assess_mlx_antigravity
    from .predecessor_antigravity import capture_exp448_authority_antigravity

    prepare_attempt_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
    audit_path = Path("/Users/mac/Documents/Dataset/exp449_failure_audit_antigravity.json")
    try:
        predecessor = capture_exp448_authority_antigravity()
        families = _families_antigravity(predecessor)
        if predecessor.target_dataset_names and len(predecessor.target_dataset_names) != 48:
            raise ValueError("Exp448 target dataset names differ")
        methods = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
        jobs = tuple((dataset, family, method) for dataset, family in zip(predecessor.target_dataset_names, families, strict=True) for method in methods[family])
        if len(jobs) != 96 or len(set((dataset, method) for dataset, _family, method in jobs)) != 96:
            raise ValueError("frozen production job inventory differs")
        inventory = tuple(f"{sequence:03d}_{dataset}__{method}.json" for sequence, (dataset, _family, method) in enumerate(jobs))
        initialise_attempt_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, {"schema": "exp449", "reviewed_commit": reviewed_commit, "logical_cpus": _LOGICAL_CPUS_ANTIGRAVITY, "workers": _WORKERS_ANTIGRAVITY, "label_access_count": 0, "method_audit": [item.__dict__ if hasattr(item, "__dict__") else repr(item) for item in audit_methods_antigravity()], "mlx": repr(assess_mlx_antigravity())})
        pending: dict[int, tuple[str, dict[str, object]]] = {}
        names: list[str] = []
        digests: list[tuple[str, str]] = []
        next_sequence = 0
        with ProcessPoolExecutor(max_workers=_WORKERS_ANTIGRAVITY) as executor:
            futures = {executor.submit(_production_worker_antigravity, sequence, dataset, family, method): sequence for sequence, (dataset, family, method) in enumerate(jobs)}
            for future in as_completed(futures):
                sequence, name, payload = future.result()
                pending[sequence] = (name, payload)
                while next_sequence in pending:
                    item_name, item_payload = pending.pop(next_sequence)
                    digest = commit_candidate_leaf_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity", inventory[next_sequence], item_payload)
                    names.append(item_name); digests.append(("candidates_antigravity/" + inventory[next_sequence], digest))
                    next_sequence += 1
                    record_verified_progress_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, inventory, next_sequence)
        if next_sequence != 96:
            raise RuntimeError("production worker completion did not reach the frozen inventory")
        _set_pid_inactive_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
        aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
        manifest = {"schema": "exp449_prediction_manifest", "reviewed_commit": reviewed_commit, "inventory": [name for name, _ in digests], "leaf_sha256": dict(digests), "aggregate_sha256": aggregate, "label_access_count": 0, "dataset_units": 96}
        write_create_only_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY / "prediction_manifest_antigravity.json", canonical_json_bytes_antigravity(manifest))
        return FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(names), tuple(name for name, _ in digests), tuple(digests), aggregate, 0)
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            if (_OFFICIAL_ROOT_ANTIGRAVITY / "pid_antigravity.json").exists():
                _set_pid_inactive_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
        except BaseException as error:
            cleanup_error = error
        prefix = verified_prefix_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity", tuple(item.name for item in (_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity").glob("*.json"))) if (_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity").is_dir() else 0
        write_failure_audit_antigravity(audit_path, primary_error, cleanup_error, exact_next_sequence=prefix)
        raise


def validate_frozen_prediction_authority_antigravity(authority: object) -> tuple[str, ...]:
    """Fail closed without opening a database; official root validation is intentionally descriptor-only."""
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction authority is invalid")
    if not _OFFICIAL_ROOT_ANTIGRAVITY.is_dir():
        raise ValueError("official prediction root is unavailable")
    if authority.family_method_pairs != 16 or authority.dataset_units != 96 or authority.label_access_count != 0:
        raise ValueError("prediction authority is invalid")
    datasets = tuple(name.split("__", 1)[0] for name in authority.candidate_names[::2])
    if len(datasets) != 48 or len(set(datasets)) != 48:
        raise ValueError("prediction authority is invalid")
    return datasets


__all__ = ["FrozenPredictionAuthorityAntigravity", "PredictionDependenciesAntigravity", "predict_antigravity", "predict_with_dependencies_antigravity", "schedule_synthetic_units_antigravity", "validate_frozen_prediction_authority_antigravity"]
