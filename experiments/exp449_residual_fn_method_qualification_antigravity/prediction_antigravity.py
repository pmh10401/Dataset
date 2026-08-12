"""Frozen, label-free Exp449 prediction lifecycle; official entry is intentionally guarded."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity, write_create_only_antigravity
from .progress_antigravity import (
    commit_candidate_leaf_antigravity, initialise_attempt_antigravity,
    record_verified_progress_antigravity, verified_prefix_antigravity,
    write_failure_audit_antigravity,
)


_OFFICIAL_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812")
_LOGICAL_CPUS_ANTIGRAVITY = 12
_WORKERS_ANTIGRAVITY = 11


def _candidate_payload_antigravity(sequence: int, dataset: str, family: str, method: str, indices: list[int], target_length: int, train_count: int, score_identities: list[dict[str, int]]) -> dict[str, object]:
    """The one canonical worker/fixture candidate representation accepted by validation."""
    return {"sequence": sequence, "candidate_name": f"{dataset}__{method}", "dataset": dataset,
            "family": family, "method": method, "indices": indices, "contains_test_labels": False,
            "target_length": target_length, "train_count": train_count, "test_count": target_length,
            "seeds": list(FROZEN_CONFIG_ANTIGRAVITY["seeds"]), "score_identities": score_identities,
            "alpha": 0.005}


def _semantic_context_antigravity(reviewed_commit: str, *, source_identity: object, method_audit_sha256: str, mlx: object) -> dict[str, object]:
    """Canonical label-free context that the manifest and descriptor validator both bind."""
    return {
        "schema": "exp449", "reviewed_commit": reviewed_commit,
        "logical_cpus": _LOGICAL_CPUS_ANTIGRAVITY, "workers": _WORKERS_ANTIGRAVITY,
        "label_access_count": 0, "source_identity": source_identity,
        "candidate_matrix": [[family, list(methods)] for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY],
        "seeds": list(FROZEN_CONFIG_ANTIGRAVITY["seeds"]), "alpha": 0.005,
        "method_audit_sha256": method_audit_sha256, "mlx": mlx,
    }


def _fault_antigravity(hook: Callable[[str, int], None] | None, tag: str, sequence: int) -> None:
    """Private test-only seam placed immediately at a named durable lifecycle boundary."""
    if hook is not None:
        hook(tag, sequence)


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
    return sequence, name, _candidate_payload_antigravity(sequence, dataset, family, method, indices, bundle.target_length, int(len(bundle.train_normal)), [{"seed": row.seed, "feature_dimension": row.feature_dimension} for row in rows])


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


def _remove_exact_manifest_antigravity(root: Path, identity: tuple[int, int]) -> None:
    """Remove only the manifest inode just committed, then persist its namespace change."""
    parent = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.stat("prediction_manifest_antigravity.json", dir_fd=parent, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == identity:
            os.unlink("prediction_manifest_antigravity.json", dir_fd=parent)
            os.fsync(parent)
    finally:
        os.close(parent)


def _assert_terminal_channels_antigravity(root: Path, inventory: tuple[str, ...]) -> None:
    """Gate manifest creation on leaf-derived terminal progress and a durably inactive PID."""
    if verified_prefix_antigravity(root / "candidates_antigravity", inventory) != len(inventory):
        raise ValueError("terminal candidate prefix differs")
    if strict_json_bytes_antigravity((root / "progress_antigravity.json").read_bytes()) != {"dataset_unit_completed": len(inventory), "next_sequence": len(inventory), "terminal": True}:
        raise ValueError("terminal progress differs")
    if strict_json_bytes_antigravity((root / "pid_antigravity.json").read_bytes()) != {"active": False}:
        raise ValueError("terminal PID differs")
    events = (root / "progress_antigravity.jsonl").read_bytes().splitlines()
    text = (root / "progress_antigravity.txt").read_bytes().splitlines()
    if len(events) != len(inventory) or len(text) != len(inventory) or any(strict_json_bytes_antigravity(line + b"\n") != {"dataset_unit_completed": index} for index, line in enumerate(events, 1)) or text != [str(index).encode() for index in range(1, len(inventory) + 1)]:
        raise ValueError("terminal event or text differs")


def run_synthetic_lifecycle_antigravity(root: Path, reviewed_commit: str, *, fail_after_leaf: int | None = None, audit_path: Path | None = None, fault_hook: Callable[[str, int], None] | None = None) -> "FrozenPredictionAuthorityAntigravity":
    """Private test lifecycle with real fixed workers and parent-only ordered durable commits."""
    if not isinstance(root, Path) or type(reviewed_commit) is not str or len(reviewed_commit) != 40:
        raise ValueError("synthetic lifecycle inputs are invalid")
    if fail_after_leaf is not None and (type(fail_after_leaf) is not int or not 0 <= fail_after_leaf < 96):
        raise ValueError("synthetic failure sequence is invalid")
    inventory = tuple(f"{sequence:03d}_synthetic_{sequence:03d}.json" for sequence in range(96))
    synthetic_families = tuple(family for family, count in zip(tuple(name for name, _ in CANDIDATE_MATRIX_ANTIGRAVITY), (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
    synthetic_names = tuple(f"{family}_synthetic_{index:03d}" for index, family in enumerate(synthetic_families))
    try:
        # Synthetic data use the production descriptor shape; their fixed digests are
        # opaque identities, never a source/database escape hatch.
        context = _semantic_context_antigravity(
            reviewed_commit,
            source_identity={"artifact_map_sha256": "1" * 64, "evaluation_sha256": "2" * 64,
                             "decision_sha256": "3" * 64, "target_dataset_names": list(synthetic_names)},
            method_audit_sha256="0" * 64,
            mlx={"backend": "cpu", "reason": "synthetic_cpu", "tensor_eligible": False,
                 "mlx_version": None, "mlx_metal_version": None, "torch_version": None,
                 "mps_built": None, "mps_available": None, "device": None, "parity_run": False,
                 "candidate_indices_identical": None, "numerical_tolerance_met": None,
                 "benchmark_includes_transfer_sync_warmup_compile": False, "speedup": None},
        )
        initialise_attempt_antigravity(root, context)
        _fault_antigravity(fault_hook, "worker_exception", 0)
        _fault_antigravity(fault_hook, "scorer_exception", 0)
        _fault_antigravity(fault_hook, "nonfinite_scores", 0)
        ordered, _completed, _provenance = schedule_synthetic_units_antigravity(tuple(range(96)))
        leaves = root / "candidates_antigravity"
        digests: list[tuple[str, str]] = []
        for sequence in ordered:
            _fault_antigravity(fault_hook, "predecessor_drift", sequence)
            _fault_antigravity(fault_hook, "before_commit", sequence)
            family = synthetic_families[sequence // 2]; dataset = f"{family}_synthetic_{sequence // 2:03d}"; method = dict(CANDIDATE_MATRIX_ANTIGRAVITY)[family][sequence % 2]
            payload = _candidate_payload_antigravity(sequence, dataset, family, method, [], 3, 5, [{"seed": 20260721, "feature_dimension": 1}, {"seed": 20261721, "feature_dimension": 1}])
            digest = commit_candidate_leaf_antigravity(leaves, inventory[sequence], payload, fault_hook=fault_hook, sequence=sequence)
            digests.append(("candidates_antigravity/" + inventory[sequence], digest))
            _fault_antigravity(fault_hook, "after_commit", sequence)
            record_verified_progress_antigravity(root, inventory, sequence + 1, fault_hook=fault_hook)
            if sequence == fail_after_leaf:
                raise RuntimeError(f"synthetic post-leaf failure exact_next_sequence={sequence + 1}")
        _fault_antigravity(fault_hook, "pid_close", 96)
        _fault_antigravity(fault_hook, "pid", 96)
        _set_pid_inactive_antigravity(root)
        _assert_terminal_channels_antigravity(root, inventory)
        aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
        authority = FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(f"{synthetic_families[sequence // 2]}_synthetic_{sequence // 2:03d}__{dict(CANDIDATE_MATRIX_ANTIGRAVITY)[synthetic_families[sequence // 2]][sequence % 2]}" for sequence in range(96)), tuple(name for name, _ in digests), tuple(digests), aggregate, 0)
        validate_prediction_root_antigravity(root, authority, require_manifest=False)
        manifest = {"schema": "exp449_prediction_manifest", "reviewed_commit": reviewed_commit, "inventory": [name for name, _ in digests], "leaf_sha256": dict(digests), "aggregate_sha256": aggregate, "context_sha256": hashlib.sha256(canonical_json_bytes_antigravity(context)).hexdigest(), "label_access_count": 0, "dataset_units": 96}
        commit_candidate_leaf_antigravity(root, "prediction_manifest_antigravity.json", manifest, fault_hook=fault_hook, sequence=96, fault_prefix="manifest_")
        manifest_info = os.stat(root / "prediction_manifest_antigravity.json", follow_symlinks=False)
        try:
            _fault_antigravity(fault_hook, "post_manifest_validation", 96)
            validate_prediction_root_antigravity(root, authority)
        except BaseException:
            _remove_exact_manifest_antigravity(root, (manifest_info.st_dev, manifest_info.st_ino))
            raise
        return authority
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            if root.exists() and (root / "pid_antigravity.json").exists():
                _set_pid_inactive_antigravity(root)
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            primary_error.add_note(f"pid_cleanup_error={type(cleanup_error).__name__}: {cleanup_error}")
        target_audit = audit_path if audit_path is not None else root.parent / "exp449_failure_audit_antigravity.json"
        try:
            prefix = verified_prefix_antigravity(root / "candidates_antigravity", tuple(f"{sequence:03d}_synthetic_{sequence:03d}.json" for sequence in range(96))) if (root / "candidates_antigravity").is_dir() else 0
            _fault_antigravity(fault_hook, "failure_audit", prefix)
            write_failure_audit_antigravity(target_audit, primary_error, cleanup_error, exact_next_sequence=prefix)
        except BaseException as audit_error:
            # Audit persistence is evidence, never a replacement for the run's cause.
            primary_error.add_note(f"failure_audit_error={type(audit_error).__name__}: {audit_error}")
        raise primary_error


def _read_nofollow_json_antigravity(path: Path) -> object:
    """Read one regular single-link canonical JSON leaf with no symlink traversal."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("prediction leaf namespace is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if opened.st_ino != info.st_ino or opened.st_dev != info.st_dev or opened.st_nlink != 1:
            raise ValueError("prediction leaf namespace changed")
        payload = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    if current.st_ino != info.st_ino or current.st_dev != info.st_dev or current.st_size != info.st_size:
        raise ValueError("prediction leaf namespace changed")
    return strict_json_bytes_antigravity(payload)


def _read_nofollow_bytes_antigravity(path: Path) -> bytes:
    """Read a retained regular single-link status channel without interpreting it as a path."""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("prediction channel namespace is unsafe")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = b"".join(iter(lambda: os.read(descriptor, 1 << 20), b""))
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size) or (current.st_dev, current.st_ino, current.st_size) != (info.st_dev, info.st_ino, info.st_size):
        raise ValueError("prediction channel namespace changed")
    return payload


def _fd_identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_pinned_antigravity(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(iter(lambda: os.read(fd, 1 << 20), b""))


def validate_prediction_root_antigravity(root: Path, authority: object, *, require_manifest: bool = True) -> tuple[str, ...]:
    """Validate every leaf under one retained no-follow root/candidates descriptor transaction."""
    if not isinstance(root, Path) or type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction root authority is invalid")
    if (type(authority.reviewed_commit) is not str or type(authority.family_method_pairs) is not int or type(authority.dataset_units) is not int
            or type(authority.candidate_names) is not tuple or type(authority.inventory) is not tuple or type(authority.leaf_sha256) is not tuple
            or type(authority.aggregate_sha256) is not str or type(authority.label_access_count) is not int):
        raise ValueError("prediction authority field type differs")
    if (authority.family_method_pairs != 16 or authority.dataset_units != 96 or authority.label_access_count != 0
            or len(authority.reviewed_commit) != 40 or authority.reviewed_commit.lower() != authority.reviewed_commit
            or len(authority.inventory) != 96 or len(authority.candidate_names) != 96 or len(authority.leaf_sha256) != 96
            or len(authority.aggregate_sha256) != 64):
        raise ValueError("prediction authority field value differs")
    root_fd = candidates_fd = -1
    pinned: dict[str, tuple[int, tuple[int, int, int, int, int, int, int], int]] = {}
    expected_root = {"context_antigravity.json", "progress_antigravity.json", "progress_antigravity.txt", "progress_antigravity.jsonl", "pid_antigravity.json", "candidates_antigravity"} | ({"prediction_manifest_antigravity.json"} if require_manifest else set())
    expected_candidates = tuple(Path(relative).name for relative in authority.inventory)
    try:
        root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        root_identity = _fd_identity_antigravity(os.fstat(root_fd))
        if not stat.S_ISDIR(root_identity[2]) or set(os.listdir(root_fd)) != expected_root:
            raise ValueError("prediction root inventory differs")
        candidates_fd = os.open("candidates_antigravity", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=root_fd)
        candidates_identity = _fd_identity_antigravity(os.fstat(candidates_fd))
        if not stat.S_ISDIR(candidates_identity[2]) or len(expected_candidates) != 96 or len(set(expected_candidates)) != 96 or set(os.listdir(candidates_fd)) != set(expected_candidates):
            raise ValueError("prediction candidate inventory differs")
        def pin(name: str, parent_fd: int, parent_tag: int) -> bytes:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            key = f"{parent_tag}:{name}"; pinned[key] = (fd, (), parent_tag)
            opened = os.fstat(fd); identity = _fd_identity_antigravity(opened)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or _fd_identity_antigravity(before) != identity:
                raise ValueError("prediction leaf namespace is unsafe")
            pinned[key] = (fd, identity, parent_tag)
            return _read_pinned_antigravity(fd)
        leaves = {name: pin(name, root_fd, 0) for name in expected_root - {"candidates_antigravity"}}
        candidate_bytes = {name: pin(name, candidates_fd, 1) for name in expected_candidates}
        context = strict_json_bytes_antigravity(leaves["context_antigravity.json"])
        progress = strict_json_bytes_antigravity(leaves["progress_antigravity.json"])
        pid = strict_json_bytes_antigravity(leaves["pid_antigravity.json"])
        manifest = strict_json_bytes_antigravity(leaves["prediction_manifest_antigravity.json"]) if require_manifest else None
        if not all(type(value) is dict for value in (context, progress, pid)) or (require_manifest and type(manifest) is not dict): raise ValueError("prediction root JSON schema differs")
        expected_matrix = [[family, list(methods)] for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY]
        context_keys = {"schema", "reviewed_commit", "logical_cpus", "workers", "label_access_count", "source_identity", "candidate_matrix", "seeds", "alpha", "method_audit_sha256", "mlx"}
        source_keys = {"artifact_map_sha256", "evaluation_sha256", "decision_sha256", "target_dataset_names"}
        mlx_keys = {"backend", "reason", "tensor_eligible", "mlx_version", "mlx_metal_version", "torch_version", "mps_built", "mps_available", "device", "parity_run", "candidate_indices_identical", "numerical_tolerance_met", "benchmark_includes_transfer_sync_warmup_compile", "speedup"}
        source = context.get("source_identity")
        mlx = context.get("mlx")
        source_invalid = (type(source) is not dict or set(source) != source_keys or any(type(source.get(key)) is not str or len(source[key]) != 64 or any(character not in "0123456789abcdef" for character in source[key]) for key in ("artifact_map_sha256", "evaluation_sha256", "decision_sha256")) or type(source.get("target_dataset_names")) is not list or any(type(name) is not str or not name for name in source["target_dataset_names"]))
        mlx_invalid = (type(mlx) is not dict or set(mlx) != mlx_keys or mlx.get("backend") not in ("cpu", "mlx", "mps") or type(mlx.get("reason")) is not str or not mlx["reason"] or type(mlx.get("tensor_eligible")) is not bool or type(mlx.get("parity_run")) is not bool or type(mlx.get("benchmark_includes_transfer_sync_warmup_compile")) is not bool or any(mlx.get(key) is not None and type(mlx.get(key)) is not str for key in ("mlx_version", "mlx_metal_version", "torch_version", "device")) or any(mlx.get(key) is not None and type(mlx.get(key)) is not bool for key in ("mps_built", "mps_available", "candidate_indices_identical", "numerical_tolerance_met")) or (mlx.get("speedup") is not None and type(mlx.get("speedup")) is not float))
        if (set(context) != context_keys or context.get("schema") != "exp449" or context.get("reviewed_commit") != authority.reviewed_commit or type(context.get("reviewed_commit")) is not str or len(context["reviewed_commit"]) != 40 or type(context.get("label_access_count")) is not int or context.get("label_access_count") != 0 or type(context.get("logical_cpus")) is not int or context.get("logical_cpus") != 12 or type(context.get("workers")) is not int or context.get("workers") != 11 or context.get("candidate_matrix") != expected_matrix or context.get("seeds") != [20260721, 20261721] or type(context.get("alpha")) is not float or context.get("alpha") != 0.005 or type(context.get("method_audit_sha256")) is not str or len(context["method_audit_sha256"]) != 64 or source_invalid or mlx_invalid or progress != {"dataset_unit_completed": 96, "next_sequence": 96, "terminal": True} or pid != {"active": False}): raise ValueError("prediction root terminal channels differ")
        event_records = tuple(strict_json_bytes_antigravity(line + b"\n") for line in leaves["progress_antigravity.jsonl"].splitlines())
        text_records = leaves["progress_antigravity.txt"].splitlines()
        if len(event_records) != 96 or any(type(record) is not dict or record != {"dataset_unit_completed": index} for index, record in enumerate(event_records, 1)) or text_records != [str(index).encode() for index in range(1, 97)]: raise ValueError("prediction event or text channel differs")
        digests: list[tuple[str, str]] = []; names: list[str] = []
        for sequence, relative in enumerate(authority.inventory):
            if type(relative) is not str or not relative.startswith("candidates_antigravity/"): raise ValueError("prediction authority inventory differs")
            payload = strict_json_bytes_antigravity(candidate_bytes[Path(relative).name])
            candidate_keys = {"sequence", "candidate_name", "dataset", "family", "method", "indices", "contains_test_labels", "target_length", "train_count", "test_count", "seeds", "score_identities", "alpha"}
            invalid = (type(payload) is not dict or set(payload) != candidate_keys or type(payload.get("sequence")) is not int or payload["sequence"] != sequence or type(payload.get("candidate_name")) is not str or type(payload.get("dataset")) is not str or type(payload.get("family")) is not str or type(payload.get("method")) is not str or payload.get("candidate_name") != payload["dataset"] + "__" + payload["method"] or payload.get("contains_test_labels") is not False or type(payload.get("target_length")) is not int or type(payload.get("train_count")) is not int or type(payload.get("test_count")) is not int or min(payload["target_length"], payload["train_count"], payload["test_count"]) < 1 or payload.get("seeds") != [20260721, 20261721] or type(payload.get("alpha")) is not float or payload["alpha"] != 0.005 or type(payload.get("indices")) is not list or any(type(index) is not int or index < 0 or index >= payload["target_length"] for index in payload["indices"]) or payload["indices"] != sorted(set(payload["indices"])))
            identities = payload.get("score_identities") if type(payload) is dict else None
            if invalid or type(identities) is not list or len(identities) != 2 or any(type(item) is not dict or set(item) != {"seed", "feature_dimension"} or type(item["seed"]) is not int or type(item["feature_dimension"]) is not int or item["feature_dimension"] < 1 for item in identities) or [item["seed"] for item in identities] != [20260721, 20261721]: raise ValueError("prediction candidate leaf differs")
            expected_family = next((family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY if payload["dataset"].startswith(family + "_")), None)
            if expected_family != payload["family"] or payload["method"] not in dict(CANDIDATE_MATRIX_ANTIGRAVITY).get(payload["family"], ()): raise ValueError("prediction candidate binding differs")
            digests.append((relative, hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest())); names.append(payload["candidate_name"])
        aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
        if tuple(digests) != authority.leaf_sha256 or tuple(names) != authority.candidate_names or aggregate != authority.aggregate_sha256: raise ValueError("prediction authority digest differs")
        if require_manifest and manifest != {"schema": "exp449_prediction_manifest", "reviewed_commit": authority.reviewed_commit, "inventory": [name for name, _ in digests], "leaf_sha256": dict(digests), "aggregate_sha256": aggregate, "context_sha256": hashlib.sha256(canonical_json_bytes_antigravity(context)).hexdigest(), "label_access_count": 0, "dataset_units": 96}: raise ValueError("prediction manifest differs")
        datasets = tuple(name.split("__", 1)[0] for name in names[::2]); candidate_pairs = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
        def expected_pair(dataset: str) -> list[str]:
            if dataset.startswith("synthetic_dataset_"): return [dataset + "__method_a", dataset + "__method_b"]
            family = next((candidate_family for candidate_family in candidate_pairs if dataset.startswith(candidate_family + "_")), None)
            return [] if family is None else [dataset + "__" + method for method in candidate_pairs[family]]
        if len(datasets) != 48 or len(set(datasets)) != 48 or any(names[index:index + 2] != expected_pair(dataset) for index, dataset in zip(range(0, 96, 2), datasets, strict=True)): raise ValueError("prediction dataset order differs")
        if source["target_dataset_names"] != list(datasets):
            raise ValueError("prediction source target order differs")
        for key, (fd, identity, parent_tag) in pinned.items():
            name = key.split(":", 1)[1]; parent_fd = root_fd if parent_tag == 0 else candidates_fd
            if _fd_identity_antigravity(os.fstat(fd)) != identity or _fd_identity_antigravity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity: raise ValueError("prediction leaf namespace changed")
        if _fd_identity_antigravity(os.fstat(root_fd)) != root_identity or _fd_identity_antigravity(os.fstat(candidates_fd)) != candidates_identity or set(os.listdir(root_fd)) != expected_root or set(os.listdir(candidates_fd)) != set(expected_candidates): raise ValueError("prediction directory namespace changed")
        if _fd_identity_antigravity(os.stat(root, follow_symlinks=False)) != root_identity: raise ValueError("prediction root namespace changed")
        return datasets
    finally:
        for fd, _identity, _parent in pinned.values(): os.close(fd)
        if candidates_fd >= 0: os.close(candidates_fd)
        if root_fd >= 0: os.close(root_fd)


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
    predecessor_capture: Callable[[], object] | None = None


def _families_antigravity(authority: object) -> tuple[str, ...]:
    rows = getattr(authority, "target_baseline_rows", None)
    if type(rows) is not tuple or len(rows) != 48:
        raise ValueError("Exp448 target authority differs")
    families = tuple(getattr(row, "family", None) for row in rows)
    expected = tuple(family for family, _ in CANDIDATE_MATRIX_ANTIGRAVITY)
    if any(type(family) is not str for family in families) or tuple(sorted(set(families), key=expected.index)) != expected:
        raise ValueError("Exp448 family order differs")
    return families


def _revalidate_predecessor_before_commit_antigravity(original: object, capture: Callable[[], object]) -> None:
    """Require a fresh descriptor authority equal to the pre-scheduling Exp448 capture before every leaf."""
    fresh = capture()
    if type(fresh) is not type(original) or fresh != original:
        raise ValueError("Exp448 predecessor drift before candidate commit")


def _consume_pending_executor_antigravity(futures: dict[object, int], authority: object, capture: Callable[[], object], commit: Callable[[int, object], None], sequence_of: Callable[[object], int]) -> None:
    """The sole ProcessPool completion/pending/ordered-commit loop used by production and its private seam."""
    pending: dict[int, object] = {}
    next_sequence = 0
    for future in as_completed(futures):
        result = future.result()
        sequence = sequence_of(result)
        if type(sequence) is not int or sequence not in futures.values() or sequence in pending:
            raise ValueError("production worker sequence differs")
        pending[sequence] = result
        while next_sequence in pending:
            _revalidate_predecessor_before_commit_antigravity(authority, capture)
            commit(next_sequence, pending.pop(next_sequence))
            next_sequence += 1
    if next_sequence != len(futures):
        raise RuntimeError("production worker completion did not reach the frozen inventory")


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
    unit_sequence = 0
    for dataset_sequence, family in enumerate(families):
        dataset, train, test = dependencies.load_bundle(dataset_sequence, family)
        for method in candidates[family]:
            if dependencies.predecessor_capture is not None:
                _revalidate_predecessor_before_commit_antigravity(dependencies.exp448_authority, dependencies.predecessor_capture)
            name = f"{dataset}__{method}"
            relative = f"{unit_sequence:03d}_{name}.json"
            score_rows = [dependencies.score(train, test, method, seed) for seed in FROZEN_CONFIG_ANTIGRAVITY["seeds"]]
            train_median = np.median(np.stack([item[0] for item in score_rows]), axis=0)
            test_median = np.median(np.stack([item[1] for item in score_rows]), axis=0)
            if not np.isfinite(train_median).all() or not np.isfinite(test_median).all():
                raise ValueError("non-finite score")
            indices = tuple(int(item) for item in empirical_candidates(train_median, test_median, 0.005).indices)
            payload = _candidate_payload_antigravity(unit_sequence, dataset, family, method, list(indices), int(len(test)), int(len(train)), [{"seed": seed, "feature_dimension": int(train.shape[1])} for seed in FROZEN_CONFIG_ANTIGRAVITY["seeds"]])
            digest = commit_candidate_leaf_antigravity(leaves, relative, payload)
            names.append(name)
            inventory.append("candidates_antigravity/" + relative)
            digests.append(("candidates_antigravity/" + relative, digest))
            unit_sequence += 1
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
    inventory: tuple[str, ...] = ()
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
        audit_payload = [asdict(item) for item in audit_methods_antigravity()]
        audit_sha256 = hashlib.sha256(canonical_json_bytes_antigravity(audit_payload)).hexdigest()
        mlx_payload = asdict(assess_mlx_antigravity())
        source_identity = {"artifact_map_sha256": predecessor.artifact_map_sha256, "evaluation_sha256": predecessor.evaluation_sha256, "decision_sha256": predecessor.decision_sha256, "target_dataset_names": list(predecessor.target_dataset_names)}
        context = _semantic_context_antigravity(reviewed_commit, source_identity=source_identity, method_audit_sha256=audit_sha256, mlx=mlx_payload)
        initialise_attempt_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, context)
        names: list[str] = []
        digests: list[tuple[str, str]] = []
        with ProcessPoolExecutor(max_workers=_WORKERS_ANTIGRAVITY) as executor:
            futures = {executor.submit(_production_worker_antigravity, sequence, dataset, family, method): sequence for sequence, (dataset, family, method) in enumerate(jobs)}
            def commit(sequence: int, result: object) -> None:
                item_sequence, item_name, item_payload = result
                if item_sequence != sequence:
                    raise ValueError("production worker sequence differs")
                digest = commit_candidate_leaf_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity", inventory[sequence], item_payload)
                names.append(item_name); digests.append(("candidates_antigravity/" + inventory[sequence], digest))
                record_verified_progress_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, inventory, sequence + 1)
            _consume_pending_executor_antigravity(futures, predecessor, capture_exp448_authority_antigravity, commit, lambda result: result[0])
        _set_pid_inactive_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
        _assert_terminal_channels_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, inventory)
        aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
        authority = FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(names), tuple(name for name, _ in digests), tuple(digests), aggregate, 0)
        validate_prediction_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, authority, require_manifest=False)
        manifest = {"schema": "exp449_prediction_manifest", "reviewed_commit": reviewed_commit, "inventory": [name for name, _ in digests], "leaf_sha256": dict(digests), "aggregate_sha256": aggregate, "context_sha256": hashlib.sha256(canonical_json_bytes_antigravity(context)).hexdigest(), "label_access_count": 0, "dataset_units": 96}
        commit_candidate_leaf_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, "prediction_manifest_antigravity.json", manifest)
        manifest_info = os.stat(_OFFICIAL_ROOT_ANTIGRAVITY / "prediction_manifest_antigravity.json", follow_symlinks=False)
        try:
            validate_prediction_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, authority)
        except BaseException:
            _remove_exact_manifest_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, (manifest_info.st_dev, manifest_info.st_ino))
            raise
        return authority
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            if (_OFFICIAL_ROOT_ANTIGRAVITY / "pid_antigravity.json").exists():
                _set_pid_inactive_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY)
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            primary_error.add_note(f"pid_cleanup_error={type(cleanup_error).__name__}: {cleanup_error}")
        prefix = verified_prefix_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity", inventory) if inventory and (_OFFICIAL_ROOT_ANTIGRAVITY / "candidates_antigravity").is_dir() else 0
        try:
            write_failure_audit_antigravity(audit_path, primary_error, cleanup_error, exact_next_sequence=prefix)
        except BaseException as audit_error:
            primary_error.add_note(f"failure_audit_error={type(audit_error).__name__}: {audit_error}")
        raise primary_error


def validate_frozen_prediction_authority_antigravity(authority: object) -> tuple[str, ...]:
    """Fail closed without opening a database; official root validation is intentionally descriptor-only."""
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction authority is invalid")
    if not _OFFICIAL_ROOT_ANTIGRAVITY.is_dir():
        raise ValueError("official prediction root is unavailable")
    if authority.family_method_pairs != 16 or authority.dataset_units != 96 or authority.label_access_count != 0:
        raise ValueError("prediction authority is invalid")
    return validate_prediction_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, authority)


__all__ = ["FrozenPredictionAuthorityAntigravity", "PredictionDependenciesAntigravity", "predict_antigravity", "predict_with_dependencies_antigravity", "schedule_synthetic_units_antigravity", "validate_frozen_prediction_authority_antigravity"]
