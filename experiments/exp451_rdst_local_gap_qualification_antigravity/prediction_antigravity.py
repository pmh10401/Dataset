"""Create-only, descriptor-verified and resumable label-free Exp451 prediction."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates

from .artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    read_canonical_file_antigravity,
    read_exact_bytes_antigravity,
    write_atomic_antigravity,
    write_create_only_antigravity,
)
from .backend_gate_antigravity import (
    BackendDecisionAntigravity,
    BackendProbeAntigravity,
    decide_backend_antigravity,
    installed_backend_probe_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY
from .contracts_antigravity import CandidateScoresAntigravity, DatasetBundleAntigravity, FrozenPredictionAuthorityAntigravity
from .predecessor_antigravity import verify_exp448_predecessor_antigravity
from .progress_antigravity import ProgressWriterAntigravity, render_progress_line_antigravity, select_outer_workers_antigravity, validate_progress_antigravity
from .scoring_antigravity import aggregate_seed_scores_antigravity, score_candidate_antigravity
from .sources_antigravity import load_target_bundle_antigravity


ALLOWED_PREDICTION_KEYS = {
    "sequence", "dataset", "family", "target_length", "train_count",
    "test_count", "indices", "score_identities", "alpha",
    "attainable_p", "resolution_blocked",
}
_FORBIDDEN_KEYS_ANTIGRAVITY = {"label", "labels", "test_label", "actual_anomaly", "truth"}
_OFFICIAL_ROOT_ANTIGRAVITY = Path(FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.prediction_root)
_CONTEXT_KEYS_ANTIGRAVITY = {"schema", "reviewed_commit", "dataset_identities", "target_lengths", "backend",
    "experiment", "config_schema", "kind", "run_id", "design_sha256", "preregistration_sha256", "runbook_sha256",
    "config_sha256", "method_sha256", "source_sha256", "predecessor_sha256", "expected_snapshot_sha256",
    "backend_decision", "logical_cpus", "outer_workers", "inner_threads", "label_access_count"}
_PID_KEYS_ANTIGRAVITY = {"schema", "pid", "active", "reviewed_commit"}
_CHECKPOINT_KEYS_ANTIGRAVITY = {"schema", "reviewed_commit", "next_sequence", "terminal"}
_RECEIPT_KEYS_ANTIGRAVITY = {"schema", "sequence", "dataset_identity_sha256", "leaf_sha256", "previous_chain_sha256", "chain_sha256"}
_CHAIN_ORIGIN_ANTIGRAVITY = "0" * 64
_SEED_IDENTITY_KEYS_ANTIGRAVITY = {"kind","seed","train_dtype","train_shape","test_dtype","test_shape","expected_train_length","expected_test_length","feature_dimension","retained_shapelets","train_bytes_sha256","test_bytes_sha256","canonical_sha256"}
_AGGREGATE_IDENTITY_KEYS_ANTIGRAVITY = (_SEED_IDENTITY_KEYS_ANTIGRAVITY - {"seed"}) | {"seeds"}

def _file_identity_antigravity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class CommittedPrefixAntigravity:
    completed: int


@dataclass(frozen=True, slots=True)
class PredictionDependenciesAntigravity:
    fault_hook: Callable[[str, int], None] | None
    prefix_reader: Callable[[], CommittedPrefixAntigravity | None]
    backend_decision: BackendDecisionAntigravity
    root: Path | None = None
    bundles: tuple[DatasetBundleAntigravity, ...] | None = None
    scorer: Callable[..., CandidateScoresAntigravity] = score_candidate_antigravity
    logical_cpus: int | None = None
    completion_observer: Callable[[int, int], None] | None = None


@dataclass(frozen=True, slots=True)
class PredictionProgressAntigravity:
    completed: int
    reused: int
    recomputed: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class PredictionRunResultAntigravity:
    authority: FrozenPredictionAuthorityAntigravity
    progress: PredictionProgressAntigravity


@dataclass(frozen=True, slots=True)
class ExpectedPredecessorSnapshotAntigravity:
    dataset_identities: tuple[tuple[int, str, str], ...]
    target_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.dataset_identities) != 48 or len(self.target_lengths) != 48:
            raise ValueError("expected predecessor must contain exact ordered 48")
        if tuple(item[0] for item in self.dataset_identities) != tuple(range(48)):
            raise ValueError("expected predecessor order differs")


class PinnedPredictionTreeAntigravity:
    """One retained descriptor authority for an entire prediction-tree validation."""

    def __init__(self, root: Path):
        if not isinstance(root, Path):
            raise ValueError("prediction root must be Path")
        self.root = root
        self.root_fd = self.leaves_fd = self.receipts_fd = -1
        self._root_identity = self._leaves_identity = self._receipts_identity = None

    @staticmethod
    def _directory_identity_antigravity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    @classmethod
    def _require_directory_antigravity(cls, info: os.stat_result) -> tuple[int, ...]:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_nlink < 2:
            raise ValueError("unsafe prediction directory")
        return cls._directory_identity_antigravity(info)

    def __enter__(self):
        before = os.lstat(self.root)
        self._root_identity = self._require_directory_antigravity(before)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self.root_fd = os.open(self.root, flags)
            if self._require_directory_antigravity(os.fstat(self.root_fd)) != self._root_identity:
                raise ValueError("prediction root changed during open")
            self.leaves_fd = os.open("leaves_antigravity", flags, dir_fd=self.root_fd)
            self.receipts_fd = os.open("receipts_antigravity", flags, dir_fd=self.root_fd)
            self._leaves_identity = self._require_directory_antigravity(os.fstat(self.leaves_fd))
            self._receipts_identity = self._require_directory_antigravity(os.fstat(self.receipts_fd))
            if self._leaves_identity != self._require_directory_antigravity(os.stat("leaves_antigravity", dir_fd=self.root_fd, follow_symlinks=False)):
                raise ValueError("leaves namespace changed")
            if self._receipts_identity != self._require_directory_antigravity(os.stat("receipts_antigravity", dir_fd=self.root_fd, follow_symlinks=False)):
                raise ValueError("receipts namespace changed")
            return self
        except BaseException:
            self._close_antigravity()
            raise

    def inventory_antigravity(self, directory_fd: int) -> set[str]:
        if directory_fd not in {self.root_fd, self.leaves_fd, self.receipts_fd}:
            raise ValueError("directory descriptor not retained")
        return set(os.listdir(directory_fd))

    def read_bytes_antigravity(self, parent_fd: int, name: str) -> bytes:
        if parent_fd not in {self.root_fd, self.leaves_fd, self.receipts_fd} or not name or "/" in name:
            raise ValueError("unsafe retained-tree read")
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("authority file must be single-link regular")
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            if _file_identity_antigravity(opened) != _file_identity_antigravity(before):
                raise ValueError("authority file changed during open")
            payload = __import__("experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity", fromlist=["_read_fd_antigravity"])._read_fd_antigravity(fd)
            if len(payload) != opened.st_size or _file_identity_antigravity(os.fstat(fd)) != _file_identity_antigravity(opened):
                raise ValueError("authority file changed during read")
            reopened = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
            try:
                if _file_identity_antigravity(os.fstat(reopened)) != _file_identity_antigravity(opened):
                    raise ValueError("authority namespace changed")
            finally:
                os.close(reopened)
            return payload
        finally:
            os.close(fd)

    def read_canonical_antigravity(self, parent_fd: int, name: str) -> object:
        from .artifacts_antigravity import strict_json_bytes_antigravity
        return strict_json_bytes_antigravity(self.read_bytes_antigravity(parent_fd, name))

    def _finalize_antigravity(self) -> None:
        failures = []
        checks = ((self.root_fd, self._root_identity), (self.leaves_fd, self._leaves_identity), (self.receipts_fd, self._receipts_identity))
        for fd, expected in checks:
            try:
                if fd < 0 or self._require_directory_antigravity(os.fstat(fd)) != expected:
                    raise ValueError("retained directory changed")
            except BaseException as error:
                failures.append(error)
        try:
            if self._require_directory_antigravity(os.lstat(self.root)) != self._root_identity:
                raise ValueError("root namespace changed")
            if self._require_directory_antigravity(os.stat("leaves_antigravity", dir_fd=self.root_fd, follow_symlinks=False)) != self._leaves_identity:
                raise ValueError("leaves namespace changed at finalize")
            if self._require_directory_antigravity(os.stat("receipts_antigravity", dir_fd=self.root_fd, follow_symlinks=False)) != self._receipts_identity:
                raise ValueError("receipts namespace changed at finalize")
        except BaseException as error:
            failures.append(error)
        if failures:
            raise failures[0] if len(failures) == 1 else ExceptionGroup("pinned prediction tree changed", failures)

    def _close_antigravity(self) -> None:
        for name in ("receipts_fd", "leaves_fd", "root_fd"):
            fd = getattr(self, name)
            if fd >= 0:
                os.close(fd); setattr(self, name, -1)

    def __exit__(self, exc_type, exc, traceback):
        verification = None
        try:
            self._finalize_antigravity()
        except BaseException as error:
            verification = error
        finally:
            self._close_antigravity()
        if exc is not None and verification is not None:
            raise ExceptionGroup("validation and pinned-tree finalization failed", [exc, verification])
        if verification is not None:
            raise verification
        return False


def _exact_file_antigravity(path: Path, *, canonical: bool) -> object | bytes:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"unsafe artifact leaf: {path.name}")
    return read_canonical_file_antigravity(path) if canonical else read_exact_bytes_antigravity(path)


def _exact_directory_antigravity(path: Path) -> None:
    before = os.stat(path, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode) or before.st_nlink < 2:
        raise ValueError(f"unsafe artifact directory: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        after = os.stat(path, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (opened.st_dev, opened.st_ino, opened.st_mtime_ns, opened.st_ctime_ns):
            raise ValueError("directory identity changed during open")
        if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("directory namespace changed")
    finally:
        os.close(descriptor)


def _reject_forbidden_antigravity(value: object) -> None:
    if type(value) is dict:
        if set(value) & _FORBIDDEN_KEYS_ANTIGRAVITY:
            raise ValueError("forbidden label key")
        for child in value.values():
            _reject_forbidden_antigravity(child)
    elif type(value) is list:
        for child in value:
            _reject_forbidden_antigravity(child)


def _score_identity_antigravity(score: CandidateScoresAntigravity) -> dict[str, object]:
    payload = {
        "kind": "seed", "seed": score.seed,
        "train_dtype": score.train_oof.dtype.str, "train_shape": list(score.train_oof.shape),
        "test_dtype": score.test_scores.dtype.str, "test_shape": list(score.test_scores.shape),
        "expected_train_length": score.expected_train_length,
        "expected_test_length": score.expected_test_length,
        "feature_dimension": score.feature_dimension,
        "retained_shapelets": score.retained_shapelets,
        "train_bytes_sha256": hashlib.sha256(score.train_oof.tobytes(order="C")).hexdigest(),
        "test_bytes_sha256": hashlib.sha256(score.test_scores.tobytes(order="C")).hexdigest(),
    }
    return {**payload, "canonical_sha256": hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()}


def _aggregate_identity_antigravity(aggregate: object) -> dict[str, object]:
    payload = {
        "kind": "aggregate", "seeds": list(aggregate.seeds),
        "train_dtype": aggregate.train_oof.dtype.str, "train_shape": list(aggregate.train_oof.shape),
        "test_dtype": aggregate.test_scores.dtype.str, "test_shape": list(aggregate.test_scores.shape),
        "expected_train_length": aggregate.expected_train_length,
        "expected_test_length": aggregate.expected_test_length,
        "feature_dimension": aggregate.feature_dimension,
        "retained_shapelets": aggregate.retained_shapelets,
        "train_bytes_sha256": hashlib.sha256(aggregate.train_oof.tobytes(order="C")).hexdigest(),
        "test_bytes_sha256": hashlib.sha256(aggregate.test_scores.tobytes(order="C")).hexdigest(),
    }
    return {**payload, "canonical_sha256": hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()}


def _leaf_antigravity(sequence: int, bundle: DatasetBundleAntigravity, scorer: Callable[..., CandidateScoresAntigravity], backend: str) -> dict[str, object]:
    scores = tuple(scorer(bundle.train_normal, bundle.test_values, seed, backend) for seed in FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.seeds)
    aggregate = aggregate_seed_scores_antigravity(scores)
    if not np.isfinite(aggregate.train_oof).all() or not np.isfinite(aggregate.test_scores).all():
        raise ValueError("nonfinite scores")
    attainable = 1.0 / (len(aggregate.train_oof) + 1)
    blocked = attainable > FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.alpha
    indices = () if blocked else tuple(int(index) for index in empirical_candidates(aggregate.train_oof, aggregate.test_scores, FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.alpha).indices)
    return {
        "sequence": sequence, "dataset": bundle.dataset, "family": bundle.family,
        "target_length": bundle.target_length, "train_count": len(bundle.train_normal),
        "test_count": len(bundle.test_values), "indices": list(indices),
        "score_identities": [*(_score_identity_antigravity(item) for item in scores), _aggregate_identity_antigravity(aggregate)],
        "alpha": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.alpha,
        "attainable_p": attainable, "resolution_blocked": blocked,
    }


def _prediction_worker_antigravity(sequence: int, bundle: DatasetBundleAntigravity, scorer: Callable[..., CandidateScoresAntigravity], backend: str) -> tuple[int, int, dict[str, object]]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    return sequence, os.getpid(), _leaf_antigravity(sequence, bundle, scorer, backend)


def _validate_score_identity_antigravity(value: object, bundle: DatasetBundleAntigravity, *, seed: int | None) -> None:
    expected_keys = _AGGREGATE_IDENTITY_KEYS_ANTIGRAVITY if seed is None else _SEED_IDENTITY_KEYS_ANTIGRAVITY
    if type(value) is not dict or set(value) != expected_keys or type(value.get("canonical_sha256")) is not str:
        raise ValueError("score identity malformed")
    payload = dict(value)
    digest = payload.pop("canonical_sha256")
    if hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest() != digest:
        raise ValueError("score identity canonical SHA differs")
    expected_common = {
        "expected_train_length": len(bundle.train_normal), "expected_test_length": len(bundle.test_values),
    }
    if any(payload.get(key) != expected for key, expected in expected_common.items()):
        raise ValueError("score identity lengths differ")
    if type(payload.get("feature_dimension")) is not int or type(payload.get("retained_shapelets")) is not int or payload["feature_dimension"] != 3 * payload["retained_shapelets"]:
        raise ValueError("score identity RDST metadata differs")
    for key in ("train_dtype", "test_dtype", "train_bytes_sha256", "test_bytes_sha256"):
        if type(payload.get(key)) is not str or not payload[key]:
            raise ValueError("score identity string differs")
    for key in ("train_bytes_sha256", "test_bytes_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", payload[key]):
            raise ValueError("score SHA differs")
    if payload.get("train_shape") != [len(bundle.train_normal)] or payload.get("test_shape") != [len(bundle.test_values)]:
        raise ValueError("score identity shape differs")
    if seed is None:
        if payload.get("kind") != "aggregate" or payload.get("seeds") != list(FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.seeds) or "seed" in payload:
            raise ValueError("aggregate identity differs")
    elif payload.get("kind") != "seed" or payload.get("seed") != seed or "seeds" in payload:
        raise ValueError("seed identity differs")


def _validate_leaf_antigravity(value: object, sequence: int, bundle: DatasetBundleAntigravity) -> None:
    _reject_forbidden_antigravity(value)
    if type(value) is not dict or set(value) != ALLOWED_PREDICTION_KEYS:
        raise ValueError("leaf keys differ")
    exact = {
        "sequence": sequence, "dataset": bundle.dataset, "family": bundle.family,
        "target_length": bundle.target_length, "train_count": len(bundle.train_normal),
        "test_count": len(bundle.test_values), "alpha": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.alpha,
        "attainable_p": 1.0 / (len(bundle.train_normal) + 1),
        "resolution_blocked": 1.0 / (len(bundle.train_normal) + 1) > FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.alpha,
    }
    for key, expected in exact.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError(f"leaf {key} differs")
    indices = value["indices"]
    if type(indices) is not list or any(type(index) is not int for index in indices) or indices != sorted(set(indices)) or any(index < 0 or index >= bundle.target_length for index in indices):
        raise ValueError("candidate indices invalid")
    if value["resolution_blocked"] and indices:
        raise ValueError("resolution-blocked leaf has candidates")
    identities = value["score_identities"]
    if type(identities) is not list or len(identities) != 3:
        raise ValueError("score identity count differs")
    _validate_score_identity_antigravity(identities[0], bundle, seed=FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.seeds[0])
    _validate_score_identity_antigravity(identities[1], bundle, seed=FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.seeds[1])
    _validate_score_identity_antigravity(identities[2], bundle, seed=None)


def _snapshot_sha_antigravity(bundles: tuple[DatasetBundleAntigravity, ...]) -> str:
    payload = {"dataset_identities":[list(bundle.dataset_identity) for bundle in bundles], "target_lengths":[bundle.target_length for bundle in bundles]}
    return hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()


def _context_antigravity(reviewed_commit: str, bundles: tuple[DatasetBundleAntigravity, ...], decision: BackendDecisionAntigravity, logical_cpus: int) -> dict[str, object]:
    backend = decision.selected_name
    snapshot_sha = _snapshot_sha_antigravity(bundles)
    return {
        "schema": "exp451.prediction.context.v1", "reviewed_commit": reviewed_commit,
        "dataset_identities": [list(bundle.dataset_identity) for bundle in bundles],
        "target_lengths": [bundle.target_length for bundle in bundles], "backend": backend,
        "experiment": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.experiment, "config_schema": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.schema,
        "kind": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.kind, "run_id": FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.run_id,
        "design_sha256": "63b8880afbfe09601c241e6c80ab7d2883bc554318f9b8489352021a1dbd7fc4",
        "preregistration_sha256": "2860a8f2d6a317b99bdcbaba152068b631671b6c447819421ff5de2d452e797a",
        "runbook_sha256": "1778f118b687c078f737775a1ab2ac1bdf8b2522359309b374c69709435a9853",
        "config_sha256": "4a2b7502bcffcd056014bddc7bf3bba5ad0bbe6dea7ab8838fe08fdf867a1c80",
        "method_sha256": "4966b42ae33492170c3442f2772ae6ff0ba9e5a568e50928248e8f685a986da5",
        "source_sha256": "e43bca8fbf99be08551ee0c53785f5b492e70d9089ed1fee16377d3e73a6ad4d",
        "predecessor_sha256": "9bffb6e31e2fe84026c02f79619247d2c1732cef6f81efc27f8797b842a873c1",
        "expected_snapshot_sha256": snapshot_sha, "backend_decision": _backend_payload_antigravity(decision),
        "logical_cpus": logical_cpus, "outer_workers": select_outer_workers_antigravity(logical_cpus, backend),
        "inner_threads": 1, "label_access_count": 0,
    }


def _pid_antigravity(reviewed_commit: str, active: bool) -> dict[str, object]:
    return {"schema": "exp451.prediction.pid.v1", "pid": os.getpid(), "active": active, "reviewed_commit": reviewed_commit}


def _checkpoint_antigravity(reviewed_commit: str, next_sequence: int, terminal: bool) -> dict[str, object]:
    return {"schema": "exp451.prediction.checkpoint.v1", "reviewed_commit": reviewed_commit, "next_sequence": next_sequence, "terminal": terminal}


def _progress_payload_antigravity(bundle: DatasetBundleAntigravity, *, completed: int, reused: int, stage: str, backend: str, started: float, latest_error: str | None) -> dict[str, object]:
    elapsed = float(max(0.0, time.monotonic() - started))
    rate = float(completed / elapsed) if elapsed > 0.0 else 0.0
    eta = 0.0 if stage == "complete" else (float((48 - completed) / rate) if rate > 0.0 else 0.0)
    terminal = stage in {"complete", "failed", "setup_failed"}
    return {
        "pid": os.getpid(), "dataset": None if terminal else bundle.dataset, "family": None if terminal else bundle.family,
        "train_rows": 0 if terminal else len(bundle.train_normal), "test_rows": 0 if terminal else len(bundle.test_values),
        "series_length": 0 if terminal else bundle.train_normal.shape[1], "seed": None,
        "fold": None, "stage": stage, "backend": backend, "completed": completed,
        "total": 48, "success": completed, "error": 0 if latest_error is None else 1,
        "reused": reused, "recomputed": completed - reused, "datasets_per_second": rate, "elapsed_seconds": elapsed,
        "eta_seconds": eta, "checkpoint": "checkpoint_antigravity.json", "latest_error": latest_error,
    }


def _root_names_antigravity(root: Path) -> set[str]:
    _exact_directory_antigravity(root)
    return set(os.listdir(root))


def _read_prefix_antigravity(root: Path, bundles: tuple[DatasetBundleAntigravity, ...], reviewed_commit: str, decision: BackendDecisionAntigravity, logical_cpus: int) -> list[dict[str, object]]:
    with PinnedPredictionTreeAntigravity(root) as tree:
        return _read_prefix_from_tree_antigravity(tree, bundles, reviewed_commit, decision, logical_cpus)


def _read_prefix_from_tree_antigravity(tree: PinnedPredictionTreeAntigravity, bundles: tuple[DatasetBundleAntigravity, ...], reviewed_commit: str, decision: BackendDecisionAntigravity, logical_cpus: int) -> list[dict[str, object]]:
    names = tree.inventory_antigravity(tree.root_fd)
    complete = "manifest_antigravity.json" in names
    failed = "failure_antigravity.json" in names or "setup_failure_antigravity.json" in names
    if complete and failed:
        raise ValueError("multiple terminal artifacts")
    allowed = {"context_antigravity.json", "pid_antigravity.json", "checkpoint_antigravity.json", "progress_antigravity.json", "progress_antigravity.txt", "leaves_antigravity", "receipts_antigravity"}
    if complete: allowed.add("manifest_antigravity.json")
    if "failure_antigravity.json" in names: allowed.add("failure_antigravity.json")
    if "setup_failure_antigravity.json" in names: allowed.add("setup_failure_antigravity.json")
    if names != allowed:
        raise ValueError("prediction root inventory differs")
    context = tree.read_canonical_antigravity(tree.root_fd, "context_antigravity.json")
    if type(context) is not dict or set(context) != _CONTEXT_KEYS_ANTIGRAVITY or context != _context_antigravity(reviewed_commit, bundles, decision, logical_cpus):
        raise ValueError("prediction context differs")
    pid = tree.read_canonical_antigravity(tree.root_fd, "pid_antigravity.json")
    checkpoint = tree.read_canonical_antigravity(tree.root_fd, "checkpoint_antigravity.json")
    progress = validate_progress_antigravity(tree.read_canonical_antigravity(tree.root_fd, "progress_antigravity.json"))
    text = tree.read_bytes_antigravity(tree.root_fd, "progress_antigravity.txt")
    if not text.endswith(render_progress_line_antigravity(progress)) or progress["pid"] != pid["pid"]:
        raise ValueError("progress text or PID differs")
    if type(pid) is not dict or set(pid) != _PID_KEYS_ANTIGRAVITY or pid["reviewed_commit"] != reviewed_commit or type(pid["active"]) is not bool or type(pid["pid"]) is not int:
        raise ValueError("PID evidence differs")
    if type(checkpoint) is not dict or set(checkpoint) != _CHECKPOINT_KEYS_ANTIGRAVITY or checkpoint["reviewed_commit"] != reviewed_commit or type(checkpoint["next_sequence"]) is not int or type(checkpoint["terminal"]) is not bool:
        raise ValueError("checkpoint differs")
    leaf_names = sorted(tree.inventory_antigravity(tree.leaves_fd))
    receipt_names = sorted(tree.inventory_antigravity(tree.receipts_fd))
    if leaf_names != [f"{sequence:04d}_antigravity.json" for sequence in range(len(leaf_names))] or len(leaf_names) > 48:
        raise ValueError("leaf prefix is not contiguous")
    if receipt_names != leaf_names:
        raise ValueError("receipt prefix differs from leaf prefix")
    leaves: list[dict[str, object]] = []
    previous_chain = _CHAIN_ORIGIN_ANTIGRAVITY
    for sequence, name in enumerate(leaf_names):
        leaf_bytes = tree.read_bytes_antigravity(tree.leaves_fd, name)
        from .artifacts_antigravity import strict_json_bytes_antigravity
        value = strict_json_bytes_antigravity(leaf_bytes)
        _validate_leaf_antigravity(value, sequence, bundles[sequence])
        receipt = tree.read_canonical_antigravity(tree.receipts_fd, name)
        previous_chain = _validate_receipt_antigravity(receipt, sequence, bundles[sequence], leaf_bytes, previous_chain)
        leaves.append(value)
    if checkpoint["next_sequence"] != len(leaves):
        # Disk leaves are authoritative; only nonterminal stale mutable state is reconciled by caller.
        if checkpoint["terminal"]:
            raise ValueError("terminal checkpoint differs from durable prefix")
    if complete:
        tree.read_canonical_antigravity(tree.root_fd, "manifest_antigravity.json")
    if failed:
        failure_name = "failure_antigravity.json" if "failure_antigravity.json" in names else "setup_failure_antigravity.json"
        failure = tree.read_canonical_antigravity(tree.root_fd, failure_name)
        if type(failure) is not dict or set(failure) != {"schema", "active", "error", "next_sequence"} or failure["schema"] != "exp451.prediction.failure.v1" or failure["active"] is not False or type(failure["error"]) is not str or type(failure["next_sequence"]) is not int:
            raise ValueError("failure terminal evidence differs")
    return leaves


def _identity_digest_antigravity(bundle: DatasetBundleAntigravity) -> str:
    payload = {"sequence": bundle.dataset_order, "family": bundle.family, "dataset": bundle.dataset, "target_length": bundle.target_length, "train_count": len(bundle.train_normal), "test_count": len(bundle.test_values)}
    return hashlib.sha256(canonical_json_bytes_antigravity(payload)).hexdigest()


def _receipt_antigravity(sequence: int, bundle: DatasetBundleAntigravity, leaf_bytes: bytes, previous_chain: str) -> dict[str, object]:
    core = {"schema":"exp451.prediction.receipt.v1","sequence":sequence,"dataset_identity_sha256":_identity_digest_antigravity(bundle),"leaf_sha256":hashlib.sha256(leaf_bytes).hexdigest(),"previous_chain_sha256":previous_chain}
    return {**core, "chain_sha256": hashlib.sha256(canonical_json_bytes_antigravity(core)).hexdigest()}


def _validate_receipt_antigravity(receipt: object, sequence: int, bundle: DatasetBundleAntigravity, leaf_bytes: bytes, previous_chain: str) -> str:
    if type(receipt) is not dict or set(receipt) != _RECEIPT_KEYS_ANTIGRAVITY:
        raise ValueError("receipt keys differ")
    expected = _receipt_antigravity(sequence, bundle, leaf_bytes, previous_chain)
    for key, value in expected.items():
        if type(receipt[key]) is not type(value) or receipt[key] != value:
            raise ValueError(f"receipt {key} differs")
    return receipt["chain_sha256"]


def _authority_from_leaves_antigravity(reviewed_commit: str, bundles: tuple[DatasetBundleAntigravity, ...], leaves: list[dict[str, object]]) -> FrozenPredictionAuthorityAntigravity:
    return FrozenPredictionAuthorityAntigravity(
        reviewed_commit, tuple(bundle.dataset_identity for bundle in bundles),
        tuple(bundle.target_length for bundle in bundles),
        tuple(tuple(value["indices"]) for value in leaves),
    )


def _probe_payload_antigravity(probe: BackendProbeAntigravity) -> dict[str, object]:
    return {
        "name": probe.name, "available": probe.available, "finite": probe.finite,
        "candidate_indices_match": probe.candidate_indices_match,
        "maximum_normalized_error": probe.maximum_normalized_error,
        "inclusive_cpu_seconds": probe.inclusive_cpu_seconds,
        "inclusive_accelerator_seconds": probe.inclusive_accelerator_seconds,
        "speedup": probe.speedup, "versions": [list(item) for item in probe.versions],
        "device": probe.device, "reason": probe.reason,
        "implementation_qualified": probe.implementation_qualified,
    }


def _backend_payload_antigravity(decision: BackendDecisionAntigravity) -> dict[str, object]:
    return {"selected_name": decision.selected_name, "fallback_reason": decision.fallback_reason, "probes": [_probe_payload_antigravity(probe) for probe in decision.probes]}


def _write_mutable_state_antigravity(root: Path, reviewed_commit: str, completed: int, terminal: bool, active: bool, writer: ProgressWriterAntigravity, bundle: DatasetBundleAntigravity, reused: int, stage: str, backend: str, started: float, latest_error: str | None) -> None:
    write_atomic_antigravity(root / "checkpoint_antigravity.json", canonical_json_bytes_antigravity(_checkpoint_antigravity(reviewed_commit, completed, terminal)), fault_hook=writer.fault_hook, sequence=max(0, completed - 1))
    writer.write_antigravity(_progress_payload_antigravity(bundle, completed=completed, reused=reused, stage=stage, backend=backend, started=started, latest_error=latest_error), max(0, completed - 1))
    write_atomic_antigravity(root / "pid_antigravity.json", canonical_json_bytes_antigravity(_pid_antigravity(reviewed_commit, active)), fault_hook=writer.fault_hook, sequence=max(0, completed - 1))


def _terminal_failure_antigravity(root: Path, reviewed_commit: str, leaves: list[dict[str, object]], error: BaseException, writer: ProgressWriterAntigravity, bundle: DatasetBundleAntigravity, reused: int, backend: str, started: float) -> None:
    message = f"{type(error).__name__}: {error}"
    secondary: BaseException | None = None
    try:
        _write_mutable_state_antigravity(root, reviewed_commit, len(leaves), True, False, writer, bundle, reused, "failed", backend, started, message)
        write_create_only_antigravity(root / "failure_antigravity.json", canonical_json_bytes_antigravity({"schema":"exp451.prediction.failure.v1","active":False,"error":message,"next_sequence":len(leaves)}))
    except BaseException as cleanup:
        secondary = cleanup
    if secondary is not None:
        raise ExceptionGroup("prediction and terminal-evidence failures", [error, secondary]) from error


def _setup_failure_antigravity(root: Path, reviewed_commit: str, bundles: tuple[DatasetBundleAntigravity, ...], decision: BackendDecisionAntigravity, logical_cpus: int, started: float, error: BaseException) -> None:
    backend = decision.selected_name
    message = f"{type(error).__name__}: {error}"
    try:
        (root / "leaves_antigravity").mkdir(mode=0o700, exist_ok=True)
        (root / "receipts_antigravity").mkdir(mode=0o700, exist_ok=True)
        if not (root / "context_antigravity.json").exists():
            write_create_only_antigravity(root / "context_antigravity.json", canonical_json_bytes_antigravity(_context_antigravity(reviewed_commit, bundles, decision, logical_cpus)))
        if not (root / "progress_antigravity.txt").exists():
            write_create_only_antigravity(root / "progress_antigravity.txt", b"")
        writer = ProgressWriterAntigravity(root)
        _write_mutable_state_antigravity(root, reviewed_commit, 0, True, False, writer, bundles[0], 0, "setup_failed", backend, started, message)
        write_create_only_antigravity(root / "setup_failure_antigravity.json", canonical_json_bytes_antigravity({"schema":"exp451.prediction.setup_failure.v1","active":False,"error":message,"next_sequence":0}))
    except BaseException as cleanup:
        raise ExceptionGroup("setup and setup-closeout failures", [error, cleanup]) from error


def _run_prediction_with_dependencies_antigravity(reviewed_commit: str, dependencies: PredictionDependenciesAntigravity) -> PredictionRunResultAntigravity:
    if type(reviewed_commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", reviewed_commit):
        raise ValueError("reviewed commit invalid")
    root = dependencies.root or _OFFICIAL_ROOT_ANTIGRAVITY
    bundles = dependencies.bundles
    if bundles is None:
        predecessor = verify_exp448_predecessor_antigravity()
        bundles = tuple(load_target_bundle_antigravity(name) for name in predecessor.target_dataset_names)
    if type(bundles) is not tuple or len(bundles) != 48 or tuple(bundle.dataset_order for bundle in bundles) != tuple(range(48)):
        raise ValueError("exact ordered 48 bundles required")
    backend = dependencies.backend_decision.selected_name
    logical_cpus = dependencies.logical_cpus or os.cpu_count() or 1
    outer_workers = select_outer_workers_antigravity(logical_cpus, backend)
    started = time.monotonic()
    writer = ProgressWriterAntigravity(root, dependencies.fault_hook)
    leaves: list[dict[str, object]] = []
    reused = 0
    try:
        if dependencies.fault_hook:
            dependencies.fault_hook("setup", 0)
        if root.exists():
            leaves = _read_prefix_antigravity(root, bundles, reviewed_commit, dependencies.backend_decision, logical_cpus)
            names = _root_names_antigravity(root)
            if "manifest_antigravity.json" in names or "failure_antigravity.json" in names:
                raise ValueError("prediction root is terminal")
            reused = len(leaves)
            external_prefix = dependencies.prefix_reader()
            if external_prefix is not None and (type(external_prefix) is not CommittedPrefixAntigravity or external_prefix.completed != reused):
                raise ValueError("prefix reader differs from descriptor prefix")
            _write_mutable_state_antigravity(root, reviewed_commit, reused, False, True, writer, bundles[min(reused, 47)], reused, "active", backend, started, None)
        else:
            root.mkdir(mode=0o700)
            try:
                if dependencies.fault_hook: dependencies.fault_hook("setup_root", 0)
                if dependencies.fault_hook: dependencies.fault_hook("setup_leaves", 0)
                (root / "leaves_antigravity").mkdir(mode=0o700)
                if dependencies.fault_hook: dependencies.fault_hook("setup_receipts", 0)
                (root / "receipts_antigravity").mkdir(mode=0o700)
                if dependencies.fault_hook: dependencies.fault_hook("setup_context", 0)
                write_create_only_antigravity(root / "context_antigravity.json", canonical_json_bytes_antigravity(_context_antigravity(reviewed_commit, bundles, dependencies.backend_decision, logical_cpus)))
                if dependencies.fault_hook: dependencies.fault_hook("setup_checkpoint", 0)
                write_atomic_antigravity(root / "checkpoint_antigravity.json", canonical_json_bytes_antigravity(_checkpoint_antigravity(reviewed_commit, 0, False)))
                if dependencies.fault_hook: dependencies.fault_hook("setup_progress_json", 0)
                write_atomic_antigravity(root / "progress_antigravity.json", canonical_json_bytes_antigravity(_progress_payload_antigravity(bundles[0], completed=0, reused=0, stage="starting", backend=backend, started=started, latest_error=None)))
                if dependencies.fault_hook: dependencies.fault_hook("setup_progress_text", 0)
                write_create_only_antigravity(root / "progress_antigravity.txt", b"")
                if dependencies.fault_hook: dependencies.fault_hook("setup_pid", 0)
                write_atomic_antigravity(root / "pid_antigravity.json", canonical_json_bytes_antigravity(_pid_antigravity(reviewed_commit, True)))
            except BaseException as setup_error:
                _setup_failure_antigravity(root, reviewed_commit, bundles, dependencies.backend_decision, logical_cpus, started, setup_error)
                raise

        pending: dict[int, dict[str, object]] = {}
        next_commit = reused
        previous_chain = _CHAIN_ORIGIN_ANTIGRAVITY
        if reused:
            final_receipt = read_canonical_file_antigravity(root / "receipts_antigravity" / f"{reused - 1:04d}_antigravity.json")
            previous_chain = final_receipt["chain_sha256"]

        def receive(result: tuple[int, int, dict[str, object]]) -> None:
            nonlocal next_commit, previous_chain
            sequence, worker_pid, leaf = result
            if sequence in pending or sequence < next_commit or sequence >= 48:
                raise ValueError("duplicate or mismatched worker sequence")
            _validate_leaf_antigravity(leaf, sequence, bundles[sequence])
            pending[sequence] = leaf
            if dependencies.completion_observer is not None:
                dependencies.completion_observer(sequence, worker_pid)
            while next_commit in pending:
                committed = pending.pop(next_commit)
                leaf_bytes = canonical_json_bytes_antigravity(committed)
                write_create_only_antigravity(root / "leaves_antigravity" / f"{next_commit:04d}_antigravity.json", leaf_bytes, fault_hook=dependencies.fault_hook, sequence=next_commit)
                receipt = _receipt_antigravity(next_commit, bundles[next_commit], leaf_bytes, previous_chain)
                write_create_only_antigravity(root / "receipts_antigravity" / f"{next_commit:04d}_antigravity.json", canonical_json_bytes_antigravity(receipt), fault_hook=dependencies.fault_hook, sequence=next_commit)
                previous_chain = receipt["chain_sha256"]
                leaves.append(committed)
                if dependencies.fault_hook:
                    dependencies.fault_hook("after_leaf", next_commit)
                next_commit += 1
                _write_mutable_state_antigravity(root, reviewed_commit, next_commit, False, True, writer, bundles[next_commit - 1], reused, "active", backend, started, None)

        if backend == "cpu":
            executor = ProcessPoolExecutor(max_workers=outer_workers)
            futures = []
            try:
                for sequence in range(reused, 48):
                    if dependencies.fault_hook:
                        dependencies.fault_hook("worker", sequence)
                    futures.append(executor.submit(_prediction_worker_antigravity, sequence, bundles[sequence], dependencies.scorer, backend))
                for future in as_completed(futures):
                    receive(future.result())
            finally:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
        else:
            for sequence in range(reused, 48):
                if dependencies.fault_hook:
                    dependencies.fault_hook("worker", sequence)
                receive(_prediction_worker_antigravity(sequence, bundles[sequence], dependencies.scorer, backend))

        if next_commit != 48 or pending:
            raise ValueError("worker coverage differs")
        authority = _authority_from_leaves_antigravity(reviewed_commit, bundles, leaves)
        leaf_shas = [hashlib.sha256(canonical_json_bytes_antigravity(value)).hexdigest() for value in leaves]
        context_sha = hashlib.sha256(canonical_json_bytes_antigravity(_context_antigravity(reviewed_commit, bundles, dependencies.backend_decision, logical_cpus))).hexdigest()
        receipt_shas = []
        manifest_chain = _CHAIN_ORIGIN_ANTIGRAVITY
        for sequence, leaf in enumerate(leaves):
            receipt = _receipt_antigravity(sequence, bundles[sequence], canonical_json_bytes_antigravity(leaf), manifest_chain)
            receipt_shas.append(hashlib.sha256(canonical_json_bytes_antigravity(receipt)).hexdigest())
            manifest_chain = receipt["chain_sha256"]
        if manifest_chain != previous_chain:
            raise ValueError("receipt chain construction differs")
        manifest = {
            "schema": "exp451.prediction.manifest.v1", "authority": authority.to_payload_antigravity(),
            "leaf_sha256": leaf_shas, "logical_cpus": logical_cpus, "outer_workers": outer_workers,
            "inner_threads": 1, "backend_decision": _backend_payload_antigravity(dependencies.backend_decision),
            "context_sha256": context_sha, "receipt_sha256": receipt_shas, "receipt_chain_head_sha256": previous_chain,
            "authority_sha256": authority.canonical_sha256_antigravity,
        }
        # Complete mutable lifecycle core first; manifest is the publication point.
        _write_mutable_state_antigravity(root, reviewed_commit, 48, True, False, writer, bundles[-1], reused, "complete", backend, started, None)
        manifest_bytes = canonical_json_bytes_antigravity(manifest)
        expected = ExpectedPredecessorSnapshotAntigravity(
            tuple(bundle.dataset_identity for bundle in bundles),
            tuple(bundle.target_length for bundle in bundles),
        )
        _validate_prospective_prediction_root_antigravity(root, manifest_bytes, authority, expected)
        if dependencies.fault_hook:
            dependencies.fault_hook("publish_manifest", 48)
        write_create_only_antigravity(root / "manifest_antigravity.json", manifest_bytes)
        # Publication is last: no fallible I/O, logging, or validation follows.
        return PredictionRunResultAntigravity(authority, PredictionProgressAntigravity(48, reused, 48 - reused, True))
    except KeyboardInterrupt as interruption:
        secondary: BaseException | None = None
        try:
            if root.exists() and (root / "context_antigravity.json").exists():
                _write_mutable_state_antigravity(root, reviewed_commit, len(leaves), False, False, writer, bundles[min(len(leaves), 47)], reused, "interrupted", backend, started, f"KeyboardInterrupt: {interruption}")
        except BaseException as cleanup:
            secondary = cleanup
        if secondary is not None:
            interruption.add_note(f"interruption_cleanup_error={type(secondary).__name__}: {secondary}")
        raise
    except BaseException as error:
        if root.exists() and (root / "context_antigravity.json").exists() and not (root / "failure_antigravity.json").exists() and not (root / "setup_failure_antigravity.json").exists() and not (root / "manifest_antigravity.json").exists():
            _terminal_failure_antigravity(root, reviewed_commit, leaves, error, writer, bundles[min(len(leaves), 47)], reused, backend, started)
        raise


def _validate_complete_prediction_antigravity(tree: PinnedPredictionTreeAntigravity, authority: FrozenPredictionAuthorityAntigravity, expected_snapshot: ExpectedPredecessorSnapshotAntigravity, manifest: object, *, prospective: bool) -> FrozenPredictionAuthorityAntigravity:
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("authority type differs")
    identities = expected_snapshot.dataset_identities
    if authority.dataset_identities != identities or authority.target_lengths != expected_snapshot.target_lengths:
        raise ValueError("authority differs from expected predecessor")
    bundles = tuple(DatasetBundleAntigravity(order, family, dataset, np.ones((1, 1)), np.ones((expected_snapshot.target_lengths[order], 1)), expected_snapshot.target_lengths[order]) for order, family, dataset in identities)
    # Validation uses manifest-bound lengths/identities; TRAIN counts are recovered from leaves.
    context = tree.read_canonical_antigravity(tree.root_fd, "context_antigravity.json")
    backend = context.get("backend") if type(context) is dict else None
    expected_snapshot_sha = hashlib.sha256(canonical_json_bytes_antigravity({"dataset_identities":[list(item) for item in identities], "target_lengths":list(expected_snapshot.target_lengths)})).hexdigest()
    if type(context) is not dict or set(context) != _CONTEXT_KEYS_ANTIGRAVITY or backend not in {"cpu", "mlx", "mps"} or context["reviewed_commit"] != authority.reviewed_commit or context["expected_snapshot_sha256"] != expected_snapshot_sha or context["label_access_count"] != 0:
        raise ValueError("context backend differs")
    names = tree.inventory_antigravity(tree.root_fd)
    expected = {"context_antigravity.json","pid_antigravity.json","checkpoint_antigravity.json","progress_antigravity.json","progress_antigravity.txt","leaves_antigravity","receipts_antigravity"}
    if not prospective: expected.add("manifest_antigravity.json")
    if names != expected:
        raise ValueError("complete root inventory differs")
    if type(manifest) is not dict or set(manifest) != {"schema","authority","leaf_sha256","receipt_sha256","receipt_chain_head_sha256","context_sha256","authority_sha256","logical_cpus","outer_workers","inner_threads","backend_decision"} or manifest["schema"] != "exp451.prediction.manifest.v1":
        raise ValueError("manifest invalid")
    restored = FrozenPredictionAuthorityAntigravity.from_payload_antigravity(manifest["authority"])
    if restored != authority or manifest["inner_threads"] != 1 or type(manifest["logical_cpus"]) is not int or type(manifest["outer_workers"]) is not int:
        raise ValueError("manifest authority or scheduler differs")
    if manifest["logical_cpus"] != context["logical_cpus"] or manifest["outer_workers"] != context["outer_workers"] or manifest["backend_decision"] != context["backend_decision"] or manifest["outer_workers"] != select_outer_workers_antigravity(manifest["logical_cpus"], backend):
        raise ValueError("manifest context evidence differs")
    if manifest["context_sha256"] != hashlib.sha256(canonical_json_bytes_antigravity(context)).hexdigest() or manifest["authority_sha256"] != authority.canonical_sha256_antigravity:
        raise ValueError("manifest authority SHA differs")
    # Reconstruct exact bundle row counts from leaf metadata only after descriptor authentication.
    names = sorted(tree.inventory_antigravity(tree.leaves_fd))
    if names != [f"{sequence:04d}_antigravity.json" for sequence in range(48)] or sorted(tree.inventory_antigravity(tree.receipts_fd)) != names:
        raise ValueError("complete leaf inventory differs")
    leaves = []
    receipt_shas = []
    previous_chain = _CHAIN_ORIGIN_ANTIGRAVITY
    for sequence, name in enumerate(names):
        leaf_bytes = tree.read_bytes_antigravity(tree.leaves_fd, name)
        from .artifacts_antigravity import strict_json_bytes_antigravity
        value = strict_json_bytes_antigravity(leaf_bytes)
        train_count = value.get("train_count") if type(value) is dict else None
        if type(train_count) is not int or train_count < 1:
            raise ValueError("leaf train count differs")
        bundle = DatasetBundleAntigravity(sequence, identities[sequence][1], identities[sequence][2], np.ones((train_count, 1)), np.ones((expected_snapshot.target_lengths[sequence], 1)), expected_snapshot.target_lengths[sequence])
        _validate_leaf_antigravity(value, sequence, bundle)
        receipt_bytes = tree.read_bytes_antigravity(tree.receipts_fd, name)
        receipt = strict_json_bytes_antigravity(receipt_bytes)
        receipt_shas.append(hashlib.sha256(receipt_bytes).hexdigest())
        previous_chain = _validate_receipt_antigravity(receipt, sequence, bundle, leaf_bytes, previous_chain)
        leaves.append(value)
    if manifest["leaf_sha256"] != [hashlib.sha256(canonical_json_bytes_antigravity(value)).hexdigest() for value in leaves]:
        raise ValueError("ordered leaf SHA sequence differs")
    if manifest["receipt_sha256"] != receipt_shas or manifest["receipt_chain_head_sha256"] != previous_chain:
        raise ValueError("ordered receipt authority differs")
    checkpoint = tree.read_canonical_antigravity(tree.root_fd, "checkpoint_antigravity.json")
    pid = tree.read_canonical_antigravity(tree.root_fd, "pid_antigravity.json")
    progress = validate_progress_antigravity(tree.read_canonical_antigravity(tree.root_fd, "progress_antigravity.json"))
    progress_text = tree.read_bytes_antigravity(tree.root_fd, "progress_antigravity.txt")
    if checkpoint != _checkpoint_antigravity(authority.reviewed_commit, 48, True) or type(pid) is not dict or set(pid) != _PID_KEYS_ANTIGRAVITY or pid["active"] is not False or pid["reviewed_commit"] != authority.reviewed_commit:
        raise ValueError("terminal lifecycle evidence differs")
    if progress["stage"] != "complete" or progress["completed"] != 48 or progress["reused"] + progress["recomputed"] != 48 or progress["pid"] != pid["pid"] or not progress_text.endswith(render_progress_line_antigravity(progress)):
        raise ValueError("terminal progress differs")
    if previous_chain != tree.read_canonical_antigravity(tree.receipts_fd, names[-1])["chain_sha256"]:
        raise ValueError("receipt chain head differs")
    return authority


def _validate_prospective_prediction_root_antigravity(root: Path, manifest_bytes: bytes, authority: FrozenPredictionAuthorityAntigravity, expected_snapshot: ExpectedPredecessorSnapshotAntigravity) -> FrozenPredictionAuthorityAntigravity:
    manifest = __import__("experiments.exp451_rdst_local_gap_qualification_antigravity.artifacts_antigravity", fromlist=["strict_json_bytes_antigravity"]).strict_json_bytes_antigravity(manifest_bytes)
    with PinnedPredictionTreeAntigravity(root) as tree:
        return _validate_complete_prediction_antigravity(tree, authority, expected_snapshot, manifest, prospective=True)


def _validate_frozen_prediction_root_antigravity(root: Path, authority: FrozenPredictionAuthorityAntigravity, expected_snapshot: ExpectedPredecessorSnapshotAntigravity) -> FrozenPredictionAuthorityAntigravity:
    with PinnedPredictionTreeAntigravity(root) as tree:
        manifest = tree.read_canonical_antigravity(tree.root_fd, "manifest_antigravity.json")
        return _validate_complete_prediction_antigravity(tree, authority, expected_snapshot, manifest, prospective=False)


def validate_frozen_prediction_authority_antigravity(authority: FrozenPredictionAuthorityAntigravity) -> FrozenPredictionAuthorityAntigravity:
    predecessor = verify_exp448_predecessor_antigravity()
    identities = tuple((row.dataset_order, row.family, row.dataset) for row in predecessor.target_baseline_rows)
    expected = ExpectedPredecessorSnapshotAntigravity(identities, tuple(predecessor.target_test_lengths))
    return _validate_frozen_prediction_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY, authority, expected)


def run_prediction_antigravity(reviewed_commit: str) -> FrozenPredictionAuthorityAntigravity:
    decision = decide_backend_antigravity(installed_backend_probe_antigravity("mlx"))
    dependencies = PredictionDependenciesAntigravity(None, lambda: None, decision)
    return _run_prediction_with_dependencies_antigravity(reviewed_commit, dependencies).authority


__all__ = [
    "CommittedPrefixAntigravity", "ExpectedPredecessorSnapshotAntigravity", "PredictionDependenciesAntigravity",
    "PredictionProgressAntigravity", "PredictionRunResultAntigravity",
    "PinnedPredictionTreeAntigravity", "run_prediction_antigravity", "validate_frozen_prediction_authority_antigravity",
]
