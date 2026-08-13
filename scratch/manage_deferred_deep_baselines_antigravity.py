#!/usr/bin/env python3
"""Launch, inspect, and verify deferred one-method deep public baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paper_public_baselines.artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    validate_score_payload,
    write_create_only_bytes,
)
from experiments.paper_public_baselines.config import (
    ALPHAS,
    EXPECTED_DATASETS,
    EXPECTED_DB_SHA256,
    SEED,
)
from experiments.paper_public_baselines.runner import verify_score_root


ALLOWED_METHODS = ("deep_svdd_1d", "drocc_1d")
OFFICIAL_EPOCHS = 50
OFFICIAL_WORKERS = 1
HEARTBEAT_INTERVAL_SECONDS = 15.0
HALF_PAIR_GRACE_SECONDS = 0.05
AUTHORITY_SCHEMA = "paper-public-deep-launch-authority-antigravity.v1"
STATE_SCHEMA = "paper-public-deep-launch-state-antigravity.v1"
PUBLIC_BASELINE_SCRIPT = REPOSITORY_ROOT / "scratch/run_paper_public_baselines.py"
NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
SCORING_CLOSURE_PATHS = (
    "scratch/run_paper_public_baselines.py",
    "experiments/paper_public_baselines/acceleration.py",
    "experiments/paper_public_baselines/artifacts.py",
    "experiments/paper_public_baselines/calibration.py",
    "experiments/paper_public_baselines/cli.py",
    "experiments/paper_public_baselines/config.py",
    "experiments/paper_public_baselines/contracts.py",
    "experiments/paper_public_baselines/data.py",
    "experiments/paper_public_baselines/methods/__init__.py",
    "experiments/paper_public_baselines/methods/classical.py",
    "experiments/paper_public_baselines/methods/deep_common.py",
    "experiments/paper_public_baselines/methods/deep_svdd.py",
    "experiments/paper_public_baselines/methods/drocc.py",
    "experiments/paper_public_baselines/preprocessing.py",
    "experiments/paper_public_baselines/runner.py",
)
RUN_CONFIG_SOURCE_PATHS = (
    "acceleration.py",
    "artifacts.py",
    "calibration.py",
    "config.py",
    "contracts.py",
    "data.py",
    "methods/__init__.py",
    "methods/classical.py",
    "methods/deep_common.py",
    "methods/deep_svdd.py",
    "methods/drocc.py",
    "preprocessing.py",
    "runner.py",
)
AUTHORITY_FIELDS = {
    "schema",
    "method",
    "database_path",
    "run_root",
    "repository_root",
    "launcher_path",
    "public_baseline_script_path",
    "database_sha256",
    "expected_dataset_count",
    "expected_pair_count",
    "epochs",
    "seed",
    "workers",
    "alphas",
    "executable",
    "python_version",
    "git_commit",
    "git_clean",
    "wrapper_sha256",
    "scoring_closure_sha256",
    "logical_cpu_count",
    "logical_cpu_minus_one_worker_count",
    "actual_worker_count",
    "cpu_worker_policy_assessment",
    "backend",
}
BACKEND_FIELDS = {
    "selected_backend",
    "pytorch_version",
    "mlx_version",
    "mlx_metal_version",
    "mps_built",
    "mps_available",
    "tensor_eligible",
    "exact_accelerated_path_available",
    "transfer_compile_overhead_benchmarked",
    "cpu_fallback",
    "fallback_reason",
    "canonical_gate",
}
CPU_WORKER_POLICY_ASSESSMENT = (
    "logical-CPU-minus-one is inapplicable to the frozen sequential deep "
    "stage; official actual worker count remains one"
)
ACCELERATION_FALLBACK_REASON = (
    "No exact MLX/MPS path preserves the frozen PyTorch initialization, "
    "optimizer, autograd, score, and routing."
)


@dataclass(frozen=True, slots=True)
class OpsPaths:
    root: Path
    authority: Path
    authority_sha256: Path
    pid: Path
    state: Path
    progress_log: Path
    worker_lock: Path


@dataclass(frozen=True, slots=True)
class PairProgress:
    completed: int
    abstained: int
    datasets: tuple[str, ...]
    run_config_sha256: str | None


class PairVerificationError(ValueError):
    failure_kind = "shard_verification_failed"


class PartialShardPairError(PairVerificationError):
    failure_kind = "partial_shard_pair"


class ChildStageError(RuntimeError):
    def __init__(self, stage: str, returncode: int) -> None:
        super().__init__(f"{stage} exited with return code {returncode}")
        self.stage = stage
        self.returncode = returncode


def ops_paths(run_root: Path) -> OpsPaths:
    run = Path(run_root)
    root = run.parent / f"{run.name}_ops_antigravity"
    return OpsPaths(
        root=root,
        authority=root / "authority_antigravity.json",
        authority_sha256=root / "authority_sha256_antigravity.txt",
        pid=root / "pid_antigravity.txt",
        state=root / "state_antigravity.json",
        progress_log=root / "progress_antigravity.log",
        worker_lock=root / "worker_lock_antigravity.lock",
    )


def _canonical_run_root(run_root: Path) -> Path:
    run = Path(run_root)
    if run.is_symlink():
        raise ValueError("run-root must not be a symlink")
    return run.resolve(strict=False)


def parse_method(value: str) -> str:
    method = str(value)
    if method not in ALLOWED_METHODS:
        raise ValueError(
            "deep method must be exactly deep_svdd_1d or drocc_1d"
        )
    return method


def stage_commands(
    method: str,
    db_path: Path,
    run_root: Path,
    executable: Path,
) -> tuple[tuple[str, list[str]], ...]:
    selected = parse_method(method)
    prefix = [
        str(Path(executable)),
        "-B",
        "-u",
        str(PUBLIC_BASELINE_SCRIPT),
    ]
    database = str(Path(db_path))
    root = str(Path(run_root))
    return (
        (
            "prepare",
            prefix
            + [
                "prepare",
                "--db-path",
                database,
                "--run-root",
                root,
                "--methods",
                selected,
            ],
        ),
        (
            "assess_mlx",
            prefix + ["assess-mlx", "--run-root", root],
        ),
        (
            "score",
            prefix
            + [
                "score",
                "--run-root",
                root,
                "--methods",
                selected,
                "--workers",
                str(OFFICIAL_WORKERS),
                "--epochs",
                str(OFFICIAL_EPOCHS),
                "--db-path",
                database,
            ],
        ),
        ("freeze", prefix + ["freeze", "--run-root", root]),
    )


def launch(
    method: str,
    db_path: Path,
    run_root: Path,
    *,
    executable: Path | None = None,
    spawn: Callable[..., object] = subprocess.Popen,
) -> dict[str, object]:
    selected = parse_method(method)
    database_input = Path(db_path)
    run_input = Path(run_root)
    if not database_input.is_absolute() or not run_input.is_absolute():
        raise ValueError("database and run-root paths must be absolute")
    resolved_run = _canonical_run_root(run_input)
    paths = ops_paths(resolved_run)
    if (resolved_run / "frozen_score_manifest.json").exists():
        return verify_frozen(resolved_run)
    resolved_database = database_input.resolve(strict=True)
    if resolved_database == resolved_run:
        raise ValueError("database path and run root must differ")
    selected_executable = Path(executable or sys.executable).resolve(strict=True)

    paths.root.mkdir(parents=True, exist_ok=True)
    lock_stream = paths.worker_lock.open("a+b")
    try:
        try:
            fcntl.flock(
                lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            raise ValueError("deferred deep worker is already running") from error
        authority = _build_authority(
            selected, resolved_database, resolved_run, selected_executable
        )
        authority_sha = _publish_or_verify_authority(paths, authority)
        environment = _thread_environment()
        command = [
            str(selected_executable),
            "-B",
            "-u",
            str(Path(__file__).resolve()),
            "_worker",
            "--method",
            selected,
            "--db-path",
            str(resolved_database),
            "--run-root",
            str(resolved_run),
            "--authority-sha256",
            authority_sha,
            "--lock-fd",
            str(lock_stream.fileno()),
        ]
        log_stream = paths.progress_log.open("ab", buffering=0)
        try:
            process = spawn(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                start_new_session=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                pass_fds=(lock_stream.fileno(),),
            )
        finally:
            log_stream.close()
        pid = getattr(process, "pid", None)
        if type(pid) is not int or pid < 1:
            raise ValueError("spawn did not return a valid worker PID")
        _atomic_replace(paths.pid, f"{pid}\n".encode("ascii"))
        lock_stream.close()
    except BaseException:
        if not lock_stream.closed:
            lock_stream.close()
        raise
    return {
        "status": "launched",
        "method": selected,
        "pid": pid,
        "run_root": str(resolved_run),
        "ops_root": str(paths.root.resolve()),
        "log_path": str(paths.progress_log.resolve()),
    }


def inspect_status(
    run_root: Path, now_epoch: float | None = None
) -> dict[str, object]:
    run = _canonical_run_root(run_root)
    paths = ops_paths(run)
    now = float(time.time() if now_epoch is None else now_epoch)
    if not paths.authority.is_file():
        return {
            "status": "not_started",
            "run_root": str(run),
            "ops_root": str(paths.root),
            "log_path": str(paths.progress_log),
            "completed_pairs": 0,
            "expected_pairs": EXPECTED_DATASETS,
            "abstained_pairs": 0,
        }
    try:
        authority = _load_authority_core(paths, run)
    except ValueError as error:
        return _failed_status(
            run, paths, None, EXPECTED_DATASETS, "authority_invalid", str(error)
        )
    method = str(authority["method"])
    expected = int(authority["expected_pair_count"])
    try:
        progress = _scan_verified_pairs(run, authority)
    except PairVerificationError as error:
        return _failed_status(
            run, paths, method, expected, error.failure_kind, str(error)
        )

    base: dict[str, object] = {
        "method": method,
        "run_root": str(run),
        "ops_root": str(paths.root),
        "log_path": str(paths.progress_log),
        "completed_pairs": progress.completed,
        "expected_pairs": expected,
        "abstained_pairs": progress.abstained,
        "completion_fraction": progress.completed / expected,
        "pid": None,
        "heartbeat_age_seconds": None,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "recent_dataset": progress.datasets[-1] if progress.datasets else None,
        "failure_kind": None,
        "last_error": None,
        "child_returncode": None,
    }
    manifest = run / "frozen_score_manifest.json"
    manifest_sha = run / "frozen_score_manifest.sha256"
    if manifest.exists() != manifest_sha.exists():
        return {**base, "status": "failed", "failure_kind": "manifest_incomplete"}
    if manifest.is_file() and manifest_sha.is_file():
        try:
            frozen = verify_score_root(run)
            _verify_frozen_identity(run, frozen, authority)
        except ValueError as error:
            return {
                **base,
                "status": "failed",
                "failure_kind": "manifest_verification_failed",
                "last_error": str(error),
            }
        return {**base, "status": "frozen"}

    try:
        state = read_strict_json(paths.state) if paths.state.is_file() else {}
    except ValueError as error:
        return {
            **base,
            "status": "failed",
            "failure_kind": "state_invalid",
            "last_error": str(error),
        }
    if state and (
        state.get("schema") != STATE_SCHEMA or state.get("method") != method
    ):
        return {
            **base,
            "status": "failed",
            "failure_kind": "state_invalid",
            "last_error": "operational state identity is invalid",
        }
    for key in (
        "elapsed_seconds",
        "eta_seconds",
        "recent_dataset",
        "last_error",
        "child_returncode",
    ):
        if key in state:
            base[key] = state[key]
    if state.get("status") == "failed":
        return {
            **base,
            "status": "failed",
            "failure_kind": state.get("failure_kind") or "worker_failed",
        }
    try:
        pid = _read_pid_if_present(paths.pid)
    except ValueError:
        pid = None
    base["pid"] = pid
    heartbeat = state.get("heartbeat_epoch")
    if type(heartbeat) in (int, float):
        base["heartbeat_age_seconds"] = max(0.0, now - float(heartbeat))
    if _worker_lock_is_held(paths.worker_lock):
        return {**base, "status": "running"}
    if progress.completed == expected:
        return {**base, "status": "score_complete"}
    return {**base, "status": "failed", "failure_kind": "interrupted"}


def verify_frozen(run_root: Path) -> dict[str, object]:
    run = _canonical_run_root(run_root)
    frozen = verify_score_root(run)
    paths = ops_paths(run)
    authority = _load_authority_core(paths, run)
    method = str(authority["method"])
    expected = int(authority["expected_pair_count"])
    if expected != EXPECTED_DATASETS:
        raise ValueError("authority pair count is not the frozen 1,117")
    _verify_frozen_identity(run, frozen, authority)
    return {
        "status": "frozen_verified",
        "method": method,
        "pair_count": frozen.pair_count,
        "manifest_sha256": sha256_file(run / "frozen_score_manifest.json"),
        "run_config_sha256": frozen.run_config_sha256,
        "run_root": str(run),
    }


def _build_authority(
    method: str, db_path: Path, run_root: Path, executable: Path
) -> dict[str, object]:
    git_commit, git_clean = _git_identity()
    if not git_clean:
        raise ValueError("official deferred launch requires a clean Git worktree")
    database_sha = sha256_file(db_path)
    if database_sha != EXPECTED_DB_SHA256:
        raise ValueError("database SHA256 differs from frozen paper authority")
    logical_cpus = int(os.cpu_count() or 1)
    policy_workers = max(1, logical_cpus - 1)
    return {
        "schema": AUTHORITY_SCHEMA,
        "method": method,
        "database_path": str(db_path),
        "run_root": str(run_root),
        "repository_root": str(REPOSITORY_ROOT.resolve()),
        "launcher_path": str(Path(__file__).resolve()),
        "public_baseline_script_path": str(PUBLIC_BASELINE_SCRIPT.resolve(strict=True)),
        "database_sha256": database_sha,
        "expected_dataset_count": EXPECTED_DATASETS,
        "expected_pair_count": EXPECTED_DATASETS,
        "epochs": OFFICIAL_EPOCHS,
        "seed": SEED,
        "workers": OFFICIAL_WORKERS,
        "alphas": list(ALPHAS),
        "executable": str(executable),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "scoring_closure_sha256": {
            relative: sha256_file(REPOSITORY_ROOT / relative)
            for relative in SCORING_CLOSURE_PATHS
        },
        "logical_cpu_count": logical_cpus,
        "logical_cpu_minus_one_worker_count": policy_workers,
        "actual_worker_count": OFFICIAL_WORKERS,
        "cpu_worker_policy_assessment": CPU_WORKER_POLICY_ASSESSMENT,
        "backend": _backend_assessment(),
    }


def _backend_assessment() -> dict[str, object]:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return {
        "selected_backend": "cpu",
        "pytorch_version": _version("torch"),
        "mlx_version": _version("mlx"),
        "mlx_metal_version": _version("mlx-metal"),
        "mps_built": bool(mps is not None and mps.is_built()),
        "mps_available": bool(mps is not None and mps.is_available()),
        "tensor_eligible": True,
        "exact_accelerated_path_available": False,
        "transfer_compile_overhead_benchmarked": False,
        "cpu_fallback": "frozen deterministic PyTorch CPU implementation",
        "fallback_reason": ACCELERATION_FALLBACK_REASON,
        "canonical_gate": {
            "same_input_and_seed": True,
            "finite_outputs": True,
            "identical_candidates_all_alphas": True,
            "score_absolute_tolerance": 1e-5,
            "minimum_end_to_end_speedup": 1.1,
            "include_transfer_and_synchronization": True,
        },
    }


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_backend_authority(value: object) -> None:
    expected = {
        "selected_backend": "cpu",
        "tensor_eligible": True,
        "exact_accelerated_path_available": False,
        "transfer_compile_overhead_benchmarked": False,
        "cpu_fallback": "frozen deterministic PyTorch CPU implementation",
        "fallback_reason": ACCELERATION_FALLBACK_REASON,
    }
    expected_gate = {
        "same_input_and_seed": True,
        "finite_outputs": True,
        "identical_candidates_all_alphas": True,
        "score_absolute_tolerance": 1e-5,
        "minimum_end_to_end_speedup": 1.1,
        "include_transfer_and_synchronization": True,
    }
    if not isinstance(value, dict) or set(value) != BACKEND_FIELDS:
        raise ValueError("launch authority backend schema is invalid")
    if any(
        value.get(key) != item or type(value.get(key)) is not type(item)
        for key, item in expected.items()
    ) or any(
        type(value.get(key)) is not expected_type
        for key, expected_type in (
            ("pytorch_version", str),
            ("mlx_version", str),
            ("mlx_metal_version", str),
            ("mps_built", bool),
            ("mps_available", bool),
        )
    ):
        raise ValueError("launch authority backend official identity is invalid")
    gate = value.get("canonical_gate")
    if (
        not isinstance(gate, dict)
        or set(gate) != set(expected_gate)
        or any(
            gate.get(key) != item or type(gate.get(key)) is not type(item)
            for key, item in expected_gate.items()
        )
    ):
        raise ValueError("launch authority backend gate is invalid")


def _publish_or_verify_authority(
    paths: OpsPaths, authority: Mapping[str, object]
) -> str:
    encoded = canonical_json_bytes(authority)
    digest = hashlib.sha256(encoded).hexdigest()
    if paths.authority.exists() or paths.authority_sha256.exists():
        if not paths.authority.is_file() or not paths.authority_sha256.is_file():
            raise ValueError("existing launch authority is incomplete")
        if paths.authority.read_bytes() != encoded:
            raise ValueError("existing launch authority differs from current identity")
        if paths.authority_sha256.read_text(encoding="ascii").strip() != digest:
            raise ValueError("existing launch authority SHA256 differs")
    else:
        write_create_only_bytes(paths.authority, encoded)
        write_create_only_bytes(
            paths.authority_sha256, (digest + "\n").encode("ascii")
        )
    return digest


def _load_authority_core(paths: OpsPaths, run_root: Path) -> dict[str, object]:
    if not paths.authority.is_file() or not paths.authority_sha256.is_file():
        raise ValueError("launch authority is absent or incomplete")
    expected_sha = paths.authority_sha256.read_text(encoding="ascii").strip()
    if len(expected_sha) != 64 or sha256_file(paths.authority) != expected_sha:
        raise ValueError("launch authority SHA256 mismatch")
    try:
        authority = read_strict_json(paths.authority)
    except ValueError as error:
        raise ValueError("launch authority schema is invalid") from error
    if set(authority) != AUTHORITY_FIELDS or authority.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("launch authority schema is invalid")
    method = authority.get("method")
    if type(method) is not str:
        raise ValueError("launch authority method is invalid")
    parse_method(method)
    if authority.get("run_root") != str(_canonical_run_root(run_root)):
        raise ValueError("launch authority run root differs")
    official = {
        "database_sha256": EXPECTED_DB_SHA256,
        "expected_dataset_count": EXPECTED_DATASETS,
        "expected_pair_count": EXPECTED_DATASETS,
        "epochs": OFFICIAL_EPOCHS,
        "seed": SEED,
        "workers": OFFICIAL_WORKERS,
        "alphas": list(ALPHAS),
        "actual_worker_count": OFFICIAL_WORKERS,
        "cpu_worker_policy_assessment": CPU_WORKER_POLICY_ASSESSMENT,
    }
    for key, value in official.items():
        if authority.get(key) != value:
            qualifier = "database" if key == "database_sha256" else "official"
            raise ValueError(f"launch authority {qualifier} identity is invalid")
    expected_paths = {
        "repository_root": str(REPOSITORY_ROOT.resolve()),
        "launcher_path": str(Path(__file__).resolve()),
        "public_baseline_script_path": str(PUBLIC_BASELINE_SCRIPT.resolve(strict=True)),
    }
    if any(authority.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("launch authority repository paths are invalid")
    if any(
        type(authority.get(key)) is not str
        or not Path(str(authority[key])).is_absolute()
        for key in ("database_path", "executable")
    ):
        raise ValueError("launch authority executable or database path is invalid")
    if (
        type(authority.get("git_commit")) is not str
        or len(str(authority["git_commit"])) != 40
        or authority.get("git_clean") is not True
        or not _is_sha256(authority.get("wrapper_sha256"))
        or type(authority.get("python_version")) is not str
    ):
        raise ValueError("launch authority source identity is invalid")
    logical_cpus = authority.get("logical_cpu_count")
    if (
        type(logical_cpus) is not int
        or logical_cpus < 1
        or authority.get("logical_cpu_minus_one_worker_count")
        != max(1, logical_cpus - 1)
    ):
        raise ValueError("launch authority CPU worker assessment is invalid")
    closure = authority.get("scoring_closure_sha256")
    if (
        not isinstance(closure, dict)
        or set(closure) != set(SCORING_CLOSURE_PATHS)
        or any(not _is_sha256(value) for value in closure.values())
    ):
        raise ValueError("launch authority source closure is invalid")
    _validate_backend_authority(authority.get("backend"))
    return authority


def _verify_full_authority(
    paths: OpsPaths,
    method: str,
    db_path: Path,
    run_root: Path,
    executable: Path,
    expected_sha: str,
) -> dict[str, object]:
    authority = _build_authority(method, db_path, run_root, executable)
    actual_sha = hashlib.sha256(canonical_json_bytes(authority)).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("worker authority differs from launch authority")
    _publish_or_verify_authority(paths, authority)
    return authority


def _scan_verified_pairs(
    run_root: Path,
    authority: Mapping[str, object],
    *,
    sleep: Callable[[float], object] = time.sleep,
    half_pair_grace_seconds: float = HALF_PAIR_GRACE_SECONDS,
) -> PairProgress:
    try:
        return _scan_verified_pairs_unchecked(run_root, authority)
    except PartialShardPairError:
        sleep(half_pair_grace_seconds)
        return _scan_verified_pairs_unchecked(run_root, authority)
    except PairVerificationError:
        raise
    except ValueError as error:
        raise PairVerificationError(str(error)) from error


def _scan_verified_pairs_unchecked(
    run_root: Path, authority: Mapping[str, object]
) -> PairProgress:
    shard_root = run_root / "shards"
    config_path = run_root / "run_config.json"
    config_sha_path = run_root / "run_config.sha256"
    if config_path.exists() != config_sha_path.exists():
        raise PairVerificationError("run config pair is incomplete")
    if not config_path.is_file():
        if shard_root.exists():
            raise PairVerificationError("shards exist before run config authority")
        return PairProgress(0, 0, (), None)
    expected_config_sha = config_sha_path.read_text(encoding="ascii").strip()
    actual_config_sha = sha256_file(config_path)
    if expected_config_sha != actual_config_sha:
        raise PairVerificationError("run config SHA256 mismatch")
    config = read_strict_json(config_path)
    datasets = config.get("datasets")
    methods = config.get("methods")
    if (
        config.get("schema") != "paper-public-baselines.score-run.v1"
        or not isinstance(datasets, list)
        or any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(datasets) != len(set(datasets))
        or methods != [authority["method"]]
        or config.get("database_sha256") != authority["database_sha256"]
        or config.get("seed") != SEED
        or config.get("epochs") != OFFICIAL_EPOCHS
        or config.get("alphas") != list(ALPHAS)
    ):
        raise PairVerificationError("run config identity is invalid")
    if len(datasets) != authority["expected_pair_count"]:
        raise PairVerificationError("run config pair count differs from authority")

    method = str(authority["method"])
    actual_inventory = (
        {
            path.relative_to(shard_root)
            for path in shard_root.rglob("*")
            if path.is_file()
        }
        if shard_root.is_dir()
        else set()
    )
    inventory_pairs: dict[Path, set[str]] = {}
    for relative in actual_inventory:
        if relative.suffix not in {".json", ".meta"}:
            raise PairVerificationError(
                f"score shard inventory contains an unknown file: {relative}"
            )
        inventory_pairs.setdefault(relative.with_suffix(""), set()).add(
            relative.suffix
        )
    if any(suffixes != {".json", ".meta"} for suffixes in inventory_pairs.values()):
        raise PartialShardPairError("score shard inventory contains a partial pair")
    allowed_inventory: set[Path] = set()
    completed: list[str] = []
    abstained = 0
    for dataset in datasets:
        token = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
        shard = shard_root / method / f"{token}.json"
        metadata = shard.with_suffix(".meta")
        shard_exists, metadata_exists = shard.is_file(), metadata.is_file()
        if shard_exists != metadata_exists:
            raise PartialShardPairError(
                f"partial shard/meta pair for method={method} dataset={dataset}"
            )
        if not shard_exists:
            continue
        allowed_inventory.update((shard.relative_to(shard_root), metadata.relative_to(shard_root)))
        raw_metadata = read_strict_json(metadata)
        if (
            set(raw_metadata)
            != {
                "schema",
                "run_config_sha256",
                "database_sha256",
                "shard_sha256",
            }
            or raw_metadata.get("schema")
            != "paper-public-baselines.score-shard-meta.v1"
            or raw_metadata.get("run_config_sha256") != actual_config_sha
            or raw_metadata.get("database_sha256")
            != authority["database_sha256"]
            or raw_metadata.get("shard_sha256") != sha256_file(shard)
        ):
            raise PairVerificationError(
                f"score shard metadata is invalid for dataset={dataset}"
            )
        payload = validate_score_payload(read_strict_json(shard))
        if payload["method"] != method or payload["dataset"] != dataset:
            raise PairVerificationError(
                f"score shard coordinate mismatch for dataset={dataset}"
            )
        completed.append(dataset)
        abstained += int(payload["status"] == "abstain")
    if actual_inventory != allowed_inventory:
        extras = sorted(path.as_posix() for path in actual_inventory - allowed_inventory)
        raise PairVerificationError(f"score shard inventory is invalid: {extras}")
    return PairProgress(
        completed=len(completed),
        abstained=abstained,
        datasets=tuple(completed),
        run_config_sha256=actual_config_sha,
    )


def _verify_frozen_identity(
    run_root: Path, frozen, authority: Mapping[str, object]
) -> None:
    method = str(authority["method"])
    expected = int(authority["expected_pair_count"])
    if frozen.pair_count != expected:
        raise ValueError("frozen manifest pair count differs from authority")
    if {entry.get("method") for entry in frozen.shards} != {method}:
        raise ValueError("frozen manifest method differs from authority")
    if frozen.database_sha256 != authority["database_sha256"]:
        raise ValueError("frozen manifest database SHA256 differs from authority")
    config_path = run_root / "run_config.json"
    config = read_strict_json(config_path)
    identity = {
        "schema": "paper-public-baselines.score-run.v1",
        "database_path": authority["database_path"],
        "database_sha256": authority["database_sha256"],
        "methods": [method],
        "alphas": list(ALPHAS),
        "seed": SEED,
        "epochs": OFFICIAL_EPOCHS,
        "worker_policy": "classical_up_to_7_deep_exactly_1",
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    if any(config.get(key) != value for key, value in identity.items()):
        raise ValueError("verified run config identity differs from authority")
    expected_source = {
        name: authority["scoring_closure_sha256"][
            f"experiments/paper_public_baselines/{name}"
        ]
        for name in RUN_CONFIG_SOURCE_PATHS
    }
    if config.get("source_sha256") != expected_source:
        raise ValueError("verified run config source identity differs from authority")
    versions = config.get("versions")
    if not isinstance(versions, dict) or (
        versions.get("python") != authority["python_version"]
        or versions.get("torch") != authority["backend"]["pytorch_version"]
    ):
        raise ValueError("verified run config versions differ from authority")


def _forward_signal_to_child(child, signum: int, wait_timeout: float) -> int | None:
    if child is None or child.poll() is not None:
        return None
    child.send_signal(signum)
    try:
        return child.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def _install_child_signal_handlers(
    active_child: Callable[[], object | None],
    *,
    register: Callable[[int, object], object] = signal.signal,
    wait_timeout: float = 10.0,
) -> Callable[[], None]:
    def forward(signum, _frame) -> None:
        _forward_signal_to_child(active_child(), signum, wait_timeout)

    previous = {
        signum: register(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        for signum in (signal.SIGTERM, signal.SIGINT):
            register(signum, previous[signum])
        restored = True

    return restore


def _worker(
    method: str,
    db_path: Path,
    run_root: Path,
    executable: Path,
    authority_sha256: str,
    lock_fd: int,
) -> int:
    selected = parse_method(method)
    database = Path(db_path).resolve(strict=True)
    run = _canonical_run_root(run_root)
    paths = ops_paths(run)
    lock_stream = os.fdopen(lock_fd, "a+b", closefd=True)
    child: subprocess.Popen | None = None
    restore_signals = _install_child_signal_handlers(lambda: child)
    child_returncode: int | None = None
    started_epoch = time.time()
    score_started_epoch = started_epoch
    recent_dataset: str | None = None
    initial_completed = 0
    try:
        authority = _verify_full_authority(
            paths,
            selected,
            database,
            run,
            executable,
            authority_sha256,
        )
        _write_state(
            paths,
            _state_payload(
                selected,
                "running",
                started_epoch,
                completed=0,
                expected=EXPECTED_DATASETS,
                abstained=0,
                recent_dataset=None,
                eta_seconds=None,
            ),
        )
        for stage, command in stage_commands(
            selected, database, run, executable
        ):
            _append_log(paths, f"stage_start {stage} {shlex.join(command)}")
            if stage != "score":
                child = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=_thread_environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = child.communicate()
                child_returncode = child.returncode
                child = None
                if stdout:
                    _append_log(paths, f"stage_stdout {stage} {stdout.rstrip()}")
                if stderr:
                    _append_log(paths, f"stage_stderr {stage} {stderr.rstrip()}")
                if child_returncode:
                    raise ChildStageError(stage, child_returncode)
                _append_log(paths, f"stage_finish {stage} returncode=0")
                continue

            score_started_epoch = time.time()
            initial = _scan_verified_pairs(run, authority)
            initial_completed = initial.completed
            known_datasets = set(initial.datasets)
            child = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=_thread_environment(),
                stdout=_append_stream(paths),
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                now = time.time()
                progress = _scan_verified_pairs(run, authority)
                new_datasets = [
                    dataset
                    for dataset in progress.datasets
                    if dataset not in known_datasets
                ]
                if new_datasets:
                    recent_dataset = new_datasets[-1]
                    known_datasets.update(new_datasets)
                elapsed = max(0.0, now - score_started_epoch)
                newly_completed = progress.completed - initial_completed
                eta = None
                if newly_completed >= 3 and elapsed >= 60.0:
                    rate = newly_completed / elapsed
                    eta = (EXPECTED_DATASETS - progress.completed) / rate
                state = _state_payload(
                    selected,
                    "running",
                    started_epoch,
                    completed=progress.completed,
                    expected=EXPECTED_DATASETS,
                    abstained=progress.abstained,
                    recent_dataset=recent_dataset,
                    eta_seconds=eta,
                    heartbeat_epoch=now,
                )
                _write_state(paths, state)
                _append_progress(paths, state)
                try:
                    child_returncode = child.wait(
                        timeout=HEARTBEAT_INTERVAL_SECONDS
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            child = None
            if child_returncode:
                raise ChildStageError(stage, child_returncode)
            final_progress = _scan_verified_pairs(run, authority)
            if final_progress.completed != EXPECTED_DATASETS:
                raise PairVerificationError(
                    "score command completed without exact 1,117-pair coverage"
                )
            final_state = _state_payload(
                selected,
                "score_complete",
                started_epoch,
                completed=final_progress.completed,
                expected=EXPECTED_DATASETS,
                abstained=final_progress.abstained,
                recent_dataset=recent_dataset,
                eta_seconds=0.0,
            )
            _write_state(paths, final_state)
            _append_progress(paths, final_state)
            _append_log(paths, "stage_finish score returncode=0")

        verified = verify_frozen(run)
        terminal = _state_payload(
            selected,
            "frozen",
            started_epoch,
            completed=int(verified["pair_count"]),
            expected=EXPECTED_DATASETS,
            abstained=_scan_verified_pairs(run, authority).abstained,
            recent_dataset=recent_dataset,
            eta_seconds=0.0,
        )
        _write_state(paths, terminal)
        _append_progress(paths, terminal)
        return 0
    except BaseException as error:
        forwarded_returncode = _forward_signal_to_child(
            child, signal.SIGTERM, 10.0
        )
        if forwarded_returncode is not None:
            child_returncode = forwarded_returncode
        failure_kind = (
            error.failure_kind
            if isinstance(error, PairVerificationError)
            else "child_stage_failed"
            if isinstance(error, ChildStageError)
            else "worker_exception"
        )
        failed = _state_payload(
            selected,
            "failed",
            started_epoch,
            completed=0,
            expected=EXPECTED_DATASETS,
            abstained=0,
            recent_dataset=recent_dataset,
            eta_seconds=None,
            failure_kind=failure_kind,
            last_error=str(error),
            child_returncode=(
                error.returncode
                if isinstance(error, ChildStageError)
                else child_returncode
            ),
        )
        try:
            status = inspect_status(run)
            failed["completed_pairs"] = status.get("completed_pairs", 0)
            failed["abstained_pairs"] = status.get("abstained_pairs", 0)
        except BaseException:
            pass
        _write_state(paths, failed)
        _append_progress(paths, failed)
        _append_log(paths, f"worker_failed {type(error).__name__}: {error}")
        return 1
    finally:
        restore_signals()
        lock_stream.close()


def _state_payload(
    method: str,
    status: str,
    started_epoch: float,
    *,
    completed: int,
    expected: int,
    abstained: int,
    recent_dataset: str | None,
    eta_seconds: float | None,
    heartbeat_epoch: float | None = None,
    failure_kind: str | None = None,
    last_error: str | None = None,
    child_returncode: int | None = None,
) -> dict[str, object]:
    heartbeat = time.time() if heartbeat_epoch is None else heartbeat_epoch
    return {
        "schema": STATE_SCHEMA,
        "method": method,
        "status": status,
        "pid": os.getpid(),
        "started_epoch": started_epoch,
        "heartbeat_epoch": heartbeat,
        "completed_pairs": completed,
        "expected_pairs": expected,
        "abstained_pairs": abstained,
        "elapsed_seconds": max(0.0, heartbeat - started_epoch),
        "eta_seconds": eta_seconds,
        "recent_dataset": recent_dataset,
        "failure_kind": failure_kind,
        "last_error": last_error,
        "child_returncode": child_returncode,
    }


def _write_state(paths: OpsPaths, payload: Mapping[str, object]) -> None:
    _atomic_replace(paths.state, canonical_json_bytes(payload))


def _append_progress(paths: OpsPaths, state: Mapping[str, object]) -> None:
    _append_log(
        paths,
        " ".join(
            (
                str(state["method"]),
                f"{state['completed_pairs']}/{state['expected_pairs']}",
                str(state["elapsed_seconds"]),
                str(state["eta_seconds"]),
                str(state["recent_dataset"]),
                str(state["status"]),
            )
        ),
    )


def _append_log(paths: OpsPaths, message: str) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with paths.progress_log.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_stream(paths: OpsPaths):
    paths.root.mkdir(parents=True, exist_ok=True)
    return paths.progress_log.open("a", encoding="utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp_antigravity"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _failed_status(
    run: Path,
    paths: OpsPaths,
    method: str | None,
    expected: int,
    failure_kind: str,
    last_error: str,
) -> dict[str, object]:
    return {
        "status": "failed",
        "method": method,
        "run_root": str(run),
        "ops_root": str(paths.root),
        "log_path": str(paths.progress_log),
        "completed_pairs": 0,
        "expected_pairs": expected,
        "abstained_pairs": 0,
        "failure_kind": failure_kind,
        "last_error": last_error,
    }


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    return commit, not bool(dirty.strip())


def _thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in NUMERICAL_THREAD_VARIABLES:
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = str(SEED)
    return environment


def _read_pid_if_present(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("worker PID file is invalid") from error
    if value < 1:
        raise ValueError("worker PID file is invalid")
    return value


def _worker_lock_is_held(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manage_deferred_deep_baselines_antigravity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--method", required=True)
    launch_parser.add_argument("--db-path", required=True)
    launch_parser.add_argument("--run-root", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-root", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-root", required=True)
    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--method", required=True)
    worker_parser.add_argument("--db-path", required=True)
    worker_parser.add_argument("--run-root", required=True)
    worker_parser.add_argument("--authority-sha256", required=True)
    worker_parser.add_argument("--lock-fd", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "launch":
        result = launch(
            parse_method(arguments.method),
            Path(arguments.db_path),
            Path(arguments.run_root),
        )
    elif arguments.command == "status":
        result = inspect_status(Path(arguments.run_root))
    elif arguments.command == "verify":
        result = verify_frozen(Path(arguments.run_root))
    elif arguments.command == "_worker":
        return _worker(
            parse_method(arguments.method),
            Path(arguments.db_path),
            Path(arguments.run_root),
            Path(sys.executable).resolve(strict=True),
            arguments.authority_sha256,
            arguments.lock_fd,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
