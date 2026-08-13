#!/usr/bin/env python3
"""Fixed-authority operations surface for the paper proposal r4 run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

OFFICIAL_INTERPRETER = Path(
    "/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314/bin/python"
)
OFFICIAL_DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
OFFICIAL_R3_ROOT = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813")
OFFICIAL_R4_ROOT = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813")
OFFICIAL_OPS_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"
)
OFFICIAL_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813"
)
EXPECTED_DATABASE_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
EXPECTED_DATASET_COUNT = 1117
INNER_NUMERICAL_THREADS = 1
PIPELINE_MODULE = "paper.equipment_anomaly_detection.pipeline_paper"
AUTHORITY_SCHEMA = "paper-r4-launch-authority-antigravity.v1"
STATE_SCHEMA = "paper-r4-ops-state-antigravity.v1"
PROGRESS_SCHEMA = "paper-r4-progress-event-antigravity.v1"
SMOKE_RESULT_SCHEMA = "paper-r4-smoke-result-antigravity.v1"
SMOKE_PREAUTHORIZATION_SCHEMA = "paper-r4-smoke-preauthorization-antigravity.v1"
SCIENTIFIC_DISTRIBUTION_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "aeon",
    "pandas",
    "numba",
    "joblib",
    "scikit-image",
    "pycatch22",
    "torch",
    "mlx",
    "mlx-metal",
)
_SYSTEM_PATH_PREFIXES = frozenset(
    {
        Path("/"),
        Path("/Users"),
        Path("/var"),
        Path("/tmp"),
        Path("/private"),
        Path("/private/var"),
        Path("/private/tmp"),
        Path("/System"),
        Path("/Library"),
    }
)
FORBIDDEN_LABEL_MODULE = "paper.equipment_anomaly_detection.label_access_paper"
STOP_SIGINT_SECONDS = 5.0
STOP_SIGTERM_SECONDS = 5.0
REVIEWED_COMMIT = re.compile(r"^[0-9a-f]{40}$")
NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
SMOKE_DATASETS = (
    "ChlorineConcentration_normal_3",
    "DodgerLoopDay_normal_1",
    "DodgerLoopDay_normal_2",
    "DodgerLoopDay_normal_4",
    "DodgerLoopDay_normal_5",
    "DodgerLoopDay_normal_6",
    "DodgerLoopDay_normal_7",
    "DodgerLoopGame_normal_1",
    "DodgerLoopGame_normal_2",
    "DodgerLoopWeekend_normal_1",
    "DodgerLoopWeekend_normal_2",
    "MelbournePedestrian_normal_1",
    "MelbournePedestrian_normal_2",
    "MelbournePedestrian_normal_3",
    "MelbournePedestrian_normal_4",
    "MelbournePedestrian_normal_5",
    "MelbournePedestrian_normal_6",
    "MelbournePedestrian_normal_7",
    "MelbournePedestrian_normal_8",
    "MelbournePedestrian_normal_9",
    "Phoneme_normal_39",
    "PigAirwayPressure_normal_1",
)
SCORING_CLOSURE_PATHS = (
    "scratch/manage_paper_r4_proposal_antigravity.py",
    "paper/equipment_anomaly_detection/pipeline_paper.py",
    "paper/equipment_anomaly_detection/runner_paper.py",
    "paper/equipment_anomaly_detection/proposal_score_paper.py",
    "paper/equipment_anomaly_detection/artifacts_paper.py",
    "paper/equipment_anomaly_detection/acceleration_paper.py",
    "paper/equipment_anomaly_detection/catalog_paper.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/hard_route_paper.py",
    "paper/equipment_anomaly_detection/config_paper.py",
)
AUTHORITY_FIELDS = {
    "schema",
    "reviewed_commit",
    "repository_root",
    "launcher_path",
    "pipeline_module",
    "database_path",
    "database_sha256",
    "r3_root",
    "r3_digest",
    "r4_root",
    "r4_absent",
    "ops_root",
    "smoke_root",
    "smoke_result_sha256",
    "executable",
    "executable_realpath",
    "python_prefix",
    "python_base_prefix",
    "python_identity_sha256",
    "scientific_distribution_inventory",
    "scientific_distribution_inventory_sha256",
    "python_version",
    "platform",
    "git_commit",
    "git_clean",
    "launcher_sha256",
    "scoring_closure_sha256",
    "logical_cpu_count",
    "dataset_workers",
    "inner_numerical_threads",
    "expected_dataset_count",
    "backend",
}
SMOKE_RESULT_FIELDS = {
    "schema",
    "authority_sha256",
    "pipeline",
    "smoke_root",
    "label_module_loaded",
    "result_sha256",
}
SMOKE_PREAUTHORIZATION_FIELDS = {
    "schema",
    "authority_sha256",
    "smoke_root",
    "smoke_root_manifest_sha256",
    "pipeline_status",
    "pipeline_sha256",
    "marker_sha256",
}
STATE_FIELDS = {
    "schema",
    "status",
    "mode",
    "started_epoch",
    "heartbeat_epoch",
    "pid",
    "child_returncode",
    "last_error",
    "reviewed_commit",
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


@dataclass(frozen=True, slots=True)
class OpsPaths:
    root: Path
    authority: Path
    authority_sha256: Path
    pid: Path
    state: Path
    progress_log: Path
    progress_events: Path
    worker_lock: Path
    smoke_result: Path
    preauthorization: Path


def ops_paths(root: Path | None = None) -> OpsPaths:
    base = Path(OFFICIAL_OPS_ROOT if root is None else root)
    return OpsPaths(
        root=base,
        authority=base / "authority_antigravity.json",
        authority_sha256=base / "authority_sha256_antigravity.txt",
        pid=base / "pid_antigravity.txt",
        state=base / "state_antigravity.json",
        progress_log=base / "progress_antigravity.log",
        progress_events=base / "progress_events_antigravity.jsonl",
        worker_lock=base / "worker_lock_antigravity.lock",
        smoke_result=base / "smoke_result_antigravity.json",
        preauthorization=base / "smoke_preauthorization_antigravity.json",
    )


def smoke_ops_paths() -> OpsPaths:
    return ops_paths(Path(str(OFFICIAL_SMOKE_ROOT) + "_ops_antigravity"))


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _reviewed_commit(value: str) -> str:
    if type(value) is not str or REVIEWED_COMMIT.fullmatch(value) is None:
        raise ValueError("reviewed commit must be a lowercase 40-hex SHA")
    return value


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        text=True,
    )
    return commit, not bool(dirty.strip())


def _reject_symlink(path: Path, name: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")


def _require_absent(path: Path, name: str) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"{name} must be absent")


def _lstat(path: Path):
    try:
        return Path(path).lstat()
    except FileNotFoundError:
        return None


def _operational_components(path: Path) -> tuple[Path, ...]:
    current = Path(path)
    items: list[Path] = []
    while current not in _SYSTEM_PATH_PREFIXES and current.parent != current:
        items.append(current)
        current = current.parent
    return tuple(items)


def _validate_directory_components(path: Path, name: str) -> None:
    for component in _operational_components(path):
        info = _lstat(component)
        if info is None:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{name} path component is not a real directory")


def _validate_ops_root(path: Path, name: str) -> None:
    _reject_symlink(path, name)
    _validate_directory_components(path, name)
    info = _lstat(path)
    if info is not None and (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)):
        raise ValueError(f"{name} is not a real directory")


def _require_regular_file(path: Path, name: str) -> None:
    info = _lstat(path)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")


def _validate_regular_file_if_present(path: Path, name: str) -> None:
    _validate_directory_components(path.parent, f"{name} parent")
    info = _lstat(path)
    if info is None:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")


def _ensure_real_directory(path: Path, name: str) -> None:
    _validate_directory_components(path, name)
    _reject_symlink(path, name)
    info = _lstat(path)
    if info is not None and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} is not a directory")
    if info is None:
        Path(path).mkdir(parents=True, exist_ok=True)
        _validate_directory_components(path, name)
        _reject_symlink(path, name)
        created = _lstat(path)
        if created is None or stat.S_ISLNK(created.st_mode) or not stat.S_ISDIR(created.st_mode):
            raise ValueError(f"{name} is not a real directory")


def _validate_ops_confinement(paths: OpsPaths, *, create_root: bool = False) -> None:
    _validate_ops_root(paths.root, "ops root")
    if create_root:
        _ensure_real_directory(paths.root, "ops root")
    for artifact, name in (
        (paths.authority, "authority"),
        (paths.authority_sha256, "authority hash"),
        (paths.pid, "PID"),
        (paths.state, "state"),
        (paths.progress_log, "progress log"),
        (paths.progress_events, "progress events"),
        (paths.worker_lock, "lock"),
        (paths.smoke_result, "smoke result"),
        (paths.preauthorization, "smoke preauthorization"),
    ):
        _validate_regular_file_if_present(artifact, name)


def _tree_digest(path: Path, name: str) -> str:
    _reject_symlink(path, name)
    if not path.is_dir():
        raise ValueError(f"{name} is absent")
    hasher = sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if item.is_symlink():
            raise ValueError(f"{name} tree must not contain a symlink")
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            kind, size, content = "dir", 0, b""
        elif item.is_file():
            content = item.read_bytes()
            kind, size = "file", len(content)
        else:
            raise ValueError(f"{name} tree contains an unsupported entry")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(kind.encode("ascii"))
        hasher.update(b"\0")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _identity_digest(path: Path) -> str:
    return _tree_digest(path, "r3 root")


def _scientific_distribution_inventory() -> dict[str, str]:
    return {name: _version(name) for name in SCIENTIFIC_DISTRIBUTION_PACKAGES}


def _inventory_digest(inventory: Mapping[str, object]) -> str:
    return sha256(canonical_json_bytes(dict(inventory))).hexdigest()


def _smoke_root_manifest_sha256() -> str:
    root = Path(OFFICIAL_SMOKE_ROOT)
    _reject_symlink(root, "smoke root")
    if not root.exists():
        return sha256(canonical_json_bytes({"state": "absent"})).hexdigest()
    return _tree_digest(root, "smoke root")


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _backend_facts() -> dict[str, object]:
    mps_built = False
    mps_available = False
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch

            mps = getattr(torch.backends, "mps", None)
            mps_built = bool(mps is not None and mps.is_built())
            mps_available = bool(mps is not None and mps.is_available())
    except Exception:
        mps_built = False
        mps_available = False
    return {
        "selected_backend": "cpu",
        "pytorch_version": _version("torch"),
        "mlx_version": _version("mlx"),
        "mlx_metal_version": _version("mlx-metal"),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "tensor_eligible": False,
        "exact_accelerated_path_available": False,
        "transfer_compile_overhead_benchmarked": False,
        "cpu_fallback": "existing paper acceleration assessment remains canonical",
        "fallback_reason": "launcher does not introduce a new score source or accelerated algorithm",
        "canonical_gate": {
            "same_input_and_seed": True,
            "finite_outputs": True,
            "identical_candidates_all_alphas": True,
            "score_absolute_tolerance": 1e-6,
            "minimum_end_to_end_speedup": 1.1,
            "include_transfer_and_synchronization": True,
        },
    }


def _validate_label_firewall() -> None:
    if FORBIDDEN_LABEL_MODULE in sys.modules:
        raise ValueError("label-capable module is loaded")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in NUMERICAL_THREAD_VARIABLES:
        environment[name] = "1"
    environment["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not environment.get("PYTHONPATH")
        else str(REPOSITORY_ROOT) + os.pathsep + environment["PYTHONPATH"]
    )
    return environment


def _logical_workers() -> tuple[int, int]:
    logical = int(os.cpu_count() or 1)
    return logical, max(1, logical - 1)


def _require_official_runtime(*, inventory_sha256: str) -> dict[str, str]:
    official = Path(OFFICIAL_INTERPRETER)
    lexical = str(official)
    if sys.executable != lexical:
        raise ValueError("running interpreter is not the official interpreter")
    expected_prefix = str(official.parent.parent)
    if sys.prefix != expected_prefix:
        raise ValueError("running interpreter prefix is not the official venv")
    if not official.is_symlink():
        raise ValueError("official interpreter must be a venv symlink")
    if not official.exists():
        raise ValueError("official interpreter is absent")
    resolved = official.resolve()
    if resolved.is_symlink() or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("official interpreter resolved target is not a regular executable")
    pyvenv = official.parent.parent / "pyvenv.cfg"
    _reject_symlink(pyvenv, "official venv identity")
    if not pyvenv.is_file():
        raise ValueError("official venv identity is absent")
    if not _is_sha256(inventory_sha256):
        raise ValueError("scientific distribution inventory hash is invalid")
    identity = {
        "executable": lexical,
        "executable_realpath": str(resolved),
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": " ".join(platform.python_build()),
        "pyvenv_cfg_sha256": sha256_file(pyvenv),
        "scientific_distribution_inventory_sha256": inventory_sha256,
    }
    return {
        "executable": identity["executable"],
        "executable_realpath": identity["executable_realpath"],
        "python_prefix": identity["python_prefix"],
        "python_base_prefix": identity["python_base_prefix"],
        "python_identity_sha256": sha256(canonical_json_bytes(identity)).hexdigest(),
    }


def build_authority(reviewed_commit: str, *, smoke_result_sha256: str | None = None) -> dict[str, object]:
    commit = _reviewed_commit(reviewed_commit)
    head, clean = _git_identity()
    if commit != head:
        raise ValueError("reviewed commit must match clean HEAD")
    if not clean:
        raise ValueError("official launch requires a clean Git worktree")
    inventory = _scientific_distribution_inventory()
    inventory_sha256 = _inventory_digest(inventory)
    runtime = _require_official_runtime(inventory_sha256=inventory_sha256)
    _reject_symlink(OFFICIAL_DATABASE, "database")
    if not OFFICIAL_DATABASE.is_file():
        raise ValueError("official database is absent")
    digest = sha256_file(OFFICIAL_DATABASE)
    if digest != EXPECTED_DATABASE_SHA256:
        raise ValueError("database SHA256 differs from frozen paper authority")
    r3_digest = _identity_digest(OFFICIAL_R3_ROOT)
    logical, workers = _logical_workers()
    closure = {relative: sha256_file(REPOSITORY_ROOT / relative) for relative in SCORING_CLOSURE_PATHS}
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "reviewed_commit": commit,
        "repository_root": str(Path(REPOSITORY_ROOT).resolve()),
        "launcher_path": str(Path(__file__).resolve()),
        "pipeline_module": PIPELINE_MODULE,
        "database_path": str(OFFICIAL_DATABASE),
        "database_sha256": digest,
        "r3_root": str(OFFICIAL_R3_ROOT),
        "r3_digest": r3_digest,
        "r4_root": str(OFFICIAL_R4_ROOT),
        "r4_absent": True,
        "ops_root": str(OFFICIAL_OPS_ROOT),
        "smoke_root": str(OFFICIAL_SMOKE_ROOT),
        "smoke_result_sha256": smoke_result_sha256,
        "executable": runtime["executable"],
        "executable_realpath": runtime["executable_realpath"],
        "python_prefix": runtime["python_prefix"],
        "python_base_prefix": runtime["python_base_prefix"],
        "python_identity_sha256": runtime["python_identity_sha256"],
        "scientific_distribution_inventory": inventory,
        "scientific_distribution_inventory_sha256": inventory_sha256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": head,
        "git_clean": True,
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "scoring_closure_sha256": closure,
        "logical_cpu_count": logical,
        "dataset_workers": workers,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS,
        "expected_dataset_count": EXPECTED_DATASET_COUNT,
        "backend": _backend_facts(),
    }
    validate_authority(authority)
    _validate_label_firewall()
    return authority


def validate_authority(authority: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(authority, Mapping) or set(authority) != AUTHORITY_FIELDS or authority.get("schema") != AUTHORITY_SCHEMA:
        raise ValueError("launch authority schema is invalid")
    extra = set(authority) - AUTHORITY_FIELDS
    if extra:
        raise ValueError("launch authority has unknown fields")
    if authority.get("pipeline_module") != PIPELINE_MODULE:
        raise ValueError("launch authority pipeline module is invalid")
    if authority.get("expected_dataset_count") != EXPECTED_DATASET_COUNT:
        raise ValueError("launch authority dataset count is invalid")
    if authority.get("inner_numerical_threads") != INNER_NUMERICAL_THREADS:
        raise ValueError("launch authority inner threads are invalid")
    if authority.get("r4_absent") is not True:
        raise ValueError("launch authority r4 absence identity is invalid")
    if authority.get("git_clean") is not True:
        raise ValueError("launch authority source identity is invalid")
    _reviewed_commit(str(authority.get("reviewed_commit")))
    if authority.get("git_commit") != authority.get("reviewed_commit"):
        raise ValueError("launch authority reviewed commit differs")
    logical = authority.get("logical_cpu_count")
    if type(logical) is not int or logical < 1 or authority.get("dataset_workers") != max(1, logical - 1):
        raise ValueError("launch authority CPU worker assessment is invalid")
    closure = authority.get("scoring_closure_sha256")
    if not isinstance(closure, dict) or set(closure) != set(SCORING_CLOSURE_PATHS) or any(not _is_sha256(value) for value in closure.values()):
        raise ValueError("launch authority source closure is invalid")
    if not _is_sha256(authority.get("launcher_sha256")) or not _is_sha256(authority.get("r3_digest")) or not _is_sha256(authority.get("database_sha256")):
        raise ValueError("launch authority digest is invalid")
    if (
        type(authority.get("executable")) is not str
        or not authority["executable"]
        or type(authority.get("executable_realpath")) is not str
        or not authority["executable_realpath"]
        or type(authority.get("python_prefix")) is not str
        or not authority["python_prefix"]
        or type(authority.get("python_base_prefix")) is not str
        or not authority["python_base_prefix"]
        or not _is_sha256(authority.get("python_identity_sha256"))
    ):
        raise ValueError("launch authority python identity is invalid")
    inventory = authority.get("scientific_distribution_inventory")
    if (
        not isinstance(inventory, dict)
        or set(inventory) != set(SCIENTIFIC_DISTRIBUTION_PACKAGES)
        or any(type(value) is not str or not value for value in inventory.values())
        or not _is_sha256(authority.get("scientific_distribution_inventory_sha256"))
        or authority.get("scientific_distribution_inventory_sha256") != _inventory_digest(inventory)
    ):
        raise ValueError("launch authority scientific distribution inventory is invalid")
    smoke = authority.get("smoke_result_sha256")
    if smoke is not None and not _is_sha256(smoke):
        raise ValueError("launch authority smoke result hash is invalid")
    backend = authority.get("backend")
    if not isinstance(backend, dict) or set(backend) != BACKEND_FIELDS or backend.get("selected_backend") != "cpu":
        raise ValueError("launch authority backend schema is invalid")
    return dict(authority)


def check(reviewed_commit: str) -> dict[str, object]:
    _require_absent(OFFICIAL_R4_ROOT, "r4 root")
    _reject_symlink(OFFICIAL_R4_ROOT, "r4 root")
    _validate_ops_root(OFFICIAL_OPS_ROOT, "ops root")
    _validate_ops_root(OFFICIAL_SMOKE_ROOT, "smoke root")
    _validate_ops_root(smoke_ops_paths().root, "smoke-ops root")
    _validate_ops_confinement(ops_paths(), create_root=False)
    _validate_ops_confinement(smoke_ops_paths(), create_root=False)
    return build_authority(reviewed_commit)


def pipeline_command(mode: str, run_root: Path) -> list[str]:
    if mode not in {"smoke", "full"}:
        raise ValueError("pipeline mode is invalid")
    command = [
        str(OFFICIAL_INTERPRETER),
        "-B",
        "-u",
        "-m",
        PIPELINE_MODULE,
        "run",
        "--db-path",
        str(OFFICIAL_DATABASE),
        "--run-root",
        str(run_root),
        "--mode",
        mode,
    ]
    if mode == "smoke":
        if SMOKE_DATASETS != tuple(sorted(SMOKE_DATASETS)) or len(SMOKE_DATASETS) != 22:
            raise ValueError("smoke dataset identity is invalid")
        for dataset in SMOKE_DATASETS:
            command.extend(["--smoke-dataset", dataset])
    return command


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_create_only(path: Path, payload: bytes) -> None:
    destination = Path(path)
    _validate_regular_file_if_present(destination, "create-only destination")
    if destination.exists() or destination.is_symlink():
        _require_regular_file(destination, "create-only destination")
        if destination.read_bytes() != payload:
            raise ValueError("existing destination differs")
        return
    _ensure_real_directory(destination.parent, "create-only parent")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp_antigravity")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise ValueError("create-only write was incomplete")
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        _require_regular_file(destination, "create-only destination")
        if destination.read_bytes() != payload:
            raise ValueError("existing destination differs") from error
    finally:
        _fsync_directory(destination.parent)
        temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp_antigravity")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_event(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(payload).decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _worker_lock_is_held(path: Path) -> bool:
    info = _lstat(path)
    if info is None:
        return False
    _require_regular_file(path, "lock")
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False


def _read_pid(path: Path) -> int | None:
    info = _lstat(path)
    if info is None:
        return None
    _require_regular_file(path, "PID")
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("worker PID file is invalid") from error
    if value < 1:
        raise ValueError("worker PID file is invalid")
    return value


def _publish_authority(paths: OpsPaths, authority: Mapping[str, object]) -> str:
    _validate_ops_confinement(paths, create_root=True)
    encoded = canonical_json_bytes(authority)
    digest = sha256(encoded).hexdigest()
    digest_bytes = (digest + "\n").encode("ascii")
    auth_present = paths.authority.exists() or paths.authority.is_symlink()
    hash_present = paths.authority_sha256.exists() or paths.authority_sha256.is_symlink()
    if auth_present:
        _require_regular_file(paths.authority, "authority")
    if hash_present:
        _require_regular_file(paths.authority_sha256, "authority hash")
    if auth_present and hash_present:
        if paths.authority.read_bytes() != encoded:
            raise ValueError("existing launch authority differs from current identity")
        if paths.authority_sha256.read_text(encoding="ascii").strip() != digest:
            raise ValueError("existing launch authority SHA256 differs")
        return digest
    if auth_present:
        if paths.authority.read_bytes() != encoded:
            raise ValueError("partial launch authority differs from current identity")
        _write_create_only(paths.authority_sha256, digest_bytes)
        return digest
    if hash_present:
        if paths.authority_sha256.read_text(encoding="ascii").strip() != digest:
            raise ValueError("partial launch authority SHA256 differs")
        _write_create_only(paths.authority, encoded)
        return digest
    _write_create_only(paths.authority, encoded)
    _write_create_only(paths.authority_sha256, digest_bytes)
    return digest


def _load_authority(paths: OpsPaths) -> dict[str, object]:
    _validate_ops_confinement(paths, create_root=False)
    if _lstat(paths.authority) is None or _lstat(paths.authority_sha256) is None:
        raise ValueError("launch authority is absent or incomplete")
    _require_regular_file(paths.authority, "authority")
    _require_regular_file(paths.authority_sha256, "authority hash")
    expected = paths.authority_sha256.read_text(encoding="ascii").strip()
    if sha256_file(paths.authority) != expected:
        raise ValueError("launch authority SHA256 mismatch")
    authority = json.loads(paths.authority.read_text(encoding="utf-8"))
    return validate_authority(authority)


def _pipeline_status(run_root: Path, mode: str) -> dict[str, object]:
    from paper.equipment_anomaly_detection.runner_paper import PipelineConfig, verify_completed

    datasets = SMOKE_DATASETS if mode == "smoke" else ()
    return verify_completed(PipelineConfig(OFFICIAL_DATABASE, run_root, mode, datasets))


def verify_smoke_authorization(reviewed_commit: str) -> dict[str, object]:
    paths = smoke_ops_paths()
    _validate_ops_confinement(paths, create_root=False)
    _validate_regular_file_if_present(paths.smoke_result, "smoke result")
    if paths.smoke_result.is_symlink() or not paths.smoke_result.is_file():
        raise ValueError("verified smoke result is absent")
    _require_regular_file(paths.smoke_result, "smoke result")
    payload = json.loads(paths.smoke_result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != SMOKE_RESULT_FIELDS or payload.get("schema") != SMOKE_RESULT_SCHEMA:
        raise ValueError("smoke result schema is invalid")
    digest = payload.get("result_sha256")
    bare = {key: value for key, value in payload.items() if key != "result_sha256"}
    if sha256(canonical_json_bytes(bare)).hexdigest() != digest:
        raise ValueError("smoke result hash differs")
    current = build_authority(reviewed_commit)
    if payload.get("authority_sha256") != sha256(canonical_json_bytes(current)).hexdigest():
        raise ValueError("smoke result authority differs")
    if payload.get("smoke_root") != str(OFFICIAL_SMOKE_ROOT):
        raise ValueError("smoke result root differs")
    if payload.get("label_module_loaded") is not False:
        raise ValueError("smoke result label boundary is invalid")
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, dict) or pipeline.get("status") != "completed":
        raise ValueError("smoke result is not completed")
    verified = _pipeline_status(OFFICIAL_SMOKE_ROOT, "smoke")
    if verified.get("status") != "completed":
        raise ValueError("smoke root semantic verification is not completed")
    return payload


def _smoke_preauthorization_payload(authority: Mapping[str, object], pipeline: Mapping[str, object]) -> dict[str, object]:
    bare = {
        "schema": SMOKE_PREAUTHORIZATION_SCHEMA,
        "authority_sha256": sha256(canonical_json_bytes(authority)).hexdigest(),
        "smoke_root": str(OFFICIAL_SMOKE_ROOT),
        "smoke_root_manifest_sha256": _smoke_root_manifest_sha256(),
        "pipeline_status": "completed",
        "pipeline_sha256": sha256(canonical_json_bytes(dict(pipeline))).hexdigest(),
    }
    digest = sha256(canonical_json_bytes(bare)).hexdigest()
    return {**bare, "marker_sha256": digest}


def _load_smoke_preauthorization(paths: OpsPaths) -> dict[str, object] | None:
    _validate_regular_file_if_present(paths.preauthorization, "smoke preauthorization")
    if _lstat(paths.preauthorization) is None:
        return None
    _require_regular_file(paths.preauthorization, "smoke preauthorization")
    payload = json.loads(paths.preauthorization.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != SMOKE_PREAUTHORIZATION_FIELDS
        or payload.get("schema") != SMOKE_PREAUTHORIZATION_SCHEMA
    ):
        raise ValueError("smoke preauthorization schema is invalid")
    digest = payload.get("marker_sha256")
    bare = {key: value for key, value in payload.items() if key != "marker_sha256"}
    if sha256(canonical_json_bytes(bare)).hexdigest() != digest:
        raise ValueError("smoke preauthorization marker hash differs")
    return payload


def _publish_smoke_preauthorization(
    paths: OpsPaths,
    authority: Mapping[str, object],
    pipeline: Mapping[str, object],
) -> dict[str, object]:
    payload = _smoke_preauthorization_payload(authority, pipeline)
    _write_create_only(paths.preauthorization, canonical_json_bytes(payload))
    return payload


def _verify_smoke_preauthorization(
    paths: OpsPaths,
    authority: Mapping[str, object],
    pipeline: Mapping[str, object],
) -> dict[str, object]:
    stored = _load_smoke_preauthorization(paths)
    if stored is None:
        raise ValueError("smoke preauthorization marker is absent")
    expected = _smoke_preauthorization_payload(authority, pipeline)
    if canonical_json_bytes(stored) != canonical_json_bytes(expected):
        raise ValueError("smoke preauthorization marker does not match current authority")
    if stored.get("pipeline_status") != "completed" or pipeline.get("status") != "completed":
        raise ValueError("smoke preauthorization is not completed")
    return stored


def _publish_smoke_result(authority: Mapping[str, object], pipeline: Mapping[str, object]) -> dict[str, object]:
    paths = smoke_ops_paths()
    _validate_ops_confinement(paths, create_root=True)
    _validate_regular_file_if_present(paths.smoke_result, "smoke result")
    bare = {
        "schema": SMOKE_RESULT_SCHEMA,
        "authority_sha256": sha256(canonical_json_bytes(authority)).hexdigest(),
        "pipeline": dict(pipeline),
        "smoke_root": str(OFFICIAL_SMOKE_ROOT),
        "label_module_loaded": FORBIDDEN_LABEL_MODULE in sys.modules,
    }
    digest = sha256(canonical_json_bytes(bare)).hexdigest()
    payload = {**bare, "result_sha256": digest}
    encoded = canonical_json_bytes(payload)
    if paths.smoke_result.exists() or paths.smoke_result.is_symlink():
        if paths.smoke_result.is_symlink() or not paths.smoke_result.is_file():
            raise ValueError("existing smoke result is invalid")
        existing = paths.smoke_result.read_bytes()
        if existing != encoded:
            raise ValueError("existing smoke result differs from the newly verified result")
        return json.loads(existing.decode("utf-8"))
    _write_create_only(paths.smoke_result, encoded)
    return payload


def _authenticate_inherited_lock_fd(lock_fd: int, lock_path: Path):
    owned = False
    try:
        if type(lock_fd) is not int or lock_fd < 0:
            raise ValueError("inherited lock fd is invalid")
        try:
            info = os.fstat(lock_fd)
        except OSError as error:
            raise ValueError("inherited lock fd is invalid") from error
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("inherited lock fd is not a regular file")
        _require_regular_file(lock_path, "canonical lock")
        lock_info = Path(lock_path).lstat()
        if (info.st_dev, info.st_ino) != (lock_info.st_dev, lock_info.st_ino):
            raise ValueError("inherited lock fd does not match canonical lock")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("inherited lock fd is contended") from error
        stream = os.fdopen(lock_fd, "a+b", closefd=True)
        owned = True
        return stream
    except BaseException:
        if not owned:
            try:
                os.close(lock_fd)
            except Exception:
                pass
        raise


def _acquire_lock(paths: OpsPaths):
    _validate_ops_confinement(paths, create_root=True)
    _validate_regular_file_if_present(paths.worker_lock, "lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(paths.worker_lock, flags, 0o600)
    except OSError as error:
        raise ValueError("worker lock is invalid") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("worker lock must be a regular non-symlink file")
        stream = os.fdopen(descriptor, "r+b", closefd=True)
    except Exception:
        os.close(descriptor)
        raise
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise ValueError("paper r4 worker is already running") from error
    return stream


def _spawn_worker(
    reviewed_commit: str,
    authority_sha256: str,
    mode: str,
    lock_stream,
    *,
    spawn=subprocess.Popen,
) -> dict[str, object]:
    paths = ops_paths() if mode == "full" else smoke_ops_paths()
    command = [
        str(OFFICIAL_INTERPRETER),
        "-B",
        "-u",
        str(Path(__file__).resolve()),
        "_worker",
        "--reviewed-commit",
        reviewed_commit,
        "--authority-sha256",
        authority_sha256,
        "--lock-fd",
        str(lock_stream.fileno()),
        "--mode",
        mode,
    ]
    log_stream = paths.progress_log.open("ab", buffering=0)
    try:
        process = spawn(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=_thread_environment(),
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
    return {"status": "launched", "pid": pid, "command": command, "mode": mode}


def launch(reviewed_commit: str, *, spawn=subprocess.Popen) -> dict[str, object]:
    check(reviewed_commit)
    smoke = verify_smoke_authorization(reviewed_commit)
    authority = build_authority(reviewed_commit, smoke_result_sha256=str(smoke["result_sha256"]))
    if OFFICIAL_R4_ROOT.exists() or OFFICIAL_R4_ROOT.is_symlink():
        raise ValueError("r4 root already exists; use resume")
    paths = ops_paths()
    lock_stream = _acquire_lock(paths)
    try:
        digest = _publish_authority(paths, authority)
        return {
            **_spawn_worker(reviewed_commit, digest, "full", lock_stream, spawn=spawn),
            "run_root": str(OFFICIAL_R4_ROOT),
            "ops_root": str(paths.root),
            "authority_sha256": digest,
        }
    finally:
        lock_stream.close()


def _verified_pre_genesis_shape(root: Path) -> bool:
    info = _lstat(root)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    children = sorted(Path(root).iterdir(), key=lambda item: item.name)
    if not children:
        return True
    if len(children) != 1:
        return False
    journal = children[0]
    if journal.name != "pipeline_journal_paper":
        return False
    journal_info = _lstat(journal)
    if journal_info is None or stat.S_ISLNK(journal_info.st_mode) or not stat.S_ISDIR(journal_info.st_mode):
        return False
    return not any(journal.iterdir())


def resume(reviewed_commit: str, *, spawn=subprocess.Popen) -> dict[str, object]:
    if not OFFICIAL_R4_ROOT.is_dir() or OFFICIAL_R4_ROOT.is_symlink():
        raise ValueError("resume requires an incomplete verified r4 root")
    paths = ops_paths()
    stored = _load_authority(paths)
    smoke = verify_smoke_authorization(reviewed_commit)
    current = build_authority(reviewed_commit, smoke_result_sha256=str(smoke["result_sha256"]))
    if canonical_json_bytes(stored) != canonical_json_bytes(current):
        raise ValueError("resume authority differs from the stored launch identity")
    if not _verified_pre_genesis_shape(OFFICIAL_R4_ROOT):
        try:
            pipeline = _pipeline_status(OFFICIAL_R4_ROOT, "full")
        except Exception as error:
            raise ValueError("resume root is not a verified incomplete or pre-genesis paper run") from error
        if pipeline.get("status") == "completed":
            raise ValueError("resume requires an incomplete verified r4 root")
        if pipeline.get("status") not in {"proposal_score_incomplete", "completed_prefix"}:
            raise ValueError("resume root is not a verified incomplete paper run")
    digest = _publish_authority(paths, current)
    lock_stream = _acquire_lock(paths)
    try:
        return {
            **_spawn_worker(reviewed_commit, digest, "full", lock_stream, spawn=spawn),
            "status": "resumed",
            "run_root": str(OFFICIAL_R4_ROOT),
            "ops_root": str(paths.root),
            "authority_sha256": digest,
        }
    finally:
        lock_stream.close()


def _read_process_argv(pid: int) -> tuple[str, ...]:
    linux = Path(f"/proc/{pid}/cmdline")
    if linux.is_file():
        parts = linux.read_bytes().split(b"\0")
        return tuple(part.decode("utf-8", "surrogateescape") for part in parts if part)
    try:
        import ctypes
        import ctypes.util
        import struct

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, int(pid))
        length = ctypes.c_size_t(1 << 16)
        buffer = ctypes.create_string_buffer(length.value)
        if libc.sysctl(mib, 3, buffer, ctypes.byref(length), None, 0) != 0 or length.value < 4:
            return ()
        raw = buffer.raw[: length.value]
        argc = struct.unpack_from("<i", raw, 0)[0]
        rest = raw[4:]
        nul = rest.find(b"\0")
        if nul < 0:
            return ()
        rest = rest[nul + 1 :].lstrip(b"\0")
        values: list[str] = []
        for _ in range(max(0, argc)):
            nul = rest.find(b"\0")
            if nul < 0:
                break
            values.append(rest[:nul].decode("utf-8", "surrogateescape"))
            rest = rest[nul + 1 :]
        return tuple(values)
    except (OSError, ValueError, TypeError):
        return ()


def _process_command_identity(pid: int) -> tuple[str, ...]:
    argv = _read_process_argv(pid)
    if not argv:
        raise ValueError("exact OS argv cannot be read")
    return argv


def _official_interpreter_tokens() -> frozenset[str]:
    official = Path(OFFICIAL_INTERPRETER)
    tokens = {str(official), str(official.resolve())}
    try:
        if Path(sys.executable).resolve() == official.resolve():
            self_argv = _read_process_argv(os.getpid())
            if self_argv:
                tokens.add(self_argv[0])
    except (OSError, ValueError):
        pass
    app = official.resolve().parent.parent / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    if app.is_file():
        tokens.add(str(app))
    return frozenset(tokens)


def _command_is_this_worker(command: object, reviewed_commit: str, authority_sha256: str) -> bool:
    if type(command) is not tuple:
        return False
    if any(type(token) is not str for token in command):
        return False
    try:
        commit = _reviewed_commit(reviewed_commit)
    except ValueError:
        return False
    if type(authority_sha256) is not str or not _is_sha256(authority_sha256):
        return False
    if len(command) != 13:
        return False
    launcher = str(Path(__file__).resolve())
    expected = (
        "-B",
        "-u",
        launcher,
        "_worker",
        "--reviewed-commit",
        commit,
        "--authority-sha256",
        authority_sha256,
        "--lock-fd",
    )
    if command[1:10] != expected:
        return False
    if command[0] not in _official_interpreter_tokens():
        return False
    lock_fd = command[10]
    if not lock_fd.isdigit() or lock_fd != str(int(lock_fd)):
        return False
    return command[11:13] == ("--mode", "full")


def _state_payload(
    *,
    status: str,
    mode: str,
    started_epoch: float,
    pid: int,
    reviewed_commit: str,
    child_returncode: int | None,
    last_error: str | None,
) -> dict[str, object]:
    payload = {
        "schema": STATE_SCHEMA,
        "status": status,
        "mode": mode,
        "started_epoch": started_epoch,
        "heartbeat_epoch": time.time(),
        "pid": pid,
        "child_returncode": child_returncode,
        "last_error": last_error,
        "reviewed_commit": reviewed_commit,
    }
    return _validate_state(payload)


def _validate_state(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != STATE_FIELDS or payload.get("schema") != STATE_SCHEMA:
        raise ValueError("ops state schema is invalid")
    if payload.get("status") not in {"running", "exited", "failed"}:
        raise ValueError("ops state status is invalid")
    if payload.get("mode") not in {"smoke", "full"}:
        raise ValueError("ops state mode is invalid")
    if type(payload.get("pid")) is not int or payload["pid"] < 1:
        raise ValueError("ops state pid is invalid")
    _reviewed_commit(str(payload.get("reviewed_commit")))
    if type(payload.get("started_epoch")) not in (int, float) or type(payload.get("heartbeat_epoch")) not in (int, float):
        raise ValueError("ops state timestamps are invalid")
    if payload.get("child_returncode") is not None and type(payload.get("child_returncode")) is not int:
        raise ValueError("ops state child returncode is invalid")
    if payload.get("last_error") is not None and type(payload.get("last_error")) is not str:
        raise ValueError("ops state last_error is invalid")
    return payload


def stop(reviewed_commit: str) -> dict[str, object]:
    commit = _reviewed_commit(reviewed_commit)
    paths = ops_paths()
    try:
        _validate_ops_confinement(paths, create_root=False)
    except ValueError:
        return {"status": "stop_refused", "pid": None, "signaled": False}
    if not _worker_lock_is_held(paths.worker_lock):
        return {"status": "already_stopped", "pid": None, "signaled": False}
    refused = {"status": "stop_refused", "pid": None, "signaled": False}
    pid = None
    try:
        stored = _load_authority(paths)
        current = build_authority(commit, smoke_result_sha256=stored.get("smoke_result_sha256"))
        if canonical_json_bytes(stored) != canonical_json_bytes(current):
            return refused
        digest = paths.authority_sha256.read_text(encoding="ascii").strip()
        if digest != sha256(canonical_json_bytes(stored)).hexdigest():
            return refused
        pid = _read_pid(paths.pid)
        state = _validate_state(json.loads(paths.state.read_text(encoding="utf-8")))
        identity = _process_command_identity(pid) if type(pid) is int else ()
    except (OSError, ValueError, json.JSONDecodeError):
        return {**refused, "pid": pid}
    if (
        type(pid) is not int
        or state.get("status") != "running"
        or state.get("pid") != pid
        or state.get("reviewed_commit") != commit
        or not _command_is_this_worker(identity, commit, digest)
    ):
        return {**refused, "pid": pid}
    os.kill(pid, signal.SIGINT)
    deadline = time.time() + STOP_SIGINT_SECONDS
    while time.time() < deadline:
        if not _pid_exists(pid):
            return {"status": "stopped", "pid": pid, "signaled": "SIGINT"}
        time.sleep(0.05)
    try:
        identity = _process_command_identity(pid)
    except (OSError, ValueError):
        return {**refused, "pid": pid, "signaled": "SIGINT"}
    if not _command_is_this_worker(identity, commit, digest):
        return {**refused, "pid": pid, "signaled": "SIGINT"}
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + STOP_SIGTERM_SECONDS
    while time.time() < deadline:
        if not _pid_exists(pid):
            return {"status": "stopped", "pid": pid, "signaled": "SIGTERM"}
        time.sleep(0.05)
    return {"status": "stop_timeout", "pid": pid, "signaled": "SIGTERM"}


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_status(reviewed_commit: str, now_epoch: float | None = None) -> dict[str, object]:
    commit = _reviewed_commit(reviewed_commit)
    now = time.time() if now_epoch is None else float(now_epoch)
    paths = ops_paths()
    try:
        _validate_ops_confinement(paths, create_root=False)
    except ValueError as error:
        return {
            "run_root": str(OFFICIAL_R4_ROOT),
            "ops_root": str(paths.root),
            "pid": None,
            "lock_held": False,
            "completed": 0,
            "failed": 0,
            "pending": EXPECTED_DATASET_COUNT,
            "percentage": 0.0,
            "elapsed_seconds": None,
            "rate_per_second": None,
            "eta_seconds": None,
            "status": "tampered",
            "failure_kind": "ops_confinement",
            "last_error": str(error),
        }
    base: dict[str, object] = {
        "run_root": str(OFFICIAL_R4_ROOT),
        "ops_root": str(paths.root),
        "pid": None,
        "lock_held": _worker_lock_is_held(paths.worker_lock),
        "completed": 0,
        "failed": 0,
        "pending": EXPECTED_DATASET_COUNT,
        "percentage": 0.0,
        "elapsed_seconds": None,
        "rate_per_second": None,
        "eta_seconds": None,
    }
    if not OFFICIAL_R4_ROOT.exists() and not paths.authority.is_file():
        return {**base, "status": "not_started"}
    try:
        pid = _read_pid(paths.pid)
    except ValueError:
        pid = None
    base["pid"] = pid
    if paths.authority.is_file():
        try:
            stored = _load_authority(paths)
            current = build_authority(commit, smoke_result_sha256=stored.get("smoke_result_sha256"))
            if canonical_json_bytes(stored) != canonical_json_bytes(current):
                return {**base, "status": "tampered", "failure_kind": "authority_differs"}
        except ValueError as error:
            return {**base, "status": "tampered", "failure_kind": "authority_invalid", "last_error": str(error)}
    if OFFICIAL_R4_ROOT.exists():
        try:
            pipeline = _pipeline_status(OFFICIAL_R4_ROOT, "full")
        except Exception as error:
            return {**base, "status": "tampered", "failure_kind": "artifact_invalid", "last_error": str(error)}
        completed = int(pipeline.get("completed") or (EXPECTED_DATASET_COUNT if pipeline.get("status") == "completed" else 0))
        failed = int(pipeline.get("failed") or 0)
        pending = int(pipeline.get("pending") or (0 if pipeline.get("status") == "completed" else EXPECTED_DATASET_COUNT - completed))
        base.update({"completed": completed, "failed": failed, "pending": pending})
        base["percentage"] = 0.0 if EXPECTED_DATASET_COUNT == 0 else 100.0 * completed / EXPECTED_DATASET_COUNT
        state: dict[str, object] = {}
        if paths.state.is_file():
            try:
                state = _validate_state(json.loads(paths.state.read_text(encoding="utf-8")))
            except ValueError as error:
                return {**base, "status": "tampered", "failure_kind": "state_invalid", "last_error": str(error)}
        started = state.get("started_epoch")
        if type(started) in (int, float):
            elapsed = max(0.0, now - float(started))
            base["elapsed_seconds"] = elapsed
            if completed > 0 and elapsed > 0:
                rate = completed / elapsed
                base["rate_per_second"] = rate
                base["eta_seconds"] = max(0.0, pending / rate)
        if base["lock_held"]:
            return {**base, "status": "running", "pipeline": pipeline}
        if pipeline.get("status") == "completed":
            return {**base, "status": "completed", "pipeline": pipeline}
        if pipeline.get("status") == "proposal_score_incomplete":
            return {**base, "status": "incomplete", "pipeline": pipeline}
        return {**base, "status": str(pipeline.get("status")), "pipeline": pipeline}
    if base["lock_held"]:
        return {**base, "status": "running"}
    return {**base, "status": "not_started"}


def smoke(reviewed_commit: str, *, run_child=None) -> dict[str, object]:
    check(reviewed_commit)
    if OFFICIAL_SMOKE_ROOT.is_symlink():
        raise ValueError("smoke root must not be a symlink")
    smoke_paths = smoke_ops_paths()
    lock_stream = _acquire_lock(smoke_paths)
    try:
        existing = None
        if OFFICIAL_SMOKE_ROOT.exists() or OFFICIAL_SMOKE_ROOT.is_symlink():
            _reject_symlink(OFFICIAL_SMOKE_ROOT, "smoke root")
            try:
                existing = _pipeline_status(OFFICIAL_SMOKE_ROOT, "smoke")
            except Exception as error:
                raise ValueError("existing smoke root does not match this authority") from error
        authority = build_authority(reviewed_commit)
        _publish_authority(smoke_paths, authority)
        marker = _load_smoke_preauthorization(smoke_paths)
        already_completed = bool(existing is not None and existing.get("status") == "completed")
        if marker is not None:
            pipeline = _pipeline_status(OFFICIAL_SMOKE_ROOT, "smoke")
            if pipeline.get("status") != "completed":
                raise ValueError("smoke preauthorization is present but the smoke root is not completed")
            _verify_smoke_preauthorization(smoke_paths, authority, pipeline)
            _validate_label_firewall()
            result = _publish_smoke_result(authority, pipeline)
            return {
                "status": "smoke_verified",
                "child_returncode": None,
                "pipeline": pipeline,
                "result_sha256": result["result_sha256"],
                "smoke_root": str(OFFICIAL_SMOKE_ROOT),
                "reuse": True,
            }
        runner = _run_pipeline_child if run_child is None else run_child
        child_code = None
        if not already_completed:
            child_code = runner("smoke", OFFICIAL_SMOKE_ROOT, smoke_paths)
        pipeline = _pipeline_status(OFFICIAL_SMOKE_ROOT, "smoke")
        _validate_label_firewall()
        if pipeline.get("status") == "proposal_score_incomplete":
            return {
                "status": "smoke_incomplete",
                "child_returncode": child_code,
                "pipeline": pipeline,
                "smoke_root": str(OFFICIAL_SMOKE_ROOT),
            }
        if pipeline.get("status") != "completed":
            raise ValueError("smoke result is not a verified completed or incomplete wave")
        _publish_smoke_preauthorization(smoke_paths, authority, pipeline)
        return {
            "status": "smoke_confirmation_required",
            "child_returncode": child_code,
            "pipeline": pipeline,
            "smoke_root": str(OFFICIAL_SMOKE_ROOT),
        }
    finally:
        lock_stream.close()


def _run_pipeline_child(mode: str, run_root: Path, paths: OpsPaths) -> int:
    command = pipeline_command(mode, run_root)
    _append_log(paths.progress_log, "child_start " + " ".join(command))
    child = subprocess.Popen(
        command,
        cwd=str(REPOSITORY_ROOT),
        env=_thread_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert child.stdout is not None
    for line in child.stdout:
        text = line.rstrip("\n")
        _append_log(paths.progress_log, text)
        _append_event(
            paths.progress_events,
            {
                "schema": PROGRESS_SCHEMA,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stream": "stdout",
                "line": text,
                "child_pid": child.pid,
            },
        )
    child.stdout.close()
    code = child.wait()
    _append_log(paths.progress_log, f"child_exit {code}")
    return code


def _install_child_signal_handlers(active_child, *, register=signal.signal):
    def forward(signum, _frame) -> None:
        child = active_child()
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous = {signum: register(signum, forward) for signum in (signal.SIGTERM, signal.SIGINT)}

    def restore() -> None:
        for signum, handler in previous.items():
            register(signum, handler)

    return restore


def _worker(reviewed_commit: str, authority_sha256: str, lock_fd: int, mode: str) -> int:
    paths = ops_paths() if mode == "full" else smoke_ops_paths()
    lock_stream = None
    child = None
    restore = _install_child_signal_handlers(lambda: child)
    started = time.time()
    authenticated = False
    try:
        lock_stream = _authenticate_inherited_lock_fd(lock_fd, paths.worker_lock)
        stored = _load_authority(paths)
        smoke_hash = None
        if mode == "full":
            smoke_hash = str(verify_smoke_authorization(reviewed_commit)["result_sha256"])
        authority = build_authority(reviewed_commit, smoke_result_sha256=smoke_hash)
        digest = sha256(canonical_json_bytes(authority)).hexdigest()
        if digest != authority_sha256:
            raise ValueError("worker authority differs from launch authority")
        if canonical_json_bytes(stored) != canonical_json_bytes(authority):
            raise ValueError("worker authority differs from stored launch identity")
        authenticated = True
        _atomic_replace(paths.pid, f"{os.getpid()}\n".encode("ascii"))
        _atomic_replace(
            paths.state,
            canonical_json_bytes(
                _state_payload(
                    status="running",
                    mode=mode,
                    started_epoch=started,
                    pid=os.getpid(),
                    reviewed_commit=reviewed_commit,
                    child_returncode=None,
                    last_error=None,
                )
            ),
        )
        run_root = OFFICIAL_R4_ROOT if mode == "full" else OFFICIAL_SMOKE_ROOT
        command = pipeline_command(mode, run_root)
        _append_event(
            paths.progress_events,
            {
                "schema": PROGRESS_SCHEMA,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stream": "state",
                "line": "child_start",
                "command": command,
                "child_pid": None,
            },
        )
        child = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=_thread_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert child.stdout is not None
        for line in child.stdout:
            text = line.rstrip("\n")
            _append_log(paths.progress_log, text)
            _append_event(
                paths.progress_events,
                {
                    "schema": PROGRESS_SCHEMA,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stream": "stdout",
                    "line": text,
                    "child_pid": child.pid,
                },
            )
        child.stdout.close()
        code = child.wait()
        child = None
        _atomic_replace(
            paths.state,
            canonical_json_bytes(
                _state_payload(
                    status="exited",
                    mode=mode,
                    started_epoch=started,
                    pid=os.getpid(),
                    reviewed_commit=reviewed_commit,
                    child_returncode=code,
                    last_error=None,
                )
            ),
        )
        return int(code)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=STOP_SIGTERM_SECONDS)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if not authenticated:
            print(f"worker_preauth_failed {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        _atomic_replace(
            paths.state,
            canonical_json_bytes(
                _state_payload(
                    status="failed",
                    mode=mode,
                    started_epoch=started,
                    pid=os.getpid(),
                    reviewed_commit=reviewed_commit,
                    child_returncode=None,
                    last_error=str(error),
                )
            ),
        )
        _append_log(paths.progress_log, f"worker_failed {type(error).__name__}: {error}")
        return 1
    finally:
        restore()
        if lock_stream is not None:
            lock_stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manage_paper_r4_proposal_antigravity", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "smoke", "launch", "status", "stop", "resume"):
        child = commands.add_parser(name, allow_abbrev=False)
        child.add_argument("--reviewed-commit", required=True, type=_reviewed_commit)
    worker = commands.add_parser("_worker", allow_abbrev=False)
    worker.add_argument("--reviewed-commit", required=True, type=_reviewed_commit)
    worker.add_argument("--authority-sha256", required=True)
    worker.add_argument("--lock-fd", required=True, type=int)
    worker.add_argument("--mode", required=True, choices=("smoke", "full"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "check":
        result = check(arguments.reviewed_commit)
        code = 0
    elif arguments.command == "smoke":
        result = smoke(arguments.reviewed_commit)
        code = 0 if result["status"] == "smoke_verified" else 2
    elif arguments.command == "launch":
        result = launch(arguments.reviewed_commit)
        code = 0
    elif arguments.command == "status":
        result = inspect_status(arguments.reviewed_commit)
        code = 0
    elif arguments.command == "stop":
        result = stop(arguments.reviewed_commit)
        code = 0
    elif arguments.command == "resume":
        result = resume(arguments.reviewed_commit)
        code = 0
    elif arguments.command == "_worker":
        return _worker(arguments.reviewed_commit, arguments.authority_sha256, arguments.lock_fd, arguments.mode)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
