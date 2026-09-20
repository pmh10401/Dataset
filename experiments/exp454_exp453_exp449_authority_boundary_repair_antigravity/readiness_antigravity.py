"""Label-incapable readiness authority for Exp454's exact-runtime lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from typing import Callable, TypeVar

from .config_antigravity import (
    EXP454_EVALUATION_ROOT_ANTIGRAVITY,
    EXP454_FAILURE_AUDIT_ANTIGRAVITY,
    EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY,
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_INTERPRETER_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
    RUNTIME_VERSIONS_ANTIGRAVITY,
)


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138
_DATABASE_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MAP_TRIPLET_ANTIGRAVITY = frozenset((
    "docs/codemap/codemap_exp454_antigravity.html",
    "docs/codemap/codemap_exp454_antigravity.json",
    "docs/codemap/codemap_exp454_antigravity.lock",
))
_DOCUMENTS_ANTIGRAVITY = (
    (
        "approved_design",
        "docs/superpowers/specs/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity-design.html",
        "b76806b3ff9e89abfa4547f0ab5bb3ef707c28713b39675a2d1afad9e933f78f",
    ),
    (
        "implementation_report",
        "docs/EXP454_IMPLEMENTATION_REPORT_antigravity.html",
        "9c85fd7b3a8fc648e6f169f0e96d58f1680f91e1f7ddb35660c543ecf30aaa10",
    ),
    (
        "preregistration",
        "docs/EXP454_PREREGISTRATION_antigravity_KO.html",
        "653968cd94836f4da38f97801f3e98091c7b9f0e33699fc792d50ef15c557da7",
    ),
    (
        "runbook",
        "docs/EXP454_RUNBOOK_antigravity_KO.html",
        "aaa4b05303c50559795a4738f07e23f561eaa2b0c58ec9ddd6610dd618332081",
    ),
)
_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_FROZEN_PYTHON_ANTIGRAVITY = RUNTIME_VERSIONS_ANTIGRAVITY[0]
_FROZEN_NUMPY_ANTIGRAVITY = RUNTIME_VERSIONS_ANTIGRAVITY[1]
_FROZEN_AEON_ANTIGRAVITY = RUNTIME_VERSIONS_ANTIGRAVITY[2]
_FROZEN_SKLEARN_ANTIGRAVITY = RUNTIME_VERSIONS_ANTIGRAVITY[3]
_T = TypeVar("_T")


def _lower_hash_antigravity(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class _DatabaseIdentityAntigravity:
    sha256: str
    inode: int
    nlink: int
    size: int

    def __post_init__(self) -> None:
        if (
            not _lower_hash_antigravity(self.sha256)
            or type(self.inode) is not int
            or self.inode < 1
            or self.nlink != 1
            or type(self.nlink) is not int
            or type(self.size) is not int
            or self.size < 1
        ):
            raise ValueError("database readiness identity is invalid")


@dataclass(frozen=True, slots=True)
class _RuntimeStateAntigravity:
    executable: str
    prefix: str
    python_version: str
    numpy_version: str
    aeon_version: str
    sklearn_version: str

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value for value in (
            self.executable, self.prefix, self.python_version, self.numpy_version,
            self.aeon_version, self.sklearn_version,
        )):
            raise ValueError("runtime readiness state is invalid")


@dataclass(frozen=True, slots=True)
class _BackendStateAntigravity:
    pytorch_version: str | None
    mlx_version: str | None
    mlx_metal_version: str | None
    mps_built: bool
    mps_available: bool

    def __post_init__(self) -> None:
        versions = (self.pytorch_version, self.mlx_version, self.mlx_metal_version)
        if (
            any(value is not None and (type(value) is not str or not value) for value in versions)
            or type(self.mps_built) is not bool
            or type(self.mps_available) is not bool
            or (self.pytorch_version is None and (self.mps_built or self.mps_available))
            or (self.mps_available and not self.mps_built)
        ):
            raise ValueError("backend readiness state is invalid")


def _after_database_read_antigravity(_path: Path) -> None:
    """Private deterministic replacement seam for descriptor tests."""


def _database_stat_identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _verify_database_descriptor_antigravity(
    path: Path,
    expected_sha256: str,
    expected_inode: int,
) -> _DatabaseIdentityAntigravity:
    """Digest one exact regular database descriptor without constructing a query API."""
    if (
        not isinstance(path, Path)
        or not _lower_hash_antigravity(expected_sha256)
        or type(expected_inode) is not int
        or expected_inode < 1
    ):
        raise ValueError("database readiness inputs are invalid")
    descriptor = -1
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, _DATABASE_FLAGS_ANTIGRAVITY)
        opened = os.fstat(descriptor)
        identity = _database_stat_identity_antigravity(opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_ino != expected_inode
            or _database_stat_identity_antigravity(before) != identity
        ):
            raise ValueError("source database inode/link contract is invalid")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        _after_database_read_antigravity(path)
        current = os.stat(path, follow_symlinks=False)
        if (
            digest.hexdigest() != expected_sha256
            or _database_stat_identity_antigravity(os.fstat(descriptor)) != identity
            or _database_stat_identity_antigravity(current) != identity
        ):
            raise ValueError("source database changed during readiness verification")
        return _DatabaseIdentityAntigravity(
            digest.hexdigest(), opened.st_ino, opened.st_nlink, opened.st_size
        )
    except OSError as error:
        raise ValueError("source database is unavailable or unsafe") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_fixed_database_antigravity() -> _DatabaseIdentityAntigravity:
    return _verify_database_descriptor_antigravity(
        _DATABASE_ANTIGRAVITY,
        _DATABASE_SHA256_ANTIGRAVITY,
        _DATABASE_INODE_ANTIGRAVITY,
    )


def _git_antigravity(*arguments: str, repository: Path) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repository), *arguments), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("reviewed repository state cannot be verified") from error


def _validate_current_map_antigravity(repository: Path) -> None:
    script = repository / "scratch/build_codemap_exp454_antigravity.py"
    try:
        result = subprocess.run(
            (FIXED_INTERPRETER_ANTIGRAVITY, str(script), "--validate-only"),
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("current Exp454 map cannot be validated") from error
    if result.returncode != 0:
        raise ValueError("current Exp454 map is invalid")


def _validate_reviewed_map_antigravity(
    reviewed_commit: str,
    *,
    repository: Path = _REPOSITORY_ANTIGRAVITY,
    git: Callable[..., str] = _git_antigravity,
    validate_current_map: Callable[[Path], None] = _validate_current_map_antigravity,
) -> None:
    """Require the reviewed clean HEAD to be the exact Exp454 map-only child."""
    if (
        type(reviewed_commit) is not str
        or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None
        or not isinstance(repository, Path)
        or not callable(git)
        or not callable(validate_current_map)
    ):
        raise ValueError("reviewed map inputs are invalid")
    if git("rev-parse", "HEAD", repository=repository) != reviewed_commit:
        raise ValueError("reviewed commit does not equal current HEAD")
    if git("status", "--porcelain=v1", repository=repository):
        raise ValueError("reviewed working tree must be clean")
    parent = git("rev-parse", "HEAD^", repository=repository)
    if type(parent) is not str or _COMMIT_ANTIGRAVITY.fullmatch(parent) is None:
        raise ValueError("reviewed map source parent is invalid")
    changed = frozenset(
        item
        for item in git("diff", "--name-only", "HEAD^", "HEAD", repository=repository).splitlines()
        if item
    )
    if changed != _MAP_TRIPLET_ANTIGRAVITY:
        raise ValueError("reviewed HEAD is not the exact map-only child")
    validate_current_map(repository)


def _document_hashes_antigravity() -> tuple[tuple[str, str], ...]:
    observed: list[tuple[str, str]] = []
    for name, relative, expected in _DOCUMENTS_ANTIGRAVITY:
        try:
            digest = hashlib.sha256((_REPOSITORY_ANTIGRAVITY / relative).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("frozen Exp454 document is unavailable") from error
        if digest != expected:
            raise ValueError("frozen Exp454 document digest changed")
        observed.append((name, digest))
    return tuple(observed)


def _require_absent_antigravity(path: Path) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Exp454 path absence cannot be verified") from error
    raise ValueError("Exp454 path is already occupied")


def _require_initial_paths_absent_antigravity() -> None:
    for path in (
        Path(EXP454_EVALUATION_ROOT_ANTIGRAVITY),
        Path(EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY),
        Path(EXP454_FAILURE_AUDIT_ANTIGRAVITY),
    ):
        _require_absent_antigravity(path)


def _require_runtime_paths_absent_antigravity() -> None:
    _require_absent_antigravity(Path(EXP454_EVALUATION_ROOT_ANTIGRAVITY))
    _require_absent_antigravity(Path(EXP454_FAILURE_AUDIT_ANTIGRAVITY))


def _runtime_state_antigravity() -> _RuntimeStateAntigravity:
    try:
        numpy_version = importlib.metadata.version("numpy")
        aeon_version = importlib.metadata.version("aeon")
        sklearn_version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("runtime distribution metadata is unavailable") from error
    python_version = platform.python_version()
    if (
        sys.executable != FIXED_EXECUTABLE_ANTIGRAVITY
        or sys.prefix != FIXED_PREFIX_ANTIGRAVITY
        or python_version != _FROZEN_PYTHON_ANTIGRAVITY
        or numpy_version != _FROZEN_NUMPY_ANTIGRAVITY
        or aeon_version != _FROZEN_AEON_ANTIGRAVITY
        or sklearn_version != _FROZEN_SKLEARN_ANTIGRAVITY
    ):
        raise ValueError("Exp454 requires the exact fixed executable, prefix, and runtime versions")
    return _RuntimeStateAntigravity(
        sys.executable,
        sys.prefix,
        python_version,
        numpy_version,
        aeon_version,
        sklearn_version,
    )


def _distribution_version_antigravity(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_backend_probe_process_antigravity(arguments: tuple[str, ...]):  # type: ignore[no-untyped-def]
    return subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _probe_backend_metadata_antigravity(
    interpreter: str,
    *,
    distribution_version: Callable[[str], str | None] = _distribution_version_antigravity,
    run_process: Callable[[tuple[str, ...]], object] = _run_backend_probe_process_antigravity,
) -> _BackendStateAntigravity:
    """Read package metadata and isolate the MPS status import; no tensor is created."""
    if type(interpreter) is not str or not interpreter or not callable(distribution_version) or not callable(run_process):
        raise ValueError("backend probe inputs are invalid")
    pytorch_distribution = distribution_version("torch")
    mlx_version = distribution_version("mlx")
    mlx_metal_version = distribution_version("mlx-metal")
    if pytorch_distribution is None:
        return _BackendStateAntigravity(None, mlx_version, mlx_metal_version, False, False)
    probe = (
        "import json;import torch;"
        "print(json.dumps({'pytorch_version':str(torch.__version__),"
        "'mps_built':bool(torch.backends.mps.is_built()),"
        "'mps_available':bool(torch.backends.mps.is_available())},"
        "sort_keys=True,separators=(',',':')))"
    )
    result = run_process((interpreter, "-I", "-c", probe))
    if (
        getattr(result, "returncode", None) != 0
        or type(getattr(result, "stdout", None)) is not str
    ):
        raise ValueError("isolated PyTorch/MPS metadata probe failed")
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("isolated PyTorch/MPS metadata probe is invalid") from error
    if (
        type(payload) is not dict
        or set(payload) != {"pytorch_version", "mps_built", "mps_available"}
        or type(payload["pytorch_version"]) is not str
        or not payload["pytorch_version"]
        or type(payload["mps_built"]) is not bool
        or type(payload["mps_available"]) is not bool
    ):
        raise ValueError("isolated PyTorch/MPS metadata state differs")
    return _BackendStateAntigravity(
        payload["pytorch_version"],
        mlx_version,
        mlx_metal_version,
        payload["mps_built"],
        payload["mps_available"],
    )


def _capture_predecessors_antigravity() -> object:
    from .predecessor_antigravity import capture_predecessors_antigravity

    return capture_predecessors_antigravity()


def _validate_predecessors_antigravity(authority: object) -> tuple[str, ...]:
    from .predecessor_antigravity import validate_predecessors_antigravity

    return validate_predecessors_antigravity(authority)


def _run_production_readiness_phase_antigravity(
    reviewed_commit: str,
    operation: Callable[[], _T],
    precondition: Callable[[], None],
) -> _T:
    from .lifecycle_antigravity import (
        _production_paths_antigravity,
        _run_phase_with_paths_antigravity,
    )

    return _run_phase_with_paths_antigravity(
        "readiness",
        reviewed_commit,
        operation,
        _production_paths_antigravity(),
        path_precondition=precondition,
    )


@dataclass(frozen=True, slots=True)
class _ReadinessDependenciesAntigravity:
    validate_reviewed_map: Callable[[str], None]
    document_hashes: Callable[[], tuple[tuple[str, str], ...]]
    require_initial_paths_absent: Callable[[], None]
    require_runtime_paths_absent: Callable[[], None]
    verify_database: Callable[[], _DatabaseIdentityAntigravity]
    capture_predecessors: Callable[[], object]
    validate_predecessors: Callable[[object], tuple[str, ...]]
    runtime_state: Callable[[], _RuntimeStateAntigravity]
    backend_state: Callable[[], _BackendStateAntigravity]
    run_readiness_phase: Callable[[str, Callable[[], dict[str, object]], Callable[[], None]], dict[str, object]]

    def __post_init__(self) -> None:
        if any(not callable(value) for value in (
            self.validate_reviewed_map,
            self.document_hashes,
            self.require_initial_paths_absent,
            self.require_runtime_paths_absent,
            self.verify_database,
            self.capture_predecessors,
            self.validate_predecessors,
            self.runtime_state,
            self.backend_state,
            self.run_readiness_phase,
        )):
            raise ValueError("readiness dependencies are invalid")


def _production_dependencies_antigravity() -> _ReadinessDependenciesAntigravity:
    return _ReadinessDependenciesAntigravity(
        validate_reviewed_map=_validate_reviewed_map_antigravity,
        document_hashes=_document_hashes_antigravity,
        require_initial_paths_absent=_require_initial_paths_absent_antigravity,
        require_runtime_paths_absent=_require_runtime_paths_absent_antigravity,
        verify_database=_verify_fixed_database_antigravity,
        capture_predecessors=_capture_predecessors_antigravity,
        validate_predecessors=_validate_predecessors_antigravity,
        runtime_state=_runtime_state_antigravity,
        backend_state=lambda: _probe_backend_metadata_antigravity(FIXED_INTERPRETER_ANTIGRAVITY),
        run_readiness_phase=_run_production_readiness_phase_antigravity,
    )


def _readiness_with_dependencies_antigravity(
    reviewed_commit: str,
    dependencies: _ReadinessDependenciesAntigravity,
) -> dict[str, object]:
    """Private synthetic seam; production paths remain unreachable from tests."""
    if (
        type(reviewed_commit) is not str
        or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None
        or type(dependencies) is not _ReadinessDependenciesAntigravity
    ):
        raise ValueError("readiness inputs are invalid")

    def first_review_antigravity() -> None:
        dependencies.validate_reviewed_map(reviewed_commit)
        dependencies.require_initial_paths_absent()

    def operation_antigravity() -> dict[str, object]:
        documents = dependencies.document_hashes()
        if (
            type(documents) is not tuple
            or tuple(name for name, _digest in documents) != (
                "approved_design", "implementation_report", "preregistration", "runbook"
            )
            or any(type(name) is not str or not _lower_hash_antigravity(digest) for name, digest in documents)
        ):
            raise ValueError("Exp454 document readiness state is invalid")
        database = dependencies.verify_database()
        if type(database) is not _DatabaseIdentityAntigravity:
            raise ValueError("Exp454 database readiness state is invalid")
        authority = dependencies.capture_predecessors()
        dataset_order = dependencies.validate_predecessors(authority)
        if (
            type(dataset_order) is not tuple
            or len(dataset_order) != 48
            or len(set(dataset_order)) != 48
            or any(type(dataset) is not str or not dataset for dataset in dataset_order)
            or not _lower_hash_antigravity(getattr(authority, "predecessor_attestation_sha256", None))
            or not _lower_hash_antigravity(getattr(authority, "prediction_aggregate_sha256", None))
            or not _lower_hash_antigravity(getattr(authority, "exp453_stopped_receipt_sha256", None))
            or not _lower_hash_antigravity(getattr(authority, "exp453_failure_audit_sha256", None))
            or getattr(authority, "exp453_evaluation_root_absent", None) is not True
            or getattr(authority, "dataset_count", None) != 48
            or type(getattr(authority, "dataset_count", None)) is not int
            or getattr(authority, "prediction_count", None) != 96
            or type(getattr(authority, "prediction_count", None)) is not int
            or len(getattr(getattr(authority, "normalized_target", None), "target_dataset_names", ())) != 48
            or len(getattr(getattr(authority, "normalized_target", None), "target_baseline_rows", ())) != 48
        ):
            raise ValueError("Exp454 predecessor readiness state is invalid")
        runtime = dependencies.runtime_state()
        backend = dependencies.backend_state()
        if type(runtime) is not _RuntimeStateAntigravity or type(backend) is not _BackendStateAntigravity:
            raise ValueError("Exp454 runtime/backend readiness state is invalid")
        dependencies.require_runtime_paths_absent()
        dependencies.validate_reviewed_map(reviewed_commit)
        return {
            "schema": "exp454.readiness.v1",
            "experiment_id": "Exp454",
            "reviewed_commit": reviewed_commit,
            "label_access_count": 0,
            "dataset_count": 48,
            "prediction_count": 96,
            "normalized_target_count": 48,
            "exp453_stopped_receipt_sha256": authority.exp453_stopped_receipt_sha256,
            "exp453_failure_audit_sha256": authority.exp453_failure_audit_sha256,
            "exp453_evaluation_root_absent": True,
            "predecessor_attestation_sha256": authority.predecessor_attestation_sha256,
            "prediction_aggregate_sha256": authority.prediction_aggregate_sha256,
            "documents_sha256": dict(documents),
            "database": {
                "sha256": database.sha256,
                "inode": database.inode,
                "nlink": database.nlink,
                "size": database.size,
            },
            "runtime": {
                "executable": runtime.executable,
                "prefix": runtime.prefix,
                "python_version": runtime.python_version,
                "numpy_version": runtime.numpy_version,
                "aeon_version": runtime.aeon_version,
                "sklearn_version": runtime.sklearn_version,
            },
            "backend": {
                "canonical_backend": "cpu",
                "workers": 1,
                "single_process": True,
                "tensor_eligible": False,
                "tensor_ineligible_reason": "authority boundary repair copies integer confusion values and has no tensor stage",
                "pytorch_version": backend.pytorch_version,
                "mlx_version": backend.mlx_version,
                "mlx_metal_version": backend.mlx_metal_version,
                "mps_built": backend.mps_built,
                "mps_available": backend.mps_available,
                "backend_probe_isolated": True,
                "mlx_mps_workload_run": False,
                "benchmark_run": False,
            },
            "evaluation_root_absent": True,
            "failure_audit_absent": True,
        }

    return dependencies.run_readiness_phase(
        reviewed_commit, operation_antigravity, first_review_antigravity
    )


def readiness_antigravity(reviewed_commit: str) -> dict[str, object]:
    """Run the official fixed-path no-label readiness phase exactly once."""
    return _readiness_with_dependencies_antigravity(
        reviewed_commit, _production_dependencies_antigravity()
    )


__all__ = ["readiness_antigravity"]
