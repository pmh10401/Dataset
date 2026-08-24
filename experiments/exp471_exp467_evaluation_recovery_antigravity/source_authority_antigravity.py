"""Descriptor-bound reconstruction and publication of frozen Exp467 authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import subprocess
import sys
from types import MappingProxyType, ModuleType
from weakref import WeakKeyDictionary, WeakSet

from .config_antigravity import (
    CONDITION_ORDER_ANTIGRAVITY,
    EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
    EXP467_NATIVE_TOTAL_ANTIGRAVITY,
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
    EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)

_EXP467_LAUNCHER_PATH_ANTIGRAVITY = "scratch/run_exp467_numba_calibration_resume_antigravity.py"
_EXP467_VERIFIER_MODULE_ANTIGRAVITY = "experiments.exp467_numba_calibration_resume_antigravity.verifier_antigravity"
_EXP450_RUNNER_MODULE_ANTIGRAVITY = "experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity"
_PAPER_CONFIG_MODULE_ANTIGRAVITY = "paper.equipment_anomaly_detection.config_paper"
_PAPER_CONTRACTS_MODULE_ANTIGRAVITY = "paper.equipment_anomaly_detection.contracts_paper"
_PAPER_DATA_MODULE_ANTIGRAVITY = "paper.equipment_anomaly_detection.data_paper"
_SOURCE_AUTHORITY_NAME_ANTIGRAVITY = "source_authority_antigravity.json"
if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
    raise RuntimeError("Exp471 requires descriptor-relative no-follow support")
_READ_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW
_DIRECTORY_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _race_hook_antigravity(event: str, relative: str) -> None:
    """No-op seam for deterministic synthetic swap tests."""


def _safe_relative_path_antigravity(value: object) -> bool:
    if type(value) is not str or not value or value.startswith(("/", "\\")):
        return False
    normalized = value.replace("\\", "/")
    return all(part not in {"", ".", ".."} for part in normalized.split("/")) and not PurePosixPath(normalized).is_absolute()


def _digest_antigravity(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _same_identity_antigravity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


@dataclass(slots=True)
class _DirectoryAnchorAntigravity:
    descriptor: int
    descriptors: list[int]
    validators: list[tuple[int, str, os.stat_result]]

    def validate(self) -> None:
        for parent, name, expected in self.validators:
            try:
                observed = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except OSError as error:
                raise ValueError("descriptor-bound directory path changed") from error
            if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(expected, observed):
                raise ValueError("descriptor-bound directory path changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _open_absolute_directory_antigravity(path: Path, *, create: bool, scope: str) -> _DirectoryAnchorAntigravity:
    path = Path(path)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{scope} root must be absolute and non-escaping")
    descriptors: list[int] = []
    validators: list[tuple[int, str, os.stat_result]] = []
    try:
        current = os.open("/", _DIRECTORY_FLAGS_ANTIGRAVITY)
        descriptors.append(current)
        relative_parts: list[str] = []
        for part in path.parts[1:]:
            relative_parts.append(part)
            relative = "/".join(relative_parts)
            _race_hook_antigravity(f"{scope}:before_directory_open", relative)
            try:
                child = os.open(part, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise ValueError(f"{scope} root is missing")
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(part, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=current)
            except OSError as error:
                raise ValueError(f"{scope} root contains an unsafe component") from error
            descriptors.append(child)
            opened = os.fstat(child)
            if not stat.S_ISDIR(opened.st_mode):
                raise ValueError(f"{scope} root contains a non-directory")
            _race_hook_antigravity(f"{scope}:after_directory_open", relative)
            observed = os.stat(part, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(opened, observed):
                raise ValueError(f"{scope} root component changed")
            validators.append((current, part, opened))
            current = child
        anchor = _DirectoryAnchorAntigravity(current, descriptors, validators)
        anchor.validate()
        return anchor
    except Exception:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _read_descriptor_antigravity(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _read_leaf_at_antigravity(root_descriptor: int, relative: str, *, scope: str) -> bytes:
    if not _safe_relative_path_antigravity(relative):
        raise ValueError(f"{scope} path escapes its root")
    parts = relative.replace("\\", "/").split("/")
    current = root_descriptor
    opened_directories: list[int] = []
    validators: list[tuple[int, str, os.stat_result]] = []
    descriptor: int | None = None
    try:
        for index, part in enumerate(parts[:-1]):
            nested = "/".join(parts[: index + 1])
            child = os.open(part, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=current)
            opened_directories.append(child)
            opened = os.fstat(child)
            _race_hook_antigravity(f"{scope}:after_directory_open", nested)
            observed = os.stat(part, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(opened, observed):
                raise ValueError(f"{scope} directory changed")
            validators.append((current, part, opened))
            current = child
        _race_hook_antigravity(f"{scope}:before_leaf_open", relative)
        descriptor = os.open(parts[-1], _READ_FLAGS_ANTIGRAVITY, dir_fd=current)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{scope} leaf is not regular")
        _race_hook_antigravity(f"{scope}:after_leaf_open", relative)
        raw = _read_descriptor_antigravity(descriptor)
        after = os.fstat(descriptor)
        observed = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or not _same_identity_antigravity(before, after)
            or not _same_identity_antigravity(before, observed)
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError(f"{scope} leaf changed while open")
        for parent, name, expected in validators:
            observed_directory = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(observed_directory.st_mode) or not _same_identity_antigravity(expected, observed_directory):
                raise ValueError(f"{scope} directory changed while open")
        return raw
    except OSError as error:
        raise ValueError(f"{scope} path is unavailable or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for opened in reversed(opened_directories):
            os.close(opened)


def _git_bytes_from_anchor_antigravity(repository_descriptor: int, path: str) -> bytes:
    if not _safe_relative_path_antigravity(path):
        raise ValueError("Exp467 historical closure path is unsafe")
    result = subprocess.run(
        ("git", "show", f"{EXP467_SOURCE_COMMIT_ANTIGRAVITY}:{path}"),
        capture_output=True,
        check=False,
        pass_fds=(repository_descriptor,),
        preexec_fn=lambda: os.fchdir(repository_descriptor),
    )
    if result.returncode != 0:
        raise ValueError("Exp467 historical closure path is unavailable")
    return result.stdout


def _git_bytes_antigravity(repository: Path, path: str) -> bytes:
    anchor = _open_absolute_directory_antigravity(Path(repository), create=False, scope="repository")
    try:
        raw = _git_bytes_from_anchor_antigravity(anchor.descriptor, path)
        anchor.validate()
        return raw
    finally:
        anchor.close()


def _fixed_launcher_from_bytes_antigravity(repository: Path, source: bytes) -> ModuleType:
    module = ModuleType("_exp471_fixed_exp467_launcher_antigravity")
    module.__file__ = str(repository / _EXP467_LAUNCHER_PATH_ANTIGRAVITY)
    module.__package__ = ""
    exec(compile(source, f"{EXP467_SOURCE_COMMIT_ANTIGRAVITY}:{_EXP467_LAUNCHER_PATH_ANTIGRAVITY}", "exec"), module.__dict__)
    return module


def _closed_snapshot_antigravity(repository: Path) -> tuple[ModuleType, Mapping[str, bytes], Mapping[str, str]]:
    repository = Path(repository)
    anchor = _open_absolute_directory_antigravity(repository, create=False, scope="repository")
    try:
        launcher_git = _git_bytes_from_anchor_antigravity(anchor.descriptor, _EXP467_LAUNCHER_PATH_ANTIGRAVITY)
        launcher = _fixed_launcher_from_bytes_antigravity(repository, launcher_git)
        path_reader = getattr(launcher, "runtime_dependency_relative_paths_antigravity", None)
        if not callable(path_reader):
            raise ValueError("Exp467 fixed launcher has no runtime closure")
        paths = tuple(path_reader())
        if not paths or len(paths) != len(set(paths)) or any(not _safe_relative_path_antigravity(path) for path in paths):
            raise ValueError("Exp467 fixed launcher runtime closure is invalid")
        if _EXP467_LAUNCHER_PATH_ANTIGRAVITY not in paths:
            paths = tuple(sorted((*paths, _EXP467_LAUNCHER_PATH_ANTIGRAVITY)))
        captured: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for path in paths:
            historical = _git_bytes_from_anchor_antigravity(anchor.descriptor, path)
            worktree = _read_leaf_at_antigravity(anchor.descriptor, path, scope="worktree")
            digest = sha256(historical).hexdigest()
            if sha256(worktree).hexdigest() != digest:
                raise ValueError("Exp467 worktree closure differs from fixed Git bytes")
            captured[path] = worktree
            digests[path] = digest
        anchor.validate()
        _race_hook_antigravity("worktree:after_snapshot", "")
        return launcher, MappingProxyType(captured), MappingProxyType(digests)
    finally:
        anchor.close()


def _module_name_for_path_antigravity(path: str) -> tuple[str, bool] | None:
    if not path.endswith(".py"):
        return None
    parts = list(PurePosixPath(path).with_suffix("").parts)
    package = parts[-1] == "__init__"
    if package:
        parts.pop()
    return ".".join(parts), package


class _CapturedModuleFinderAntigravity(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, repository: Path, captured: Mapping[str, bytes]) -> None:
        self.repository = repository
        self.modules: dict[str, tuple[str, bytes, bool]] = {}
        for path, source in captured.items():
            details = _module_name_for_path_antigravity(path)
            if details is not None:
                module_name, package = details
                self.modules[module_name] = (path, source, package)

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        details = self.modules.get(fullname)
        if details is None:
            return None
        relative, _source, package = details
        return importlib.util.spec_from_loader(fullname, self, origin=str(self.repository / relative), is_package=package)

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        relative, source, package = self.modules[module.__name__]
        module.__file__ = str(self.repository / relative)
        module.__package__ = module.__name__ if package else module.__name__.rpartition(".")[0]
        if package:
            module.__path__ = [str((self.repository / relative).parent)]
        exec(compile(source, module.__file__, "exec"), module.__dict__)


@contextmanager
def _captured_modules_antigravity(repository: Path, captured: Mapping[str, bytes]):
    finder = _CapturedModuleFinderAntigravity(repository, captured)
    previous = {name: sys.modules.pop(name) for name in finder.modules if name in sys.modules}
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in finder.modules:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


def _require_fixed_context_antigravity(context: object) -> object:
    if getattr(context, "sha256", None) != EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY:
        raise ValueError("Exp467 reconstructed source context differs")
    return context


def reconstruct_exp467_source_context_antigravity(repository: Path) -> object:
    repository = Path(repository)
    launcher, captured, _digests = _closed_snapshot_antigravity(repository)
    with _captured_modules_antigravity(repository, captured):
        full_inputs = getattr(launcher, "_full_inputs_antigravity", None)
        if not callable(full_inputs):
            raise ValueError("Exp467 fixed launcher cannot reconstruct full inputs")
        context, _catalog = full_inputs()
        return _require_fixed_context_antigravity(context)


def _source_tree_from_directory_antigravity(descriptor: int, prefix: str, result: dict[str, str]) -> None:
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as error:
        raise ValueError("Exp467 source directory cannot be listed safely") from error
    for name in names:
        if name in {"", ".", ".."} or "/" in name:
            raise ValueError("Exp467 source tree name is unsafe")
        relative = f"{prefix}/{name}" if prefix else name
        details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(details.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS_ANTIGRAVITY, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                _race_hook_antigravity("source:after_directory_open", relative)
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(observed.st_mode) or not _same_identity_antigravity(opened, observed):
                    raise ValueError("Exp467 source directory changed")
                _source_tree_from_directory_antigravity(child, relative, result)
                final = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISDIR(final.st_mode) or not _same_identity_antigravity(opened, final):
                    raise ValueError("Exp467 source directory changed")
            finally:
                os.close(child)
        elif stat.S_ISREG(details.st_mode):
            raw = _read_leaf_at_antigravity(descriptor, name, scope="source")
            result[relative] = sha256(raw).hexdigest()
        else:
            raise ValueError("Exp467 source tree contains a non-regular entry")


def _source_file_map_antigravity(source_root: Path) -> Mapping[str, str]:
    anchor = _open_absolute_directory_antigravity(Path(source_root), create=False, scope="source")
    try:
        result: dict[str, str] = {}
        _source_tree_from_directory_antigravity(anchor.descriptor, "", result)
        anchor.validate()
        if not result:
            raise ValueError("Exp467 source tree contains no regular files")
        return MappingProxyType(dict(sorted(result.items())))
    finally:
        anchor.close()


def _authority_record_antigravity(authority: Exp471SourceAuthorityAntigravity, *, include_digest: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "source_commit": authority.source_commit,
        "source_context_sha256": authority.source_context_sha256,
        "inventory_sha256": authority.inventory_sha256,
        "catalog_sha256": authority.catalog_sha256,
        "database_sha256": authority.database_sha256,
        "source_file_sha256": dict(authority.source_file_sha256),
        "source_tree_sha256": authority.source_tree_sha256,
        "condition_manifest_sha256": dict(authority.condition_manifest_sha256),
        "total_successes": authority.total_successes,
        "imported_total": authority.imported_total,
        "native_total": authority.native_total,
    }
    if include_digest:
        record["authority_sha256"] = authority.authority_sha256
    return record


def _require_same_directory_identity_antigravity(left: Path, right: Path) -> None:
    left_anchor = _open_absolute_directory_antigravity(left, create=False, scope="source")
    right_anchor = _open_absolute_directory_antigravity(right, create=False, scope="context")
    try:
        if not _same_identity_antigravity(os.fstat(left_anchor.descriptor), os.fstat(right_anchor.descriptor)):
            raise ValueError("Exp467 supplied source root differs from its fixed context")
        left_anchor.validate()
        right_anchor.validate()
    finally:
        left_anchor.close()
        right_anchor.close()


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp471CapturedValuesApiAntigravity:
    """Values-only database capability retained from the captured source graph."""

    database_config_antigravity: object
    bind_preverified_database_read_only_antigravity: Callable[..., object]
    load_dataset_values_antigravity: Callable[..., object]
    raw_dataset_type_antigravity: type


def _captured_values_api_antigravity(
    verifier: ModuleType, context: object
) -> Exp471CapturedValuesApiAntigravity:
    config_module = importlib.import_module(_PAPER_CONFIG_MODULE_ANTIGRAVITY)
    contracts_module = importlib.import_module(_PAPER_CONTRACTS_MODULE_ANTIGRAVITY)
    data_module = importlib.import_module(_PAPER_DATA_MODULE_ANTIGRAVITY)
    runner_module = importlib.import_module(_EXP450_RUNNER_MODULE_ANTIGRAVITY)
    config_type = getattr(contracts_module, "FrozenPaperConfig", None)
    raw_dataset_type = getattr(contracts_module, "RawDataset", None)
    config = getattr(config_module, "FROZEN_CONFIG", None)
    binder = getattr(data_module, "bind_preverified_database_read_only_paper", None)
    loader = getattr(data_module, "load_dataset_values", None)
    compose = getattr(runner_module, "compose_all_conditions_antigravity", None)
    if (
        type(config_type) is not type
        or type(raw_dataset_type) is not type
        or type(config) is not config_type
        or not callable(binder)
        or not callable(loader)
        or not callable(compose)
        or getattr(data_module, "FrozenPaperConfig", None) is not config_type
        or getattr(data_module, "RawDataset", None) is not raw_dataset_type
        or getattr(data_module, "FROZEN_CONFIG", None) is not config
        or getattr(binder, "__globals__", {}).get("FROZEN_CONFIG") is not config
        or getattr(loader, "__globals__", {}).get("RawDataset") is not raw_dataset_type
        or getattr(loader, "__globals__", {}).get("FROZEN_CONFIG") is not config
        or getattr(compose, "__globals__", {}).get("RawDataset") is not raw_dataset_type
        or getattr(verifier, "compose_all_conditions_antigravity", None) is not compose
        or type(getattr(config, "database_path", None)) is not str
        or not Path(config.database_path).is_absolute()
        or not _digest_antigravity(getattr(config, "database_sha256", None))
        or config.database_sha256 != getattr(context, "database_sha256", None)
    ):
        raise ValueError("Exp467 captured values API is invalid")
    return Exp471CapturedValuesApiAntigravity(
        config,
        binder,
        loader,
        raw_dataset_type,
    )


def _verify_exp467_source_components_antigravity(repository: Path, source_root: Path) -> tuple[Exp471SourceAuthorityAntigravity, object, object, object, object, ModuleType, Exp471CapturedValuesApiAntigravity]:
    repository = Path(repository)
    source_root = Path(source_root)
    launcher, captured, _digests = _closed_snapshot_antigravity(repository)
    source_files_before = _source_file_map_antigravity(source_root)
    with _captured_modules_antigravity(repository, captured):
        verify_migration = getattr(launcher, "_verify_migration_antigravity", None)
        if not callable(verify_migration):
            raise ValueError("Exp467 fixed launcher cannot verify migration")
        context, catalog, migration_manifest = verify_migration()
        _require_fixed_context_antigravity(context)
        _require_same_directory_identity_antigravity(source_root, Path(context.run_root))
        verifier = importlib.import_module(_EXP467_VERIFIER_MODULE_ANTIGRAVITY)
        verify_completion = getattr(verifier, "verify_exp467_completion_antigravity", None)
        reopen_manifests = getattr(verifier, "read_persisted_exp467_condition_manifests_antigravity", None)
        summarize = getattr(verifier, "provenance_summary_antigravity", None)
        if not all(callable(value) for value in (verify_completion, reopen_manifests, summarize)):
            raise ValueError("Exp467 verifier API is incomplete")
        values_api = _captured_values_api_antigravity(verifier, context)
        verification = verify_completion(source_root, context=context, catalog=catalog, migration_manifest=migration_manifest)
        opened = reopen_manifests(source_root, verification)
        summary = summarize(verification)
    if (
        summary.get("total_successes") != EXP467_TOTAL_SUCCESSES_ANTIGRAVITY
        or summary.get("imported_total") != EXP467_IMPORTED_TOTAL_ANTIGRAVITY
        or summary.get("native_total") != EXP467_NATIVE_TOTAL_ANTIGRAVITY
    ):
        raise ValueError("Exp467 verified origin population differs")
    if (
        not _digest_antigravity(getattr(verification, "inventory_sha256", None))
        or getattr(verification, "catalog_sha256", None) != getattr(context, "catalog_sha256", None)
        or getattr(verification, "database_sha256", None) != getattr(context, "database_sha256", None)
    ):
        raise ValueError("Exp467 verification identity differs from its context")
    if not isinstance(opened, Mapping) or tuple(opened) != tuple(CONDITION_ORDER_ANTIGRAVITY):
        raise ValueError("Exp467 persisted condition manifest inventory differs")
    condition_digests: dict[str, str] = {}
    for condition in CONDITION_ORDER_ANTIGRAVITY:
        manifest = opened[condition]
        digest = manifest.get("manifest_sha256") if isinstance(manifest, Mapping) else None
        if not _digest_antigravity(digest):
            raise ValueError("Exp467 persisted condition manifest digest is invalid")
        condition_digests[condition] = digest
    source_files = _source_file_map_antigravity(source_root)
    if source_files != source_files_before:
        raise ValueError("Exp467 source tree changed during verification")
    source_tree_sha256 = sha256(canonical_json_bytes_exp471_antigravity(source_files)).hexdigest()
    bare = Exp471SourceAuthorityAntigravity(
        source_commit=EXP467_SOURCE_COMMIT_ANTIGRAVITY,
        source_context_sha256=EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
        inventory_sha256=verification.inventory_sha256,
        catalog_sha256=verification.catalog_sha256,
        database_sha256=verification.database_sha256,
        source_file_sha256=source_files,
        source_tree_sha256=source_tree_sha256,
        condition_manifest_sha256=condition_digests,
        total_successes=summary["total_successes"],
        imported_total=summary["imported_total"],
        native_total=summary["native_total"],
        authority_sha256="0" * 64,
    )
    authority_sha256 = sha256(canonical_json_bytes_exp471_antigravity(_authority_record_antigravity(bare, include_digest=False))).hexdigest()
    authority = Exp471SourceAuthorityAntigravity(**_authority_record_antigravity(bare, include_digest=False), authority_sha256=authority_sha256)
    return authority, verification, context, catalog, migration_manifest, verifier, values_api


def verify_exp467_source_antigravity(repository: Path, source_root: Path) -> Exp471SourceAuthorityAntigravity:
    return _verify_exp467_source_components_antigravity(repository, source_root)[0]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp471SourceSessionAntigravity:
    """Issued preflight capability; callbacks remain inaccessible until issuance."""

    authority: Exp471SourceAuthorityAntigravity
    verification: object
    context: object
    catalog: object
    migration_manifest: object
    verifier: ModuleType
    score_reader_antigravity: Callable[[], object] | None = None
    label_callback_antigravity: Callable[[], object] | None = None
    captured_values_api_antigravity: Exp471CapturedValuesApiAntigravity | None = None


def _make_source_session_api_antigravity():
    issued: WeakSet[object] = WeakSet()
    values_by_session: WeakKeyDictionary[object, object] = WeakKeyDictionary()
    expected_values_members: WeakKeyDictionary[object, tuple[object, ...]] = (
        WeakKeyDictionary()
    )

    def is_issued(value: object) -> bool:
        try:
            return value in issued
        except TypeError:
            return False

    def require_values(
        value: object,
    ) -> Exp471CapturedValuesApiAntigravity:
        if not isinstance(value, Exp471SourceSessionAntigravity) or not is_issued(
            value
        ):
            raise ValueError("Exp471 captured values API requires an issued session")
        api = value.captured_values_api_antigravity
        if (
            not isinstance(api, Exp471CapturedValuesApiAntigravity)
            or values_by_session.get(value) is not api
        ):
            raise ValueError("Exp471 captured values API is missing or replaced")
        expected = expected_values_members.get(api)
        observed = (
            api.database_config_antigravity,
            api.bind_preverified_database_read_only_antigravity,
            api.load_dataset_values_antigravity,
            api.raw_dataset_type_antigravity,
        )
        if (
            expected is None
            or len(expected) != len(observed)
            or any(left is not right for left, right in zip(observed, expected))
            or not callable(api.bind_preverified_database_read_only_antigravity)
            or not callable(api.load_dataset_values_antigravity)
            or type(api.raw_dataset_type_antigravity) is not type
            or type(getattr(api.database_config_antigravity, "database_path", None))
            is not str
            or not Path(api.database_config_antigravity.database_path).is_absolute()
            or api.database_config_antigravity.database_sha256
            != value.authority.database_sha256
        ):
            raise ValueError("Exp471 captured values API members differ")
        return api

    def open_session(
        repository: Path,
        source_root: Path,
        *,
        score_reader_antigravity: Callable[[], object] | None = None,
        label_callback_antigravity: Callable[[], object] | None = None,
    ) -> Exp471SourceSessionAntigravity:
        if score_reader_antigravity is not None and not callable(score_reader_antigravity):
            raise ValueError("Exp471 score-reader sentinel must be callable")
        if label_callback_antigravity is not None and not callable(label_callback_antigravity):
            raise ValueError("Exp471 label callback sentinel must be callable")
        authority, verification, context, catalog, migration, verifier, values_api = _verify_exp467_source_components_antigravity(repository, source_root)
        session = Exp471SourceSessionAntigravity(
            authority=authority,
            verification=verification,
            context=context,
            catalog=catalog,
            migration_manifest=migration,
            verifier=verifier,
            score_reader_antigravity=score_reader_antigravity,
            label_callback_antigravity=label_callback_antigravity,
            captured_values_api_antigravity=values_api,
        )
        issued.add(session)
        values_by_session[session] = values_api
        expected_values_members[values_api] = (
            values_api.database_config_antigravity,
            values_api.bind_preverified_database_read_only_antigravity,
            values_api.load_dataset_values_antigravity,
            values_api.raw_dataset_type_antigravity,
        )
        return session

    return open_session, is_issued, require_values


(
    open_exp467_source_session_antigravity,
    _source_session_is_issued_antigravity,
    _require_captured_values_api_antigravity,
) = _make_source_session_api_antigravity()
del _make_source_session_api_antigravity


def _canonical_authority_bytes_antigravity(authority: Exp471SourceAuthorityAntigravity) -> bytes:
    expected_digest = sha256(canonical_json_bytes_exp471_antigravity(_authority_record_antigravity(authority, include_digest=False))).hexdigest()
    if authority.authority_sha256 != expected_digest:
        raise ValueError("Exp471 source authority digest differs")
    return canonical_json_bytes_exp471_antigravity(_authority_record_antigravity(authority, include_digest=True))


def _write_all_antigravity(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Exp471 authority temporary write made no progress")
        offset += written


def _read_authority_bytes_at_antigravity(root_descriptor: int) -> bytes:
    return _read_leaf_at_antigravity(root_descriptor, _SOURCE_AUTHORITY_NAME_ANTIGRAVITY, scope="authority")


def publish_source_authority_antigravity(root: Path, authority: Exp471SourceAuthorityAntigravity) -> Path:
    if not isinstance(authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 source authority has the wrong type")
    root = Path(root)
    payload = _canonical_authority_bytes_antigravity(authority)
    anchor = _open_absolute_directory_antigravity(root, create=True, scope="authority")
    temporary_name = f".source_authority_temporary_antigravity_{secrets.token_hex(16)}"
    temporary_descriptor: int | None = None
    linked = False
    try:
        try:
            existing = _read_authority_bytes_at_antigravity(anchor.descriptor)
        except ValueError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise ValueError("Exp471 existing source authority differs")
            anchor.validate()
            return root / _SOURCE_AUTHORITY_NAME_ANTIGRAVITY
        temporary_descriptor = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=anchor.descriptor)
        opened = os.fstat(temporary_descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("Exp471 authority temporary leaf is unsafe")
        _write_all_antigravity(temporary_descriptor, payload)
        os.fchmod(temporary_descriptor, 0o444)
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(
                temporary_name,
                _SOURCE_AUTHORITY_NAME_ANTIGRAVITY,
                src_dir_fd=anchor.descriptor,
                dst_dir_fd=anchor.descriptor,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            existing = _read_authority_bytes_at_antigravity(anchor.descriptor)
            if existing != payload:
                raise ValueError("Exp471 existing source authority differs")
        if linked:
            os.fsync(anchor.descriptor)
        anchor.validate()
        return root / _SOURCE_AUTHORITY_NAME_ANTIGRAVITY
    except Exception:
        if linked:
            try:
                temporary_details = os.stat(
                    temporary_name,
                    dir_fd=anchor.descriptor,
                    follow_symlinks=False,
                )
                final_details = os.stat(
                    _SOURCE_AUTHORITY_NAME_ANTIGRAVITY,
                    dir_fd=anchor.descriptor,
                    follow_symlinks=False,
                )
                if _same_identity_antigravity(temporary_details, final_details):
                    os.unlink(
                        _SOURCE_AUTHORITY_NAME_ANTIGRAVITY,
                        dir_fd=anchor.descriptor,
                    )
                    os.fsync(anchor.descriptor)
            except OSError:
                pass
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=anchor.descriptor)
        except FileNotFoundError:
            pass
        anchor.close()


def _authority_from_bytes_antigravity(raw: bytes) -> Exp471SourceAuthorityAntigravity:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp471 source authority JSON is malformed") from error
    required = {
        "source_commit", "source_context_sha256", "inventory_sha256", "catalog_sha256",
        "database_sha256", "source_file_sha256", "source_tree_sha256",
        "condition_manifest_sha256", "total_successes", "imported_total",
        "native_total", "authority_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Exp471 source authority schema differs")
    try:
        authority = Exp471SourceAuthorityAntigravity(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 source authority fields are invalid") from error
    if raw != _canonical_authority_bytes_antigravity(authority):
        raise ValueError("Exp471 source authority bytes are noncanonical")
    return authority


def read_source_authority_antigravity(root: Path, repository: Path, source_root: Path) -> Exp471SourceAuthorityAntigravity:
    anchor = _open_absolute_directory_antigravity(Path(root), create=False, scope="authority")
    try:
        persisted = _authority_from_bytes_antigravity(_read_authority_bytes_at_antigravity(anchor.descriptor))
        anchor.validate()
    finally:
        anchor.close()
    observed = verify_exp467_source_antigravity(repository, source_root)
    if _canonical_authority_bytes_antigravity(persisted) != _canonical_authority_bytes_antigravity(observed):
        raise ValueError("Exp471 persisted source authority differs from current source")
    return persisted


__all__ = [
    "Exp471CapturedValuesApiAntigravity",
    "Exp471SourceSessionAntigravity",
    "_require_captured_values_api_antigravity",
    "_source_session_is_issued_antigravity",
    "open_exp467_source_session_antigravity",
    "publish_source_authority_antigravity",
    "read_source_authority_antigravity",
    "reconstruct_exp467_source_context_antigravity",
    "verify_exp467_source_antigravity",
]
