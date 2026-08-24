"""Fail-closed reconstruction and publication of the frozen Exp467 source authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from types import MappingProxyType, ModuleType
from weakref import WeakSet

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


_EXP467_LAUNCHER_PATH_ANTIGRAVITY = (
    "scratch/run_exp467_numba_calibration_resume_antigravity.py"
)
_EXP467_VERIFIER_MODULE_ANTIGRAVITY = (
    "experiments.exp467_numba_calibration_resume_antigravity.verifier_antigravity"
)
_SOURCE_AUTHORITY_NAME_ANTIGRAVITY = "source_authority_antigravity.json"


def _safe_relative_path_antigravity(value: object) -> bool:
    if type(value) is not str or not value or value.startswith(("/", "\\")):
        return False
    parts = value.replace("\\", "/").split("/")
    return (
        all(part not in {"", ".", ".."} for part in parts)
        and not PurePosixPath(value.replace("\\", "/")).is_absolute()
    )


def _digest_antigravity(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_bytes_antigravity(repository: Path, path: str) -> bytes:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "show",
            f"{EXP467_SOURCE_COMMIT_ANTIGRAVITY}:{path}",
        ),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Exp467 historical closure path is unavailable")
    return result.stdout


def _require_matching_worktree_closure_antigravity(
    repository: Path, paths: Sequence[str]
) -> Mapping[str, str]:
    fixed = {
        path: sha256(_git_bytes_antigravity(repository, path)).hexdigest()
        for path in paths
    }
    for path, digest in fixed.items():
        target = repository / path
        try:
            details = target.lstat()
        except FileNotFoundError as error:
            raise ValueError(
                "Exp467 worktree closure differs from fixed Git bytes"
            ) from error
        if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise ValueError("Exp467 worktree closure differs from fixed Git bytes")
        if sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Exp467 worktree closure differs from fixed Git bytes")
    return MappingProxyType(dict(sorted(fixed.items())))


def _fixed_launcher_antigravity(repository: Path) -> ModuleType:
    source = _git_bytes_antigravity(repository, _EXP467_LAUNCHER_PATH_ANTIGRAVITY)
    module = ModuleType("_exp471_fixed_exp467_launcher_antigravity")
    module.__file__ = str(repository / _EXP467_LAUNCHER_PATH_ANTIGRAVITY)
    module.__package__ = ""
    exec(
        compile(
            source,
            f"{EXP467_SOURCE_COMMIT_ANTIGRAVITY}:{_EXP467_LAUNCHER_PATH_ANTIGRAVITY}",
            "exec",
        ),
        module.__dict__,
    )
    return module


def _closed_launcher_antigravity(repository: Path) -> ModuleType:
    repository = Path(repository).resolve()
    launcher = _fixed_launcher_antigravity(repository)
    path_reader = getattr(
        launcher, "runtime_dependency_relative_paths_antigravity", None
    )
    if not callable(path_reader):
        raise ValueError("Exp467 fixed launcher has no runtime closure")
    paths = tuple(path_reader())
    if (
        not paths
        or len(paths) != len(set(paths))
        or any(not _safe_relative_path_antigravity(path) for path in paths)
    ):
        raise ValueError("Exp467 fixed launcher runtime closure is invalid")
    if _EXP467_LAUNCHER_PATH_ANTIGRAVITY not in paths:
        paths = tuple(sorted((*paths, _EXP467_LAUNCHER_PATH_ANTIGRAVITY)))
    _require_matching_worktree_closure_antigravity(repository, paths)
    return launcher


def _require_fixed_context_antigravity(context: object) -> object:
    if getattr(context, "sha256", None) != EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY:
        raise ValueError("Exp467 reconstructed source context differs")
    return context


def reconstruct_exp467_source_context_antigravity(repository: Path) -> object:
    """Rebuild the fixed Exp467 full context only after its Git closure matches."""

    launcher = _closed_launcher_antigravity(Path(repository))
    full_inputs = getattr(launcher, "_full_inputs_antigravity", None)
    if not callable(full_inputs):
        raise ValueError("Exp467 fixed launcher cannot reconstruct full inputs")
    context, _catalog = full_inputs()
    return _require_fixed_context_antigravity(context)


def _import_exp467_verifier_antigravity(repository: Path) -> ModuleType:
    """Import the unchanged verifier only after the historical byte gate."""

    previous = sys.modules.pop(_EXP467_VERIFIER_MODULE_ANTIGRAVITY, None)
    repository_text = str(repository)
    sys.path.insert(0, repository_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(_EXP467_VERIFIER_MODULE_ANTIGRAVITY)
    except Exception:
        if previous is not None:
            sys.modules[_EXP467_VERIFIER_MODULE_ANTIGRAVITY] = previous
        raise
    finally:
        try:
            sys.path.remove(repository_text)
        except ValueError:
            pass
    expected_path = (
        repository
        / "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py"
    ).resolve()
    module_path = Path(getattr(module, "__file__", "")).resolve()
    if module_path != expected_path:
        sys.modules.pop(_EXP467_VERIFIER_MODULE_ANTIGRAVITY, None)
        if previous is not None:
            sys.modules[_EXP467_VERIFIER_MODULE_ANTIGRAVITY] = previous
        raise ValueError("Exp467 verifier did not load from the byte-gated repository")
    if previous is not None and previous is not module:
        # The newly imported, byte-gated module is authoritative for this session.
        # The prior object remains alive for existing callers but is not restored.
        pass
    return module


def _source_file_map_antigravity(source_root: Path) -> Mapping[str, str]:
    source_root = Path(source_root)
    try:
        root_details = source_root.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp467 source root is missing") from error
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise ValueError("Exp467 source root is unsafe")
    resolved_root = source_root.resolve()
    result: dict[str, str] = {}
    for directory, names, files in os.walk(resolved_root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            details = (directory_path / name).lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise ValueError("Exp467 source tree contains an unsafe directory")
        for name in files:
            path = directory_path / name
            details = path.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise ValueError("Exp467 source tree contains an unsafe file")
            relative = path.relative_to(resolved_root).as_posix()
            if not _safe_relative_path_antigravity(relative):
                raise ValueError("Exp467 source tree path escapes its root")
            digest = sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            result[relative] = digest.hexdigest()
    if not result:
        raise ValueError("Exp467 source tree contains no regular files")
    return MappingProxyType(dict(sorted(result.items())))


def _authority_record_antigravity(
    authority: Exp471SourceAuthorityAntigravity, *, include_digest: bool
) -> dict[str, object]:
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


def _verify_exp467_source_components_antigravity(
    repository: Path, source_root: Path
) -> tuple[Exp471SourceAuthorityAntigravity, object, object, object, object, ModuleType]:
    repository = Path(repository).resolve()
    source_root = Path(source_root)
    launcher = _closed_launcher_antigravity(repository)
    verify_migration = getattr(launcher, "_verify_migration_antigravity", None)
    if not callable(verify_migration):
        raise ValueError("Exp467 fixed launcher cannot verify migration")
    context, catalog, migration_manifest = verify_migration()
    _require_fixed_context_antigravity(context)
    try:
        if source_root.resolve(strict=True) != Path(context.run_root).resolve(strict=True):
            raise ValueError("Exp467 supplied source root differs from its fixed context")
    except FileNotFoundError as error:
        raise ValueError("Exp467 supplied source root is missing") from error

    source_files_before = _source_file_map_antigravity(source_root)

    verifier = _import_exp467_verifier_antigravity(repository)
    verify_completion = getattr(verifier, "verify_exp467_completion_antigravity", None)
    reopen_manifests = getattr(
        verifier, "read_persisted_exp467_condition_manifests_antigravity", None
    )
    summarize = getattr(verifier, "provenance_summary_antigravity", None)
    if not all(callable(value) for value in (verify_completion, reopen_manifests, summarize)):
        raise ValueError("Exp467 verifier API is incomplete")
    verification = verify_completion(
        source_root,
        context=context,
        catalog=catalog,
        migration_manifest=migration_manifest,
    )
    opened = reopen_manifests(source_root, verification)
    summary = summarize(verification)
    if (
        summary.get("total_successes") != EXP467_TOTAL_SUCCESSES_ANTIGRAVITY
        or summary.get("imported_total") != EXP467_IMPORTED_TOTAL_ANTIGRAVITY
        or summary.get("native_total") != EXP467_NATIVE_TOTAL_ANTIGRAVITY
    ):
        raise ValueError("Exp467 verified origin population differs")
    if (
        getattr(verification, "inventory_sha256", None) is None
        or getattr(verification, "catalog_sha256", None)
        != getattr(context, "catalog_sha256", None)
        or getattr(verification, "database_sha256", None)
        != getattr(context, "database_sha256", None)
    ):
        raise ValueError("Exp467 verification identity differs from its context")
    condition_digests: dict[str, str] = {}
    if not isinstance(opened, Mapping) or tuple(opened) != tuple(CONDITION_ORDER_ANTIGRAVITY):
        raise ValueError("Exp467 persisted condition manifest inventory differs")
    for condition in CONDITION_ORDER_ANTIGRAVITY:
        manifest = opened[condition]
        digest = manifest.get("manifest_sha256") if isinstance(manifest, Mapping) else None
        if not _digest_antigravity(digest):
            raise ValueError("Exp467 persisted condition manifest digest is invalid")
        condition_digests[condition] = digest

    source_files = _source_file_map_antigravity(source_root)
    if source_files != source_files_before:
        raise ValueError("Exp467 source tree changed during verification")
    source_tree_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(source_files)
    ).hexdigest()
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
    authority_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(
            _authority_record_antigravity(bare, include_digest=False)
        )
    ).hexdigest()
    authority = Exp471SourceAuthorityAntigravity(
        **_authority_record_antigravity(bare, include_digest=False),
        authority_sha256=authority_sha256,
    )
    return authority, verification, context, catalog, migration_manifest, verifier


def verify_exp467_source_antigravity(
    repository: Path, source_root: Path
) -> Exp471SourceAuthorityAntigravity:
    """Reverify every fixed Exp467 source input and return its immutable digest record."""

    return _verify_exp467_source_components_antigravity(repository, source_root)[0]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class Exp471SourceSessionAntigravity:
    """Issued in-memory capability binding Exp471 to one verified Exp467 authority."""

    authority: Exp471SourceAuthorityAntigravity
    verification: object
    context: object
    catalog: object
    migration_manifest: object
    verifier: ModuleType


def _make_source_session_api_antigravity():
    issued: WeakSet[object] = WeakSet()

    def is_issued(value: object) -> bool:
        try:
            return value in issued
        except TypeError:
            return False

    def open_session(
        repository: Path, source_root: Path
    ) -> Exp471SourceSessionAntigravity:
        authority, verification, context, catalog, migration, verifier = (
            _verify_exp467_source_components_antigravity(repository, source_root)
        )
        session = Exp471SourceSessionAntigravity(
            authority, verification, context, catalog, migration, verifier
        )
        issued.add(session)
        return session

    return open_session, is_issued


(
    open_exp467_source_session_antigravity,
    _source_session_is_issued_antigravity,
) = _make_source_session_api_antigravity()
del _make_source_session_api_antigravity


def _canonical_authority_bytes_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
) -> bytes:
    expected_digest = sha256(
        canonical_json_bytes_exp471_antigravity(
            _authority_record_antigravity(authority, include_digest=False)
        )
    ).hexdigest()
    if authority.authority_sha256 != expected_digest:
        raise ValueError("Exp471 source authority digest differs")
    return canonical_json_bytes_exp471_antigravity(
        _authority_record_antigravity(authority, include_digest=True)
    )


def publish_source_authority_antigravity(
    root: Path, authority: Exp471SourceAuthorityAntigravity
) -> Path:
    """Publish canonical authority bytes once, reusing only an exact existing leaf."""

    if not isinstance(authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 source authority has the wrong type")
    root = Path(root)
    if root.exists():
        details = root.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError("Exp471 authority root is unsafe")
    else:
        root.mkdir(parents=True, exist_ok=False)
    path = root / _SOURCE_AUTHORITY_NAME_ANTIGRAVITY
    payload = _canonical_authority_bytes_antigravity(authority)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        try:
            details = path.lstat()
        except FileNotFoundError as error:
            raise ValueError("Exp471 source authority publication raced") from error
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise ValueError("Exp471 existing source authority is unsafe")
        if path.read_bytes() != payload:
            raise ValueError("Exp471 existing source authority differs")
        return path
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A failed exclusive creation is deliberately left visible; callers may
        # never silently delete or replace authority evidence.
        raise
    return path


def _read_authority_artifact_antigravity(path: Path) -> Exp471SourceAuthorityAntigravity:
    try:
        details = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp471 source authority is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Exp471 source authority is unsafe")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp471 source authority JSON is malformed") from error
    required = {
        "source_commit",
        "source_context_sha256",
        "inventory_sha256",
        "catalog_sha256",
        "database_sha256",
        "source_file_sha256",
        "source_tree_sha256",
        "condition_manifest_sha256",
        "total_successes",
        "imported_total",
        "native_total",
        "authority_sha256",
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


def read_source_authority_antigravity(
    root: Path, repository: Path, source_root: Path
) -> Exp471SourceAuthorityAntigravity:
    """Reopen canonical bytes and require a fresh source/Git verification match."""

    persisted = _read_authority_artifact_antigravity(
        Path(root) / _SOURCE_AUTHORITY_NAME_ANTIGRAVITY
    )
    observed = verify_exp467_source_antigravity(repository, source_root)
    if _canonical_authority_bytes_antigravity(persisted) != _canonical_authority_bytes_antigravity(observed):
        raise ValueError("Exp471 persisted source authority differs from current source")
    return persisted


__all__ = [
    "Exp471SourceSessionAntigravity",
    "_source_session_is_issued_antigravity",
    "open_exp467_source_session_antigravity",
    "publish_source_authority_antigravity",
    "read_source_authority_antigravity",
    "reconstruct_exp467_source_context_antigravity",
    "verify_exp467_source_antigravity",
]
