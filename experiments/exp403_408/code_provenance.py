"""Deterministic AST closure for repository-local execution code."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from collections.abc import Iterable


CLOSURE_METHOD = "ast_recursive_repository_local_import_closure/v1"


def local_code_closure(repository: Path, roots: Iterable[Path]) -> dict[str, object]:
    """Hash every repository-local Python module reachable from ``roots``.

    The resolver follows static ``import`` and ``from ... import ...`` edges,
    including relative imports. Standard-library and site-package imports are
    excluded because only paths under ``repository`` are eligible.
    """
    root = Path(repository).resolve()
    root_paths = tuple(Path(path) for path in roots)
    pending = sorted({_local_path(root, path) for path in root_paths}, key=str)
    pending = [path for path in pending if path is not None]
    if not pending:
        raise ValueError("code provenance requires at least one local root")
    visited: set[Path] = set()
    while pending:
        path = pending.pop(0)
        if path in visited:
            continue
        visited.add(path)
        for dependency in _local_imports(root, path):
            if dependency not in visited:
                pending.append(dependency)
        pending.sort(key=str)
    files = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(visited, key=str)
    }
    canonical = json.dumps(files, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    return {
        "method": CLOSURE_METHOD,
        "roots": [str(path) for path in sorted({_local_path(root, path) for path in root_paths if _local_path(root, path) is not None}, key=str)],
        "files": files,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _local_imports(repository: Path, path: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot parse local provenance module: {path}") from error
    package = _module_name(repository, path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    candidates: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_module(repository, alias.name, candidates)
        elif isinstance(node, ast.ImportFrom):
            base = _import_from_base(package, node.level, node.module)
            if base:
                _add_module(repository, base, candidates)
            for alias in node.names:
                if alias.name != "*" and base:
                    _add_module(repository, f"{base}.{alias.name}", candidates)
                elif alias.name != "*" and not base:
                    _add_module(repository, alias.name, candidates)
    return tuple(sorted(candidates, key=str))


def _import_from_base(package: list[str], level: int, module: str | None) -> str:
    if level:
        keep = len(package) - (level - 1)
        if keep < 0:
            return ""
        parts = package[:keep]
        if module:
            parts.extend(module.split("."))
        return ".".join(parts)
    return module or ""


def _add_module(repository: Path, module: str, paths: set[Path]) -> None:
    parts = module.split(".")
    for size in range(1, len(parts) + 1):
        path = _module_path(repository, ".".join(parts[:size]))
        if path is not None:
            paths.add(path)


def _module_path(repository: Path, module: str) -> Path | None:
    if not module:
        return None
    base = repository.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        local = _local_path(repository, candidate)
        if local is not None:
            return local
    return None


def _module_name(repository: Path, path: Path) -> str:
    relative = path.resolve().relative_to(repository)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def _local_path(repository: Path, path: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(repository)
    except ValueError:
        return None
    return resolved if resolved.is_file() and resolved.suffix == ".py" else None
