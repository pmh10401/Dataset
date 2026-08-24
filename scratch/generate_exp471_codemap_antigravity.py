"""Generate and verify the factual, synchronized Exp471 code-map triplet."""

from __future__ import annotations

import ast
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import subprocess
import sys
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

VERSION_ANTIGRAVITY = "exp471"
PACKAGE_ANTIGRAVITY = "experiments/exp471_exp467_evaluation_recovery_antigravity"
CLI_ANTIGRAVITY = "scratch/run_exp471_exp467_evaluation_recovery_antigravity.py"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp471_antigravity.html",
    "docs/codemap/codemap_exp471_antigravity.json",
    "docs/codemap/codemap_exp471_antigravity.lock",
)

DESIGN_PATH_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-24-exp471-exp467-evaluation-recovery-antigravity-design.md"
)
PLAN_PATH_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-24-exp471-exp467-evaluation-recovery-antigravity.md"
)

CODEMAP_DIRECTORY = REPOSITORY_ROOT / "docs" / "codemap"
JSON_PATH = CODEMAP_DIRECTORY / "codemap_exp471_antigravity.json"
HTML_PATH = CODEMAP_DIRECTORY / "codemap_exp471_antigravity.html"
LOCK_PATH = CODEMAP_DIRECTORY / "codemap_exp471_antigravity.lock"
_EXP471_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp471_exp467_evaluation_recovery_antigravity"
)
_SCRATCH_PATHS_ANTIGRAVITY = (
    "scratch/generate_exp471_codemap_antigravity.py",
)


def _relative_antigravity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _module_name_antigravity(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _repository_tracked_paths_antigravity(root: Path) -> frozenset[Path]:
    """Resolve index-tracked paths without admitting untracked worktree files."""
    resolved_root = root.resolve()
    try:
        result = subprocess.run(
            ("git", "-C", str(resolved_root), "ls-files", "--cached", "-z"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Exp471 code map cannot list repository-tracked paths") from exc
    paths: set[Path] = set()
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        try:
            relative = encoded_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Exp471 code map found a non-UTF-8 tracked path") from exc
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError("repository-tracked path escapes the Exp471 worktree")
        if candidate.is_file():
            paths.add(candidate)
    return frozenset(paths)


def tracked_existing_paths_antigravity(root: Path) -> tuple[Path, ...]:
    """Return the tracked Exp471 namespace and its scratch closure."""
    resolved_root = root.resolve()
    tracked_paths = _repository_tracked_paths_antigravity(resolved_root)
    namespace = resolved_root / _EXP471_NAMESPACE_ANTIGRAVITY
    exp471_paths = {
        path
        for path in tracked_paths
        if path.suffix == ".py" and path.is_relative_to(namespace)
    }
    required_paths = {
        resolved_root / relative
        for relative in _SCRATCH_PATHS_ANTIGRAVITY
    }
    missing = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in required_paths
        if path not in tracked_paths
    )
    if missing:
        raise ValueError(f"Exp471 code map is missing tracked closure paths: {missing}")
    paths = tuple(sorted(exp471_paths | required_paths))
    return paths


def _imported_modules_antigravity(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_name_antigravity(path).split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = list(package)
                prefix = prefix[: max(0, len(prefix) - node.level + 1)]
            else:
                prefix = []
            if node.module:
                prefix.extend(node.module.split("."))
            if prefix:
                imported.add(".".join(prefix))
                imported.update(
                    ".".join((*prefix, alias.name))
                    for alias in node.names
                    if alias.name != "*"
                )
    return tuple(sorted(imported))


def ast_callers_antigravity(paths: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    """Derive direct callers from imports across all index-tracked Python files."""
    selected = {_module_name_antigravity(path): _relative_antigravity(path) for path in paths}
    if len(selected) != len(paths):
        raise ValueError("Exp471 code map contains duplicate module names")
    callers: dict[str, set[str]] = defaultdict(set)
    sources = sorted(
        path
        for path in _repository_tracked_paths_antigravity(REPOSITORY_ROOT)
        if path.suffix == ".py"
    )
    for source in sources:
        try:
            imports = _imported_modules_antigravity(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        caller = _relative_antigravity(source)
        for imported in imports:
            for module, target in selected.items():
                if imported == module or imported.startswith(f"{module}."):
                    callers[target].add(caller)
    return {path: tuple(sorted(callers[path])) for path in sorted(selected.values())}


def _module_docstring_antigravity(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return ast.get_docstring(tree) or "No module docstring is present."


def _is_test_path_antigravity(relative_path: str) -> bool:
    return any(
        part == "tests" or part.startswith("tests_")
        for part in PurePosixPath(relative_path).parts
    )


def _canonical_mapping_bytes_antigravity(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def governance_fingerprint_evidence_antigravity(root: Path) -> dict[str, object]:
    """Verify the approved design and plan hashes as governance inputs."""
    resolved_root = root.resolve()
    design_path = resolved_root / DESIGN_PATH_ANTIGRAVITY
    plan_path = resolved_root / PLAN_PATH_ANTIGRAVITY
    if not design_path.is_file():
        raise ValueError(f"Exp471 governance design missing: {DESIGN_PATH_ANTIGRAVITY}")
    if not plan_path.is_file():
        raise ValueError(f"Exp471 governance plan missing: {PLAN_PATH_ANTIGRAVITY}")
    design_bytes = design_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    design_sha256 = sha256(design_bytes).hexdigest()
    plan_sha256 = sha256(plan_bytes).hexdigest()
    return {
        "design_path": DESIGN_PATH_ANTIGRAVITY,
        "design_sha256": design_sha256,
        "plan_path": PLAN_PATH_ANTIGRAVITY,
        "plan_sha256": plan_sha256,
        "frozen_exp467_source_commit": "1e616731a88da9bc0268823e1d2cef2a53c1054f",
        "frozen_exp467_source_context_sha256": "31341d8e14cf58a486cf7418276c432ac2dbda1af139699ff2afbd85969442af",
        "exp471_production_modules_exist": False,
    }


def build_codemap_payload_antigravity(
    paths: Sequence[Path], callers: dict[str, tuple[str, ...]]
) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    for path in paths:
        relative = _relative_antigravity(path)
        direct_callers = callers.get(relative, ())
        tests = tuple(caller for caller in direct_callers if _is_test_path_antigravity(caller))
        nodes.append(
            {
                "path": relative,
                "module": _module_name_antigravity(path),
                "callers": list(direct_callers),
                "effects": _module_docstring_antigravity(path),
                "tests": list(tests),
            }
        )
        edges.extend(
            {"caller": caller, "callee": relative, "kind": "ast_import"}
            for caller in direct_callers
        )
    return {
        "schema": "exp471-codemap.v1",
        "scope": {
            "experiment": "Exp471",
            "namespace": _EXP471_NAMESPACE_ANTIGRAVITY,
            "node_count": len(nodes),
            "production_node_count": sum(
                path.is_relative_to(REPOSITORY_ROOT / _EXP471_NAMESPACE_ANTIGRAVITY)
                and not _is_test_path_antigravity(_relative_antigravity(path))
                for path in paths
            ),
            "test_node_count": sum(
                path.is_relative_to(REPOSITORY_ROOT / _EXP471_NAMESPACE_ANTIGRAVITY)
                and _is_test_path_antigravity(_relative_antigravity(path))
                for path in paths
            ),
            "derived_from_index_tracked_paths_only": True,
            "external_closure_policy": "no_production_modules_yet",
            "governance_fingerprint_evidence": governance_fingerprint_evidence_antigravity(REPOSITORY_ROOT),
        },
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["caller"], edge["callee"])),
        "flows": [
            {
                "name": "exp471_source_verification_flow",
                "from": "fixed-commit Exp467 source authority",
                "through": "reconstructed Git bytes and context digest",
                "to": "verified 4,468 source envelopes",
            },
            {
                "name": "exp471_resumable_score_cache_flow",
                "from": "verified source authority",
                "through": "label-free raw/micro score shards with create-only publication",
                "to": "complete score-cache manifest",
            },
            {
                "name": "exp471_evaluation_flow",
                "from": "complete score-cache manifest",
                "through": "frozen Exp450 metrics with JSON null normalization",
                "to": "durable evaluation artifact",
            },
            {
                "name": "exp471_report_flow",
                "from": "durable evaluation artifact",
                "through": "atomic canonical JSON/CSV/figure publication",
                "to": "paper report manifest",
            },
        ],
        "unknowns": [],
    }


def module_fingerprints_antigravity(paths: Sequence[Path]) -> dict[str, str]:
    return {
        _relative_antigravity(path): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def render_triplet_antigravity(
    payload: dict[str, object], fingerprints: dict[str, str]
) -> dict[Path, bytes]:
    json_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    embedded = json_bytes.decode("utf-8").rstrip("\n")
    if "</script" in embedded.lower():
        raise ValueError("Exp471 code-map JSON cannot be embedded safely in HTML")
    html_bytes = (
        "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
        "<title>Exp471 code map</title></head>\n<body><h1>Exp471 code map</h1>"
        '<script id="codemap-data" type="application/json">'
        f"{embedded}</script></body>\n</html>\n"
    ).encode("utf-8")
    lock = {
        "schema": "exp471-codemap-lock.v1",
        "fingerprint_algorithm": "sha256",
        "artifact_sha256": {
            _relative_antigravity(HTML_PATH): sha256(html_bytes).hexdigest(),
            _relative_antigravity(JSON_PATH): sha256(json_bytes).hexdigest(),
        },
        "module_fingerprints": fingerprints,
        "scanned_scope": list(fingerprints),
        "governance_fingerprint_evidence": governance_fingerprint_evidence_antigravity(REPOSITORY_ROOT),
    }
    lock_bytes = (
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {JSON_PATH: json_bytes, HTML_PATH: html_bytes, LOCK_PATH: lock_bytes}


def require_existing_triplet_equal_antigravity(generated: dict[Path, bytes]) -> None:
    for path, expected in generated.items():
        if not path.is_file():
            raise ValueError(f"missing Exp471 code-map artifact: {_relative_antigravity(path)}")
        if path.read_bytes() != expected:
            raise ValueError(f"stale Exp471 code-map artifact: {_relative_antigravity(path)}")
    payload = json.loads(JSON_PATH.read_bytes())
    html_text = HTML_PATH.read_text(encoding="utf-8")
    marker = '<script id="codemap-data" type="application/json">'
    if marker not in html_text or "</script>" not in html_text:
        raise ValueError("Exp471 HTML does not contain the code-map JSON payload")
    embedded = html_text.split(marker, 1)[1].split("</script>", 1)[0]
    if json.loads(embedded) != payload:
        raise ValueError("Exp471 HTML embedded JSON differs from standalone JSON")


def _write_file_fsynced_antigravity(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_triplet_together_antigravity(generated: dict[Path, bytes]) -> None:
    """Stage all bytes first and publish the lock last as the triplet commit marker."""
    CODEMAP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(8)
    staged = {
        path: path.with_name(f".{path.name}.{nonce}.tmp")
        for path in (JSON_PATH, HTML_PATH, LOCK_PATH)
    }
    created: list[Path] = []
    try:
        for path in (JSON_PATH, HTML_PATH, LOCK_PATH):
            _write_file_fsynced_antigravity(staged[path], generated[path])
            created.append(staged[path])
        for path in (JSON_PATH, HTML_PATH, LOCK_PATH):
            os.replace(staged[path], path)
        directory_fd = os.open(CODEMAP_DIRECTORY, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for temporary in created:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments not in ((), ("--check",)):
        raise ValueError("usage: generate_exp471_codemap_antigravity.py [--check]")
    paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
    payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
    generated = render_triplet_antigravity(payload, module_fingerprints_antigravity(paths))
    if arguments == ("--check",):
        require_existing_triplet_equal_antigravity(generated)
    else:
        write_triplet_together_antigravity(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())