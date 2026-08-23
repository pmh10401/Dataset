"""Generate and verify the factual, synchronized Exp451 code-map triplet."""

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
CODEMAP_DIRECTORY = REPOSITORY_ROOT / "docs" / "codemap"
JSON_PATH = CODEMAP_DIRECTORY / "codemap_exp451_antigravity.json"
HTML_PATH = CODEMAP_DIRECTORY / "codemap_exp451_antigravity.html"
LOCK_PATH = CODEMAP_DIRECTORY / "codemap_exp451_antigravity.lock"
FROZEN_EXP450_LOCK_PATH_ANTIGRAVITY = (
    "docs/codemap/codemap_exp450_antigravity.lock"
)
FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY = (
    "026ee148b0b9db0e7be5cdfe7e311815896eb4eb8b9a468e7fb5d2a0decd0c3a"
)
_EXP451_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp451_verified_accelerated_resume_antigravity"
)
_RETAINED_PATHS_ANTIGRAVITY = (
    "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/artifacts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/evaluation_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/reporting_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py",
    "paper/equipment_anomaly_detection/calibration_paper.py",
    "paper/equipment_anomaly_detection/catalog_paper.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/hard_route_paper.py",
    "paper/equipment_anomaly_detection/routing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/knn_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
)
_SCRATCH_PATHS_ANTIGRAVITY = (
    "scratch/generate_exp451_codemap_antigravity.py",
    "scratch/run_exp451_verified_accelerated_resume_antigravity.py",
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
        raise ValueError("Exp451 code map cannot list repository-tracked paths") from exc
    paths: set[Path] = set()
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        try:
            relative = encoded_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Exp451 code map found a non-UTF-8 tracked path") from exc
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError("repository-tracked path escapes the Exp451 worktree")
        if candidate.is_file():
            paths.add(candidate)
    return frozenset(paths)


def tracked_existing_paths_antigravity(root: Path) -> tuple[Path, ...]:
    """Return the tracked Exp451 namespace and its exact direct source closure."""
    resolved_root = root.resolve()
    tracked_paths = _repository_tracked_paths_antigravity(resolved_root)
    namespace = resolved_root / _EXP451_NAMESPACE_ANTIGRAVITY
    exp451_paths = {
        path
        for path in tracked_paths
        if path.suffix == ".py" and path.is_relative_to(namespace)
    }
    required_paths = {
        resolved_root / relative
        for relative in (*_RETAINED_PATHS_ANTIGRAVITY, *_SCRATCH_PATHS_ANTIGRAVITY)
    }
    missing = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in required_paths
        if path not in tracked_paths
    )
    if missing:
        raise ValueError(f"Exp451 code map is missing tracked closure paths: {missing}")
    paths = tuple(sorted(exp451_paths | required_paths))
    if not exp451_paths:
        raise ValueError("Exp451 code map has no tracked namespace source paths")
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
        raise ValueError("Exp451 code map contains duplicate module names")
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


def frozen_exp450_fingerprint_evidence_antigravity(root: Path) -> dict[str, object]:
    """Verify the frozen Exp450 lock and all 36 recorded source fingerprints."""
    resolved_root = root.resolve()
    lock_path = resolved_root / FROZEN_EXP450_LOCK_PATH_ANTIGRAVITY
    lock_bytes = lock_path.read_bytes()
    lock_sha256 = sha256(lock_bytes).hexdigest()
    if lock_sha256 != FROZEN_EXP450_LOCK_SHA256_ANTIGRAVITY:
        raise ValueError("frozen Exp450 code-map lock bytes changed")
    try:
        lock = json.loads(lock_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen Exp450 code-map lock is not canonical JSON") from exc
    if lock.get("schema") != "exp450-codemap-lock.v1":
        raise ValueError("frozen Exp450 code-map lock schema changed")
    if lock.get("fingerprint_algorithm") != "sha256":
        raise ValueError("frozen Exp450 code-map fingerprint algorithm changed")
    fingerprints = lock.get("module_fingerprints")
    scanned_scope = lock.get("scanned_scope")
    if not isinstance(fingerprints, dict) or len(fingerprints) != 36:
        raise ValueError("frozen Exp450 code-map lock must contain exactly 36 fingerprints")
    if scanned_scope != sorted(fingerprints):
        raise ValueError("frozen Exp450 scanned scope differs from its fingerprint keys")
    for relative, expected_sha256 in sorted(fingerprints.items()):
        if not isinstance(relative, str) or not isinstance(expected_sha256, str):
            raise ValueError("frozen Exp450 code-map fingerprint entry is malformed")
        candidate = resolved_root / relative
        resolved_candidate = candidate.resolve()
        if not resolved_candidate.is_relative_to(resolved_root) or candidate.is_symlink():
            raise ValueError("frozen Exp450 code-map path escapes or aliases the worktree")
        if not candidate.is_file():
            raise ValueError(f"frozen Exp450 code-map source is missing: {relative}")
        if sha256(candidate.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"frozen Exp450 code-map source fingerprint changed: {relative}")
    fingerprints_sha256 = sha256(
        _canonical_mapping_bytes_antigravity(fingerprints)
    ).hexdigest()
    return {
        "lock_path": FROZEN_EXP450_LOCK_PATH_ANTIGRAVITY,
        "lock_sha256": lock_sha256,
        "module_fingerprint_count": 36,
        "module_fingerprints_sha256": fingerprints_sha256,
        "caller_edge_policy": "frozen_triplet_not_regenerated_after_exp451_callers",
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
        "schema": "exp451-codemap.v1",
        "scope": {
            "experiment": "Exp451",
            "namespace": _EXP451_NAMESPACE_ANTIGRAVITY,
            "node_count": len(nodes),
            "production_node_count": 12,
            "test_node_count": 7,
            "external_direct_node_count": len(_RETAINED_PATHS_ANTIGRAVITY),
            "derived_from_index_tracked_paths_only": True,
            "external_closure_policy": "direct_imports_used_by_exp451_production_or_cli",
            "frozen_exp450_fingerprint_evidence": (
                frozen_exp450_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
            ),
        },
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["caller"], edge["callee"])),
        "flows": [
            {
                "name": "exp451_verified_migration_flow",
                "from": "fixed-commit Exp450 checkpoint envelopes",
                "through": "historical source-context reconstruction and create-only migration",
                "to": "4,234 verified imported Exp451 checkpoints",
            },
            {
                "name": "exp451_accelerated_resume_flow",
                "from": "verified migration inventory plus values-only raw datasets",
                "through": "deterministic missing-coordinate schedule and MPS-to-native-CPU pooling fallback",
                "to": "4,468 durable Exp451 checkpoints",
            },
            {
                "name": "exp451_label_gated_publication_flow",
                "from": "four reopened frozen condition manifests",
                "through": "checkpoint rehash, score reconstruction, frozen Exp450 metrics and report wrapper",
                "to": "create-only evaluation and report publications",
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
        raise ValueError("Exp451 code-map JSON cannot be embedded safely in HTML")
    html_bytes = (
        "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
        "<title>Exp451 code map</title></head>\n<body><h1>Exp451 code map</h1>"
        '<script id="codemap-data" type="application/json">'
        f"{embedded}</script></body>\n</html>\n"
    ).encode("utf-8")
    lock = {
        "schema": "exp451-codemap-lock.v1",
        "fingerprint_algorithm": "sha256",
        "artifact_sha256": {
            _relative_antigravity(HTML_PATH): sha256(html_bytes).hexdigest(),
            _relative_antigravity(JSON_PATH): sha256(json_bytes).hexdigest(),
        },
        "module_fingerprints": fingerprints,
        "scanned_scope": list(fingerprints),
        "frozen_exp450_fingerprint_evidence": (
            frozen_exp450_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
        ),
    }
    lock_bytes = (
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {JSON_PATH: json_bytes, HTML_PATH: html_bytes, LOCK_PATH: lock_bytes}


def require_existing_triplet_equal_antigravity(generated: dict[Path, bytes]) -> None:
    for path, expected in generated.items():
        if not path.is_file():
            raise ValueError(f"missing Exp451 code-map artifact: {_relative_antigravity(path)}")
        if path.read_bytes() != expected:
            raise ValueError(f"stale Exp451 code-map artifact: {_relative_antigravity(path)}")
    payload = json.loads(JSON_PATH.read_bytes())
    html_text = HTML_PATH.read_text(encoding="utf-8")
    marker = '<script id="codemap-data" type="application/json">'
    if marker not in html_text or "</script>" not in html_text:
        raise ValueError("Exp451 HTML does not contain the code-map JSON payload")
    embedded = html_text.split(marker, 1)[1].split("</script>", 1)[0]
    if json.loads(embedded) != payload:
        raise ValueError("Exp451 HTML embedded JSON differs from standalone JSON")


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
        raise ValueError("usage: generate_exp451_codemap_antigravity.py [--check]")
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
