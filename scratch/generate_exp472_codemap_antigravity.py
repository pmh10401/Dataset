"""Generate and verify the factual, synchronized Exp472 code-map triplet."""

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
VERSION_ANTIGRAVITY = "exp472"
TRIPLET_ANTIGRAVITY = tuple(
    REPOSITORY_ROOT / "docs" / "codemap" / f"codemap_exp472_antigravity.{suffix}"
    for suffix in ("html", "json", "lock")
)
HTML_PATH, JSON_PATH, LOCK_PATH = TRIPLET_ANTIGRAVITY
CODEMAP_DIRECTORY = REPOSITORY_ROOT / "docs" / "codemap"

_EXP472_NAMESPACE_ANTIGRAVITY = (
    "experiments/exp472_makima_upsampling_ablation_antigravity"
)
_DESIGN_PATH_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-28-exp472-makima-upsampling-ablation-antigravity-design.md"
)
_PLAN_PATH_ANTIGRAVITY = (
    "docs/superpowers/plans/"
    "2026-08-28-exp472-makima-upsampling-ablation-antigravity.md"
)
_RETAINED_PATHS_ANTIGRAVITY = (
    _DESIGN_PATH_ANTIGRAVITY,
    _PLAN_PATH_ANTIGRAVITY,
    "experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/imaging_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/runner_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/cache_artifacts_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/score_cache_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/evaluation_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/reporting_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/label_access_paper.py",
)
_GENERATOR_PATH_ANTIGRAVITY = "scratch/generate_exp472_codemap_antigravity.py"


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
        raise ValueError("Exp472 code map cannot list repository-tracked paths") from exc
    paths: set[Path] = set()
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        try:
            relative = encoded_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Exp472 code map found a non-UTF-8 tracked path") from exc
        candidate = (resolved_root / relative).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError("repository-tracked path escapes the Exp472 worktree")
        if candidate.is_file():
            paths.add(candidate)
    return frozenset(paths)


def tracked_existing_paths_antigravity(root: Path) -> tuple[Path, ...]:
    """Return every tracked Exp472 file and its frozen scientific closure."""
    resolved_root = root.resolve()
    tracked_paths = _repository_tracked_paths_antigravity(resolved_root)
    exp472_paths = {
        path
        for path in tracked_paths
        if "exp472" in path.relative_to(resolved_root).as_posix().casefold()
        and path not in TRIPLET_ANTIGRAVITY
    }
    required_paths = {
        resolved_root / relative
        for relative in (*_RETAINED_PATHS_ANTIGRAVITY, _GENERATOR_PATH_ANTIGRAVITY)
    }
    missing = sorted(
        path.relative_to(resolved_root).as_posix()
        for path in required_paths
        if path not in tracked_paths
    )
    if missing:
        raise ValueError(f"Exp472 code map is missing tracked closure paths: {missing}")
    return tuple(sorted(exp472_paths | required_paths))


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
    selected = {
        _module_name_antigravity(path): _relative_antigravity(path)
        for path in paths
        if path.suffix == ".py"
    }
    if len(selected) != sum(path.suffix == ".py" for path in paths):
        raise ValueError("Exp472 code map contains duplicate module names")
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
    return {
        _relative_antigravity(path): tuple(
            sorted(callers[_relative_antigravity(path)])
        )
        for path in sorted(paths)
    }


def _node_name_antigravity(path: Path) -> str:
    if path.suffix == ".py":
        return _module_name_antigravity(path)
    return _relative_antigravity(path)


def _node_effect_antigravity(path: Path) -> str:
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return ast.get_docstring(tree) or "No module docstring is present."
    if path.suffix.casefold() in {".csv", ".html", ".json", ".md", ".txt"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return f"Governance document: {line[2:]}"
    return f"Tracked Exp472 artifact: {_relative_antigravity(path)}"


def _is_test_path_antigravity(relative_path: str) -> bool:
    return any(
        part == "tests" or part.startswith("tests_")
        for part in PurePosixPath(relative_path).parts
    )


def governance_fingerprint_evidence_antigravity(root: Path) -> dict[str, str]:
    """Bind the approved Exp472 design and implementation plan."""
    resolved_root = root.resolve()
    return {
        "design_path": _DESIGN_PATH_ANTIGRAVITY,
        "design_sha256": sha256(
            (resolved_root / _DESIGN_PATH_ANTIGRAVITY).read_bytes()
        ).hexdigest(),
        "plan_path": _PLAN_PATH_ANTIGRAVITY,
        "plan_sha256": sha256(
            (resolved_root / _PLAN_PATH_ANTIGRAVITY).read_bytes()
        ).hexdigest(),
    }


def build_codemap_payload_antigravity(
    paths: Sequence[Path], callers: dict[str, tuple[str, ...]]
) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    for path in paths:
        relative = _relative_antigravity(path)
        direct_callers = callers.get(relative, ())
        tests = tuple(
            caller for caller in direct_callers if _is_test_path_antigravity(caller)
        )
        nodes.append(
            {
                "path": relative,
                "module": _node_name_antigravity(path),
                "callers": list(direct_callers),
                "effects": _node_effect_antigravity(path),
                "tests": list(tests),
            }
        )
        edges.extend(
            {"caller": caller, "callee": relative, "kind": "ast_import"}
            for caller in direct_callers
        )
    namespace_root = REPOSITORY_ROOT / _EXP472_NAMESPACE_ANTIGRAVITY
    return {
        "schema": f"{VERSION_ANTIGRAVITY}-codemap.v1",
        "scope": {
            "experiment": VERSION_ANTIGRAVITY.replace("exp", "Exp", 1),
            "version": VERSION_ANTIGRAVITY,
            "namespace": _EXP472_NAMESPACE_ANTIGRAVITY,
            "node_count": len(nodes),
            "production_node_count": sum(
                path.suffix == ".py"
                and path.is_relative_to(namespace_root)
                and not _is_test_path_antigravity(_relative_antigravity(path))
                for path in paths
            ),
            "test_node_count": sum(
                path.suffix == ".py"
                and path.is_relative_to(namespace_root)
                and _is_test_path_antigravity(_relative_antigravity(path))
                for path in paths
            ),
            "retained_boundary_node_count": sum(
                _relative_antigravity(path) in _RETAINED_PATHS_ANTIGRAVITY
                for path in paths
            ),
            "derived_from_index_tracked_paths_only": True,
            "external_closure_policy": (
                "approved_design_plus_exp450_scoring_exp471_reuse_and_label_boundaries"
            ),
            "governance_fingerprint_evidence": (
                governance_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
            ),
        },
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["caller"], edge["callee"])),
        "flows": [
            {
                "name": "exp472_condition_e_score_flow",
                "from": "values-only frozen dataset population",
                "through": (
                    "Exp450 scoring and image boundaries with exact SciPy "
                    "polyphase/Makima alignment"
                ),
                "to": "one finite float64 condition-E score shard per dataset",
            },
            {
                "name": "exp472_exp471_reuse_flow",
                "from": "immutable Exp471 source and score-cache authorities",
                "through": "coordinate- and digest-bound Exp471 evaluation reuse",
                "to": "paired E-minus-D evaluation and contextual report inputs",
            },
            {
                "name": "exp472_label_gate_flow",
                "from": "complete verified E and Exp471 authorities",
                "through": "lazy paper TEST-label boundary",
                "to": "evaluation and hash-bound deterministic report",
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
    embedded = json_bytes.decode("utf-8")
    if "</script" in embedded.lower():
        raise ValueError("Exp472 code-map JSON cannot be embedded safely in HTML")
    html_bytes = (
        "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
        "<title>Exp472 code map</title></head>\n<body><h1>Exp472 code map</h1>"
        '<script id="codemap-data" type="application/json">'
        f"{embedded}</script></body>\n</html>\n"
    ).encode("utf-8")
    lock = {
        "schema": f"{VERSION_ANTIGRAVITY}-codemap-lock.v1",
        "fingerprint_algorithm": "sha256",
        "artifact_sha256": {
            _relative_antigravity(HTML_PATH): sha256(html_bytes).hexdigest(),
            _relative_antigravity(JSON_PATH): sha256(json_bytes).hexdigest(),
        },
        "module_fingerprints": fingerprints,
        "scanned_scope": list(fingerprints),
        "governance_fingerprint_evidence": (
            governance_fingerprint_evidence_antigravity(REPOSITORY_ROOT)
        ),
    }
    lock_bytes = (
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return {HTML_PATH: html_bytes, JSON_PATH: json_bytes, LOCK_PATH: lock_bytes}


def require_existing_triplet_equal_antigravity(generated: dict[Path, bytes]) -> None:
    for path, expected in generated.items():
        if not path.is_file():
            raise ValueError(f"missing Exp472 code-map artifact: {_relative_antigravity(path)}")
        if path.read_bytes() != expected:
            raise ValueError(f"stale Exp472 code-map artifact: {_relative_antigravity(path)}")
    payload = json.loads(JSON_PATH.read_bytes())
    html_text = HTML_PATH.read_text(encoding="utf-8")
    marker = '<script id="codemap-data" type="application/json">'
    if marker not in html_text or "</script>" not in html_text:
        raise ValueError("Exp472 HTML does not contain the code-map JSON payload")
    embedded = html_text.split(marker, 1)[1].split("</script>", 1)[0]
    if json.loads(embedded) != payload:
        raise ValueError("Exp472 HTML embedded JSON differs from standalone JSON")


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
        for path in (HTML_PATH, JSON_PATH, LOCK_PATH)
    }
    created: list[Path] = []
    try:
        for path in (HTML_PATH, JSON_PATH, LOCK_PATH):
            _write_file_fsynced_antigravity(staged[path], generated[path])
            created.append(staged[path])
        for path in (HTML_PATH, JSON_PATH, LOCK_PATH):
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
        raise ValueError("usage: generate_exp472_codemap_antigravity.py [--check]")
    paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
    payload = build_codemap_payload_antigravity(
        paths, ast_callers_antigravity(paths)
    )
    triplet = render_triplet_antigravity(
        payload, module_fingerprints_antigravity(paths)
    )
    if arguments == ("--check",):
        require_existing_triplet_equal_antigravity(triplet)
    else:
        write_triplet_together_antigravity(triplet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
