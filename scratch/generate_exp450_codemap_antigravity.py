"""Generate and verify the self-contained, factual Exp450 code-map triplet."""

from __future__ import annotations

import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODEMAP_DIRECTORY = REPOSITORY_ROOT / "docs" / "codemap"
JSON_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.json"
HTML_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.html"
LOCK_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.lock"
_RETAINED_PATHS = (
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
    "paper/equipment_anomaly_detection/hard_route_paper.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/routing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/validation_antigravity.py",
    "experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py",
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
        raise ValueError("Exp450 code map cannot list repository-tracked paths") from exc
    paths: set[Path] = set()
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        candidate = (resolved_root / encoded_path.decode("utf-8")).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError("repository-tracked path escapes the Exp450 worktree")
        if candidate.is_file():
            paths.add(candidate)
    return frozenset(paths)


def tracked_existing_paths_antigravity(root: Path) -> tuple[Path, ...]:
    """Return only tracked source paths in the new namespace and retained closure."""
    resolved_root = root.resolve()
    tracked_paths = _repository_tracked_paths_antigravity(resolved_root)
    exp450_directory = resolved_root / "experiments" / "exp450_resampling_image_ablation_antigravity"
    exp450_paths = {
        path
        for path in tracked_paths
        if path.suffix == ".py" and path.is_relative_to(exp450_directory)
    }
    retained_paths = {resolved_root / relative for relative in _RETAINED_PATHS}
    generator_path = resolved_root / "scratch" / "generate_exp450_codemap_antigravity.py"
    cli_path = resolved_root / "scratch" / "run_exp450_resampling_image_ablation_antigravity.py"
    path_set = exp450_paths | (retained_paths & tracked_paths)
    if generator_path in tracked_paths:
        path_set.add(generator_path)
    if cli_path in tracked_paths:
        path_set.add(cli_path)
    paths = tuple(sorted(path_set))
    if not paths:
        raise ValueError("Exp450 code map has no existing source paths")
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
    """Derive callers from imports across all index-tracked Python files."""
    selected = {_module_name_antigravity(path): _relative_antigravity(path) for path in paths}
    callers: dict[str, set[str]] = defaultdict(set)
    sources = sorted(
        path for path in _repository_tracked_paths_antigravity(REPOSITORY_ROOT) if path.suffix == ".py"
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
    """Recognize conventional test directories, including retained ``tests_paper``."""
    return any(
        part == "tests" or part.startswith("tests_")
        for part in PurePosixPath(relative_path).parts
    )


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
        edges.extend({"caller": caller, "callee": relative, "kind": "ast_import"} for caller in direct_callers)
    return {
        "schema": "exp450-codemap.v1",
        "scope": {
            "experiment": "Exp450",
            "namespace": "experiments/exp450_resampling_image_ablation_antigravity",
            "retained_exp449_immutable": True,
            "derived_from_existing_paths_only": True,
        },
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["caller"], edge["callee"])),
        "flows": [
            {
                "name": "exp450_label_free_score_flow",
                "from": "values-only raw datasets",
                "through": "conditions A,B,C,D and retained Hard-source/route stages",
                "to": "frozen condition score manifests before evaluation",
            },
            {
                "name": "exp450_cli_lifecycle_flow",
                "from": "readiness --mode smoke|full",
                "through": "smoke/run durable progress, verify v2 manifests, evaluate label gate",
                "to": "canonical evaluation artifact and self-contained report",
            },
        ],
        "unknowns": [],
    }


def module_fingerprints_antigravity(paths: Sequence[Path]) -> dict[str, str]:
    return {_relative_antigravity(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def render_triplet_antigravity(
    payload: dict[str, object], fingerprints: dict[str, str]
) -> dict[Path, bytes]:
    json_bytes = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    embedded = json_bytes.decode("utf-8").rstrip("\n")
    html_bytes = (
        "<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\"><title>Exp450 code map</title></head>\n"
        "<body><h1>Exp450 code map</h1><script id=\"codemap-data\" type=\"application/json\">"
        f"{embedded}</script></body>\n</html>\n"
    ).encode("utf-8")
    lock = {
        "schema": "exp450-codemap-lock.v1",
        "fingerprint_algorithm": "sha256",
        "module_fingerprints": fingerprints,
        "scanned_scope": list(fingerprints),
    }
    lock_bytes = (json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {JSON_PATH: json_bytes, HTML_PATH: html_bytes, LOCK_PATH: lock_bytes}


def require_existing_triplet_equal_antigravity(generated: dict[Path, bytes]) -> None:
    for path, expected in generated.items():
        if not path.is_file():
            raise ValueError(f"missing Exp450 code-map artifact: {_relative_antigravity(path)}")
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(f"stale Exp450 code-map artifact: {_relative_antigravity(path)}")
    payload = json.loads(generated[JSON_PATH])
    html_text = generated[HTML_PATH].decode("utf-8")
    marker = '<script id="codemap-data" type="application/json">'
    embedded = html_text.split(marker, 1)[1].split("</script>", 1)[0]
    if json.loads(embedded) != payload:
        raise ValueError("Exp450 HTML embedded JSON differs from standalone JSON")


def write_triplet_together_antigravity(generated: dict[Path, bytes]) -> None:
    CODEMAP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path in (JSON_PATH, HTML_PATH, LOCK_PATH):
        path.write_bytes(generated[path])


def main(argv: Sequence[str] | None = None) -> int:
    check = tuple(sys.argv[1:] if argv is None else argv) == ("--check",)
    paths = tracked_existing_paths_antigravity(REPOSITORY_ROOT)
    payload = build_codemap_payload_antigravity(paths, ast_callers_antigravity(paths))
    generated = render_triplet_antigravity(payload, module_fingerprints_antigravity(paths))
    if check:
        require_existing_triplet_equal_antigravity(generated)
    else:
        write_triplet_together_antigravity(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
