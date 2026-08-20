"""Generate and verify the self-contained, factual Exp450 code-map triplet."""

from __future__ import annotations

import ast
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODEMAP_DIRECTORY = REPOSITORY_ROOT / "docs" / "codemap"
JSON_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.json"
HTML_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.html"
LOCK_PATH = CODEMAP_DIRECTORY / "codemap_exp450_antigravity.lock"
EXP450_DIRECTORY = REPOSITORY_ROOT / "experiments" / "exp450_resampling_image_ablation_antigravity"
_RETAINED_PATHS = (
    "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
    "paper/equipment_anomaly_detection/hard_route_paper.py",
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


def tracked_existing_paths_antigravity(root: Path) -> tuple[Path, ...]:
    """Return only factual source paths in the new namespace and retained closure."""
    exp450_paths = tuple(sorted(path for path in EXP450_DIRECTORY.rglob("*.py") if path.is_file()))
    retained_paths = tuple(root / relative for relative in _RETAINED_PATHS)
    generator_path = root / "scratch" / "generate_exp450_codemap_antigravity.py"
    path_set = {*exp450_paths, *(path for path in retained_paths if path.is_file())}
    if generator_path.is_file():
        path_set.add(generator_path)
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
    return tuple(sorted(imported))


def ast_callers_antigravity(paths: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    """Derive callers from imports across all existing Python files in the repository."""
    selected = {_module_name_antigravity(path): _relative_antigravity(path) for path in paths}
    callers: dict[str, set[str]] = defaultdict(set)
    for source in REPOSITORY_ROOT.rglob("*.py"):
        if any(part in {".git", "__pycache__"} for part in source.parts):
            continue
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
            }
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
