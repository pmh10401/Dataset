"""Generate and verify the synchronized Exp473 code-map triplet."""

from __future__ import annotations

import ast
from collections import defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path, PurePosixPath
import sys


REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
NAMESPACE_ANTIGRAVITY = "experiments/exp473_contemporary_pchip_control_antigravity"
JSON_PATH_ANTIGRAVITY = (
    REPOSITORY_ROOT_ANTIGRAVITY
    / "docs/codemap/codemap_exp473_antigravity.json"
)
HTML_PATH_ANTIGRAVITY = (
    REPOSITORY_ROOT_ANTIGRAVITY
    / "docs/codemap/codemap_exp473_antigravity.html"
)
LOCK_PATH_ANTIGRAVITY = (
    REPOSITORY_ROOT_ANTIGRAVITY
    / "docs/codemap/codemap_exp473_antigravity.lock"
)
GENERATOR_PATH_ANTIGRAVITY = "scratch/generate_exp473_codemap_antigravity.py"
RETAINED_PATHS_ANTIGRAVITY = (
    "scratch/run_exp473_contemporary_pchip_control_antigravity.py",
    "docs/EXP472_D_PRIME_FOLLOWUP_DESIGN_SOL_HIGH_antigravity_KO.md",
    "docs/EXP472_PAPER_DRAFT_SOL_HIGH_antigravity_KO.md",
    "docs/codemap/codemap_exp472_antigravity.lock",
    "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/config_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/scoring_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/source_authority_antigravity.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/routing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
)


def _relative_antigravity(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT_ANTIGRAVITY).as_posix()


def _module_antigravity(path: Path) -> str:
    relative = path.relative_to(REPOSITORY_ROOT_ANTIGRAVITY).with_suffix("")
    return ".".join(relative.parts)


def _selected_paths_antigravity() -> tuple[Path, ...]:
    namespace = REPOSITORY_ROOT_ANTIGRAVITY / NAMESPACE_ANTIGRAVITY
    paths = {
        path
        for path in namespace.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    paths.add(REPOSITORY_ROOT_ANTIGRAVITY / GENERATOR_PATH_ANTIGRAVITY)
    for relative in RETAINED_PATHS_ANTIGRAVITY:
        path = REPOSITORY_ROOT_ANTIGRAVITY / relative
        if not path.is_file():
            raise ValueError(f"Exp473 code-map closure is missing {relative}")
        paths.add(path)
    return tuple(sorted(paths))


def _imports_antigravity(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_antigravity(path).split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = list(package)
            if node.level:
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


def _repository_python_paths_antigravity() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in REPOSITORY_ROOT_ANTIGRAVITY.rglob("*.py")
            if ".git" not in path.parts
            and ".worktrees" not in path.relative_to(REPOSITORY_ROOT_ANTIGRAVITY).parts
            and "__pycache__" not in path.parts
        )
    )


def _callers_antigravity(
    selected: tuple[Path, ...]
) -> dict[str, tuple[str, ...]]:
    modules = {
        _module_antigravity(path): _relative_antigravity(path)
        for path in selected
        if path.suffix == ".py"
    }
    callers: dict[str, set[str]] = defaultdict(set)
    for source in _repository_python_paths_antigravity():
        try:
            imports = _imports_antigravity(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        caller = _relative_antigravity(source)
        for imported in imports:
            for module, target in modules.items():
                if imported == module or imported.startswith(f"{module}."):
                    callers[target].add(caller)
    return {
        _relative_antigravity(path): tuple(
            sorted(callers[_relative_antigravity(path)])
        )
        for path in selected
    }


def _is_test_antigravity(path: str) -> bool:
    return "tests" in PurePosixPath(path).parts


def _effect_antigravity(path: Path) -> str:
    if path.suffix == ".py":
        return (
            ast.get_docstring(
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            )
            or "No module docstring is present."
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return f"Governance document: {line[2:]}"
    return f"Retained Exp473 closure artifact: {_relative_antigravity(path)}"


def build_payload_exp473_antigravity() -> dict[str, object]:
    paths = _selected_paths_antigravity()
    callers = _callers_antigravity(paths)
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, str]] = []
    for path in paths:
        relative = _relative_antigravity(path)
        direct = callers[relative]
        tests = tuple(caller for caller in direct if _is_test_antigravity(caller))
        nodes.append(
            {
                "path": relative,
                "module": _module_antigravity(path)
                if path.suffix == ".py"
                else relative,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "callers": list(direct),
                "effects": _effect_antigravity(path),
                "tests": list(tests),
            }
        )
        edges.extend(
            {"caller": caller, "callee": relative, "kind": "ast_import"}
            for caller in direct
        )
    namespace_prefix = f"{NAMESPACE_ANTIGRAVITY}/"
    return {
        "schema": "exp473-codemap.v1",
        "scope": {
            "experiment": "Exp473",
            "version": "exp473",
            "namespace": NAMESPACE_ANTIGRAVITY,
            "node_count": len(nodes),
            "production_node_count": sum(
                node["path"].startswith(namespace_prefix)
                and not _is_test_antigravity(node["path"])
                for node in nodes
            ),
            "test_node_count": sum(
                node["path"].startswith(namespace_prefix)
                and _is_test_antigravity(node["path"])
                for node in nodes
            ),
            "questions": {
                "what_calls_it": "nodes[].callers and edges",
                "what_does_it_affect": "nodes[].effects and flows",
                "which_tests_cover_it": "nodes[].tests",
            },
        },
        "nodes": nodes,
        "edges": sorted(edges, key=lambda item: (item["caller"], item["callee"])),
        "flows": [
            {
                "name": "exp473_d_prime_score_flow",
                "from": "verified Exp467 values-only dataset bytes",
                "through": "current Exp450 condition-D polyphase/PCHIP/area scorer",
                "to": "finite float64 hard-plus-stable Exp473 score shards",
            },
            {
                "name": "exp473_durable_execution_flow",
                "from": "frozen smoke or full catalog",
                "through": "bounded two-wave CPU-minus-one or single-MPS runner",
                "to": "create-only digest-bound terminal manifest",
            },
        ],
    }


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _html_bytes_antigravity(payload: object) -> bytes:
    embedded = html.escape(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    document = f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><title>Time Series · Exp473 Code Map</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;line-height:1.5}}pre{{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:.5rem}}</style></head>
<body><h1>Exp473 Code Map</h1><p>D′ 동시대 PCHIP 통제군의 호출자, 영향 범위, 테스트를 결속한 지도입니다.</p><pre>{embedded}</pre></body></html>
"""
    return document.encode("utf-8")


def generate_exp473_codemap_antigravity(*, check: bool = False) -> None:
    payload = build_payload_exp473_antigravity()
    json_bytes = _json_bytes_antigravity(payload)
    html_bytes = _html_bytes_antigravity(payload)
    lock_payload = {
        "schema": "exp473-codemap-lock.v1",
        "files": {
            _relative_antigravity(HTML_PATH_ANTIGRAVITY): sha256(html_bytes).hexdigest(),
            _relative_antigravity(JSON_PATH_ANTIGRAVITY): sha256(json_bytes).hexdigest(),
        },
    }
    lock_bytes = _json_bytes_antigravity(lock_payload)
    expected = (
        (HTML_PATH_ANTIGRAVITY, html_bytes),
        (JSON_PATH_ANTIGRAVITY, json_bytes),
        (LOCK_PATH_ANTIGRAVITY, lock_bytes),
    )
    if check:
        mismatched = [
            _relative_antigravity(path)
            for path, raw in expected
            if not path.is_file() or path.read_bytes() != raw
        ]
        if mismatched:
            raise ValueError(f"Exp473 code-map triplet is stale: {mismatched}")
        return
    HTML_PATH_ANTIGRAVITY.parent.mkdir(parents=True, exist_ok=True)
    for path, raw in expected:
        path.write_bytes(raw)


if __name__ == "__main__":
    arguments = tuple(sys.argv[1:])
    if arguments not in {(), ("--check",)}:
        raise SystemExit("usage: generate_exp473_codemap_antigravity.py [--check]")
    generate_exp473_codemap_antigravity(check=arguments == ("--check",))
