#!/usr/bin/env python3
"""Build and validate the bounded Exp461 Task 1 governance code map."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html as html_module
import html.parser
import json
import re
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp461_antigravity.html",
    "docs/codemap/codemap_exp461_antigravity.json",
    "docs/codemap/codemap_exp461_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP461_DASHBOARD_antigravity.html",
    "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP461_PREREGISTRATION_antigravity.html",
    "docs/EXP461_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html",
    "docs/superpowers/specs/2026-08-20-exp461-ecg-period-signature-qualification-antigravity-design.html",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp461_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "tests"}


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def strict_json_antigravity(text_or_path: str | Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    text = (
        text_or_path.read_text(encoding="utf-8")
        if isinstance(text_or_path, Path)
        else text_or_path
    )
    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def python_symbols_antigravity(text: str) -> set[str]:
    tree = ast.parse(text)
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
    return symbols


def evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".html":
        parser = HtmlTextExtractorAntigravity()
        parser.feed(text)
        return symbol in " ".join(parser.parts)
    if target.suffix == ".py":
        return symbol in python_symbols_antigravity(text)
    return symbol in text


def tracked_fingerprints_antigravity() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT_ANTIGRAVITY / path).read_bytes()).hexdigest()
        for path in TRACKED_PATHS_ANTIGRAVITY
    }


def module_fingerprints_antigravity() -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for path in TRACKED_PATHS_ANTIGRAVITY:
        groups.setdefault(path.split("/", 1)[0] if "/" in path else "root", []).append(path)
    result: dict[str, str] = {}
    for group, paths in sorted(groups.items()):
        digest = hashlib.sha256()
        for path in sorted(paths):
            data = (ROOT_ANTIGRAVITY / path).read_bytes()
            digest.update(path.encode() + b"\0" + str(len(data)).encode() + b"\0" + data + b"\0")
        result[group] = digest.hexdigest()
    return result


def html_antigravity(graph: dict[str, object]) -> str:
    payload = html_module.escape(canonical_bytes_antigravity(graph).decode("utf-8"))
    nodes = "".join(
        f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp461 Task 1 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp461 Task 1 governance 코드맵</h1><p>Task 1은 governance docs, planned ledgers, always-visible dashboard, bounded map, and the Task 1 regression test만 보존한다. 공식 실행, 결과 root, scientific source implementation은 아직 없다.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        '</main></body></html>'
    )


def build_map_antigravity(
    source_commit: str,
    generated_at: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=ROOT_ANTIGRAVITY,
        check=False,
    ).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must end in Z")

    design = "docs/superpowers/specs/2026-08-20-exp461-ecg-period-signature-qualification-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html"
    preregistration = "docs/EXP461_PREREGISTRATION_antigravity.html"
    runbook = "docs/EXP461_RUNBOOK_antigravity.html"
    report = "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html"
    dashboard = "docs/EXP461_DASHBOARD_antigravity.html"
    governance_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    builder = "scratch/build_codemap_exp461_antigravity.py"

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Task 1-6 governance and implementation. Effect: freezes the three-method comparison, TRAIN-only boundary, predecessor failed folds, and official-root absence rule.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp461 — ECG 주기 특징 성능 검증")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Task 1-6 execution. Effect: records the owned files, Task 1 step checkboxes, current pending-external-review state, and later task boundaries.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 1 — 문서, 진행 대시보드, 코드맵")],
        },
        {
            "id": "preregistration",
            "path": preregistration,
            "role": "Caller: governance history. Effect: preserves planned-only experiment truth, official command count 0, label query count 0, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(preregistration, "planned_implementation_not_started")],
        },
        {
            "id": "runbook",
            "path": runbook,
            "role": "Caller: future operators. Effect: records the Task 1 operator boundary, exact root that must stay absent, and the commands that remain forbidden for now.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(runbook, "Task 1 operator boundary")],
        },
        {
            "id": "implementation_report",
            "path": report,
            "role": "Caller: Task 2 handoff. Effect: records the immutable design/preregistration/runbook SHA-256 authority and states that living progress files are read directly.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(report, "Task 1 governance report")],
        },
        {
            "id": "dashboard",
            "path": dashboard,
            "role": "Caller: always-visible status review. Effect: shows Task 1/6, planned status, official execution 0, null artifact hashes, and the two inherited period-change failures.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(dashboard, "Task 1/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records the Exp461 planned entry with zero official counts, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp461\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp461 as planned-not-started and non-promoted with the frozen predecessor reference to Exp460.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp461\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp461 planned history without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp461\"")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: Task 1 verification. Effect: rejects missing docs, wrong planned ledger truth, wrong Task 1 checkbox state, or invalid stored map bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_task1_governance_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 1 validation. Effect: rebuilds and validates the bounded Exp461 governance graph plus the stored HTML/JSON/lock triplet bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
    ]

    edges = [
        {
            "from": "approved_design",
            "to": "preregistration",
            "type": "governs",
            "evidence": evidence_antigravity(
                preregistration,
                "frozen predecessor truth",
            ),
        },
        {
            "from": "approved_design",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity(dashboard, "hydra_period_rank_max"),
        },
        {
            "from": "implementation_plan",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity(dashboard, "공식 실행 0회"),
        },
        {
            "from": "implementation_plan",
            "to": "governance_regression",
            "type": "governs",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "preregistration",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "runbook",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "implementation_report",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "dashboard",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "research_state",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "experiment_catalog",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "champion_ledger",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "map_builder",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
    ]

    graph = {
        "schema": "exp461_task1_governance_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp461_task1_governance_codemap_lock_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "tracked_fingerprints": tracked_fingerprints_antigravity(),
        "module_fingerprints": module_fingerprints_antigravity(),
        "working_tree_dirty": False,
    }
    return graph, lock, html_antigravity(graph)


def validate_map_antigravity(
    graph: dict[str, object],
    lock: dict[str, object],
    rendered_html: str,
) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph/lock must be dicts")
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("nodes and edges must be lists")
    if graph.get("unknown_relationships") != []:
        raise ValueError("unknown relationships must stay empty")
    ids = set()
    for node in graph["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not isinstance(node.get("path"), str)
            or not isinstance(node.get("role"), str)
            or not isinstance(node.get("tests"), list)
            or not node["tests"]
            or not isinstance(node.get("evidence"), list)
            or not node["evidence"]
        ):
            raise ValueError("invalid node")
        ids.add(node["id"])
        for item in node["evidence"]:
            if not isinstance(item, dict) or set(item) != {"path", "symbol"}:
                raise ValueError("invalid evidence")
            if not evidence_resolves_antigravity(item["path"], item["symbol"]):
                raise ValueError("unresolved evidence")
    for edge in graph["edges"]:
        if (
            not isinstance(edge, dict)
            or edge.get("from") not in ids
            or edge.get("to") not in ids
            or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY
        ):
            raise ValueError("invalid edge")
        item = edge.get("evidence")
        if not isinstance(item, dict) or set(item) != {"path", "symbol"}:
            raise ValueError("invalid edge evidence")
        if not evidence_resolves_antigravity(item["path"], item["symbol"]):
            raise ValueError("unresolved edge evidence")

    source = lock.get("source_parent_commit")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()
    if source != graph.get("source_parent_commit") or source not in {head, parent}:
        raise ValueError("invalid source parent")
    if (
        lock.get("working_tree_dirty") is not False
        or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY)
    ):
        raise ValueError("invalid lock")
    if (
        lock.get("tracked_fingerprints") != tracked_fingerprints_antigravity()
        or lock.get("module_fingerprints") != module_fingerprints_antigravity()
    ):
        raise ValueError("fingerprint drift")

    embedded = re.search(
        r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>',
        rendered_html,
        flags=re.DOTALL,
    )
    if not embedded:
        raise ValueError("embedded graph missing")
    embedded_graph = strict_json_antigravity(html_module.unescape(embedded.group(1)))
    if canonical_bytes_antigravity(embedded_graph) != canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-20T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(
            strict_json_antigravity(JSON_PATH_ANTIGRAVITY),
            strict_json_antigravity(LOCK_PATH_ANTIGRAVITY),
            HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8"),
        )
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at)
    JSON_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(graph) + b"\n")
    LOCK_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(lock) + b"\n")
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    main_antigravity()
