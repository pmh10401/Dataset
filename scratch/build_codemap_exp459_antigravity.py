#!/usr/bin/env python3
"""Build and validate the bounded Exp459 Task 1 governance code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp459_antigravity.html",
    "docs/codemap/codemap_exp459_antigravity.json",
    "docs/codemap/codemap_exp459_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP459_DASHBOARD_antigravity.html",
    "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP459_PREREGISTRATION_antigravity.html",
    "docs/EXP459_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html",
    "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp459_antigravity.py",
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


def strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

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
        '<title>Exp459 Task 1 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp459 Task 1 거버넌스 코드맵</h1><p>Task 1은 Exp459를 planned_not_started로 고정하고, self-contained HTML 문서 4개와 세 ledger, bounded governance regression, code map triplet만 추가한다. 공식 evaluate/inspect, label query, 공식 root 생성, champion 변경은 모두 아직 없다.</p>'
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

    design = "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html"
    governance_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    builder = "scratch/build_codemap_exp459_antigravity.py"

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Task 1 governance and future Tasks 2–6. Effect: freezes the blocked-validation identity, predecessor, fold sizes, and stable decision rule.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp459")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Task 1 execution. Effect: records checked Task 1 steps while Tasks 2–6 remain unchecked pending later work.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 1: Exp458 완료 상태 정리와 Exp459 거버넌스")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP459_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: states planned_not_started identity, fixed thresholds, zero official actions, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_PREREGISTRATION_antigravity.html", "planned_not_started")],
        },
        {
            "id": "runbook",
            "path": "docs/EXP459_RUNBOOK_antigravity.html",
            "role": "Caller: future operators. Effect: says Task 1 allows docs and ledgers only, and blocks official commands, root creation, and label access.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_RUNBOOK_antigravity.html", "Task 1에서 금지한 일")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports the Exp458 test correction, new governance artifacts, and the still-unstarted Exp459 lifecycle.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html", "Task 1 거버넌스와 문서화만 완료했다")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP459_DASHBOARD_antigravity.html",
            "role": "Caller: reader-facing review. Effect: shows Task 1/6 progress, planned_not_started status, zero official commands, and unchanged champion state.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_DASHBOARD_antigravity.html", "Task 1/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Exp459 as planned_not_started with zero official invocations, label_query_count 0, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp459\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp459 documents, frozen predecessor link, planned status, and no-result truth.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"id\": \"Exp459\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp459 without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp459\"")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: Task 1 verification. Effect: freezes the self-contained documents, planned-not-started ledger truth, official-root absence, and bounded code-map parity.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 1 validation. Effect: recomputes the governance-only graph, fingerprints, source-parent rule, and HTML/JSON/lock parity.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
    ]

    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity(design, "Exp459")},
        {"from": "approved_design", "to": "runbook", "type": "governs", "evidence": evidence_antigravity(design, "기준 커밋: Exp458 최종 커밋")},
        {"from": "approved_design", "to": "research_state", "type": "governs", "evidence": evidence_antigravity(design, "Exp458 공식 artifact map")},
        {"from": "implementation_plan", "to": "implementation_report", "type": "governs", "evidence": evidence_antigravity(plan, "Step 4 — 네 HTML 문서, 세 ledger의 Exp459 non-promotion 항목, 진행 대시보드, code map을 최소 구현한다.")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(plan, "Step 5 — governance GREEN을 두 번 실행하고 HTML parse, JSON duplicate-key, map validate, diff check를 실행한다.")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": evidence_antigravity(plan, "scratch/build_codemap_exp459_antigravity.py")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "runbook", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "implementation_report", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "experiment_catalog", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "champion_ledger", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
    ]

    graph = {
        "schema": "exp459_task1_governance_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp459_task1_governance_codemap_lock_v1",
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
            raise ValueError("unresolved evidence")

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
    parser.add_argument("--generated-at", default="2026-08-18T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(
            strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
            strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
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
