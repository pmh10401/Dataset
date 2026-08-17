#!/usr/bin/env python3
"""Build and validate the bounded Task 1 Exp457 governance code map."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp457_antigravity.html",
    "docs/codemap/codemap_exp457_antigravity.json",
    "docs/codemap/codemap_exp457_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP457_DASHBOARD_antigravity.html",
    "docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP457_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP457_RUNBOOK_antigravity_KO.html",
    "docs/superpowers/plans/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity.html",
    "docs/superpowers/specs/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity-design.html",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp457_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "tests", "future_unknown"}


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def tracked_fingerprints_antigravity() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT_ANTIGRAVITY / path).read_bytes()).hexdigest() for path in TRACKED_PATHS_ANTIGRAVITY}


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
        '<title>Exp457 Task 1 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp457 Task 1 governance 코드맵</h1><p>현재 구현된 governance, dashboard, ledger, test, builder 경계와 향후 과학 source를 분리한다. '
        'future nodes는 구현 증거가 없으며 관계도 future/unknown이다.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        '</main></body></html>'
    )


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT_ANTIGRAVITY, check=False).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must end in Z")
    plan = "docs/superpowers/plans/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity.html"
    design = "docs/superpowers/specs/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity-design.html"
    package = "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity"
    governance_test = f"{package}/tests/test_governance_antigravity.py"
    builder = "scratch/build_codemap_exp457_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": design, "role": "Caller: Task 1 governance. Effect: freezes the only scientific change and exact confidence/result gates.", "tests": [governance_test], "evidence": [evidence_antigravity(design, "Exp457")]},
        {"id": "implementation_plan", "path": plan, "role": "Caller: Tasks 1–6. Effect: sequences TDD implementation and keeps official execution outside Task 1.", "tests": [governance_test], "evidence": [evidence_antigravity(plan, "Task 1: Exp457 등록")]},
        {"id": "preregistration", "path": "docs/EXP457_PREREGISTRATION_antigravity_KO.html", "role": "Caller: governance review. Effect: records planned identity, frozen gates, and TEST routing firewall.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_PREREGISTRATION_antigravity_KO.html", "planned_awaiting_implementation")]},
        {"id": "runbook", "path": "docs/EXP457_RUNBOOK_antigravity_KO.html", "role": "Caller: future task operators. Effect: prohibits Task 1 official execution and result-root use.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_RUNBOOK_antigravity_KO.html", "Task 1 governance only")]},
        {"id": "implementation_report", "path": "docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html", "role": "Caller: Task 1 review. Effect: reports governance completion without claiming scientific implementation or results.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html", "Task 1/6")]},
        {"id": "dashboard", "path": "docs/EXP457_DASHBOARD_antigravity.html", "role": "Caller: result-facing review. Effect: shows Task 1/6, no result, Exp456 comparators, gates, and Task 2 next action.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_DASHBOARD_antigravity.html", "아직 공식 결과 없음")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Caller: governance review. Effect: registers planned lifecycle with zero official invocations and null result hashes.", "tests": [governance_test], "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp457\"")]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Caller: catalog review. Effect: registers Exp457 documents, frozen contract, and no-result truth.", "tests": [governance_test], "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"id\": \"Exp457\"")]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "Caller: champion review. Effect: records Exp457 without changing champion or pending candidate.", "tests": [governance_test], "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp457\"")]},
        {"id": "governance_regression", "path": governance_test, "role": "Caller: Task 1 verification. Effect: rejects gate drift, fabricated results, or implemented-source claims.", "tests": [governance_test], "evidence": [evidence_antigravity(governance_test, "class GovernanceAntigravityTest")]},
        {"id": "map_builder", "path": builder, "role": "Caller: Task 1 map validation. Effect: validates evidence, future boundaries, fingerprints, and HTML/JSON parity.", "tests": [governance_test], "evidence": [evidence_antigravity(builder, "validate_map_antigravity")]},
        {"id": "future_config", "path": f"{package}/config_antigravity.py", "role": "Future Task 2 frozen configuration and contract boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_synthetic", "path": f"{package}/perturbations_antigravity.py", "role": "Future Task 2 Exp417-compatible synthetic anomaly boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_router", "path": f"{package}/router_antigravity.py", "role": "Future Task 4 two-gate routing boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_evaluation", "path": f"{package}/evaluation_antigravity.py", "role": "Future Task 4 fixed-route result aggregation boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_artifacts", "path": f"{package}/artifacts_antigravity.py", "role": "Future Task 4 publication and replay evidence boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_cli", "path": f"{package}/cli_antigravity.py", "role": "Future Task 5 closed evaluate/inspect command boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
    ]
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity("docs/EXP457_PREREGISTRATION_antigravity_KO.html", "고정 게이트")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(plan, "대시보드는 Task 1/6")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp457_scientific_contract_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_dashboard_and_map_expose_only_planned_boundaries_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": evidence_antigravity(governance_test, "builder.validate_map_antigravity")},
        {"from": "future_config", "to": "future_synthetic", "type": "future_unknown", "evidence": evidence_antigravity(plan, "build_primary_perturbations_antigravity")},
        {"from": "future_synthetic", "to": "future_router", "type": "future_unknown", "evidence": evidence_antigravity(plan, "RouteDecisionAntigravity")},
        {"from": "future_router", "to": "future_evaluation", "type": "future_unknown", "evidence": evidence_antigravity(plan, "evaluate_routes_antigravity")},
        {"from": "future_evaluation", "to": "future_artifacts", "type": "future_unknown", "evidence": evidence_antigravity(plan, "publish_result_antigravity")},
        {"from": "future_artifacts", "to": "future_cli", "type": "future_unknown", "evidence": evidence_antigravity(plan, "evaluate --reviewed-commit")},
    ]
    unknowns = [
        {"from": "future_config", "to": "future_synthetic", "reason": "Task 2 source is not implemented."},
        {"from": "future_synthetic", "to": "future_router", "reason": "Task 4 routing source is not implemented."},
        {"from": "future_router", "to": "future_evaluation", "reason": "Task 4 evaluation source is not implemented."},
        {"from": "future_evaluation", "to": "future_artifacts", "reason": "Task 4 publication source is not implemented."},
        {"from": "future_artifacts", "to": "future_cli", "reason": "Task 5 CLI source is not implemented."},
    ]
    graph: dict[str, object] = {"schema": "exp457_task1_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp457_task1_codemap_lock_v1", "source_parent_commit": source_commit, "working_tree_dirty": False, "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "module_fingerprints": module_fingerprints_antigravity(), "tracked_fingerprints": tracked_fingerprints_antigravity()}
    return graph, lock, html_antigravity(graph)


def validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str):
        raise ValueError("invalid evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("evidence not found")


def validate_map_antigravity(graph: object, lock: object, rendered_html: str) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes, edges, unknowns = graph.get("nodes"), graph.get("edges"), graph.get("unknown_relationships")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(unknowns, list) or len(nodes) > 20:
        raise ValueError("invalid bounded topology")
    ids: set[str] = set()
    future_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in ids:
            raise ValueError("invalid node")
        ids.add(node["id"])
        if node.get("status") == "future_unknown":
            future_ids.add(node["id"])
            if node.get("evidence") != [] or (ROOT_ANTIGRAVITY / str(node.get("path"))).exists():
                raise ValueError("future source evidence claimed")
        else:
            if not isinstance(node.get("path"), str) or not (ROOT_ANTIGRAVITY / node["path"]).is_file() or not node.get("evidence"):
                raise ValueError("missing known evidence")
            for item in node["evidence"]:
                validate_evidence_antigravity(item)
    unknown_pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    future_pairs: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY:
            raise ValueError("invalid edge")
        validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in unknown_pairs:
                raise ValueError("unmarked future relationship")
            future_pairs.add(pair)
        elif edge["type"] == "future_unknown":
            raise ValueError("known edge marked future")
    if unknown_pairs != future_pairs:
        raise ValueError("unknown relationship mismatch")
    source = lock.get("source_parent_commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    if source != graph.get("source_parent_commit") or source not in {head, parent}:
        raise ValueError("invalid source parent")
    if lock.get("working_tree_dirty") is not False or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != module_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    embedded = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, flags=re.DOTALL)
    if not embedded or canonical_bytes_antigravity(strict_json_antigravity(html_module.unescape(embedded.group(1)))) != canonical_bytes_antigravity(graph):
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
