#!/usr/bin/env python3
"""Build and validate the bounded initial Exp455 code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp455_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp455_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp455_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp455_antigravity.html",
    "docs/codemap/codemap_exp455_antigravity.json",
    "docs/codemap/codemap_exp455_antigravity.lock",
}
TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP455_DASHBOARD_antigravity.html",
    "docs/EXP455_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP455_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP455_RUNBOOK_antigravity_KO.html",
    "docs/superpowers/plans/2026-08-17-exp455-conservative-label-free-routing-antigravity.html",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/config_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/contracts_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/authority_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/features_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_authority_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_features_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_config_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp455_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "reads", "tests", "future_unknown"}


def _canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_ANTIGRAVITY, text=True).strip()


def _tracked_fingerprints_antigravity() -> dict[str, str]:
    return {
        relative: hashlib.sha256((ROOT_ANTIGRAVITY / relative).read_bytes()).hexdigest()
        for relative in TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY
    }


def _module_fingerprints_antigravity() -> dict[str, str]:
    groups: dict[str, list[str]] = {"root": []}
    for relative in TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY:
        groups.setdefault("root" if "/" not in relative else relative.split("/", 1)[0], []).append(relative)
    result: dict[str, str] = {}
    for group, relatives in sorted(groups.items()):
        digest = hashlib.sha256()
        for relative in sorted(relatives):
            data = (ROOT_ANTIGRAVITY / relative).read_bytes()
            digest.update(relative.encode() + b"\0" + str(len(data)).encode() + b"\0" + data + b"\0")
        result[group] = digest.hexdigest()
    return result


def _evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def _html_antigravity(graph: dict[str, object]) -> str:
    payload = html_module.escape(_canonical_bytes_antigravity(graph).decode("utf-8"))
    nodes = "".join(
        f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp455 초기 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp455 Task 2 코드맵</h1><p>구현된 Exp454 authority와 label-free feature 경계, 그리고 미래 router/evaluation 경계를 분리한다. '
        'label, confusion, dataset variant number는 router inputs가 아니다.</p>'
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
    package = "experiments/exp455_exp454_conservative_label_free_routing_antigravity"
    plan = "docs/superpowers/plans/2026-08-17-exp455-conservative-label-free-routing-antigravity.html"
    governance_test = f"{package}/tests/test_governance_antigravity.py"
    config_test = f"{package}/tests/test_config_antigravity.py"
    contracts_test = f"{package}/tests/test_contracts_antigravity.py"
    authority = f"{package}/authority_antigravity.py"
    features = f"{package}/features_antigravity.py"
    authority_test = f"{package}/tests/test_authority_antigravity.py"
    features_test = f"{package}/tests/test_features_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html", "role": "Caller: Task 1 governance. Effect: fixes Exp455 scope, candidates, gates, and label-free firewall.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html", "Exp455")]},
        {"id": "implementation_plan", "path": plan, "role": "Caller: Tasks 1–6. Effect: freezes sequential TDD and official-evaluation boundaries.", "tests": [governance_test], "evidence": [_evidence_antigravity(plan, "Task 1: Governance")]},
        {"id": "preregistration", "path": "docs/EXP455_PREREGISTRATION_antigravity_KO.html", "role": "Caller: governance review. Effect: records planned-only frozen identity and firewall.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP455_PREREGISTRATION_antigravity_KO.html", "Exp455")]},
        {"id": "runbook", "path": "docs/EXP455_RUNBOOK_antigravity_KO.html", "role": "Caller: future operators. Effect: prohibits Task 1 official execution and TEST-label access.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP455_RUNBOOK_antigravity_KO.html", "Task 1")]},
        {"id": "dashboard", "path": "docs/EXP455_DASHBOARD_antigravity.html", "role": "Caller: planned status display. Effect: exposes 8/8 label-free features, Tasks 2/6, and not_started result state.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP455_DASHBOARD_antigravity.html", "Tasks 2/6")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Caller: governance review. Effect: records Exp455 as planned, non-promoted, and not officially run.", "tests": [governance_test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "\"exp455\"")]},
        {"id": "config", "path": f"{package}/config_antigravity.py", "role": "Caller: future authority, router, and evaluation. Effect: freezes approved identity, scope, gates, evidence hashes, and official Python.", "tests": [config_test, governance_test], "evidence": [_evidence_antigravity(f"{package}/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "contracts", "path": f"{package}/contracts_antigravity.py", "role": "Caller: future authority, router, and evaluation. Effect: freezes label-free feature, authority, rule, decision, and result records.", "tests": [contracts_test, governance_test], "evidence": [_evidence_antigravity(f"{package}/contracts_antigravity.py", "DatasetRoutingFeaturesAntigravity")]},
        {"id": "governance_regression", "path": governance_test, "role": "Caller: Task 1 focused verification. Effect: rejects missing planned governance and unmarked downstream boundaries.", "tests": [governance_test], "evidence": [_evidence_antigravity(governance_test, "class GovernanceAntigravityTest")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp455_antigravity.py", "role": "Caller: Task 1 map validation. Effect: checks evidence, future boundaries, fingerprints, and HTML/JSON parity.", "tests": [governance_test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp455_antigravity.py", "validate_map_antigravity")]},
        {"id": "authority", "path": authority, "role": "Caller: Task 2 feature builder and future router/evaluation. Effect: verifies Exp454 12 leaves and combines Exp446 stable-26 plus Exp404 Hard coordinates without labels.", "tests": [authority_test], "evidence": [_evidence_antigravity(authority, "capture_routing_authority_antigravity")]},
        {"id": "features", "path": features, "role": "Caller: future Task 3 router. Effect: emits exactly eight ECG5000-then-ChlorineConcentration integer feature rows without confusion or labels.", "tests": [features_test], "evidence": [_evidence_antigravity(features, "build_label_free_features_antigravity")]},
        {"id": "future_router", "path": f"{package}/router_antigravity.py", "role": "Future Task 3 leave-one-dataset-out routing boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
        {"id": "future_evaluation", "path": f"{package}/evaluation_antigravity.py", "role": "Future Task 4 evaluation and frozen-result boundary.", "tests": [], "evidence": [], "status": "future_unknown"},
    ]
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": _evidence_antigravity("docs/EXP455_PREREGISTRATION_antigravity_KO.html", "Exp455")},
        {"from": "implementation_plan", "to": "config", "type": "governs", "evidence": _evidence_antigravity(plan, "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "implementation_plan", "to": "contracts", "type": "governs", "evidence": _evidence_antigravity(plan, "ConfusionAntigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": _evidence_antigravity(governance_test, "test_task_one_governance_is_planned_and_non_promoted_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": _evidence_antigravity(governance_test, "builder.validate_map_antigravity")},
        {"from": "authority", "to": "config", "type": "reads", "evidence": _evidence_antigravity(authority, "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "authority", "to": "contracts", "type": "reads", "evidence": _evidence_antigravity(authority, "FrozenRoutingAuthorityAntigravity")},
        {"from": "features", "to": "authority", "type": "reads", "evidence": _evidence_antigravity(features, "FrozenRoutingAuthorityAntigravity")},
        {"from": "future_router", "to": "features", "type": "future_unknown", "evidence": _evidence_antigravity(plan, "Task 3: 가족별 leave-one-dataset-out 라우터")},
        {"from": "future_router", "to": "future_evaluation", "type": "future_unknown", "evidence": _evidence_antigravity(plan, "evaluate_routing_antigravity")},
    ]
    unknowns = [
        {"from": "future_router", "to": "features", "reason": "Task 3 router is not implemented."},
        {"from": "future_router", "to": "future_evaluation", "reason": "Task 4 evaluation is not implemented."},
    ]
    graph: dict[str, object] = {"schema": "exp455_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp455_codemap_map_child_v1", "source_parent_commit": source_commit, "working_tree_dirty": False, "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "module_fingerprints": _module_fingerprints_antigravity(), "tracked_fingerprints": _tracked_fingerprints_antigravity()}
    return graph, lock, _html_antigravity(graph)


def _validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str):
        raise ValueError("missing evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("invalid evidence")


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
            if node.get("evidence") != []:
                raise ValueError("future node claimed evidence")
        else:
            if not isinstance(node.get("path"), str) or not (ROOT_ANTIGRAVITY / node["path"]).is_file() or not node.get("evidence"):
                raise ValueError("missing node evidence")
            for evidence in node["evidence"]:
                _validate_evidence_antigravity(evidence)
    unknown_pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    future_edges: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY:
            raise ValueError("invalid edge")
        _validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in unknown_pairs:
                raise ValueError("unmarked future relationship")
            future_edges.add(pair)
        elif edge["type"] == "future_unknown":
            raise ValueError("known edge marked future")
    if unknown_pairs != future_edges:
        raise ValueError("unknown relationship mismatch")
    source = lock.get("source_parent_commit")
    head = _git_output_antigravity("rev-parse", "HEAD")
    if source != graph.get("source_parent_commit") or not isinstance(source, str) or source not in {head, _git_output_antigravity("rev-parse", "HEAD^")}:
        raise ValueError("invalid source parent")
    if lock.get("working_tree_dirty") is not False or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != _module_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    match = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, flags=re.DOTALL)
    if not match or _canonical_bytes_antigravity(_strict_json_antigravity(html_module.unescape(match.group(1)))) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-17T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(_strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), _strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at)
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph) + b"\n")
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock) + b"\n")
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    main_antigravity()
