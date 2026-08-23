#!/usr/bin/env python3
"""Build and validate the bounded Exp465 Task 1 code-map triplet."""
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
from typing import Any

ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
HTML_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp465_antigravity.html"
JSON_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp465_antigravity.json"
LOCK_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp465_antigravity.lock"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp465_antigravity.html",
    "docs/codemap/codemap_exp465_antigravity.json",
    "docs/codemap/codemap_exp465_antigravity.lock",
)
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP465_DASHBOARD_antigravity.html",
    "docs/EXP465_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP465_PREREGISTRATION_antigravity.html",
    "docs/EXP465_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html",
    "docs/superpowers/specs/2026-08-22-exp465-ecg200-c8-external-validation-antigravity-design.html",
    ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-1-report-antigravity.html",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/config_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/contracts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp465_antigravity.py",
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp465_task1_contract_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp465_task1_contract_codemap_lock_v1"
GENERATED_AT_ANTIGRAVITY = "2026-08-22T00:00:00Z"


def _canonical_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class _HtmlTextAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _python_symbols_antigravity(text: str) -> set[str]:
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


def _evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".html":
        parser = _HtmlTextAntigravity()
        parser.feed(text)
        return symbol in " ".join(parser.parts)
    if target.suffix == ".py":
        return symbol in _python_symbols_antigravity(text)
    return symbol in text


def _node_antigravity(node_id: str, path: str, role: str, symbol: str, test: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "tests": [test],
        "evidence": [{"path": path, "symbol": symbol}],
    }


def _edge_antigravity(source: str, relation: str, target: str, evidence_path: str, evidence_symbol: str) -> dict[str, str]:
    return {
        "from": source,
        "to": target,
        "type": relation,
        "evidence": {"path": evidence_path, "symbol": evidence_symbol},
    }


def _tracked_fingerprints_antigravity() -> list[dict[str, str]]:
    return [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                (ROOT_ANTIGRAVITY / relative).read_bytes(),
            ).hexdigest(),
        }
        for relative in TRACKED_PATHS_ANTIGRAVITY
    ]


def _source_parent_antigravity() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()


def build_graph_antigravity(source_parent: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_parent) is None:
        raise ValueError("source_parent must be a lowercase 40-hex commit")
    config = "experiments/exp465_ecg200_c8_external_validation_antigravity/config_antigravity.py"
    contracts = "experiments/exp465_ecg200_c8_external_validation_antigravity/contracts_antigravity.py"
    config_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_config_antigravity.py"
    contracts_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_contracts_antigravity.py"
    governance_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_governance_antigravity.py"
    design = "docs/superpowers/specs/2026-08-22-exp465-ecg200-c8-external-validation-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html"
    dashboard = "docs/EXP465_DASHBOARD_antigravity.html"
    preregistration = "docs/EXP465_PREREGISTRATION_antigravity.html"
    runbook = "docs/EXP465_RUNBOOK_antigravity.html"
    report = "docs/EXP465_IMPLEMENTATION_REPORT_antigravity.html"
    state = "RESEARCH_STATE.json"
    catalog = "EXPERIMENT_CATALOG.json"
    champion = "CHAMPION_LEDGER.json"
    builder = "scratch/build_codemap_exp465_antigravity.py"
    task1_report = ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-1-report-antigravity.html"
    governance_symbol = "test_task1_truth_documents_ledgers_and_readonly_codemap_antigravity"
    nodes = [
        _node_antigravity("approved_design", design, "Caller: Exp465 plan and Task 1 contract. Effect: freezes ECG200 TRAIN 100x96 normal69, C0/C6/C8 order with C8-only primary, seeds 20260721/20261721, five folds, onset 56, cosine 0.95, four AUC gates 0.85, mean 0.90, finite/non-degradation/zero-confirmation gates, and 10-of-10 requirement.", "Exp465 — ECG200에서 C8을 그대로 외부 검증", governance_test),
        _node_antigravity("implementation_plan", plan, "Caller: implementers and reviewers. Effect: bounds Task 1 to frozen config/contracts, three focused tests, Korean HTML truth documents, map triplet, ledgers, dashboard, and only actually completed checkbox updates; Tasks 2–5 remain unchecked.", "Task 1 — 계약, 문서와 코드맵", governance_test),
        _node_antigravity("dashboard", dashboard, "Caller: user-facing status review. Effect: reports Task 1 implementation verified pending fresh review, exact gates and counts, absent official root, official0/private0/label0, null result hashes, and TEST/routing/promotion boundaries.", "현재 상태", governance_test),
        _node_antigravity("preregistration", preregistration, "Caller: governance review. Effect: freezes immutable data/method/gate contract and records Wilson as explanatory only plus CPU fallback acceleration assessment before any scoring run.", "Frozen qualification gates for C8", governance_test),
        _node_antigravity("runbook", runbook, "Caller: operators. Effect: keeps official root absent, forbids TEST access and post-failure contract drift, requires later sequential complete private reproductions, and lists Task 1 read-only validation commands.", "must stay absent until separate approval", governance_test),
        _node_antigravity("implementation_report", report, "Caller: fresh review and audit. Effect: records Task 1 scope, RED-to-GREEN twice, codemap validation, diff hygiene, CPU fallback assessment, and remaining review boundary without claiming qualification.", "task_1_implementation_verified_fresh_review_pending", governance_test),
        _node_antigravity("task1_report", task1_report, "Caller: controller and fresh reviewer. Effect: records Task 1 precommit boundary, exact double-GREEN/codemap/diff-check evidence, unchanged official0/private0/label0/null hashes, and unchecked review-only heading/Step 6 truth.", "implementation_verified_precommit_fresh_review_pending", governance_test),
        _node_antigravity("research_state", state, "Caller: active research ledger. Effect: adds Exp465 implementation status while preserving prior experiments; stores label0/official0/null artifact-map governance truth and blocks routing/promotion.", "\"exp465\"", governance_test),
        _node_antigravity("experiment_catalog", catalog, "Caller: experiment inventory. Effect: registers exactly one Exp465 entry with approved design/plan, package, scientific contract, zero-run counts, and no result hash.", "\"id\": \"Exp465\"", governance_test),
        _node_antigravity("champion_ledger", champion, "Caller: champion history. Effect: preserves champion and pending state while recording that Exp464 C8 remains a non-promotable clue and Exp465 changes neither routing nor promotion.", "\"experiment\": \"Exp465\"", governance_test),
        _node_antigravity("task1_config", config, "Caller: contracts/tests and future source/scoring modules. Effect: freezes exact-type Exp465 identity/data/method/gate/governance values and rejects aliases, mutable/reordered mappings, C6 primary, changed thresholds, and altered counts.", "FROZEN_CONFIG_ANTIGRAVITY", config_test),
        _node_antigravity("task1_contracts", contracts, "Caller: future scoring, qualification, evaluation, artifacts, replay, and CLI tasks. Effect: freezes immutable score/fold/qualification containers and returns initial governance with zero label query, zero official evaluate, and null artifact map; does not load data or score.", "initial_governance_state_antigravity", contracts_test),
        _node_antigravity("task1_config_test", config_test, "Caller: Task 1 TDD verification. Effect: preserves controller RED import failure and covers exact truth, immutability, alias rejection, mutable/reordered mapping rejection, and changed-gate rejection.", "Exp465ConfigTestAntigravity", config_test),
        _node_antigravity("task1_contracts_test", contracts_test, "Caller: Task 1 TDD verification. Effect: covers field inventories, tuple immutability, score/result aliases, nonfinite AUCs, mean consistency, reason/pass agreement, ten-row shape, seed order, and initial governance.", "ContractsAntigravityTest", contracts_test),
        _node_antigravity("task1_governance_test", governance_test, "Caller: required double GREEN and Task 1 review. Effect: verifies files, HTML, ledger truth, document phrases, checkbox completion truth, map graph/evidence/source binding, and read-only validate behavior.", "GovernanceAntigravityTest", governance_test),
        _node_antigravity("map_builder", builder, "Caller: Task 1 boundary validation. Effect: writes canonical HTML/JSON/lock triplet or validates it read-only via exact --validate-only, resolving evidence and fingerprinting tracked paths.", "build_triplet_antigravity", governance_test),
    ]
    edges = [
        _edge_antigravity("approved_design", "governs", "task1_config", design, "C8 합격 조건"),
        _edge_antigravity("approved_design", "governs", "preregistration", design, "C8 합격 조건"),
        _edge_antigravity("implementation_plan", "governs", "task1_config", plan, "Step 2 — Immutable config/contracts."),
        _edge_antigravity("implementation_plan", "governs", "map_builder", plan, "Step 4 — Code map."),
        _edge_antigravity("task1_config", "governs", "task1_contracts", config, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", config_test, "test_frozen_contract_preserves_all_exp465_truth"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", contracts_test, "test_qualification_freezes_rows_seed_map_and_initial_governance"),
        _edge_antigravity("task1_governance_test", "tests", "dashboard", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "preregistration", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "runbook", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "implementation_report", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "task1_report", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "research_state", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "experiment_catalog", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "champion_ledger", governance_test, governance_symbol),
        _edge_antigravity("task1_governance_test", "tests", "map_builder", governance_test, governance_symbol),
        _edge_antigravity("map_builder", "documents_boundary", "task1_config", builder, "build_triplet_antigravity"),
    ]
    return {
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent_commit": source_parent,
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
        "flows": [
            "design/plan -> config -> contracts -> future Task 2+ inputs/scoring/qualification",
            "config/contracts tests + governance regression -> documents, ledgers, dashboard, plan truth",
            "map_builder --validate-only -> read-only triplet/evidence/fingerprint verification",
        ],
    }


def _html_antigravity(graph: dict[str, Any]) -> str:
    payload = html_module.escape(
        _canonical_bytes_antigravity(graph).decode("utf-8").rstrip("\n"),
    )
    node_items = []
    for node in graph["nodes"]:
        path_text = html_module.escape(str(node["path"]))
        role_text = html_module.escape(str(node["role"]))
        node_items.append(f"<li><code>{path_text}</code> — {role_text}</li>")
    nodes = "".join(node_items)
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp465 Task 1 코드맵</title><style>body{max-width:1160px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}pre{padding:1rem}</style></head><body><main>'
        '<h1>Exp465 Task 1 코드맵</h1><p>이 map은 frozen ECG200 C8 external-validation contract, immutable score/result/governance contracts, focused tests, truth documents, ledgers, dashboard, plan checkbox truth, and the read-only map boundary를 설명한다. 자료 loading과 scoring은 아직 구현하지 않았다.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        "</main></body></html>"
    )


def build_lock_antigravity(source_parent: str) -> dict[str, Any]:
    fingerprints = _tracked_fingerprints_antigravity()
    return {
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent_commit": source_parent,
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "tracked_fingerprints": fingerprints,
        "fingerprint_count": len(fingerprints),
        "working_tree_dirty": False,
    }


def validate_map_antigravity(graph: dict[str, Any], lock: dict[str, Any], rendered: str) -> None:
    if graph.get("unknown_relationships") != []:
        raise ValueError("unknown relationships must stay empty")
    allowed_types = {"governs", "tests", "documents_boundary"}
    ids = set()
    for node in graph["nodes"]:
        if not isinstance(node, dict) or not node.get("id") or not node.get("path") or not node.get("role"):
            raise ValueError("invalid node")
        ids.add(node["id"])
        for item in node["evidence"]:
            if set(item) != {"path", "symbol"} or not _evidence_resolves_antigravity(item["path"], item["symbol"]):
                raise ValueError(f"unresolved evidence: {item}")
        if not node["tests"] or any(test not in TRACKED_PATHS_ANTIGRAVITY for test in node["tests"]):
            raise ValueError("invalid node test")
    for edge in graph["edges"]:
        if edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in allowed_types:
            raise ValueError("invalid edge")
        item = edge.get("evidence")
        if set(item) != {"path", "symbol"} or not _evidence_resolves_antigravity(item["path"], item["symbol"]):
            raise ValueError(f"unresolved edge evidence: {item}")
    source = lock.get("source_parent_commit")
    if source != graph.get("source_parent_commit"):
        raise ValueError("graph/lock source mismatch")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY) or lock.get("working_tree_dirty") is not False:
        raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    embedded = re.search(
        r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>',
        rendered,
        flags=re.DOTALL,
    )
    if not embedded:
        raise ValueError("embedded graph missing")
    embedded_graph = json.loads(html_module.unescape(embedded.group(1)))
    if _canonical_bytes_antigravity(embedded_graph) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def build_triplet_antigravity(source_parent: str, *, validate_only: bool = False) -> None:
    if validate_only:
        validate_map_antigravity(
            json.loads(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
            json.loads(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
            HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8"),
        )
        return
    graph = build_graph_antigravity(source_parent)
    lock = build_lock_antigravity(source_parent)
    rendered = _html_antigravity(graph)
    validate_map_antigravity(graph, lock, rendered)
    HTML_PATH_ANTIGRAVITY.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph))
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock))


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        build_triplet_antigravity(_source_parent_antigravity(), validate_only=True)
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    build_triplet_antigravity(args.source_commit)


if __name__ == "__main__":
    main_antigravity()
