#!/usr/bin/env python3
"""Build and validate the bounded Exp460 Task 1 governance code map."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import html.parser
import json
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.html"
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP460_DASHBOARD_antigravity.html",
    "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP460_PREREGISTRATION_antigravity.html",
    "docs/EXP460_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html",
    "docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp460_antigravity.py",
)


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates_antigravity,
    )


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".html":
        parser = HtmlTextExtractorAntigravity()
        parser.feed(text)
        haystack = " ".join(parser.parts)
    else:
        haystack = text
    return symbol in haystack


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
        '<title>Exp460 Task 1 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp460 Task 1 governance 코드맵</h1><p>Task 1 creates only the governance surface: four HTML documents, planned ledger state, one governance regression, and one bounded Exp460 map builder. Future scientific modules are shown as planned downstream consumers only.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        '</main></body></html>'
    )


def build_map_antigravity(source_parent_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    nodes = [
        {
            "id": "approved_design",
            "path": "docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html",
            "role": "Caller: Task 1 governance and future Tasks 2-6. Effect: freezes target dataset, rule order, failed predecessor folds, and strict-only official decision rule.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "10/10 엄격 복귀")],
        },
        {
            "id": "implementation_plan",
            "path": "docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html",
            "role": "Caller: Task 1-6 execution. Effect: records Task 1 completion while leaving future tasks unchecked.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 1 — 문서, 진행 대시보드, 코드맵")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP460_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: states planned-only Exp460 contract, zero official counts, and null result hashes.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_PREREGISTRATION_antigravity.html", "governance_complete_implementation_not_started")],
        },
        {
            "id": "runbook",
            "path": "docs/EXP460_RUNBOOK_antigravity.html",
            "role": "Caller: future operators. Effect: records Task 1-only allowed scope and official execution firewall.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_RUNBOOK_antigravity.html", "Task 1 operating boundary")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports Task 1 progress, zero official actions, and no scientific implementation yet.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html", "Task 1 lifecycle")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP460_DASHBOARD_antigravity.html",
            "role": "Caller: result-facing governance review. Effect: shows Task 1/6 progress, four-rule frame, unstable predecessor truth, and design-expectation-only boundary.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_DASHBOARD_antigravity.html", "Task 1/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Exp460 as planned-only with zero official actions and unchanged pending candidate.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"task_1_governance_complete\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp460 documents, scientific contract, and planned-only lifecycle.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp460\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp460 as non-promoted planned work without changing champion or pending candidate.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"task_1_governance_complete\"")],
        },
        {
            "id": "governance_regression",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
            "role": "Caller: Task 1 verification. Effect: rejects missing docs, wrong rule order, nonzero official counters, changed champion state, stale plan checkboxes, or invalid map triplet.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity")],
        },
        {
            "id": "map_builder",
            "path": "scratch/build_codemap_exp460_antigravity.py",
            "role": "Caller: Task 1 validation. Effect: rebuilds and validates the bounded governance-only Exp460 graph and triplet fingerprints.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("scratch/build_codemap_exp460_antigravity.py", "validate_map_antigravity")],
        },
        {
            "id": "future_predecessor_capture",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py",
            "role": "Caller: future Task 2 scientific implementation. Effect: will capture frozen Exp459 authority into Exp460-owned contracts.",
            "tests": [],
            "evidence": [],
            "status": "future_planned",
        },
        {
            "id": "future_rule_evaluation",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py",
            "role": "Caller: future Tasks 3-4. Effect: will compare the four rules and derive the strict fallback decision from frozen Exp459 evidence.",
            "tests": [],
            "evidence": [],
            "status": "future_planned",
        },
        {
            "id": "future_artifacts_cli",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py",
            "role": "Caller: future Tasks 4-6. Effect: will expose evaluate and inspect only after Task 5 implementation and approval boundaries are met.",
            "tests": [],
            "evidence": [],
            "status": "future_planned",
        },
    ]
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "10/10 엄격 복귀")},
        {"from": "approved_design", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "현재 Exp459 자료에서는 1번과 2번을 만족하지 못하므로 엄격 규칙은 baseline을 선택한다")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 1 — 문서, 진행 대시보드, 코드맵")},
        {"from": "implementation_plan", "to": "research_state", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Exp460을 non-promotion 실험으로 ledger에 추가")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_plan_dashboard_and_map_antigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity")},
        {"from": "governance_regression", "to": "champion_ledger", "type": "tests", "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity")},
        {"from": "map_builder", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity("scratch/build_codemap_exp460_antigravity.py", "validate_map_antigravity")},
        {"from": "research_state", "to": "future_predecessor_capture", "type": "governs", "evidence": evidence_antigravity("RESEARCH_STATE.json", "\"ecg_strict_fallback\"")},
        {"from": "future_predecessor_capture", "to": "future_rule_evaluation", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "capture_exp459_fallback_authority_antigravity()")},
        {"from": "future_rule_evaluation", "to": "future_artifacts_cli", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "evaluate --reviewed-commit LOWER40HEX")},
    ]
    graph = {
        "schema": "exp460_task1_governance_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_parent_commit,
        "unknown_relationships": [],
        "nodes": nodes,
        "edges": edges,
    }
    lock = {
        "schema": "exp460_task1_governance_codemap_lock_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_parent_commit,
        "represented_paths": [
            "docs/codemap/codemap_exp460_antigravity.html",
            "docs/codemap/codemap_exp460_antigravity.json",
            "docs/codemap/codemap_exp460_antigravity.lock",
        ],
        "tracked_fingerprints": tracked_fingerprints_antigravity(),
        "module_fingerprints": module_fingerprints_antigravity(),
        "working_tree_dirty": False,
    }
    return graph, lock, html_antigravity(graph)


def validate_map_antigravity(graph: dict[str, object], lock: dict[str, object], rendered: str) -> None:
    if graph["schema"] != "exp460_task1_governance_codemap_v1":
        raise ValueError("unexpected graph schema")
    if lock["schema"] != "exp460_task1_governance_codemap_lock_v1":
        raise ValueError("unexpected lock schema")
    if graph["source_parent_commit"] != lock["source_parent_commit"]:
        raise ValueError("source_parent_commit mismatch")
    escaped_payload = html_module.escape(canonical_bytes_antigravity(graph).decode("utf-8"))
    if escaped_payload not in rendered:
        raise ValueError("rendered HTML is not in sync with graph JSON")
    node_ids = set()
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id in node_ids:
            raise ValueError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        for evidence in node.get("evidence", []):
            if not evidence_resolves_antigravity(evidence["path"], evidence["symbol"]):
                raise ValueError(f"unresolved evidence: {evidence}")
        if node.get("status") != "future_planned":
            if not (ROOT_ANTIGRAVITY / node["path"]).is_file():
                raise ValueError(f"missing current node path: {node['path']}")
    if len(graph["nodes"]) > 20:
        raise ValueError("graph exceeds 20 nodes")
    for edge in graph["edges"]:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            raise ValueError(f"edge references unknown node: {edge}")
        if not evidence_resolves_antigravity(edge["evidence"]["path"], edge["evidence"]["symbol"]):
            raise ValueError(f"unresolved edge evidence: {edge['evidence']}")
    expected_fingerprints = tracked_fingerprints_antigravity()
    if lock["tracked_fingerprints"] != expected_fingerprints:
        raise ValueError("tracked_fingerprints mismatch")
    if lock["module_fingerprints"] != module_fingerprints_antigravity():
        raise ValueError("module_fingerprints mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source_parent_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    generated_at = "2026-08-19T00:00:00Z"
    graph, lock, rendered = build_map_antigravity(source_parent_commit, generated_at)
    if args.validate_only:
        validate_map_antigravity(graph, lock, rendered)
        return
    JSON_PATH_ANTIGRAVITY.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOCK_PATH_ANTIGRAVITY.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    HTML_PATH_ANTIGRAVITY.write_text(rendered + "\n", encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    main()
