#!/usr/bin/env python3
"""Build and validate the Exp449 Task 2 candidate and method-audit map."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "docs/codemap"
JSON_PATH = MAP_DIR / "codemap_exp449_antigravity.json"
LOCK_PATH = MAP_DIR / "codemap_exp449_antigravity.lock"
HTML_PATH = MAP_DIR / "codemap_exp449_antigravity.html"
SCANNED_TOP_LEVEL = (".superpowers", "docs", "experiments", "operational", "reports", "scratch", "scripts", "vercel-dashboard")
EXCLUDED_PARTS = {".git", ".worktrees", "vendor", "node_modules", "build", "cache", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "outputs", "results"}
EDGE_TYPES = {"imports", "reads", "governs", "future_unknown"}


def strict_json_antigravity(text: str) -> Any:
    def reject_duplicate_antigravity(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicate_antigravity)


def tracked_paths_antigravity() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        parts = relative.parts
        if any(part in EXCLUDED_PARTS for part in parts) or parts[:2] == ("docs", "codemap"):
            continue
        if len(parts) == 1 or parts[0] in SCANNED_TOP_LEVEL:
            paths.append(relative)
    return sorted(paths)


def fingerprint_modules_antigravity() -> dict[str, str]:
    grouped: dict[str, list[Path]] = {"root": []}
    for name in SCANNED_TOP_LEVEL:
        grouped[name] = []
    for relative in tracked_paths_antigravity():
        key = "root" if len(relative.parts) == 1 else relative.parts[0]
        if key in grouped:
            grouped[key].append(relative)
    result: dict[str, str] = {}
    for key, paths in grouped.items():
        digest = hashlib.sha256()
        for relative in sorted(paths):
            contents = (ROOT / relative).read_bytes()
            encoded_path = relative.as_posix().encode("utf-8")
            digest.update(encoded_path + b"\0" + str(len(contents)).encode("ascii") + b"\0" + contents + b"\0")
        result[key] = digest.hexdigest()
    return result


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return JSON graph, lock, and self-contained HTML for the same graph."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    nodes: list[dict[str, object]] = [
        {"id": "research_control", "path": "RESEARCH_STATE.json", "role": "Planned Exp449 active state with zero label access and closed official lifecycle.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("RESEARCH_STATE.json", '"exp449"')]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Registered planned Exp449 identity and immutable predecessor relationship.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", '"id": "Exp449"')]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "Non-promotion ledger retaining the unchanged pending candidate.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", '"experiment": "Exp449"')]},
        {"id": "exp448_predecessor", "path": "docs/EXP448_IMPLEMENTATION_REPORT_antigravity.md", "role": "Immutable stable-26 predecessor aggregate truth.", "tests": [], "evidence": [evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "immutable Exp448 stable-26")]},
        {"id": "exp449_preregistration", "path": "docs/EXP449_PREREGISTRATION_antigravity_KO.html", "role": "Frozen candidates, settings, success gates, label boundary, CPU policy, and no-promotion contract.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "official_label_sessions = 1")]},
        {"id": "exp449_runbook", "path": "docs/EXP449_RUNBOOK_antigravity_KO.html", "role": "Single-use future command order, seven thread variables, stop rules, and dashboard firewall.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_RUNBOOK_antigravity_KO.html", "AEON_NUM_THREADS=1")]},
        {"id": "exp449_report", "path": "docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "role": "Governance-only implementation report with no official execution claim.", "tests": [], "evidence": [evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "Task 1 governance-only report")]},
        {"id": "candidate_registry", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "role": "Frozen eight-family, two-candidate registry and pinned environment authority.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "immutable_contracts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "role": "Frozen slots exact-type authorities for candidates, audits, bundles, baselines, and metrics.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "class MethodAuditAntigravity")]},
        {"id": "canonical_artifacts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "role": "Canonical JSON/JSONL and create-only no-follow artifact inventory boundary.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "def artifact_map_bytes_antigravity")]},
        {"id": "method_audit", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "role": "Subprocess-only fixed-interpreter audit binding versions, source SHA256 values, and frozen public methods.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "def audit_methods_antigravity")]},
        {"id": "future_score_prediction", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "role": "Future score/prediction boundary; no source module or call exists at Task 1.", "tests": ["future tests/test_scoring_antigravity.py", "future tests/test_prediction_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "scoring/prediction")]},
        {"id": "future_evaluation_replay", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "role": "Future one-session evaluation/replay boundary; no source module or call exists at Task 1.", "tests": ["future tests/test_evaluation_antigravity.py", "future tests/test_replay_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "evaluation/replay")]},
        {"id": "future_cli", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/cli_antigravity.py", "role": "Future closed CLI/launcher boundary; no source module or call exists at Task 1.", "tests": ["future tests/test_cli_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "CLI")]},
        {"id": "dashboard_presentation", "path": "existing owner-only dashboard", "role": "Future aggregate-only presentation boundary; it is not a scientific gate and no site source is mapped.", "tests": [], "evidence": [evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "aggregate-only owner-only snapshot")]},
    ]
    edges: list[dict[str, object]] = [
        {"from": "research_control", "to": "exp449_preregistration", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_PREREGISTRATION_antigravity_KO.html")},
        {"from": "research_control", "to": "exp449_runbook", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_RUNBOOK_antigravity_KO.html")},
        {"from": "research_control", "to": "exp449_report", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_IMPLEMENTATION_REPORT_antigravity.html")},
        {"from": "experiment_catalog", "to": "exp449_preregistration", "type": "reads", "evidence": evidence_antigravity("EXPERIMENT_CATALOG.json", "EXP449_PREREGISTRATION_antigravity_KO.html")},
        {"from": "champion_ledger", "to": "research_control", "type": "governs", "evidence": evidence_antigravity("CHAMPION_LEDGER.json", "pending_candidate_changed")},
        {"from": "exp449_preregistration", "to": "exp448_predecessor", "type": "reads", "evidence": evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "immutable Exp448 stable-26")},
        {"from": "exp449_preregistration", "to": "candidate_registry", "type": "governs", "evidence": evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "global_fp_max = 314")},
        {"from": "method_audit", "to": "candidate_registry", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "config_antigravity import")},
        {"from": "method_audit", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "contracts_antigravity import")},
        {"from": "exp449_preregistration", "to": "future_score_prediction", "type": "future_unknown", "evidence": evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "scoring/prediction")},
        {"from": "exp449_runbook", "to": "future_evaluation_replay", "type": "future_unknown", "evidence": evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "evaluation/replay")},
        {"from": "exp449_runbook", "to": "future_cli", "type": "future_unknown", "evidence": evidence_antigravity("docs/EXP449_IMPLEMENTATION_REPORT_antigravity.html", "CLI")},
        {"from": "exp449_preregistration", "to": "dashboard_presentation", "type": "future_unknown", "evidence": evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "aggregate-only owner-only snapshot")},
    ]
    graph: dict[str, object] = {"generated_at": generated_at, "generated_from_commit": source_commit, "scope": ["Exp449 Task-2 candidate registry, immutable contracts, canonical artifacts, fixed method audit, and future product boundaries"], "nodes": nodes, "edges": edges, "flows": [{"trigger": "Task-1 governance freeze", "steps": ["exp449_preregistration", "exp449_runbook", "exp449_report", "research_control", "experiment_catalog", "champion_ledger"], "outcome": "Planned Exp449 truth only: no official command, label access, promotion, or routing."}, {"trigger": "Task-2 contract freeze", "steps": ["exp449_preregistration", "candidate_registry", "immutable_contracts", "canonical_artifacts", "method_audit"], "outcome": "The exact candidate inventory and fixed environment authority are implemented without prediction, readiness, evaluation, or official lifecycle execution."}, {"trigger": "Future implementation boundary", "steps": ["candidate_registry", "future_score_prediction", "future_evaluation_replay", "future_cli", "dashboard_presentation"], "outcome": "Later product relations remain explicitly future/unknown until their source modules and tests exist."}], "unknown_relationships": ["Prediction, readiness, source authority, score, evaluation, replay, CLI, and dashboard product-boundary relationships remain future/unknown; no calls, routes, databases, queues, or dashboard source edges are claimed."], "unknowns": ["No Exp449 official lifecycle or prediction/readiness source exists at Task 2.", "The existing owner-only dashboard has no mapped repository source and is presentation-only."]}
    lock: dict[str, object] = {"current_commit": source_commit, "working_tree_dirty": False, "generated_at": generated_at, "scanned_scope": ["tracked root files including AGENTS.md", *SCANNED_TOP_LEVEL], "excluded_directories": sorted(EXCLUDED_PARTS | {"docs/codemap"}), "fingerprint_algorithm": "sha256-v1: sort tracked repo-relative paths by module; append UTF-8 path, NUL, decimal byte length, NUL, current bytes, NUL; docs/codemap is excluded to prevent self-reference.", "module_fingerprints": fingerprint_modules_antigravity(), "changed_modules_since_prior_lock": ["docs", "experiments", "scratch"]}
    payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    html = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Exp449 Code Map</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.55 system-ui}pre{white-space:pre-wrap;overflow:auto;background:#eef2f7;padding:1rem}</style></head><body><main><h1>Exp449 Task-2 Code Map</h1><p>Candidate, contract, artifact, and method-audit boundaries are implemented; later product boundaries remain unknown, not calls.</p><script id=\"codemap-data\" type=\"application/json\">" + payload + "</script><pre id=\"graph\"></pre></main><script>const data=JSON.parse(document.getElementById('codemap-data').textContent);document.getElementById('graph').textContent=JSON.stringify(data,null,2);</script></body></html>"
    return graph, lock, html


def validate_map_antigravity(graph: object, lock: object, html: str) -> None:
    """Reject missing evidence, unresolved refs, >20 primary nodes, and fingerprint drift."""
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    flows = graph.get("flows")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(flows, list) or len(nodes) > 20:
        raise ValueError("invalid primary node/edge/flow topology")
    node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if len(node_ids) != len(nodes) or None in node_ids:
        raise ValueError("nodes require unique ids")
    def check_evidence(item: object) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str):
            raise ValueError("missing evidence")
        candidate = ROOT / item["path"]
        if not candidate.is_file() or item["symbol"] not in candidate.read_text(encoding="utf-8"):
            raise ValueError(f"unresolved evidence: {item}")
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("evidence"), list):
            raise ValueError("node requires evidence")
        for item in node["evidence"]:
            check_evidence(item)
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in node_ids or edge.get("to") not in node_ids or edge.get("type") not in EDGE_TYPES:
            raise ValueError("unsupported or unresolved edge")
        check_evidence(edge.get("evidence"))
        if str(edge["from"]).startswith("future_"):
            raise ValueError("no future product-source edge is permitted")
    for flow in flows:
        if not isinstance(flow, dict) or not isinstance(flow.get("steps"), list) or any(step not in node_ids for step in flow["steps"]):
            raise ValueError("unresolved flow reference")
    match = re.search(r'<script id="codemap-data" type="application/json">(.*?)</script>', html)
    if not match or strict_json_antigravity(match.group(1)) != graph:
        raise ValueError("HTML does not embed the exact JSON graph")
    if lock.get("module_fingerprints") != fingerprint_modules_antigravity():
        raise ValueError("fingerprint drift")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(strict_json_antigravity(JSON_PATH.read_text(encoding="utf-8")), strict_json_antigravity(LOCK_PATH.read_text(encoding="utf-8")), HTML_PATH.read_text(encoding="utf-8"))
        return
    if not args.source_commit:
        parser.error("--source-commit is required unless --validate-only is used")
    generated_at = args.generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    graph, lock, html = build_map_antigravity(args.source_commit, generated_at)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HTML_PATH.write_text(html + "\n", encoding="utf-8")
    validate_map_antigravity(graph, lock, html)


if __name__ == "__main__":
    main_antigravity()
