#!/usr/bin/env python3
"""Build and validate the Task 4 Exp456 candidate code map."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp456_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp456_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp456_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp456_antigravity.html",
    "docs/codemap/codemap_exp456_antigravity.json",
    "docs/codemap/codemap_exp456_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "RESEARCH_STATE.json", "EXPERIMENT_CATALOG.json", "CHAMPION_LEDGER.json",
    "docs/EXP456_PREREGISTRATION_antigravity_KO.html", "docs/EXP456_RUNBOOK_antigravity_KO.html",
    "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html", "docs/EXP456_DASHBOARD_antigravity.html",
    "docs/superpowers/specs/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity-design.html",
    "docs/superpowers/plans/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity.html",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/config_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/contracts_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/authority_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/features_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/router_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/evaluation_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/artifacts_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/replay_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/dashboard_export_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/cli_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_config_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_authority_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_features_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_router_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_controlled_fork_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_dataset_key_join_antigravity.py",
    "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_cli_antigravity.py",
    "scratch/build_codemap_exp456_antigravity.py",
    "scratch/run_exp456_exp455_dataset_key_order_repair_antigravity.py",
)
FUTURE_RELATIONSHIPS_ANTIGRAVITY = "Task 4 implements the closed CLI and outside-CWD launcher; no Exp456 source relationship remains future or unknown."


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


def fingerprints_antigravity() -> dict[str, str]:
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
    payload = html.escape(canonical_bytes_antigravity(graph).decode("utf-8"))
    nodes = "".join(f'<li><code>{html.escape(str(node.get("path", "unimplemented")))}</code> — {html.escape(str(node["role"]))}</li>' for node in graph["nodes"])
    return "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>Exp456 Task 4 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main><h1>Exp456 Task 4 candidate 코드맵</h1><p>현재 map은 Exp455 controlled-fork foundations, Task 3 dataset-key evaluation, 그리고 Task 4 closed CLI/launcher를 표시한다. 모든 CLI caller, effect, test 관계는 concrete evidence로 연결되며 future source는 없다.</p><h2>노드</h2><ul>" + nodes + "</ul><h2>Machine-readable graph</h2><pre id=\"codemap-json-antigravity\" type=\"application/json\">" + payload + "</pre></main></body></html>"


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT_ANTIGRAVITY, check=False).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must end in Z")
    test = "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_governance_antigravity.py"
    package = "experiments/exp456_exp455_dataset_key_order_repair_antigravity"
    foundation_tests = {
        "config": f"{package}/tests/test_config_antigravity.py",
        "contracts": f"{package}/tests/test_contracts_antigravity.py",
        "authority": f"{package}/tests/test_authority_antigravity.py",
        "features": f"{package}/tests/test_features_antigravity.py",
        "router": f"{package}/tests/test_router_antigravity.py",
        "controlled_fork": f"{package}/tests/test_controlled_fork_antigravity.py",
    }
    builder = "scratch/build_codemap_exp456_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity-design.html", "role": "Caller: Task 3 join review. Effect: freezes dataset-key repair as the only scientific change.", "tests": [test], "evidence": [evidence_antigravity("docs/superpowers/specs/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity-design.html", "Exp456")]},
        {"id": "implementation_plan", "path": "docs/superpowers/plans/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity.html", "role": "Caller: Task 4 candidate review. Effect: sequences the completed closed CLI and later approval gate.", "tests": [test], "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity.html", "Task 4:")]},
        {"id": "preregistration", "path": "docs/EXP456_PREREGISTRATION_antigravity_KO.html", "role": "Caller: Task 4 governance review. Effect: records frozen identity and the continuing execution firewall.", "tests": [test], "evidence": [evidence_antigravity("docs/EXP456_PREREGISTRATION_antigravity_KO.html", "planned_awaiting_implementation")]},
        {"id": "runbook", "path": "docs/EXP456_RUNBOOK_antigravity_KO.html", "role": "Caller: Task 4 candidate review. Effect: blocks official commands until Task 5 approval.", "tests": [test], "evidence": [evidence_antigravity("docs/EXP456_RUNBOOK_antigravity_KO.html", "official_evaluate_started=false")]},
        {"id": "implementation_report", "path": "docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html", "role": "Caller: Task 4 review. Effect: records map preflight, CLI RED/GREEN, full verification, and non-actions.", "tests": [test], "evidence": [evidence_antigravity("docs/EXP456_IMPLEMENTATION_REPORT_antigravity.html", "Map preflight")]},
        {"id": "dashboard", "path": "docs/EXP456_DASHBOARD_antigravity.html", "role": "Caller: Task 4 status review. Effect: shows Task 4 candidate progress and only not_available official performance.", "tests": [test], "evidence": [evidence_antigravity("docs/EXP456_DASHBOARD_antigravity.html", "Task 4/5")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Caller: governance review. Effect: records Exp456 planned lifecycle truth.", "tests": [test], "evidence": [evidence_antigravity("RESEARCH_STATE.json", "exp456")]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Caller: governance review. Effect: registers the planned Exp456 document and lifecycle surface.", "tests": [test], "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "Exp456")]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "Caller: champion review. Effect: records no champion or pending-candidate change.", "tests": [test], "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "Exp456")]},
        {"id": "governance_regression", "path": test, "role": "Caller: Task 4 verification. Effect: rejects candidate-truth drift, invented official performance, stale map progress, and Exp455/champion mutation.", "tests": [test], "evidence": [evidence_antigravity(test, "GovernanceAntigravityTest")]},
        {"id": "map_builder", "path": builder, "role": "Caller: Task 4 map validation. Effect: validates current evidence, concrete CLI boundaries, fingerprints, and HTML/JSON parity.", "tests": [test], "evidence": [evidence_antigravity(builder, "validate_map_antigravity")]},
    ]
    for node_id, role, symbol in (
        ("foundation_config", "Caller: authority capture and later evaluation. Effect: freezes Exp456 identity while retaining Exp455 scientific gates and Exp454 evidence hashes.", "FROZEN_CONFIG_ANTIGRAVITY"),
        ("foundation_contracts", "Caller: authority, features, router, and later evaluation. Effect: defines immutable routing authority, feature, rule, and result boundary shapes.", "FrozenRoutingAuthorityAntigravity"),
        ("foundation_authority", "Caller: Task 3 evaluation. Effect: captures Exp454 predecessor authority without TEST labels and preserves its source order.", "capture_routing_authority_antigravity"),
        ("foundation_features", "Caller: Task 3 evaluation and router. Effect: derives label-free inputs in ECG5000×5 then ChlorineConcentration×3 public order.", "build_label_free_features_antigravity"),
        ("foundation_router", "Caller: Task 3 evaluation. Effect: preserves family-specific LODO rules, threshold handling, objective, and tie break.", "leave_one_dataset_out_antigravity"),
    ):
        name = node_id.removeprefix("foundation_")
        nodes.append({"id": node_id, "path": f"{package}/{name}_antigravity.py", "role": role, "tests": [foundation_tests[name], foundation_tests["controlled_fork"]], "evidence": [evidence_antigravity(f"{package}/{name}_antigravity.py", symbol)]})
    for name, path in foundation_tests.items():
        nodes.append({"id": f"{name}_regression", "path": path, "role": f"Caller: Task 3 verification. Effect: rejects foundation drift for {name}.", "tests": [path], "evidence": [evidence_antigravity(path, "AntigravityTest")]})
    task3_sources = (
        ("evaluation", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/evaluation_antigravity.py", "Caller: Task 3 artifact publication and replay. Effect: joins frozen targets, features, and decisions by exact dataset key in feature/public order.", "align_targets_by_dataset_antigravity", ["experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_evaluation_antigravity.py", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_dataset_key_join_antigravity.py"]),
        ("artifacts", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/artifacts_antigravity.py", "Caller: Task 3 result publication. Effect: writes the unchanged eight-leaf replay evidence inventory.", "publish_evaluation_antigravity", ["experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_artifacts_antigravity.py", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_replay_antigravity.py"]),
        ("replay", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/replay_antigravity.py", "Caller: Task 3 inspection. Effect: reconstructs frozen evidence and re-evaluates it through the dataset-key join contract.", "inspect_replay_at_antigravity", ["experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_replay_antigravity.py"]),
        ("dashboard_export", "experiments/exp456_exp455_dataset_key_order_repair_antigravity/dashboard_export_antigravity.py", "Caller: Task 3 publication. Effect: exposes only approved aggregate comparison and family fields.", "export_aggregate_dashboard_antigravity", ["experiments/exp456_exp455_dataset_key_order_repair_antigravity/tests/test_dashboard_export_antigravity.py"]),
    )
    for node_id, path, role, symbol, tests in task3_sources:
        nodes.append({"id": node_id, "path": path, "role": role, "tests": tests, "evidence": [evidence_antigravity(path, symbol)]})
    for path in sorted({test for _node_id, _path, _role, _symbol, tests in task3_sources for test in tests}):
        nodes.append({"id": Path(path).stem, "path": path, "role": "Caller: Task 3 verification. Effect: rejects dataset-key join, publication, replay, or dashboard aggregate drift.", "tests": [path], "evidence": [evidence_antigravity(path, "Antigravity")]})
    cli_test = f"{package}/tests/test_cli_antigravity.py"
    cli_path = f"{package}/cli_antigravity.py"
    launcher = "scratch/run_exp456_exp455_dataset_key_order_repair_antigravity.py"
    nodes.extend([
        {"id": "cli", "path": cli_path, "role": "Caller: outside-CWD launcher. Effect: accepts exactly reviewed evaluate or bare inspect; evaluates authority→features→LODO→evaluation→publication and emits aggregate JSON, or replays and emits the same aggregate.", "tests": [cli_test], "evidence": [evidence_antigravity(cli_path, "main_antigravity")]},
        {"id": "launcher", "path": launcher, "role": "Caller: user command outside repository CWD. Effect: resolves the repository package and invokes only the closed CLI main.", "tests": [cli_test], "evidence": [evidence_antigravity(launcher, "main_antigravity")]},
        {"id": "cli_regression", "path": cli_test, "role": "Caller: Task 4 verification. Effect: rejects parser expansion and validates injected evaluate/replay ordering, aggregate output, and no official test root.", "tests": [cli_test], "evidence": [evidence_antigravity(cli_test, "CliAntigravityTests")]},
    ])
    unknowns: list[dict[str, str]] = []
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity("docs/EXP456_PREREGISTRATION_antigravity_KO.html", "dataset")},
        {"from": "implementation_plan", "to": "research_state", "type": "governs", "evidence": evidence_antigravity("RESEARCH_STATE.json", "implementation_verified_awaiting_official_evaluation")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(test, "test_candidate_governance_dashboard_and_map_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": evidence_antigravity(test, "builder.validate_map_antigravity")},
    ]
    for name, path in foundation_tests.items():
        target = f"foundation_{name}" if name != "controlled_fork" else "foundation_config"
        edges.append({"from": f"{name}_regression", "to": target, "type": "tests", "evidence": evidence_antigravity(path, "AntigravityTest")})
    for target in ("foundation_contracts", "foundation_authority", "foundation_features", "foundation_router"):
        edges.append({"from": "controlled_fork_regression", "to": target, "type": "tests", "evidence": evidence_antigravity(foundation_tests["controlled_fork"], "ControlledForkAntigravityTest")})
    for node_id, _path, _role, _symbol, tests in task3_sources:
        for test in tests:
            edges.append({"from": Path(test).stem, "to": node_id, "type": "tests", "evidence": evidence_antigravity(test, "Antigravity")})
    for target in ("foundation_authority", "foundation_features", "foundation_router"):
        edges.append({"from": "evaluation", "to": target, "type": "calls", "evidence": evidence_antigravity(f"{package}/evaluation_antigravity.py", "evaluate_routing_antigravity")})
    edges.extend([
        {"from": "artifacts", "to": "evaluation", "type": "consumes", "evidence": evidence_antigravity(f"{package}/artifacts_antigravity.py", "evidence_for_evaluation_antigravity")},
        {"from": "artifacts", "to": "dashboard_export", "type": "calls", "evidence": evidence_antigravity(f"{package}/artifacts_antigravity.py", "export_aggregate_dashboard_antigravity")},
        {"from": "replay", "to": "evaluation", "type": "calls", "evidence": evidence_antigravity(f"{package}/replay_antigravity.py", "evaluate_routing_antigravity")},
        {"from": "launcher", "to": "cli", "type": "calls", "evidence": evidence_antigravity(launcher, "main_antigravity")},
        {"from": "cli_regression", "to": "cli", "type": "tests", "evidence": evidence_antigravity(cli_test, "CliAntigravityTests")},
        {"from": "cli_regression", "to": "launcher", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_launcher_is_outside_cwd_safe_and_only_calls_package_cli_antigravity")},
        {"from": "cli", "to": "foundation_authority", "type": "calls", "evidence": evidence_antigravity(cli_path, "capture_routing_authority_antigravity")},
        {"from": "cli", "to": "foundation_features", "type": "calls", "evidence": evidence_antigravity(cli_path, "build_label_free_features_antigravity")},
        {"from": "cli", "to": "foundation_router", "type": "calls", "evidence": evidence_antigravity(cli_path, "leave_one_dataset_out_antigravity")},
        {"from": "cli", "to": "evaluation", "type": "calls", "evidence": evidence_antigravity(cli_path, "evaluate_routing_antigravity")},
        {"from": "cli", "to": "artifacts", "type": "calls", "evidence": evidence_antigravity(cli_path, "publish_evaluation_antigravity")},
        {"from": "cli", "to": "replay", "type": "calls", "evidence": evidence_antigravity(cli_path, "inspect_replay_at_antigravity")},
        {"from": "cli", "to": "dashboard_export", "type": "calls", "evidence": evidence_antigravity(cli_path, "export_aggregate_dashboard_antigravity")},
    ])
    graph: dict[str, object] = {"schema": "exp456_task4_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp456_task4_codemap_lock_v1", "source_parent_commit": source_commit, "working_tree_dirty": False, "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "module_fingerprints": module_fingerprints_antigravity(), "tracked_fingerprints": fingerprints_antigravity()}
    return graph, lock, html_antigravity(graph)


def validate_map_antigravity(graph: object, lock: object, rendered_html: str) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes, edges, unknowns = graph.get("nodes"), graph.get("edges"), graph.get("unknown_relationships")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(unknowns, list):
        raise ValueError("invalid topology")
    ids: set[str] = set(); future_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in ids:
            raise ValueError("invalid node")
        ids.add(node["id"])
        if node.get("status") == "future_unknown":
            future_ids.add(node["id"])
            if node.get("evidence") != [] or (ROOT_ANTIGRAVITY / str(node.get("path"))).exists():
                raise ValueError("future source evidence claimed")
        else:
            evidence = node.get("evidence")
            if not isinstance(evidence, list) or not evidence or not (ROOT_ANTIGRAVITY / str(node.get("path"))).is_file():
                raise ValueError("missing known evidence")
            for item in evidence: validate_evidence_antigravity(item)
    pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    actual_future: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids:
            raise ValueError("invalid edge")
        validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        involves_future = edge["from"] in future_ids or edge["to"] in future_ids
        if involves_future:
            if edge.get("type") != "future_unknown" or pair not in pairs: raise ValueError("unmarked future relationship")
            actual_future.add(pair)
        elif edge.get("type") == "future_unknown": raise ValueError("known edge marked future")
    if pairs != actual_future: raise ValueError("unknown relationship mismatch")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY) or lock.get("working_tree_dirty") is not False: raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != fingerprints_antigravity() or lock.get("module_fingerprints") != module_fingerprints_antigravity(): raise ValueError("fingerprint drift")
    embedded = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, re.DOTALL)
    if not embedded or canonical_bytes_antigravity(strict_json_antigravity(html.unescape(embedded.group(1)))) != canonical_bytes_antigravity(graph): raise ValueError("HTML/JSON mismatch")


def validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str): raise ValueError("invalid evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"): raise ValueError("evidence not found")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-commit"); parser.add_argument("--generated-at", default="2026-08-17T00:00:00Z"); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8")); return
    if not args.source_commit: parser.error("--source-commit is required")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at)
    JSON_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(graph) + b"\n"); LOCK_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(lock) + b"\n"); HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__": main_antigravity()
