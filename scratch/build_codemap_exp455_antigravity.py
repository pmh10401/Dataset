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
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/router_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/evaluation_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/artifacts_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/replay_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/dashboard_export_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/cli_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_authority_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_features_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_router_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_config_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp455_exp454_conservative_label_free_routing_antigravity/tests/test_cli_antigravity.py",
    "scratch/run_exp455_exp454_conservative_label_free_routing_antigravity.py",
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
        '<h1>Exp455 Task 5 코드맵</h1><p>Exp454 authority, label-free 특징, 가족별 LODO router, 48행 evaluation, create-only artifact, semantic replay, aggregate dashboard와 닫힌 CLI 경계를 분리한다. '
        'label, confusion, dataset variant number는 router inputs가 아니며 holdout outcome은 rule 선택에 쓰지 않는다.</p>'
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
    router = f"{package}/router_antigravity.py"
    router_test = f"{package}/tests/test_router_antigravity.py"
    evaluation = f"{package}/evaluation_antigravity.py"
    evaluation_test = f"{package}/tests/test_evaluation_antigravity.py"
    artifacts = f"{package}/artifacts_antigravity.py"
    artifacts_test = f"{package}/tests/test_artifacts_antigravity.py"
    replay = f"{package}/replay_antigravity.py"
    replay_test = f"{package}/tests/test_replay_antigravity.py"
    dashboard_export = f"{package}/dashboard_export_antigravity.py"
    dashboard_export_test = f"{package}/tests/test_dashboard_export_antigravity.py"
    cli = f"{package}/cli_antigravity.py"
    cli_test = f"{package}/tests/test_cli_antigravity.py"
    launcher = "scratch/run_exp455_exp454_conservative_label_free_routing_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html", "role": "Caller: Task 1 governance. Effect: fixes Exp455 scope, candidates, gates, and label-free firewall.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html", "Exp455")]},
        {"id": "implementation_plan", "path": plan, "role": "Caller: Tasks 1–6. Effect: freezes sequential TDD and official-evaluation boundaries.", "tests": [governance_test], "evidence": [_evidence_antigravity(plan, "Task 1: Governance")]},
        {"id": "preregistration", "path": "docs/EXP455_PREREGISTRATION_antigravity_KO.html", "role": "Caller: governance review. Effect: records planned-only frozen identity and firewall.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP455_PREREGISTRATION_antigravity_KO.html", "Exp455")]},
        {"id": "runbook", "path": "docs/EXP455_RUNBOOK_antigravity_KO.html", "role": "Caller: future operators. Effect: prohibits Task 1 official execution and TEST-label access.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP455_RUNBOOK_antigravity_KO.html", "Task 1")]},
        {"id": "dashboard", "path": "docs/EXP455_DASHBOARD_antigravity.html", "role": "Caller: reviewed implementation status display. Effect: exposes 8/8 label-free features, Tasks 5/6, clean independent reviews, 8/8 folds, and still-not-started official result.", "tests": [governance_test, dashboard_export_test], "evidence": [_evidence_antigravity("docs/EXP455_DASHBOARD_antigravity.html", "Tasks 5/6")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Caller: governance review. Effect: records Exp455 as planned, non-promoted, and not officially run.", "tests": [governance_test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "\"exp455\"")]},
        {"id": "config", "path": f"{package}/config_antigravity.py", "role": "Caller: future authority, router, and evaluation. Effect: freezes approved identity, scope, gates, evidence hashes, and official Python.", "tests": [config_test, governance_test], "evidence": [_evidence_antigravity(f"{package}/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "contracts", "path": f"{package}/contracts_antigravity.py", "role": "Caller: future authority, router, and evaluation. Effect: freezes label-free feature, authority, rule, decision, and result records.", "tests": [contracts_test, governance_test], "evidence": [_evidence_antigravity(f"{package}/contracts_antigravity.py", "DatasetRoutingFeaturesAntigravity")]},
        {"id": "governance_regression", "path": governance_test, "role": "Caller: Task 1 focused verification. Effect: rejects missing planned governance and unmarked downstream boundaries.", "tests": [governance_test], "evidence": [_evidence_antigravity(governance_test, "class GovernanceAntigravityTest")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp455_antigravity.py", "role": "Caller: Task 1 map validation. Effect: checks evidence, future boundaries, fingerprints, and HTML/JSON parity.", "tests": [governance_test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp455_antigravity.py", "validate_map_antigravity")]},
        {"id": "authority", "path": authority, "role": "Caller: Task 2 feature builder and future router/evaluation. Effect: verifies Exp454 12 leaves and combines Exp446 stable-26 plus Exp404 Hard coordinates without labels.", "tests": [authority_test], "evidence": [_evidence_antigravity(authority, "capture_routing_authority_antigravity")]},
        {"id": "features", "path": features, "role": "Caller: Task 3 router. Effect: emits exactly eight ECG5000-then-ChlorineConcentration integer feature rows without confusion or labels.", "tests": [features_test], "evidence": [_evidence_antigravity(features, "build_label_free_features_antigravity")]},
        {"id": "router", "path": router, "role": "Caller: Task 3 fold selection and Task 4 replay. Effect: chooses deterministic family LODO rules from allowed integer features without the held-out outcome.", "tests": [router_test, replay_test], "evidence": [_evidence_antigravity(router, "leave_one_dataset_out_antigravity")]},
        {"id": "evaluation", "path": evaluation, "role": "Caller: Task 4 artifacts and dashboard export. Effect: sums each of 48 rows once for four comparisons and checks gates without activating routing.", "tests": [evaluation_test, artifacts_test, replay_test, dashboard_export_test], "evidence": [_evidence_antigravity(evaluation, "evaluate_routing_antigravity")]},
        {"id": "artifacts", "path": artifacts, "role": "Caller: injected temporary publication and replay. Effect: writes exactly eight canonical create-only leaves with a self-excluding hash map.", "tests": [artifacts_test, replay_test], "evidence": [_evidence_antigravity(artifacts, "publish_evaluation_antigravity")]},
        {"id": "replay", "path": replay, "role": "Caller: Task 4 artifact inspection. Effect: captures frozen Task 2 authority once, compares all 48 rows plus eight label-free features, then re-fits LODO and recomputes aggregates without TEST labels.", "tests": [replay_test], "evidence": [_evidence_antigravity(replay, "inspect_replay_at_antigravity")]},
        {"id": "dashboard_export", "path": dashboard_export, "role": "Caller: artifact dashboard snapshot. Effect: limits output to aggregate method and family rows.", "tests": [dashboard_export_test], "evidence": [_evidence_antigravity(dashboard_export, "export_aggregate_dashboard_antigravity")]},
        {"id": "closed_cli", "path": cli, "role": "Caller: outside-CWD launcher and CLI regression. Effect: admits only reviewed evaluate or bare inspect; selected phases load lazily and official publication remains fixed-root only.", "tests": [cli_test], "evidence": [_evidence_antigravity(cli, "main_antigravity"), _evidence_antigravity(launcher, "main_antigravity")]},
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
        {"from": "router", "to": "features", "type": "reads", "evidence": _evidence_antigravity(router, "DatasetRoutingFeaturesAntigravity")},
        {"from": "router", "to": "contracts", "type": "reads", "evidence": _evidence_antigravity(router, "FoldDecisionAntigravity")},
        {"from": "router", "to": "evaluation", "type": "reads", "evidence": _evidence_antigravity(evaluation, "FoldDecisionAntigravity")},
        {"from": "evaluation", "to": "artifacts", "type": "reads", "evidence": _evidence_antigravity(artifacts, "publish_evaluation_antigravity")},
        {"from": "evaluation", "to": "dashboard_export", "type": "reads", "evidence": _evidence_antigravity(dashboard_export, "RoutingEvaluationAntigravity")},
        {"from": "artifacts", "to": "replay", "type": "reads", "evidence": _evidence_antigravity(replay, "_LEAVES_ANTIGRAVITY")},
        {"from": "router", "to": "replay", "type": "reads", "evidence": _evidence_antigravity(replay, "leave_one_dataset_out_antigravity")},
        {"from": "authority", "to": "replay", "type": "reads", "evidence": _evidence_antigravity(replay, "capture_routing_authority_antigravity")},
        {"from": "features", "to": "replay", "type": "reads", "evidence": _evidence_antigravity(replay, "build_label_free_features_antigravity")},
        {"from": "closed_cli", "to": "authority", "type": "reads", "evidence": _evidence_antigravity(cli, "capture_routing_authority_antigravity")},
        {"from": "closed_cli", "to": "features", "type": "reads", "evidence": _evidence_antigravity(cli, "build_label_free_features_antigravity")},
        {"from": "closed_cli", "to": "router", "type": "reads", "evidence": _evidence_antigravity(cli, "leave_one_dataset_out_antigravity")},
        {"from": "closed_cli", "to": "evaluation", "type": "reads", "evidence": _evidence_antigravity(cli, "evaluate_routing_antigravity")},
        {"from": "closed_cli", "to": "artifacts", "type": "reads", "evidence": _evidence_antigravity(cli, "publish_evaluation_antigravity")},
        {"from": "closed_cli", "to": "replay", "type": "reads", "evidence": _evidence_antigravity(cli, "inspect_replay_at_antigravity")},
    ]
    unknowns: list[dict[str, str]] = []
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
