#!/usr/bin/env python3
"""Build and validate the bounded Exp454 code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp454_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp454_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp454_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp454_antigravity.html",
    "docs/codemap/codemap_exp454_antigravity.json",
    "docs/codemap/codemap_exp454_antigravity.lock",
}
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "reads", "tests", "future_unknown"}
TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP454_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP454_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP454_RUNBOOK_antigravity_KO.html",
    "docs/superpowers/plans/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity.html",
    "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp454_antigravity.py",
)


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


def _tracked_paths_antigravity() -> list[Path]:
    return [Path(item) for item in TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY]


def _tracked_fingerprints_antigravity() -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((ROOT_ANTIGRAVITY / path).read_bytes()).hexdigest()
        for path in _tracked_paths_antigravity()
    }


def _module_fingerprints_antigravity() -> dict[str, str]:
    groups: dict[str, list[Path]] = {"root": []}
    for path in _tracked_paths_antigravity():
        groups.setdefault("root" if len(path.parts) == 1 else path.parts[0], []).append(path)
    result: dict[str, str] = {}
    for group, paths in sorted(groups.items()):
        digest = hashlib.sha256()
        for path in sorted(paths):
            data = (ROOT_ANTIGRAVITY / path).read_bytes()
            digest.update(path.as_posix().encode() + b"\0" + str(len(data)).encode() + b"\0" + data + b"\0")
        result[group] = digest.hexdigest()
    return result


def _evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def _html_antigravity(graph: dict[str, object]) -> str:
    payload = html_module.escape(_canonical_bytes_antigravity(graph).decode())
    items = "".join(
        f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Exp454 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}"
        "code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>"
        "<h1>Exp454 Task 5 코드맵</h1>"
        "<p>HTML governance, Task 2 foundations/normalize 경계, Task 3 runtime/sources/lifecycle/readiness, "
        "Task 4 evaluator/replay, 그리고 Task 5 closed CLI가 구현되었다. "
        "CLI callers는 official launcher와 CLI regression이다. effect는 readiness/evaluate에 lower 40-hex "
        "--reviewed-commit만, inspect는 인자 없이 허용하고 new/resume/retry/override/abbreviation을 거부한다. "
        "evaluate는 Task 2 capture_normalized_target_authority_antigravity만 load한다. "
        "tests는 test_cli_antigravity다. Task 6 official lifecycle은 아직 실행하지 않는다.</p>"
        f"<h2>노드</h2><ul>{items}</ul>"
        f'<h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        "</main></body></html>"
    )


def _allowed_source_parent_antigravity(source_parent: str) -> bool:
    head = _git_output_antigravity("rev-parse", "HEAD")
    if source_parent == head:
        return True
    if source_parent != _git_output_antigravity("rev-parse", "HEAD^"):
        return False
    changed = set(_git_output_antigravity("diff", "--name-only", "HEAD^", "HEAD").splitlines())
    return changed == TRIPLET_ANTIGRAVITY


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT_ANTIGRAVITY, check=False).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must end in Z")

    test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_governance_antigravity.py"
    differential = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_controlled_fork_differential_antigravity.py"
    predecessor_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_predecessor_antigravity.py"
    runtime_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_runtime_binding_antigravity.py"
    sources_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_sources_antigravity.py"
    lifecycle_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_lifecycle_antigravity.py"
    readiness_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_readiness_antigravity.py"
    evaluation_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_evaluation_antigravity.py"
    replay_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_replay_antigravity.py"
    cli_test = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_cli_antigravity.py"
    launcher = "scratch/run_exp454_exp453_exp449_authority_boundary_repair_antigravity.py"
    plan = "docs/superpowers/plans/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity.html"
    future_path = "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity"
    package = future_path
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity-design.html", "role": "승인된 단일 authority 변환 경계.", "tests": [test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity-design.html", "Exp454")]},
        {"id": "implementation_plan", "path": plan, "role": "Task 1–6의 순차 TDD와 공식 실행 경계.", "tests": [test], "evidence": [_evidence_antigravity(plan, "Task 1: HTML governance")]},
        {"id": "preregistration", "path": "docs/EXP454_PREREGISTRATION_antigravity_KO.html", "role": "고정 runtime과 과학 불변값.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP454_PREREGISTRATION_antigravity_KO.html", "NumPy 2.3.5")]},
        {"id": "runbook", "path": "docs/EXP454_RUNBOOK_antigravity_KO.html", "role": "미래 one-shot 명령 순서.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP454_RUNBOOK_antigravity_KO.html", "readiness --reviewed-commit")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Exp454 implementation_verified_awaiting_official_readiness, zero-label, null-result truth.", "tests": [test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", '"exp454"')]},
        {"id": "governance_regression", "path": test, "role": "HTML, ledger, hash, stale-Exp453-map, future-map regression.", "tests": [test], "evidence": [_evidence_antigravity(test, "class GovernanceAntigravityTest")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp454_antigravity.py", "role": "bounded HTML/JSON/lock generator.", "tests": [test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp454_antigravity.py", "def validate_map_antigravity")]},
        {"id": "task2_regressions", "path": differential, "role": "Caller: Task 2 focused verification. Effect: rejects scientific drift across all foundations.", "tests": [differential], "evidence": [_evidence_antigravity(differential, "class ControlledForkDifferentialAntigravityTest")]},
        {"id": "task2_config", "path": f"{future_path}/config_antigravity.py", "role": "Caller: readiness and evaluator. Effect: freezes runtime identity and unchanged scientific profile.", "tests": [f"{future_path}/tests/test_config_antigravity.py", differential], "evidence": [_evidence_antigravity(f"{future_path}/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "task2_contracts", "path": f"{future_path}/contracts_antigravity.py", "role": "Caller: sources, evaluator and replay. Effect: freezes exact metric and dataset records.", "tests": [f"{future_path}/tests/test_contracts_antigravity.py", differential], "evidence": [_evidence_antigravity(f"{future_path}/contracts_antigravity.py", "class CandidateMetricAntigravity")]},
        {"id": "task2_artifacts", "path": f"{future_path}/artifacts_antigravity.py", "role": "Caller: lifecycle, evaluator and replay. Effect: fixes canonical bytes and artifact-map construction.", "tests": [f"{future_path}/tests/test_artifacts_antigravity.py", differential], "evidence": [_evidence_antigravity(f"{future_path}/artifacts_antigravity.py", "def canonical_json_bytes_antigravity")]},
        {"id": "task2_bootstrap", "path": f"{future_path}/bootstrap_antigravity.py", "role": "Caller: evaluator. Effect: fixes the paired draw bytes and FN/FP/F1 intervals.", "tests": [f"{future_path}/tests/test_bootstrap_antigravity.py", differential], "evidence": [_evidence_antigravity(f"{future_path}/bootstrap_antigravity.py", "def paired_delta_antigravity")]},
        {"id": "task2_dashboard", "path": f"{future_path}/dashboard_export_antigravity.py", "role": "Caller: evaluator result projection. Effect: exposes only the same aggregate qualification result.", "tests": [f"{future_path}/tests/test_dashboard_export_antigravity.py", differential], "evidence": [_evidence_antigravity(f"{future_path}/dashboard_export_antigravity.py", "def public_result_antigravity")]},
        {"id": "task2_predecessor", "path": f"{future_path}/predecessor_antigravity.py", "role": "Caller: sources, readiness, and evaluator. Effect: single normalize_exp449_target_authority_antigravity conversion of real Exp449 authority. Tests: test_predecessor_antigravity.", "tests": [predecessor_test, sources_test, evaluation_test], "evidence": [_evidence_antigravity(f"{future_path}/predecessor_antigravity.py", "def normalize_exp449_target_authority_antigravity"), _evidence_antigravity(f"{future_path}/predecessor_antigravity.py", "def capture_predecessors_antigravity")]},
        {"id": "task2_predecessor_tests", "path": predecessor_test, "role": "Caller: Task 2 predecessor verification. Effect: rejects missing conversion and mutated real authority.", "tests": [predecessor_test], "evidence": [_evidence_antigravity(predecessor_test, "class PredecessorAntigravityTest")]},
        {"id": "task3_runtime_binding", "path": "scratch/exp454_runtime_binding_antigravity.py", "role": "Caller: official launcher and all closed commands. Effect: exact executable/prefix re-exec before any Exp454 import.", "tests": [runtime_test], "evidence": [_evidence_antigravity("scratch/exp454_runtime_binding_antigravity.py", "def ensure_fixed_runtime_antigravity"), _evidence_antigravity(launcher, "ensure_fixed_runtime_antigravity")]},
        {"id": "task3_sources", "path": f"{package}/sources_antigravity.py", "role": "Caller: evaluator. Effect: opens one synthetic or official label session only after Exp453 evidence, Exp449 prediction, and normalized target succeed.", "tests": [sources_test, predecessor_test, evaluation_test], "evidence": [_evidence_antigravity(f"{package}/sources_antigravity.py", "def open_test_labels_once_antigravity")]},
        {"id": "task3_lifecycle_readiness", "path": f"{package}/readiness_antigravity.py", "role": "Caller: closed CLI readiness/evaluate map gate. Effect: label-free readiness with exact runtime, Exp453 terminal hashes, and normalized target 48.", "tests": [lifecycle_test, readiness_test], "evidence": [_evidence_antigravity(f"{package}/lifecycle_antigravity.py", "def run_phase_antigravity"), _evidence_antigravity(f"{package}/readiness_antigravity.py", "def readiness_antigravity")]},
        {"id": "task4_evaluation_replay", "path": f"{package}/evaluation_antigravity.py", "role": "Caller: closed CLI evaluate/inspect. Effect: consumes Task 2 capture_normalized_target_authority_antigravity and Task 3 label opener to publish the same 12-leaf result and source-free replay.", "tests": [evaluation_test, replay_test], "evidence": [_evidence_antigravity(f"{package}/evaluation_antigravity.py", "capture_normalized_target_authority_antigravity"), _evidence_antigravity(f"{package}/replay_antigravity.py", "def inspect_replay_at_antigravity")]},
        {"id": "task5_cli", "path": f"{package}/cli_antigravity.py", "role": "Caller: official launcher and CLI regression. Effect: readiness/evaluate accept one lower 40-hex --reviewed-commit, inspect is bare, and evaluate loads capture_normalized_target_authority_antigravity only.", "tests": [cli_test], "evidence": [_evidence_antigravity(f"{package}/cli_antigravity.py", "def main_antigravity"), _evidence_antigravity(launcher, "main_antigravity")]},
    ]
    future_pairs: list[tuple[str, str]] = []
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": _evidence_antigravity("docs/EXP454_PREREGISTRATION_antigravity_KO.html", "고정 정체성")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": _evidence_antigravity(plan, "build_codemap_exp454_antigravity.py")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": _evidence_antigravity(test, "test_planned_truth_preserves_science_and_champion_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": _evidence_antigravity(test, "builder.validate_map_antigravity")},
        *[
            {"from": "implementation_plan", "to": node_id, "type": "governs", "evidence": _evidence_antigravity(plan, "Task 2: Controlled fork")}
            for node_id in ("task2_config", "task2_contracts", "task2_artifacts", "task2_bootstrap", "task2_dashboard", "task2_predecessor")
        ],
        *[
            {"from": "task2_regressions", "to": node_id, "type": "tests", "evidence": _evidence_antigravity(differential, test_symbol)}
            for node_id, test_symbol in (
                ("task2_config", "test_scientific_config_is_identical_after_named_identity_fields_are_removed_antigravity"),
                ("task2_contracts", "test_contract_artifact_and_bootstrap_asts_are_identity_only_antigravity"),
                ("task2_artifacts", "test_contract_artifact_and_bootstrap_asts_are_identity_only_antigravity"),
                ("task2_bootstrap", "test_contract_artifact_and_bootstrap_asts_are_identity_only_antigravity"),
                ("task2_dashboard", "test_dashboard_projection_ast_changes_identity_only_antigravity"),
            )
        ],
        {"from": "task2_predecessor", "to": "task2_contracts", "type": "reads", "evidence": _evidence_antigravity(f"{future_path}/predecessor_antigravity.py", "DatasetBaselineAntigravity")},
        {"from": "task2_predecessor_tests", "to": "task2_predecessor", "type": "tests", "evidence": _evidence_antigravity(predecessor_test, "test_real_exp449_authority_normalizes_without_test_fixture_conversion_antigravity")},
        *[
            {"from": "implementation_plan", "to": node_id, "type": "governs", "evidence": _evidence_antigravity(plan, "Task 3: Runtime")}
            for node_id in ("task3_runtime_binding", "task3_sources", "task3_lifecycle_readiness")
        ],
        {"from": "task3_sources", "to": "task2_predecessor", "type": "reads", "evidence": _evidence_antigravity(f"{package}/sources_antigravity.py", "_claim_label_capability_antigravity")},
        {"from": "task3_lifecycle_readiness", "to": "task2_predecessor", "type": "reads", "evidence": _evidence_antigravity(f"{package}/readiness_antigravity.py", "capture_predecessors_antigravity")},
        {"from": "task3_lifecycle_readiness", "to": "task3_lifecycle_readiness", "type": "reads", "evidence": _evidence_antigravity(f"{package}/readiness_antigravity.py", "_run_phase_with_paths_antigravity")},
        {"from": "implementation_plan", "to": "task4_evaluation_replay", "type": "governs", "evidence": _evidence_antigravity(plan, "Task 4: 실제 authority evaluator")},
        {"from": "task4_evaluation_replay", "to": "task2_predecessor", "type": "reads", "evidence": _evidence_antigravity(f"{package}/evaluation_antigravity.py", "capture_normalized_target_authority_antigravity")},
        {"from": "task4_evaluation_replay", "to": "task3_sources", "type": "reads", "evidence": _evidence_antigravity(f"{package}/evaluation_antigravity.py", "open_test_labels_once_antigravity")},
        {"from": "implementation_plan", "to": "task5_cli", "type": "governs", "evidence": _evidence_antigravity(plan, "Task 5: Closed CLI")},
        {"from": "task3_runtime_binding", "to": "task5_cli", "type": "reads", "evidence": _evidence_antigravity(launcher, "cli_antigravity")},
        {"from": "task5_cli", "to": "task3_lifecycle_readiness", "type": "reads", "evidence": _evidence_antigravity(f"{package}/cli_antigravity.py", "_load_readiness_antigravity")},
        {"from": "task5_cli", "to": "task4_evaluation_replay", "type": "reads", "evidence": _evidence_antigravity(f"{package}/cli_antigravity.py", "_load_evaluate_antigravity")},
        {"from": "task5_cli", "to": "task4_evaluation_replay", "type": "reads", "evidence": _evidence_antigravity(f"{package}/cli_antigravity.py", "_load_inspect_antigravity")},
    ]
    unknowns = [{"from": left, "to": right, "reason": "Task 5 does not implement this later product boundary."} for left, right in future_pairs]
    graph: dict[str, object] = {"schema": "exp454_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp454_codemap_map_child_v1", "source_parent_commit": source_commit, "working_tree_dirty": False, "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "module_fingerprints": _module_fingerprints_antigravity(), "tracked_fingerprints": _tracked_fingerprints_antigravity()}
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
            if not (ROOT_ANTIGRAVITY / node["path"]).is_file() or not node.get("evidence"):
                raise ValueError("missing node evidence")
            for evidence in node["evidence"]:
                _validate_evidence_antigravity(evidence)
    pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    future_edges: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY:
            raise ValueError("invalid edge")
        _validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in pairs:
                raise ValueError("unmarked future relationship")
            future_edges.add(pair)
        elif edge["type"] == "future_unknown":
            raise ValueError("known edge marked future")
    if pairs != future_edges:
        raise ValueError("unknown relationship mismatch")
    source = lock.get("source_parent_commit")
    if source != graph.get("source_parent_commit") or not isinstance(source, str) or not _allowed_source_parent_antigravity(source):
        raise ValueError("invalid source parent")
    if lock.get("working_tree_dirty") is not False or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != _module_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    if len(lock.get("tracked_fingerprints") or {}) != len(TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY):
        raise ValueError("tracked fingerprint count")
    match = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, flags=re.DOTALL)
    if not match or _canonical_bytes_antigravity(_strict_json_antigravity(html_module.unescape(match.group(1)))) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-14T12:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(_strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text()), _strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text()), HTML_PATH_ANTIGRAVITY.read_text())
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    if set(_git_output_antigravity("diff", "--name-only").splitlines()) - TRIPLET_ANTIGRAVITY:
        raise ValueError("map generation requires a clean functional commit")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at)
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph) + b"\n")
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock) + b"\n")
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    main_antigravity()
