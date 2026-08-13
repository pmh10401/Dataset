#!/usr/bin/env python3
"""Build and validate the bounded Exp452 governance and Task 3 code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp452_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp452_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp452_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp452_antigravity.html",
    "docs/codemap/codemap_exp452_antigravity.json",
    "docs/codemap/codemap_exp452_antigravity.lock",
}
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "reads", "tests", "future_unknown"}


def _canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_ANTIGRAVITY, text=True).strip()


def _tracked_paths_antigravity() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT_ANTIGRAVITY)
    paths = [Path(entry.decode("utf-8")) for entry in raw.split(b"\0") if entry]
    return sorted(path for path in paths if path.as_posix() not in TRIPLET_ANTIGRAVITY)


def _tracked_fingerprints_antigravity() -> dict[str, str]:
    return {path.as_posix(): hashlib.sha256((ROOT_ANTIGRAVITY / path).read_bytes()).hexdigest() for path in _tracked_paths_antigravity()}


def _module_fingerprints_antigravity() -> dict[str, str]:
    groups: dict[str, list[Path]] = {"root": []}
    for path in _tracked_paths_antigravity():
        groups.setdefault("root" if len(path.parts) == 1 else path.parts[0], []).append(path)
    result: dict[str, str] = {}
    for group, paths in sorted(groups.items()):
        digest = hashlib.sha256()
        for path in sorted(paths):
            data = (ROOT_ANTIGRAVITY / path).read_bytes()
            digest.update(path.as_posix().encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data + b"\0")
        result[group] = digest.hexdigest()
    return result


def _evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def _html_antigravity(graph: dict[str, object]) -> str:
    payload = html_module.escape(_canonical_bytes_antigravity(graph).decode("utf-8"))
    items = "".join(f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>' for node in graph["nodes"])
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp452 코드맵</title><style>body{{max-width:1050px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}}code,pre{{background:#eef2f7;padding:.15rem .3rem;overflow:auto}}</style></head><body><main><h1>Exp452 Task 5 코드맵</h1><p>Task 2–4의 frozen foundation, stopped predecessor, lifecycle, evaluator와 source-free replay 위에 Task 5 closed CLI와 outside-CWD launcher가 추가됐다. 공식 lifecycle invocation은 아직 0이며 별도 authorization 전에는 실행하지 않는다.</p><h2>노드</h2><ul>{items}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre></main></body></html>'''


def _allowed_source_parent_antigravity(source_parent: str) -> bool:
    head = _git_output_antigravity("rev-parse", "HEAD")
    if source_parent == head:
        return True
    if source_parent != _git_output_antigravity("rev-parse", "HEAD^"):
        return False
    changed = set(_git_output_antigravity("diff", "--name-only", "HEAD^", "HEAD").splitlines())
    return changed == TRIPLET_ANTIGRAVITY


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return a source-parent-bound Exp452 governance map and lock."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT_ANTIGRAVITY, check=False).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must be UTC ISO-8601 ending in Z")
    test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_governance_antigravity.py"
    differential = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_controlled_fork_differential_antigravity.py"
    config_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_config_antigravity.py"
    contracts_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_contracts_antigravity.py"
    artifacts_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_artifacts_antigravity.py"
    bootstrap_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_bootstrap_antigravity.py"
    dashboard_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_dashboard_export_antigravity.py"
    predecessor_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_predecessor_antigravity.py"
    sources_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_sources_antigravity.py"
    lifecycle_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_lifecycle_antigravity.py"
    readiness_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_readiness_antigravity.py"
    evaluation_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_evaluation_antigravity.py"
    replay_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_replay_antigravity.py"
    cli_test = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_cli_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-13-exp452-exp449-index-order-repair-antigravity-design.html", "role": "Exp449 index-order repair의 승인된 고정 경계.", "tests": [test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-13-exp452-exp449-index-order-repair-antigravity-design.html", "Exp452: Exp449 인덱스 순서 복구 설계")]},
        {"id": "implementation_plan", "path": "docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "role": "Task 1 TDD와 map-only child 규칙.", "tests": [test], "evidence": [_evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 1: Governance HTML과 Exp452 코드맵")]},
        {"id": "preregistration", "path": "docs/EXP452_PREREGISTRATION_antigravity_KO.html", "role": "frozen repair, label firewall, qualified-2 non-input 계약.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP452_PREREGISTRATION_antigravity_KO.html", "cannot guide")]},
        {"id": "runbook", "path": "docs/EXP452_RUNBOOK_antigravity_KO.html", "role": "future lifecycle의 one-shot stop 규칙.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP452_RUNBOOK_antigravity_KO.html", "cannot guide")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Exp452 planned truth와 zero official state.", "tests": [test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "\"exp452\"")]},
        {"id": "governance_regression", "path": test, "role": "strict JSON, no Markdown twin, zero official counts 및 map validation test.", "tests": [test], "evidence": [_evidence_antigravity(test, "test_map_is_strict_embedded_and_records_task_five_cli_antigravity")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp452_antigravity.py", "role": "bounded graph, strict embedded JSON와 tracked fingerprint lock generator.", "tests": [test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp452_antigravity.py", "def validate_map_antigravity")]},
        {"id": "task2_config", "path": "experiments/exp452_exp449_index_order_repair_antigravity/config_antigravity.py", "role": "Exp452 identity/path/predecessor/index-order nodes를 추가하고 Exp449 science와 48/96/16을 고정한다.", "tests": [config_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "task2_contracts", "path": "experiments/exp452_exp449_index_order_repair_antigravity/contracts_antigravity.py", "role": "strict exact-type confusion과 candidate metric records; 현재 callers는 focused/differential tests뿐이다.", "tests": [contracts_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/contracts_antigravity.py", "class CandidateMetricAntigravity")]},
        {"id": "task2_artifacts", "path": "experiments/exp452_exp449_index_order_repair_antigravity/artifacts_antigravity.py", "role": "canonical JSON/JSONL, create-only no-follow write와 descriptor-validated exact inventory hashing.", "tests": [artifacts_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/artifacts_antigravity.py", "def artifact_map_bytes_antigravity")]},
        {"id": "task2_bootstrap", "path": "experiments/exp452_exp449_index_order_repair_antigravity/bootstrap_antigravity.py", "role": "Exp449-exact PCG64 paired FN/FP/F1 descriptive bootstrap; selection effect는 없다.", "tests": [bootstrap_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/bootstrap_antigravity.py", "def paired_delta_antigravity")]},
        {"id": "task2_dashboard", "path": "experiments/exp452_exp449_index_order_repair_antigravity/dashboard_export_antigravity.py", "role": "aggregate-only pure projection; label/index/dataset/path/hash/log/per-dataset material을 recursive reject하고 write하지 않는다.", "tests": [dashboard_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/dashboard_export_antigravity.py", "def public_result_antigravity")]},
        {"id": "task4_evaluator", "path": "experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "role": "Callers: Task-3 predecessor/source/lifecycle and future Task-5 CLI. Effects: 12-leaf evaluation publication with the sole ascending-evidence producer.", "tests": [evaluation_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "sorted(current[datasets[position]])")]},
        {"id": "task4_replay", "path": "experiments/exp452_exp449_index_order_repair_antigravity/replay_antigravity.py", "role": "Callers: evaluator private publication check and future inspect CLI. Effects: source-free semantic replay with exact ordered integer-list equality.", "tests": [replay_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/replay_antigravity.py", "candidate[\"indices\"] != row[\"indices\"]")]},
        {"id": "task3_predecessor", "path": "experiments/exp452_exp449_index_order_repair_antigravity/predecessor_antigravity.py", "role": "Callers: sources and readiness. Effects: twice captures immutable stopped Exp449 evidence and issues a non-metric authority.", "tests": [predecessor_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/predecessor_antigravity.py", "def capture_exp449_stopped_authority_antigravity")]},
        {"id": "task3_sources", "path": "experiments/exp452_exp449_index_order_repair_antigravity/sources_antigravity.py", "role": "Callers: focused tests and the future evaluator. Effects: sole one-shot 48-query/72,300-row TEST-label session boundary.", "tests": [sources_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/sources_antigravity.py", "def open_test_labels_once_antigravity")]},
        {"id": "task3_lifecycle", "path": "experiments/exp452_exp449_index_order_repair_antigravity/lifecycle_antigravity.py", "role": "Callers: readiness and the future CLI. Effects: create-only readiness→evaluate→inspect receipts and permanent stop.", "tests": [lifecycle_test], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/lifecycle_antigravity.py", "def run_phase_antigravity")]},
        {"id": "task3_readiness", "path": "experiments/exp452_exp449_index_order_repair_antigravity/readiness_antigravity.py", "role": "Callers: the future official CLI. Effects: label-incapable clean-map, document, path, DB, predecessor, runtime and backend metadata gates.", "tests": [readiness_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/readiness_antigravity.py", "def readiness_antigravity")]},
        {"id": "task2_regressions", "path": differential, "role": "Callers: Task 2 foundation coverage. Effects: constrained fork differential rejects any scientific delta outside identity, predecessor, and ordered evidence.", "tests": [differential], "evidence": [_evidence_antigravity(differential, "test_task4_runtime_same_label_differential_preserves_scientific_payloads_antigravity")]},
        {"id": "task5_cli", "path": "experiments/exp452_exp449_index_order_repair_antigravity/cli_antigravity.py", "role": "Callers: exact outside-CWD launcher and CLI regression. Effects: closed three-form parser, lazy phase imports, and clean reviewed-map gate before evaluator import.", "tests": [cli_test], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/cli_antigravity.py", "def main_antigravity"), _evidence_antigravity("scratch/run_exp452_exp449_index_order_repair_antigravity.py", "main_antigravity")]},
    ]
    edges: list[dict[str, object]] = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": _evidence_antigravity("docs/EXP452_PREREGISTRATION_antigravity_KO.html", "Exp452")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "map-only child")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": _evidence_antigravity(test, "test_implementation_candidate_truth_is_label_free_and_non_promoting_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": _evidence_antigravity(test, "builder.validate_map_antigravity")},
        {"from": "research_state", "to": "preregistration", "type": "reads", "evidence": _evidence_antigravity("RESEARCH_STATE.json", "EXP452_PREREGISTRATION_antigravity_KO.html")},
        {"from": "implementation_plan", "to": "task2_config", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "implementation_plan", "to": "task2_contracts", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "ConfusionAntigravity")},
        {"from": "implementation_plan", "to": "task2_artifacts", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "canonical JSON/JSONL")},
        {"from": "implementation_plan", "to": "task2_bootstrap", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "paired_delta_antigravity()")},
        {"from": "implementation_plan", "to": "task2_dashboard", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Dashboard allowlist")},
        {"from": "implementation_plan", "to": "task4_evaluator", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "sorted(current[datasets[position]])")},
        {"from": "task4_evaluator", "to": "task2_config", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "from .config_antigravity import")},
        {"from": "task4_evaluator", "to": "task2_artifacts", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "from .artifacts_antigravity import")},
        {"from": "task4_evaluator", "to": "task2_bootstrap", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "from .bootstrap_antigravity import")},
        {"from": "task4_evaluator", "to": "task3_predecessor", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "capture_exp449_stopped_authority_antigravity")},
        {"from": "task4_evaluator", "to": "task3_sources", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "open_test_labels_once_antigravity")},
        {"from": "task4_evaluator", "to": "task2_dashboard", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "public_result_antigravity")},
        {"from": "task4_evaluator", "to": "task3_lifecycle", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "run_phase_antigravity")},
        {"from": "task4_evaluator", "to": "task4_replay", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/evaluation_antigravity.py", "inspect_replay_at_fd_antigravity")},
        {"from": "task4_replay", "to": "task2_contracts", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/replay_antigravity.py", "from .contracts_antigravity import")},
        {"from": "implementation_plan", "to": "task3_predecessor", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "capture_exp449_stopped_authority_antigravity()")},
        {"from": "implementation_plan", "to": "task3_sources", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "open_test_labels_once_antigravity(authority)")},
        {"from": "implementation_plan", "to": "task3_lifecycle", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "run_phase_antigravity()")},
        {"from": "implementation_plan", "to": "task3_readiness", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "readiness_antigravity(reviewed_commit)")},
        {"from": "task3_predecessor", "to": "task2_config", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/predecessor_antigravity.py", "from .config_antigravity import")},
        {"from": "task3_predecessor", "to": "task2_artifacts", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/predecessor_antigravity.py", "from .artifacts_antigravity import")},
        {"from": "task3_sources", "to": "task3_predecessor", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/sources_antigravity.py", "from .predecessor_antigravity import")},
        {"from": "task3_lifecycle", "to": "task2_artifacts", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/lifecycle_antigravity.py", "from .artifacts_antigravity import")},
        {"from": "task3_readiness", "to": "task3_predecessor", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/readiness_antigravity.py", "from .predecessor_antigravity import")},
        {"from": "task3_readiness", "to": "task3_lifecycle", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/readiness_antigravity.py", "from .lifecycle_antigravity import")},
        {"from": "task2_regressions", "to": "task2_config", "type": "tests", "evidence": _evidence_antigravity(differential, "test_config_ast_allows_only_named_identity_path_predecessor_and_index_nodes_antigravity")},
        {"from": "task2_regressions", "to": "task2_contracts", "type": "tests", "evidence": _evidence_antigravity(differential, "test_contract_artifact_and_bootstrap_asts_are_scientifically_identical_antigravity")},
        {"from": "task2_regressions", "to": "task2_artifacts", "type": "tests", "evidence": _evidence_antigravity(differential, "test_contract_artifact_and_bootstrap_asts_are_scientifically_identical_antigravity")},
        {"from": "task2_regressions", "to": "task2_bootstrap", "type": "tests", "evidence": _evidence_antigravity(differential, "test_contract_artifact_and_bootstrap_asts_are_scientifically_identical_antigravity")},
        {"from": "task2_regressions", "to": "task2_dashboard", "type": "tests", "evidence": _evidence_antigravity(differential, "test_dashboard_aggregate_validation_ast_changes_identity_only_antigravity")},
        {"from": "task5_cli", "to": "task3_readiness", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/cli_antigravity.py", "_load_readiness_antigravity")},
        {"from": "task5_cli", "to": "task4_evaluator", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/cli_antigravity.py", "_load_evaluate_antigravity")},
        {"from": "task5_cli", "to": "task4_replay", "type": "reads", "evidence": _evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/cli_antigravity.py", "_load_inspect_antigravity")},
    ]
    graph: dict[str, object] = {"schema": "exp452_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "topology_note": "Task 5 map: frozen foundations, stopped predecessor, one-shot label boundary, evaluator, strict source-free replay, and closed CLI are implemented. Official lifecycle invocation remains zero pending separate authorization.", "nodes": nodes, "edges": edges, "unknown_relationships": []}
    changed_paths = set(_git_output_antigravity("diff", "--name-only").splitlines())
    lock: dict[str, object] = {"schema": "exp452_codemap_same_commit_v1", "source_parent_commit": source_commit, "working_tree_dirty": bool(changed_paths - TRIPLET_ANTIGRAVITY), "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "fingerprint_algorithm": "sha256-v1: tracked bytes excluding exactly the Exp452 map triplet", "module_fingerprints": _module_fingerprints_antigravity(), "tracked_fingerprints": _tracked_fingerprints_antigravity()}
    return graph, lock, _html_antigravity(graph)


def _validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str) or not item["symbol"]:
        raise ValueError("missing evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("invalid evidence reference")


def validate_map_antigravity(graph: object, lock: object, rendered_html: str) -> None:
    """Reject duplicate, unsupported, stale, or falsely-known map records."""
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes, edges, unknowns = graph.get("nodes"), graph.get("edges"), graph.get("unknown_relationships")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(unknowns, list) or len(nodes) > 20:
        raise ValueError("invalid bounded topology")
    ids: set[str] = set(); future_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not isinstance(node.get("path"), str) or node["id"] in ids:
            raise ValueError("invalid node")
        ids.add(node["id"])
        if node.get("status") == "future_unknown":
            future_ids.add(node["id"])
            if node.get("evidence") not in ([], None): raise ValueError("future unknown node cannot claim evidence")
        else:
            if not (ROOT_ANTIGRAVITY / node["path"]).is_file() or not isinstance(node.get("evidence"), list) or not node["evidence"]: raise ValueError("missing node evidence")
            for evidence in node["evidence"]: _validate_evidence_antigravity(evidence)
    pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    if len(pairs) != len(unknowns) or any(left not in ids or right not in ids for left, right in pairs): raise ValueError("invalid unknown reference")
    future_edges: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY: raise ValueError("invalid edge reference")
        _validate_evidence_antigravity(edge.get("evidence")); pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in pairs: raise ValueError("unmarked unknown relationship")
            future_edges.add(pair)
        elif edge["type"] == "future_unknown" or pair in pairs: raise ValueError("unknown relationship asserted as known")
    if pairs != future_edges: raise ValueError("unmarked unknown relationship")
    source = lock.get("source_parent_commit")
    if source != graph.get("source_parent_commit") or not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source) or not _allowed_source_parent_antigravity(source): raise ValueError("invalid source parent relationship")
    if lock.get("working_tree_dirty") is not False or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY): raise ValueError("invalid map lock")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != _module_fingerprints_antigravity(): raise ValueError("fingerprint drift")
    match = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, flags=re.DOTALL)
    if not match: raise ValueError("HTML lacks embedded graph")
    embedded = _strict_json_antigravity(html_module.unescape(match.group(1)))
    if _canonical_bytes_antigravity(embedded) != _canonical_bytes_antigravity(graph): raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-commit"); parser.add_argument("--generated-at", default="2026-08-13T00:00:00Z"); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(_strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), _strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")), HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8")); return
    if not args.source_commit: parser.error("--source-commit is required unless --validate-only")
    changed_paths = set(_git_output_antigravity("diff", "--name-only").splitlines())
    if changed_paths - TRIPLET_ANTIGRAVITY: raise ValueError("map generation requires a clean functional commit")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at); validate_map_antigravity(graph, lock, rendered)
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True); JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph) + b"\n"); LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock) + b"\n"); HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")


if __name__ == "__main__": main_antigravity()
