#!/usr/bin/env python3
"""Build and validate the bounded Exp452 governance and foundation code map."""

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
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp452 코드맵</title><style>body{{max-width:1050px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}}code,pre{{background:#eef2f7;padding:.15rem .3rem;overflow:auto}}</style></head><body><main><h1>Exp452 거버넌스와 고정 기반 코드맵</h1><p>Task 2의 config, contract, canonical artifact IO, paired bootstrap, aggregate-only dashboard projection은 구현됐다. predecessor, lifecycle, evaluator, replay, CLI와 official lifecycle 관계는 future/unknown으로 유지한다.</p><h2>노드</h2><ul>{items}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre></main></body></html>'''


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
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-13-exp452-exp449-index-order-repair-antigravity-design.html", "role": "Exp449 index-order repair의 승인된 고정 경계.", "tests": [test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-13-exp452-exp449-index-order-repair-antigravity-design.html", "Exp452: Exp449 인덱스 순서 복구 설계")]},
        {"id": "implementation_plan", "path": "docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "role": "Task 1 TDD와 map-only child 규칙.", "tests": [test], "evidence": [_evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 1: Governance HTML과 Exp452 코드맵")]},
        {"id": "preregistration", "path": "docs/EXP452_PREREGISTRATION_antigravity_KO.html", "role": "frozen repair, label firewall, qualified-2 non-input 계약.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP452_PREREGISTRATION_antigravity_KO.html", "cannot guide")]},
        {"id": "runbook", "path": "docs/EXP452_RUNBOOK_antigravity_KO.html", "role": "future lifecycle의 one-shot stop 규칙.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP452_RUNBOOK_antigravity_KO.html", "cannot guide")]},
        {"id": "implementation_report", "path": "docs/EXP452_IMPLEMENTATION_REPORT_antigravity.html", "role": "Task 1 scope와 absent history-ledger resolution.", "tests": [test], "evidence": [_evidence_antigravity("docs/EXP452_IMPLEMENTATION_REPORT_antigravity.html", "repository-assumption mismatch")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Exp452 planned truth와 zero official state.", "tests": [test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "\"exp452\"")]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Exp452 identity와 immutable predecessor catalog.", "tests": [test], "evidence": [_evidence_antigravity("EXPERIMENT_CATALOG.json", "\"id\": \"Exp452\"")]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "non-promotion 및 unchanged pending candidate record.", "tests": [test], "evidence": [_evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp452\"")]},
        {"id": "governance_regression", "path": test, "role": "strict JSON, no Markdown twin, zero official counts 및 map validation test.", "tests": [test], "evidence": [_evidence_antigravity(test, "test_map_is_strict_embedded_and_marks_product_relations_future_unknown_antigravity")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp452_antigravity.py", "role": "bounded graph, strict embedded JSON와 tracked fingerprint lock generator.", "tests": [test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp452_antigravity.py", "def validate_map_antigravity")]},
        {"id": "task2_config", "path": "experiments/exp452_exp449_index_order_repair_antigravity/config_antigravity.py", "role": "Exp452 identity/path/predecessor/index-order nodes를 추가하고 Exp449 science와 48/96/16을 고정한다.", "tests": [config_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "task2_contracts", "path": "experiments/exp452_exp449_index_order_repair_antigravity/contracts_antigravity.py", "role": "strict exact-type confusion과 candidate metric records; 현재 callers는 focused/differential tests뿐이다.", "tests": [contracts_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/contracts_antigravity.py", "class CandidateMetricAntigravity")]},
        {"id": "task2_artifacts", "path": "experiments/exp452_exp449_index_order_repair_antigravity/artifacts_antigravity.py", "role": "canonical JSON/JSONL, create-only no-follow write와 descriptor-validated exact inventory hashing.", "tests": [artifacts_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/artifacts_antigravity.py", "def artifact_map_bytes_antigravity")]},
        {"id": "task2_bootstrap", "path": "experiments/exp452_exp449_index_order_repair_antigravity/bootstrap_antigravity.py", "role": "Exp449-exact PCG64 paired FN/FP/F1 descriptive bootstrap; selection effect는 없다.", "tests": [bootstrap_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/bootstrap_antigravity.py", "def paired_delta_antigravity")]},
        {"id": "task2_dashboard", "path": "experiments/exp452_exp449_index_order_repair_antigravity/dashboard_export_antigravity.py", "role": "aggregate-only pure projection; label/index/dataset/path/hash/log/per-dataset material을 recursive reject하고 write하지 않는다.", "tests": [dashboard_test, differential], "evidence": [_evidence_antigravity("experiments/exp452_exp449_index_order_repair_antigravity/dashboard_export_antigravity.py", "def public_result_antigravity")]},
        {"id": "task2_regressions", "path": differential, "role": "six focused suites와 controlled-fork AST guard의 caller boundary.", "tests": [config_test, contracts_test, artifacts_test, bootstrap_test, dashboard_test, differential], "evidence": [_evidence_antigravity(differential, "class ControlledForkDifferentialAntigravityTest")]},
        {"id": "future_product_modules", "path": "Exp452 Task 3-5 predecessor/lifecycle/evaluator/replay/CLI", "role": "Task 3–5 consumers and effects remain unimplemented and unknown after Task 2.", "status": "future_unknown", "tests": [], "evidence": []},
        {"id": "future_official_lifecycle", "path": "Exp452 official readiness/evaluate/inspect", "role": "official lifecycle is not run or implemented in Task 2.", "status": "future_unknown", "tests": [], "evidence": []},
    ]
    edges: list[dict[str, object]] = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": _evidence_antigravity("docs/EXP452_PREREGISTRATION_antigravity_KO.html", "Exp452")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "map-only child")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": _evidence_antigravity(test, "test_planned_truth_is_label_free_and_non_promoting_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": _evidence_antigravity(test, "builder.validate_map_antigravity")},
        {"from": "research_state", "to": "preregistration", "type": "reads", "evidence": _evidence_antigravity("RESEARCH_STATE.json", "EXP452_PREREGISTRATION_antigravity_KO.html")},
        {"from": "implementation_plan", "to": "task2_config", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "implementation_plan", "to": "task2_contracts", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "ConfusionAntigravity")},
        {"from": "implementation_plan", "to": "task2_artifacts", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "canonical JSON/JSONL")},
        {"from": "implementation_plan", "to": "task2_bootstrap", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "paired_delta_antigravity()")},
        {"from": "implementation_plan", "to": "task2_dashboard", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Dashboard allowlist")},
        {"from": "task2_regressions", "to": "task2_config", "type": "tests", "evidence": _evidence_antigravity(config_test, "test_candidate_matrix_and_scientific_settings_match_exp449_antigravity")},
        {"from": "task2_regressions", "to": "task2_contracts", "type": "tests", "evidence": _evidence_antigravity(contracts_test, "test_bool_is_not_an_integer_metric_antigravity")},
        {"from": "task2_regressions", "to": "task2_artifacts", "type": "tests", "evidence": _evidence_antigravity(artifacts_test, "test_canonical_json_and_jsonl_are_exact_antigravity")},
        {"from": "task2_regressions", "to": "task2_bootstrap", "type": "tests", "evidence": _evidence_antigravity(bootstrap_test, "test_bootstrap_authority_is_byte_exact_antigravity")},
        {"from": "task2_regressions", "to": "task2_dashboard", "type": "tests", "evidence": _evidence_antigravity(dashboard_test, "test_public_result_returns_only_the_complete_aggregate_allowlist_antigravity")},
        {"from": "task2_config", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 3: Exp449 읽기 전용 predecessor와 lifecycle")},
        {"from": "task2_contracts", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 4: 정렬 복구 evaluator와 source-free replay")},
        {"from": "task2_artifacts", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 4: 정렬 복구 evaluator와 source-free replay")},
        {"from": "task2_bootstrap", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 4: 정렬 복구 evaluator와 source-free replay")},
        {"from": "task2_dashboard", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html", "Task 4: 정렬 복구 evaluator와 source-free replay")},
        {"from": "approved_design", "to": "future_product_modules", "type": "future_unknown", "evidence": _evidence_antigravity("docs/superpowers/specs/2026-08-13-exp452-exp449-index-order-repair-antigravity-design.html", "구현 단계")},
        {"from": "runbook", "to": "future_official_lifecycle", "type": "future_unknown", "evidence": _evidence_antigravity("docs/EXP452_RUNBOOK_antigravity_KO.html", "현재는 실행 금지")},
    ]
    future_sources = ("approved_design", "task2_config", "task2_contracts", "task2_artifacts", "task2_bootstrap", "task2_dashboard")
    unknowns = [{"from": source, "to": "future_product_modules", "reason": "Task 3–5 consumer/effect relationships remain unimplemented after Task 2."} for source in future_sources]
    unknowns.append({"from": "runbook", "to": "future_official_lifecycle", "reason": "Task 2 forbids official lifecycle execution."})
    graph: dict[str, object] = {"schema": "exp452_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "topology_note": "Task 2 map: frozen foundations and their focused/differential test callers are implemented; Task 3–5 effects remain explicitly future/unknown.", "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp452_codemap_same_commit_v1", "source_parent_commit": source_commit, "working_tree_dirty": bool(_git_output_antigravity("status", "--porcelain")), "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "fingerprint_algorithm": "sha256-v1: tracked bytes excluding exactly the Exp452 map triplet", "module_fingerprints": _module_fingerprints_antigravity(), "tracked_fingerprints": _tracked_fingerprints_antigravity()}
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
    if _git_output_antigravity("status", "--porcelain"): raise ValueError("map generation requires a clean functional commit")
    graph, lock, rendered = build_map_antigravity(args.source_commit, args.generated_at); validate_map_antigravity(graph, lock, rendered)
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True); JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph) + b"\n"); LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock) + b"\n"); HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")


if __name__ == "__main__": main_antigravity()
