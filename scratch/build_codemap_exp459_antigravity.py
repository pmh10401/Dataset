#!/usr/bin/env python3
"""Build and validate the bounded Exp459 Task 2 contracts/folds code map."""

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


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp459_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp459_antigravity.html",
    "docs/codemap/codemap_exp459_antigravity.json",
    "docs/codemap/codemap_exp459_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP459_DASHBOARD_antigravity.html",
    "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP459_PREREGISTRATION_antigravity.html",
    "docs/EXP459_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html",
    "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/blocked_folds_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/config_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/contracts_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_blocked_folds_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_controlled_fork_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp459_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "tests"}


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def python_symbols_antigravity(text: str) -> set[str]:
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


def evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".html":
        parser = HtmlTextExtractorAntigravity()
        parser.feed(text)
        return symbol in " ".join(parser.parts)
    if target.suffix == ".py":
        return symbol in python_symbols_antigravity(text)
    return symbol in text


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
        '<title>Exp459 Task 2 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp459 Task 2 contracts/folds 코드맵</h1><p>Task 2는 Exp459를 planned_not_started로 유지한 채 frozen config, immutable contracts, deterministic blocked folds, bounded tests, 그리고 동기화된 ledgers/map만 추가한다. 공식 evaluate/inspect, label query, 공식 root 생성, champion 변경은 여전히 없다.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        '</main></body></html>'
    )


def build_map_antigravity(
    source_commit: str,
    generated_at: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        cwd=ROOT_ANTIGRAVITY,
        check=False,
    ).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must end in Z")

    design = "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html"
    governance_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    config_path = "experiments/exp459_ecg_normal2_blocked_validation_antigravity/config_antigravity.py"
    contracts_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/contracts_antigravity.py"
    )
    blocked_folds_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/blocked_folds_antigravity.py"
    )
    config_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_config_antigravity.py"
    )
    contracts_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_contracts_antigravity.py"
    )
    blocked_folds_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_blocked_folds_antigravity.py"
    )
    controlled_fork_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_controlled_fork_antigravity.py"
    )
    builder = "scratch/build_codemap_exp459_antigravity.py"

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Task 2 implementation and future Tasks 3–6. Effect: freezes the blocked-validation identity, predecessor, fold sizes/ranges, thresholds, and stable decision rule.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp459")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Task 1–2 execution. Effect: records checked Task 1 and Task 2 steps while Tasks 3–6 remain unchecked pending later work.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 2: 고정 계약과 연속 fold 생성")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP459_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: states planned_not_started identity, fixed thresholds, zero official actions, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_PREREGISTRATION_antigravity.html", "planned_not_started")],
        },
        {
            "id": "runbook",
            "path": "docs/EXP459_RUNBOOK_antigravity.html",
            "role": "Caller: future operators. Effect: records that Task 2 completed frozen contracts and blocked folds while official commands, root creation, and label access remain blocked.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_RUNBOOK_antigravity.html", "Task 3 이전까지 금지한 일")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports the Task 2 frozen config/contracts/folds additions while the official Exp459 lifecycle remains unstarted.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html", "Task 2의 고정 계약과 연속 fold 생성까지 완료했다")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP459_DASHBOARD_antigravity.html",
            "role": "Caller: reader-facing review. Effect: shows Task 2/6 progress, planned_not_started status, zero official commands, and unchanged champion state.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_DASHBOARD_antigravity.html", "Task 2/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Exp459 as planned_not_started with Task 2 complete, zero official invocations, label_query_count 0, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"task_2_contracts_and_blocked_folds_complete\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp459 documents, Task 2 implementation status, frozen predecessor link, planned status, and no-result truth.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"task_2_contracts_and_blocked_folds_complete\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp459 Task 2 completion without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"task_2_contracts_and_blocked_folds_complete\"")],
        },
        {
            "id": "task2_config",
            "path": config_path,
            "role": "Caller: contracts, blocked_folds, and future qualification. Effect: freezes Exp459 identity plus the inherited Exp458 candidate, alpha, Hydra settings, runtime, fold ranges, seeds, types, and AUC/Wilson thresholds.",
            "tests": [config_test, controlled_fork_test],
            "evidence": [evidence_antigravity(config_path, "Exp459ConfigAntigravity")],
        },
        {
            "id": "task2_contracts",
            "path": contracts_path,
            "role": "Caller: blocked_folds, future qualification/evaluation, and Task 2 tests. Effect: validates immutable blocked-fold, fold-evidence, seed-summary, qualification, and decision semantics while binding Wilson/AUC science to Exp458.",
            "tests": [contracts_test, controlled_fork_test],
            "evidence": [evidence_antigravity(contracts_path, "FoldEvidenceAntigravity")],
        },
        {
            "id": "task2_blocked_folds",
            "path": blocked_folds_path,
            "role": "Caller: Task 3 qualification and Task 2 tests. Effect: generates deterministic contiguous query ranges and sorted complement fit indices without randomness.",
            "tests": [blocked_folds_test, controlled_fork_test],
            "evidence": [evidence_antigravity(blocked_folds_path, "blocked_plan_antigravity")],
        },
        {
            "id": "task2_config_tests",
            "path": config_test,
            "role": "Caller: Task 2 verification. Effect: freezes every Exp459 config constant and rejects mutated values, types, and ordering.",
            "tests": [config_test],
            "evidence": [evidence_antigravity(config_test, "test_frozen_config_preserves_the_approved_blocked_validation_contract_antigravity")],
        },
        {
            "id": "task2_contract_tests",
            "path": contracts_test,
            "role": "Caller: Task 2 verification. Effect: proves immutable dataclasses, array ownership, exact blocked-fold evidence, and stable/unstable recomputation.",
            "tests": [contracts_test],
            "evidence": [evidence_antigravity(contracts_test, "test_contract_fields_are_exact_and_instances_are_immutable_antigravity")],
        },
        {
            "id": "task2_blocked_fold_tests",
            "path": blocked_folds_test,
            "role": "Caller: Task 2 verification. Effect: proves the exact 1413-row ranges, general array_split behavior, coverage, and no-RNG blocked planner boundary.",
            "tests": [blocked_folds_test],
            "evidence": [evidence_antigravity(blocked_folds_test, "test_blocked_plan_matches_the_exact_1413_contiguous_ranges_antigravity")],
        },
        {
            "id": "task2_controlled_fork_tests",
            "path": controlled_fork_test,
            "role": "Caller: Task 2 verification. Effect: proves only identity/fold-planner deltas exist while inherited candidate, alpha, Hydra settings, Exp458 AUC/Wilson bindings, runtime, seed, type, threshold, and no-randomness boundaries stay paired.",
            "tests": [controlled_fork_test],
            "evidence": [evidence_antigravity(controlled_fork_test, "test_exp459_config_changes_only_identity_predecessor_and_fold_planner_antigravity")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: governance verification. Effect: freezes the self-contained documents, Task 2 ledger truth, official-root absence, and bounded code-map parity.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 2 validation. Effect: recomputes the bounded Task 2 graph, fingerprints, source-parent rule, and HTML/JSON/lock parity.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
    ]

    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity(design, "Exp459")},
        {"from": "approved_design", "to": "runbook", "type": "governs", "evidence": evidence_antigravity(design, "기준 커밋: Exp458 최종 커밋")},
        {"from": "approved_design", "to": "research_state", "type": "governs", "evidence": evidence_antigravity(design, "Exp458 공식 artifact map")},
        {"from": "approved_design", "to": "task2_config", "type": "governs", "evidence": evidence_antigravity(design, "실행 이름:")},
        {"from": "approved_design", "to": "task2_contracts", "type": "governs", "evidence": evidence_antigravity(design, "stable/unstable")},
        {"from": "approved_design", "to": "task2_blocked_folds", "type": "governs", "evidence": evidence_antigravity(design, "numpy.array_split")},
        {"from": "implementation_plan", "to": "implementation_report", "type": "governs", "evidence": evidence_antigravity(plan, "Step 6 — focused GREEN 두 번, Exp458 contract/AUC 회귀, compile/diff/map 검증 후 체크박스와 map을 갱신하고 커밋·검토한다.")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(plan, "Task 2 완료")},
        {"from": "implementation_plan", "to": "task2_config", "type": "governs", "evidence": evidence_antigravity(plan, "exact config와 immutable dataclass 계약을 구현한다.")},
        {"from": "implementation_plan", "to": "task2_contracts", "type": "governs", "evidence": evidence_antigravity(plan, "FoldEvidenceAntigravity")},
        {"from": "implementation_plan", "to": "task2_blocked_folds", "type": "governs", "evidence": evidence_antigravity(plan, "def blocked_plan_antigravity(n_rows: int):")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": evidence_antigravity(plan, "map을 갱신하고 커밋·검토한다.")},
        {"from": "task2_config_tests", "to": "task2_config", "type": "tests", "evidence": evidence_antigravity(config_test, "test_config_rejects_any_mutated_value_type_or_order_antigravity")},
        {"from": "task2_contract_tests", "to": "task2_contracts", "type": "tests", "evidence": evidence_antigravity(contracts_test, "test_fold_evidence_summary_qualification_and_decision_recompute_semantics_antigravity")},
        {"from": "task2_blocked_fold_tests", "to": "task2_blocked_folds", "type": "tests", "evidence": evidence_antigravity(blocked_folds_test, "test_blocked_plan_source_uses_array_split_without_random_calls_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_config", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_config_changes_only_identity_predecessor_and_fold_planner_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_contracts", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_modules_avoid_forbidden_runtime_expansions_and_bind_science_to_exp458_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_blocked_folds", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_modules_avoid_forbidden_runtime_expansions_and_bind_science_to_exp458_antigravity")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "runbook", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "implementation_report", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "experiment_catalog", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "champion_ledger", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_planned_truth_and_non_promotion_antigravity")},
    ]

    graph = {
        "schema": "exp459_task2_contracts_and_folds_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp459_task2_contracts_and_folds_codemap_lock_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "tracked_fingerprints": tracked_fingerprints_antigravity(),
        "module_fingerprints": module_fingerprints_antigravity(),
        "working_tree_dirty": False,
    }
    return graph, lock, html_antigravity(graph)


def validate_map_antigravity(
    graph: dict[str, object],
    lock: dict[str, object],
    rendered_html: str,
) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph/lock must be dicts")
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("nodes and edges must be lists")
    if graph.get("unknown_relationships") != []:
        raise ValueError("unknown relationships must stay empty")
    ids = set()
    for node in graph["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or not isinstance(node.get("path"), str)
            or not isinstance(node.get("role"), str)
            or not isinstance(node.get("tests"), list)
            or not node["tests"]
            or not isinstance(node.get("evidence"), list)
            or not node["evidence"]
        ):
            raise ValueError("invalid node")
        ids.add(node["id"])
        for item in node["evidence"]:
            if not isinstance(item, dict) or set(item) != {"path", "symbol"}:
                raise ValueError("invalid evidence")
            if not evidence_resolves_antigravity(item["path"], item["symbol"]):
                raise ValueError("unresolved evidence")
    for edge in graph["edges"]:
        if (
            not isinstance(edge, dict)
            or edge.get("from") not in ids
            or edge.get("to") not in ids
            or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY
        ):
            raise ValueError("invalid edge")
        item = edge.get("evidence")
        if not isinstance(item, dict) or set(item) != {"path", "symbol"}:
            raise ValueError("invalid edge evidence")
        if not evidence_resolves_antigravity(item["path"], item["symbol"]):
            raise ValueError("unresolved evidence")

    source = lock.get("source_parent_commit")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()
    if source != graph.get("source_parent_commit") or source not in {head, parent}:
        raise ValueError("invalid source parent")
    if (
        lock.get("working_tree_dirty") is not False
        or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY)
    ):
        raise ValueError("invalid lock")
    if (
        lock.get("tracked_fingerprints") != tracked_fingerprints_antigravity()
        or lock.get("module_fingerprints") != module_fingerprints_antigravity()
    ):
        raise ValueError("fingerprint drift")

    embedded = re.search(
        r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>',
        rendered_html,
        flags=re.DOTALL,
    )
    if not embedded:
        raise ValueError("embedded graph missing")
    embedded_graph = strict_json_antigravity(html_module.unescape(embedded.group(1)))
    if canonical_bytes_antigravity(embedded_graph) != canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-18T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(
            strict_json_antigravity(JSON_PATH_ANTIGRAVITY),
            strict_json_antigravity(LOCK_PATH_ANTIGRAVITY),
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
