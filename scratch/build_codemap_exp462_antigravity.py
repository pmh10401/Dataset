#!/usr/bin/env python3
"""Build and validate the bounded Exp462 Task 3 code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp462_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp462_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp462_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp462_antigravity.html",
    "docs/codemap/codemap_exp462_antigravity.json",
    "docs/codemap/codemap_exp462_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP462_DASHBOARD_antigravity.html",
    "docs/EXP462_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP462_PREREGISTRATION_antigravity.html",
    "docs/EXP462_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html",
    "docs/superpowers/specs/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity-design.html",
    ".superpowers/sdd/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html/task-2-report-antigravity.html",
    ".superpowers/sdd/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html/task-3-report-antigravity.html",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/config_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/contracts_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/gradual_features_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/predecessor_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/scoring_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/sources_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_config_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_gradual_features_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_scoring_antigravity.py",
    "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/test_sources_antigravity.py",
    "scratch/build_codemap_exp462_antigravity.py",
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


def strict_json_antigravity(text_or_path: str | Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    text = (
        text_or_path.read_text(encoding="utf-8")
        if isinstance(text_or_path, Path)
        else text_or_path
    )
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
        f'<li><code>{html_module.escape(str(node["path"]))}</code> - {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp462 Task 3 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}pre{padding:1rem}</style></head><body><main>'
        '<h1>Exp462 Task 3 코드맵</h1><p>Task 3 is implemented pending review and maps governance documents, frozen Task 2 authority/source nodes, gradual feature, scoring boundary, and focused tests. Future Task 4-5 boundaries remain explicitly unknown and not implemented.</p>'
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

    design = "docs/superpowers/specs/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html"
    preregistration = "docs/EXP462_PREREGISTRATION_antigravity.html"
    runbook = "docs/EXP462_RUNBOOK_antigravity.html"
    report = "docs/EXP462_IMPLEMENTATION_REPORT_antigravity.html"
    dashboard = "docs/EXP462_DASHBOARD_antigravity.html"
    task2_report = ".superpowers/sdd/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html/task-2-report-antigravity.html"
    task3_report = ".superpowers/sdd/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html/task-3-report-antigravity.html"
    governance_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    builder = "scratch/build_codemap_exp462_antigravity.py"
    config = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/config_antigravity.py"
    )
    contracts = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/contracts_antigravity.py"
    )
    predecessor = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/predecessor_antigravity.py"
    )
    sources = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/sources_antigravity.py"
    )
    gradual_features = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/gradual_features_antigravity.py"
    )
    scoring = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/scoring_antigravity.py"
    )
    config_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_config_antigravity.py"
    )
    contracts_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_contracts_antigravity.py"
    )
    predecessor_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_predecessor_antigravity.py"
    )
    sources_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_sources_antigravity.py"
    )
    gradual_features_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_gradual_features_antigravity.py"
    )
    scoring_test = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/tests/"
        "test_scoring_antigravity.py"
    )
    task4_boundary = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/qualification_antigravity.py"
    )
    task5_boundary = (
        "experiments/exp462_ecg_gradual_signature_qualification_antigravity/cli_antigravity.py"
    )

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Task 1-6 governance and future implementation. Effect: freezes Exp462 identity, TRAIN-only boundary, gradual and period preservation targets, and official-root absence rule.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp462 — ECG 점진적 변화 특징 검증 설계")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Task 1-6 execution. Effect: records Task 3 implemented pending review, checked Task 1-2 controls, unchecked Task 3 controls, official0/label0, and future Task 4-5 boundaries.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 3 implemented pending review")],
        },
        {
            "id": "preregistration",
            "path": preregistration,
            "role": "Caller: governance history. Effect: preserves planned-only governance, target thresholds, official command count 0, label query count 0, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(preregistration, "planned-only governance")],
        },
        {
            "id": "runbook",
            "path": runbook,
            "role": "Caller: future operators. Effect: blocks official evaluate/inspect, source loading, labels, routing, champion mutation, and pending-candidate mutation during Task 1.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(runbook, "Task 1 governance ready / Task 2 next")],
        },
        {
            "id": "implementation_report",
            "path": report,
            "role": "Caller: Task 3 handoff. Effect: records frozen authority, Task 3 feature/scoring public interfaces, backend decision, null official hashes, official0/label0, and review-pending state.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(report, "Task 3 implementation")],
        },
        {
            "id": "dashboard",
            "path": dashboard,
            "role": "Caller: always-visible status review. Effect: shows Task 3 implemented pending review, public interfaces, selected CPU backend, official0, label0, and null hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(dashboard, "Task 3 implemented pending review")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records the Exp462 planned entry with zero official counts, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp462\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp462 planned governance without official result or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp462\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp462 planned governance history without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp462\"")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: Task 1 validation. Effect: rejects missing docs, ledger drift, checked pending controls, official lifecycle drift, label access, result hashes, and invalid map bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_task1_planned_governance_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 3 validation. Effect: rebuilds and validates the bounded Exp462 graph plus stored HTML/JSON/lock triplet bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
        {
            "id": "task2_report",
            "path": task2_report,
            "role": "Caller: Task 2 review. Effect: records complete state, RED/GREEN boundary, frozen Exp461 authority, and Task 3 next.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(task2_report, "Exp462 Task 2 report")],
        },
        {
            "id": "task3_report",
            "path": task3_report,
            "role": "Caller: Task 3 review. Effect: records implemented-pending-review state, RED/GREEN evidence, backend decision, public interfaces, official0, label0, and absent official root.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(task3_report, "Exp462 Task 3 report")],
        },
        {
            "id": "task2_config",
            "path": config,
            "role": "Caller: Task 3 scoring and future qualification. Effect: freezes Exp462 identity, TRAIN shape, seeds, folds, method order, gates, Exp461 reviewed code commit, Exp461 terminal closeout commit, private hashes, official0, and label0.",
            "tests": [config_test, governance_test],
            "evidence": [evidence_antigravity(config, "FROZEN_CONFIG_ANTIGRAVITY")],
        },
        {
            "id": "task2_contracts",
            "path": contracts,
            "role": "Caller: predecessor capture and future replay. Effect: rejects mutable or drifted Exp461 authority and exposes strict period preservation rows.",
            "tests": [contracts_test, governance_test],
            "evidence": [evidence_antigravity(contracts, "Exp461GradualAuthorityAntigravity")],
        },
        {
            "id": "task2_predecessor",
            "path": predecessor,
            "role": "Caller: Task 3 scoring and future context artifacts. Effect: returns the exact source-free Exp461 terminal authority consumed by Exp462.",
            "tests": [predecessor_test, governance_test],
            "evidence": [evidence_antigravity(predecessor, "capture_exp461_gradual_authority_antigravity")],
        },
        {
            "id": "task2_sources",
            "path": sources,
            "role": "Caller: Task 3 scoring and future qualification. Effect: reuses reviewed Exp461 TRAIN loading and returns an owned read-only finite float32 1413x140 matrix without direct TEST or label queries.",
            "tests": [sources_test, governance_test],
            "evidence": [evidence_antigravity(sources, "load_normal2_train_antigravity")],
        },
        {
            "id": "task2_config_tests",
            "path": config_test,
            "role": "Caller: Task 2 focused regression. Effect: covers frozen config identity, order, gates, and strict mutation rejection.",
            "tests": [config_test],
            "evidence": [evidence_antigravity(config_test, "test_frozen_config_preserves_task2_identity_and_gates_antigravity")],
        },
        {
            "id": "task2_contract_tests",
            "path": contracts_test,
            "role": "Caller: Task 2 focused regression. Effect: covers authority dataclass immutability, hash mapping immutability, and validator drift rejection.",
            "tests": [contracts_test],
            "evidence": [evidence_antigravity(contracts_test, "test_authority_contract_fields_and_immutability_antigravity")],
        },
        {
            "id": "task2_predecessor_tests",
            "path": predecessor_test,
            "role": "Caller: Task 2 focused regression. Effect: covers exact Exp461 reviewed code and terminal closeout commits, values, hashes, historical period folds, and source-free capture.",
            "tests": [predecessor_test],
            "evidence": [evidence_antigravity(predecessor_test, "test_capture_returns_exact_frozen_exp461_authority_antigravity")],
        },
        {
            "id": "task2_source_tests",
            "path": sources_test,
            "role": "Caller: Task 2 focused regression. Effect: covers TRAIN source shape, dtype, finite values, ownership, immutability, no aliasing, and label-free module surface.",
            "tests": [sources_test],
            "evidence": [evidence_antigravity(sources_test, "test_loader_reuses_exp461_train_matrix_without_aliasing_antigravity")],
        },
        {
            "id": "task3_gradual_signature",
            "path": gradual_features,
            "role": "Caller: Task 3 scoring and future qualification. Effect: validates one finite 140-length row and returns exact 24D gradual signatures for full/front/back views as owned read-only float64.",
            "tests": [gradual_features_test, governance_test],
            "evidence": [evidence_antigravity(gradual_features, "gradual_signature_antigravity")],
        },
        {
            "id": "task3_scoring",
            "path": scoring,
            "role": "Caller: future Task 4 qualification. Effect: delegates Exp461 hydra_period_rank_max, computes gradual_signature_local_gap with TRAIN-only LocalGap, ranks scores, and returns hydra_period_gradual_rank_max as exact elementwise maximum.",
            "tests": [scoring_test, governance_test],
            "evidence": [evidence_antigravity(scoring, "score_fold_methods_antigravity")],
        },
        {
            "id": "task3_gradual_feature_tests",
            "path": gradual_features_test,
            "role": "Caller: Task 3 focused regression. Effect: covers independent NumPy feature reference, order, shape, dtype, finite values, ownership, immutability, input preservation, and rejection matrix.",
            "tests": [gradual_features_test],
            "evidence": [evidence_antigravity(gradual_features_test, "test_gradual_signature_matches_independent_reference_and_contract_antigravity")],
        },
        {
            "id": "task3_scoring_tests",
            "path": scoring_test,
            "role": "Caller: Task 3 focused regression. Effect: covers method order, Exp461 baseline delegation, LocalGap index/seed flow, exact rank formula, primary max fusion, and public boundary restrictions.",
            "tests": [scoring_test],
            "evidence": [evidence_antigravity(scoring_test, "test_fold_scoring_delegates_baseline_and_combines_gradual_rank_antigravity")],
        },
        {
            "id": "task4_qualification_unknown",
            "path": task4_boundary,
            "role": "Caller: future Task 4 only. Effect: unknown and not implemented in Task 1; expected to implement blocked qualification and replay later.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 4 — blocked qualification, seven leaves, replay")],
        },
        {
            "id": "task5_cli_unknown",
            "path": task5_boundary,
            "role": "Caller: future Task 5 only. Effect: unknown and not implemented in Task 1; expected to implement the closed CLI and private TRAIN reproductions later.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 5 — closed CLI and two private TRAIN reproductions")],
        },
    ]

    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity(preregistration, "gradual_drift AUC ≥ 0.85")},
        {"from": "approved_design", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(dashboard, "0.9216148584075248")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(dashboard, "Task 3 implemented pending review")},
        {"from": "implementation_plan", "to": "governance_regression", "type": "governs", "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_planned_governance_antigravity")},
        {"from": "governance_regression", "to": "runbook", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity")},
        {"from": "governance_regression", "to": "implementation_report", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_planned_governance_antigravity")},
        {"from": "governance_regression", "to": "experiment_catalog", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_planned_governance_antigravity")},
        {"from": "governance_regression", "to": "champion_ledger", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_planned_governance_antigravity")},
        {"from": "governance_regression", "to": "map_builder", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity")},
        {"from": "implementation_plan", "to": "task2_config", "type": "governs", "evidence": evidence_antigravity(plan, "Task 2 — frozen Exp461 authority and TRAIN source")},
        {"from": "implementation_plan", "to": "task2_report", "type": "governs", "evidence": evidence_antigravity(task2_report, "Task 2 complete / Task 3 next")},
        {"from": "task2_config_tests", "to": "task2_config", "type": "tests", "evidence": evidence_antigravity(config_test, "test_frozen_config_preserves_task2_identity_and_gates_antigravity")},
        {"from": "task2_contract_tests", "to": "task2_contracts", "type": "tests", "evidence": evidence_antigravity(contracts_test, "test_validator_rejects_snapshot_drift_and_mutable_containers_antigravity")},
        {"from": "task2_predecessor_tests", "to": "task2_predecessor", "type": "tests", "evidence": evidence_antigravity(predecessor_test, "test_capture_returns_exact_frozen_exp461_authority_antigravity")},
        {"from": "task2_source_tests", "to": "task2_sources", "type": "tests", "evidence": evidence_antigravity(sources_test, "test_loader_reuses_exp461_train_matrix_without_aliasing_antigravity")},
        {"from": "task2_config", "to": "task2_contracts", "type": "governs", "evidence": evidence_antigravity(config, "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "task2_contracts", "to": "task2_predecessor", "type": "governs", "evidence": evidence_antigravity(contracts, "Exp461GradualAuthorityAntigravity")},
        {"from": "task2_config", "to": "task2_sources", "type": "governs", "evidence": evidence_antigravity(sources, "load_normal2_train_antigravity")},
        {"from": "implementation_plan", "to": "task3_report", "type": "governs", "evidence": evidence_antigravity(task3_report, "Task 3 implemented pending review")},
        {"from": "implementation_plan", "to": "task3_gradual_signature", "type": "governs", "evidence": evidence_antigravity(plan, "Task 3 — 24D gradual signature and three scores")},
        {"from": "implementation_plan", "to": "task3_scoring", "type": "governs", "evidence": evidence_antigravity(plan, "Task 3 — 24D gradual signature and three scores")},
        {"from": "task3_gradual_feature_tests", "to": "task3_gradual_signature", "type": "tests", "evidence": evidence_antigravity(gradual_features_test, "test_gradual_signature_matches_independent_reference_and_contract_antigravity")},
        {"from": "task3_scoring_tests", "to": "task3_scoring", "type": "tests", "evidence": evidence_antigravity(scoring_test, "test_fold_scoring_delegates_baseline_and_combines_gradual_rank_antigravity")},
        {"from": "task2_config", "to": "task3_scoring", "type": "governs", "evidence": evidence_antigravity(config, "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "task2_predecessor", "to": "task3_scoring", "type": "governs", "evidence": evidence_antigravity(predecessor, "capture_exp461_gradual_authority_antigravity")},
        {"from": "task2_sources", "to": "task3_scoring", "type": "governs", "evidence": evidence_antigravity(sources, "load_normal2_train_antigravity")},
        {"from": "task3_gradual_signature", "to": "task3_scoring", "type": "governs", "evidence": evidence_antigravity(scoring, "score_fold_methods_antigravity")},
        {"from": "implementation_plan", "to": "task4_qualification_unknown", "type": "governs", "evidence": evidence_antigravity(plan, "Task 4 — blocked qualification, seven leaves, replay")},
        {"from": "implementation_plan", "to": "task5_cli_unknown", "type": "governs", "evidence": evidence_antigravity(plan, "Task 5 — closed CLI and two private TRAIN reproductions")},
    ]

    graph = {
        "schema": "exp462_task3_boundary_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp462_task3_boundary_codemap_lock_v1",
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
                raise ValueError(f"unresolved evidence: {item}")
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
            raise ValueError(f"unresolved edge evidence: {item}")

    source = lock.get("source_parent_commit")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()
    parents = subprocess.check_output(
        ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip().split()[1:]
    if source != graph.get("source_parent_commit") or source not in {head, *parents}:
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


def write_map_antigravity(source_commit: str, generated_at: str) -> None:
    graph, lock, rendered = build_map_antigravity(source_commit, generated_at)
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    JSON_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(graph) + b"\n")
    LOCK_PATH_ANTIGRAVITY.write_bytes(canonical_bytes_antigravity(lock) + b"\n")
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


def main_antigravity_validate_only_for_tests() -> None:
    validate_map_antigravity(
        strict_json_antigravity(JSON_PATH_ANTIGRAVITY),
        strict_json_antigravity(LOCK_PATH_ANTIGRAVITY),
        HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8"),
    )


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-20T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        main_antigravity_validate_only_for_tests()
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    write_map_antigravity(args.source_commit, args.generated_at)


if __name__ == "__main__":
    main_antigravity()
