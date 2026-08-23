#!/usr/bin/env python3
"""Build and validate the bounded Exp465 Task 5 CLI code-map triplet."""
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
    ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-2-report-antigravity.html",
    ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-3-report-antigravity.html",
    ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-4-report-antigravity.html",
    ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-5-report-antigravity.html",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/config_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/contracts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/interpolation_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/scoring_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/qualification_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/evaluation_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/artifacts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/replay_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/dashboard_export_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/sources_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/cli_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_interpolation_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_sources_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_scoring_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_cli_antigravity.py",
    "scratch/build_codemap_exp465_antigravity.py",
    "scratch/run_exp465_ecg200_c8_external_validation_antigravity.py",
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp465_task5_private_failure_closeout_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp465_task5_private_failure_closeout_codemap_lock_v1"
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
    task2_report = ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-2-report-antigravity.html"
    task3_report = ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-3-report-antigravity.html"
    task4_report = ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-4-report-antigravity.html"
    task5_report = ".superpowers/sdd/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html/task-5-report-antigravity.html"
    sources = "experiments/exp465_ecg200_c8_external_validation_antigravity/sources_antigravity.py"
    interpolation = "experiments/exp465_ecg200_c8_external_validation_antigravity/interpolation_antigravity.py"
    scoring = "experiments/exp465_ecg200_c8_external_validation_antigravity/scoring_antigravity.py"
    sources_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_sources_antigravity.py"
    interpolation_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_interpolation_antigravity.py"
    scoring_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_scoring_antigravity.py"
    qualification = "experiments/exp465_ecg200_c8_external_validation_antigravity/qualification_antigravity.py"
    evaluation = "experiments/exp465_ecg200_c8_external_validation_antigravity/evaluation_antigravity.py"
    artifacts = "experiments/exp465_ecg200_c8_external_validation_antigravity/artifacts_antigravity.py"
    replay = "experiments/exp465_ecg200_c8_external_validation_antigravity/replay_antigravity.py"
    dashboard_export = "experiments/exp465_ecg200_c8_external_validation_antigravity/dashboard_export_antigravity.py"
    qualification_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_qualification_antigravity.py"
    evaluation_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_evaluation_antigravity.py"
    artifacts_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_artifacts_antigravity.py"
    replay_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_replay_antigravity.py"
    dashboard_export_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_dashboard_export_antigravity.py"
    cli = "experiments/exp465_ecg200_c8_external_validation_antigravity/cli_antigravity.py"
    cli_test = "experiments/exp465_ecg200_c8_external_validation_antigravity/tests/test_cli_antigravity.py"
    launcher = "scratch/run_exp465_ecg200_c8_external_validation_antigravity.py"
    governance_symbol = "test_task1_truth_documents_ledgers_and_readonly_codemap_antigravity"
    nodes = [
        _node_antigravity("approved_design", design, "Caller: Exp465 plan and Task 1 contract. Effect: freezes ECG200 TRAIN 100x96 normal69, C0/C6/C8 order with C8-only primary, seeds 20260721/20261721, five folds, onset 56, cosine 0.95, four AUC gates 0.85, mean 0.90, finite/non-degradation/zero-confirmation gates, and 10-of-10 requirement.", "Exp465 — ECG200에서 C8을 그대로 외부 검증", governance_test),
        _node_antigravity("implementation_plan", plan, "Caller: implementers and reviewers. Effect: records Tasks 1–4 reviewed complete, Task 5 CLI and exact two-run private reproduction complete through frozen decision, C8 qualification failure, and Task 6 remaining forbidden.", "Task 5 — 닫힌 CLI와 private 재현 2회", governance_test),
        _node_antigravity("dashboard", dashboard, "Caller: user-facing status review. Effect: reports the byte-identical two-run private result, C8 0/10 and exact minima, official0/private2/label0, absent official root, and Task 6/routing/promotion block.", "Task 5 private qualification은 두 번 똑같이 재현됐지만", governance_test),
        _node_antigravity("preregistration", preregistration, "Caller: governance review. Effect: freezes immutable data/method/gate contract and records Wilson as explanatory only plus CPU fallback acceleration assessment before any scoring run.", "Frozen qualification gates for C8", governance_test),
        _node_antigravity("runbook", runbook, "Caller: operators. Effect: records the exact two-run private qualification failure, keeps official root absent, and blocks Task 6, TEST access, routing research, and post-failure contract drift.", "Task 5 private qualification은 두 번 정확히 재현됐지만", governance_test),
        _node_antigravity("implementation_report", report, "Caller: fresh review and audit. Effect: records Tasks 1–5 scope, exact two-run private reproduction, frozen C8 gate failure, and the resulting Task 6 and routing-research block.", "task_5_private_reproduction_complete_awaiting_final_review", governance_test),
        _node_antigravity("task1_report", task1_report, "Caller: controller and fresh reviewer. Effect: records exact double-GREEN/codemap/diff-check evidence, unchanged official0/private0/label0/null hashes, review PASS, and completed Task 1 checkbox truth.", "task_1_review_pass_complete", governance_test),
        _node_antigravity("task2_report", task2_report, "Caller: controller and reviewer. Effect: records Task 2 source/interpolation RED-to-GREEN, real Zenodo TRAIN smoke hashes and shapes, CPU fallback assessment, fresh review approval, unchanged official0/private0/label0/null governance, and completed Task 2 checkbox truth.", "task_2_review_pass_complete", governance_test),
        _node_antigravity("research_state", state, "Caller: active research ledger. Effect: stores official0/private2/label0, exact two-run private hashes and C8 failure metrics while preserving null official hashes and blocking Task 6/routing/promotion.", "private_qualification_failed_blocked_before_routing_research", governance_test),
        _node_antigravity("experiment_catalog", catalog, "Caller: experiment inventory. Effect: registers one Exp465 entry with the approved contract, private2 exact-reproduction failure summary, official0/label0, and no official result hash.", "private_qualification_failed_blocked_before_routing_research", governance_test),
        _node_antigravity("champion_ledger", champion, "Caller: champion history. Effect: records the reproducible C8 0/10 private failure while preserving champion, pending candidate, routing, promotion, and official result state.", "private_qualification_failed_blocked_before_routing_research", governance_test),
        _node_antigravity("task1_config", config, "Caller: contracts/tests and future source/scoring modules. Effect: freezes exact-type Exp465 identity/data/method/gate/governance values and rejects aliases, mutable/reordered mappings, C6 primary, changed thresholds, and altered counts.", "FROZEN_CONFIG_ANTIGRAVITY", config_test),
        _node_antigravity("task1_contracts", contracts, "Caller: future scoring, qualification, evaluation, artifacts, replay, and CLI tasks. Effect: freezes immutable score/fold/qualification containers and returns initial governance with zero label query, zero official evaluate, and null artifact map; does not load data or score.", "initial_governance_state_antigravity", contracts_test),
        _node_antigravity("task1_config_test", config_test, "Caller: Task 1 TDD verification. Effect: preserves controller RED import failure and covers exact truth, immutability, alias rejection, mutable/reordered mapping rejection, and changed-gate rejection.", "Exp465ConfigTestAntigravity", config_test),
        _node_antigravity("task1_contracts_test", contracts_test, "Caller: Task 1 TDD verification. Effect: covers field inventories, tuple immutability, score/result aliases, nonfinite AUCs, mean consistency, reason/pass agreement, ten-row shape, seed order, and initial governance.", "ContractsAntigravityTest", contracts_test),
        _node_antigravity("task1_governance_test", governance_test, "Caller: required double GREEN and Task 1 review. Effect: verifies files, HTML, ledger truth, document phrases, checkbox completion truth, map graph/evidence/source binding, and read-only validate behavior.", "GovernanceAntigravityTest", governance_test),
        _node_antigravity("map_builder", builder, "Caller: Task 1 boundary validation. Effect: writes canonical HTML/JSON/lock triplet or validates it read-only via exact --validate-only, resolving evidence and fingerprinting tracked paths.", "build_triplet_antigravity", governance_test),
        _node_antigravity("task2_sources", sources, "Caller: future Exp465 scoring/evaluation tasks. Effect: loads only the official ECG200 TRAIN file after TEST-name rejection, frozen MD5, exact rsplit parsing, finite float64 values, exact 100x96 shape and 69/31 labels; returns owned C-contiguous read-only normal 69x96 without opening or deriving TEST data.", "load_ecg200_normal_train_antigravity", sources_test),
        _node_antigravity("task2_interpolation", interpolation, "Caller: future Exp465 scoring/evaluation tasks. Effect: applies only the fixed np.linspace grids and one np.interp per row to map width 96 to 140, preserves endpoints, rejects aliases/nonfinite/wrong shapes, and returns owned C-contiguous read-only float64 output.", "interpolate_to_140_antigravity", interpolation_test),
        _node_antigravity("task2_sources_test", sources_test, "Caller: Task 2 TDD verification. Effect: covers TRAIN-only export, exact fixture contract, ownership/read-only flags, TEST-name rejection before reads, wrong MD5/width/labels/count/nonfinite rejection, and alias-value rejection.", "SourcesAntigravityTest", sources_test),
        _node_antigravity("task2_interpolation_test", interpolation_test, "Caller: Task 2 TDD verification. Effect: covers single-entrypoint boundary, exact np.interp equality, endpoints, constant/nonlinear rows, no input aliasing, owned read-only flags, and shape/dtype/finiteness rejection.", "InterpolationAntigravityTest", interpolation_test),
        _node_antigravity("task3_report", task3_report, "Caller: controller and reviewer. Effect: records Task 3 scoring RED-to-GREEN, exact Exp464 C0/C6/C8 parity across both frozen seeds and five folds, immutable 13-field projection, CPU fallback ruling, unchanged official0/private0/label0/null governance, and independent review approval.", "task_3_review_pass_complete", governance_test),
        _node_antigravity("task3_scoring", scoring, "Caller: Task 4 qualification/replay. Effect: calls the exact Exp464 fold scorer once, projects C0/C6/C8 in frozen order, and retains canonical train140 plus fit/query index evidence for exact C0 replay.", "score_fold_methods_antigravity", scoring_test),
        _node_antigravity("task3_scoring_test", scoring_test, "Caller: Task 3 TDD verification. Effect: proves fixed entrypoint shape, exact Exp464 parity for every required field on both frozen seeds and all five folds, mapping key order, read-only mappings, and a spy showing every entrypoint invocation calls the imported Exp464 scorer exactly once.", "test_entrypoint_spy_proves_one_exp464_scorer_call_per_invocation_antigravity", scoring_test),
        _node_antigravity("task4_report", task4_report, "Caller: controller and reviewer. Effect: records final Task 4 review approval while keeping qualification execution pending.", "task_4_review_pass_complete", qualification_test),
        _node_antigravity("task4_qualification", qualification, "Caller: Task 4 evaluation, publication, replay, and dashboard. Effect: recomputes AUC, mean, finite state, confirmations, 69 OOF totals, seed Wilson explanatory value, C0/C8 pair diagnostics, C8 non-degradation, exact C0 input replay, and C6/C8 raw-evidence replay.", "qualify_methods_antigravity", qualification_test),
        _node_antigravity("task4_evaluation", evaluation, "Caller: publication, replay, and dashboard. Effect: applies only the C8 10/10, zero-confirmation, finite, and C0 gradual/period non-degradation decision; it does not route or access labels.", "evaluate_qualification_antigravity", evaluation_test),
        _node_antigravity("task4_artifacts", artifacts, "Caller: temporary Task 4 publication and replay. Effect: writes canonical context, metrics, failed rows, summaries, decision, dashboard, and SHA-256 map leaves to a new path.", "publish_result_antigravity", artifacts_test),
        _node_antigravity("task4_replay", replay, "Caller: artifact inspection. Effect: reconstructs score-derived evidence and rejects noncanonical or semantically altered output.", "inspect_replay_at_antigravity", replay_test),
        _node_antigravity("task4_dashboard_export", dashboard_export, "Caller: canonical publication dashboard leaf. Effect: creates a simple Korean, label-free C8 qualification snapshot from recomputed evidence.", "export_dashboard_antigravity", dashboard_export_test),
        _node_antigravity("task4_qualification_test", qualification_test, "Caller: Task 4 focused TDD verification. Effect: tests ten C8 records and rejects inconsistent confirmation evidence without opening data files.", "QualificationAntigravityTest", qualification_test),
        _node_antigravity("task4_evaluation_test", evaluation_test, "Caller: Task 4 focused TDD verification. Effect: proves C8 is the sole decision candidate and zero normal confirmation is required.", "EvaluationAntigravityTest", evaluation_test),
        _node_antigravity("task4_artifacts_test", artifacts_test, "Caller: Task 4 focused TDD verification. Effect: proves canonical publication creates exactly seven leaves.", "ArtifactsAntigravityTest", artifacts_test),
        _node_antigravity("task4_replay_test", replay_test, "Caller: Task 4 focused TDD verification. Effect: proves semantic inspection accepts its own temporary publication and rejects a changed decision leaf.", "ReplayAntigravityTest", replay_test),
        _node_antigravity("task4_dashboard_export_test", dashboard_export_test, "Caller: Task 4 focused TDD verification. Effect: proves the C8 dashboard snapshot uses simple Korean summary fields.", "DashboardExportAntigravityTest", dashboard_export_test),
        _node_antigravity("task5_report", task5_report, "Caller: controller and fresh reviewer. Effect: preserves the pre-execution Task 5 CLI RED/GREEN and review-correction evidence; current two-run private failure truth is recorded by the dashboard, implementation report, plan, and ledgers.", "Exp465 Task 5 — 닫힌 CLI 구현 보고", cli_test),
        _node_antigravity("task5_cli", cli, "Caller: fixed-runtime launcher and private qualification harness. Effect: accepts only bare inspect or reviewed evaluate, reads the fixed TRAIN path only during evaluate, scores the frozen 2×5 folds, publishes one new private root, and inspects persisted leaves without calling the Exp464/Task 3 scorer. It never exposes root/seed/fold/gate/retry/resume arguments or creates the official root.", "main_antigravity", cli_test),
        _node_antigravity("task5_launcher", launcher, "Caller: private Task 5 operator. Effect: re-execs the fixed interpreter outside the caller CWD and maps only preregistered run1/run2 capabilities to their fixed private roots; arbitrary root selection remains unavailable.", "_install_private_capability_antigravity", cli_test),
        _node_antigravity("task5_cli_test", cli_test, "Caller: Task 5 TDD and fresh review. Effect: covers exact parser surface, fixed runtime, authorized launcher capability, root-only seam, create-only root, clean commit, duplicate process, real lower-layer private fixture evaluate/inspect equality, label0, and scorer-free inspect.", "CliAntigravityTest", cli_test),
    ]
    edges = [
        _edge_antigravity("approved_design", "governs", "task1_config", design, "C8 합격 조건"),
        _edge_antigravity("approved_design", "governs", "preregistration", design, "C8 합격 조건"),
        _edge_antigravity("implementation_plan", "governs", "task1_config", plan, "Step 2 — Immutable config/contracts."),
        _edge_antigravity("implementation_plan", "governs", "map_builder", plan, "Step 4 — Code map."),
        _edge_antigravity("task1_config", "governs", "task1_contracts", config, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("implementation_plan", "governs", "task2_sources", plan, "Step 2 — Parser."),
        _edge_antigravity("implementation_plan", "governs", "task2_interpolation", plan, "Step 4 — Minimal interpolation."),
        _edge_antigravity("implementation_plan", "governs", "task3_scoring", plan, "Step 3 — Projection only."),
        _edge_antigravity("approved_design", "governs", "task2_sources", design, "자료 계약"),
        _edge_antigravity("approved_design", "governs", "task2_interpolation", design, "길이 96을 140으로 바꾸는 고정 규칙"),
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
        _edge_antigravity("task2_sources_test", "tests", "task2_sources", sources_test, "test_loader_returns_exact_normal_train_contract_antigravity"),
        _edge_antigravity("task2_interpolation_test", "tests", "task2_interpolation", interpolation_test, "test_exact_formula_matches_reference_rows_endpoints_and_flags_antigravity"),
        _edge_antigravity("task3_scoring_test", "tests", "task3_scoring", scoring_test, "test_golden_ten_seed_fold_calls_have_exact_exp464_parity_antigravity"),
        _edge_antigravity("task3_scoring_test", "tests", "task3_scoring", scoring_test, "test_entrypoint_spy_proves_one_exp464_scorer_call_per_invocation_antigravity"),
        _edge_antigravity("task1_governance_test", "tests", "task3_report", governance_test, governance_symbol),
        _edge_antigravity("map_builder", "documents_boundary", "task1_config", builder, "build_triplet_antigravity"),
        _edge_antigravity("implementation_plan", "governs", "task4_qualification", plan, "Task 4 — 판정, 결과, 재생, 대시보드"),
        _edge_antigravity("task3_scoring", "governs", "task4_qualification", scoring, "decode_train140_antigravity"),
        _edge_antigravity("task1_config", "governs", "task4_evaluation", config, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task4_qualification", "governs", "task4_evaluation", qualification, "recompute_qualification_antigravity"),
        _edge_antigravity("task4_qualification", "governs", "task4_artifacts", qualification, "Exp465QualificationResultAntigravity"),
        _edge_antigravity("task4_evaluation", "governs", "task4_artifacts", evaluation, "QualificationDecisionAntigravity"),
        _edge_antigravity("task4_evaluation", "governs", "task4_dashboard_export", evaluation, "QualificationDecisionAntigravity"),
        _edge_antigravity("task4_artifacts", "governs", "task4_replay", artifacts, "_LEAVES_ANTIGRAVITY"),
        _edge_antigravity("task4_qualification_test", "tests", "task4_qualification", qualification_test, "test_recomputes_c8_ten_of_ten_from_raw_scores_antigravity"),
        _edge_antigravity("task4_evaluation_test", "tests", "task4_evaluation", evaluation_test, "test_c8_only_decision_requires_zero_confirmation_antigravity"),
        _edge_antigravity("task4_artifacts_test", "tests", "task4_artifacts", artifacts_test, "test_publish_creates_exact_seven_leaves_antigravity"),
        _edge_antigravity("task4_replay_test", "tests", "task4_replay", replay_test, "test_semantic_replay_rejects_tampered_decision_antigravity"),
        _edge_antigravity("task4_dashboard_export_test", "tests", "task4_dashboard_export", dashboard_export_test, "test_dashboard_uses_simple_korean_and_c8_summary_antigravity"),
        _edge_antigravity("task4_report", "documents_boundary", "task4_qualification", task4_report, "task_4_review_pass_complete"),
        _edge_antigravity("implementation_plan", "governs", "task5_cli", plan, "Task 5 — 닫힌 CLI와 private 재현 2회"),
        _edge_antigravity("task2_sources", "governs", "task5_cli", sources, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("task2_interpolation", "governs", "task5_cli", interpolation, "interpolate_to_140_antigravity"),
        _edge_antigravity("task3_scoring", "governs", "task5_cli", scoring, "score_fold_methods_antigravity"),
        _edge_antigravity("task4_qualification", "governs", "task5_cli", qualification, "qualify_methods_antigravity"),
        _edge_antigravity("task4_evaluation", "governs", "task5_cli", evaluation, "evaluate_qualification_antigravity"),
        _edge_antigravity("task4_artifacts", "governs", "task5_cli", artifacts, "publish_result_antigravity"),
        _edge_antigravity("task5_launcher", "governs", "task5_cli", launcher, "_install_private_capability_antigravity"),
        _edge_antigravity("task5_cli", "governs", "task4_artifacts", cli, "_evaluate_at_root_antigravity"),
        _edge_antigravity("task5_cli", "governs", "dashboard", cli, "_emit_payload_antigravity"),
        _edge_antigravity("task5_cli", "governs", "research_state", cli, "_emit_payload_antigravity"),
        _edge_antigravity("task5_cli", "governs", "experiment_catalog", cli, "_emit_payload_antigravity"),
        _edge_antigravity("task5_cli", "governs", "champion_ledger", cli, "_emit_payload_antigravity"),
        _edge_antigravity("task5_cli_test", "tests", "task5_cli", cli_test, "test_private_evaluate_then_inspect_uses_real_lower_layers_and_label_zero_antigravity"),
        _edge_antigravity("task5_cli_test", "tests", "task5_launcher", cli_test, "test_launcher_accepts_only_preregistered_private_capabilities_antigravity"),
        _edge_antigravity("task5_report", "documents_boundary", "task5_cli", task5_report, "Exp465 Task 5 — 닫힌 CLI 구현 보고"),
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
            "design/plan -> config -> contracts -> Task2 sources/interpolation -> future scoring/qualification",
            "Task2 source RED/GREEN + interpolation RED/GREEN -> real TRAIN smoke and document truth",
            "Task3 entrypoint -> imported Exp464 scorer once -> C0/C6/C8 exact projection",
            "Task3 input evidence + raw scores -> 69-OOF/C0/C6/C8 semantic qualification -> C8-only decision -> seven canonical leaves -> semantic replay/dashboard snapshot",
            "Task5 closed CLI -> fixed TRAIN and 2x5 fold assembly -> Task3/Task4 publication -> scorer-free persisted-leaf inspection -> byte-identical private2 C8 0/10 failure -> Task6 and routing research blocked",
            "config/contracts/scoring tests + governance regression -> documents, ledgers, dashboard, plan truth",
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
        '<title>Exp465 Task 5 코드맵</title><style>body{max-width:1160px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}pre{padding:1rem}</style></head><body><main>'
        '<h1>Exp465 Task 5 코드맵</h1><p>이 map은 frozen ECG200 C8 contract, TRAIN-only loader, 96→140 interpolation, Exp464-bound C0/C6/C8 projection, Task 4 qualification/publication, 그리고 Task 5의 닫힌 evaluate/inspect CLI를 설명한다. launcher는 fixed runtime과 run1/run2 private capability만 연결하고 임의 root 인자를 허용하지 않는다. inspect는 저장된 leaves를 scorer 없이 재검산한다. 두 private 실행은 byte-for-byte 일치했지만 C8이 0/10으로 실패했으므로 Task 6과 라우팅 연구는 차단됐다. 공식 실행 결과는 없다.</p>'
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
