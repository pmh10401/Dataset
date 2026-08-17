#!/usr/bin/env python3
"""Build and validate the bounded Task 5 Exp457 candidate code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp457_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp457_antigravity.html",
    "docs/codemap/codemap_exp457_antigravity.json",
    "docs/codemap/codemap_exp457_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP457_DASHBOARD_antigravity.html",
    "docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP457_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP457_RUNBOOK_antigravity_KO.html",
    "docs/superpowers/plans/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity.html",
    "docs/superpowers/specs/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity-design.html",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/config_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/cli_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/contracts_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/dashboard_export_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/evaluation_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/artifacts_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/perturbations_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/qualification_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/replay_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/router_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/sources_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_config_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_cli_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_perturbations_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_router_antigravity.py",
    "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity/tests/test_sources_antigravity.py",
    "scratch/build_codemap_exp457_antigravity.py",
    "scratch/run_exp457_ecg_train_synthetic_confidence_routing_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "tests", "future_unknown"}


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


def tracked_fingerprints_antigravity() -> dict[str, str]:
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
    payload = html_module.escape(canonical_bytes_antigravity(graph).decode("utf-8"))
    nodes = "".join(
        f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp457 Task 5 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp457 Task 5 candidate 코드맵</h1><p>구현된 TRAIN qualification, conservative router, immutable Exp456 aggregation, 9-leaf replay, closed CLI와 outside-CWD launcher를 연결한다. '
        '실제 TRAIN 후보 검증은 비공식 임시 폴더에서 두 번 재현되었고 공식 result는 아직 없다.</p>'
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
    plan = "docs/superpowers/plans/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity.html"
    design = "docs/superpowers/specs/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity-design.html"
    package = "experiments/exp457_ecg_train_synthetic_confidence_routing_antigravity"
    governance_test = f"{package}/tests/test_governance_antigravity.py"
    config_test = f"{package}/tests/test_config_antigravity.py"
    contracts_test = f"{package}/tests/test_contracts_antigravity.py"
    perturbations_test = f"{package}/tests/test_perturbations_antigravity.py"
    qualification_test = f"{package}/tests/test_qualification_antigravity.py"
    sources_test = f"{package}/tests/test_sources_antigravity.py"
    router_test = f"{package}/tests/test_router_antigravity.py"
    evaluation_test = f"{package}/tests/test_evaluation_antigravity.py"
    artifacts_test = f"{package}/tests/test_artifacts_antigravity.py"
    replay_test = f"{package}/tests/test_replay_antigravity.py"
    dashboard_export_test = f"{package}/tests/test_dashboard_export_antigravity.py"
    cli_test = f"{package}/tests/test_cli_antigravity.py"
    config = f"{package}/config_antigravity.py"
    contracts = f"{package}/contracts_antigravity.py"
    perturbations = f"{package}/perturbations_antigravity.py"
    qualification = f"{package}/qualification_antigravity.py"
    sources = f"{package}/sources_antigravity.py"
    router = f"{package}/router_antigravity.py"
    evaluation = f"{package}/evaluation_antigravity.py"
    artifacts = f"{package}/artifacts_antigravity.py"
    replay = f"{package}/replay_antigravity.py"
    dashboard_export = f"{package}/dashboard_export_antigravity.py"
    cli = f"{package}/cli_antigravity.py"
    cli_launcher = "scratch/run_exp457_ecg_train_synthetic_confidence_routing_antigravity.py"
    builder = "scratch/build_codemap_exp457_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": design, "role": "Caller: Tasks 1–5 governance, qualification, routing, evaluation, and candidate flow. Effect: freezes the only scientific change and exact confidence/result gates.", "tests": [governance_test, config_test, qualification_test, router_test, evaluation_test, cli_test], "evidence": [evidence_antigravity(design, "Exp457")]},
        {"id": "implementation_plan", "path": plan, "role": "Caller: Tasks 1–6. Effect: sequences TDD implementation, permits Task 5 temporary candidate qualification, and keeps official execution in Task 6.", "tests": [governance_test, router_test, evaluation_test, artifacts_test, replay_test, dashboard_export_test, cli_test], "evidence": [evidence_antigravity(plan, "Task 5: 닫힌 CLI와 공식 후보 검증")]},
        {"id": "preregistration", "path": "docs/EXP457_PREREGISTRATION_antigravity_KO.html", "role": "Caller: governance review. Effect: records planned identity, frozen gates, and TEST routing firewall.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_PREREGISTRATION_antigravity_KO.html", "planned_awaiting_implementation")]},
        {"id": "runbook", "path": "docs/EXP457_RUNBOOK_antigravity_KO.html", "role": "Caller: future task operators. Effect: prohibits Task 1 official execution and result-root use.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_RUNBOOK_antigravity_KO.html", "Task 1 governance only")]},
        {"id": "implementation_report", "path": "docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html", "role": "Caller: Task 5 review. Effect: reports exact twice-reproduced TRAIN candidate evidence without claiming an official result.", "tests": [], "evidence": [evidence_antigravity("docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html", "Task 5/6")]},
        {"id": "dashboard", "path": "docs/EXP457_DASHBOARD_antigravity.html", "role": "Caller: result-facing review. Effect: shows Task 5/6, exact TRAIN gates/routes and temporary aggregate, no official result, and Task 6 approval boundary.", "tests": [governance_test], "evidence": [evidence_antigravity("docs/EXP457_DASHBOARD_antigravity.html", "Task 5/6")]},
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Caller: governance review. Effect: records implementation verification while retaining zero official invocations and null result hashes.", "tests": [governance_test], "evidence": [evidence_antigravity("RESEARCH_STATE.json", "implementation_verified_awaiting_official_evaluation")]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Caller: catalog review. Effect: registers Exp457 documents, frozen contract, and no-result truth.", "tests": [governance_test], "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"id\": \"Exp457\"")]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "Caller: champion review. Effect: records Exp457 without changing champion or pending candidate.", "tests": [governance_test], "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp457\"")]},
        {"id": "governance_regression", "path": governance_test, "role": "Caller: Tasks 1–5 scientific-governance verification. Effect: rejects gate drift, fabricated official results, lifecycle changes, or stale Task 5 boundaries.", "tests": [governance_test], "evidence": [evidence_antigravity(governance_test, "test_exp457_scientific_contract_antigravity")]},
        {"id": "map_builder", "path": builder, "role": "Caller: Task 5 map validation. Effect: validates implemented CLI/launcher evidence, fingerprints, and HTML/JSON parity.", "tests": [governance_test], "evidence": [evidence_antigravity(builder, "validate_map_antigravity")]},
        {"id": "frozen_config", "path": config, "role": "Caller: Task 3 qualification and Task 4 evaluation. Effect: supplies exact datasets, Hydra parameters, gates, primary order, and final thresholds.", "tests": [config_test], "evidence": [evidence_antigravity(config, "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "result_contracts", "path": contracts, "role": "Caller: qualification, router, evaluation, publication, and replay. Effect: recomputes Wilson, pooled and per-type Gate 1 semantics and rejects mismatched aggregate flags or malformed result records.", "tests": [contracts_test, qualification_test, router_test, evaluation_test, artifacts_test, replay_test], "evidence": [evidence_antigravity(contracts, "validate_train_qualification_semantics_antigravity")]},
        {"id": "primary_perturbations", "path": perturbations, "role": "Caller: Task 3 TRAIN qualification. Effect: creates four deterministic Exp417-compatible primary perturbations and excludes spike.", "tests": [perturbations_test, qualification_test], "evidence": [evidence_antigravity(perturbations, "build_primary_perturbations_antigravity")]},
        {"id": "config_regression", "path": config_test, "role": "Caller: Task 2 verification. Effect: rejects drift in every result-affecting frozen configuration value.", "tests": [config_test], "evidence": [evidence_antigravity(config_test, "class ConfigAntigravityTest")]},
        {"id": "contracts_regression", "path": contracts_test, "role": "Caller: Task 2 and Task 4 verification. Effect: rejects wrong types, datasets, seeds, primary order, sensitivities, recomputed Gate 1 numerics and aggregate flags.", "tests": [contracts_test], "evidence": [evidence_antigravity(contracts_test, "test_qualification_recomputes_numeric_seed_and_aggregate_gate_semantics_antigravity")]},
        {"id": "perturbations_regression", "path": perturbations_test, "role": "Caller: Task 2 verification. Effect: freezes onset, robust scale, magnitudes, order, shape, immutability, and period interpolation.", "tests": [perturbations_test], "evidence": [evidence_antigravity(perturbations_test, "class PrimaryPerturbationsAntigravityTest")]},
        {"id": "train_source", "path": sources, "role": "Caller: Task 3 qualification and read-only smoke. Effect: selects only exact-index ECG TRAIN values into owned immutable float32 matrices.", "tests": [sources_test], "evidence": [evidence_antigravity(sources, "load_ecg_train_antigravity")]},
        {"id": "train_qualification", "path": qualification, "role": "Caller: Task 4 router. Effect: computes two-seed five-fold Hydra+LocalGap OOF and synthetic Gate 1 evidence using the complete normal OOF reference.", "tests": [qualification_test, router_test], "evidence": [evidence_antigravity(qualification, "qualify_ecg_dataset_antigravity")]},
        {"id": "source_regression", "path": sources_test, "role": "Caller: Task 3 verification. Effect: rejects label/TEST SQL, row-index drift, malformed values, aliases, and writable output.", "tests": [sources_test], "evidence": [evidence_antigravity(sources_test, "class SourcesAntigravityTest")]},
        {"id": "qualification_regression", "path": qualification_test, "role": "Caller: Task 3 verification. Effect: freezes plus-one math, Wilson 95%, same-model folds, complete OOF reference, and every Gate 1 criterion.", "tests": [qualification_test], "evidence": [evidence_antigravity(qualification_test, "class QualificationMathAntigravityTest")]},
        {"id": "router", "path": router, "role": "Caller: Task 5 qualification flow, evaluation, and semantic replay. Effect: recomputes exact Gate 1 plus added>removed routing and binds every decision to its authoritative qualification and feature evidence.", "tests": [router_test, artifacts_test, replay_test], "evidence": [evidence_antigravity(router, "_evidence_for_route_decision_antigravity")]},
        {"id": "evaluation", "path": evaluation, "role": "Caller: publication, replay, dashboard, and CLI. Effect: verifies router provenance, reuses Exp456 oracle/F1 math, recomputes the frozen comparisons and exposes one shared semantic result validator.", "tests": [evaluation_test, artifacts_test, replay_test, dashboard_export_test, cli_test], "evidence": [evidence_antigravity(evaluation, "validate_evaluation_result_semantics_antigravity")]},
        {"id": "artifacts", "path": artifacts, "role": "Caller: Task 5 evaluate command. Effect: validates shared result provenance against publication evidence and creates the canonical nine-leaf inventory.", "tests": [artifacts_test, replay_test, cli_test], "evidence": [evidence_antigravity(artifacts, "validate_evaluation_result_semantics_antigravity")]},
        {"id": "replay", "path": replay, "role": "Caller: Task 5 inspect command. Effect: verifies inventory and hashes, recomputes Gate 1 numerics, reroutes exact evidence, and recomputes comparisons, dashboard, and decision.", "tests": [replay_test, cli_test], "evidence": [evidence_antigravity(replay, "route_ecg_antigravity")]},
        {"id": "dashboard_export", "path": dashboard_export, "role": "Caller: artifact publication and future result display. Effect: invokes the shared semantic result validator before exposing only the ordered aggregate and TRAIN-gate allowlist.", "tests": [dashboard_export_test, artifacts_test, replay_test], "evidence": [evidence_antigravity(dashboard_export, "validate_evaluation_result_semantics_antigravity")]},
        {"id": "task4_regressions", "path": router_test, "role": "Caller: Task 4 verification suite. Effect: covers exact Gate 1 recomputation, route/evaluation provenance, frozen comparison rejection, nine-leaf replay, and dashboard allowlist.", "tests": [contracts_test, router_test, evaluation_test, artifacts_test, replay_test, dashboard_export_test], "evidence": [evidence_antigravity(evaluation_test, "test_rejects_self_consistent_caller_flip_of_normal_two_train_gate_and_route_antigravity")]},
        {"id": "cli", "path": cli, "role": "Caller: Task 5/6 operator. Effect: exposes only reviewed evaluate and bare inspect, lazily runs Gate 1→route→aggregate→publication, and emits the aggregate dashboard allowlist.", "tests": [cli_test], "evidence": [evidence_antigravity(cli, "_evaluate_with_dependencies_antigravity")]},
        {"id": "cli_launcher", "path": cli_launcher, "role": "Caller: repository-external working directories. Effect: resolves the repository before importing only the package CLI.", "tests": [cli_test], "evidence": [evidence_antigravity(cli_launcher, "Path(__file__).resolve().parents[1]")]},
        {"id": "cli_regression", "path": cli_test, "role": "Caller: Task 5 verification. Effect: freezes the exact command tokens, lazy phase loading, result-affecting candidate order, label-query firewall, temporary publication, and outside-CWD launch.", "tests": [cli_test], "evidence": [evidence_antigravity(cli_test, "class CliAntigravityTest")]},
    ]
    edges = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": evidence_antigravity("docs/EXP457_PREREGISTRATION_antigravity_KO.html", "고정 게이트")},
        {"from": "approved_design", "to": "frozen_config", "type": "governs", "evidence": evidence_antigravity(design, "Hydra의 두 고정 seed")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(plan, "Task 5: 닫힌 CLI와 공식 후보 검증")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp457_scientific_contract_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_dashboard_plan_and_map_expose_task5_candidate_without_official_results_antigravity")},
        {"from": "config_regression", "to": "frozen_config", "type": "tests", "evidence": evidence_antigravity(config_test, "test_frozen_config_preserves_the_approved_scientific_contract_antigravity")},
        {"from": "contracts_regression", "to": "result_contracts", "type": "tests", "evidence": evidence_antigravity(contracts_test, "test_seed_evidence_rejects_wrong_dataset_seed_and_integer_contracts_antigravity")},
        {"from": "perturbations_regression", "to": "primary_perturbations", "type": "tests", "evidence": evidence_antigravity(perturbations_test, "test_length_100_fixture_matches_exp417_math_and_exact_order_antigravity")},
        {"from": "source_regression", "to": "train_source", "type": "tests", "evidence": evidence_antigravity(sources_test, "test_sqlite_loader_reads_only_ordered_train_values_without_labels_antigravity")},
        {"from": "qualification_regression", "to": "train_qualification", "type": "tests", "evidence": evidence_antigravity(qualification_test, "test_plus_one_upper_tail_pvalues_and_exact_alpha_are_hand_derived_antigravity")},
        {"from": "frozen_config", "to": "result_contracts", "type": "governs", "evidence": evidence_antigravity(contracts, "FROZEN_CONFIG_ANTIGRAVITY")},
        {"from": "frozen_config", "to": "primary_perturbations", "type": "governs", "evidence": evidence_antigravity(perturbations, "FROZEN_CONFIG_ANTIGRAVITY.primary_types")},
        {"from": "frozen_config", "to": "train_source", "type": "governs", "evidence": evidence_antigravity(sources, "FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets")},
        {"from": "frozen_config", "to": "train_qualification", "type": "governs", "evidence": evidence_antigravity(qualification, "FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds")},
        {"from": "result_contracts", "to": "train_qualification", "type": "governs", "evidence": evidence_antigravity(qualification, "TrainSeedEvidenceAntigravity")},
        {"from": "primary_perturbations", "to": "train_qualification", "type": "governs", "evidence": evidence_antigravity(qualification, "build_primary_perturbations_antigravity")},
        {"from": "train_source", "to": "train_qualification", "type": "governs", "evidence": evidence_antigravity(qualification, "load_ecg_train_antigravity")},
        {"from": "result_contracts", "to": "router", "type": "governs", "evidence": evidence_antigravity(router, "TrainQualificationAntigravity")},
        {"from": "train_qualification", "to": "router", "type": "governs", "evidence": evidence_antigravity(router, "route_ecg_antigravity")},
        {"from": "router", "to": "evaluation", "type": "governs", "evidence": evidence_antigravity(evaluation, "RouteDecisionAntigravity")},
        {"from": "evaluation", "to": "artifacts", "type": "governs", "evidence": evidence_antigravity(artifacts, "evaluate_routes_antigravity")},
        {"from": "evaluation", "to": "replay", "type": "governs", "evidence": evidence_antigravity(replay, "evaluate_routes_antigravity")},
        {"from": "evaluation", "to": "dashboard_export", "type": "governs", "evidence": evidence_antigravity(dashboard_export, "EvaluationResultAntigravity")},
        {"from": "dashboard_export", "to": "artifacts", "type": "governs", "evidence": evidence_antigravity(artifacts, "export_dashboard_antigravity")},
        {"from": "artifacts", "to": "replay", "type": "governs", "evidence": evidence_antigravity(replay, "_payloads_antigravity")},
        {"from": "task4_regressions", "to": "router", "type": "tests", "evidence": evidence_antigravity(router_test, "test_candidate_requires_both_recomputed_gates_antigravity")},
        {"from": "task4_regressions", "to": "evaluation", "type": "tests", "evidence": evidence_antigravity(evaluation_test, "test_reproduces_exp456_comparisons_before_hand_derived_exp457_result_antigravity")},
        {"from": "task4_regressions", "to": "artifacts", "type": "tests", "evidence": evidence_antigravity(artifacts_test, "test_publishes_exact_ordered_canonical_nine_leaf_inventory_create_only_antigravity")},
        {"from": "task4_regressions", "to": "replay", "type": "tests", "evidence": evidence_antigravity(replay_test, "test_rejects_every_rehashed_semantic_leaf_tamper_antigravity")},
        {"from": "task4_regressions", "to": "dashboard_export", "type": "tests", "evidence": evidence_antigravity(dashboard_export_test, "test_exact_aggregate_and_train_gate_allowlist_antigravity")},
        {"from": "train_qualification", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli, "qualify_all_ecg_antigravity")},
        {"from": "router", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli, "route_ecg_antigravity")},
        {"from": "evaluation", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli, "evaluate_routes_antigravity")},
        {"from": "artifacts", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli, "publish_result_antigravity")},
        {"from": "replay", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli, "inspect_result_at_antigravity")},
        {"from": "cli_launcher", "to": "cli", "type": "governs", "evidence": evidence_antigravity(cli_launcher, "main_antigravity")},
        {"from": "cli_regression", "to": "cli", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_candidate_flow_qualifies_routes_evaluates_and_publishes_once_antigravity")},
        {"from": "cli_regression", "to": "cli_launcher", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_launcher_imports_package_from_outside_repository_cwd_antigravity")},
    ]
    unknowns: list[dict[str, object]] = []
    graph: dict[str, object] = {"schema": "exp457_task5_codemap_v1", "generated_at": generated_at, "source_parent_commit": source_commit, "nodes": nodes, "edges": edges, "unknown_relationships": unknowns}
    lock: dict[str, object] = {"schema": "exp457_task5_codemap_lock_v1", "source_parent_commit": source_commit, "working_tree_dirty": False, "represented_paths": sorted(TRIPLET_ANTIGRAVITY), "generated_at": generated_at, "module_fingerprints": module_fingerprints_antigravity(), "tracked_fingerprints": tracked_fingerprints_antigravity()}
    return graph, lock, html_antigravity(graph)


def validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str):
        raise ValueError("invalid evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("evidence not found")


def validate_map_antigravity(graph: object, lock: object, rendered_html: str) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes, edges, unknowns = graph.get("nodes"), graph.get("edges"), graph.get("unknown_relationships")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(unknowns, list) or len(nodes) > 30:
        raise ValueError("invalid bounded topology")
    ids: set[str] = set()
    future_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in ids:
            raise ValueError("invalid node")
        ids.add(node["id"])
        if node.get("status") == "future_unknown":
            future_ids.add(node["id"])
            if node.get("evidence") != [] or (ROOT_ANTIGRAVITY / str(node.get("path"))).exists():
                raise ValueError("future source evidence claimed")
        else:
            if not isinstance(node.get("path"), str) or not (ROOT_ANTIGRAVITY / node["path"]).is_file() or not node.get("evidence"):
                raise ValueError("missing known evidence")
            for item in node["evidence"]:
                validate_evidence_antigravity(item)
    unknown_pairs = {(item.get("from"), item.get("to")) for item in unknowns if isinstance(item, dict) and isinstance(item.get("reason"), str)}
    future_pairs: set[tuple[object, object]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY:
            raise ValueError("invalid edge")
        validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in unknown_pairs:
                raise ValueError("unmarked future relationship")
            future_pairs.add(pair)
        elif edge["type"] == "future_unknown":
            raise ValueError("known edge marked future")
    if unknown_pairs != future_pairs:
        raise ValueError("unknown relationship mismatch")
    source = lock.get("source_parent_commit")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    if source != graph.get("source_parent_commit") or source not in {head, parent}:
        raise ValueError("invalid source parent")
    if lock.get("working_tree_dirty") is not False or lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("invalid lock")
    if lock.get("tracked_fingerprints") != tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != module_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    embedded = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered_html, flags=re.DOTALL)
    if not embedded or canonical_bytes_antigravity(strict_json_antigravity(html_module.unescape(embedded.group(1)))) != canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-18T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(
            strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
            strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
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
