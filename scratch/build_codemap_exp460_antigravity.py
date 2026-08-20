#!/usr/bin/env python3
"""Build and validate the bounded Exp460 Task 5 strict fallback code map."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html as html_module
import html.parser
import json
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp460_antigravity.html"
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP460_DASHBOARD_antigravity.html",
    "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP460_PREREGISTRATION_antigravity.html",
    "docs/EXP460_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html",
    "docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html",
    "experiments/exp460_ecg_strict_fallback_antigravity/config_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/fallback_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/artifacts_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/replay_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/dashboard_export_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_fallback_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py",
    "scratch/build_codemap_exp460_antigravity.py",
    "scratch/run_exp460_ecg_strict_fallback_antigravity.py",
)


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates_antigravity,
    )


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        return False
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".html":
        parser = HtmlTextExtractorAntigravity()
        parser.feed(text)
        haystack = " ".join(parser.parts)
    else:
        haystack = text
    return symbol in haystack


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
        '<title>Exp460 Task 5 strict fallback 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp460 Task 5 strict fallback 코드맵</h1><p>Task 5 preserves the frozen Task 2-4 science while adding the exact lazy CLI, fixed-runtime launcher, private two-run reproduction, and implementation-verified ledgers. Official execution remains separately approved Task 6 work.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2><pre id="codemap-json-antigravity" type="application/json">{payload}</pre>'
        '</main></body></html>'
    )


def build_map_antigravity(
    source_parent_commit: str,
    generated_at: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    nodes = [
        {
            "id": "approved_design",
            "path": "docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html",
            "role": "Caller: Task 1-6 governance and implementation. Effect: freezes the target dataset, exact rule order, strict-only official decision policy, and the two failed Exp459 folds that Task 2 must preserve.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "10/10 엄격 복귀")],
        },
        {
            "id": "implementation_plan",
            "path": "docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html",
            "role": "Caller: Task 1-6 execution. Effect: records Task 5 Steps 1-5 complete while keeping its parent-review Step 6 and all of Task 6 unchecked.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 5 — CLI, 비공식 재현, 최종 후보")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP460_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: still states planned-only Exp460 official execution with zero official counts and null Exp460 result hashes.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_PREREGISTRATION_antigravity.html", "governance_complete_implementation_not_started")],
        },
        {
            "id": "runbook",
            "path": "docs/EXP460_RUNBOOK_antigravity.html",
            "role": "Caller: future operators. Effect: documents the exact Task 5 CLI/runtime boundary and prohibits official execution until separately approved Task 6.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_RUNBOOK_antigravity.html", "Task 5 operating boundary")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports the Task 5 CLI, two private exact-equal roots, six private leaf hashes, verification counts, and zero official actions.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html", "Task 5 lifecycle")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP460_DASHBOARD_antigravity.html",
            "role": "Caller: result-facing implementation review. Effect: shows Task 5/6 progress, private exact equality, the four-rule table, two failed folds, strict fallback status, and official-zero state.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_DASHBOARD_antigravity.html", "Task 5/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Task 5 implementation verification and two private exact-equal runs while preserving zero official actions and unchanged pending candidate.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"task_5_cli_private_reproduction_verified\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers the implementation-verified Task 5 lifecycle without claiming official evaluation.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp460\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: keeps champion and pending candidate unchanged while recording Task 5 private reproduction only.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"task_5_cli_private_reproduction_verified\"")],
        },
        {
            "id": "governance_regression",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
            "role": "Caller: Task 1-5 verification. Effect: rejects stale docs, wrong Task 5 progress, false official state, changed champion/pending state, or invalid stored codemap bytes.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task5_plan_dashboard_report_runbook_and_map_antigravity")],
        },
        {
            "id": "map_builder",
            "path": "scratch/build_codemap_exp460_antigravity.py",
            "role": "Caller: Task 5 validation. Effect: rebuilds and validates the bounded Exp460 Task 5 graph plus the stored HTML/JSON/lock triplet bytes.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("scratch/build_codemap_exp460_antigravity.py", "validate_map_antigravity")],
        },
        {
            "id": "task2_config",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/config_antigravity.py",
            "role": "Caller: Task 2 contracts and predecessor capture. Effect: freezes Exp460 rule order, exact predecessor hashes, baseline/candidate confusions, and the fixed interpreter boundary.",
            "tests": [
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
            ],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/config_antigravity.py", "exp459_official_artifact_map_sha256")],
        },
        {
            "id": "task2_contracts",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py",
            "role": "Caller: predecessor capture and Task 3 evaluation. Effect: validates the exact ordered Exp459 snapshot, rejects bool-as-int/container drift, and defines immutable Task 3 result contracts.",
            "tests": [
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
            ],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py", "validate_fallback_evaluation_antigravity")],
        },
        {
            "id": "task2_predecessor_capture",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py",
            "role": "Caller: Task 3 evaluation and future Tasks 4-6. Effect: returns the exact frozen Exp459 fallback authority by reusing the version-controlled Exp459 frozen authority and predecessor confusion context without touching TRAIN rows or TEST labels.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py", "capture_exp459_fallback_authority_antigravity")],
        },
        {
            "id": "task2_config_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py",
            "role": "Caller: Task 2 verification. Effect: rejects missing config import surface, mutated rule order/confusions/hashes, and bool-as-int drift in the frozen Exp460 configuration.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py", "test_import_surface_and_frozen_values_antigravity")],
        },
        {
            "id": "task2_contracts_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py",
            "role": "Caller: Task 2 verification. Effect: rejects post-construction mutation, reordered/missing/duplicated records, and changed decision/confusion/hash semantics at the Exp460 authority boundary.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py", "test_validator_rejects_bool_as_int_container_and_semantic_drift_antigravity")],
        },
        {
            "id": "task2_predecessor_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
            "role": "Caller: Task 2 verification. Effect: proves the captured authority equals the exact ordered Exp459 snapshot, includes the two failed rows, reuses frozen predecessor science, and remains source-free.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py", "test_capture_returns_exact_frozen_exp459_authority_antigravity")],
        },
        {
            "id": "task3_strict_route",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/fallback_antigravity.py",
            "role": "Caller: Task 3 rule evaluation and future Tasks 4-5. Effect: permits candidate only when all ten records pass and rejects malformed authority drift.",
            "tests": [
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_fallback_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py",
            ],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/fallback_antigravity.py", "strict_route_antigravity")],
        },
        {
            "id": "task3_rule_evaluation",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py",
            "role": "Caller: Task 3 implementation, Task 4 artifacts, and Task 5 evaluate. Effect: recomputes the four ordered rules, strict fallback status, and joint-success gate from the frozen Task 2 authority.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py", "compare_fallback_rules_antigravity")],
        },
        {
            "id": "task4_artifacts_publication",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/artifacts_antigravity.py",
            "role": "Caller: Task 4 implementation and Task 5 evaluate. Effect: publishes the exact six leaves with canonical JSON/JSONL bytes and an artifact hash map that excludes itself.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_artifacts_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/artifacts_antigravity.py", "publish_result_antigravity")],
        },
        {
            "id": "task4_replay",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/replay_antigravity.py",
            "role": "Caller: Task 4 verification and Task 5 evaluate/inspect. Effect: parses the stored six-leaf inventory, replays semantics from the frozen authority, and rejects rehashed tamper beyond hash-map equality.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_replay_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/replay_antigravity.py", "inspect_replay_at_antigravity")],
        },
        {
            "id": "task4_dashboard_export",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/dashboard_export_antigravity.py",
            "role": "Caller: Task 4 publication and Task 5 CLI output. Effect: exports the bounded scientific-only payload with dataset/predecessor identifiers, four comparison rows, two failed folds, strict status, and official-zero counters.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_dashboard_export_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/dashboard_export_antigravity.py", "export_dashboard_antigravity")],
        },
        {
            "id": "task5_cli",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py",
            "role": "Caller: fixed launcher and private dependency interfaces. Effect: exposes only evaluate --reviewed-commit LOWER40HEX and bare inspect, chains existing Task 2-4 functions, rejects resolved official-root aliases and overwrites, and returns the bounded payload.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "_evaluate_with_dependencies_antigravity")],
        },
        {
            "id": "task5_launcher",
            "path": "scratch/run_exp460_ecg_strict_fallback_antigravity.py",
            "role": "Caller: future separately approved Task 6 command. Effect: re-execs the exact frozen Python and makes repository imports independent of caller CWD.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py"],
            "evidence": [evidence_antigravity("scratch/run_exp460_ecg_strict_fallback_antigravity.py", "_ensure_fixed_runtime_antigravity")],
        },
        {
            "id": "task5_cli_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py",
            "role": "Caller: Task 5 verification. Effect: proves exact CLI surface, lazy dispatch, resolved official-root alias rejection in evaluate/inspect seams, private equality, create-only roots, exact fixed-interpreter re-exec, and outside-CWD launcher success.",
            "tests": [],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py", "test_launcher_private_evaluate_then_inspect_from_outside_cwd_antigravity")],
        },
    ]
    edges = [
        {
            "from": "approved_design",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "현재 Exp459 자료에서는 1번과 2번을 만족하지 못하므로 엄격 규칙은 baseline을 선택한다"),
        },
        {
            "from": "approved_design",
            "to": "task2_config",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/specs/2026-08-20-exp460-ecg-strict-fallback-antigravity-design.html", "Exp459의 다섯 연속 fold, 두 seed, 네 합성 이상 종류와 AUC 값"),
        },
        {
            "from": "implementation_plan",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 3 — 엄격 복귀 규칙과 성능 계산"),
        },
        {
            "from": "implementation_plan",
            "to": "task2_predecessor_capture",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Exp459 공식 10개 기록, 두 실패 AUC, 혼동행렬, hash를 읽어 authority로 만드는 함수를 구현한다."),
        },
        {
            "from": "governance_regression",
            "to": "dashboard",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task5_plan_dashboard_report_runbook_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "preregistration",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task5_governance_surface_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "research_state",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task5_governance_surface_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "champion_ledger",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task5_governance_surface_antigravity"),
        },
        {
            "from": "map_builder",
            "to": "dashboard",
            "type": "tests",
            "evidence": evidence_antigravity("scratch/build_codemap_exp460_antigravity.py", "validate_map_antigravity"),
        },
        {
            "from": "task2_config",
            "to": "task2_contracts",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py", "from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY"),
        },
        {
            "from": "task2_contracts",
            "to": "task2_predecessor_capture",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py", "from .contracts_antigravity import ("),
        },
        {
            "from": "task2_config_tests",
            "to": "task2_config",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py", "test_import_surface_and_frozen_values_antigravity"),
        },
        {
            "from": "task2_contracts_tests",
            "to": "task2_contracts",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py", "test_validator_rejects_bool_as_int_container_and_semantic_drift_antigravity"),
        },
        {
            "from": "task2_predecessor_tests",
            "to": "task2_predecessor_capture",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py", "test_capture_returns_exact_frozen_exp459_authority_antigravity"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity("docs/EXP460_DASHBOARD_antigravity.html", "Task 5 CLI and private reproduction"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "task3_strict_route",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "strict_route_antigravity"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "task3_rule_evaluation",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "compare_fallback_rules_antigravity"),
        },
        {
            "from": "task3_strict_route",
            "to": "task3_rule_evaluation",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py", "from .fallback_antigravity import strict_route_antigravity"),
        },
        {
            "from": "task3_rule_evaluation",
            "to": "task4_artifacts_publication",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "정확한 6개 결과 파일"),
        },
        {
            "from": "task4_artifacts_publication",
            "to": "task4_replay",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/replay_antigravity.py", "from .artifacts_antigravity import ("),
        },
        {
            "from": "task4_artifacts_publication",
            "to": "task4_dashboard_export",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/artifacts_antigravity.py", "from .dashboard_export_antigravity import export_dashboard_antigravity"),
        },
        {
            "from": "task4_replay",
            "to": "task5_cli",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "from .replay_antigravity import inspect_replay_at_antigravity"),
        },
        {
            "from": "task4_dashboard_export",
            "to": "task5_cli",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "from .dashboard_export_antigravity import export_dashboard_antigravity"),
        },
        {
            "from": "task3_rule_evaluation",
            "to": "task5_cli",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "from .evaluation_antigravity import compare_fallback_rules_antigravity"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "task5_cli",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "capture_exp459_fallback_authority_antigravity"),
        },
        {
            "from": "task4_artifacts_publication",
            "to": "task5_cli",
            "type": "governs",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py", "from .artifacts_antigravity import publish_result_antigravity"),
        },
        {
            "from": "task5_launcher",
            "to": "task5_cli",
            "type": "calls",
            "evidence": evidence_antigravity("scratch/run_exp460_ecg_strict_fallback_antigravity.py", "main_antigravity"),
        },
        {
            "from": "task5_cli_tests",
            "to": "task5_cli",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py", "test_private_evaluate_then_inspect_returns_exact_payload_and_six_leaves_antigravity"),
        },
        {
            "from": "task5_cli_tests",
            "to": "task5_launcher",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py", "test_launcher_reexecs_the_exact_fixed_interpreter_antigravity"),
        },
    ]
    graph = {
        "schema": "exp460_task5_strict_fallback_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_parent_commit,
        "unknown_relationships": [],
        "nodes": nodes,
        "edges": edges,
    }
    lock = {
        "schema": "exp460_task5_strict_fallback_codemap_lock_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_parent_commit,
        "represented_paths": [
            "docs/codemap/codemap_exp460_antigravity.html",
            "docs/codemap/codemap_exp460_antigravity.json",
            "docs/codemap/codemap_exp460_antigravity.lock",
        ],
        "tracked_fingerprints": tracked_fingerprints_antigravity(),
        "module_fingerprints": module_fingerprints_antigravity(),
        "working_tree_dirty": False,
    }
    return graph, lock, html_antigravity(graph)


def load_triplet_antigravity() -> tuple[dict[str, object], dict[str, object], str]:
    graph = strict_json_antigravity(JSON_PATH_ANTIGRAVITY)
    lock = strict_json_antigravity(LOCK_PATH_ANTIGRAVITY)
    rendered = HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
    assert isinstance(graph, dict) and isinstance(lock, dict)
    return graph, lock, rendered


def validate_map_antigravity(graph: dict[str, object], lock: dict[str, object], rendered: str) -> None:
    if graph["schema"] != "exp460_task5_strict_fallback_codemap_v1":
        raise ValueError("unexpected graph schema")
    if lock["schema"] != "exp460_task5_strict_fallback_codemap_lock_v1":
        raise ValueError("unexpected lock schema")
    if graph["source_parent_commit"] != lock["source_parent_commit"]:
        raise ValueError("source_parent_commit mismatch")
    expected_rendered = html_antigravity(graph)
    if rendered.rstrip("\n") != expected_rendered:
        raise ValueError("rendered HTML is not in sync with graph JSON")
    node_ids = set()
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id in node_ids:
            raise ValueError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        for evidence in node.get("evidence", []):
            if not evidence_resolves_antigravity(evidence["path"], evidence["symbol"]):
                raise ValueError(f"unresolved evidence: {evidence}")
        if node.get("status") != "future_planned" and not (ROOT_ANTIGRAVITY / node["path"]).is_file():
            raise ValueError(f"missing current node path: {node['path']}")
    if len(graph["nodes"]) > 25:
        raise ValueError("graph exceeds 25 nodes")
    for edge in graph["edges"]:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            raise ValueError(f"edge references unknown node: {edge}")
        if not evidence_resolves_antigravity(edge["evidence"]["path"], edge["evidence"]["symbol"]):
            raise ValueError(f"unresolved edge evidence: {edge['evidence']}")
    if lock["tracked_fingerprints"] != tracked_fingerprints_antigravity():
        raise ValueError("tracked_fingerprints mismatch")
    if lock["module_fingerprints"] != module_fingerprints_antigravity():
        raise ValueError("module_fingerprints mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        graph, lock, rendered = load_triplet_antigravity()
        validate_map_antigravity(graph, lock, rendered)
        return
    source_parent_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_ANTIGRAVITY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    graph, lock, rendered = build_map_antigravity(source_parent_commit, generated_at)
    JSON_PATH_ANTIGRAVITY.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOCK_PATH_ANTIGRAVITY.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    HTML_PATH_ANTIGRAVITY.write_text(rendered + "\n", encoding="utf-8")
    validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    main()
