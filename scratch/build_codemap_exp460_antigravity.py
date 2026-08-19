#!/usr/bin/env python3
"""Build and validate the bounded Exp460 Task 2 foundations code map."""

from __future__ import annotations

import argparse
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
    "experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_config_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
    "scratch/build_codemap_exp460_antigravity.py",
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
        '<title>Exp460 Task 2 foundations 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp460 Task 2 foundations 코드맵</h1><p>Task 2 keeps the governance surface intact while adding the exact predecessor input freeze: one config module, one contract validator, one predecessor capture module, and three current Task 2 tests. Future rule evaluation, artifact publication, and CLI work remain planned only.</p>'
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
            "role": "Caller: Task 1-6 execution. Effect: records Task 1 and Task 2 completion while keeping Tasks 3-6 unchecked.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 2 — 고정 입력과 자료형")],
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
            "role": "Caller: future operators. Effect: keeps the Task 1 official-execution firewall while Task 2 only freezes scientific inputs.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_RUNBOOK_antigravity.html", "Task 1 operating boundary")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports Task 2 foundations, predecessor hashes, zero official actions, and unchanged label-access state.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html", "Task 2 lifecycle")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP460_DASHBOARD_antigravity.html",
            "role": "Caller: result-facing implementation review. Effect: shows Task 2/6 progress, frozen predecessor digests, unchanged official-zero state, and the design-only rule comparison frame.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("docs/EXP460_DASHBOARD_antigravity.html", "Task 2/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: keeps Exp460 in planned-only ledger state with zero official actions and unchanged pending candidate while Task 2 works only in local files.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"task_1_governance_complete\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: still registers Exp460 as planned-only lifecycle despite Task 2 local implementation progress.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp460\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: keeps champion and pending candidate unchanged while Task 2 adds only local scientific input code.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"task_1_governance_complete\"")],
        },
        {
            "id": "governance_regression",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py",
            "role": "Caller: Task 1-2 verification. Effect: rejects stale docs, wrong progress/checkbox state, changed planned-only ledgers, or invalid stored codemap bytes.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py"],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task2_plan_dashboard_and_map_antigravity")],
        },
        {
            "id": "map_builder",
            "path": "scratch/build_codemap_exp460_antigravity.py",
            "role": "Caller: Task 2 validation. Effect: rebuilds and validates the bounded Exp460 Task 2 graph plus the stored HTML/JSON/lock triplet bytes.",
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
            "role": "Caller: predecessor capture and future Tasks 3-6. Effect: validates the exact ordered Exp459 snapshot, rejects bool-as-int/container drift, and localizes the frozen predecessor truth into Exp460-owned dataclasses.",
            "tests": [
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_contracts_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_predecessor_antigravity.py",
            ],
            "evidence": [evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/contracts_antigravity.py", "validate_exp459_fallback_authority_antigravity")],
        },
        {
            "id": "task2_predecessor_capture",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/predecessor_antigravity.py",
            "role": "Caller: future Tasks 3-6. Effect: returns the exact frozen Exp459 fallback authority by reusing the version-controlled Exp459 frozen authority and predecessor confusion context without touching TRAIN rows or TEST labels.",
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
            "id": "future_rule_evaluation",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/evaluation_antigravity.py",
            "role": "Caller: future Tasks 3-4. Effect: will compare the four rules and derive the strict fallback decision from the frozen Task 2 authority.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "compare_fallback_rules_antigravity")],
            "status": "future_planned",
        },
        {
            "id": "future_rule_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_evaluation_antigravity.py",
            "role": "Caller: future Task 3 verification. Effect: will prove 10/10, 9/10, 8/10, and 0/10 route truth plus confusion-matrix F1 recomputation.",
            "tests": [],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "10/10, 9/10, 8/10, 0/10 route RED와 성능 재계산 RED를 작성한다.")],
            "status": "future_planned",
        },
        {
            "id": "future_artifacts_replay_dashboard",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/artifacts_antigravity.py",
            "role": "Caller: future Tasks 4-5. Effect: will publish the six result leaves, replay them from frozen authority, and export the bounded dashboard payload.",
            "tests": [
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_artifacts_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_replay_antigravity.py",
                "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_dashboard_export_antigravity.py",
            ],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "정확한 6개 결과 파일")],
            "status": "future_planned",
        },
        {
            "id": "future_artifacts_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_artifacts_antigravity.py",
            "role": "Caller: future Task 4 verification. Effect: will reject stored-result, replay, route, AUC, confusion, and dashboard tamper across the six-leaf publication boundary.",
            "tests": [],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "6개 파일 저장과 AUC·route·혼동행렬·결정 변조 replay RED를 작성한다.")],
            "status": "future_planned",
        },
        {
            "id": "future_cli_launcher",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/cli_antigravity.py",
            "role": "Caller: future Tasks 5-6. Effect: will expose only evaluate --reviewed-commit LOWER40HEX and inspect after implementation and approval boundaries are met.",
            "tests": ["experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py"],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "evaluate --reviewed-commit LOWER40HEX")],
            "status": "future_planned",
        },
        {
            "id": "future_cli_tests",
            "path": "experiments/exp460_ecg_strict_fallback_antigravity/tests/test_cli_antigravity.py",
            "role": "Caller: future Task 5 verification. Effect: will prove exact CLI surface, outside-CWD launcher behavior, and repeated unofficial evaluate→inspect equality.",
            "tests": [],
            "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "CLI·outside-CWD launcher 부재 RED를 작성한다.")],
            "status": "future_planned",
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
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "Task 2 — 고정 입력과 자료형"),
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
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task2_plan_dashboard_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "preregistration",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "research_state",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "champion_ledger",
            "type": "tests",
            "evidence": evidence_antigravity("experiments/exp460_ecg_strict_fallback_antigravity/tests/test_governance_antigravity.py", "test_task1_governance_surface_antigravity"),
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
            "evidence": evidence_antigravity("docs/EXP460_DASHBOARD_antigravity.html", "Task 2 foundations"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "future_rule_evaluation",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "compare_fallback_rules_antigravity"),
        },
        {
            "from": "future_rule_tests",
            "to": "future_rule_evaluation",
            "type": "tests",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "10/10, 9/10, 8/10, 0/10 route RED와 성능 재계산 RED를 작성한다."),
        },
        {
            "from": "future_rule_evaluation",
            "to": "future_artifacts_replay_dashboard",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "정확한 6개 결과 파일"),
        },
        {
            "from": "future_artifacts_tests",
            "to": "future_artifacts_replay_dashboard",
            "type": "tests",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "6개 파일 저장과 AUC·route·혼동행렬·결정 변조 replay RED를 작성한다."),
        },
        {
            "from": "future_artifacts_replay_dashboard",
            "to": "future_cli_launcher",
            "type": "governs",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "dashboard payload를 진행 상태, 네 비교 행, 두 실패 fold, 최종 route, label_query_count 0으로 제한한다."),
        },
        {
            "from": "future_cli_tests",
            "to": "future_cli_launcher",
            "type": "tests",
            "evidence": evidence_antigravity("docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html", "CLI·outside-CWD launcher 부재 RED를 작성한다."),
        },
    ]
    graph = {
        "schema": "exp460_task2_foundations_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_parent_commit,
        "unknown_relationships": [],
        "nodes": nodes,
        "edges": edges,
    }
    lock = {
        "schema": "exp460_task2_foundations_codemap_lock_v1",
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
    if graph["schema"] != "exp460_task2_foundations_codemap_v1":
        raise ValueError("unexpected graph schema")
    if lock["schema"] != "exp460_task2_foundations_codemap_lock_v1":
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
    if len(graph["nodes"]) > 24:
        raise ValueError("graph exceeds 24 nodes")
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
    generated_at = "2026-08-19T00:00:00Z"
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
