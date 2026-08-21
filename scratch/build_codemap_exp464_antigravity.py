"""Build and validate the bounded Exp464 Task 2 closeout code-map triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp464_antigravity.html",
    "docs/codemap/codemap_exp464_antigravity.json",
    "docs/codemap/codemap_exp464_antigravity.lock",
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp464_task2_foundation_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp464_task2_foundation_codemap_lock_v1"
GENERATED_AT_ANTIGRAVITY = "2026-08-21T00:00:00Z"
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP464_DASHBOARD_antigravity.html",
    "docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP464_PREREGISTRATION_antigravity.html",
    "docs/EXP464_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html",
    "docs/superpowers/specs/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity-design.html",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/sources_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_sources_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp464_antigravity.py",
)


def _canonical_json_antigravity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes_antigravity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_antigravity(path: Path) -> str:
    return _sha256_bytes_antigravity(path.read_bytes())


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_ANTIGRAVITY, text=True).strip()


def _current_head_antigravity() -> str:
    return _git_output_antigravity("rev-parse", "HEAD")


def _allowed_source_commits_antigravity() -> list[str]:
    commits = _git_output_antigravity("rev-list", "--parents", "-n", "1", "HEAD").split()
    return commits


def _node_antigravity(
    node_id: str,
    path: str,
    role: str,
    *,
    symbol: str,
    tests: tuple[str, ...] = (),
    evidence_path: str | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "evidence": [
            {
                "path": path if evidence_path is None else evidence_path,
                "symbol": symbol,
            }
        ],
        "tests": list(tests),
    }


def _build_graph_antigravity(source_parent: str) -> dict[str, Any]:
    governance_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    plan = (
        "docs/superpowers/plans/"
        "2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html"
    )
    nodes = [
        _node_antigravity(
            "approved_design",
            "docs/superpowers/specs/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity-design.html",
            "Caller: Task 2 foundation and later Task 3 scoring. Effect: freezes Exp464 identity, dataset, onset 56, cosine >= 0.95, C0–C8 order, and official-root absence.",
            symbol="Exp464 — 공통 ECG 파형을 제거한 뒤 실제 상승 모양만 확인",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "implementation_plan",
            plan,
            "Caller: Task 2 closeout and later Task 3 planning. Effect: records task_2_complete_pending_task_3 while preserving exactly fourteen checked boxes total for Task 1 and Task 2 only.",
            symbol="task_2_complete_pending_task_3",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP464_DASHBOARD_antigravity.html",
            "Caller: status review. Effect: explains Exp463 M0/M1/M2/M4 evidence in easy Korean, states that Task 2 is review-closed, and keeps Task 3+ unstarted.",
            symbol="Exp464 진행 대시보드",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "preregistration",
            "docs/EXP464_PREREGISTRATION_antigravity.html",
            "Caller: governance review. Effect: freezes C0–C8 order, C6 sole primary, and all gates while Task 2 foundation exists and Task 3+ science code remains absent.",
            symbol="Frozen C0–C8 comparison",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "runbook",
            "docs/EXP464_RUNBOOK_antigravity.html",
            "Caller: future operators. Effect: blocks scoring, private run, official commands, TEST labels, routing, and promotion before Task 2 review.",
            symbol="Exp464 Task 2 operator boundary",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "implementation_report",
            "docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html",
            "Caller: Task 2 closeout review. Effect: records frozen Exp463 authority capture, TRAIN source boundary, zero official/private/label activity, and Task 3 next.",
            symbol="Exp464 Task 2 구현 보고서",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: active research ledger. Effect: stores the task_2_complete_pending_task_3 Exp464 row with official/private/label counts at zero and null result hashes.",
            symbol="\"exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment catalog. Effect: registers the task_2_complete_pending_task_3 Exp464 record without any scientific result or promotion.",
            symbol="\"id\": \"Exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion history. Effect: appends the task_2_complete_pending_task_3 Exp464 row while keeping champion, pending candidate, routing, and promotion unchanged.",
            symbol="\"experiment\": \"Exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "governance_test",
            governance_test,
            "Caller: Task 2 closeout verification. Effect: rejects missing foundation modules, ledger drift, any checkbox count other than Task 1 plus Task 2 only, missing map triplet, root creation, or nonzero official/private/label counts.",
            symbol="test_exp464_is_planned_without_results_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "map_builder",
            "scratch/build_codemap_exp464_antigravity.py",
            "Caller: Task 2 closeout validation. Effect: writes and validates the Exp464 Task 2 bounded graph, exact embedded JSON, lock fingerprints, and source-parent binding.",
            symbol="build_triplet_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task2_config",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/config_antigravity.py",
            "Caller: Task 2 tests, predecessor validator, and future Task 3 scoring. Effect: freezes Exp464 identity, interpreter, TRAIN shape, C0–C8 order, gates, and exact C0/C1/C2/C7 Exp463 summaries.",
            symbol="FROZEN_CONFIG_ANTIGRAVITY",
            tests=(
                governance_test,
                "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_config_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task2_contracts",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/contracts_antigravity.py",
            "Caller: Task 2 predecessor capture and future Task 3 scoring. Effect: freezes exact-type Exp463 summaries, copies/finalizes control mappings, and rejects forged authority fields.",
            symbol="Exp463AuthorityAntigravity",
            tests=(
                governance_test,
                "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_contracts_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task2_predecessor",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/predecessor_antigravity.py",
            "Caller: Task 2 tests and future Task 3 scoring. Effect: captures frozen Exp463 terminal authority from the tracked Task 5 report, implementation report, and three ledgers only.",
            symbol="capture_exp463_authority_antigravity",
            tests=(
                governance_test,
                "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_predecessor_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task2_sources",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/sources_antigravity.py",
            "Caller: Task 2 tests and future Task 3 scoring. Effect: replays the exact TRAIN-only float32 1413x140 matrix as an owned read-only copy without label access.",
            symbol="load_normal2_train_antigravity",
            tests=(
                governance_test,
                "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_sources_antigravity.py",
            ),
        ),
        _node_antigravity(
            "future_task3_scoring",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/scoring_antigravity.py",
            "Caller: future Task 3 only. Effect: next planned consumer of the Task 2 foundation boundary.",
            symbol="Task 3 — centered residual ramp and C0–C8 scores",
            tests=(governance_test,),
            evidence_path=plan,
        ),
    ]
    edges = [
        {"from": "approved_design", "to": "preregistration", "relation": "freezes"},
        {"from": "approved_design", "to": "dashboard", "relation": "freezes"},
        {"from": "implementation_plan", "to": "task2_config", "relation": "bounds_current_work"},
        {"from": "implementation_plan", "to": "future_task3_scoring", "relation": "bounds_future_work"},
        {"from": "research_state", "to": "dashboard", "relation": "reports_status"},
        {"from": "experiment_catalog", "to": "dashboard", "relation": "reports_identity"},
        {"from": "champion_ledger", "to": "dashboard", "relation": "preserves_champion_state"},
        {"from": "governance_test", "to": "dashboard", "relation": "covers"},
        {"from": "governance_test", "to": "preregistration", "relation": "covers"},
        {"from": "governance_test", "to": "runbook", "relation": "covers"},
        {"from": "governance_test", "to": "implementation_report", "relation": "covers"},
        {"from": "governance_test", "to": "research_state", "relation": "covers"},
        {"from": "governance_test", "to": "experiment_catalog", "relation": "covers"},
        {"from": "governance_test", "to": "champion_ledger", "relation": "covers"},
        {"from": "governance_test", "to": "map_builder", "relation": "covers"},
        {"from": "governance_test", "to": "task2_config", "relation": "covers"},
        {"from": "governance_test", "to": "task2_contracts", "relation": "covers"},
        {"from": "governance_test", "to": "task2_predecessor", "relation": "covers"},
        {"from": "governance_test", "to": "task2_sources", "relation": "covers"},
        {"from": "task2_config", "to": "task2_contracts", "relation": "freezes_inputs_for"},
        {"from": "task2_config", "to": "task2_predecessor", "relation": "freezes_inputs_for"},
        {"from": "task2_config", "to": "task2_sources", "relation": "freezes_inputs_for"},
        {"from": "task2_predecessor", "to": "future_task3_scoring", "relation": "feeds"},
        {"from": "task2_sources", "to": "future_task3_scoring", "relation": "feeds"},
        {"from": "map_builder", "to": "future_task3_scoring", "relation": "documents_boundary"},
    ]
    flows = [
        "governance_test -> docs/ledgers/map",
        "approved_design -> Task 2 foundation docs and ledgers",
        "Task 2 foundation -> future Task 3 scorer",
    ]
    return {
        "graph_schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent": source_parent,
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "flow_count": len(flows),
        "nodes": nodes,
        "edges": edges,
        "flows": flows,
    }


def _verify_evidence_antigravity(graph: dict[str, Any]) -> None:
    for node in graph["nodes"]:
        for evidence in node["evidence"]:
            evidence_path = ROOT_ANTIGRAVITY / evidence["path"]
            if not evidence_path.is_file():
                raise FileNotFoundError(f"missing evidence path: {evidence_path}")
            text = evidence_path.read_text(encoding="utf-8")
            if evidence["symbol"] not in text:
                raise ValueError(
                    f"missing evidence symbol {evidence['symbol']!r} in {evidence_path}"
                )


def _fingerprints_antigravity() -> list[dict[str, str]]:
    fingerprints: list[dict[str, str]] = []
    for relative in TRACKED_PATHS_ANTIGRAVITY:
        path = ROOT_ANTIGRAVITY / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing tracked path: {path}")
        fingerprints.append({"path": relative, "sha256": _sha256_file_antigravity(path)})
    return fingerprints


def _html_bytes_antigravity(graph_json: str) -> bytes:
    html_text = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp464 Task 2 Foundation Code Map</title>
<style>body{{max-width:1160px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}}code,pre{{background:#eef2f7;padding:.1rem .3rem;overflow-wrap:anywhere}}table{{width:100%;border-collapse:collapse;margin:1rem 0}}th,td{{border:1px solid #aab3bf;padding:.55rem;text-align:left;vertical-align:top}}</style></head>
<body><main>
<h1>Exp464 Task 2 closeout code map</h1>
<p>이 map은 Task 2 closeout 경계를 다루며, 다음 소비자가 Task 3 scorer인 상태를 설명한다.</p>
<script id="graph-data" type="application/json">{graph_json}</script>
</main></body></html>
"""
    return html_text.encode("utf-8")


def build_triplet_antigravity(validate_only: bool = False) -> dict[str, Any]:
    triplet_paths = [ROOT_ANTIGRAVITY / relative for relative in TRIPLET_ANTIGRAVITY]
    existing_lock_path = triplet_paths[2]
    current_head = _current_head_antigravity()
    if validate_only and existing_lock_path.is_file():
        lock_data = json.loads(existing_lock_path.read_text(encoding="utf-8"))
        source_parent = lock_data["source_parent"]
        allowed = _allowed_source_commits_antigravity()
        if source_parent not in allowed:
            raise ValueError(
                f"lock source_parent {source_parent} is not reachable from current HEAD {current_head}"
            )
        allowed_source_commits = [source_parent] + [commit for commit in allowed if commit != source_parent]
    else:
        source_parent = current_head
        allowed_source_commits = [source_parent]

    graph = _build_graph_antigravity(source_parent)
    _verify_evidence_antigravity(graph)
    fingerprints = _fingerprints_antigravity()
    graph_json = _canonical_json_antigravity(graph)
    html_bytes = _html_bytes_antigravity(graph_json)
    json_bytes = graph_json.encode("utf-8")
    lock = {
        "lock_schema": LOCK_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent": source_parent,
        "generated_files": list(TRIPLET_ANTIGRAVITY),
        "tracked_fingerprints": fingerprints,
        "fingerprint_count": len(fingerprints),
        "graph_sha256": _sha256_bytes_antigravity(json_bytes),
        "html_sha256": _sha256_bytes_antigravity(html_bytes),
    }
    lock_bytes = (_canonical_json_antigravity(lock) + "\n").encode("utf-8")
    expected_bytes = (html_bytes, json_bytes, lock_bytes)

    if validate_only:
        for path, expected in zip(triplet_paths, expected_bytes):
            if not path.is_file():
                raise FileNotFoundError(f"missing generated file: {path}")
            actual = path.read_bytes()
            if actual != expected:
                raise ValueError(f"generated bytes differ for {path}")
    else:
        triplet_paths[0].write_bytes(html_bytes)
        triplet_paths[1].write_bytes(json_bytes)
        triplet_paths[2].write_bytes(lock_bytes)

    return {
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "source_parent": source_parent,
        "allowed_source_commits": allowed_source_commits,
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "flow_count": graph["flow_count"],
        "fingerprint_count": lock["fingerprint_count"],
    }


def main_antigravity(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the existing triplet without writing files",
    )
    args = parser.parse_args(argv)
    build_triplet_antigravity(validate_only=args.validate_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
