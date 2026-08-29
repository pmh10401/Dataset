"""Build and validate the bounded Exp464 Task 5 closed-CLI code-map triplet."""

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
GRAPH_SCHEMA_ANTIGRAVITY = "exp464_task5_closed_cli_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp464_task5_closed_cli_codemap_lock_v1"
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
    ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-4-report-antigravity.md",
    ".superpowers/sdd/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/task-5-report-antigravity.md",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/control_row_evidence_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/exp463_control_row_evidence_antigravity.jsonl",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/raw_evidence_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/sources_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/centered_ramp_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/scoring_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/qualification_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/evaluation_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/artifacts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/replay_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/dashboard_export_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_sources_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_centered_ramp_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_scoring_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_cli_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp464_antigravity.py",
    "scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py",
)


def _canonical_json_antigravity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json_bytes_antigravity(value: Any) -> bytes:
    return (_canonical_json_antigravity(value) + "\n").encode("utf-8")


def _sha256_bytes_antigravity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_antigravity(path: Path) -> str:
    return _sha256_bytes_antigravity(path.read_bytes())


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT_ANTIGRAVITY,
        text=True,
    ).strip()


def _source_parent_antigravity() -> str:
    return _git_output_antigravity(
        "rev-list",
        "-n",
        "1",
        "HEAD",
        "--",
        *TRACKED_PATHS_ANTIGRAVITY,
    )


def _allowed_source_commits_antigravity(source_parent: str) -> list[str]:
    return _git_output_antigravity(
        "rev-list",
        "--parents",
        "-n",
        "1",
        source_parent,
    ).split()


def _node_antigravity(
    node_id: str,
    path: str,
    role: str,
    *,
    symbol: str,
    tests: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "evidence": [{"path": path, "symbol": symbol}],
        "tests": list(tests),
    }


def _build_graph_antigravity(source_parent: str) -> dict[str, Any]:
    governance_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    task3_feature_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_centered_ramp_antigravity.py"
    )
    task3_scoring_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_scoring_antigravity.py"
    )
    task2_predecessor_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_predecessor_antigravity.py"
    )
    task4_qualification_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_qualification_antigravity.py"
    )
    task4_evaluation_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_evaluation_antigravity.py"
    )
    task4_artifacts_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_artifacts_antigravity.py"
    )
    task4_replay_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_replay_antigravity.py"
    )
    task4_dashboard_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_dashboard_export_antigravity.py"
    )
    task5_cli_test = (
        "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/"
        "test_cli_antigravity.py"
    )
    exp463_feature_regression = (
        "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/"
        "test_ramp_features_antigravity.py"
    )
    exp463_scoring_regression = (
        "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/"
        "test_scoring_antigravity.py"
    )
    nodes = [
        _node_antigravity(
            "approved_design",
            "docs/superpowers/specs/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity-design.html",
            "Caller: Task 3 scorer, Task 4 qualification, and Task 5 closed CLI. Effect: freezes Exp464 identity, dataset, onset 56, C0–C8 order, C6-only promotion contract, and public official-root absence.",
            symbol="Exp464 — 공통 ECG 파형 제거 뒤 중심화 ramp 확인",
            tests=(governance_test, task3_scoring_test, task4_qualification_test),
        ),
        _node_antigravity(
            "implementation_plan",
            "docs/superpowers/plans/2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html",
            "Caller: Task 5 reviewed closeout. Effect: records stopped_before_official_evaluation_qualification_failed with exactly thirty-five checked boxes for Tasks 1–5 while Task 6 remains 0/7 unchecked and permanently forbidden.",
            symbol="stopped_before_official_evaluation_qualification_failed",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP464_DASHBOARD_antigravity.html",
            "Caller: status review. Effect: summarizes the two exact-equal private Task 5 reproductions, the frozen C6 qualification failure, the SHIP review, and the removed-after-review recoverable private-root boundary.",
            symbol="Exp464 대시보드 — Task 5 최종 closeout, Task 6 금지",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "preregistration",
            "docs/EXP464_PREREGISTRATION_antigravity.html",
            "Caller: governance review. Effect: freezes the scientific contract, C6 gate thresholds, and the final 35/42 checkbox truth after Task 5 review SHIP closeout.",
            symbol="Frozen qualification gates for C6",
            tests=(governance_test, task4_qualification_test, task4_evaluation_test),
        ),
        _node_antigravity(
            "runbook",
            "docs/EXP464_RUNBOOK_antigravity.html",
            "Caller: future operators. Effect: records that exactly two sequential private TRAIN reproductions already finished, that both explicit private roots were recoverably moved after review, and that official evaluate/inspect, TEST labels, promotion, and Task 6 remain forbidden.",
            symbol="Exp464 Task 5 post-run operator boundary",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "implementation_report",
            "docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html",
            "Caller: Task 5 reviewed closeout. Effect: records exact private-run hashes, timings, retained pre-science failures, the sole C6 failure row, the C8 non-promotable clue, tracked numpy_cpu frozen-interpreter fallback authority, and the qualification-failed terminal boundary.",
            symbol="Exp464 구현 보고서 — Task 5 reviewed closeout complete",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: active research ledger. Effect: stores stopped_before_official_evaluation_qualification_failed, the two exact-equal private non-official reproductions, removed-after-review recoverable root locations, still-zero official/label counts, and all-official-result hashes null.",
            symbol="\"exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment catalog. Effect: registers the reviewed Task 5 private reproducibility closeout with qualification failure and no official result.",
            symbol="\"id\": \"Exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion history. Effect: preserves champion and pending state while recording the Task 5 qualification-failed stop with no official result.",
            symbol="\"experiment\": \"Exp464\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "governance_test",
            governance_test,
            "Caller: Task 5 result verification. Effect: rejects stale ledgers, wrong checkbox truth, stale map truth, incorrect private-run hashes, official/private drift, or active Exp464 work.",
            symbol="test_exp464_task5_private_runs_complete_and_task6_blocked_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "map_builder",
            "scratch/build_codemap_exp464_antigravity.py",
            "Caller: Task 5 boundary validation. Effect: writes and validates the Exp464 closed-CLI graph, embedded JSON, lock fingerprints, and source-parent binding.",
            symbol="build_triplet_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task2_config",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/config_antigravity.py",
            "Caller: Task 3 scorer, Task 4 qualification, and Task 5 closed CLI. Effect: freezes the method order, seed pair, gates, and C0/C1/C2/C7 Exp463 control summaries.",
            symbol="FROZEN_CONFIG_ANTIGRAVITY",
            tests=(governance_test, task3_scoring_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task2_control_row_evidence_file",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/exp463_control_row_evidence_antigravity.jsonl",
            "Caller: tracked control-evidence loader. Effect: freezes the reviewed forty Exp463 C0/C1/C2/C7 control rows plus canonical provenance and digests for source-free authority replay.",
            symbol="exp463_control_row_evidence_antigravity.jsonl",
            tests=(governance_test, task2_predecessor_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task2_control_row_evidence_module",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/control_row_evidence_antigravity.py",
            "Caller: Task 2 contracts and predecessor capture. Effect: parses the tracked control-row evidence file, validates metadata/order, and re-derives exact per-fold control digests.",
            symbol="load_control_row_evidence_antigravity",
            tests=(governance_test, task2_predecessor_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task2_contracts",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/contracts_antigravity.py",
            "Caller: Task 3 scorer and Task 4 qualification. Effect: freezes exact-type Exp463 summaries and rejects forged authority fields.",
            symbol="Exp463AuthorityAntigravity",
            tests=(governance_test, task2_predecessor_test, task3_scoring_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task2_predecessor",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/predecessor_antigravity.py",
            "Caller: Task 3 scorer and Task 4 qualification. Effect: captures frozen Exp463 terminal authority from the approved tracked truth only.",
            symbol="capture_exp463_authority_antigravity",
            tests=(governance_test, task2_predecessor_test, task3_scoring_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task2_sources",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/sources_antigravity.py",
            "Caller: Task 3 scorer and Task 4 qualification. Effect: replays the exact TRAIN-only 1413x140 matrix without label access.",
            symbol="load_normal2_train_antigravity",
            tests=(governance_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task3_centered_ramp_feature",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/centered_ramp_antigravity.py",
            "Caller: task3_scoring. Effect: computes centered residual amplitude/cosine evidence that Task 4 consumes only through frozen fold scores.",
            symbol="CenteredRampEvidenceAntigravity",
            tests=(task3_feature_test, task3_scoring_test, exp463_feature_regression),
        ),
        _node_antigravity(
            "task3_raw_evidence",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/raw_evidence_antigravity.py",
            "Caller: Task 3 scoring and Task 4 qualification. Effect: encodes and decodes raw vector evidence so replay can recompute amplitudes, cosines, counts, and score arrays source-free.",
            symbol="encode_vector_matrix_antigravity",
            tests=(governance_test, task3_scoring_test, task4_qualification_test),
        ),
        _node_antigravity(
            "task3_scoring",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/scoring_antigravity.py",
            "Caller: Task 4 qualification. Effect: produces the exact per-fold score arrays, confirmation counts, and provenance for C0–C8.",
            symbol="score_fold_methods_antigravity",
            tests=(governance_test, task3_scoring_test, exp463_scoring_regression, task4_qualification_test),
        ),
        _node_antigravity(
            "task4_qualification",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/qualification_antigravity.py",
            "Caller: Task 4 evaluation, artifacts, replay, dashboard export, and Task 5 closed CLI. Effect: recomputes AUC/Wilson/finite gates from score arrays and normalizes C6 against C0 plus historical period floors; reviewed Task 4 is SHIP.",
            symbol="qualify_methods_antigravity",
            tests=(task4_qualification_test, task4_evaluation_test, task4_artifacts_test, task4_replay_test, governance_test),
        ),
        _node_antigravity(
            "task4_evaluation",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/evaluation_antigravity.py",
            "Caller: Task 4 artifacts and dashboard export. Effect: turns the normalized qualification contract into one fixed C6-only decision while routing remains blocked until later lifecycle authorization.",
            symbol="evaluate_qualification_antigravity",
            tests=(task4_evaluation_test, task4_artifacts_test, task4_dashboard_test, governance_test),
        ),
        _node_antigravity(
            "task4_artifacts",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/artifacts_antigravity.py",
            "Caller: Task 4 replay, Task 5 inspect, and internal Task 6 official lifecycle. Effect: publishes exactly seven canonical leaves with a self-excluding artifact map and reserves the exact official root for internal lifecycle authority only.",
            symbol="publish_result_antigravity",
            tests=(task4_artifacts_test, task4_replay_test, governance_test),
        ),
        _node_antigravity(
            "task4_replay",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/replay_antigravity.py",
            "Caller: Task 4 verification, Task 5 inspect, and internal Task 6 official lifecycle. Effect: reopens only a supplied root, semantically rejects rehashed tampering, and allows the exact official root only through internal lifecycle authority.",
            symbol="inspect_replay_at_antigravity",
            tests=(task4_replay_test, governance_test),
        ),
        _node_antigravity(
            "task4_dashboard_export",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/dashboard_export_antigravity.py",
            "Caller: Task 4 artifacts and review UI. Effect: exposes aggregate-only dashboard scalars with registered method and metric names only.",
            symbol="export_dashboard_antigravity",
            tests=(task4_dashboard_test, task4_artifacts_test, governance_test),
        ),
        _node_antigravity(
            "task4_qualification_test",
            task4_qualification_test,
            "Caller: Task 4 RED→GREEN metric verification. Effect: covers row ordering, score-derived recomputation, normalization, and pair-diagnostic accounting.",
            symbol="QualificationAntigravityTest",
            tests=(task4_qualification_test,),
        ),
        _node_antigravity(
            "task4_evaluation_test",
            task4_evaluation_test,
            "Caller: Task 4 RED→GREEN decision verification. Effect: covers fixed C6-only evaluation and blocked-before-routing status.",
            symbol="EvaluationAntigravityTest",
            tests=(task4_evaluation_test,),
        ),
        _node_antigravity(
            "task4_artifacts_test",
            task4_artifacts_test,
            "Caller: Task 4 RED→GREEN publication verification. Effect: covers exact seven-leaf inventory, canonical JSON/JSONL, and create-only private roots.",
            symbol="ArtifactsAntigravityTest",
            tests=(task4_artifacts_test,),
        ),
        _node_antigravity(
            "task4_replay_test",
            task4_replay_test,
            "Caller: Task 4 RED→GREEN replay verification. Effect: covers source-free reconstruction and the canonical rehash tamper matrix.",
            symbol="ReplayAntigravityTest",
            tests=(task4_replay_test,),
        ),
        _node_antigravity(
            "task4_dashboard_test",
            task4_dashboard_test,
            "Caller: Task 4 RED→GREEN dashboard verification. Effect: covers aggregate-only export with no raw dataset names, labels, or unrestricted caller strings.",
            symbol="DashboardExportAntigravityTest",
            tests=(task4_dashboard_test,),
        ),
        _node_antigravity(
            "task5_closed_cli",
            "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/cli_antigravity.py",
            "Caller: Task 5 launcher and the public evaluate/inspect dispatch. Effect: the closed CLI binds frozen validation and scientific dependencies, rejects invalid private roots before validation or heavy compute, permits only non-official private root injection, and carries the exact official root only through an internal identity-backed Task 6 evaluate→inspect path.",
            symbol="main_antigravity",
            tests=(governance_test, task5_cli_test),
        ),
        _node_antigravity(
            "task5_launcher",
            "scratch/run_exp464_ecg_residual_centered_ramp_confirmation_antigravity.py",
            "Caller: outside-repository operators and launcher regression. Effect: re-execs the fixed governed interpreter and injects the repository path before the closed CLI imports.",
            symbol="main_antigravity",
            tests=(task5_cli_test,),
        ),
        _node_antigravity(
            "task5_cli_test",
            task5_cli_test,
            "Caller: Task 5 Phase A RED→GREEN boundary verification. Effect: covers exact parser rejection, lazy imports, outside-CWD launcher, pre-validation private-root rejection, public-dispatch official evaluate→inspect wiring, and label-query count zero.",
            symbol="CliAntigravityTest",
            tests=(task5_cli_test,),
        ),
    ]
    edges = [
        {"from": "approved_design", "relation": "freezes", "to": "preregistration"},
        {"from": "approved_design", "relation": "freezes", "to": "task3_scoring"},
        {"from": "approved_design", "relation": "freezes", "to": "task4_qualification"},
        {"from": "implementation_plan", "relation": "bounds_current_work", "to": "task4_qualification"},
        {"from": "implementation_plan", "relation": "bounds_current_work", "to": "task4_evaluation"},
        {"from": "implementation_plan", "relation": "bounds_current_work", "to": "task4_artifacts"},
        {"from": "implementation_plan", "relation": "bounds_current_work", "to": "task4_replay"},
        {"from": "implementation_plan", "relation": "bounds_current_work", "to": "task4_dashboard_export"},
        {"from": "implementation_plan", "relation": "bounds_future_work", "to": "task5_closed_cli"},
        {"from": "research_state", "relation": "reports_status", "to": "dashboard"},
        {"from": "experiment_catalog", "relation": "reports_identity", "to": "dashboard"},
        {"from": "champion_ledger", "relation": "preserves_champion_state", "to": "dashboard"},
        {"from": "governance_test", "relation": "covers", "to": "dashboard"},
        {"from": "governance_test", "relation": "covers", "to": "preregistration"},
        {"from": "governance_test", "relation": "covers", "to": "runbook"},
        {"from": "governance_test", "relation": "covers", "to": "implementation_report"},
        {"from": "governance_test", "relation": "covers", "to": "research_state"},
        {"from": "governance_test", "relation": "covers", "to": "experiment_catalog"},
        {"from": "governance_test", "relation": "covers", "to": "champion_ledger"},
        {"from": "governance_test", "relation": "covers", "to": "map_builder"},
        {"from": "governance_test", "relation": "covers", "to": "task4_qualification"},
        {"from": "governance_test", "relation": "covers", "to": "task4_evaluation"},
        {"from": "governance_test", "relation": "covers", "to": "task4_artifacts"},
        {"from": "governance_test", "relation": "covers", "to": "task4_replay"},
        {"from": "governance_test", "relation": "covers", "to": "task4_dashboard_export"},
        {"from": "task2_config", "relation": "freezes_inputs_for", "to": "task2_contracts"},
        {"from": "task2_control_row_evidence_file", "relation": "freezes_inputs_for", "to": "task2_control_row_evidence_module"},
        {"from": "task2_control_row_evidence_module", "relation": "feeds", "to": "task2_contracts"},
        {"from": "task2_control_row_evidence_module", "relation": "feeds", "to": "task2_predecessor"},
        {"from": "task2_config", "relation": "freezes_inputs_for", "to": "task2_predecessor"},
        {"from": "task2_config", "relation": "freezes_inputs_for", "to": "task3_scoring"},
        {"from": "task2_contracts", "relation": "guards", "to": "task4_qualification"},
        {"from": "task2_predecessor", "relation": "feeds", "to": "task4_qualification"},
        {"from": "task2_sources", "relation": "feeds", "to": "task4_qualification"},
        {"from": "task3_centered_ramp_feature", "relation": "feeds", "to": "task3_scoring"},
        {"from": "task3_raw_evidence", "relation": "feeds", "to": "task3_scoring"},
        {"from": "task3_raw_evidence", "relation": "feeds", "to": "task4_qualification"},
        {"from": "task3_scoring", "relation": "feeds", "to": "task4_qualification"},
        {"from": "task4_qualification", "relation": "feeds", "to": "task4_evaluation"},
        {"from": "task4_qualification", "relation": "feeds", "to": "task4_artifacts"},
        {"from": "task4_qualification", "relation": "feeds", "to": "task4_replay"},
        {"from": "task4_evaluation", "relation": "feeds", "to": "task4_artifacts"},
        {"from": "task4_evaluation", "relation": "feeds", "to": "task4_dashboard_export"},
        {"from": "task4_artifacts", "relation": "feeds", "to": "task4_replay"},
        {"from": "task4_artifacts", "relation": "feeds", "to": "task5_closed_cli"},
        {"from": "task4_replay", "relation": "guards", "to": "task5_closed_cli"},
        {"from": "task5_launcher", "relation": "invokes", "to": "task5_closed_cli"},
        {"from": "task5_cli_test", "relation": "covers", "to": "task5_closed_cli"},
        {"from": "task5_cli_test", "relation": "covers", "to": "task5_launcher"},
        {"from": "map_builder", "relation": "documents_boundary", "to": "task5_closed_cli"},
    ]
    graph = {
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "graph_schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "source_parent": source_parent,
        "allowed_source_commits": _allowed_source_commits_antigravity(source_parent),
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "nodes": nodes,
        "node_count": len(nodes),
        "edges": edges,
        "edge_count": len(edges),
        "flows": [
            "governance_test -> docs/ledgers/map",
            "Task 2 foundation -> Task 3 scoring -> Task 4 qualification/evaluation/artifacts/replay/dashboard",
            "Task 4 artifacts/replay -> Task 5 closed CLI",
        ],
        "flow_count": 3,
    }
    graph["flows"] = [
        "governance_test -> docs/ledgers/map",
        "tracked Exp463 control evidence file -> control evidence loader -> contracts/predecessor",
        "raw evidence codecs -> Task 3 scoring -> Task 4 qualification/replay",
        "Task 2 foundation -> Task 3 scoring -> Task 4 qualification/evaluation/artifacts/replay/dashboard",
        "Task 5 launcher/test -> closed CLI -> Task 4 qualification/evaluation/artifacts/replay -> private non-official result truth",
        "Task 4 artifacts/replay -> Task 5 closed CLI",
        "internal official lifecycle authority -> Task 4 artifacts/replay -> exact official root",
        "Task 5 closed CLI -> two completed exact-equal private TRAIN reproductions -> final review SHIP -> Task 6 forbidden",
    ]
    graph["flow_count"] = len(graph["flows"])
    return graph


def _build_html_antigravity(graph: dict[str, Any]) -> str:
    payload = _canonical_json_antigravity(graph)
    return (
        "<!doctype html>\n"
        '<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>Exp464 Task 5 Closed CLI Code Map</title>\n"
        "<style>body{max-width:1160px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.1rem .3rem;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #aab3bf;padding:.55rem;text-align:left;vertical-align:top}</style></head>\n"
        "<body><main>\n"
        "<h1>Exp464 Task 5 closed-CLI boundary code map</h1>\n"
        "<p>이 map은 <code>stopped_before_official_evaluation_qualification_failed</code> 상태에서, 구현된 Task 5 closed CLI·launcher·CLI test boundary와 이미 끝난 두 번의 exact-equal private TRAIN reproduction truth를 설명한다. private seam은 non-official root만 허용하며 invalid private root는 validation이나 heavy compute 전에 거부된다. public dispatch는 exact official root를 내부 official lifecycle identity 경로로만 운반한다. final review는 SHIP이며 explicit private roots는 review 후 recoverable move 되었고 Task 6은 permanently forbidden이다.</p>\n"
        f'<script id="graph-data" type="application/json">{payload}</script>\n'
        "</main></body></html>\n"
    )


def _tracked_fingerprints_antigravity() -> list[dict[str, str]]:
    fingerprints: list[dict[str, str]] = []
    for relative in TRACKED_PATHS_ANTIGRAVITY:
        path = ROOT_ANTIGRAVITY / relative
        fingerprints.append({"path": relative, "sha256": _sha256_file_antigravity(path)})
    return fingerprints


def _build_lock_antigravity(
    graph_bytes: bytes,
    html_bytes: bytes,
    source_parent: str,
) -> dict[str, Any]:
    return {
        "lock_schema": LOCK_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent": source_parent,
        "generated_files": list(TRIPLET_ANTIGRAVITY),
        "graph_sha256": _sha256_bytes_antigravity(graph_bytes),
        "html_sha256": _sha256_bytes_antigravity(html_bytes),
        "tracked_fingerprints": _tracked_fingerprints_antigravity(),
        "fingerprint_count": len(TRACKED_PATHS_ANTIGRAVITY),
    }


def _render_triplet_antigravity() -> tuple[dict[str, Any], bytes, bytes, bytes]:
    source_parent = _source_parent_antigravity()
    graph = _build_graph_antigravity(source_parent)
    graph_bytes = _canonical_json_bytes_antigravity(graph)
    html_bytes = _build_html_antigravity(graph).encode("utf-8")
    lock_bytes = _canonical_json_bytes_antigravity(
        _build_lock_antigravity(graph_bytes, html_bytes, source_parent),
    )
    return graph, graph_bytes, html_bytes, lock_bytes


def build_triplet_antigravity(*, validate_only: bool = False) -> dict[str, Any]:
    graph, graph_bytes, html_bytes, lock_bytes = _render_triplet_antigravity()
    targets = (
        (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0], html_bytes),
        (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1], graph_bytes),
        (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2], lock_bytes),
    )
    if validate_only:
        mismatches = [
            str(path)
            for path, expected in targets
            if (not path.is_file()) or path.read_bytes() != expected
        ]
        if mismatches:
            raise SystemExit(
                "Exp464 map triplet is stale; rebuild required: " + ", ".join(mismatches),
            )
    else:
        for path, payload in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return {
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "source_parent": graph["source_parent"],
        "allowed_source_commits": graph["allowed_source_commits"],
    }


def _parse_args_antigravity() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="fail if the checked-in triplet differs from the generated content",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args_antigravity()
    build_triplet_antigravity(validate_only=args.validate_only)


if __name__ == "__main__":
    main()
