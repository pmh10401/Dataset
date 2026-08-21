"""Build and validate the bounded Exp463 Task 5 code-map triplet."""

from __future__ import annotations

import hashlib
import html
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp463_antigravity.html",
    "docs/codemap/codemap_exp463_antigravity.json",
    "docs/codemap/codemap_exp463_antigravity.lock",
)
GENERATED_AT_ANTIGRAVITY = "2026-08-21T00:00:00Z"
GRAPH_SCHEMA_ANTIGRAVITY = "exp463_task5_boundary_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp463_task5_boundary_codemap_lock_v1"
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP463_DASHBOARD_antigravity.html",
    "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP463_PREREGISTRATION_antigravity.html",
    "docs/EXP463_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html",
    "docs/superpowers/specs/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity-design.html",
    ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-2-report-antigravity.html",
    ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-3-report-antigravity.html",
    ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-5-report-antigravity.html",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/qualification_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/evaluation_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/artifacts_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/replay_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/dashboard_export_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/config_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/contracts_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/predecessor_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/sources_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/ramp_features_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/scoring_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_config_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_sources_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_ramp_features_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_scoring_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_dashboard_export_antigravity.py",
    "scratch/build_codemap_exp463_antigravity.py",
    "scratch/run_exp463_ecg_prefix_ramp_gated_qualification_antigravity.py",
)
STABLE_FINGERPRINT_EXCLUDED_PREFIXES_ANTIGRAVITY = (
    ".superpowers",
    ".codex",
    ".worktrees",
)


def _sha256_bytes_antigravity(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_antigravity(path: Path) -> str:
    return _sha256_bytes_antigravity(path.read_bytes())


def _strict_json_antigravity(path: Path) -> Any:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates_antigravity,
    )


def _canonical_json_antigravity(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT_ANTIGRAVITY, text=True
    ).strip()


def _current_head_antigravity() -> str:
    return _git_output_antigravity("rev-parse", "HEAD")


def _allowed_source_commits_antigravity() -> set[str]:
    parent_line = _git_output_antigravity(
        "rev-list", "--parents", "-n", "1", "HEAD"
    ).split()
    return set(parent_line)


def _node_antigravity(
    node_id: str,
    path: str,
    role: str,
    *,
    symbol: str,
    tests: tuple[str, ...] = (),
    evidence_path: str | None = None,
) -> dict[str, Any]:
    evidence_path = path if evidence_path is None else evidence_path
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "evidence": [{"path": evidence_path, "symbol": symbol}],
        "tests": list(tests),
    }


def _build_graph_antigravity(source_parent_commit: str | None = None) -> dict[str, Any]:
    governance_test = (
        "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    plan = (
        "docs/superpowers/plans/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html"
    )
    nodes = [
        _node_antigravity(
            "approved_design",
            "docs/superpowers/specs/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity-design.html",
            "Caller: Task 1–6 governance. Effect: freezes Exp463 identity, TRAIN-only data, M0–M6 descriptions, M2 primary gate, and official-root absence.",
            symbol="Exp463 — 앞부분 기준 경사와 보수적 결합 검증",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "implementation_plan",
            plan,
            "Caller: Task 1–6 execution. Effect: records bounded files, frozen M0–M6 IDs, Task 5 review SHIP after private qualification failure, checked Task5 controls, and forbidden unchecked Task6 controls.",
            symbol="Task 1 — governance, dashboard, code map",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "preregistration",
            "docs/EXP463_PREREGISTRATION_antigravity.html",
            "Caller: governance review. Effect: freezes TRAIN-only data, M0–M6 descriptions, M2 sole primary, fixed gates, and null pre-result hashes.",
            symbol="Frozen M0–M6 comparison",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "runbook",
            "docs/EXP463_RUNBOOK_antigravity.html",
            "Caller: future operators. Effect: blocks official evaluate/inspect, TEST labels, root creation, routing, and promotion before separate approval.",
            symbol="Task 1 operator boundary",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "implementation_report",
            "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
            "Caller: Task 2/3/4/5 closeout. Effect: records frozen Exp462 authority, Task 3 scoring boundary, Task 4 evaluation/publication/replay evidence, Task 5 review SHIP, and terminal pre-official stop.",
            symbol="Exp463 Task 4",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP463_DASHBOARD_antigravity.html",
            "Caller: status review. Effect: shows Task 5 review SHIP after M2 qualification failure, Task6 forbidden/no-next-action, easy Korean explanation, frozen authority inventory, aggregate gates, official0, label0, null official hashes, and private cleanup.",
            symbol="task_4_complete_pending_task_5",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: governance ledger. Effect: records terminal Task5 review SHIP qualification failure with null official hashes, official0, label0, absent root, private cleanup, and unchanged pending state.",
            symbol="\"exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment catalog. Effect: registers the same terminal Task5 review SHIP Exp463 record without official result materialization or promotion.",
            symbol="\"Exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion review. Effect: records Task5 review SHIP qualification failure while preserving champion, pending candidate, routing, and promotion.",
            symbol="\"experiment\": \"Exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "governance_regression",
            governance_test,
            "Caller: Task 1–5 regression. Effect: rejects missing governance/foundation/evaluation/CLI surfaces, ledger drift, nonzero official lifecycle, label access, root presence, map mismatch, checkbox drift, and champion/pending mutation.",
            symbol="test_task1_governance_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "map_builder",
            "scratch/build_codemap_exp463_antigravity.py",
            "Caller: governance validation. Effect: builds and validates the Task5 bounded graph, resolving evidence, exact HTML-embedded JSON, and clean source-parent lock.",
            symbol="main_antigravity_validate_only_for_tests",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task2_foundation",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/config_antigravity.py",
            "Caller: future Exp463 scoring/qualification. Effect: freezes identity, M0–M6, gates, Exp462 authority, and TRAIN shape; Task 2–5 reviews are SHIP and no next official action is allowed.",
            symbol="FROZEN_CONFIG_ANTIGRAVITY",
            tests=(
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_config_antigravity.py",
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_contracts_antigravity.py",
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_predecessor_antigravity.py",
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_sources_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task2_report",
            ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-2-report-antigravity.html",
            "Caller: Task 2 review. Effect: records frozen Exp462 authority, immutable TRAIN-source contracts, synchronized M0–M6 definitions, and the closed next-stage boundary.",
            symbol="Exp463 Task 2 — frozen Exp462 authority and TRAIN source",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task3_report",
            ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-3-report-antigravity.html",
            "Caller: Task 3 review. Effect: records complete-pending-Task4 state, RED/GREEN evidence, exact reference parity, MLX decision, public boundaries, and official0/label0 guards.",
            symbol="Exp463 Task 3 — prefix-scaled ramp and seven method scores",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task3_ramp_feature",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/ramp_features_antigravity.py",
            "Caller: Task 3 scoring and future qualification. Effect: validates one finite 140-length row and returns the exact prefix-normalized signed ramp score plus bounded diagnostic variants.",
            symbol="prefix_ramp_score_antigravity",
            tests=(
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_ramp_features_antigravity.py",
                governance_test,
            ),
        ),
        _node_antigravity(
            "task3_scoring",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/scoring_antigravity.py",
            "Caller: future Task 4 qualification. Effect: delegates M0 and M6 reviewed controls, computes deterministic M1/M2/M3/M4/M5 ramp branches, applies the exact rank and 0.995 gate, and returns immutable ordered M0-M6 score evidence.",
            symbol="score_fold_methods_antigravity",
            tests=(
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_scoring_antigravity.py",
                governance_test,
            ),
        ),
        _node_antigravity(
            "task3_ramp_feature_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_ramp_features_antigravity.py",
            "Caller: Task 3 focused regression. Effect: covers independent NumPy reference parity, onset/scale rejection, finite deterministic output, and input preservation.",
            symbol="test_prefix_ramp_matches_independent_reference_and_contract_antigravity",
            tests=(
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_ramp_features_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task3_scoring_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_scoring_antigravity.py",
            "Caller: Task 3 focused regression. Effect: covers exact rank/tie formula, seed-fold dependency flow, M0-M6 order, gate/max semantics, immutability, and public boundary restrictions.",
            symbol="test_fold_scoring_preserves_order_gate_semantics_and_dependency_flow_antigravity",
            tests=(
                "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_scoring_antigravity.py",
            ),
        ),
        _node_antigravity(
            "task4_qualification",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/qualification_antigravity.py",
            "Caller: Task 4 evaluator and Task 5 private run. Effect: recomputes four AUCs, pooled mean, Wilson bound, finite status, ordered 70 rows/seven summaries, and frozen M0/M6 control reproduction from score evidence.",
            symbol="_qualify_methods_with_dependencies_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_qualification_antigravity.py",),
        ),
        _node_antigravity(
            "task4_evaluation",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/evaluation_antigravity.py",
            "Caller: Task 4 publication and Task 5 private run. Effect: evaluates only M2, applies every frozen gate, and permanently blocks routing/label access in the decision.",
            symbol="evaluate_qualification_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_evaluation_antigravity.py",),
        ),
        _node_antigravity(
            "task4_artifacts",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/artifacts_antigravity.py",
            "Caller: Task 5 private run and Task 6 approved preservation. Effect: writes exactly seven canonical create-only leaves with a self-excluding SHA-256 inventory.",
            symbol="publish_result_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_artifacts_antigravity.py",),
        ),
        _node_antigravity(
            "task4_replay",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/replay_antigravity.py",
            "Caller: Task 5/6 inspect. Effect: performs source-free canonical parsing and recomputes metrics, summaries, controls, decision, dashboard, and every stored payload.",
            symbol="inspect_replay_at_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_replay_antigravity.py",),
        ),
        _node_antigravity(
            "task4_dashboard",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/dashboard_export_antigravity.py",
            "Caller: publication and status review. Effect: exports only ordered aggregate method summaries and historical period comparisons; raw score evidence remains private.",
            symbol="export_dashboard_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_dashboard_export_antigravity.py",),
        ),
        _node_antigravity(
            "task4_qualification_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_qualification_antigravity.py",
            "Caller: Task 4 qualification regression. Effect: covers exact 70-row order, score-derived metrics, M2 gate reasons, and M0/M6 control rejection.",
            symbol="test_qualification_emits_exact_ordered_70_rows_and_seven_summaries_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_qualification_antigravity.py",),
        ),
        _node_antigravity(
            "task4_evaluation_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_evaluation_antigravity.py",
            "Caller: Task 4 evaluation regression. Effect: covers blocked M2 decision, all frozen gates, routing false, and label-query zero.",
            symbol="test_evaluation_is_blocked_without_routing_or_test_labels_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_evaluation_antigravity.py",),
        ),
        _node_antigravity(
            "task4_artifacts_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_artifacts_antigravity.py",
            "Caller: Task 4 publication regression. Effect: covers exact seven leaves, canonical bytes, self-excluding inventory, and create-only behavior.",
            symbol="test_publish_is_exact_seven_leaf_canonical_create_only_and_self_excluding_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_artifacts_antigravity.py",),
        ),
        _node_antigravity(
            "task4_replay_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_replay_antigravity.py",
            "Caller: Task 4 replay regression. Effect: covers source-free semantic equality and canonical-rehashed score-evidence, numeric-metric, summary, decision, dashboard, failed-fold, missing/extra-leaf, reordered-row, bool, and string-bool tamper rejection.",
            symbol="test_replay_rejects_rehashed_tamper_matrix_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_replay_antigravity.py",),
        ),
        _node_antigravity(
            "task4_dashboard_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_dashboard_export_antigravity.py",
            "Caller: Task 4 dashboard regression. Effect: covers aggregate-only output, four-type minima, and no source/evidence leakage.",
            symbol="test_dashboard_is_aggregate_only_and_exposes_ordered_four_type_minima_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_dashboard_export_antigravity.py",),
        ),
        _node_antigravity(
            "task5_cli",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
            "Caller: fixed-runtime launcher and private controller. Effect: accepts exactly evaluate --reviewed-commit LOWER40HEX or bare inspect, lazily dispatches the selected phase, rejects official-root aliases, and keeps TEST labels closed.",
            symbol="main_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",),
        ),
        _node_antigravity(
            "task5_cli_tests",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",
            "Caller: Task 5 CLI regression. Effect: covers exact parser closure, lazy/source-free inspect, private create-only seven-leaf flow, label firewall, outside-CWD launcher, and fixed-interpreter re-exec.",
            symbol="test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",),
        ),
        _node_antigravity(
            "task5_launcher",
            "scratch/run_exp463_ecg_prefix_ramp_gated_qualification_antigravity.py",
            "Caller: future approved CLI operator. Effect: binds the fixed Python 3.14.6 interpreter and inserts the repository path so the closed CLI works outside repository CWD.",
            symbol="_ensure_fixed_runtime_antigravity",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",),
        ),
        _node_antigravity(
            "task5_report",
            ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/task-5-report-antigravity.html",
            "Caller: Task 5 closeout review. Effect: records exact two-run private TRAIN evidence, scientific gate failure, MLX manifest/fallback, no-official/label-zero boundary, and the disclosed create-only race concern.",
            symbol="Exp463 Task 5 report",
            tests=("experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_governance_antigravity.py",),
        ),
    ]
    edges = [
        {
            "from": "approved_design",
            "to": "preregistration",
            "type": "governs",
            "evidence": {"path": "docs/EXP463_PREREGISTRATION_antigravity.html", "symbol": "Frozen M0–M6 comparison"},
        },
        {
            "from": "approved_design",
            "to": "dashboard",
            "type": "governs",
            "evidence": {"path": "docs/EXP463_DASHBOARD_antigravity.html", "symbol": "hydra_period_ramp_gated_max"},
        },
        {
            "from": "implementation_plan",
            "to": "dashboard",
            "type": "governs",
            "evidence": {"path": "docs/EXP463_DASHBOARD_antigravity.html", "symbol": "공식 실행 0회"},
        },
        {
            "from": "implementation_plan",
            "to": "governance_regression",
            "type": "governs",
            "evidence": {"path": governance_test, "symbol": "test_task1_dashboard_plan_and_map_antigravity"},
        },
        *[
            {
                "from": "governance_regression",
                "to": target,
                "type": "tests",
                "evidence": {"path": governance_test, "symbol": "test_task1_governance_antigravity"},
            }
            for target in (
                "preregistration",
                "runbook",
                "implementation_report",
                "dashboard",
                "research_state",
                "experiment_catalog",
                "champion_ledger",
            )
        ],
        {
            "from": "governance_regression",
            "to": "map_builder",
            "type": "tests",
            "evidence": {"path": governance_test, "symbol": "test_task1_dashboard_plan_and_map_antigravity"},
        },
        *[
            {
                "from": "implementation_plan",
                "to": target,
                "type": "governs",
                "evidence": {"path": plan, "symbol": symbol},
            }
            for target, symbol in (
                ("task2_foundation", "Task 2 — frozen Exp462 authority and TRAIN source"),
                ("task2_report", "Task 2 — frozen Exp462 authority and TRAIN source"),
                ("task3_report", "Task 3 — prefix-scaled ramp and seven method scores"),
                ("task3_ramp_feature", "Task 3 — prefix-scaled ramp and seven method scores"),
                ("task3_scoring", "Task 3 — prefix-scaled ramp and seven method scores"),
                ("task4_qualification", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task4_evaluation", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task4_artifacts", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task4_replay", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task4_dashboard", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task5_cli", "Task 5 — closed CLI and two private TRAIN reproductions"),
                ("task5_cli_tests", "Task 5 — closed CLI and two private TRAIN reproductions"),
                ("task5_launcher", "Task 5 — closed CLI and two private TRAIN reproductions"),
                ("task5_report", "Task 5 — closed CLI and two private TRAIN reproductions"),
            )
        ],
        {
            "from": "task3_ramp_feature_tests",
            "to": "task3_ramp_feature",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_ramp_features_antigravity.py",
                "symbol": "test_prefix_ramp_matches_independent_reference_and_contract_antigravity",
            },
        },
        {
            "from": "task3_scoring_tests",
            "to": "task3_scoring",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_scoring_antigravity.py",
                "symbol": "test_fold_scoring_preserves_order_gate_semantics_and_dependency_flow_antigravity",
            },
        },
        {
            "from": "task2_foundation",
            "to": "task3_scoring",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/config_antigravity.py",
                "symbol": "FROZEN_CONFIG_ANTIGRAVITY",
            },
        },
        {
            "from": "task3_ramp_feature",
            "to": "task3_scoring",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/scoring_antigravity.py",
                "symbol": "prefix_ramp_score_antigravity",
            },
        },
        {
            "from": "task3_scoring",
            "to": "task4_qualification",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/scoring_antigravity.py",
                "symbol": "score_fold_methods_antigravity",
            },
        },
        {
            "from": "task4_qualification",
            "to": "task4_evaluation",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/qualification_antigravity.py",
                "symbol": "normalize_qualification_antigravity",
            },
        },
        {
            "from": "task4_evaluation",
            "to": "task4_artifacts",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/evaluation_antigravity.py",
                "symbol": "evaluate_qualification_antigravity",
            },
        },
        {
            "from": "task4_artifacts",
            "to": "task4_replay",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/artifacts_antigravity.py",
                "symbol": "_LEAVES_ANTIGRAVITY",
            },
        },
        {
            "from": "task4_evaluation",
            "to": "task4_dashboard",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/dashboard_export_antigravity.py",
                "symbol": "export_dashboard_antigravity",
            },
        },
        {
            "from": "task4_qualification_tests",
            "to": "task4_qualification",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_qualification_antigravity.py",
                "symbol": "test_qualification_emits_exact_ordered_70_rows_and_seven_summaries_antigravity",
            },
        },
        {
            "from": "task4_evaluation_tests",
            "to": "task4_evaluation",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_evaluation_antigravity.py",
                "symbol": "test_evaluation_is_blocked_without_routing_or_test_labels_antigravity",
            },
        },
        {
            "from": "task4_artifacts_tests",
            "to": "task4_artifacts",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_artifacts_antigravity.py",
                "symbol": "test_publish_is_exact_seven_leaf_canonical_create_only_and_self_excluding_antigravity",
            },
        },
        {
            "from": "task4_replay_tests",
            "to": "task4_replay",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_replay_antigravity.py",
                "symbol": "test_replay_rejects_rehashed_tamper_matrix_antigravity",
            },
        },
        {
            "from": "task4_dashboard_tests",
            "to": "task4_dashboard",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_dashboard_export_antigravity.py",
                "symbol": "test_dashboard_is_aggregate_only_and_exposes_ordered_four_type_minima_antigravity",
            },
        },
        {
            "from": "task5_cli_tests",
            "to": "task5_cli",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",
                "symbol": "test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity",
            },
        },
        {
            "from": "task5_cli_tests",
            "to": "task5_launcher",
            "type": "tests",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_cli_antigravity.py",
                "symbol": "test_launcher_imports_package_from_outside_repository_cwd_antigravity",
            },
        },
        {
            "from": "task5_cli",
            "to": "task4_qualification",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
                "symbol": "qualify_methods_antigravity",
            },
        },
        {
            "from": "task5_cli",
            "to": "task4_artifacts",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
                "symbol": "publish_result_antigravity",
            },
        },
        {
            "from": "task5_cli",
            "to": "task4_replay",
            "type": "governs",
            "evidence": {
                "path": "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
                "symbol": "inspect_replay_at_antigravity",
            },
        },
        *[
            {
                "from": "governance_regression",
                "to": target,
                "type": "tests",
                "evidence": {
                    "path": governance_test,
                    "symbol": "test_task1_dashboard_plan_and_map_antigravity",
                },
            }
            for target in ("task2_report", "task3_report", "task5_report")
        ],
    ]
    return {
        "edges": edges,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "nodes": nodes,
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "source_parent_commit": (
            _current_head_antigravity()
            if source_parent_commit is None
            else source_parent_commit
        ),
        "unknown_relationships": [],
    }


def _git_tracked_paths_antigravity() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached"],
        cwd=ROOT_ANTIGRAVITY,
    )
    return tuple(
        Path(item)
        for item in output.decode("utf-8").split("\0")
        if item
    )


def _is_stable_fingerprint_path_antigravity(path: Path) -> bool:
    normalized = path.as_posix()
    if any(
        normalized == prefix or normalized.startswith(f"{prefix}/")
        for prefix in STABLE_FINGERPRINT_EXCLUDED_PREFIXES_ANTIGRAVITY
    ):
        return False
    if normalized in TRIPLET_ANTIGRAVITY:
        return False
    return True


def _iter_tree_files_antigravity(relative_root: str) -> list[Path]:
    root = ROOT_ANTIGRAVITY / relative_root
    if root.is_file():
        relative = root.relative_to(ROOT_ANTIGRAVITY)
        return [root] if _is_stable_fingerprint_path_antigravity(relative) else []
    if not root.exists():
        return []
    root_prefix = Path(relative_root).as_posix()
    tracked_paths = []
    for relative in _git_tracked_paths_antigravity():
        if not _is_stable_fingerprint_path_antigravity(relative):
            continue
        if root_prefix not in {"", "."} and not (
            relative.as_posix() == root_prefix
            or relative.as_posix().startswith(f"{root_prefix}/")
        ):
            continue
        path = ROOT_ANTIGRAVITY / relative
        if path.is_file():
            tracked_paths.append(path)
    return sorted(tracked_paths)


def _fingerprint_tree_antigravity(relative_root: str) -> str:
    digest = hashlib.sha256()
    for path in _iter_tree_files_antigravity(relative_root):
        digest.update(path.relative_to(ROOT_ANTIGRAVITY).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_fingerprints_antigravity() -> dict[str, str]:
    tracked = {
        path: _sha256_file_antigravity(ROOT_ANTIGRAVITY / path)
        for path in sorted(TRACKED_PATHS_ANTIGRAVITY)
        if (ROOT_ANTIGRAVITY / path).is_file()
    }
    return tracked


def _module_fingerprints_antigravity() -> dict[str, str]:
    return {
        ".superpowers": _fingerprint_tree_antigravity(".superpowers"),
        "docs": _fingerprint_tree_antigravity("docs"),
        "experiments": _fingerprint_tree_antigravity("experiments"),
        "root": _fingerprint_tree_antigravity("."),
        "scratch": _fingerprint_tree_antigravity("scratch"),
    }


def _lock_antigravity(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "module_fingerprints": _module_fingerprints_antigravity(),
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "source_parent_commit": graph["source_parent_commit"],
        "tracked_fingerprints": _tracked_fingerprints_antigravity(),
        "working_tree_dirty": False,
    }


def _html_antigravity(graph: dict[str, Any]) -> str:
    payload = html.escape(_canonical_json_antigravity(graph), quote=False)
    node_items = "".join(
        f"<li><code>{html.escape(node['id'])}</code> — {html.escape(node['role'])}</li>"
        for node in graph["nodes"]
    )
    return (
        "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Exp463 Task 5 코드맵</title>"
        "<style>body{max-width:1180px;margin:2rem auto;padding:0 1rem;font:15px/1.6 system-ui;color:#172033}"
        "code,pre{background:#eef2f7;padding:.12rem .3rem;overflow:auto}li{margin:.35rem 0}</style></head><body><main>"
        "<h1>Exp463 Task 5 코드맵</h1>"
        "<p>실제 Exp463 Task 1 governance, Task 2 foundation, Task 3 scoring, Task 4 qualification/publication/replay, Task 5 closed CLI/private TRAIN 경계를 기록한다. Task 6은 별도 승인 전까지 실행하지 않는다.</p>"
        f"<h2>노드</h2><ul>{node_items}</ul>"
        "<h2>Machine-readable graph</h2>"
        f"<pre id=\"codemap-json-antigravity\" type=\"application/json\">{payload}</pre>"
        "</main></body></html>"
    )


def _validate_evidence_antigravity(evidence: Any, *, owner: str) -> None:
    if not isinstance(evidence, dict):
        raise AssertionError(f"{owner} evidence must be an object")
    path = evidence.get("path")
    symbol = evidence.get("symbol")
    if not isinstance(path, str) or not path:
        raise AssertionError(f"{owner} evidence path is required")
    if not isinstance(symbol, str) or not symbol:
        raise AssertionError(f"{owner} evidence symbol is required")
    target = ROOT_ANTIGRAVITY / path
    if not target.is_file():
        raise AssertionError(f"{owner} evidence path missing: {path}")
    if symbol not in target.read_text(encoding="utf-8"):
        raise AssertionError(f"{owner} evidence symbol does not resolve: {symbol}")


def _validate_graph_antigravity(graph: dict[str, Any]) -> None:
    if graph.get("schema") != GRAPH_SCHEMA_ANTIGRAVITY:
        raise AssertionError("unexpected Exp463 graph schema")
    source_parent_commit = graph.get("source_parent_commit")
    if source_parent_commit not in _allowed_source_commits_antigravity():
        raise AssertionError("source parent commit is not HEAD or its immediate parent")
    if graph.get("unknown_relationships") != []:
        raise AssertionError("unresolved graph relationships")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise AssertionError("graph nodes/edges must be lists")
    node_ids = [node.get("id") for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise AssertionError("duplicate graph node id")
    node_set = set(node_ids)
    required = {
        "approved_design",
        "implementation_plan",
        "preregistration",
        "runbook",
        "implementation_report",
        "dashboard",
        "research_state",
        "experiment_catalog",
        "champion_ledger",
        "governance_regression",
        "map_builder",
        "task2_foundation",
        "task2_report",
        "task3_report",
        "task3_ramp_feature",
        "task3_scoring",
        "task3_ramp_feature_tests",
        "task3_scoring_tests",
        "task4_qualification",
        "task4_evaluation",
        "task4_artifacts",
        "task4_replay",
        "task4_dashboard",
        "task4_qualification_tests",
        "task4_evaluation_tests",
        "task4_artifacts_tests",
        "task4_replay_tests",
        "task4_dashboard_tests",
        "task5_cli",
        "task5_cli_tests",
        "task5_launcher",
        "task5_report",
    }
    if not required.issubset(node_set):
        raise AssertionError(f"missing graph nodes: {sorted(required - node_set)}")
    for node in nodes:
        if not all(isinstance(node.get(key), str) and node.get(key) for key in ("id", "path", "role")):
            raise AssertionError("node identity/effect fields are required")
        if not isinstance(node.get("tests"), list):
            raise AssertionError("node tests must be a list")
        evidence = node.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise AssertionError(f"node evidence is required: {node.get('id')}")
        for index, item in enumerate(evidence):
            _validate_evidence_antigravity(
                item, owner=f"node {node.get('id')} evidence[{index}]"
            )
        if node["id"].endswith("_unknown") and (
            "unknown" not in node["role"] or "not implemented" not in node["role"]
        ):
            raise AssertionError(f"future node is not explicitly unknown: {node['id']}")
    for edge in edges:
        if edge.get("from") not in node_set or edge.get("to") not in node_set:
            raise AssertionError("edge references unknown node")
        if edge.get("type") not in {"governs", "tests"}:
            raise AssertionError("edge type outside bounded vocabulary")
        evidence = edge.get("evidence")
        _validate_evidence_antigravity(evidence, owner="edge")


def validate_map_antigravity(
    graph: dict[str, Any] | None = None,
    lock: dict[str, Any] | None = None,
    rendered: str | None = None,
) -> None:
    graph_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1]
    lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2]
    html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]
    graph = _strict_json_antigravity(graph_path) if graph is None else graph
    lock = _strict_json_antigravity(lock_path) if lock is None else lock
    rendered = html_path.read_text(encoding="utf-8") if rendered is None else rendered
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise AssertionError("map triplet JSON values must be objects")
    _validate_graph_antigravity(graph)
    if lock.get("schema") != LOCK_SCHEMA_ANTIGRAVITY:
        raise AssertionError("unexpected Exp463 lock schema")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise AssertionError("lock represented paths drifted")
    if lock.get("source_parent_commit") != graph["source_parent_commit"]:
        raise AssertionError("lock source parent differs from graph")
    if lock.get("working_tree_dirty") is not False:
        raise AssertionError("source-parent lock must be clean")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity():
        raise AssertionError("tracked fingerprint drift detected")
    if lock.get("module_fingerprints") != _module_fingerprints_antigravity():
        raise AssertionError("module fingerprint drift detected")
    marker = '<pre id="codemap-json-antigravity" type="application/json">'
    if marker not in rendered:
        raise AssertionError("HTML map JSON marker missing")
    payload = rendered.split(marker, 1)[1].split("</pre>", 1)[0]
    if json.loads(html.unescape(payload)) != graph:
        raise AssertionError("HTML embedded graph differs from JSON graph")


def build_map_antigravity() -> None:
    graph = _build_graph_antigravity()
    lock = _lock_antigravity(graph)
    (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1]).write_text(
        _canonical_json_antigravity(graph) + "\n", encoding="utf-8"
    )
    (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).write_text(
        _html_antigravity(graph), encoding="utf-8"
    )
    (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2]).write_text(
        _canonical_json_antigravity(lock) + "\n", encoding="utf-8"
    )
    validate_map_antigravity(graph, lock, _html_antigravity(graph))


def main_antigravity_validate_only_for_tests() -> None:
    validate_map_antigravity()


def main() -> None:
    build_map_antigravity()


if __name__ == "__main__":
    main()
