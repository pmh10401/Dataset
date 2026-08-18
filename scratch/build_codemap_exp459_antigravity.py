#!/usr/bin/env python3
"""Build and validate the bounded Exp459 Task 5 CLI/authority code map."""

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
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/artifacts_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/dashboard_export_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/evaluation_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/frozen_validation_authority_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/cli_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/qualification_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/replay_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/sources_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_blocked_folds_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_cli_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_controlled_fork_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/test_sources_antigravity.py",
    "scratch/build_codemap_exp459_antigravity.py",
    "scratch/run_exp459_ecg_normal2_blocked_validation_antigravity.py",
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
        f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>'
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp459 Task 5 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp459 Task 5 CLI/authority 코드맵</h1><p>Task 5는 CLI, fixed-runtime launcher, exact private-run frozen authority, unstable validation truth, updated ledgers/docs, and bounded tests/map을 추가한다. 공식 evaluate/inspect, label query, 공식 root 생성, champion 변경은 여전히 없다.</p>'
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
    sources_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/sources_antigravity.py"
    )
    qualification_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/qualification_antigravity.py"
    )
    evaluation_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/evaluation_antigravity.py"
    )
    artifacts_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/artifacts_antigravity.py"
    )
    replay_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/replay_antigravity.py"
    )
    dashboard_export_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/dashboard_export_antigravity.py"
    )
    frozen_authority_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/frozen_validation_authority_antigravity.py"
    )
    cli_path = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/cli_antigravity.py"
    )
    launcher_path = "scratch/run_exp459_ecg_normal2_blocked_validation_antigravity.py"
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
    sources_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_sources_antigravity.py"
    )
    qualification_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_qualification_antigravity.py"
    )
    evaluation_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_evaluation_antigravity.py"
    )
    artifacts_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_artifacts_antigravity.py"
    )
    replay_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_replay_antigravity.py"
    )
    dashboard_export_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_dashboard_export_antigravity.py"
    )
    cli_test = (
        "experiments/exp459_ecg_normal2_blocked_validation_antigravity/tests/"
        "test_cli_antigravity.py"
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
            "role": "Caller: Task 1–4 execution and future Task 5 operator sequence. Effect: records checked Task 1, Task 2, Task 3, and Task 4 steps while Task 5 now explicitly requires private qualification/evaluation twice, canonical equality, version-controlled authority, full tests, fresh review, and Task 6 approval after SHIP.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 4: stable 판단, 7-leaf 결과, replay와 대시보드")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP459_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: states Task 5 private unstable validation truth, fixed thresholds, zero official actions, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_PREREGISTRATION_antigravity.html", "implementation_verified_awaiting_official_evaluation")],
        },
        {
            "id": "runbook",
            "path": "docs/EXP459_RUNBOOK_antigravity.html",
            "role": "Caller: future operators. Effect: records that Task 4 completed evaluation, artifacts, replay, and dashboard export while official commands, root creation, label access, actual authority values, and Task 6 execution remain blocked until the exact Task 5 authority sequence completes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_RUNBOOK_antigravity.html", "Task 5 authority 순서")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: implementation review. Effect: reports the Task 5 private unstable validation result, frozen authority digest/file, CLI/import boundary, and unchanged official lifecycle.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html", "Task 5 private validation result")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP459_DASHBOARD_antigravity.html",
            "role": "Caller: reader-facing review. Effect: shows Task 5/6 progress, unstable status, zero official commands, present frozen authority file, failed fold rows, tuning prohibition, and unchanged champion state.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP459_DASHBOARD_antigravity.html", "Task 5 private validation result")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Exp459 Task 5 private unstable authority truth, zero official invocations, label_query_count 0, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"task_5_private_unstable_authority_frozen\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp459 documents, Task 5 private unstable authority status, frozen predecessor link, official-zero status, and no-result truth.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"task_5_private_unstable_authority_frozen\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp459 Task 5 private unstable authority truth without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"task_5_private_unstable_authority_frozen\"")],
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
            "id": "task3_sources",
            "path": sources_path,
            "role": "Caller: Task 3 qualification and source tests. Effect: loads exactly one ECG5000_normal_2 TRAIN values matrix as owned read-only float32 1413x140 without labels or TEST rows.",
            "tests": [sources_test],
            "evidence": [evidence_antigravity(sources_path, "load_normal2_train_antigravity")],
        },
        {
            "id": "task3_qualification",
            "path": qualification_path,
            "role": "Caller: Task 4 evaluation, future CLI, and Task 3 tests. Effect: calls the blocked fold scorer in exact seed-then-fold order, computes fold AUC/Wilson pass records, seed summaries, and overall qualification.",
            "tests": [qualification_test],
            "evidence": [evidence_antigravity(qualification_path, "qualify_blocked_normal2_antigravity")],
        },
        {
            "id": "task4_evaluation",
            "path": evaluation_path,
            "role": "Caller: Task 4 artifacts, dashboard export, replay, and evaluation tests. Effect: validates exact ordered qualification evidence, returns stable only when all ten fold-seed records pass, exposes immutable Exp458 metrics as predecessor context only, and defines the frozen external validation authority contract.",
            "tests": [evaluation_test, artifacts_test, replay_test, dashboard_export_test],
            "evidence": [evidence_antigravity(evaluation_path, "evaluate_blocked_qualification_antigravity")],
        },
        {
            "id": "task4_artifacts",
            "path": artifacts_path,
            "role": "Caller: Task 4 replay, future CLI, and artifact tests. Effect: publishes canonical seven-leaf result roots create-only only when evaluator evidence matches the trusted frozen validation authority, with a self-excluding six-leaf artifact hash map.",
            "tests": [artifacts_test, replay_test],
            "evidence": [evidence_antigravity(artifacts_path, "publish_result_antigravity")],
        },
        {
            "id": "task4_replay",
            "path": replay_path,
            "role": "Caller: Task 4 replay tests and future inspect CLI. Effect: requires a trusted frozen validation authority and rejects inventory, canonical JSON, hash-map, coordinated fold/AUC/Wilson/seed/final decision/dashboard/map tamper without reading source rows.",
            "tests": [replay_test],
            "evidence": [evidence_antigravity(replay_path, "inspect_replay_at_antigravity")],
        },
        {
            "id": "task4_dashboard_export",
            "path": dashboard_export_path,
            "role": "Caller: Task 4 artifacts, replay, dashboard tests, and future reports. Effect: exports only aggregate TRAIN evidence, two seed summaries, immutable fixed Exp458 predecessor metrics, and label_query_count 0 from evaluator or trusted authority payloads.",
            "tests": [dashboard_export_test, artifacts_test, replay_test],
            "evidence": [evidence_antigravity(dashboard_export_path, "export_dashboard_antigravity")],
        },
        {
            "id": "task5_frozen_authority",
            "path": frozen_authority_path,
            "role": "Caller: Task 5 CLI evaluate/inspect and replay. Effect: version-controls the exact unstable private-run validation authority with five fold-plan rows, ten evidence rows, two seed summaries, final decision, and canonical digest f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee.",
            "tests": [cli_test],
            "evidence": [evidence_antigravity(frozen_authority_path, "FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY")],
        },
        {
            "id": "task5_cli",
            "path": cli_path,
            "role": "Caller: fixed launcher and future official Task 6 command. Effect: exposes only evaluate --reviewed-commit LOWER40HEX and bare inspect, validates runtime and authority before qualification/root creation, publishes only authority-matching seven-leaf results, and keeps inspect source/SQLite/Hydra-free where possible.",
            "tests": [cli_test],
            "evidence": [evidence_antigravity(cli_path, "main_antigravity")],
        },
        {
            "id": "task5_launcher",
            "path": launcher_path,
            "role": "Caller: future operator command outside repository CWD. Effect: re-execs the fixed Python runtime before importing Exp459 CLI and pins numerical thread environment variables to one.",
            "tests": [cli_test],
            "evidence": [evidence_antigravity(launcher_path, "_ensure_fixed_runtime_antigravity")],
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
            "id": "task3_source_tests",
            "path": sources_test,
            "role": "Caller: Task 3 verification. Effect: proves the no-argument TRAIN-only source API, one SQL query, no label/TEST surface, exact shape, finite float32, and no aliasing.",
            "tests": [sources_test],
            "evidence": [evidence_antigravity(sources_test, "test_loader_issues_one_normal2_train_value_query_with_no_labels_or_test_antigravity")],
        },
        {
            "id": "task3_qualification_tests",
            "path": qualification_test,
            "role": "Caller: Task 3 verification. Effect: proves exact ten-call seed/fold order, row coverage, scorer validation, one fold/seed failure behavior, and inclusive threshold semantics.",
            "tests": [qualification_test],
            "evidence": [evidence_antigravity(qualification_test, "test_qualification_calls_ten_ordered_blocked_scorers_and_covers_each_row_per_seed_antigravity")],
        },
        {
            "id": "task4_evaluation_tests",
            "path": evaluation_test,
            "role": "Caller: Task 4 verification. Effect: proves stable requires ten ordered passes, unstable follows one failed record, Exp458 metrics stay immutable and predecessor-only, frozen authority carries exact payloads/digest, and forged qualification evidence is rejected.",
            "tests": [evaluation_test],
            "evidence": [evidence_antigravity(evaluation_test, "test_stable_requires_all_ten_ordered_fold_seed_records_antigravity")],
        },
        {
            "id": "task4_artifact_tests",
            "path": artifacts_test,
            "role": "Caller: Task 4 verification. Effect: proves exact seven-leaf canonical create-only publication, self-excluding artifact maps, context payload, evaluator provenance, and trusted-authority enforcement.",
            "tests": [artifacts_test],
            "evidence": [evidence_antigravity(artifacts_test, "test_publishes_exact_canonical_seven_leaf_inventory_create_only_antigravity")],
        },
        {
            "id": "task4_replay_tests",
            "path": replay_test,
            "role": "Caller: Task 4 verification. Effect: proves authority-gated source-free replay and rejection of wrong/digest-mutated authority, coordinated rehashed fold, AUC, Wilson, decision, dashboard, map, inventory, and row-order tamper.",
            "tests": [replay_test],
            "evidence": [evidence_antigravity(replay_test, "test_rejects_rehashed_semantic_tamper_across_result_leaves_antigravity")],
        },
        {
            "id": "task4_dashboard_tests",
            "path": dashboard_export_test,
            "role": "Caller: Task 4 verification. Effect: proves dashboard payload is aggregate-only, includes immutable fixed Exp458 metrics and label_query_count 0, and rejects forged decisions, mutated decisions, or predecessor metric mutation.",
            "tests": [dashboard_export_test],
            "evidence": [evidence_antigravity(dashboard_export_test, "test_dashboard_exports_only_aggregate_train_payload_and_exp458_context_antigravity")],
        },
        {
            "id": "task5_cli_tests",
            "path": cli_test,
            "role": "Caller: Task 5 verification. Effect: proves exact CLI surface, lazy dispatch, runtime/authority fail-closed ordering, synthetic publish/replay, launcher outside-CWD behavior, inspect import boundary, and exact frozen authority digest/failure folds.",
            "tests": [cli_test],
            "evidence": [evidence_antigravity(cli_test, "test_version_controlled_frozen_authority_matches_exact_private_runs_antigravity")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: governance verification. Effect: freezes the self-contained documents, Task 5 private unstable authority truth, official-root absence, non-promotion, and bounded code-map parity.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_task5_authority_sequence_and_import_surface_are_unambiguous_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 5 validation. Effect: recomputes the bounded Task 5 graph, fingerprints, source-parent rule, and HTML/JSON/lock parity.",
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
        {"from": "approved_design", "to": "task3_sources", "type": "governs", "evidence": evidence_antigravity(design, "TRAIN 입력:")},
        {"from": "approved_design", "to": "task3_qualification", "type": "governs", "evidence": evidence_antigravity(design, "점수 계산:")},
        {"from": "approved_design", "to": "task4_evaluation", "type": "governs", "evidence": evidence_antigravity(design, "최종 판단")},
        {"from": "approved_design", "to": "task4_artifacts", "type": "governs", "evidence": evidence_antigravity(design, "결과 파일과 대시보드")},
        {"from": "approved_design", "to": "task4_replay", "type": "governs", "evidence": evidence_antigravity(design, "source-free replay")},
        {"from": "approved_design", "to": "task4_dashboard_export", "type": "governs", "evidence": evidence_antigravity(design, "원시 TRAIN 행")},
        {"from": "implementation_plan", "to": "implementation_report", "type": "governs", "evidence": evidence_antigravity(plan, "SHIP 후에만 Task 6 approval")},
        {"from": "implementation_plan", "to": "dashboard", "type": "governs", "evidence": evidence_antigravity(plan, "Task 4 완료")},
        {"from": "implementation_plan", "to": "task2_config", "type": "governs", "evidence": evidence_antigravity(plan, "exact config와 immutable dataclass 계약을 구현한다.")},
        {"from": "implementation_plan", "to": "task2_contracts", "type": "governs", "evidence": evidence_antigravity(plan, "FoldEvidenceAntigravity")},
        {"from": "implementation_plan", "to": "task2_blocked_folds", "type": "governs", "evidence": evidence_antigravity(plan, "def blocked_plan_antigravity(n_rows: int):")},
        {"from": "implementation_plan", "to": "task3_sources", "type": "governs", "evidence": evidence_antigravity(plan, "sources_antigravity.py")},
        {"from": "implementation_plan", "to": "task3_qualification", "type": "governs", "evidence": evidence_antigravity(plan, "qualify_blocked_normal2_antigravity()")},
        {"from": "implementation_plan", "to": "task4_evaluation", "type": "governs", "evidence": evidence_antigravity(plan, "evaluation_antigravity.py")},
        {"from": "implementation_plan", "to": "task4_artifacts", "type": "governs", "evidence": evidence_antigravity(plan, "artifacts_antigravity.py")},
        {"from": "implementation_plan", "to": "task4_replay", "type": "governs", "evidence": evidence_antigravity(plan, "replay_antigravity.py")},
        {"from": "implementation_plan", "to": "task4_dashboard_export", "type": "governs", "evidence": evidence_antigravity(plan, "dashboard_export_antigravity.py")},
        {"from": "implementation_plan", "to": "task5_cli", "type": "governs", "evidence": evidence_antigravity(plan, "cli_antigravity.py")},
        {"from": "implementation_plan", "to": "task5_frozen_authority", "type": "governs", "evidence": evidence_antigravity(plan, "frozen_validation_authority_antigravity.py")},
        {"from": "implementation_plan", "to": "task5_launcher", "type": "governs", "evidence": evidence_antigravity(plan, "run_exp459_ecg_normal2_blocked_validation_antigravity.py")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": evidence_antigravity(plan, "map을 갱신하고 커밋·검토한다.")},
        {"from": "task2_config_tests", "to": "task2_config", "type": "tests", "evidence": evidence_antigravity(config_test, "test_config_rejects_any_mutated_value_type_or_order_antigravity")},
        {"from": "task2_contract_tests", "to": "task2_contracts", "type": "tests", "evidence": evidence_antigravity(contracts_test, "test_fold_evidence_summary_qualification_and_decision_recompute_semantics_antigravity")},
        {"from": "task2_blocked_fold_tests", "to": "task2_blocked_folds", "type": "tests", "evidence": evidence_antigravity(blocked_folds_test, "test_blocked_plan_source_uses_array_split_without_random_calls_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_config", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_config_changes_only_identity_predecessor_and_fold_planner_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_contracts", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_modules_avoid_forbidden_runtime_expansions_and_bind_science_to_exp458_antigravity")},
        {"from": "task2_controlled_fork_tests", "to": "task2_blocked_folds", "type": "tests", "evidence": evidence_antigravity(controlled_fork_test, "test_exp459_modules_avoid_forbidden_runtime_expansions_and_bind_science_to_exp458_antigravity")},
        {"from": "task2_blocked_folds", "to": "task3_qualification", "type": "governs", "evidence": evidence_antigravity(qualification_path, "_blocked_plan_for_train_antigravity")},
        {"from": "task2_contracts", "to": "task3_qualification", "type": "governs", "evidence": evidence_antigravity(qualification_path, "_fold_evidence_antigravity")},
        {"from": "task3_sources", "to": "task3_qualification", "type": "governs", "evidence": evidence_antigravity(qualification_path, "qualify_blocked_normal2_antigravity")},
        {"from": "task2_contracts", "to": "task4_evaluation", "type": "governs", "evidence": evidence_antigravity(evaluation_path, "validate_blocked_qualification_semantics_antigravity")},
        {"from": "task3_qualification", "to": "task4_evaluation", "type": "governs", "evidence": evidence_antigravity(evaluation_path, "evaluate_blocked_qualification_antigravity")},
        {"from": "task2_blocked_folds", "to": "task4_artifacts", "type": "governs", "evidence": evidence_antigravity(artifacts_path, "_fold_plan_rows_antigravity")},
        {"from": "task4_evaluation", "to": "task4_artifacts", "type": "governs", "evidence": evidence_antigravity(artifacts_path, "publish_result_antigravity")},
        {"from": "task4_dashboard_export", "to": "task4_artifacts", "type": "governs", "evidence": evidence_antigravity(artifacts_path, "_payloads_antigravity")},
        {"from": "task4_evaluation", "to": "task4_dashboard_export", "type": "governs", "evidence": evidence_antigravity(dashboard_export_path, "export_dashboard_antigravity")},
        {"from": "task4_artifacts", "to": "task4_replay", "type": "governs", "evidence": evidence_antigravity(replay_path, "inspect_replay_at_antigravity")},
        {"from": "task4_evaluation", "to": "task4_replay", "type": "governs", "evidence": evidence_antigravity(evaluation_path, "FrozenValidationAuthorityAntigravity")},
        {"from": "task4_dashboard_export", "to": "task4_replay", "type": "governs", "evidence": evidence_antigravity(dashboard_export_path, "export_dashboard_from_authority_antigravity")},
        {"from": "task2_blocked_folds", "to": "task5_frozen_authority", "type": "governs", "evidence": evidence_antigravity(frozen_authority_path, "_FOLD_PLAN_ANTIGRAVITY")},
        {"from": "task2_contracts", "to": "task5_frozen_authority", "type": "governs", "evidence": evidence_antigravity(frozen_authority_path, "_fold_evidence_antigravity")},
        {"from": "task4_evaluation", "to": "task5_frozen_authority", "type": "governs", "evidence": evidence_antigravity(frozen_authority_path, "FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY")},
        {"from": "task5_frozen_authority", "to": "task5_cli", "type": "governs", "evidence": evidence_antigravity(cli_path, "_load_frozen_validation_authority_antigravity")},
        {"from": "task3_qualification", "to": "task5_cli", "type": "governs", "evidence": evidence_antigravity(cli_path, "qualifier_antigravity")},
        {"from": "task4_artifacts", "to": "task5_cli", "type": "governs", "evidence": evidence_antigravity(cli_path, "publisher_antigravity")},
        {"from": "task4_replay", "to": "task5_cli", "type": "governs", "evidence": evidence_antigravity(cli_path, "inspector_antigravity")},
        {"from": "task5_cli", "to": "task5_launcher", "type": "governs", "evidence": evidence_antigravity(launcher_path, "_ensure_fixed_runtime_antigravity")},
        {"from": "task3_source_tests", "to": "task3_sources", "type": "tests", "evidence": evidence_antigravity(sources_test, "test_loader_rejects_malformed_train_rows_and_never_aliases_results_antigravity")},
        {"from": "task3_qualification_tests", "to": "task3_qualification", "type": "tests", "evidence": evidence_antigravity(qualification_test, "test_qualification_applies_fold_seed_and_inclusive_threshold_gates_antigravity")},
        {"from": "task4_evaluation_tests", "to": "task4_evaluation", "type": "tests", "evidence": evidence_antigravity(evaluation_test, "test_rejects_forged_or_mutated_qualification_at_evaluation_boundary_antigravity")},
        {"from": "task4_artifact_tests", "to": "task4_artifacts", "type": "tests", "evidence": evidence_antigravity(artifacts_test, "test_rejects_missing_or_wrong_authority_and_unsafe_or_forged_decision_antigravity")},
        {"from": "task4_replay_tests", "to": "task4_replay", "type": "tests", "evidence": evidence_antigravity(replay_test, "test_source_free_replay_recomputes_plan_evidence_decision_and_dashboard_antigravity")},
        {"from": "task4_dashboard_tests", "to": "task4_dashboard_export", "type": "tests", "evidence": evidence_antigravity(dashboard_export_test, "test_dashboard_rejects_forged_result_and_mutated_nested_fields_antigravity")},
        {"from": "task5_cli_tests", "to": "task5_cli", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_exact_surface_accepts_only_reviewed_evaluate_or_bare_inspect_antigravity")},
        {"from": "task5_cli_tests", "to": "task5_frozen_authority", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_version_controlled_frozen_authority_matches_exact_private_runs_antigravity")},
        {"from": "task5_cli_tests", "to": "task5_launcher", "type": "tests", "evidence": evidence_antigravity(cli_test, "test_launcher_private_synthetic_evaluate_then_inspect_from_outside_cwd_antigravity")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task1_documents_triplet_and_checkboxes_antigravity")},
        {"from": "governance_regression", "to": "runbook", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task5_authority_sequence_and_import_surface_are_unambiguous_antigravity")},
        {"from": "governance_regression", "to": "implementation_report", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task5_authority_sequence_and_import_surface_are_unambiguous_antigravity")},
        {"from": "governance_regression", "to": "dashboard", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_task5_authority_sequence_and_import_surface_are_unambiguous_antigravity")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_task5_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "experiment_catalog", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_task5_truth_and_non_promotion_antigravity")},
        {"from": "governance_regression", "to": "champion_ledger", "type": "tests", "evidence": evidence_antigravity(governance_test, "test_exp459_task5_truth_and_non_promotion_antigravity")},
    ]

    graph = {
        "schema": "exp459_task5_cli_authority_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp459_task5_cli_authority_codemap_lock_v1",
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
