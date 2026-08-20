#!/usr/bin/env python3
"""Build and validate the bounded Exp461 Task 4 qualification code map."""

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
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp461_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp461_antigravity.html",
    "docs/codemap/codemap_exp461_antigravity.json",
    "docs/codemap/codemap_exp461_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP461_DASHBOARD_antigravity.html",
    "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP461_PREREGISTRATION_antigravity.html",
    "docs/EXP461_RUNBOOK_antigravity.html",
    "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html",
    "docs/superpowers/specs/2026-08-20-exp461-ecg-period-signature-qualification-antigravity-design.html",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/config_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/contracts_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/artifacts_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/dashboard_export_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/evaluation_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/period_features_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/predecessor_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/qualification_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/replay_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/scoring_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/sources_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_artifacts_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_config_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_dashboard_export_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_evaluation_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_period_features_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_qualification_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_replay_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_scoring_antigravity.py",
    "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/test_sources_antigravity.py",
    "scratch/build_codemap_exp461_antigravity.py",
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
        '<title>Exp461 Task 4 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp461 Task 4 qualification 코드맵</h1><p>Task 4 implemented pending external review 상태는 Task 2 frozen input truth와 Task 3 scoring boundary를 유지한 채 qualification, decision, seven-leaf publication with in-row score evidence, source-free replay recomputation, restricted dashboard, and focused tests를 보존한다. Task 5 CLI, real TRAIN run, official execution은 아직 없다.</p>'
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

    design = "docs/superpowers/specs/2026-08-20-exp461-ecg-period-signature-qualification-antigravity-design.html"
    plan = "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html"
    preregistration = "docs/EXP461_PREREGISTRATION_antigravity.html"
    runbook = "docs/EXP461_RUNBOOK_antigravity.html"
    report = "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html"
    dashboard = "docs/EXP461_DASHBOARD_antigravity.html"
    governance_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    task2_config = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/config_antigravity.py"
    )
    task2_contracts = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/contracts_antigravity.py"
    )
    task2_predecessor = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/predecessor_antigravity.py"
    )
    task2_sources = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/sources_antigravity.py"
    )
    task3_period_features = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/period_features_antigravity.py"
    )
    task3_scoring = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/scoring_antigravity.py"
    )
    task4_qualification = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/qualification_antigravity.py"
    )
    task4_evaluation = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/evaluation_antigravity.py"
    )
    task4_artifacts = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/artifacts_antigravity.py"
    )
    task4_replay = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/replay_antigravity.py"
    )
    task4_dashboard = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/dashboard_export_antigravity.py"
    )
    task2_config_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_config_antigravity.py"
    )
    task2_contracts_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_contracts_antigravity.py"
    )
    task2_predecessor_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_predecessor_antigravity.py"
    )
    task2_sources_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_sources_antigravity.py"
    )
    task3_period_features_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_period_features_antigravity.py"
    )
    task3_scoring_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_scoring_antigravity.py"
    )
    task4_qualification_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_qualification_antigravity.py"
    )
    task4_evaluation_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_evaluation_antigravity.py"
    )
    task4_artifacts_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_artifacts_antigravity.py"
    )
    task4_replay_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_replay_antigravity.py"
    )
    task4_dashboard_test = (
        "experiments/exp461_ecg_period_signature_qualification_antigravity/tests/"
        "test_dashboard_export_antigravity.py"
    )
    builder = "scratch/build_codemap_exp461_antigravity.py"

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Task 1-6 governance and implementation. Effect: freezes the three-method comparison, TRAIN-only boundary, predecessor failed folds, and official-root absence rule.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp461 — ECG 주기 특징 성능 검증")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Task 1-6 execution. Effect: records the owned files, Task 2 closeout approval, checked Task 2 checkboxes, and later task boundaries.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 1 — 문서, 진행 대시보드, 코드맵")],
        },
        {
            "id": "preregistration",
            "path": preregistration,
            "role": "Caller: governance history. Effect: preserves planned-only experiment truth, official command count 0, label query count 0, and null result hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(preregistration, "planned_implementation_not_started")],
        },
        {
            "id": "runbook",
            "path": runbook,
            "role": "Caller: future operators. Effect: records the Task 1 operator boundary, exact root that must stay absent, and the commands that remain forbidden for now.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(runbook, "Task 1 operator boundary")],
        },
        {
            "id": "implementation_report",
            "path": report,
            "role": "Caller: Task 5 handoff. Effect: records the immutable document authority, Task 2 input boundary, Task 3 scoring boundary, and Task 4 qualification/publication/replay boundary pending external review.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(report, "Task 4 qualification report")],
        },
        {
            "id": "dashboard",
            "path": dashboard,
            "role": "Caller: always-visible status review. Effect: shows Task 4/6, planned official status, official execution 0, null official artifact hashes, the inherited period-change failures, Task 3 scoring entrypoints, and Task 4 qualification/replay entrypoints.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(dashboard, "Task 4/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records the Exp461 planned entry with zero official counts, null hashes, and unchanged pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp461\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp461 as planned-not-started and non-promoted with the frozen predecessor reference to Exp460.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"Exp461\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp461 planned history without changing champion, pending candidate, routing, or promotion.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp461\"")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: Task 3 governance verification. Effect: rejects missing scoring docs, wrong planned ledger truth, unchecked Task 3 boxes after review approval, checked later-task boxes, or invalid stored map bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_task1_governance_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 3 validation. Effect: rebuilds and validates the bounded Exp461 scoring graph plus the stored HTML/JSON/lock triplet bytes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
        {
            "id": "task2_config",
            "path": task2_config,
            "role": "Caller: predecessor capture, source validation, and future Task 3 scoring. Effect: freezes the exact Exp461 runtime, dataset, method order, thresholds, and predecessor hashes without opening labels or results.",
            "tests": [task2_config_test],
            "evidence": [evidence_antigravity(task2_config, "FROZEN_CONFIG_ANTIGRAVITY")],
        },
        {
            "id": "task2_contracts",
            "path": task2_contracts,
            "role": "Caller: predecessor capture and future replay consumers. Effect: enforces the exact Exp460 fallback snapshot, immutable failed-fold rows, strict baseline routing, and six-leaf hash order.",
            "tests": [task2_contracts_test],
            "evidence": [
                evidence_antigravity(
                    task2_contracts,
                    "validate_exp460_period_authority_antigravity",
                )
            ],
        },
        {
            "id": "task2_predecessor_capture",
            "path": task2_predecessor,
            "role": "Caller: future Task 3 scoring and qualification setup. Effect: replays the exact two failed Exp459 period-change folds into the frozen Exp460 fallback authority for Exp461.",
            "tests": [task2_predecessor_test],
            "evidence": [
                evidence_antigravity(
                    task2_predecessor,
                    "capture_exp460_period_authority_antigravity",
                )
            ],
        },
        {
            "id": "task2_sources",
            "path": task2_sources,
            "role": "Caller: future Task 3 feature extraction. Effect: replays the exact owned, read-only, C-contiguous float32 normal_2 TRAIN matrix from the established Exp459 loader with no new data path.",
            "tests": [task2_sources_test],
            "evidence": [evidence_antigravity(task2_sources, "load_normal2_train_antigravity")],
        },
        {
            "id": "task2_config_tests",
            "path": task2_config_test,
            "role": "Caller: Task 2 RED/GREEN verification. Effect: rejects exact-value, type, or order drift in the frozen Exp461 config surface.",
            "tests": [task2_config_test],
            "evidence": [
                evidence_antigravity(
                    task2_config_test,
                    "test_frozen_config_preserves_the_exact_task2_contract_antigravity",
                )
            ],
        },
        {
            "id": "task2_contracts_tests",
            "path": task2_contracts_test,
            "role": "Caller: Task 2 RED/GREEN verification. Effect: rejects mutable containers, bool-as-int aliases, wrong strict-route semantics, or any Exp460 snapshot drift.",
            "tests": [task2_contracts_test],
            "evidence": [
                evidence_antigravity(
                    task2_contracts_test,
                    "test_validator_rejects_snapshot_drift_and_mutable_containers_antigravity",
                )
            ],
        },
        {
            "id": "task2_predecessor_tests",
            "path": task2_predecessor_test,
            "role": "Caller: Task 2 RED/GREEN verification. Effect: rejects wrong Exp460 hashes, wrong failed-fold rows, or source drift away from the frozen Exp459 authority.",
            "tests": [task2_predecessor_test],
            "evidence": [
                evidence_antigravity(
                    task2_predecessor_test,
                    "test_capture_returns_the_exact_frozen_exp460_authority_antigravity",
                )
            ],
        },
        {
            "id": "task2_sources_tests",
            "path": task2_sources_test,
            "role": "Caller: Task 2 RED/GREEN verification. Effect: rejects TRAIN-shape drift, aliasing drift, or any new TEST or label-capable source path.",
            "tests": [task2_sources_test],
            "evidence": [
                evidence_antigravity(
                    task2_sources_test,
                    "test_loader_replays_the_exact_exp459_train_matrix_without_aliasing_antigravity",
                )
            ],
        },
        {
            "id": "task3_period_features",
            "path": task3_period_features,
            "role": "Caller: Task 3 scoring and future Task 4 qualification. Effect: converts each finite 140-wide ECG row into an owned immutable float64 120D period signature using full/front/back views, 16 Hann-rFFT bands, and 24 autocorrelation lags.",
            "tests": [task3_period_features_test],
            "evidence": [
                evidence_antigravity(
                    task3_period_features,
                    "period_signature_antigravity",
                )
            ],
        },
        {
            "id": "task3_scoring",
            "path": task3_scoring,
            "role": "Caller: future Task 4 qualification. Effect: exposes exact method-order fold scores for Exp459 Hydra LocalGap, period-signature LocalGap, and max branch-rank hybrid without creating result roots.",
            "tests": [task3_scoring_test],
            "evidence": [evidence_antigravity(task3_scoring, "score_fold_methods_antigravity")],
        },
        {
            "id": "task3_period_features_tests",
            "path": task3_period_features_test,
            "role": "Caller: Task 3 RED/GREEN verification. Effect: rejects period feature shape/order, NumPy reference, invariance, sensitivity, invalid row, or mutability drift.",
            "tests": [task3_period_features_test],
            "evidence": [
                evidence_antigravity(
                    task3_period_features_test,
                    "test_signature_matches_independent_three_view_numpy_reference_antigravity",
                )
            ],
        },
        {
            "id": "task3_scoring_tests",
            "path": task3_scoring_test,
            "role": "Caller: Task 3 RED/GREEN verification. Effect: rejects rank formula drift, method-order drift, dependency-flow drift, source access creep, or forged fold-score containers.",
            "tests": [task3_scoring_test],
            "evidence": [
                evidence_antigravity(
                    task3_scoring_test,
                    "test_fold_scoring_method_order_rank_fusion_and_dependency_flow_antigravity",
                )
            ],
        },
        {
            "id": "task4_qualification",
            "path": task4_qualification,
            "role": "Caller: Task 4 evaluation/publication. Effect: converts Task 3 fold scores into exact 30 method-seed-fold rows with retained score evidence, recomputed AUCs, Wilson upper bounds, finite flags, pass/fail reasons, and method summaries without opening labels or result roots.",
            "tests": [task4_qualification_test],
            "evidence": [evidence_antigravity(task4_qualification, "qualify_methods_antigravity")],
        },
        {
            "id": "task4_evaluation",
            "path": task4_evaluation,
            "role": "Caller: Task 4 publication/dashboard/replay. Effect: recomputes the frozen primary candidate gate, the two historical strict-improvement checks, and the non-routing decision.",
            "tests": [task4_evaluation_test],
            "evidence": [evidence_antigravity(task4_evaluation, "evaluate_qualification_antigravity")],
        },
        {
            "id": "task4_artifacts",
            "path": task4_artifacts,
            "role": "Caller: Task 4 temporary publication and replay tests. Effect: normalizes metric rows from score evidence, then writes the exact seven canonical leaves under a create-only temporary root with a self-excluding artifact hash map.",
            "tests": [task4_artifacts_test, task4_replay_test],
            "evidence": [evidence_antigravity(task4_artifacts, "publish_result_antigravity")],
        },
        {
            "id": "task4_replay",
            "path": task4_replay,
            "role": "Caller: Task 4 inspect/replay boundary. Effect: validates inventory, canonical bytes, frozen context, 30-row order, in-row score evidence, recomputed metrics, summaries, decision, dashboard, and hashes without source data or labels.",
            "tests": [task4_replay_test],
            "evidence": [evidence_antigravity(task4_replay, "inspect_replay_at_antigravity")],
        },
        {
            "id": "task4_dashboard_export",
            "path": task4_dashboard,
            "role": "Caller: Task 4 publication and visible dashboard projection. Effect: exposes aggregate method/fold qualification only, with no raw series, fold indices, TEST labels, paths, or secrets.",
            "tests": [task4_dashboard_test],
            "evidence": [evidence_antigravity(task4_dashboard, "export_dashboard_antigravity")],
        },
        {
            "id": "task4_qualification_tests",
            "path": task4_qualification_test,
            "role": "Caller: Task 4 RED/GREEN verification. Effect: rejects row-order drift, missing method/fold rows, primary gate drift, or non-strict historical improvement acceptance.",
            "tests": [task4_qualification_test],
            "evidence": [
                evidence_antigravity(
                    task4_qualification_test,
                    "test_qualification_preserves_exact_method_seed_fold_and_type_order_antigravity",
                )
            ],
        },
        {
            "id": "task4_evaluation_tests",
            "path": task4_evaluation_test,
            "role": "Caller: Task 4 RED/GREEN verification. Effect: rejects summary forgery, primary fold failures, routing application, or TEST-label access at the decision boundary.",
            "tests": [task4_evaluation_test],
            "evidence": [
                evidence_antigravity(
                    task4_evaluation_test,
                    "test_decision_recomputes_primary_summary_without_routing_or_labels_antigravity",
                )
            ],
        },
        {
            "id": "task4_artifacts_tests",
            "path": task4_artifacts_test,
            "role": "Caller: Task 4 RED/GREEN verification. Effect: rejects publication leaf-count, hash-map, canonical inventory, or create-only overwrite drift.",
            "tests": [task4_artifacts_test],
            "evidence": [
                evidence_antigravity(
                    task4_artifacts_test,
                    "test_publish_writes_exact_seven_leaf_create_only_canonical_inventory_antigravity",
                )
            ],
        },
        {
            "id": "task4_replay_tests",
            "path": task4_replay_test,
            "role": "Caller: Task 4 RED/GREEN verification. Effect: rejects replay that trusts stored metric rows, summaries, decisions, dashboards, hash maps, or rehashed tamper.",
            "tests": [task4_replay_test],
            "evidence": [
                evidence_antigravity(
                    task4_replay_test,
                    "test_inspect_recomputes_qualification_decision_dashboard_and_hashes_antigravity",
                )
            ],
        },
        {
            "id": "task4_dashboard_export_tests",
            "path": task4_dashboard_test,
            "role": "Caller: Task 4 RED/GREEN verification. Effect: rejects dashboard payloads that expose raw series, fold indices, local paths, TEST labels, or secrets.",
            "tests": [task4_dashboard_test],
            "evidence": [
                evidence_antigravity(
                    task4_dashboard_test,
                    "test_dashboard_payload_is_aggregate_only_and_label_free_antigravity",
                )
            ],
        },
    ]

    edges = [
        {
            "from": "approved_design",
            "to": "preregistration",
            "type": "governs",
            "evidence": evidence_antigravity(
                preregistration,
                "frozen predecessor truth",
            ),
        },
        {
            "from": "approved_design",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity(dashboard, "hydra_period_rank_max"),
        },
        {
            "from": "implementation_plan",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity(dashboard, "공식 실행 0회"),
        },
        {
            "from": "implementation_plan",
            "to": "governance_regression",
            "type": "governs",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "preregistration",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "runbook",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "implementation_report",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "dashboard",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "research_state",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "experiment_catalog",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "champion_ledger",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_governance_antigravity"),
        },
        {
            "from": "governance_regression",
            "to": "map_builder",
            "type": "tests",
            "evidence": evidence_antigravity(governance_test, "test_task1_dashboard_plan_and_map_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task2_config",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "Task 2 — 고정 입력, 자료형, TRAIN source"),
        },
        {
            "from": "implementation_plan",
            "to": "task2_contracts",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "capture_exp460_period_authority_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task2_predecessor_capture",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "Exp460 six-leaf official root"),
        },
        {
            "from": "implementation_plan",
            "to": "task2_sources",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "load_normal2_train_antigravity()"),
        },
        {
            "from": "implementation_plan",
            "to": "task3_period_features",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "120차원 주기 특징과 세 점수"),
        },
        {
            "from": "implementation_plan",
            "to": "task3_scoring",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "score_fold_methods_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task4_qualification",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "qualify_methods_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task4_evaluation",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "evaluate_qualification_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task4_artifacts",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "publish_result_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task4_replay",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "inspect_replay_at_antigravity"),
        },
        {
            "from": "implementation_plan",
            "to": "task4_dashboard_export",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "export_dashboard_antigravity"),
        },
        {
            "from": "task2_config",
            "to": "task3_scoring",
            "type": "governs",
            "evidence": evidence_antigravity(task2_config, "method_order"),
        },
        {
            "from": "task2_sources",
            "to": "task3_scoring",
            "type": "governs",
            "evidence": evidence_antigravity(task2_sources, "load_normal2_train_antigravity"),
        },
        {
            "from": "task3_scoring",
            "to": "task3_period_features",
            "type": "governs",
            "evidence": evidence_antigravity(task3_scoring, "_period_feature_matrix_antigravity"),
        },
        {
            "from": "task3_scoring",
            "to": "task4_qualification",
            "type": "governs",
            "evidence": evidence_antigravity(task4_qualification, "_qualify_methods_with_dependencies_antigravity"),
        },
        {
            "from": "task2_predecessor_capture",
            "to": "task4_qualification",
            "type": "governs",
            "evidence": evidence_antigravity(task4_qualification, "_metric_row_antigravity"),
        },
        {
            "from": "task4_qualification",
            "to": "task4_evaluation",
            "type": "governs",
            "evidence": evidence_antigravity(task4_evaluation, "evaluate_qualification_antigravity"),
        },
        {
            "from": "task4_evaluation",
            "to": "task4_artifacts",
            "type": "governs",
            "evidence": evidence_antigravity(task4_artifacts, "_payloads_antigravity"),
        },
        {
            "from": "task4_evaluation",
            "to": "task4_dashboard_export",
            "type": "governs",
            "evidence": evidence_antigravity(task4_dashboard, "_summary_payloads_antigravity"),
        },
        {
            "from": "task4_artifacts",
            "to": "task4_replay",
            "type": "governs",
            "evidence": evidence_antigravity(task4_replay, "inspect_replay_at_antigravity"),
        },
        {
            "from": "task2_config_tests",
            "to": "task2_config",
            "type": "tests",
            "evidence": evidence_antigravity(
                task2_config_test,
                "test_frozen_config_preserves_the_exact_task2_contract_antigravity",
            ),
        },
        {
            "from": "task2_contracts_tests",
            "to": "task2_contracts",
            "type": "tests",
            "evidence": evidence_antigravity(
                task2_contracts_test,
                "test_validator_rejects_snapshot_drift_and_mutable_containers_antigravity",
            ),
        },
        {
            "from": "task2_predecessor_tests",
            "to": "task2_predecessor_capture",
            "type": "tests",
            "evidence": evidence_antigravity(
                task2_predecessor_test,
                "test_capture_returns_the_exact_frozen_exp460_authority_antigravity",
            ),
        },
        {
            "from": "task2_sources_tests",
            "to": "task2_sources",
            "type": "tests",
            "evidence": evidence_antigravity(
                task2_sources_test,
                "test_loader_replays_the_exact_exp459_train_matrix_without_aliasing_antigravity",
            ),
        },
        {
            "from": "task3_period_features_tests",
            "to": "task3_period_features",
            "type": "tests",
            "evidence": evidence_antigravity(
                task3_period_features_test,
                "test_signature_matches_independent_three_view_numpy_reference_antigravity",
            ),
        },
        {
            "from": "task3_scoring_tests",
            "to": "task3_scoring",
            "type": "tests",
            "evidence": evidence_antigravity(
                task3_scoring_test,
                "test_fold_scoring_method_order_rank_fusion_and_dependency_flow_antigravity",
            ),
        },
        {
            "from": "task4_qualification_tests",
            "to": "task4_qualification",
            "type": "tests",
            "evidence": evidence_antigravity(
                task4_qualification_test,
                "test_qualification_preserves_exact_method_seed_fold_and_type_order_antigravity",
            ),
        },
        {
            "from": "task4_evaluation_tests",
            "to": "task4_evaluation",
            "type": "tests",
            "evidence": evidence_antigravity(
                task4_evaluation_test,
                "test_decision_recomputes_primary_summary_without_routing_or_labels_antigravity",
            ),
        },
        {
            "from": "task4_artifacts_tests",
            "to": "task4_artifacts",
            "type": "tests",
            "evidence": evidence_antigravity(
                task4_artifacts_test,
                "test_publish_writes_exact_seven_leaf_create_only_canonical_inventory_antigravity",
            ),
        },
        {
            "from": "task4_replay_tests",
            "to": "task4_replay",
            "type": "tests",
            "evidence": evidence_antigravity(
                task4_replay_test,
                "test_inspect_recomputes_qualification_decision_dashboard_and_hashes_antigravity",
            ),
        },
        {
            "from": "task4_dashboard_export_tests",
            "to": "task4_dashboard_export",
            "type": "tests",
            "evidence": evidence_antigravity(
                task4_dashboard_test,
                "test_dashboard_payload_is_aggregate_only_and_label_free_antigravity",
            ),
        },
    ]

    graph = {
        "schema": "exp461_task4_qualification_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
    }
    lock = {
        "schema": "exp461_task4_qualification_codemap_lock_v1",
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
            raise ValueError("unresolved edge evidence")

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
    parser.add_argument("--generated-at", default="2026-08-20T00:00:00Z")
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
