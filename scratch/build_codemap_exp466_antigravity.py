#!/usr/bin/env python3
"""Build and validate the bounded Exp466 Task 1-6 code-map triplet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html as html_module
import html.parser
import json
import re
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
HTML_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp466_antigravity.html"
JSON_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp466_antigravity.json"
LOCK_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp466_antigravity.lock"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp466_antigravity.html",
    "docs/codemap/codemap_exp466_antigravity.json",
    "docs/codemap/codemap_exp466_antigravity.lock",
)
CONFIG_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "config_antigravity.py"
)
CONTRACTS_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "contracts_antigravity.py"
)
CONFIG_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_config_antigravity.py"
)
CONTRACTS_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_contracts_antigravity.py"
)
SOURCES_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "sources_antigravity.py"
)
INTERPOLATION_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "interpolation_antigravity.py"
)
SYNTHETIC_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "synthetic_antigravity.py"
)
PREDECESSOR_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "predecessor_antigravity.py"
)
SOURCES_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_sources_antigravity.py"
)
INTERPOLATION_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_interpolation_antigravity.py"
)
SYNTHETIC_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_synthetic_antigravity.py"
)
PREDECESSOR_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_predecessor_antigravity.py"
)
PHASE_REGISTRATION_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "phase_registration_antigravity.py"
)
PHASE_REGISTRATION_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_phase_registration_antigravity.py"
)
SCORING_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "scoring_antigravity.py"
)
QUALIFICATION_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "qualification_antigravity.py"
)
EVALUATION_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "evaluation_antigravity.py"
)
SCORING_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_scoring_antigravity.py"
)
QUALIFICATION_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_qualification_antigravity.py"
)
EVALUATION_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_evaluation_antigravity.py"
)
BUILDER_ANTIGRAVITY = "scratch/build_codemap_exp466_antigravity.py"
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity-design.html"
)
BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-1-brief-antigravity.md"
)
TASK_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-1-report-antigravity.html"
)
TASK2_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-2-report-antigravity.html"
)
TASK3_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-3-report-antigravity.html"
)
TASK4_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-4-report-antigravity.html"
)
ARTIFACTS_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "artifacts_antigravity.py"
)
REPLAY_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "replay_antigravity.py"
)
DASHBOARD_EXPORT_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "dashboard_export_antigravity.py"
)
ARTIFACTS_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_artifacts_antigravity.py"
)
REPLAY_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_replay_antigravity.py"
)
DASHBOARD_EXPORT_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_dashboard_export_antigravity.py"
)
TASK5_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp466-ecg200-phase-registration-margin-antigravity.html/"
    "task-5-report-antigravity.html"
)
CLI_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "cli_antigravity.py"
)
CLI_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_cli_antigravity.py"
)
GOVERNANCE_TEST_ANTIGRAVITY = (
    "experiments/exp466_ecg200_phase_registration_margin_antigravity/"
    "tests/test_governance_antigravity.py"
)
LAUNCHER_ANTIGRAVITY = (
    "scratch/run_exp466_ecg200_phase_registration_margin_antigravity.py"
)
TASK6_SOURCE_BASE_COMMIT_ANTIGRAVITY = (
    "285b33f6307293c65087d4625284dd397d932a9f"
)
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP466_DASHBOARD_antigravity.html",
    "docs/EXP466_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP466_PREREGISTRATION_antigravity.html",
    "docs/EXP466_RUNBOOK_antigravity.html",
    DESIGN_ANTIGRAVITY,
    BRIEF_ANTIGRAVITY,
    TASK_REPORT_ANTIGRAVITY,
    CONFIG_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    SOURCES_ANTIGRAVITY,
    INTERPOLATION_ANTIGRAVITY,
    SYNTHETIC_ANTIGRAVITY,
    PREDECESSOR_ANTIGRAVITY,
    SOURCES_TEST_ANTIGRAVITY,
    INTERPOLATION_TEST_ANTIGRAVITY,
    SYNTHETIC_TEST_ANTIGRAVITY,
    PREDECESSOR_TEST_ANTIGRAVITY,
    TASK2_REPORT_ANTIGRAVITY,
    PHASE_REGISTRATION_ANTIGRAVITY,
    PHASE_REGISTRATION_TEST_ANTIGRAVITY,
    TASK3_REPORT_ANTIGRAVITY,
    SCORING_ANTIGRAVITY,
    QUALIFICATION_ANTIGRAVITY,
    EVALUATION_ANTIGRAVITY,
    SCORING_TEST_ANTIGRAVITY,
    QUALIFICATION_TEST_ANTIGRAVITY,
    EVALUATION_TEST_ANTIGRAVITY,
    TASK4_REPORT_ANTIGRAVITY,
    ARTIFACTS_ANTIGRAVITY,
    REPLAY_ANTIGRAVITY,
    DASHBOARD_EXPORT_ANTIGRAVITY,
    ARTIFACTS_TEST_ANTIGRAVITY,
    REPLAY_TEST_ANTIGRAVITY,
    DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
    TASK5_REPORT_ANTIGRAVITY,
    CLI_ANTIGRAVITY,
    LAUNCHER_ANTIGRAVITY,
    CLI_TEST_ANTIGRAVITY,
    GOVERNANCE_TEST_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
    "docs/codemap/codemap_exp465_antigravity.html",
    "docs/codemap/codemap_exp465_antigravity.json",
    "docs/codemap/codemap_exp465_antigravity.lock",
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp466_task6_private_failure_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp466_task6_private_failure_codemap_lock_v1"
GENERATED_AT_ANTIGRAVITY = "2026-08-23T00:00:00+09:00"
_VOID_TAGS_ANTIGRAVITY = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}


def _canonical_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class _StrictHtmlAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.parts: list[str] = []

    def handle_decl(self, decl: str) -> None:
        self.parts.append(decl)

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _VOID_TAGS_ANTIGRAVITY:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            raise ValueError(f"misnested HTML closing tag: {tag}")
        self.stack.pop()


def _validated_html_text_antigravity(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().lower().startswith("<!doctype html>"):
        raise ValueError(f"missing HTML5 doctype: {path}")
    parser = _StrictHtmlAntigravity()
    parser.feed(text)
    parser.close()
    if parser.stack:
        raise ValueError(f"unclosed HTML tags in {path}: {parser.stack}")
    return " ".join(parser.parts)


def _python_symbols_antigravity(text: str) -> set[str]:
    tree = ast.parse(text)
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".", 1)[0])
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
    if target.suffix == ".html":
        return symbol in _validated_html_text_antigravity(target)
    text = target.read_text(encoding="utf-8")
    if target.suffix == ".py":
        return symbol in _python_symbols_antigravity(text)
    return symbol in text


def _node_antigravity(
    node_id: str,
    path: str,
    role: str,
    symbol: str,
    tests: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "tests": list(tests),
        "evidence": [{"path": path, "symbol": symbol}],
    }


def _edge_antigravity(
    source: str,
    relation: str,
    target: str,
    evidence_path: str,
    evidence_symbol: str,
) -> dict[str, Any]:
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


def build_graph_antigravity(source_parent: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_parent) is None:
        raise ValueError("source_parent must be a lowercase 40-hex commit")
    if source_parent != TASK6_SOURCE_BASE_COMMIT_ANTIGRAVITY:
        raise ValueError("source_parent must be the accepted Task 6 source base")
    map_test = (BUILDER_ANTIGRAVITY,)
    config_test = (CONFIG_TEST_ANTIGRAVITY,)
    contracts_test = (CONTRACTS_TEST_ANTIGRAVITY,)
    sources_test = (SOURCES_TEST_ANTIGRAVITY,)
    interpolation_test = (INTERPOLATION_TEST_ANTIGRAVITY,)
    synthetic_test = (SYNTHETIC_TEST_ANTIGRAVITY,)
    predecessor_test = (PREDECESSOR_TEST_ANTIGRAVITY,)
    phase_registration_test = (PHASE_REGISTRATION_TEST_ANTIGRAVITY,)
    scoring_test = (SCORING_TEST_ANTIGRAVITY,)
    qualification_test = (QUALIFICATION_TEST_ANTIGRAVITY,)
    evaluation_test = (EVALUATION_TEST_ANTIGRAVITY,)
    artifacts_test = (ARTIFACTS_TEST_ANTIGRAVITY,)
    replay_test = (REPLAY_TEST_ANTIGRAVITY,)
    dashboard_export_test = (DASHBOARD_EXPORT_TEST_ANTIGRAVITY,)
    cli_test = (CLI_TEST_ANTIGRAVITY, GOVERNANCE_TEST_ANTIGRAVITY)
    launcher_test = (CLI_TEST_ANTIGRAVITY, GOVERNANCE_TEST_ANTIGRAVITY)
    nodes = [
        _node_antigravity(
            "task1_brief",
            BRIEF_ANTIGRAVITY,
            "Caller: Task 1 implementer and closeout controller. Effect: bounds work to config/contracts, two tests, three truth HTML files, additive planned ledgers, dashboard fields, map triplet, report, verification, and commit; forbids Tasks 2–6 and every data/evaluation path.",
            "Task 1 — 권한·계약·문서 골격",
            map_test,
        ),
        _node_antigravity(
            "approved_design",
            DESIGN_ANTIGRAVITY,
            "Caller: frozen Exp466 authority. Effect: fixes TRAIN normal69, 96-to-140 interpolation, seeds/folds, onset56/suffix84, warp1.25/shared67, C0/P9/C9 roles, C9 gates, zero normal confirmation, and no TEST/routing/promotion.",
            "Exp466 — ECG200 고정 위상 정렬 마진 자격 검증",
            map_test,
        ),
        _node_antigravity(
            "exp465_baseline",
            "docs/codemap/codemap_exp465_antigravity.json",
            "Caller: Exp466 Task 1 structural comparison. Effect: supplies the prior caller/effect/test graph strength without modifying Exp465; its HTML/JSON/lock fingerprints remain represented by the Exp466 lock.",
            '"task1_config"',
            map_test,
        ),
        _node_antigravity(
            "task1_config",
            CONFIG_ANTIGRAVITY,
            "Caller: Exp466 contracts/tests and future source/scoring tasks. Effect: exact-type freezes ECG200 TRAIN identity, 69 normals, 96-to-140 geometry, seeds/folds/onset/suffix, warp/shared length, C0/P9/C9 order, C9-only primary, gates, zero-use counts, and approved design digest.",
            "FROZEN_CONFIG_ANTIGRAVITY",
            config_test,
        ),
        _node_antigravity(
            "task1_contracts",
            CONTRACTS_ANTIGRAVITY,
            "Caller: future phase scoring, qualification, replay, and publication tasks. Effect: supplies immutable phase evidence, fold result, and qualification containers; rejects mutable/reordered mappings, nonfinite margin, wrong 69 count, wrong SHA, nonzero official/label counts, and non-null pre-run artifact hash.",
            "Exp466QualificationAntigravity",
            contracts_test,
        ),
        _node_antigravity(
            "task1_config_test",
            CONFIG_TEST_ANTIGRAVITY,
            "Caller: focused Task 1 RED/GREEN suite. Effect: detects identity, geometry, gate, zero-use, immutability, exact-type, and design SHA drift.",
            "Exp466ConfigTestAntigravity",
            config_test,
        ),
        _node_antigravity(
            "task1_contracts_test",
            CONTRACTS_TEST_ANTIGRAVITY,
            "Caller: focused Task 1 RED/GREEN suite. Effect: detects nonfinite/forged phase evidence, invalid fold gates, mutable/reordered seed maps, wrong normal69 aggregation, nonzero official/label counts, and result hashes before execution.",
            "Exp466ContractsTestAntigravity",
            contracts_test,
        ),
        _node_antigravity(
            "task2_sources",
            SOURCES_ANTIGRAVITY,
            "Caller: future Exp466 phase scoring input assembly and qualification replay. Effect: exports the exact Exp465 ECG200 TRAIN-normal loader function object, preserving TEST-name rejection before open and the frozen owned read-only normal69 boundary without a copied parser.",
            "load_ecg200_normal_train_antigravity",
            sources_test,
        ),
        _node_antigravity(
            "task2_interpolation",
            INTERPOLATION_ANTIGRAVITY,
            "Caller: future Exp466 phase scoring input assembly and qualification replay. Effect: exports the exact Exp465 96-to-140 interpolation function object, preserving its fixed grids, validation, and read-only float64 output without a wrapper or new resampling math.",
            "interpolate_to_140_antigravity",
            interpolation_test,
        ),
        _node_antigravity(
            "task2_synthetic",
            SYNTHETIC_ANTIGRAVITY,
            "Caller: future Exp466 phase scoring and qualification synthetic-query construction. Effect: creates only level_shift, gradual_drift, amplitude_scaling, and period_change in frozen order with Exp464 formulas, onset56, warp1.25, and owned finite C-contiguous read-only float64 outputs.",
            "primary_rows_antigravity",
            synthetic_test,
        ),
        _node_antigravity(
            "task2_predecessor",
            PREDECESSOR_ANTIGRAVITY,
            "Caller: future Exp466 phase scoring and qualification C0 control path. Effect: pins reviewed Exp465 closeout 23528af929da9dbf57cbcdf87c06abd58cb2363b and the exact Exp465 fold scorer, then projects only immutable byte-equal C0 normal and four primary score vectors.",
            "score_c0_fold_antigravity",
            predecessor_test,
        ),
        _node_antigravity(
            "task2_sources_test",
            SOURCES_TEST_ANTIGRAVITY,
            "Caller: focused Task 2 RED/GREEN and regression verification. Effect: detects copied/wrapped loader exports and proves a TEST-named path is rejected before Path.open is called.",
            "SourcesAntigravityTest",
            sources_test,
        ),
        _node_antigravity(
            "task2_interpolation_test",
            INTERPOLATION_TEST_ANTIGRAVITY,
            "Caller: focused Task 2 RED/GREEN and regression verification. Effect: detects any copied or wrapped interpolation export instead of the exact Exp465 function object.",
            "InterpolationAntigravityTest",
            interpolation_test,
        ),
        _node_antigravity(
            "task2_synthetic_test",
            SYNTHETIC_TEST_ANTIGRAVITY,
            "Caller: focused Task 2 RED/GREEN and regression verification. Effect: detects formula, key-order, onset/warp, dtype, finiteness, ownership, immutability, aliasing, and input-validation drift across all four synthetic changes.",
            "SyntheticAntigravityTest",
            synthetic_test,
        ),
        _node_antigravity(
            "task2_predecessor_test",
            PREDECESSOR_TEST_ANTIGRAVITY,
            "Caller: focused Task 2 RED/GREEN and regression verification. Effect: detects Exp465 closeout/source drift, signature drift, mutable projections, and any byte difference in C0 normal or primary scores.",
            "PredecessorAntigravityTest",
            predecessor_test,
        ),
        _node_antigravity(
            "task3_phase_registration",
            PHASE_REGISTRATION_ANTIGRAVITY,
            "Caller: focused Task 3 tests and future separately authorized Exp466 C9/qualification stages. Effect: standardizes each onset56/suffix84 independently with median and 1.4826×MAD, computes the fixed warp1.25/shared67 identity-minus-undo margin, calibrates fit with leave-one-out medians and query with the full-fit median, reuses the exact frozen anomaly rank, applies strict positive fit-max confirmation, and returns owned read-only finite outputs without MLX, data, private, or official execution.",
            "score_phase_registration_antigravity",
            phase_registration_test,
        ),
        _node_antigravity(
            "task3_phase_registration_test",
            PHASE_REGISTRATION_TEST_ANTIGRAVITY,
            "Caller: focused Task 3 RED/GREEN, mutation review, and full Exp466 regression. Effect: detects fixed-formula, affine-invariance, shared67, LOO/full-fit calibration, exact-rank, strict-confirmation, geometry/dtype/finite/MAD, ownership/immutability, and exact-zero-above-negative-fit-max drift across six tests.",
            "PhaseRegistrationAntigravityTest",
            phase_registration_test,
        ),
        _node_antigravity(
            "task3_report",
            TASK3_REPORT_ANTIGRAVITY,
            "Caller: Task 3 implementation, Terra review, and map closeout audit. Effect: consolidates original RED, two focused GREEN runs, full regression, mathematical boundaries, exact CPU/MLX manifest and CPU fallback, zero-margin mutation disposition, accepted source commit, Terra final PASS with no findings, map/JSON/fingerprint/diff evidence, preserved boundaries, and concerns.",
            "Exp466 Task 3",
            map_test,
        ),
        _node_antigravity(
            "preregistration",
            "docs/EXP466_PREREGISTRATION_antigravity.html",
            "Caller: governance and scientific reviewers. Effect: states planned_implementation_not_started, freezes C0/P9/C9 and C9 gates, records zero/null truth, CPU/MLX pre-run gate, and forbidden scope.",
            "planned_implementation_not_started",
            map_test,
        ),
        _node_antigravity(
            "runbook",
            "docs/EXP466_RUNBOOK_antigravity.html",
            "Caller: Task 6 Step 4-6 operators and reviewers. Effect: preserves the fixed run1/run2 commands as a no-rerun execution record, records private run/evaluate/inspect 2/2/2, four byte-equal stdout payloads, two byte-equal seven-leaf roots, final sequence stage run2_inspected, C9 0-of-10 deterministic failure, exact measured causes, obsolete embedded Task 5 provenance warning, zero official/label activity, and Step 6 as the only next action.",
            "Task 1 focused verification",
            map_test,
        ),
        _node_antigravity(
            "implementation_report",
            "docs/EXP466_IMPLEMENTATION_REPORT_antigravity.html",
            "Caller: Task 1-6 implementation audit. Effect: records Task 6 Step 1-5 complete, private run/evaluate/inspect 2/2/2, stdout and seven-leaf byte equality, C9 0-of-10 failure and exact causes, official/label zero/null boundaries, preserved-artifact provenance caveat, and Step 6 final verification/review pending.",
            "TDD 증거",
            map_test,
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP466_DASHBOARD_antigravity.html",
            "Caller: user-facing progress review. Effect: marks private run1 and run2/reproducibility complete, shows C9 0-of-10 and exact failure ranges in easy Korean, records private 2/2/2 with byte-equal stdout/leaves and private hashes, keeps official fields null and external effects false, warns that preserved Task 5/actual=false/synthetic wording is invalid Task 6 provenance, and leaves Step 6 final review unchecked.",
            "qualification_failed_blocked_before_fn_validation",
            map_test,
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: research-state audit. Effect: updates only the Exp466 block to the Step 5 private qualification failure, stores private roots/stdout/leaves/sequence hashes and measured causes, keeps official generic hashes null and TEST/label/routing/promotion/pending/champion effects zero or false, and points only to Step 6 final verification/review.",
            '"exp466"',
            map_test,
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment inventory. Effect: records Exp466 private run/evaluate/inspect 2/2/2, reproducible C9 0-of-10 failure evidence and private hashes while preserving zero official/label use, null official hashes, and no routing/promotion/pending/champion change.",
            '"id": "Exp466"',
            map_test,
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion-history audit. Effect: records the byte-reproduced Exp466 C9 0-of-10 private qualification failure while leaving champion and pending candidate unchanged, disallowing routing/promotion, and preserving zero official/label activity with null official hashes.",
            '"experiment": "Exp466"',
            map_test,
        ),
        _node_antigravity(
            "task1_report",
            TASK_REPORT_ANTIGRAVITY,
            "Caller: controller handoff. Effect: consolidates status, files/interfaces, exact RED and both GREEN evidence, diff/map/HTML/JSON checks, commit hash, caller/effect/test answers, preserved boundaries, and concerns.",
            "Exp466 Task 1 구현 보고",
            map_test,
        ),
        _node_antigravity(
            "task2_report",
            TASK2_REPORT_ANTIGRAVITY,
            "Caller: Task 2 implementation and map closeout audit. Effect: consolidates exact RED, two GREEN runs, C0 byte parity, TEST-before-open evidence, implementation commit, Luna SPEC/QUALITY PASS, map/diff verification, preserved boundaries, and concerns.",
            "Exp466 Task 2",
            map_test,
        ),
        _node_antigravity(
            "map_builder",
            BUILDER_ANTIGRAVITY,
            "Caller: Task 1-6 closeout and preflight verification. Effect: generates canonical HTML/JSON/lock or validates evidence, HTML structure, Task 1-6 graph relations, embedded JSON, Exp465 baseline fingerprints, and all tracked fingerprints without writing in validate-only mode.",
            "build_triplet_antigravity",
            map_test,
        ),
        _node_antigravity(
            "task4_scoring",
            SCORING_ANTIGRAVITY,
            "Caller: focused Task 4 scoring tests, exact ten-record qualification, and future explicitly authorized publication. Effect: calls only the reviewed Task 2 C0 authority, Task 2 four-change generator, and Task 3 phase scorer; preserves C0/P9/C9 order; applies C9=max(C0,P9) only on strict confirmation; and owns/freezes every retained input, score, margin, confirmation, index, and mapping without data, private, official, MLX, or governance side effects.",
            "score_fold_methods_antigravity",
            scoring_test,
        ),
        _node_antigravity(
            "task4_qualification",
            QUALIFICATION_ANTIGRAVITY,
            "Caller: focused Task 4 qualification tests and future explicitly authorized orchestration/publication. Effect: accepts only the exact frozen seed-major ten FoldMethodScores records, proves 69 unique OOF normals per seed, independently replays C0 plus raw P9 margins/ranks/confirmations and C9, computes fixed Mann-Whitney AUC gates, emits exactly ten unchanged Task 1 scalar fold rows with deterministic reasons, preserves ordered immutable per-seed normal confirmation counts 0..69 as failure evidence, and permits the pass gate only when every seed count is zero.",
            "qualify_antigravity",
            qualification_test,
        ),
        _node_antigravity(
            "task4_evaluation",
            EVALUATION_ANTIGRAVITY,
            "Caller: focused Task 4 evaluation tests and future explicitly authorized publication. Effect: reads only the frozen scalar qualification, requires exact C9 10-of-10 eligibility, rejects forged eligible 9-of-10 decisions, and returns an immutable decision with routing, champion, pending, promotion, TEST-label, and publication effects fixed false/zero.",
            "evaluate_qualification_antigravity",
            evaluation_test,
        ),
        _node_antigravity(
            "task4_scoring_test",
            SCORING_TEST_ANTIGRAVITY,
            "Caller: focused Task 4 RED/GREEN and full Exp466 regression. Effect: detects C0 authority or P9 drift, C9 fusion outside strict confirmation, false-confirm C0 byte drift, signature/order drift, and aliased or mutable raw evidence across two tests.",
            "ScoringAntigravityTest",
            scoring_test,
        ),
        _node_antigravity(
            "task4_qualification_test",
            QUALIFICATION_TEST_ANTIGRAVITY,
            "Caller: focused Task 4 RED/GREEN and full Exp466 regression. Effect: detects missing/reordered ten records, non-unique 69 OOF coverage, trusted instead of replayed C0/P9/C9 evidence, and every fixed nonfinite/AUC/mean/period/gradual/confirmation failure boundary across three tests.",
            "QualificationAntigravityTest",
            qualification_test,
        ),
        _node_antigravity(
            "task4_evaluation_test",
            EVALUATION_TEST_ANTIGRAVITY,
            "Caller: focused Task 4 RED/GREEN and full Exp466 regression. Effect: proves the decision is scorer-free and state-free, blocks a failed fold with deterministic reasons, and rejects a forged eligible 9-of-10 result across three tests.",
            "EvaluationAntigravityTest",
            evaluation_test,
        ),
        _node_antigravity(
            "task4_report",
            TASK4_REPORT_ANTIGRAVITY,
            "Caller: Task 4 implementation, Terra review, and map closeout audit. Effect: records exact RED/GREEN and regression evidence, interfaces, fixed C0/P9/C9 and replay invariants, SPEC PASS plus resolved QUALITY P1s, source boundary, caller/effect/test answers, preserved no-result boundaries, and concerns.",
            "Exp466 Task 4",
            map_test,
        ),
        _node_antigravity(
            "task5_artifacts",
            ARTIFACTS_ANTIGRAVITY,
            "Caller: Task 5 unit tests and future separately authorized Task 6 evaluate path. Effect: validates explicit frozen Task 4 records, qualification, and decision; requires exact blocked-fold indices; creates only a new root with exactly seven canonical JSON/JSONL leaves; hashes the other six leaves while excluding the hash map itself; and retains complete replay evidence without data, TEST, private, official, CLI, routing, or governance access.",
            "publish_result_antigravity",
            (ARTIFACTS_TEST_ANTIGRAVITY, REPLAY_TEST_ANTIGRAVITY),
        ),
        _node_antigravity(
            "task5_replay",
            REPLAY_ANTIGRAVITY,
            "Caller: Task 5 replay tests and future separately authorized Task 6 inspect path. Effect: reads only the exact seven leaves, validates canonical bytes and self-excluding hashes, independently reconstructs blocked-fold identity, P9 ranks, strict confirmations, C9, AUC, summaries, failed rows, qualification, decision, and dashboard, and returns equality-stable evidence without loaders or Task 2/3/4 scorers/qualifier.",
            "inspect_result_antigravity",
            replay_test,
        ),
        _node_antigravity(
            "task5_dashboard_export",
            DASHBOARD_EXPORT_ANTIGRAVITY,
            "Caller: canonical Task 5 publisher and focused dashboard tests. Effect: validates frozen qualification/decision agreement and returns a deterministic easy-Korean snapshot with embedded HTML inside an existing leaf, without an eighth file or any private/official performance claim.",
            "export_dashboard_antigravity",
            (
                DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
                ARTIFACTS_TEST_ANTIGRAVITY,
                REPLAY_TEST_ANTIGRAVITY,
            ),
        ),
        _node_antigravity(
            "task5_artifacts_test",
            ARTIFACTS_TEST_ANTIGRAVITY,
            "Caller: focused Task 5 RED/GREEN and full Exp466 regression. Effect: detects any leaf-count, canonical-byte, hash self-exclusion, create-only, record-order, retained-evidence, or exact blocked-fold-index drift across three tests.",
            "ArtifactsAntigravityTest",
            artifacts_test,
        ),
        _node_antigravity(
            "task5_replay_test",
            REPLAY_TEST_ANTIGRAVITY,
            "Caller: focused Task 5 RED/GREEN, mutation tests, and full Exp466 regression. Effect: proves dataset/scorer-free stable inspection, publishable nonzero-confirmation failure, exact index identity, stale-map rejection, and semantic rejection of re-canonicalized/rehashed mutations across six tests.",
            "ReplayAntigravityTest",
            replay_test,
        ),
        _node_antigravity(
            "task5_dashboard_export_test",
            DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
            "Caller: focused Task 5 RED/GREEN and full Exp466 regression. Effect: detects hard-to-read Korean, nondeterministic embedded/returned HTML, an eighth leaf, and private/official result claims in one test.",
            "DashboardExportAntigravityTest",
            dashboard_export_test,
        ),
        _node_antigravity(
            "task5_report",
            TASK5_REPORT_ANTIGRAVITY,
            "Caller: Task 5 implementation, prerequisite-fix, controller verification, fresh Sol review, and map closeout audit. Effect: records TDD RED/GREEN, exact focused/full/lower evidence, publish-inspect equality, normal-confirmation failure preservation, blocked-fold identity, SHIP with Critical 0 and Important 0, the accepted nonblocking independent-replay duplication minor, no-result boundaries, and concerns.",
            "Exp466 Task 5",
            map_test,
        ),
        _node_antigravity(
            "task6_cli",
            CLI_ANTIGRAVITY,
            "Caller: the fixed Task 6 launcher, focused CLI/governance tests, and the completed private run1/run2 sequence. Effect: accepted only evaluate with one exact lowercase reviewed commit or bare inspect; each private evaluate checked a new fixed root, clean matching commit, absent official root, and no duplicate process before the frozen TRAIN-only pipeline, and each inspect performed source-free replay; all four calls emitted the same canonical label-free payload and produced C9 0-of-10 failure without TEST, tuning, retry, routing, promotion, or official controls.",
            "main_antigravity",
            cli_test,
        ),
        _node_antigravity(
            "task6_launcher",
            LAUNCHER_ANTIGRAVITY,
            "Caller: the completed Step 4 operator sequence at fresh-Sol-approved commit 0aed6d753a3f4cc6f52efc6c573f5f73e4cd522a. Effect: re-executed the frozen interpreter outside the working directory, mapped only run1/run2 to two fixed private roots, enforced evaluate-then-inspect order, and finished the external canonical sequence state at run2_inspected; the preserved commands are now a no-rerun record and never permitted custom-root or official execution.",
            "_run_antigravity",
            launcher_test,
        ),
        _node_antigravity(
            "task6_cli_test",
            CLI_TEST_ANTIGRAVITY,
            "Caller: focused Task 6 RED/GREEN and full Exp466 regression. Effect: detects extra parser controls, eager science imports, forged private capabilities, roots outside the two fixed paths, current-working-directory dependence, broken run order, noncanonical decision projection, and any evaluate/inspect stdout difference across six tests without executing ECG200 or creating either private root.",
            "CliAntigravityTest",
            (CLI_TEST_ANTIGRAVITY,),
        ),
        _node_antigravity(
            "task6_governance_test",
            GOVERNANCE_TEST_ANTIGRAVITY,
            "Caller: focused Task 6 RED/GREEN and full Exp466 regression. Effect: detects extra commands, existing/symlink/official destinations, dirty or wrong commits, duplicate processes, late official guards, changed runtime/roots/order, skipped or repeated stages, commit drift, premature sequence advancement, noncanonical sequence state, non-seed-major scoring, repeated stage calls, inspect data/scorer access, and TEST/routing controls across eleven tests without a private or official run.",
            "CliGovernanceAntigravityTest",
            (GOVERNANCE_TEST_ANTIGRAVITY,),
        ),
    ]
    edges = [
        _edge_antigravity("task1_brief", "governs", "task1_config", BRIEF_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("approved_design", "governs", "task1_config", DESIGN_ANTIGRAVITY, "고정 입력"),
        _edge_antigravity("exp465_baseline", "informs_structure", "map_builder", "docs/codemap/codemap_exp465_antigravity.json", '"map_builder"'),
        _edge_antigravity("task1_config", "governs", "task1_contracts", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", CONFIG_TEST_ANTIGRAVITY, "test_frozen_config_antigravity"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", CONTRACTS_TEST_ANTIGRAVITY, "test_qualification_accepts_bounded_failure_counts_and_rejects_bad_count_shapes_antigravity"),
        _edge_antigravity("approved_design", "governs", "task2_sources", DESIGN_ANTIGRAVITY, "ECG200 TRAIN의 정상"),
        _edge_antigravity("approved_design", "governs", "task2_interpolation", DESIGN_ANTIGRAVITY, "96→140"),
        _edge_antigravity("approved_design", "governs", "task2_synthetic", DESIGN_ANTIGRAVITY, "level shift, gradual drift, amplitude scaling, period change"),
        _edge_antigravity("approved_design", "governs", "task2_predecessor", DESIGN_ANTIGRAVITY, "Exp465의 C0 점수를 그대로 재생"),
        _edge_antigravity("task1_config", "governs", "task2_sources", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config", "governs", "task2_interpolation", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config", "governs", "task2_synthetic", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config", "governs", "task2_predecessor", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("exp465_baseline", "informs_structure", "task2_sources", "docs/codemap/codemap_exp465_antigravity.json", '"task2_sources"'),
        _edge_antigravity("exp465_baseline", "informs_structure", "task2_interpolation", "docs/codemap/codemap_exp465_antigravity.json", '"task2_interpolation"'),
        _edge_antigravity("exp465_baseline", "informs_structure", "task2_predecessor", "docs/codemap/codemap_exp465_antigravity.json", '"task3_scoring"'),
        _edge_antigravity("task2_sources_test", "tests", "task2_sources", SOURCES_TEST_ANTIGRAVITY, "test_export_is_exact_exp465_loader_antigravity"),
        _edge_antigravity("task2_interpolation_test", "tests", "task2_interpolation", INTERPOLATION_TEST_ANTIGRAVITY, "test_export_is_exact_exp465_interpolator_antigravity"),
        _edge_antigravity("task2_synthetic_test", "tests", "task2_synthetic", SYNTHETIC_TEST_ANTIGRAVITY, "test_four_exp464_formulas_are_exact_owned_and_read_only_antigravity"),
        _edge_antigravity("task2_predecessor_test", "tests", "task2_predecessor", PREDECESSOR_TEST_ANTIGRAVITY, "test_c0_normal_and_primary_scores_are_byte_equal_to_exp465_antigravity"),
        _edge_antigravity("approved_design", "governs", "task3_phase_registration", DESIGN_ANTIGRAVITY, "위상 정렬 마진"),
        _edge_antigravity("task1_config", "governs", "task3_phase_registration", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task3_phase_registration_test", "tests", "task3_phase_registration", PHASE_REGISTRATION_TEST_ANTIGRAVITY, "test_zero_margin_is_not_confirmed_above_negative_fit_max_antigravity"),
        _edge_antigravity("task3_report", "verifies", "task3_phase_registration", TASK3_REPORT_ANTIGRAVITY, "수식·경계 검증"),
        _edge_antigravity("map_builder", "verifies", "preregistration", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "runbook", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "implementation_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "dashboard", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "research_state", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "experiment_catalog", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "champion_ledger", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task1_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task2_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task3_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("approved_design", "governs", "task4_scoring", DESIGN_ANTIGRAVITY, "C9"),
        _edge_antigravity("approved_design", "governs", "task4_qualification", DESIGN_ANTIGRAVITY, "10개 seed-fold"),
        _edge_antigravity("approved_design", "governs", "task4_evaluation", DESIGN_ANTIGRAVITY, "C9이 10/10"),
        _edge_antigravity("task1_config", "governs", "task4_scoring", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config", "governs", "task4_qualification", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config", "governs", "task4_evaluation", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_contracts", "governs", "task4_qualification", CONTRACTS_ANTIGRAVITY, "Exp466QualificationAntigravity"),
        _edge_antigravity("task1_contracts", "governs", "task4_evaluation", CONTRACTS_ANTIGRAVITY, "Exp466QualificationAntigravity"),
        _edge_antigravity("task2_predecessor", "governs", "task4_scoring", SCORING_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("task2_synthetic", "governs", "task4_scoring", SCORING_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("task3_phase_registration", "governs", "task4_scoring", SCORING_ANTIGRAVITY, "score_phase_registration_antigravity"),
        _edge_antigravity("task4_scoring", "governs", "task4_qualification", QUALIFICATION_ANTIGRAVITY, "FoldMethodScoresAntigravity"),
        _edge_antigravity("task4_qualification", "governs", "task4_evaluation", EVALUATION_ANTIGRAVITY, "Exp466QualificationAntigravity"),
        _edge_antigravity("task4_scoring_test", "tests", "task4_scoring", SCORING_TEST_ANTIGRAVITY, "test_fixed_c0_p9_c9_assembly_and_raw_phase_evidence_antigravity"),
        _edge_antigravity("task4_qualification_test", "tests", "task4_qualification", QUALIFICATION_TEST_ANTIGRAVITY, "test_exact_ten_records_replay_inputs_and_cover_69_unique_oof_antigravity"),
        _edge_antigravity("task4_evaluation_test", "tests", "task4_evaluation", EVALUATION_TEST_ANTIGRAVITY, "test_pure_c9_decision_accepts_ten_of_ten_without_state_changes_antigravity"),
        _edge_antigravity("task4_report", "verifies", "task4_scoring", TASK4_REPORT_ANTIGRAVITY, "고정 수식과 불변 조건"),
        _edge_antigravity("task4_report", "verifies", "task4_qualification", TASK4_REPORT_ANTIGRAVITY, "고정 수식과 불변 조건"),
        _edge_antigravity("task4_report", "verifies", "task4_evaluation", TASK4_REPORT_ANTIGRAVITY, "Decision"),
        _edge_antigravity("map_builder", "verifies", "task4_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("approved_design", "governs", "task5_artifacts", DESIGN_ANTIGRAVITY, "canonical 7 leaves"),
        _edge_antigravity("task4_scoring", "governs", "task5_artifacts", ARTIFACTS_ANTIGRAVITY, "FoldMethodScoresAntigravity"),
        _edge_antigravity("task4_qualification", "governs", "task5_artifacts", ARTIFACTS_ANTIGRAVITY, "Exp466QualificationAntigravity"),
        _edge_antigravity("task4_evaluation", "governs", "task5_artifacts", ARTIFACTS_ANTIGRAVITY, "QualificationDecisionAntigravity"),
        _edge_antigravity("task5_dashboard_export", "governs", "task5_artifacts", ARTIFACTS_ANTIGRAVITY, "export_dashboard_antigravity"),
        _edge_antigravity("task5_artifacts", "governs", "task5_replay", REPLAY_ANTIGRAVITY, "_LEAVES_ANTIGRAVITY"),
        _edge_antigravity("task5_artifacts_test", "tests", "task5_artifacts", ARTIFACTS_TEST_ANTIGRAVITY, "test_publish_creates_exact_canonical_self_excluding_seven_leaves_antigravity"),
        _edge_antigravity("task5_replay_test", "tests", "task5_replay", REPLAY_TEST_ANTIGRAVITY, "test_temporary_publish_inspect_is_stable_and_scorer_data_free_antigravity"),
        _edge_antigravity("task5_dashboard_export_test", "tests", "task5_dashboard_export", DASHBOARD_EXPORT_TEST_ANTIGRAVITY, "test_easy_korean_snapshot_embeds_deterministic_html_without_eighth_leaf_antigravity"),
        _edge_antigravity("task5_report", "verifies", "task5_artifacts", TASK5_REPORT_ANTIGRAVITY, "Fresh Sol review"),
        _edge_antigravity("task5_report", "verifies", "task5_replay", TASK5_REPORT_ANTIGRAVITY, "Fresh Sol review"),
        _edge_antigravity("task5_report", "verifies", "task5_dashboard_export", TASK5_REPORT_ANTIGRAVITY, "Fresh Sol review"),
        _edge_antigravity("map_builder", "verifies", "task5_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("approved_design", "governs", "task6_cli", DESIGN_ANTIGRAVITY, "canonical 7 leaves"),
        _edge_antigravity("approved_design", "governs", "task6_launcher", DESIGN_ANTIGRAVITY, "서로 다른 private root"),
        _edge_antigravity("task1_config", "governs", "task6_cli", CLI_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task2_sources", "governs", "task6_cli", CLI_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("task2_interpolation", "governs", "task6_cli", CLI_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("task2_predecessor", "governs", "task6_cli", CLI_ANTIGRAVITY, "capture_exp465_authority_antigravity"),
        _edge_antigravity("task4_scoring", "governs", "task6_cli", CLI_ANTIGRAVITY, "score_fold_methods_antigravity"),
        _edge_antigravity("task4_qualification", "governs", "task6_cli", CLI_ANTIGRAVITY, "qualify_antigravity"),
        _edge_antigravity("task4_evaluation", "governs", "task6_cli", CLI_ANTIGRAVITY, "evaluate_qualification_antigravity"),
        _edge_antigravity("task5_artifacts", "governs", "task6_cli", CLI_ANTIGRAVITY, "publish_result_antigravity"),
        _edge_antigravity("task5_replay", "governs", "task6_cli", CLI_ANTIGRAVITY, "inspect_result_antigravity"),
        _edge_antigravity("task6_launcher", "governs", "task6_cli", LAUNCHER_ANTIGRAVITY, "_load_main_antigravity"),
        _edge_antigravity("task6_cli_test", "tests", "task6_cli", CLI_TEST_ANTIGRAVITY, "test_exact_surface_accepts_only_reviewed_commands_antigravity"),
        _edge_antigravity("task6_cli_test", "tests", "task6_launcher", CLI_TEST_ANTIGRAVITY, "test_launcher_is_cwd_independent_and_enforces_run_order_antigravity"),
        _edge_antigravity("task6_governance_test", "tests", "task6_cli", GOVERNANCE_TEST_ANTIGRAVITY, "test_evaluate_pipeline_is_seed_major_and_calls_each_stage_once_antigravity"),
        _edge_antigravity("task6_governance_test", "tests", "task6_launcher", GOVERNANCE_TEST_ANTIGRAVITY, "test_launcher_rejects_skipped_repeated_and_commit_drift_phases_antigravity"),
    ]
    return {
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent_commit": source_parent,
        "baseline_triplet": [
            "docs/codemap/codemap_exp465_antigravity.html",
            "docs/codemap/codemap_exp465_antigravity.json",
            "docs/codemap/codemap_exp465_antigravity.lock",
        ],
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": [],
        "task1_answers": {
            "callers": "config is called by contracts/tests and future Tasks 2-4; contracts are called by future scoring/qualification/replay/publication; truth surfaces are called by governance/audit; builder is called by Task 1 closeout.",
            "effects": "freezes authority and immutable result shapes, records planned zero/null governance, and documents boundaries without loading ECG200 or calculating a score.",
            "tests": "test_config_antigravity and test_contracts_antigravity cover executable contracts; build_codemap_exp466_antigravity.py --validate-only covers evidence, HTML, graph, embedded JSON, and fingerprints.",
        },
        "task2_answers": {
            "callers": "future Exp466 phase scoring and qualification call the TRAIN source, 96-to-140 interpolation, four-change synthetic generator, and Exp465 C0 authority adapter; their four focused tests call each boundary directly.",
            "effects": "source/interpolation preserve exact Exp465 function-object behavior, synthetic creates only the four frozen owned read-only float64 changes, and predecessor pins the reviewed Exp465 closeout/scorer while projecting byte-equal immutable C0 normal/primary vectors; no P9/C9, private/official, routing, or promotion state changes.",
            "tests": "test_sources_antigravity covers exact export and TEST-before-open; test_interpolation_antigravity covers exact export; test_synthetic_antigravity covers formulas/order/ownership/validation; test_predecessor_antigravity covers closeout/source identity, signature, immutability, and C0 byte parity; the map builder validates all nodes, edges, HTML, embedded JSON, and tracked fingerprints.",
        },
        "task3_answers": {
            "callers": "the six focused Task 3 tests call the phase margin, fold scorer, and immutable result contract directly; only a future separately authorized Exp466 C9/qualification stage may consume P9 margins, ranks, and confirmations.",
            "effects": "reads validated 140-point fit/query rows and emits only LOO fit margins, full-fit query margins, the fit maximum, exact frozen anomaly ranks, and strict positive fit-max confirmations as owned read-only finite arrays; it does not mutate Task 1/2 inputs or create C9, qualification, publication, routing, private, official, or MLX state.",
            "tests": "test_phase_registration_antigravity covers fixed onset56/suffix84/warp1.25/shared67 math, affine invariance, literal LOO/full-fit calibration, exact ranks, strict confirmations including query_margin==0 above a negative fit maximum, malformed input/MAD rejection, and immutable non-aliasing outputs; the full 20-test Exp466 suite preserves Task 1/2 interfaces, and the map builder validates Task 3 nodes, edges, report HTML, embedded JSON, source boundary, and all fingerprints.",
        },
        "task4_answers": {
            "callers": "test_scoring_antigravity calls score_fold_methods_antigravity and its immutable record boundary; test_qualification_antigravity calls qualify_antigravity over the exact frozen seed-major ten-record mapping; test_evaluation_antigravity calls evaluate_qualification_antigravity over the frozen scalar qualification; future publication must receive records, qualification, and decision explicitly rather than recover hidden state.",
            "effects": "scoring reads the reviewed Task 2 C0/synthetic and Task 3 P9 interfaces to emit owned immutable C0/P9/C9 raw evidence; qualification verifies 69 unique OOF normals per seed and independently replays C0/P9/C9 before emitting exactly ten Task 1 scalar rows; evaluation makes only a frozen 10-of-10 decision with all routing/champion/pending/promotion/TEST-label effects false or zero; no real data, private, official, MLX, publication, or governance state is touched.",
            "tests": "the two scoring tests cover exact C0/P9/C9 assembly, false-confirm C0 byte identity, and non-aliased immutable evidence; the three qualification tests cover exact ten-record order, 69 unique OOF coverage, raw replay, and every fixed negative gate; the three evaluation tests cover pure acceptance, deterministic failure, and forged eligible 9-of-10 rejection; focused 8, full Exp466 28, Exp465 lower 7, map validation, HTML/JSON parsing, and independent fingerprint recomputation cover the closeout.",
        },
        "task5_answers": {
            "callers": "test_artifacts_antigravity calls publish_result_antigravity with deterministic Task 4 records, qualification, and decision; test_replay_antigravity calls inspect_result_antigravity over temporary seven-leaf roots; test_dashboard_export_antigravity calls export_dashboard_antigravity directly; the closed Task 6 evaluate/inspect boundary calls publisher and inspector only through its fixed private capability.",
            "effects": "publication validates exact blocked-fold identities and creates a new root containing exactly seven canonical leaves plus a self-excluding hash map; replay reads only those leaves and independently reconstructs every stored and derived scientific value; dashboard export embeds deterministic easy-Korean HTML without an eighth leaf; no loader, Task 2/3/4 scorer or qualifier, TEST/label, real data, private/official, CLI, routing, or governance state is accessed.",
            "tests": "the three artifact tests cover exact leaves, canonical bytes, hash-map self-exclusion, create-only behavior, record order, and blocked-fold identity; the six replay tests cover stable scorer/data-free inspection, publishable normal-confirmation failure, rehashed semantic mutations, stale hashes, exact indices, context/summary/failed-leaf mutations, and equality; the dashboard test covers easy Korean, deterministic embedded/returned HTML, no eighth leaf, and no private/official claim; controller focused 10 twice, full Exp466 39, Exp465 lower 7, temporary publish-inspect equality, fresh Sol SHIP, map validation, HTML/JSON parsing, and fingerprint recomputation cover closeout.",
        },
        "task6_answers": {
            "callers": "the fixed outside-CWD launcher called main_antigravity four times in the completed run1 evaluate→inspect then run2 evaluate→inspect sequence, injecting only the matching fixed private capability; the six CLI tests and eleven governance tests call parser, capability, preflight, pipeline, replay projection, and sequence boundaries directly; Step 6 reviewers now consume only the preserved roots, stdout, ledgers, truth HTML, and code map without rerunning ECG.",
            "effects": "the launcher fixed the interpreter, two roots, clean reviewed commit 0aed6d753a3f4cc6f52efc6c573f5f73e4cd522a, and exact order in a canonical external sequence-state file that ended at run2_inspected; each evaluate ran the frozen TRAIN-only ten-fold pipeline and each inspect replayed only seven leaves; all four stdout payloads and all seven leaves across both roots are byte-equal, C9 passed 0 of 10, and official/TEST/label/routing/promotion/champion/pending effects remain zero or false.",
            "tests": "Task 6 RED, focused 17/17 twice, full Exp466 56/56, Exp465 lower 7/7, compile/diff, fresh Sol SHIP, and clean/official/process preflight established the execution boundary. The completed private comparison verifies four identical stdout SHA-256 values, seven identical leaf SHA-256 values, final sequence stage run2_inspected, finite scores, deterministic qualification_failed_blocked_before_fn_validation, and no official/label/external mutation; Step 6 truth/full verification and final independent review remain pending.",
        },
        "flows": [
            "task1 brief + approved design -> frozen config -> immutable contracts -> future Tasks 2-6",
            "focused config/contracts RED-GREEN tests -> executable Task 1 authority verification",
            "truth HTML + additive planned ledgers + dashboard -> map builder -> canonical triplet and read-only fingerprint validation",
            "ECG200 TRAIN path -> exact Exp465 source adapter -> exact Exp465 96-to-140 interpolation adapter -> future Exp466 phase scoring/qualification input",
            "frozen config + Exp464 formulas -> four-change synthetic adapter -> owned read-only primary query rows -> future Exp466 phase scoring/qualification",
            "reviewed Exp465 closeout + exact Exp465 fold scorer -> predecessor C0 adapter -> byte-equal immutable C0 normal/primary score projection",
            "four focused Task 2 tests + Luna review + Task 2 report -> map builder -> Task 2 caller/effect/test boundaries and fingerprints",
            "frozen 140-point geometry -> independently standardized suffix84 -> fixed warp1.25/shared67 margin -> LOO/full-fit calibration -> exact ranks and strict confirmations",
            "six focused Task 3 tests + Terra final PASS + Task 3 report -> map builder -> Task 3 caller/effect/test boundaries and fingerprints",
            "Task 2 C0 authority + Task 2 four-change rows + Task 3 P9 phase evidence -> fixed C0/P9/C9 fold scoring -> owned immutable raw replay records",
            "exact seed-major two-seed x five-fold records -> 69 unique OOF normals per seed -> independent C0/P9/C9 replay -> exactly ten scalar qualification rows",
            "frozen scalar qualification -> pure C9 10-of-10 decision -> Exp467-design eligibility or deterministic block with no external state mutation",
            "eight focused Task 4 tests + Terra SPEC PASS/QUALITY P1 disposition + Task 4 report -> map builder -> Task 4 caller/effect/test boundaries and fingerprints",
            "explicit frozen Task 4 records + qualification + decision -> exact blocked-fold validation -> create-only canonical publisher -> exactly seven leaves with a self-excluding hash map",
            "exact seven leaves -> canonical byte and hash validation -> retained train/index/margin evidence -> independent P9/C9/AUC/qualification/decision replay without dataset or scorer access",
            "frozen qualification + decision -> deterministic easy-Korean dashboard and embedded HTML -> existing dashboard snapshot leaf without an eighth file or performance claim",
            "ten focused Task 5 tests + temporary publish-inspect equality + fresh Sol SHIP + Task 5 report -> map builder -> Task 5 caller/effect/test boundaries and exact fingerprints",
            "fresh Sol SHIP-reviewed clean commit 0aed6d753a3f4cc6f52efc6c573f5f73e4cd522a + clean/official/process preflight PASS -> fixed run1/run2 environment selector -> outside-CWD frozen-runtime launcher -> canonical external sequence-state run2_inspected -> completed private evaluate/inspect count 2/2",
            "two evaluate calls + exact clean reviewed commit + absent official/new private roots -> frozen TRAIN-only ten-fold pipeline -> C9 qualification 0-of-10 -> seven-leaf publication -> source-free inspection -> byte-equal canonical stdout",
            "two fixed seven-leaf roots + four stdout captures -> exact byte and SHA-256 equality -> qualification_failed_blocked_before_fn_validation -> zero official/TEST/label/routing/promotion/champion/pending effects -> Step 6 truth/full verification and final independent review only",
        ],
    }


def _render_html_antigravity(graph: dict[str, Any]) -> str:
    payload = html_module.escape(
        _canonical_bytes_antigravity(graph).decode("utf-8").rstrip("\n"),
    )
    nodes = "".join(
        "<li><code>"
        + html_module.escape(str(node["path"]))
        + "</code> — "
        + html_module.escape(str(node["role"]))
        + "</li>"
        for node in graph["nodes"]
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Exp466 Task 1-6 코드맵</title><style>body{max-width:1180px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}pre{padding:1rem}</style></head><body><main>'
        '<h1>Exp466 Task 1-6 코드맵</h1><p>이 map은 Exp466에서 어떤 코드가 무엇을 부르고, 무엇을 바꾸며, 어떤 검사로 확인하는지를 기록한다. Task 1~5는 TRAIN 입력, C0/P9/C9 점수, 10개 fold 판정, 7개 결과 파일과 독립 replay 경계를 고정했다. Task 6은 run1과 run2만 허용하는 launcher와 evaluate·inspect 두 명령만 허용하는 CLI를 연결했다. reviewed commit 0aed6d753a3f4cc6f52efc6c573f5f73e4cd522a에서 두 번의 private TRAIN-only 실행을 순서대로 마쳤고 외부 순서 상태는 run2_inspected로 끝났다. private run/evaluate/inspect는 2/2/2다. 네 stdout과 두 root의 7개 파일은 모두 byte 단위로 같았지만 C9 합격은 0/10이어서 판정은 qualification_failed_blocked_before_fn_validation이다. period AUC 0.85 미만 10개, 평균 AUC 0.90 미만 7개, C0보다 period 개선 0개, period confirmation 없음 8개, gradual 악화 2개, seed별 정상 confirmation 1개가 측정됐다. 보존 파일 안의 Task 5·actual=false·합성 단위 문구는 Task 6 출처 설명으로 쓰지 않지만 과학 점수, FAIL 판정, byte 재현성은 유효하다. official·TEST·label·routing·champion·pending·promotion은 모두 0 또는 false이며 공식 결과와 공식 hash는 null이다. Task 6 Step 6의 truth/full verify와 최종 독립 검토만 남았다.</p>'
        f'<h2>노드</h2><ul>{nodes}</ul><h2>Machine-readable graph</h2>'
        f'<pre id="codemap-json-antigravity">{payload}</pre>'
        '</main></body></html>'
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


def validate_map_antigravity(
    graph: dict[str, Any],
    lock: dict[str, Any],
    rendered: str,
) -> None:
    if graph.get("schema") != GRAPH_SCHEMA_ANTIGRAVITY:
        raise ValueError("graph schema differs from Task 6 preflight authority")
    if lock.get("schema") != LOCK_SCHEMA_ANTIGRAVITY:
        raise ValueError("lock schema differs from Task 6 preflight authority")
    if graph.get("source_parent_commit") != TASK6_SOURCE_BASE_COMMIT_ANTIGRAVITY:
        raise ValueError("graph source must be the accepted Task 6 source base")
    if graph.get("unknown_relationships") != []:
        raise ValueError("unknown relationships must stay empty")
    if graph.get("baseline_triplet") != [
        "docs/codemap/codemap_exp465_antigravity.html",
        "docs/codemap/codemap_exp465_antigravity.json",
        "docs/codemap/codemap_exp465_antigravity.lock",
    ]:
        raise ValueError("Exp465 baseline triplet is not exact")
    allowed_types = {"governs", "tests", "verifies", "informs_structure"}
    ids: set[str] = set()
    for node in graph["nodes"]:
        if not isinstance(node, dict) or not node.get("id") or not node.get("path") or not node.get("role"):
            raise ValueError("invalid node")
        ids.add(node["id"])
        if not node.get("tests") or any(path not in TRACKED_PATHS_ANTIGRAVITY for path in node["tests"]):
            raise ValueError("invalid node tests")
        for item in node.get("evidence", []):
            if set(item) != {"path", "symbol"} or not _evidence_resolves_antigravity(item["path"], item["symbol"]):
                raise ValueError(f"unresolved node evidence: {item}")
    for edge in graph["edges"]:
        if edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in allowed_types:
            raise ValueError(f"invalid edge: {edge}")
        item = edge.get("evidence")
        if set(item) != {"path", "symbol"} or not _evidence_resolves_antigravity(item["path"], item["symbol"]):
            raise ValueError(f"unresolved edge evidence: {item}")
    required_task3_nodes = {
        "task3_phase_registration",
        "task3_phase_registration_test",
        "task3_report",
    }
    if not required_task3_nodes.issubset(ids):
        raise ValueError("Task 3 nodes are incomplete")
    edge_keys = {
        (edge["from"], edge["type"], edge["to"])
        for edge in graph["edges"]
    }
    required_task3_edges = {
        ("approved_design", "governs", "task3_phase_registration"),
        ("task1_config", "governs", "task3_phase_registration"),
        ("task3_phase_registration_test", "tests", "task3_phase_registration"),
        ("task3_report", "verifies", "task3_phase_registration"),
        ("map_builder", "verifies", "task3_report"),
    }
    if not required_task3_edges.issubset(edge_keys):
        raise ValueError("Task 3 caller/effect/test edges are incomplete")
    required_task4_nodes = {
        "task4_scoring",
        "task4_qualification",
        "task4_evaluation",
        "task4_scoring_test",
        "task4_qualification_test",
        "task4_evaluation_test",
        "task4_report",
    }
    if not required_task4_nodes.issubset(ids):
        raise ValueError("Task 4 nodes are incomplete")
    required_task4_edges = {
        ("approved_design", "governs", "task4_scoring"),
        ("approved_design", "governs", "task4_qualification"),
        ("approved_design", "governs", "task4_evaluation"),
        ("task2_predecessor", "governs", "task4_scoring"),
        ("task2_synthetic", "governs", "task4_scoring"),
        ("task3_phase_registration", "governs", "task4_scoring"),
        ("task4_scoring", "governs", "task4_qualification"),
        ("task4_qualification", "governs", "task4_evaluation"),
        ("task4_scoring_test", "tests", "task4_scoring"),
        ("task4_qualification_test", "tests", "task4_qualification"),
        ("task4_evaluation_test", "tests", "task4_evaluation"),
        ("task4_report", "verifies", "task4_scoring"),
        ("task4_report", "verifies", "task4_qualification"),
        ("task4_report", "verifies", "task4_evaluation"),
        ("map_builder", "verifies", "task4_report"),
    }
    if not required_task4_edges.issubset(edge_keys):
        raise ValueError("Task 4 caller/effect/test edges are incomplete")
    required_task5_nodes = {
        "task5_artifacts",
        "task5_replay",
        "task5_dashboard_export",
        "task5_artifacts_test",
        "task5_replay_test",
        "task5_dashboard_export_test",
        "task5_report",
    }
    if not required_task5_nodes.issubset(ids):
        raise ValueError("Task 5 nodes are incomplete")
    required_task5_edges = {
        ("approved_design", "governs", "task5_artifacts"),
        ("task4_scoring", "governs", "task5_artifacts"),
        ("task4_qualification", "governs", "task5_artifacts"),
        ("task4_evaluation", "governs", "task5_artifacts"),
        ("task5_dashboard_export", "governs", "task5_artifacts"),
        ("task5_artifacts", "governs", "task5_replay"),
        ("task5_artifacts_test", "tests", "task5_artifacts"),
        ("task5_replay_test", "tests", "task5_replay"),
        ("task5_dashboard_export_test", "tests", "task5_dashboard_export"),
        ("task5_report", "verifies", "task5_artifacts"),
        ("task5_report", "verifies", "task5_replay"),
        ("task5_report", "verifies", "task5_dashboard_export"),
        ("map_builder", "verifies", "task5_report"),
    }
    if not required_task5_edges.issubset(edge_keys):
        raise ValueError("Task 5 caller/effect/test edges are incomplete")
    required_task6_nodes = {
        "task6_cli",
        "task6_launcher",
        "task6_cli_test",
        "task6_governance_test",
    }
    if not required_task6_nodes.issubset(ids):
        raise ValueError("Task 6 nodes are incomplete")
    required_task6_edges = {
        ("approved_design", "governs", "task6_cli"),
        ("approved_design", "governs", "task6_launcher"),
        ("task1_config", "governs", "task6_cli"),
        ("task2_sources", "governs", "task6_cli"),
        ("task2_interpolation", "governs", "task6_cli"),
        ("task2_predecessor", "governs", "task6_cli"),
        ("task4_scoring", "governs", "task6_cli"),
        ("task4_qualification", "governs", "task6_cli"),
        ("task4_evaluation", "governs", "task6_cli"),
        ("task5_artifacts", "governs", "task6_cli"),
        ("task5_replay", "governs", "task6_cli"),
        ("task6_launcher", "governs", "task6_cli"),
        ("task6_cli_test", "tests", "task6_cli"),
        ("task6_cli_test", "tests", "task6_launcher"),
        ("task6_governance_test", "tests", "task6_cli"),
        ("task6_governance_test", "tests", "task6_launcher"),
    }
    if not required_task6_edges.issubset(edge_keys):
        raise ValueError("Task 6 caller/effect/test edges are incomplete")
    if set(graph.get("task1_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 1 caller/effect/test answers are incomplete")
    if set(graph.get("task2_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 2 caller/effect/test answers are incomplete")
    if set(graph.get("task3_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 3 caller/effect/test answers are incomplete")
    if set(graph.get("task4_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 4 caller/effect/test answers are incomplete")
    if set(graph.get("task5_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 5 caller/effect/test answers are incomplete")
    if set(graph.get("task6_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 6 caller/effect/test answers are incomplete")
    if len(graph.get("nodes", ())) != 46:
        raise ValueError("Task 1-6 node count differs")
    if len(graph.get("edges", ())) != 84:
        raise ValueError("Task 1-6 edge count differs")
    if len(graph.get("flows", ())) != 20:
        raise ValueError("Task 1-6 flow count differs")
    if lock.get("source_parent_commit") != graph.get("source_parent_commit"):
        raise ValueError("graph/lock source mismatch")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("lock represented paths differ")
    if lock.get("working_tree_dirty") is not False:
        raise ValueError("lock must record a closed generation boundary")
    if lock.get("fingerprint_count") != len(TRACKED_PATHS_ANTIGRAVITY):
        raise ValueError("lock fingerprint count differs")
    fingerprint_paths = {
        item.get("path")
        for item in lock.get("tracked_fingerprints", ())
        if isinstance(item, dict)
    }
    required_task4_fingerprints = {
        SCORING_ANTIGRAVITY,
        QUALIFICATION_ANTIGRAVITY,
        EVALUATION_ANTIGRAVITY,
        SCORING_TEST_ANTIGRAVITY,
        QUALIFICATION_TEST_ANTIGRAVITY,
        EVALUATION_TEST_ANTIGRAVITY,
        TASK4_REPORT_ANTIGRAVITY,
    }
    if not required_task4_fingerprints.issubset(fingerprint_paths):
        raise ValueError("Task 4 fingerprints are incomplete")
    required_task5_fingerprints = {
        ARTIFACTS_ANTIGRAVITY,
        REPLAY_ANTIGRAVITY,
        DASHBOARD_EXPORT_ANTIGRAVITY,
        ARTIFACTS_TEST_ANTIGRAVITY,
        REPLAY_TEST_ANTIGRAVITY,
        DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
        TASK5_REPORT_ANTIGRAVITY,
    }
    if not required_task5_fingerprints.issubset(fingerprint_paths):
        raise ValueError("Task 5 fingerprints are incomplete")
    required_task6_fingerprints = {
        CLI_ANTIGRAVITY,
        LAUNCHER_ANTIGRAVITY,
        CLI_TEST_ANTIGRAVITY,
        GOVERNANCE_TEST_ANTIGRAVITY,
    }
    if not required_task6_fingerprints.issubset(fingerprint_paths):
        raise ValueError("Task 6 fingerprints are incomplete")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity():
        raise ValueError("tracked fingerprint drift")
    parser = _StrictHtmlAntigravity()
    parser.feed(rendered)
    parser.close()
    if parser.stack:
        raise ValueError("generated code-map HTML has unclosed tags")
    embedded = re.search(
        r'<pre id="codemap-json-antigravity">(.*?)</pre>',
        rendered,
        flags=re.DOTALL,
    )
    if not embedded:
        raise ValueError("embedded graph missing")
    embedded_graph = json.loads(html_module.unescape(embedded.group(1)))
    if _canonical_bytes_antigravity(embedded_graph) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON graph mismatch")
    for relative in (
        "docs/EXP466_DASHBOARD_antigravity.html",
        "docs/EXP466_IMPLEMENTATION_REPORT_antigravity.html",
        "docs/EXP466_PREREGISTRATION_antigravity.html",
        "docs/EXP466_RUNBOOK_antigravity.html",
        TASK_REPORT_ANTIGRAVITY,
        TASK2_REPORT_ANTIGRAVITY,
        TASK3_REPORT_ANTIGRAVITY,
        TASK4_REPORT_ANTIGRAVITY,
        TASK5_REPORT_ANTIGRAVITY,
    ):
        _validated_html_text_antigravity(ROOT_ANTIGRAVITY / relative)
    for relative in ("CHAMPION_LEDGER.json", "EXPERIMENT_CATALOG.json", "RESEARCH_STATE.json"):
        json.loads((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))


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
    rendered = _render_html_antigravity(graph)
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
        build_triplet_antigravity("0" * 40, validate_only=True)
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    build_triplet_antigravity(args.source_commit)


if __name__ == "__main__":
    main_antigravity()
