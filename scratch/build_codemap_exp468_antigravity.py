#!/usr/bin/env python3
"""Build and validate the bounded Exp468 Task 2 code-map triplet."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import html.parser
import json
import re
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
HTML_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp468_antigravity.html"
JSON_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp468_antigravity.json"
LOCK_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp468_antigravity.lock"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp468_antigravity.html",
    "docs/codemap/codemap_exp468_antigravity.json",
    "docs/codemap/codemap_exp468_antigravity.lock",
)
BASELINE_TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp466_antigravity.html",
    "docs/codemap/codemap_exp466_antigravity.json",
    "docs/codemap/codemap_exp466_antigravity.lock",
)
PACKAGE_ANTIGRAVITY = "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity"
CONFIG_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/config_antigravity.py"
CONTRACTS_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/contracts_antigravity.py"
CONFIG_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_config_antigravity.py"
CONTRACTS_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_contracts_antigravity.py"
SOURCES_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/sources_antigravity.py"
INTERPOLATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/interpolation_antigravity.py"
SYNTHETIC_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/synthetic_antigravity.py"
PREDECESSOR_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/predecessor_antigravity.py"
SOURCES_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_sources_antigravity.py"
INTERPOLATION_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_interpolation_antigravity.py"
SYNTHETIC_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_synthetic_antigravity.py"
PREDECESSOR_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_predecessor_antigravity.py"
EXP466_PACKAGE_ANTIGRAVITY = "experiments/exp466_ecg200_phase_registration_margin_antigravity"
EXP466_SOURCES_ANTIGRAVITY = f"{EXP466_PACKAGE_ANTIGRAVITY}/sources_antigravity.py"
EXP466_INTERPOLATION_ANTIGRAVITY = f"{EXP466_PACKAGE_ANTIGRAVITY}/interpolation_antigravity.py"
EXP466_SYNTHETIC_ANTIGRAVITY = f"{EXP466_PACKAGE_ANTIGRAVITY}/synthetic_antigravity.py"
EXP466_PREDECESSOR_ANTIGRAVITY = f"{EXP466_PACKAGE_ANTIGRAVITY}/predecessor_antigravity.py"
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity-design.html"
)
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html"
)
TASK1_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-1-brief-antigravity.html"
)
TASK2_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-2-brief-antigravity.html"
)
PREREGISTRATION_ANTIGRAVITY = "docs/EXP468_PREREGISTRATION_antigravity.html"
RUNBOOK_ANTIGRAVITY = "docs/EXP468_RUNBOOK_antigravity.html"
IMPLEMENTATION_REPORT_ANTIGRAVITY = "docs/EXP468_IMPLEMENTATION_REPORT_antigravity.html"
DASHBOARD_ARTIFACT_ANTIGRAVITY = "scratch/exp468_dashboard_artifact_antigravity.json"
DASHBOARD_ANTIGRAVITY = "docs/EXP468_DASHBOARD_antigravity.html"
BUILDER_ANTIGRAVITY = "scratch/build_codemap_exp468_antigravity.py"
TRACKED_PATHS_ANTIGRAVITY = (
    *BASELINE_TRIPLET_ANTIGRAVITY,
    DESIGN_ANTIGRAVITY,
    PLAN_ANTIGRAVITY,
    TASK1_BRIEF_ANTIGRAVITY,
    TASK2_BRIEF_ANTIGRAVITY,
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
    EXP466_SOURCES_ANTIGRAVITY,
    EXP466_INTERPOLATION_ANTIGRAVITY,
    EXP466_SYNTHETIC_ANTIGRAVITY,
    EXP466_PREDECESSOR_ANTIGRAVITY,
    PREREGISTRATION_ANTIGRAVITY,
    RUNBOOK_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_ANTIGRAVITY,
    DASHBOARD_ARTIFACT_ANTIGRAVITY,
    DASHBOARD_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp468_task2_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp468_task2_codemap_lock_v1"
SOURCE_PARENT_COMMIT_ANTIGRAVITY = "f82c8a31623e73935c63c982fda294e399f717cd"
GENERATED_AT_ANTIGRAVITY = "2026-08-24T00:00:00+09:00"
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


def _fingerprint_antigravity(relative: str) -> dict[str, Any]:
    payload = (ROOT_ANTIGRAVITY / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _tracked_fingerprints_antigravity() -> list[dict[str, Any]]:
    return [_fingerprint_antigravity(path) for path in sorted(TRACKED_PATHS_ANTIGRAVITY)]


class _StrictHtmlAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.add(tag)
        if tag not in _VOID_TAGS_ANTIGRAVITY:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.seen.add(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            raise ValueError(f"misnested HTML closing tag: {tag}")
        self.stack.pop()


def _validated_html_text_antigravity(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    parser = _StrictHtmlAntigravity()
    parser.feed(text)
    parser.close()
    if parser.stack:
        raise ValueError(f"unclosed HTML tags in {path}: {parser.stack[-5:]}")
    if not {"html", "head", "body"}.issubset(parser.seen):
        raise ValueError(f"required HTML structure missing in {path}")
    return text


def _evidence_resolves_antigravity(path: str, symbol: str) -> bool:
    candidate = ROOT_ANTIGRAVITY / path
    return candidate.is_file() and symbol in candidate.read_text(encoding="utf-8")


def _node_antigravity(
    node_id: str,
    path: str,
    role: str,
    symbol: str,
    tests: list[str],
) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "role": role,
        "tests": tests,
        "evidence": [{"path": path, "symbol": symbol}],
    }


def _edge_antigravity(
    source: str,
    edge_type: str,
    target: str,
    path: str,
    symbol: str,
) -> dict[str, Any]:
    return {
        "from": source,
        "type": edge_type,
        "to": target,
        "evidence": {"path": path, "symbol": symbol},
    }


def build_graph_antigravity() -> dict[str, Any]:
    nodes = [
        _node_antigravity(
            "task1_brief",
            TASK1_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 1 implementation authority. Effect: fixes file ownership, frozen values including change_types, required interfaces, RED/GREEN sequence, document/dashboard/map outputs, validation commands, report contract, and the no-data/no-score boundary.",
            "Read this first — it is your requirements",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_brief",
            TASK2_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 2 implementation authority. Effect: fixes adapter/test ownership, exact Exp466 imports, TRAIN/TEST boundary, four-scope C0 byte parity, RED/GREEN sequence, dashboard/map updates, verification commands, and report contract.",
            "Read this first. This file is the complete Task 2 requirement.",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "approved_design",
            DESIGN_ANTIGRAVITY,
            "Caller: approved scientific authority. Effect: fixes TRAIN-only identity, 5x28 local STFT geometry, symmetric-Hann reference, natural-log JS, C0/S10/C10 roles, and the period-focused qualification boundary.",
            "Exp468 — ECG200 국소 STFT 주파수 재분배 자격 검증",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "implementation_plan",
            PLAN_ANTIGRAVITY,
            "Caller: Task-by-Task controller and later implementers. Effect: records Task 1-2 checked completion, exact change_types order, backend-neutral Hann definition, file ownership, and the remaining Task 3-6 boundaries.",
            "implementation_task_2_complete",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task1_config",
            CONFIG_ANTIGRAVITY,
            "Caller: Task 1 contracts/tests and future source, STFT, scoring, and qualification modules. Effect: exact-type freezes ECG200 identity, seeds/folds, suffix/windows, symmetric-Hann reference, RFFT/epsilon/JS, change_types, method order, period threshold, and design digest without choosing a backend.",
            "FROZEN_CONFIG_ANTIGRAVITY",
            [CONFIG_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task1_contracts",
            CONTRACTS_ANTIGRAVITY,
            "Caller: future Task 4 qualification and Task 5 publication/replay. Effect: supplies immutable fold and ten-row qualification containers while enforcing exact float AUC shape/range, seed-major order, immutable count mappings, reason agreement, zero label queries, and null pre-run artifact hash without calculating the scientific gate.",
            "Exp468QualificationAntigravity",
            [CONTRACTS_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task1_config_test",
            CONFIG_TEST_ANTIGRAVITY,
            "Caller: focused Task 1 RED/GREEN suite. Effect: detects approved identity, change order, STFT geometry, Hann/backend separation, threshold, design digest, exact-type, and immutability drift.",
            "Exp468ConfigTestAntigravity",
            [CONFIG_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task1_contracts_test",
            CONTRACTS_TEST_ANTIGRAVITY,
            "Caller: focused Task 1 RED/GREEN suite. Effect: detects invalid AUC shape/type/range, seed/fold/count aliases, mutable or reordered mappings, premature Task 4 gate logic, inconsistent pass reasons, label use, and a non-null pre-run hash.",
            "Exp468ContractsTestAntigravity",
            [CONTRACTS_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_sources",
            SOURCES_ANTIGRAVITY,
            "Caller: focused Task 2 source test and future qualification runner. Effect: exposes the exact Exp466 TRAIN-only loader object, preserving MD5, 69x96 normal selection, immutability, and pre-open TEST-name rejection without adding I/O logic.",
            "load_ecg200_normal_train_antigravity",
            [SOURCES_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_interpolation",
            INTERPOLATION_ANTIGRAVITY,
            "Caller: focused Task 2 interpolation test and future Task 3-4 scoring. Effect: exposes the exact Exp466 96-to-140 interpolator object without changing grids, dtype, ownership, or interpolation logic.",
            "interpolate_to_140_antigravity",
            [INTERPOLATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_synthetic",
            SYNTHETIC_ANTIGRAVITY,
            "Caller: focused Task 2 synthetic test and future Task 4 evaluation. Effect: exposes the exact Exp466 four-change generator object, including period_change as the fourth frozen scope, without reimplementation.",
            "primary_rows_antigravity",
            [SYNTHETIC_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_predecessor",
            PREDECESSOR_ANTIGRAVITY,
            "Caller: focused Task 2 predecessor test and future Task 4 scoring. Effect: exposes the exact Exp466 Exp465-authority capture and C0 fold scorer objects, preserving normal and four-scope dtype/shape/bytes.",
            "score_c0_fold_antigravity",
            [PREDECESSOR_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_sources_test",
            SOURCES_TEST_ANTIGRAVITY,
            "Caller: Task 2 RED/GREEN suite. Effect: detects loader object replacement, approved TRAIN MD5 or 69x96 drift, and any TEST-name path that reaches file-open activity.",
            "SourcesAntigravityTest",
            [SOURCES_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_interpolation_test",
            INTERPOLATION_TEST_ANTIGRAVITY,
            "Caller: Task 2 RED/GREEN suite. Effect: detects any wrapper, copy, or replacement of the exact Exp466 interpolation function object.",
            "InterpolationAntigravityTest",
            [INTERPOLATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_synthetic_test",
            SYNTHETIC_TEST_ANTIGRAVITY,
            "Caller: Task 2 RED/GREEN suite. Effect: detects synthetic generator replacement and omission or reordering of level_shift, gradual_drift, amplitude_scaling, and period_change.",
            "SyntheticAntigravityTest",
            [SYNTHETIC_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_predecessor_test",
            PREDECESSOR_TEST_ANTIGRAVITY,
            "Caller: Task 2 RED/GREEN suite. Effect: detects authority/scorer object replacement and C0 normal or four-scope dtype, shape, order, array, or byte drift on the fixed TRAIN fold.",
            "PredecessorAntigravityTest",
            [PREDECESSOR_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "exp466_sources",
            EXP466_SOURCES_ANTIGRAVITY,
            "Caller: Exp468 sources adapter through an exact import. Effect: remains the reviewed TRAIN-only loader authority used unchanged by Exp468.",
            "load_ecg200_normal_train_antigravity",
            [SOURCES_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "exp466_interpolation",
            EXP466_INTERPOLATION_ANTIGRAVITY,
            "Caller: Exp468 interpolation adapter through an exact import. Effect: remains the reviewed 96-to-140 interpolation authority used unchanged by Exp468.",
            "interpolate_to_140_antigravity",
            [INTERPOLATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "exp466_synthetic",
            EXP466_SYNTHETIC_ANTIGRAVITY,
            "Caller: Exp468 synthetic adapter through an exact import. Effect: remains the reviewed four-change generation authority used unchanged by Exp468.",
            "primary_rows_antigravity",
            [SYNTHETIC_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "exp466_predecessor",
            EXP466_PREDECESSOR_ANTIGRAVITY,
            "Caller: Exp468 predecessor adapter through exact imports. Effect: remains the reviewed Exp465 authority capture and C0 fold score projection used unchanged by Exp468.",
            "score_c0_fold_antigravity",
            [PREDECESSOR_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "preregistration",
            PREREGISTRATION_ANTIGRAVITY,
            "Caller: governance and scientific reviewers. Effect: states implementation_task_1_complete, freezes the backend-neutral scientific definition, and records zero score/private/official/label activity plus null results and hashes.",
            "implementation_task_1_complete",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "runbook",
            RUNBOOK_ANTIGRAVITY,
            "Caller: Task 1 verifier and future Task 2 operator. Effect: provides focused read-only verification, states that no executable pipeline exists, and limits the next step to exact Exp466 imports and parity checks.",
            "Task 1 집중 검증",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "implementation_report",
            IMPLEMENTATION_REPORT_ANTIGRAVITY,
            "Caller: Task 1-2 closeout and later audit. Effect: records both RED/GREEN scopes, five exact re-exports, approved TRAIN and TEST-name boundaries, fixed-fold C0 parity, and zero/null official pre-run truth.",
            "Task 2 완료 범위",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard_artifact",
            DASHBOARD_ARTIFACT_ANTIGRAVITY,
            "Caller: portable dashboard generator. Effect: stores bounded source-backed Task 2 status, one focused parity fixture, zero official runs, null result truth, checklist state, and unchanged Exp466 comparison evidence.",
            "implementation_summary",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard",
            DASHBOARD_ANTIGRAVITY,
            "Caller: user-facing progress review. Effect: renders the source-backed Task 2 completion truth and points only to Task 3 STFT/acceleration work while preserving zero official execution, null results, and no TEST/FN claim.",
            "implementation_task_2_complete",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "map_builder",
            BUILDER_ANTIGRAVITY,
            "Caller: Task 2 closeout and every later Exp468 code-changing task preflight. Effect: generates the canonical HTML/JSON/lock triplet or validates Task 1-2 structure, evidence, exact-import edges, embedded JSON, dashboard truth, and all tracked fingerprints without writing in validate-only mode.",
            "build_triplet_antigravity",
            [BUILDER_ANTIGRAVITY],
        ),
    ]
    edges = [
        _edge_antigravity("task1_brief", "governs", "task1_config", TASK1_BRIEF_ANTIGRAVITY, "고정 설정"),
        _edge_antigravity("task1_brief", "governs", "task1_contracts", TASK1_BRIEF_ANTIGRAVITY, "필수 인터페이스"),
        _edge_antigravity("approved_design", "governs", "task1_config", DESIGN_ANTIGRAVITY, "고정 입력과 범위"),
        _edge_antigravity("implementation_plan", "governs", "task1_config", PLAN_ANTIGRAVITY, "고정 설정·결과 계약·문서 골격"),
        _edge_antigravity("implementation_plan", "governs", "task1_contracts", PLAN_ANTIGRAVITY, "Exp468QualificationAntigravity"),
        _edge_antigravity("task1_config", "governs", "task1_contracts", CONFIG_ANTIGRAVITY, "change_types"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", CONFIG_TEST_ANTIGRAVITY, "test_frozen_stft_authority_antigravity"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", CONTRACTS_TEST_ANTIGRAVITY, "test_fold_auc_tuples_are_exact_finite_and_positionally_bound_antigravity"),
        _edge_antigravity("task2_brief", "governs", "task2_sources", TASK2_BRIEF_ANTIGRAVITY, "정확한 re-export"),
        _edge_antigravity("task2_brief", "governs", "task2_interpolation", TASK2_BRIEF_ANTIGRAVITY, "정확한 re-export"),
        _edge_antigravity("task2_brief", "governs", "task2_synthetic", TASK2_BRIEF_ANTIGRAVITY, "정확한 re-export"),
        _edge_antigravity("task2_brief", "governs", "task2_predecessor", TASK2_BRIEF_ANTIGRAVITY, "정확한 re-export"),
        _edge_antigravity("implementation_plan", "governs", "task2_sources", PLAN_ANTIGRAVITY, "ECG200 입력·합성 변화·C0 재사용"),
        _edge_antigravity("implementation_plan", "governs", "task2_interpolation", PLAN_ANTIGRAVITY, "ECG200 입력·합성 변화·C0 재사용"),
        _edge_antigravity("implementation_plan", "governs", "task2_synthetic", PLAN_ANTIGRAVITY, "ECG200 입력·합성 변화·C0 재사용"),
        _edge_antigravity("implementation_plan", "governs", "task2_predecessor", PLAN_ANTIGRAVITY, "ECG200 입력·합성 변화·C0 재사용"),
        _edge_antigravity("task2_sources_test", "tests", "task2_sources", SOURCES_TEST_ANTIGRAVITY, "test_export_is_exact_exp466_loader_antigravity"),
        _edge_antigravity("task2_interpolation_test", "tests", "task2_interpolation", INTERPOLATION_TEST_ANTIGRAVITY, "test_export_is_exact_exp466_interpolator_antigravity"),
        _edge_antigravity("task2_synthetic_test", "tests", "task2_synthetic", SYNTHETIC_TEST_ANTIGRAVITY, "test_export_is_exact_exp466_primary_generator_antigravity"),
        _edge_antigravity("task2_predecessor_test", "tests", "task2_predecessor", PREDECESSOR_TEST_ANTIGRAVITY, "test_exports_are_exact_exp466_c0_authority_objects_antigravity"),
        _edge_antigravity("task2_sources", "imports_exact", "exp466_sources", SOURCES_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("task2_interpolation", "imports_exact", "exp466_interpolation", INTERPOLATION_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("task2_synthetic", "imports_exact", "exp466_synthetic", SYNTHETIC_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("task2_predecessor", "imports_exact", "exp466_predecessor", PREDECESSOR_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("task1_config", "documents", "preregistration", PREREGISTRATION_ANTIGRAVITY, "고정한 과학 정의"),
        _edge_antigravity("task1_contracts", "documents", "implementation_report", IMPLEMENTATION_REPORT_ANTIGRAVITY, "qualification은 seed-major 10개 fold"),
        _edge_antigravity("implementation_plan", "documents", "runbook", RUNBOOK_ANTIGRAVITY, "다음 허용 작업"),
        _edge_antigravity("implementation_plan", "documents", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "progress_checklist"),
        _edge_antigravity("implementation_report", "informs", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "exp468_task2_truth"),
        _edge_antigravity("dashboard_artifact", "generates", "dashboard", DASHBOARD_ARTIFACT_ANTIGRAVITY, '"surface": "dashboard"'),
        _edge_antigravity("map_builder", "verifies", "preregistration", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "runbook", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "implementation_report", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "dashboard_artifact", BUILDER_ANTIGRAVITY, "_validate_dashboard_antigravity"),
        _edge_antigravity("map_builder", "verifies", "dashboard", BUILDER_ANTIGRAVITY, "_validate_dashboard_antigravity"),
    ]
    return {
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent_commit": SOURCE_PARENT_COMMIT_ANTIGRAVITY,
        "baseline_triplet": list(BASELINE_TRIPLET_ANTIGRAVITY),
        "triplet_paths": list(TRIPLET_ANTIGRAVITY),
        "nodes": nodes,
        "edges": edges,
        "flows": [
            ["task1_brief", "task1_config", "task1_contracts"],
            ["approved_design", "task1_config", "task1_contracts"],
            ["task1_config_test", "task1_config"],
            ["task1_contracts_test", "task1_contracts"],
            ["task2_brief", "task2_sources", "exp466_sources"],
            ["task2_brief", "task2_interpolation", "exp466_interpolation"],
            ["task2_brief", "task2_synthetic", "exp466_synthetic"],
            ["task2_brief", "task2_predecessor", "exp466_predecessor"],
            ["task2_sources_test", "task2_sources"],
            ["task2_interpolation_test", "task2_interpolation"],
            ["task2_synthetic_test", "task2_synthetic"],
            ["task2_predecessor_test", "task2_predecessor"],
            ["implementation_plan", "preregistration", "runbook", "implementation_report"],
            ["implementation_report", "dashboard_artifact", "dashboard"],
            ["map_builder", "preregistration", "runbook", "implementation_report", "dashboard"],
        ],
        "task1_answers": {
            "callers": "Config is called by contracts/tests and future Tasks 2-4; contracts are called by future qualification/publication/replay; the plan calls Task 1 docs and dashboard state; the portable generator calls the source-backed artifact; Task 1 and later preflights call the map builder.",
            "effects": "Freezes backend-neutral scientific constants and immutable result shapes, records implementation_task_1_complete with zero score/private/official/label activity and null results/hashes, and leaves ECG access, scoring, qualification logic, TEST/FN, routing, and promotion untouched.",
            "tests": "test_config_antigravity and test_contracts_antigravity cover executable behavior; build_codemap_exp468_antigravity.py --validate-only covers evidence, required edges, HTML/JSON structure, embedded JSON, source-backed dashboard truth, and tracked fingerprints.",
        },
        "task2_answers": {
            "callers": "The four focused Task 2 tests call the Exp468 adapters now; future Task 3-4 scoring and qualification call the loader/interpolator/synthetic/C0 boundaries. Each Exp468 adapter directly imports its corresponding reviewed Exp466 authority.",
            "effects": "Preserves the exact Exp466 function objects and therefore the approved TRAIN MD5 and 69x96 boundary, 96-to-140 interpolation, four synthetic scopes including period_change, and C0 normal/primary dtype, shape, order, and bytes. It adds no loading, interpolation, synthetic, or scoring algorithm and leaves TEST/FN, qualification, routing, official results, and promotion untouched.",
            "tests": "test_sources_antigravity covers object identity, approved TRAIN MD5/69x96, and pre-open TEST-name rejection; test_interpolation_antigravity covers exact object identity; test_synthetic_antigravity covers exact identity and four-scope order; test_predecessor_antigravity covers both exact C0 objects and fixed-fold normal/four-scope byte parity. build_codemap_exp468_antigravity.py --validate-only covers the exact-import edges, dashboard truth, HTML/JSON structure, embedded JSON, and tracked fingerprints.",
        },
        "unknown_relationships": [],
    }


def build_lock_antigravity() -> dict[str, Any]:
    fingerprints = _tracked_fingerprints_antigravity()
    return {
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "source_parent_commit": SOURCE_PARENT_COMMIT_ANTIGRAVITY,
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "working_tree_dirty": False,
        "fingerprint_count": len(fingerprints),
        "tracked_fingerprints": fingerprints,
    }


def _render_html_antigravity(graph: dict[str, Any]) -> str:
    node_rows = "\n".join(
        "<tr><td><code>{}</code></td><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
            html_module.escape(node["id"]),
            html_module.escape(node["path"]),
            html_module.escape(node["role"]),
            html_module.escape(", ".join(node["tests"])),
        )
        for node in graph["nodes"]
    )
    edge_rows = "\n".join(
        "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td><td><code>{}</code></td></tr>".format(
            html_module.escape(edge["from"]),
            html_module.escape(edge["type"]),
            html_module.escape(edge["to"]),
            html_module.escape(f"{edge['evidence']['path']}::{edge['evidence']['symbol']}"),
        )
        for edge in graph["edges"]
    )
    answers = graph["task2_answers"]
    embedded = html_module.escape(
        json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp468 Task 2 코드맵</title>
<style>body{{max-width:1240px;margin:2rem auto;padding:0 1rem;font:15px/1.65 system-ui;color:#172033}}table{{width:100%;border-collapse:collapse;margin:1rem 0}}th,td{{border:1px solid #aab3bf;padding:.48rem;text-align:left;vertical-align:top}}code,pre{{background:#eef2f7;padding:.1rem .3rem;overflow:auto}}.box{{background:#f5f7fa;border-left:6px solid #2457a6;padding:1rem}}</style></head>
<body><main><h1>Exp468 Task 2 코드맵</h1>
<div class="box"><p><strong>상태:</strong> <code>implementation_task_2_complete</code>. 이 지도는 Task 1 설정·계약과 Task 2 exact adapters·검사·Exp466 import authority·truth 문서·source-backed 대시보드의 현재 경계를 기록한다.</p></div>
<h2>세 질문</h2><table><tr><th>무엇이 호출하는가?</th><td>{html_module.escape(answers['callers'])}</td></tr><tr><th>무엇에 영향을 주는가?</th><td>{html_module.escape(answers['effects'])}</td></tr><tr><th>어떤 검사가 덮는가?</th><td>{html_module.escape(answers['tests'])}</td></tr></table>
<h2>노드</h2><table><thead><tr><th>ID</th><th>파일</th><th>호출자와 영향</th><th>검사</th></tr></thead><tbody>{node_rows}</tbody></table>
<h2>관계</h2><table><thead><tr><th>시작</th><th>관계</th><th>도착</th><th>근거</th></tr></thead><tbody>{edge_rows}</tbody></table>
<h2>기계 판독 JSON</h2><pre id="codemap-json-antigravity">{embedded}</pre>
</main></body></html>"""


def _validate_dashboard_antigravity() -> None:
    artifact = json.loads(
        (ROOT_ANTIGRAVITY / DASHBOARD_ARTIFACT_ANTIGRAVITY).read_text(encoding="utf-8"),
    )
    if artifact.get("surface") != "dashboard" or artifact.get("manifest", {}).get("surface") != "dashboard":
        raise ValueError("dashboard artifact surface differs")
    if artifact.get("manifest", {}).get("sources") != artifact.get("sources"):
        raise ValueError("dashboard source inventories differ")
    summary = artifact.get("snapshot", {}).get("datasets", {}).get("implementation_summary")
    expected = [{
        "status": "implementation_task_2_complete",
        "score_calculations": 1,
        "score_calculation_scope": "focused_parity_test_only",
        "official_runs": 0,
        "official_result": None,
        "private_result": None,
        "result_hash": None,
    }]
    if summary != expected:
        raise ValueError("dashboard Task 2 implementation truth differs")
    source_ids = {item.get("id") for item in artifact["sources"]}
    if "exp468_task2_truth" not in source_ids:
        raise ValueError("dashboard Task 2 source is missing")
    rows = artifact["snapshot"]["datasets"].get("progress_checklist", [])
    if not any(
        row.get("task") == "Task 2 입력·합성 변화·C0 exact import와 parity"
        and row.get("checked") == "☑"
        and row.get("status") == "완료"
        for row in rows
    ):
        raise ValueError("dashboard Task 2 checklist is not complete")
    rendered = _validated_html_text_antigravity(ROOT_ANTIGRAVITY / DASHBOARD_ANTIGRAVITY)
    for required in (
        "implementation_task_2_complete",
        "exp468_task2_truth",
        "Task 2 입력·합성 변화·C0 exact import와 parity",
    ):
        if required not in rendered:
            raise ValueError(f"dashboard HTML is missing {required}")


def validate_map_antigravity(
    graph: dict[str, Any],
    lock: dict[str, Any],
    rendered: str,
    *,
    compare_files: bool,
) -> None:
    if graph.get("schema") != GRAPH_SCHEMA_ANTIGRAVITY:
        raise ValueError("graph schema differs")
    if graph.get("source_parent_commit") != SOURCE_PARENT_COMMIT_ANTIGRAVITY:
        raise ValueError("graph source parent differs")
    if graph.get("baseline_triplet") != list(BASELINE_TRIPLET_ANTIGRAVITY):
        raise ValueError("graph baseline triplet differs")
    if graph.get("triplet_paths") != list(TRIPLET_ANTIGRAVITY):
        raise ValueError("graph triplet paths differ")
    if graph.get("unknown_relationships") != []:
        raise ValueError("unknown relationships must remain empty")
    if set(graph.get("task1_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 1 caller/effect/test answers are incomplete")
    if set(graph.get("task2_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 2 caller/effect/test answers are incomplete")
    ids: set[str] = set()
    for node in graph.get("nodes", []):
        if not isinstance(node, dict) or not node.get("id") or not node.get("path") or not node.get("role"):
            raise ValueError("invalid node")
        ids.add(node["id"])
        if not node.get("tests") or any(path not in TRACKED_PATHS_ANTIGRAVITY for path in node["tests"]):
            raise ValueError("invalid node tests")
        for evidence in node.get("evidence", []):
            if set(evidence) != {"path", "symbol"} or not _evidence_resolves_antigravity(
                evidence["path"], evidence["symbol"],
            ):
                raise ValueError(f"unresolved node evidence: {evidence}")
    required_nodes = {
        "task1_brief", "task2_brief", "approved_design", "implementation_plan", "task1_config", "task1_contracts",
        "task1_config_test", "task1_contracts_test", "preregistration", "runbook",
        "implementation_report", "dashboard_artifact", "dashboard", "map_builder",
        "task2_sources", "task2_interpolation", "task2_synthetic", "task2_predecessor",
        "task2_sources_test", "task2_interpolation_test", "task2_synthetic_test",
        "task2_predecessor_test", "exp466_sources", "exp466_interpolation",
        "exp466_synthetic", "exp466_predecessor",
    }
    if ids != required_nodes:
        raise ValueError("Task 1-2 nodes differ")
    allowed_types = {"governs", "tests", "imports_exact", "documents", "informs", "generates", "verifies"}
    edge_keys: set[tuple[str, str, str]] = set()
    for edge in graph.get("edges", []):
        if edge.get("from") not in ids or edge.get("to") not in ids or edge.get("type") not in allowed_types:
            raise ValueError(f"invalid edge: {edge}")
        evidence = edge.get("evidence", {})
        if set(evidence) != {"path", "symbol"} or not _evidence_resolves_antigravity(
            evidence["path"], evidence["symbol"],
        ):
            raise ValueError(f"unresolved edge evidence: {evidence}")
        edge_keys.add((edge["from"], edge["type"], edge["to"]))
    required_edges = {
        ("task1_brief", "governs", "task1_config"),
        ("task1_brief", "governs", "task1_contracts"),
        ("approved_design", "governs", "task1_config"),
        ("implementation_plan", "governs", "task1_config"),
        ("implementation_plan", "governs", "task1_contracts"),
        ("task1_config", "governs", "task1_contracts"),
        ("task1_config_test", "tests", "task1_config"),
        ("task1_contracts_test", "tests", "task1_contracts"),
        ("task2_brief", "governs", "task2_sources"),
        ("task2_brief", "governs", "task2_interpolation"),
        ("task2_brief", "governs", "task2_synthetic"),
        ("task2_brief", "governs", "task2_predecessor"),
        ("implementation_plan", "governs", "task2_sources"),
        ("implementation_plan", "governs", "task2_interpolation"),
        ("implementation_plan", "governs", "task2_synthetic"),
        ("implementation_plan", "governs", "task2_predecessor"),
        ("task2_sources_test", "tests", "task2_sources"),
        ("task2_interpolation_test", "tests", "task2_interpolation"),
        ("task2_synthetic_test", "tests", "task2_synthetic"),
        ("task2_predecessor_test", "tests", "task2_predecessor"),
        ("task2_sources", "imports_exact", "exp466_sources"),
        ("task2_interpolation", "imports_exact", "exp466_interpolation"),
        ("task2_synthetic", "imports_exact", "exp466_synthetic"),
        ("task2_predecessor", "imports_exact", "exp466_predecessor"),
        ("implementation_plan", "documents", "dashboard_artifact"),
        ("implementation_report", "informs", "dashboard_artifact"),
        ("dashboard_artifact", "generates", "dashboard"),
        ("map_builder", "verifies", "preregistration"),
        ("map_builder", "verifies", "runbook"),
        ("map_builder", "verifies", "implementation_report"),
        ("map_builder", "verifies", "dashboard_artifact"),
        ("map_builder", "verifies", "dashboard"),
    }
    if not required_edges.issubset(edge_keys):
        raise ValueError("Task 1-2 required edges are incomplete")
    if lock.get("schema") != LOCK_SCHEMA_ANTIGRAVITY:
        raise ValueError("lock schema differs")
    if lock.get("source_parent_commit") != graph.get("source_parent_commit"):
        raise ValueError("graph/lock source parent differs")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("lock represented paths differ")
    if lock.get("working_tree_dirty") is not False:
        raise ValueError("lock must record a closed generation boundary")
    fingerprints = _tracked_fingerprints_antigravity()
    if lock.get("fingerprint_count") != len(TRACKED_PATHS_ANTIGRAVITY):
        raise ValueError("lock fingerprint count differs")
    if lock.get("tracked_fingerprints") != fingerprints:
        raise ValueError("tracked fingerprint drift")
    parser = _StrictHtmlAntigravity()
    parser.feed(rendered)
    parser.close()
    if parser.stack or not {"html", "head", "body", "main", "pre"}.issubset(parser.seen):
        raise ValueError("generated code-map HTML structure differs")
    embedded = re.search(
        r'<pre id="codemap-json-antigravity">(.*?)</pre>',
        rendered,
        flags=re.DOTALL,
    )
    if embedded is None:
        raise ValueError("embedded code-map JSON missing")
    embedded_graph = json.loads(html_module.unescape(embedded.group(1)))
    if _canonical_bytes_antigravity(embedded_graph) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON graph mismatch")
    for relative in (
        DESIGN_ANTIGRAVITY,
        PLAN_ANTIGRAVITY,
        TASK1_BRIEF_ANTIGRAVITY,
        TASK2_BRIEF_ANTIGRAVITY,
        PREREGISTRATION_ANTIGRAVITY,
        RUNBOOK_ANTIGRAVITY,
        IMPLEMENTATION_REPORT_ANTIGRAVITY,
    ):
        _validated_html_text_antigravity(ROOT_ANTIGRAVITY / relative)
    _validate_dashboard_antigravity()
    if compare_files:
        if JSON_PATH_ANTIGRAVITY.read_bytes() != _canonical_bytes_antigravity(graph):
            raise ValueError("code-map JSON is not canonical")
        if LOCK_PATH_ANTIGRAVITY.read_bytes() != _canonical_bytes_antigravity(lock):
            raise ValueError("code-map lock is not canonical")


def build_triplet_antigravity(*, validate_only: bool = False) -> None:
    if validate_only:
        graph = json.loads(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        lock = json.loads(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        rendered = HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
        validate_map_antigravity(graph, lock, rendered, compare_files=True)
        return
    graph = build_graph_antigravity()
    lock = build_lock_antigravity()
    rendered = _render_html_antigravity(graph)
    validate_map_antigravity(graph, lock, rendered, compare_files=False)
    HTML_PATH_ANTIGRAVITY.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH_ANTIGRAVITY.write_text(rendered, encoding="utf-8")
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph))
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock))


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    build_triplet_antigravity(validate_only=args.validate_only)


if __name__ == "__main__":
    main_antigravity()
