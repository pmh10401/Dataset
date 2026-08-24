#!/usr/bin/env python3
"""Build and validate the versioned Exp471 Task 2 code-map triplet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import html.parser
import json
from pathlib import Path
import re
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
PACKAGE_ANTIGRAVITY = "experiments/exp471_ecg200_acf_peak_lag_shift_numba_antigravity"
PREDECESSOR_ANTIGRAVITY = "experiments/exp470_ecg200_spectral_entropy_numba_antigravity"
SDD_ANTIGRAVITY = ".superpowers/sdd/2026-08-24-exp471-ecg200-acf-peak-lag-shift-numba-antigravity.html"


def _package_file_antigravity(name: str) -> str:
    return PACKAGE_ANTIGRAVITY + "/" + name


def _package_test_antigravity(name: str) -> str:
    return PACKAGE_ANTIGRAVITY + "/tests/" + name


def _predecessor_file_antigravity(name: str) -> str:
    return PREDECESSOR_ANTIGRAVITY + "/" + name


CONFIG_ANTIGRAVITY = _package_file_antigravity("config_antigravity.py")
SOURCES_ANTIGRAVITY = _package_file_antigravity("sources_antigravity.py")
INTERPOLATION_ANTIGRAVITY = _package_file_antigravity("interpolation_antigravity.py")
SYNTHETIC_ANTIGRAVITY = _package_file_antigravity("synthetic_antigravity.py")
C0_AUTHORITY_ANTIGRAVITY = _package_file_antigravity("predecessor_antigravity.py")
CONTRACTS_ANTIGRAVITY = _package_file_antigravity("contracts_antigravity.py")
ACF_PEAK_LAG_ANTIGRAVITY = _package_file_antigravity("acf_peak_lag_antigravity.py")
PYTHON_REFERENCE_ANTIGRAVITY = _package_file_antigravity("python_reference_antigravity.py")
CONFIG_TEST_ANTIGRAVITY = _package_test_antigravity("test_config_antigravity.py")
CONTRACTS_TEST_ANTIGRAVITY = _package_test_antigravity("test_contracts_antigravity.py")
AUTHORITY_TEST_ANTIGRAVITY = _package_test_antigravity("test_predecessor_antigravity.py")
ACF_TEST_ANTIGRAVITY = _package_test_antigravity("test_acf_peak_lag_antigravity.py")
PYTHON_REFERENCE_TEST_ANTIGRAVITY = _package_test_antigravity(
    "test_python_reference_antigravity.py"
)
ACCELERATION_ANTIGRAVITY = _package_file_antigravity("acceleration_antigravity.py")
ACCELERATION_TEST_ANTIGRAVITY = _package_test_antigravity(
    "test_acceleration_antigravity.py"
)
ACCELERATION_ASSESSMENT_ANTIGRAVITY = (
    "scratch/exp471_acceleration_assessment_antigravity.json"
)
BUILDER_ANTIGRAVITY = "scratch/build_codemap_exp471_antigravity.py"
IMPLEMENTATION_REPORT_ANTIGRAVITY = "docs/EXP471_IMPLEMENTATION_REPORT_antigravity.html"
TASK_BRIEF_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-1-brief-antigravity.html"
TASK_REPORT_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-1-report-antigravity.html"
TASK2_BRIEF_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-2-brief-antigravity.html"
TASK2_REPORT_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-2-report-antigravity.html"
TASK3_BRIEF_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-3-brief-antigravity.html"
TASK3_REPORT_ANTIGRAVITY = SDD_ANTIGRAVITY + "/task-3-report-antigravity.html"

FUTURE_DECLARED_PATHS_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-24-exp471-ecg200-acf-peak-lag-shift-numba-antigravity.html",
    "docs/superpowers/specs/2026-08-24-exp471-ecg200-acf-peak-lag-shift-numba-antigravity-design.html",
    "scratch/exp471_dashboard_artifact_antigravity.json",
    "docs/EXP471_DASHBOARD_antigravity.html",
    _package_file_antigravity("scoring_antigravity.py"),
    _package_file_antigravity("qualification_antigravity.py"),
    _package_file_antigravity("evaluation_antigravity.py"),
    _package_file_antigravity("artifacts_antigravity.py"),
    _package_file_antigravity("replay_antigravity.py"),
    _package_file_antigravity("dashboard_export_antigravity.py"),
    _package_test_antigravity("test_scoring_antigravity.py"),
    _package_test_antigravity("test_qualification_antigravity.py"),
    _package_test_antigravity("test_evaluation_antigravity.py"),
    _package_test_antigravity("test_artifacts_antigravity.py"),
    _package_test_antigravity("test_replay_antigravity.py"),
    _package_test_antigravity("test_dashboard_export_antigravity.py"),
    _package_file_antigravity("cli_antigravity.py"),
    "scratch/run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity.py",
    _package_test_antigravity("test_cli_antigravity.py"),
    _package_test_antigravity("test_pipeline_antigravity.py"),
    "docs/EXP471_RUNBOOK_antigravity.html",
)

TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp471_antigravity.html",
    "docs/codemap/codemap_exp471_antigravity.json",
    "docs/codemap/codemap_exp471_antigravity.lock",
)
EXISTING_REPRESENTED_PATHS_ANTIGRAVITY = (
    TASK_BRIEF_ANTIGRAVITY,
    TASK_REPORT_ANTIGRAVITY,
    TASK2_BRIEF_ANTIGRAVITY,
    TASK2_REPORT_ANTIGRAVITY,
    TASK3_BRIEF_ANTIGRAVITY,
    TASK3_REPORT_ANTIGRAVITY,
    CONFIG_ANTIGRAVITY,
    SOURCES_ANTIGRAVITY,
    INTERPOLATION_ANTIGRAVITY,
    SYNTHETIC_ANTIGRAVITY,
    C0_AUTHORITY_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    ACF_PEAK_LAG_ANTIGRAVITY,
    PYTHON_REFERENCE_ANTIGRAVITY,
    ACCELERATION_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    AUTHORITY_TEST_ANTIGRAVITY,
    ACF_TEST_ANTIGRAVITY,
    PYTHON_REFERENCE_TEST_ANTIGRAVITY,
    ACCELERATION_TEST_ANTIGRAVITY,
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_ANTIGRAVITY,
) + TRIPLET_ANTIGRAVITY
EXISTING_PRE_TRIPLET_PATHS_ANTIGRAVITY = tuple(
    relative for relative in EXISTING_REPRESENTED_PATHS_ANTIGRAVITY
    if relative not in TRIPLET_ANTIGRAVITY
)
HASHED_SOURCE_PATHS_ANTIGRAVITY = (
    TASK2_BRIEF_ANTIGRAVITY,
    TASK2_REPORT_ANTIGRAVITY,
    TASK3_BRIEF_ANTIGRAVITY,
    TASK3_REPORT_ANTIGRAVITY,
    CONFIG_ANTIGRAVITY,
    SOURCES_ANTIGRAVITY,
    INTERPOLATION_ANTIGRAVITY,
    SYNTHETIC_ANTIGRAVITY,
    C0_AUTHORITY_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    ACF_PEAK_LAG_ANTIGRAVITY,
    PYTHON_REFERENCE_ANTIGRAVITY,
    ACCELERATION_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    AUTHORITY_TEST_ANTIGRAVITY,
    ACF_TEST_ANTIGRAVITY,
    PYTHON_REFERENCE_TEST_ANTIGRAVITY,
    ACCELERATION_TEST_ANTIGRAVITY,
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
    _predecessor_file_antigravity("sources_antigravity.py"),
    _predecessor_file_antigravity("interpolation_antigravity.py"),
    _predecessor_file_antigravity("synthetic_antigravity.py"),
    _predecessor_file_antigravity("predecessor_antigravity.py"),
    _predecessor_file_antigravity("contracts_antigravity.py"),
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp471_task3_runtime_assessment_codemap_v3"
LOCK_SCHEMA_ANTIGRAVITY = "exp471_task3_runtime_assessment_lock_v3"
_VOID_TAGS_ANTIGRAVITY = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}


def _canonical_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


def _fingerprints_antigravity() -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for relative in HASHED_SOURCE_PATHS_ANTIGRAVITY:
        payload = (ROOT_ANTIGRAVITY / relative).read_bytes()
        fingerprints.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    return fingerprints


def _node_antigravity(
    node_id: str,
    path: str,
    caller: str,
    effect: str,
    tests: tuple[str, ...],
    symbol: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "caller": caller,
        "effect": effect,
        "tests": list(tests),
        "evidence": {"path": path, "symbol": symbol},
    }


def _edge_antigravity(source: str, edge_type: str, target: str, path: str, symbol: str) -> dict[str, Any]:
    return {
        "from": source,
        "type": edge_type,
        "to": target,
        "evidence": {"path": path, "symbol": symbol},
    }


def _contains_symbol_antigravity(path: str, symbol: str) -> bool:
    return symbol in (ROOT_ANTIGRAVITY / path).read_text(encoding="utf-8")


def _imports_exact_authority_antigravity(source_path: str, target_path: str, symbol: str) -> bool:
    source = (ROOT_ANTIGRAVITY / source_path).read_text(encoding="utf-8")
    expected_module = target_path.removesuffix(".py").replace("/", ".")
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == expected_module
        and any(alias.name == symbol and alias.asname is None for alias in node.names)
        for node in ast.walk(ast.parse(source))
    )


def build_graph_antigravity() -> dict[str, Any]:
    nodes = [
        _node_antigravity(
            "task1_brief", TASK_BRIEF_ANTIGRAVITY, "Task 1B implementer",
            "Bounds Task 1 to exact Exp470 authority, immutable contracts, a scoped code map, and no experiment execution.",
            (), "Task 1",
        ),
        _node_antigravity(
            "config", CONFIG_ANTIGRAVITY, "contracts and later Exp471 stages",
            "Freezes seeds, folds, ECG200 geometry, windows, ACF lag bounds, method order, change order, and required speedup.",
            (CONFIG_TEST_ANTIGRAVITY,), "Exp471ConfigAntigravity",
        ),
        _node_antigravity(
            "sources", SOURCES_ANTIGRAVITY, "later input stage",
            "Re-exports only the exact Exp470 normal TRAIN loader object.",
            (AUTHORITY_TEST_ANTIGRAVITY,), "load_ecg200_normal_train_antigravity",
        ),
        _node_antigravity(
            "interpolation", INTERPOLATION_ANTIGRAVITY, "later input stage",
            "Re-exports only the exact Exp470 96-to-140 interpolator object.",
            (AUTHORITY_TEST_ANTIGRAVITY,), "interpolate_to_140_antigravity",
        ),
        _node_antigravity(
            "synthetic", SYNTHETIC_ANTIGRAVITY, "later qualification stage",
            "Re-exports only the exact Exp470 primary synthetic-row object.",
            (AUTHORITY_TEST_ANTIGRAVITY,), "primary_rows_antigravity",
        ),
        _node_antigravity(
            "c0_authority", C0_AUTHORITY_ANTIGRAVITY, "later C0 comparison",
            "Re-exports only exact Exp470 C0 authority-capture and fold-score objects.",
            (AUTHORITY_TEST_ANTIGRAVITY,), "score_c0_fold_antigravity",
        ),
        _node_antigravity(
            "contracts", CONTRACTS_ANTIGRAVITY, "Task 2 scorer and later qualification",
            "Defines immutable pair counts, seed-major fold results, finite AUC evidence, zero label queries, optional SHA-256, and failure reasons.",
            (CONTRACTS_TEST_ANTIGRAVITY,), "Exp471FoldResultAntigravity",
        ),
        _node_antigravity(
            "task2_brief", TASK2_BRIEF_ANTIGRAVITY, "Task 2 implementer",
            "Bounds Task 2 to the frozen Python reference formula and float64 Numba kernels with no evaluation or private/official execution.",
            (), "Task 2",
        ),
        _node_antigravity(
            "acf_peak_lag", ACF_PEAK_LAG_ANTIGRAVITY, "Task 3 acceleration assessment and later fold scoring",
            "Computes frozen median/MAD windows, biased ACF, smallest-tie peak selection, integer shifts, LOO raw ranks, strict confirmation, and immutable result arrays.",
            (ACF_TEST_ANTIGRAVITY,), "ACFPeakLagScoresAntigravity",
        ),
        _node_antigravity(
            "python_reference", PYTHON_REFERENCE_ANTIGRAVITY, "Task 2 parity tests",
            "Provides the list, math.fsum, and sorted calculation authority with the same result shape as the Numba backend.",
            (PYTHON_REFERENCE_TEST_ANTIGRAVITY,), "score_acf_peak_lag_python_reference_antigravity",
        ),
        _node_antigravity(
            "task3_brief", TASK3_BRIEF_ANTIGRAVITY, "Task 3 implementer",
            "Bounds Task 3 to public-fixture Python parity, Numba determinism and timing, MLX eligibility evidence, and no experiment execution.",
            (), "Task 3",
        ),
        _node_antigravity(
            "acceleration", ACCELERATION_ANTIGRAVITY, "later fold scoring and runtime governance",
            "Measures fixed-fixture Python reference parity, Numba full-output bytes determinism, cold/warm timing, MLX float64/parity/speed gates, and records a strict manifest.",
            (ACCELERATION_TEST_ANTIGRAVITY,), "assess_runtime_antigravity",
        ),
        _node_antigravity(
            "acceleration_assessment", ACCELERATION_ASSESSMENT_ANTIGRAVITY, "Task 3 evidence readers",
            "Records the public fixture shape, environment versions, Numba timings/parity, MLX eligibility reason, selected backend, and zero restricted-run counters.",
            (), "selected_backend",
        ),
        _node_antigravity(
            "config_test", CONFIG_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers frozen values, exact runtime types, tuple order, mutation rejection, and type/order drift rejection.",
            (), "Exp471ConfigTestAntigravity",
        ),
        _node_antigravity(
            "contracts_test", CONTRACTS_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers ordered mappings, ten seed-major rows, strict counts and bools, bounded AUCs, governance, digest, and failure tuples.",
            (), "Exp471ContractsTestAntigravity",
        ),
        _node_antigravity(
            "authority_test", AUTHORITY_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers object identity for five direct Exp470 authority exports.",
            (), "Exp471PredecessorTestAntigravity",
        ),
        _node_antigravity(
            "acf_test", ACF_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers known peak shift, tie/sentinel rules, scale invariance, input rejection, immutability, dispatcher signatures, and Numba scoring.",
            (), "Exp471AcfPeakLagNumbaTestAntigravity",
        ),
        _node_antigravity(
            "python_reference_test", PYTHON_REFERENCE_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers exact peaks, presence, shifts, ranks, confirmations, raw tolerance, and immutable output parity against Numba.",
            (), "Exp471PythonReferenceTestAntigravity",
        ),
        _node_antigravity(
            "acceleration_test", ACCELERATION_TEST_ANTIGRAVITY, "focused unittest command",
            "Covers frozen fixture authority, full MLX CPU parity, exact lag boundaries, full-output determinism bytes, strict arguments, manifest fields, and GPU fallback.",
            (), "Exp471AccelerationTestAntigravity",
        ),
        _node_antigravity(
            "implementation_report", IMPLEMENTATION_REPORT_ANTIGRAVITY, "Task 1–3 closeout",
            "Records Task 1–3 scope, parity evidence, code-map relations, changed files, dashboard checks, and pending boundaries.",
            (), "Exp471 구현 보고",
        ),
        _node_antigravity(
            "task3_report", TASK3_REPORT_ANTIGRAVITY, "Task 3 parent handoff",
            "Preserves Task 3 RED import failure, two-pass 12-test GREEN evidence, Numba/MLX eligibility outcome, and execution boundary.",
            (), "Task 3",
        ),
        _node_antigravity(
            "task_report", TASK_REPORT_ANTIGRAVITY, "Task 1 parent handoff",
            "Preserves Task 1A RED/GREEN, path reconciliation, focused tests, baseline, self-review, and execution boundary.",
            (), "Exp471 Task 1A",
        ),
        _node_antigravity(
            "task2_report", TASK2_REPORT_ANTIGRAVITY, "Task 2 parent handoff",
            "Preserves Task 2 RED import failures, intermediate fixes, two-pass 14-test evidence, regression, code map update, and execution boundary.",
            (), "Task 2",
        ),
        _node_antigravity(
            "map_builder", BUILDER_ANTIGRAVITY, "Task 2 generation and validation",
            "Writes or validates only this HTML/JSON/lock triplet, existing represented paths, declared future paths, hashes, relations, and AST-checked authority imports.",
            (), "build_graph_antigravity",
        ),
        _node_antigravity(
            "exp470_load_ecg200_normal_train", _predecessor_file_antigravity("sources_antigravity.py"), "Exp471 sources export",
            "Provides the reviewed Exp470 normal TRAIN loader object.", (),
            "load_ecg200_normal_train_antigravity",
        ),
        _node_antigravity(
            "exp470_interpolate_to_140", _predecessor_file_antigravity("interpolation_antigravity.py"), "Exp471 interpolation export",
            "Provides the reviewed Exp470 interpolator object.", (),
            "interpolate_to_140_antigravity",
        ),
        _node_antigravity(
            "exp470_primary_rows", _predecessor_file_antigravity("synthetic_antigravity.py"), "Exp471 synthetic export",
            "Provides the reviewed Exp470 primary synthetic-row object.", (),
            "primary_rows_antigravity",
        ),
        _node_antigravity(
            "exp470_capture_exp465_authority", _predecessor_file_antigravity("predecessor_antigravity.py"), "Exp471 predecessor export",
            "Provides the reviewed Exp470 C0 capture object.", (),
            "capture_exp465_authority_antigravity",
        ),
        _node_antigravity(
            "exp470_score_c0_fold", _predecessor_file_antigravity("predecessor_antigravity.py"), "Exp471 predecessor export",
            "Provides the reviewed Exp470 C0 scorer object.", (),
            "score_c0_fold_antigravity",
        ),
        _node_antigravity(
            "exp470_config_contract", _predecessor_file_antigravity("config_antigravity.py"), "predecessor map review",
            "Preserves the predecessor config used by exact C0 authority.", (),
            "FROZEN_CONFIG_ANTIGRAVITY",
        ),
        _node_antigravity(
            "exp470_contracts_pattern", _predecessor_file_antigravity("contracts_antigravity.py"), "predecessor map review",
            "Preserves the immutable pattern adapted to Exp471 C12 score fields.", (),
            "Exp470QualificationAntigravity",
        ),
    ]
    edges = [
        _edge_antigravity("contracts", "calls", "config", CONTRACTS_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task2_brief", "documents", "acf_peak_lag", TASK2_BRIEF_ANTIGRAVITY, "score_acf_peak_lag_numba_antigravity"),
        _edge_antigravity("task2_brief", "documents", "python_reference", TASK2_BRIEF_ANTIGRAVITY, "python_reference_antigravity.py"),
        _edge_antigravity("python_reference", "calls", "acf_peak_lag", PYTHON_REFERENCE_ANTIGRAVITY, "ACFPeakLagScoresAntigravity"),
        _edge_antigravity("task3_brief", "documents", "acceleration", TASK3_BRIEF_ANTIGRAVITY, "assess_runtime_antigravity"),
        _edge_antigravity("acceleration", "calls", "python_reference", ACCELERATION_ANTIGRAVITY, "score_acf_peak_lag_python_reference_antigravity"),
        _edge_antigravity("acceleration", "writes", "acceleration_assessment", ACCELERATION_ANTIGRAVITY, "manifest_payload_antigravity"),
        _edge_antigravity("acceleration_test", "tests", "acceleration", ACCELERATION_TEST_ANTIGRAVITY, "assess_runtime_antigravity"),
        _edge_antigravity("python_reference_test", "tests", "python_reference", PYTHON_REFERENCE_TEST_ANTIGRAVITY, "score_acf_peak_lag_python_reference_antigravity"),
        _edge_antigravity("python_reference_test", "tests", "acf_peak_lag", PYTHON_REFERENCE_TEST_ANTIGRAVITY, "score_acf_peak_lag_numba_antigravity"),
        _edge_antigravity("acf_test", "tests", "acf_peak_lag", ACF_TEST_ANTIGRAVITY, "ACFPeakLagScoresAntigravity"),
        _edge_antigravity("config_test", "tests", "config", CONFIG_TEST_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("contracts_test", "tests", "contracts", CONTRACTS_TEST_ANTIGRAVITY, "Exp471QualificationAntigravity"),
        _edge_antigravity("authority_test", "tests", "sources", AUTHORITY_TEST_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("authority_test", "tests", "interpolation", AUTHORITY_TEST_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("authority_test", "tests", "synthetic", AUTHORITY_TEST_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("authority_test", "tests", "c0_authority", AUTHORITY_TEST_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("sources", "authority", "exp470_load_ecg200_normal_train", SOURCES_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("interpolation", "authority", "exp470_interpolate_to_140", INTERPOLATION_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("synthetic", "authority", "exp470_primary_rows", SYNTHETIC_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("c0_authority", "authority", "exp470_capture_exp465_authority", C0_AUTHORITY_ANTIGRAVITY, "capture_exp465_authority_antigravity"),
        _edge_antigravity("c0_authority", "authority", "exp470_score_c0_fold", C0_AUTHORITY_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("map_builder", "documents", "config", BUILDER_ANTIGRAVITY, "Exp471ConfigAntigravity"),
        _edge_antigravity("map_builder", "documents", "sources", BUILDER_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("map_builder", "documents", "interpolation", BUILDER_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("map_builder", "documents", "synthetic", BUILDER_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("map_builder", "documents", "c0_authority", BUILDER_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("map_builder", "documents", "contracts", BUILDER_ANTIGRAVITY, "Exp471FoldResultAntigravity"),
        _edge_antigravity("map_builder", "documents", "acf_peak_lag", BUILDER_ANTIGRAVITY, "ACFPeakLagScoresAntigravity"),
        _edge_antigravity("map_builder", "documents", "python_reference", BUILDER_ANTIGRAVITY, "score_acf_peak_lag_python_reference_antigravity"),
        _edge_antigravity("map_builder", "documents", "acceleration", BUILDER_ANTIGRAVITY, "assess_runtime_antigravity"),
        _edge_antigravity("map_builder", "documents", "acceleration_assessment", BUILDER_ANTIGRAVITY, "ACCELERATION_ASSESSMENT_ANTIGRAVITY"),
        _edge_antigravity("map_builder", "reviews", "exp470_config_contract", BUILDER_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("map_builder", "reviews", "exp470_contracts_pattern", BUILDER_ANTIGRAVITY, "Exp470QualificationAntigravity"),
    ]
    return {
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "represented_paths": list(EXISTING_REPRESENTED_PATHS_ANTIGRAVITY + FUTURE_DECLARED_PATHS_ANTIGRAVITY),
        "future_declared_paths": list(FUTURE_DECLARED_PATHS_ANTIGRAVITY),
        "source_fingerprints": _fingerprints_antigravity(),
        "nodes": nodes,
        "edges": edges,
    }


def _render_html_antigravity(graph: dict[str, Any]) -> str:
    rows = "".join(
        "<tr><td>" + html.escape(node["id"]) + "</td><td>" + html.escape(node["path"])
        + "</td><td>" + html.escape(node["caller"]) + "</td><td>" + html.escape(node["effect"]) + "</td></tr>"
        for node in graph["nodes"]
    )
    embedded = _canonical_bytes_antigravity(graph).decode("utf-8").rstrip()
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        "<title>Exp471 code map</title></head><body><main>"
        "<h1>Exp471 Task 2 Code Map</h1>"
        "<p>Versioned authority, ACF scorer, and runtime-assessment map; future Task 4–6 files are explicitly pending.</p>"
        "<table><thead><tr><th>ID</th><th>Path</th><th>Caller</th><th>Effect</th></tr></thead><tbody>"
        + rows + "</tbody></table>"
        '<script id="exp471-codemap-json-antigravity" type="application/json">'
        + embedded + "</script></main></body></html>\n"
    )


def _lock_antigravity(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "graph_schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "graph_sha256": hashlib.sha256(_canonical_bytes_antigravity(graph)).hexdigest(),
        "represented_existing_paths": list(EXISTING_REPRESENTED_PATHS_ANTIGRAVITY),
        "represented_future_declared_paths": list(FUTURE_DECLARED_PATHS_ANTIGRAVITY),
        "source_fingerprints": graph["source_fingerprints"],
        "triplet": list(TRIPLET_ANTIGRAVITY),
    }


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
            raise ValueError("misnested HTML closing tag: " + tag)
        self.stack.pop()


def _validate_graph_antigravity(graph: dict[str, Any]) -> None:
    for relative in EXISTING_PRE_TRIPLET_PATHS_ANTIGRAVITY:
        if not (ROOT_ANTIGRAVITY / relative).is_file():
            raise ValueError("existing represented path missing: " + relative)
    if graph["future_declared_paths"] != list(FUTURE_DECLARED_PATHS_ANTIGRAVITY):
        raise ValueError("future declaration is stale")
    if graph["source_fingerprints"] != _fingerprints_antigravity():
        raise ValueError("source fingerprints are stale")
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    if len(node_by_id) != len(graph["nodes"]):
        raise ValueError("duplicate node ID")
    non_predecessor_nodes = set(node_by_id) - {"exp470_config_contract", "exp470_contracts_pattern"}
    for node_id in sorted(non_predecessor_nodes):
        evidence = node_by_id[node_id]["evidence"]
        if not _contains_symbol_antigravity(evidence["path"], evidence["symbol"]):
            raise ValueError("node evidence does not resolve: " + node_id)
    for edge in graph["edges"]:
        if edge["from"] not in node_by_id or edge["to"] not in node_by_id:
            raise ValueError("edge endpoint does not resolve: " + edge["from"] + "->" + edge["to"])
        if not _contains_symbol_antigravity(edge["evidence"]["path"], edge["evidence"]["symbol"]):
            raise ValueError("edge evidence does not resolve: " + edge["from"] + "->" + edge["to"])
        if edge["type"] == "authority":
            source_node = node_by_id[edge["from"]]
            target_evidence = node_by_id[edge["to"]]["evidence"]
            expected_edge_evidence = {"path": source_node["path"], "symbol": target_evidence["symbol"]}
            if edge["evidence"] != expected_edge_evidence:
                raise ValueError("authority edge evidence is not exact: " + edge["from"] + "->" + edge["to"])
            if not _imports_exact_authority_antigravity(
                source_node["path"], target_evidence["path"], target_evidence["symbol"]
            ):
                raise ValueError("authority import does not resolve: " + edge["from"] + "->" + edge["to"])


def _validate_html_document_antigravity(path: Path) -> str:
    document = path.read_text(encoding="utf-8")
    parser = _StrictHtmlAntigravity()
    parser.feed(document)
    parser.close()
    if parser.stack or not {"html", "head", "body", "main"}.issubset(parser.seen):
        raise ValueError("invalid HTML document: " + str(path))
    return document


def validate_triplet_antigravity() -> None:
    graph = build_graph_antigravity()
    _validate_graph_antigravity(graph)
    expected_json = _canonical_bytes_antigravity(graph)
    json_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1]
    if json_path.read_bytes() != expected_json:
        raise ValueError("JSON code map is stale or nondeterministic")
    html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]
    document = _validate_html_document_antigravity(html_path)
    match = re.search(
        r'<script id="exp471-codemap-json-antigravity" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    if match is None or _canonical_bytes_antigravity(json.loads(match.group(1))) != expected_json:
        raise ValueError("HTML embedded code map differs from JSON")
    lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2]
    if lock_path.read_bytes() != _canonical_bytes_antigravity(_lock_antigravity(graph)):
        raise ValueError("code-map lock is stale or nondeterministic")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_triplet_antigravity()
        return
    graph = build_graph_antigravity()
    _validate_graph_antigravity(graph)
    json_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1]
    html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]
    lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(_canonical_bytes_antigravity(graph))
    html_path.write_text(_render_html_antigravity(graph), encoding="utf-8")
    lock_path.write_bytes(_canonical_bytes_antigravity(_lock_antigravity(graph)))
    validate_triplet_antigravity()


if __name__ == "__main__":
    main_antigravity()
